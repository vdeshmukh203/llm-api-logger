"""
LLM API Logger - Logging and analysis of LLM API calls.

Provides:
- LogEntry dataclass for structured API call tracking
- LLMLogger class with SQLite/JSONL backend storage
- Cost estimation for 20+ LLM models
- urllib.request.urlopen monkey-patching for automatic logging
- LoggingSession context manager
- CLI for querying, summarising, and exporting logs
"""

import json
import sqlite3
import csv
import sys
import argparse
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path
from urllib import request as urllib_request
from urllib.response import addinfourl
from io import BytesIO
import time
import uuid

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# Pricing is per million tokens (USD)
COST_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o":             {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":        {"input": 0.15,   "output": 0.60},
    "gpt-4-turbo":        {"input": 10.00,  "output": 30.00},
    "gpt-4":              {"input": 30.00,  "output": 60.00},
    "gpt-3.5-turbo":      {"input": 0.50,   "output": 1.50},
    "claude-3-5-sonnet":  {"input": 3.00,   "output": 15.00},
    "claude-3-opus":      {"input": 15.00,  "output": 75.00},
    "claude-3-sonnet":    {"input": 3.00,   "output": 15.00},
    "claude-3-haiku":     {"input": 0.25,   "output": 1.25},
    "claude-2.1":         {"input": 8.00,   "output": 24.00},
    "claude-2":           {"input": 8.00,   "output": 24.00},
    "claude-instant":     {"input": 0.80,   "output": 2.40},
    "gemini-pro":         {"input": 0.50,   "output": 1.50},
    "gemini-1.5-pro":     {"input": 1.25,   "output": 5.00},
    "gemini-1.5-flash":   {"input": 0.075,  "output": 0.30},
    "gemini-2.0-flash":   {"input": 0.10,   "output": 0.40},
    "palm-2":             {"input": 0.00005, "output": 0.0001},
    "llama-2-7b":         {"input": 0.10,   "output": 0.10},
    "llama-2-13b":        {"input": 0.20,   "output": 0.20},
    "llama-2-70b":        {"input": 0.65,   "output": 0.75},
    "llama-3-8b":         {"input": 0.05,   "output": 0.10},
    "llama-3-70b":        {"input": 0.50,   "output": 1.00},
    "mistral-large":      {"input": 2.00,   "output": 6.00},
    "mistral-medium":     {"input": 0.27,   "output": 0.81},
    "mistral-small":      {"input": 0.14,   "output": 0.42},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated USD cost for an API call.

    Parameters
    ----------
    model:
        Model identifier as it appears in COST_TABLE.
    tokens_in:
        Number of prompt/input tokens.
    tokens_out:
        Number of completion/output tokens.

    Raises
    ------
    ValueError
        If *model* is not present in COST_TABLE.
    """
    if model not in COST_TABLE:
        raise ValueError(f"Model '{model}' not found in cost table.")
    pricing = COST_TABLE[model]
    return (tokens_in / 1_000_000) * pricing["input"] + \
           (tokens_out / 1_000_000) * pricing["output"]


def _extract_provider(url: str) -> str:
    """Infer LLM provider name from a URL string."""
    url_lower = url.lower()
    if "openai" in url_lower:
        return "openai"
    if "anthropic" in url_lower:
        return "anthropic"
    if "google" in url_lower or "gemini" in url_lower:
        return "google"
    if "mistral" in url_lower:
        return "mistral"
    if "together" in url_lower:
        return "together"
    if "cohere" in url_lower:
        return "cohere"
    if "huggingface" in url_lower:
        return "huggingface"
    return "unknown"


def _extract_model(request_body: Optional[str], response_body: Optional[str]) -> str:
    """Extract model name from request or response JSON body."""
    for body in filter(None, [request_body, response_body]):
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                for key in ("model", "modelId", "model_id", "engine"):
                    if key in data:
                        return str(data[key])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return "unknown"


def _tok(response_body: Optional[str]) -> tuple:
    """Extract (tokens_in, tokens_out) from a JSON response body."""
    if not response_body:
        return 0, 0
    try:
        d = json.loads(response_body)
        if isinstance(d, dict):
            if "usage" in d:
                u = d["usage"]
                return (
                    int(u.get("prompt_tokens", 0)),
                    int(u.get("completion_tokens", 0)),
                )
            if "usageMetadata" in d:
                u = d["usageMetadata"]
                return (
                    int(u.get("promptTokenCount", 0)),
                    int(u.get("candidatesTokenCount", 0)),
                )
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        pass
    return 0, 0


@dataclass
class LogEntry:
    """A single LLM API call log entry."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    method: str = "POST"
    provider: str = field(default="")
    model: str = field(default="")
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    status_code: int = 200
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.provider:
            self.provider = _extract_provider(self.url)
        if not self.model:
            self.model = _extract_model(self.request_body, self.response_body)
        if self.tokens_in == 0 or self.tokens_out == 0:
            ti, to = _tok(self.response_body)
            if ti > 0:
                self.tokens_in = ti
            if to > 0:
                self.tokens_out = to
        if self.tokens_in > 0 and self.tokens_out > 0 and self.cost_usd == 0.0:
            try:
                self.cost_usd = estimate_cost(self.model, self.tokens_in, self.tokens_out)
            except ValueError:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Deserialise from a plain dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LLMLogger:
    """Store and query LLM API call log entries.

    Parameters
    ----------
    db_path:
        File path for the SQLite database or JSONL file.
        Use ``":memory:"`` for an in-memory SQLite database.
    backend:
        Either ``"sqlite"`` or ``"jsonl"``.
    """

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite") -> None:
        self.db_path = db_path
        self.backend = backend
        self.entries: List[LogEntry] = []
        self.conn: Optional[sqlite3.Connection] = None
        if backend == "sqlite":
            self._init_sqlite()
        elif backend != "jsonl":
            raise ValueError(f"Unknown backend: {backend!r}. Use 'sqlite' or 'jsonl'.")

    def _init_sqlite(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS log_entries (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                method TEXT,
                provider TEXT,
                model TEXT,
                request_body TEXT,
                response_body TEXT,
                status_code INTEGER,
                latency_ms REAL,
                tokens_in INTEGER,
                tokens_out INTEGER,
                cost_usd REAL,
                timestamp TEXT,
                error TEXT
            )
        """)
        self.conn.commit()

    def record(self, entry: LogEntry) -> None:
        """Persist a log entry to the configured backend."""
        if self.backend == "sqlite":
            self.conn.execute("""
                INSERT OR REPLACE INTO log_entries
                (id, url, method, provider, model, request_body, response_body,
                 status_code, latency_ms, tokens_in, tokens_out, cost_usd, timestamp, error)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                entry.id, entry.url, entry.method, entry.provider, entry.model,
                entry.request_body, entry.response_body, entry.status_code,
                entry.latency_ms, entry.tokens_in, entry.tokens_out,
                entry.cost_usd, entry.timestamp, entry.error,
            ))
            self.conn.commit()
        else:
            self.entries.append(entry)

    def count(self) -> int:
        """Return total number of stored entries."""
        if self.backend == "sqlite":
            row = self.conn.execute("SELECT COUNT(*) FROM log_entries").fetchone()
            return row[0]
        return len(self.entries)

    def query(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        since: Optional[str] = None,
    ) -> List[LogEntry]:
        """Return entries, optionally filtered.

        Parameters
        ----------
        model:
            Exact model name to filter on.
        provider:
            Exact provider name to filter on.
        status_code:
            HTTP status code to filter on.
        since:
            ISO-8601 timestamp; only entries at or after this time are returned.
        """
        if self.backend == "sqlite":
            self.conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM log_entries WHERE 1=1"
            params: list = []
            if model:
                sql += " AND model = ?"
                params.append(model)
            if provider:
                sql += " AND provider = ?"
                params.append(provider)
            if status_code is not None:
                sql += " AND status_code = ?"
                params.append(status_code)
            if since:
                sql += " AND timestamp >= ?"
                params.append(since)
            sql += " ORDER BY timestamp DESC"
            rows = self.conn.execute(sql, params).fetchall()
            return [LogEntry(**dict(r)) for r in rows]

        entries = self.entries[:]
        if model:
            entries = [e for e in entries if e.model == model]
        if provider:
            entries = [e for e in entries if e.provider == provider]
        if status_code is not None:
            entries = [e for e in entries if e.status_code == status_code]
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics across all stored entries."""
        entries = self.query()
        if not entries:
            return {
                "total_calls": 0,
                "total_cost_usd": 0.0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "calls_by_model": {},
                "cost_by_model": {},
                "avg_latency_ms": 0.0,
            }
        calls_by_model: Dict[str, int] = {}
        cost_by_model: Dict[str, float] = {}
        for e in entries:
            calls_by_model[e.model] = calls_by_model.get(e.model, 0) + 1
            cost_by_model[e.model] = cost_by_model.get(e.model, 0.0) + e.cost_usd
        return {
            "total_calls": len(entries),
            "total_cost_usd": sum(e.cost_usd for e in entries),
            "total_tokens_in": sum(e.tokens_in for e in entries),
            "total_tokens_out": sum(e.tokens_out for e in entries),
            "calls_by_model": calls_by_model,
            "cost_by_model": cost_by_model,
            "avg_latency_ms": sum(e.latency_ms for e in entries) / len(entries),
        }

    def export_jsonl(self, path: str) -> None:
        """Write all entries to a JSONL file at *path*."""
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.query():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Write all entries to a CSV file at *path*."""
        entries = self.query()
        if not entries:
            return
        fieldnames = [
            "id", "url", "method", "provider", "model", "status_code",
            "latency_ms", "tokens_in", "tokens_out", "cost_usd", "timestamp", "error",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow({k: getattr(entry, k) for k in fieldnames})


# ---------------------------------------------------------------------------
# urllib monkey-patching
# ---------------------------------------------------------------------------

_original_urlopen = urllib_request.urlopen
_active_logger: Optional[LLMLogger] = None


def _is_llm(url: str, request_body: Optional[str]) -> bool:
    """Return True if the URL or body suggests an LLM API endpoint."""
    url_lower = url.lower()
    llm_keywords = [
        "openai", "anthropic", "google", "gemini", "mistral",
        "cohere", "together", "huggingface", "llama",
    ]
    if any(kw in url_lower for kw in llm_keywords):
        return True
    if request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict) and any(k in data for k in ("model", "engine", "modelId")):
                return True
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return False


def _patched_urlopen(url, data=None, timeout=None, **kwargs):
    """Replacement for urllib.request.urlopen that logs LLM API calls."""
    start_time = time.time()
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    status_code = 200

    if data is not None:
        request_body = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else str(data)

    url_str: str = url if isinstance(url, str) else url.full_url
    is_llm = _is_llm(url_str, request_body)

    try:
        call_kwargs = dict(kwargs)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        response = _original_urlopen(url, data=data, **call_kwargs)
        status_code = response.status

        if is_llm:
            response_data = response.read()
            response_body = response_data.decode("utf-8", errors="ignore")
            # Reconstruct a readable response so callers can still .read() it
            response = addinfourl(
                BytesIO(response_data), response.headers, url_str, status_code
            )

        if is_llm and _active_logger is not None:
            latency_ms = (time.time() - start_time) * 1000
            _active_logger.record(LogEntry(
                url=url_str, method="POST",
                request_body=request_body, response_body=response_body,
                status_code=status_code, latency_ms=latency_ms,
            ))
        return response

    except Exception as exc:
        if is_llm and _active_logger is not None:
            latency_ms = (time.time() - start_time) * 1000
            _active_logger.record(LogEntry(
                url=url_str, method="POST",
                request_body=request_body, response_body=response_body,
                status_code=status_code, latency_ms=latency_ms,
                error=str(exc),
            ))
        raise


def patch_urllib(logger_instance: Optional[LLMLogger] = None) -> None:
    """Replace urllib.request.urlopen with a logging wrapper."""
    global _active_logger
    _active_logger = logger_instance
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore the original urllib.request.urlopen."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


@contextmanager
def session(
    log_file: Optional[str] = None,
    backend: str = "jsonl",
    auto_patch: bool = True,
):
    """Context manager that captures LLM API calls made within the block.

    Parameters
    ----------
    log_file:
        Where to store entries. Defaults to ``"llm_api.jsonl"`` (JSONL) or
        ``":memory:"`` (SQLite).
    backend:
        ``"jsonl"`` or ``"sqlite"``.
    auto_patch:
        Patch ``urllib.request.urlopen`` automatically.

    Yields
    ------
    LLMLogger
        The active logger instance.
    """
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"
    log_instance = LLMLogger(db_path=log_file, backend=backend)
    if auto_patch:
        patch_urllib(log_instance)
    try:
        yield log_instance
    finally:
        if backend == "jsonl" and log_file != ":memory:":
            log_instance.export_jsonl(log_file)
        if auto_patch:
            unpatch_urllib()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="LLM API Logger — log and analyse LLM API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="sub-command")

    # summary
    sp = subparsers.add_parser("summary", help="Print aggregate statistics")
    sp.add_argument("log_file", nargs="?", default="llm_api.jsonl")

    # query
    sp = subparsers.add_parser("query", help="List log entries")
    sp.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    sp.add_argument("--model", help="Filter by model name")
    sp.add_argument("--provider", help="Filter by provider name")

    # export
    sp = subparsers.add_parser("export", help="Export logs to another format")
    sp.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    sp.add_argument("--output", "-o", required=True, help="Output file path")
    sp.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv")

    # gui
    subparsers.add_parser("gui", help="Launch the graphical dashboard")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "gui":
        _gui_main()
        return

    log_file: str = args.log_file
    backend = "jsonl" if log_file.endswith(".jsonl") else "sqlite"
    log_instance = LLMLogger(db_path=log_file, backend=backend)

    if backend == "jsonl" and Path(log_file).exists():
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    log_instance.entries.append(LogEntry.from_dict(data))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    if args.command == "summary":
        stats = log_instance.summary()
        print("\n" + "=" * 60)
        print("LLM API CALL SUMMARY")
        print("=" * 60)
        print(f"Total API Calls  : {stats['total_calls']}")
        print(f"Total Cost (USD) : ${stats['total_cost_usd']:.4f}")
        print(f"Total Tokens In  : {stats['total_tokens_in']:,}")
        print(f"Total Tokens Out : {stats['total_tokens_out']:,}")
        print(f"Avg Latency (ms) : {stats['avg_latency_ms']:.2f}")
        print("\nCalls by Model:")
        for mdl, cnt in sorted(stats["calls_by_model"].items()):
            cost = stats["cost_by_model"].get(mdl, 0.0)
            print(f"  {mdl:<30} {cnt:>5} calls  ${cost:>8.4f}")
        print("=" * 60 + "\n")

    elif args.command == "query":
        results = log_instance.query(model=args.model, provider=args.provider)
        print(f"\nFound {len(results)} entries\n")
        for entry in results[:10]:
            print(f"  {entry.timestamp} | {entry.provider:>10} | "
                  f"{entry.model:<20} | ${entry.cost_usd:.6f}")
        if len(results) > 10:
            print(f"  … and {len(results) - 10} more")
        print()

    elif args.command == "export":
        if args.format == "csv":
            log_instance.export_csv(args.output)
        else:
            log_instance.export_jsonl(args.output)
        print(f"Exported {log_instance.count()} entries to {args.output} ({args.format.upper()})")


def _gui_main() -> None:
    """Launch the Tkinter dashboard."""
    try:
        from .gui import main as gui_main
    except ImportError:
        print("GUI not available. Install the package with: pip install -e .", file=sys.stderr)
        sys.exit(1)
    gui_main()


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
LogRecord = LogEntry
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider

if __name__ == "__main__":
    main()
