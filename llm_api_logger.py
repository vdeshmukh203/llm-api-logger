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
from urllib.error import URLError
from io import BytesIO
import time
import uuid

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# Cost in USD per 1 million tokens (input, output)
COST_TABLE: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-2.1": {"input": 8.00, "output": 24.00},
    "claude-2": {"input": 8.00, "output": 24.00},
    "claude-instant": {"input": 0.80, "output": 2.40},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-pro": {"input": 0.50, "output": 1.50},
    "palm-2": {"input": 0.00005, "output": 0.0001},
    # Meta / Llama (via hosted APIs)
    "llama-3-8b": {"input": 0.05, "output": 0.10},
    "llama-3-70b": {"input": 0.50, "output": 1.00},
    "llama-2-7b": {"input": 0.10, "output": 0.10},
    "llama-2-13b": {"input": 0.20, "output": 0.20},
    "llama-2-70b": {"input": 0.65, "output": 0.75},
    # Mistral
    "mistral-large": {"input": 2.00, "output": 6.00},
    "mistral-medium": {"input": 0.27, "output": 0.81},
    "mistral-small": {"input": 0.14, "output": 0.42},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated USD cost for an LLM API call.

    Args:
        model: Model identifier (key in COST_TABLE).
        tokens_in: Number of input/prompt tokens.
        tokens_out: Number of output/completion tokens.

    Returns:
        Estimated cost in USD.

    Raises:
        ValueError: If *model* is not present in COST_TABLE.
    """
    pricing = COST_TABLE.get(model)
    if pricing is None:
        raise ValueError(
            f"Model '{model}' not found in COST_TABLE. "
            f"Available models: {sorted(COST_TABLE)}"
        )
    return (tokens_in / 1_000_000) * pricing["input"] + (tokens_out / 1_000_000) * pricing["output"]


def _extract_provider(url: str) -> str:
    """Return a normalised provider name derived from *url*."""
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
    """Return model name extracted from request or response JSON."""
    for body in filter(None, [request_body, response_body]):
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                for key in ("model", "modelId", "model_id", "engine"):
                    if key in data:
                        return str(data[key])
        except (json.JSONDecodeError, ValueError):
            pass
    return "unknown"


def _tok(response_body: Optional[str]) -> tuple:
    """Extract (tokens_in, tokens_out) from a JSON response body.

    Supports OpenAI, Anthropic, and Google Gemini response formats.
    """
    if not response_body:
        return 0, 0
    try:
        d = json.loads(response_body)
        if not isinstance(d, dict):
            return 0, 0
        # OpenAI / OpenAI-compatible: usage.prompt_tokens / completion_tokens
        if "usage" in d:
            u = d["usage"]
            tokens_in = u.get("prompt_tokens") or u.get("input_tokens", 0)
            tokens_out = u.get("completion_tokens") or u.get("output_tokens", 0)
            return int(tokens_in), int(tokens_out)
        # Google Gemini: usageMetadata
        if "usageMetadata" in d:
            u = d["usageMetadata"]
            return int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return 0, 0


@dataclass
class LogEntry:
    """A single LLM API call log record."""

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
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        return cls(**data)


class LLMLogger:
    """Stores and queries LLM API call records.

    Supports two backends:

    * ``"sqlite"`` – persists to a SQLite database (default).
    * ``"jsonl"``  – keeps records in memory and exports to JSONL on demand.
    """

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite") -> None:
        if backend not in ("sqlite", "jsonl"):
            raise ValueError(f"Unknown backend '{backend}'. Choose 'sqlite' or 'jsonl'.")
        self.db_path = db_path
        self.backend = backend
        self.entries: List[LogEntry] = []
        self.conn: Optional[sqlite3.Connection] = None
        if backend == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS log_entries (
                id           TEXT PRIMARY KEY,
                url          TEXT NOT NULL,
                method       TEXT,
                provider     TEXT,
                model        TEXT,
                request_body TEXT,
                response_body TEXT,
                status_code  INTEGER,
                latency_ms   REAL,
                tokens_in    INTEGER,
                tokens_out   INTEGER,
                cost_usd     REAL,
                timestamp    TEXT,
                error        TEXT
            )
        """)
        self.conn.commit()

    def record(self, entry: LogEntry) -> None:
        """Persist *entry* to the chosen backend."""
        if self.backend == "sqlite":
            self.conn.execute(
                """INSERT OR REPLACE INTO log_entries
                   (id, url, method, provider, model, request_body, response_body,
                    status_code, latency_ms, tokens_in, tokens_out, cost_usd, timestamp, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (entry.id, entry.url, entry.method, entry.provider, entry.model,
                 entry.request_body, entry.response_body, entry.status_code,
                 entry.latency_ms, entry.tokens_in, entry.tokens_out,
                 entry.cost_usd, entry.timestamp, entry.error),
            )
            self.conn.commit()
        else:
            self.entries.append(entry)

    def count(self) -> int:
        """Return the total number of stored entries."""
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
        """Return entries matching the given filters, ordered newest-first."""
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
            return [LogEntry(**dict(r)) for r in self.conn.execute(sql, params).fetchall()]
        else:
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
        """Return aggregate statistics over all stored entries."""
        entries = self.query()
        if not entries:
            return {
                "total_calls": 0, "total_cost_usd": 0.0,
                "total_tokens_in": 0, "total_tokens_out": 0,
                "calls_by_model": {}, "cost_by_model": {}, "avg_latency_ms": 0.0,
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
            "avg_latency_ms": sum(e.latency_ms for e in entries) / len(entries),
            "calls_by_model": calls_by_model,
            "cost_by_model": cost_by_model,
        }

    def export_jsonl(self, path: str, append: bool = False) -> None:
        """Write all entries to a JSONL file.

        Args:
            path: Destination file path.
            append: If *True*, append to an existing file instead of overwriting.
        """
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            for entry in self.query():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Write all entries to a CSV file (excludes raw request/response bodies)."""
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

    def close(self) -> None:
        """Close the SQLite connection, if open."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# ---------------------------------------------------------------------------
# urllib monkey-patching
# ---------------------------------------------------------------------------

_original_urlopen = urllib_request.urlopen
_active_logger: Optional[LLMLogger] = None

_LLM_KEYWORDS = frozenset([
    "openai", "anthropic", "google", "gemini", "mistral",
    "cohere", "together", "huggingface", "llama",
])


def _is_llm(url: str, request_body: Optional[str]) -> bool:
    """Return True if *url* looks like an LLM provider API endpoint."""
    if any(kw in url.lower() for kw in _LLM_KEYWORDS):
        return True
    if request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict) and any(k in data for k in ("model", "engine", "modelId")):
                return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


def _patched_urlopen(url, data=None, timeout=None, **kwargs):
    """Replacement for urllib.request.urlopen that logs LLM API calls."""
    start = time.monotonic()
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
            # Rebuild a readable response object so callers can still read the body.
            response = urllib_request.addinfourl(
                BytesIO(response_data), response.headers, url_str, code=status_code
            )

        if is_llm and _active_logger is not None:
            _active_logger.record(LogEntry(
                url=url_str,
                method="POST",
                request_body=request_body,
                response_body=response_body,
                status_code=status_code,
                latency_ms=(time.monotonic() - start) * 1000,
            ))
        return response

    except Exception as exc:
        if is_llm and _active_logger is not None:
            _active_logger.record(LogEntry(
                url=url_str,
                method="POST",
                request_body=request_body,
                response_body=response_body,
                status_code=status_code,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            ))
        raise


def patch_urllib(active_logger: Optional[LLMLogger] = None) -> None:
    """Patch ``urllib.request.urlopen`` to intercept LLM API calls."""
    global _active_logger
    _active_logger = active_logger
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore the original ``urllib.request.urlopen``."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def session(log_file: Optional[str] = None, backend: str = "jsonl", auto_patch: bool = True):
    """Context manager that captures LLM API calls for the duration of the block.

    Args:
        log_file: Path to the log file. Defaults to ``llm_api.jsonl`` (JSONL)
                  or ``:memory:`` (SQLite).
        backend: ``"jsonl"`` or ``"sqlite"``.
        auto_patch: Monkey-patch ``urllib.request.urlopen`` automatically.

    Yields:
        The active :class:`LLMLogger` instance.

    Example::

        with session("my_run.jsonl") as log:
            # make LLM API calls …
            print(log.summary())
    """
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"

    active = LLMLogger(db_path=log_file, backend=backend)
    if auto_patch:
        patch_urllib(active)
    try:
        yield active
    finally:
        if auto_patch:
            unpatch_urllib()
        if backend == "jsonl" and log_file != ":memory:":
            # Append the session entries so previous runs are preserved.
            active.export_jsonl(log_file, append=True)
        if log_file != ":memory:":
            active.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_jsonl(log_file: str, active: LLMLogger) -> None:
    """Load a JSONL log file into *active* (in-memory JSONL backend)."""
    p = Path(log_file)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                active.entries.append(LogEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning("Skipping malformed JSONL line: %s", line[:80])


def main() -> None:
    """Entry point for the ``llm-api-logger`` command-line tool."""
    parser = argparse.ArgumentParser(
        prog="llm-api-logger",
        description="LLM API Logger — log, query, and export LLM API call records.",
    )
    sub = parser.add_subparsers(dest="command")

    # summary
    p_sum = sub.add_parser("summary", help="Print aggregate statistics.")
    p_sum.add_argument("log_file", nargs="?", default="llm_api.jsonl")

    # query
    p_qry = sub.add_parser("query", help="List log entries with optional filters.")
    p_qry.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    p_qry.add_argument("--model", help="Filter by model name.")
    p_qry.add_argument("--provider", help="Filter by provider.")
    p_qry.add_argument("--limit", type=int, default=20, help="Max rows to display.")

    # export
    p_exp = sub.add_parser("export", help="Export logs to CSV or JSONL.")
    p_exp.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    p_exp.add_argument("--output", "-o", required=True, help="Output file path.")
    p_exp.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    log_file: str = args.log_file
    backend = "jsonl" if log_file.endswith(".jsonl") else "sqlite"
    active = LLMLogger(db_path=log_file, backend=backend)
    if backend == "jsonl":
        _load_jsonl(log_file, active)

    if args.command == "summary":
        s = active.summary()
        print("\n" + "=" * 62)
        print("  LLM API CALL SUMMARY")
        print("=" * 62)
        print(f"  Total calls      : {s['total_calls']}")
        print(f"  Total cost (USD) : ${s['total_cost_usd']:.4f}")
        print(f"  Input tokens     : {s['total_tokens_in']:,}")
        print(f"  Output tokens    : {s['total_tokens_out']:,}")
        print(f"  Avg latency (ms) : {s['avg_latency_ms']:.1f}")
        if s["calls_by_model"]:
            print("\n  Calls by model:")
            for mdl, cnt in sorted(s["calls_by_model"].items()):
                cost = s["cost_by_model"].get(mdl, 0.0)
                print(f"    {mdl:<32} {cnt:>5} calls   ${cost:>9.4f}")
        print("=" * 62 + "\n")

    elif args.command == "query":
        results = active.query(model=args.model, provider=args.provider)
        print(f"\nFound {len(results)} entr{'y' if len(results) == 1 else 'ies'}\n")
        limit = args.limit
        for entry in results[:limit]:
            ts = entry.timestamp
            print(
                f"  {ts}  {entry.provider:>10}  {entry.model:<26}  "
                f"${entry.cost_usd:.6f}  {entry.latency_ms:.0f} ms"
            )
        if len(results) > limit:
            print(f"  … and {len(results) - limit} more (use --limit to see more)")
        print()

    elif args.command == "export":
        if args.format == "csv":
            active.export_csv(args.output)
        else:
            active.export_jsonl(args.output)
        print(f"Exported {active.count()} entries → {args.output} ({args.format.upper()})")


# Backwards-compatible aliases
LogRecord = LogEntry
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider
_cli = main

if __name__ == "__main__":
    main()
