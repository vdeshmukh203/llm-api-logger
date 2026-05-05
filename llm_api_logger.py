"""
LLM API Logger - Middleware library for logging and analyzing LLM API calls.

Provides:
- LogEntry dataclass for structured API call tracking
- LLMLogger class with SQLite/JSONL backend storage
- Cost estimation for 20+ LLM models with prefix-matched versioned names
- urllib.request.urlopen monkey-patching for automatic transparent logging
- session() context manager for scoped logging sessions
- CLI for querying, summarizing, and exporting logs
"""

import json
import sqlite3
import csv
import sys
import argparse
import logging
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from pathlib import Path
from io import BytesIO
from urllib import request as urllib_request
from urllib.response import addinfourl
from urllib.error import URLError
import time
import uuid

__version__ = "0.1.0"

_log = logging.getLogger(__name__)

# Pricing in USD per 1 million tokens (input, output).
# Versioned model IDs (e.g. "gpt-4o-2024-05-13") match via prefix lookup.
COST_TABLE: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":              {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":         {"input": 0.15,   "output": 0.60},
    "gpt-4-turbo":         {"input": 10.00,  "output": 30.00},
    "gpt-4":               {"input": 30.00,  "output": 60.00},
    "gpt-3.5-turbo":       {"input": 0.50,   "output": 1.50},
    "o1":                  {"input": 15.00,  "output": 60.00},
    "o1-mini":             {"input": 3.00,   "output": 12.00},
    "o3-mini":             {"input": 1.10,   "output": 4.40},
    # Anthropic
    "claude-opus-4":       {"input": 15.00,  "output": 75.00},
    "claude-sonnet-4":     {"input": 3.00,   "output": 15.00},
    "claude-3-5-sonnet":   {"input": 3.00,   "output": 15.00},
    "claude-3-5-haiku":    {"input": 0.80,   "output": 4.00},
    "claude-3-opus":       {"input": 15.00,  "output": 75.00},
    "claude-3-sonnet":     {"input": 3.00,   "output": 15.00},
    "claude-3-haiku":      {"input": 0.25,   "output": 1.25},
    "claude-2.1":          {"input": 8.00,   "output": 24.00},
    "claude-2":            {"input": 8.00,   "output": 24.00},
    "claude-instant":      {"input": 0.80,   "output": 2.40},
    # Google
    "gemini-2.0-flash":    {"input": 0.10,   "output": 0.40},
    "gemini-1.5-pro":      {"input": 1.25,   "output": 5.00},
    "gemini-1.5-flash":    {"input": 0.075,  "output": 0.30},
    "gemini-pro":          {"input": 0.50,   "output": 1.50},
    # Open-source / hosted
    "llama-3-70b":         {"input": 0.50,   "output": 1.00},
    "llama-3-8b":          {"input": 0.05,   "output": 0.10},
    "llama-2-70b":         {"input": 0.65,   "output": 0.75},
    "llama-2-13b":         {"input": 0.20,   "output": 0.20},
    "llama-2-7b":          {"input": 0.10,   "output": 0.10},
    "mistral-large":       {"input": 2.00,   "output": 6.00},
    "mistral-medium":      {"input": 0.27,   "output": 0.81},
    "mistral-small":       {"input": 0.14,   "output": 0.42},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated USD cost for an API call.

    Performs exact lookup first, then prefix-matches to handle versioned model
    IDs such as ``gpt-4o-2024-05-13`` -> ``gpt-4o``.

    Raises ``ValueError`` when no match is found.
    """
    pricing = COST_TABLE.get(model)
    if pricing is None:
        for key in COST_TABLE:
            if model.startswith(key):
                pricing = COST_TABLE[key]
                break
    if pricing is None:
        raise ValueError(f"Model '{model}' not found in COST_TABLE.")
    return (tokens_in / 1_000_000) * pricing["input"] + (tokens_out / 1_000_000) * pricing["output"]


def _extract_provider(url: str) -> str:
    """Identify the LLM provider from a request URL."""
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
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return "unknown"


def _tok(response_body: Optional[str]) -> Tuple[int, int]:
    """Extract (tokens_in, tokens_out) from a provider response body.

    Supports:
    - OpenAI / Together / Mistral: ``usage.prompt_tokens`` / ``usage.completion_tokens``
    - Anthropic: ``usage.input_tokens`` / ``usage.output_tokens``
    - Google Gemini: ``usageMetadata.promptTokenCount`` / ``usageMetadata.candidatesTokenCount``
    - Cohere: ``meta.billed_units.input_tokens`` / ``meta.billed_units.output_tokens``
    """
    if not response_body:
        return 0, 0
    try:
        d = json.loads(response_body)
        if not isinstance(d, dict):
            return 0, 0
        # OpenAI-compatible (prompt_tokens / completion_tokens)
        # Anthropic-compatible (input_tokens / output_tokens) — both live under "usage"
        if "usage" in d:
            u = d["usage"]
            tok_in = u.get("prompt_tokens") or u.get("input_tokens", 0)
            tok_out = u.get("completion_tokens") or u.get("output_tokens", 0)
            return int(tok_in or 0), int(tok_out or 0)
        # Google Gemini
        if "usageMetadata" in d:
            u = d["usageMetadata"]
            return int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))
        # Cohere
        if "meta" in d and isinstance(d["meta"], dict) and "billed_units" in d["meta"]:
            u = d["meta"]["billed_units"]
            return int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return 0, 0


@dataclass
class LogEntry:
    """Structured record of a single LLM API call."""

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
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Deserialize from a plain dictionary."""
        return cls(**data)


class LLMLogger:
    """Persistent store for LLM API call log entries.

    Supports two storage backends:
    - ``sqlite``: on-disk or in-memory SQLite database (supports full querying)
    - ``jsonl``: in-memory list, optionally flushed to a JSONL file at session end
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
        """Return entries matching the supplied filters, newest first."""
        if self.backend == "sqlite":
            self.conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM log_entries WHERE 1=1"
            params: List[Any] = []
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
            entries = list(self.entries)
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
                "total_calls": 0, "total_cost_usd": 0.0,
                "total_tokens_in": 0, "total_tokens_out": 0,
                "calls_by_model": {}, "cost_by_model": {},
                "calls_by_provider": {}, "avg_latency_ms": 0.0,
            }
        calls_by_model: Dict[str, int] = {}
        cost_by_model: Dict[str, float] = {}
        calls_by_provider: Dict[str, int] = {}
        for e in entries:
            calls_by_model[e.model] = calls_by_model.get(e.model, 0) + 1
            cost_by_model[e.model] = cost_by_model.get(e.model, 0.0) + e.cost_usd
            calls_by_provider[e.provider] = calls_by_provider.get(e.provider, 0) + 1
        return {
            "total_calls": len(entries),
            "total_cost_usd": sum(e.cost_usd for e in entries),
            "total_tokens_in": sum(e.tokens_in for e in entries),
            "total_tokens_out": sum(e.tokens_out for e in entries),
            "calls_by_model": calls_by_model,
            "cost_by_model": cost_by_model,
            "calls_by_provider": calls_by_provider,
            "avg_latency_ms": sum(e.latency_ms for e in entries) / len(entries),
        }

    def export_jsonl(self, path: str) -> None:
        """Write all entries to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.query():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Write all entries to a CSV file."""
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


# --- urllib monkey-patching ---

_original_urlopen = urllib_request.urlopen
_active_logger: Optional[LLMLogger] = None
_patch_lock = threading.Lock()


def _is_llm(url: str, request_body: Optional[str]) -> bool:
    """Return True if the URL/body looks like an LLM API request."""
    llm_keywords = (
        "openai", "anthropic", "google", "gemini",
        "mistral", "cohere", "together", "huggingface", "llama",
    )
    if any(kw in url.lower() for kw in llm_keywords):
        return True
    if request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict) and any(k in data for k in ("model", "engine", "modelId")):
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


def _patched_urlopen(url, data=None, timeout=None, **kwargs):
    """Replacement for urllib.request.urlopen that logs LLM API calls."""
    start = time.time()

    # Extract URL string and request body from both str and Request objects.
    url_str = url.full_url if hasattr(url, "full_url") else str(url)
    request_body: Optional[str] = None
    if data is not None:
        request_body = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else str(data)
    elif hasattr(url, "data") and url.data is not None:
        raw = url.data
        request_body = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)

    is_llm = _is_llm(url_str, request_body)
    status_code = 200
    response_body: Optional[str] = None

    try:
        call_kwargs = dict(kwargs)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        response = _original_urlopen(url, data=data, **call_kwargs)
        status_code = getattr(response, "status", None) or getattr(response, "code", 200) or 200

        if is_llm:
            response_data = response.read()
            response_body = response_data.decode("utf-8", errors="ignore")
            response.close()
            # Reconstruct a readable response object so the caller can still call .read()
            new_resp = addinfourl(BytesIO(response_data), response.headers, url_str, status_code)
            response = new_resp

        if is_llm:
            _record_entry(url_str, request_body, response_body, status_code, start)

        return response

    except Exception as exc:
        if is_llm:
            _record_entry(url_str, request_body, response_body, status_code, start, error=str(exc))
        raise


def _record_entry(
    url: str,
    request_body: Optional[str],
    response_body: Optional[str],
    status_code: int,
    start: float,
    error: Optional[str] = None,
) -> None:
    logger = _active_logger
    if logger is None:
        return
    latency_ms = (time.time() - start) * 1000
    entry = LogEntry(
        url=url,
        method="POST",
        request_body=request_body,
        response_body=response_body,
        status_code=status_code,
        latency_ms=latency_ms,
        error=error,
    )
    try:
        logger.record(entry)
    except Exception:
        _log.exception("Failed to record log entry")


def patch_urllib(logger: Optional[LLMLogger] = None) -> None:
    """Monkey-patch urllib.request.urlopen to log LLM API calls transparently."""
    global _active_logger
    with _patch_lock:
        _active_logger = logger
        urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore the original urllib.request.urlopen."""
    global _active_logger
    with _patch_lock:
        urllib_request.urlopen = _original_urlopen
        _active_logger = None


@contextmanager
def session(log_file: Optional[str] = None, backend: str = "jsonl", auto_patch: bool = True):
    """Context manager that captures all LLM API calls within its scope.

    Usage::

        with llm_api_logger.session("run.jsonl") as log:
            response = client.chat.completions.create(...)
        print(log.summary())
    """
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"
    logger = LLMLogger(db_path=log_file if backend == "sqlite" else ":memory:", backend=backend)
    if auto_patch:
        patch_urllib(logger)
    try:
        yield logger
    finally:
        if auto_patch:
            unpatch_urllib()
        if backend == "jsonl" and log_file != ":memory:":
            logger.export_jsonl(log_file)


# --- CLI ---

def _load_logger(log_file: str) -> LLMLogger:
    """Instantiate and populate an LLMLogger from a log file."""
    backend = "jsonl" if log_file.endswith(".jsonl") else "sqlite"
    if backend == "sqlite":
        return LLMLogger(db_path=log_file, backend="sqlite")
    logger = LLMLogger(db_path=":memory:", backend="jsonl")
    path = Path(log_file)
    if path.exists():
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    logger.entries.append(LogEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    _log.warning("Skipping malformed log line: %s", exc)
    return logger


def main() -> None:
    """Command-line interface for LLM API Logger."""
    parser = argparse.ArgumentParser(
        description="LLM API Logger — log, query, and visualise LLM API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # summary
    sp = sub.add_parser("summary", help="Print aggregate statistics")
    sp.add_argument("log_file", nargs="?", default="llm_api.jsonl")

    # query
    qp = sub.add_parser("query", help="Filter and list log entries")
    qp.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    qp.add_argument("--model", help="Filter by model name")
    qp.add_argument("--provider", help="Filter by provider")
    qp.add_argument("--status", type=int, dest="status_code", help="Filter by HTTP status code")

    # export
    ep = sub.add_parser("export", help="Export logs to CSV or JSONL")
    ep.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    ep.add_argument("--output", "-o", required=True, help="Destination file path")
    ep.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv")

    # gui
    gp = sub.add_parser("gui", help="Launch the web dashboard")
    gp.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    gp.add_argument("--host", default="127.0.0.1")
    gp.add_argument("--port", type=int, default=5000)
    gp.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "gui":
        try:
            import gui as _gui
            _gui.launch(args.log_file, host=args.host, port=args.port,
                        open_browser=not args.no_browser)
        except ImportError as exc:
            sys.exit(f"GUI unavailable: {exc}\nInstall with: pip install 'llm-api-logger[gui]'")
        return

    logger = _load_logger(args.log_file)

    if args.command == "summary":
        s = logger.summary()
        print("\n" + "=" * 60)
        print("LLM API CALL SUMMARY")
        print("=" * 60)
        print(f"Total API Calls   : {s['total_calls']}")
        print(f"Total Cost (USD)  : ${s['total_cost_usd']:.4f}")
        print(f"Total Tokens In   : {s['total_tokens_in']:,}")
        print(f"Total Tokens Out  : {s['total_tokens_out']:,}")
        print(f"Avg Latency (ms)  : {s['avg_latency_ms']:.2f}")
        if s["calls_by_model"]:
            print("\nCalls by Model:")
            for model, count in sorted(s["calls_by_model"].items()):
                cost = s["cost_by_model"].get(model, 0.0)
                print(f"  {model:<35} {count:>5} calls  ${cost:>10.4f}")
        print("=" * 60 + "\n")

    elif args.command == "query":
        results = logger.query(
            model=args.model,
            provider=args.provider,
            status_code=args.status_code,
        )
        print(f"\nFound {len(results)} entries\n")
        for entry in results[:20]:
            err = f"  ERR={entry.error[:40]}" if entry.error else ""
            print(
                f"  {entry.timestamp}  {entry.provider:>10}  "
                f"{entry.model:<25}  ${entry.cost_usd:.6f}{err}"
            )
        if len(results) > 20:
            print(f"  … and {len(results) - 20} more")
        print()

    elif args.command == "export":
        if args.format == "csv":
            logger.export_csv(args.output)
        else:
            logger.export_jsonl(args.output)
        print(f"Exported {logger.count()} entries to {args.output} ({args.format.upper()})")


# Backwards-compatible aliases (for code written against older interfaces)
LogRecord = LogEntry
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider


if __name__ == "__main__":
    main()
