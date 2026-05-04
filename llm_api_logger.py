"""
LLM API Logger - Complete implementation for logging and analyzing LLM API calls.

Provides:
- LogEntry dataclass for structured API call tracking
- LLMLogger class with SQLite/JSONL backend storage
- Cost estimation for 25+ LLM models
- urllib.request.urlopen monkey-patching for automatic logging
- LoggingSession context manager
- CLI for querying, summarizing, and exporting logs
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
import urllib.response
from urllib.error import URLError
from io import BytesIO
import time
import uuid

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

COST_TABLE: Dict[str, Dict[str, float]] = {
    # Prices in USD per 1 million tokens
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-2.1": {"input": 8.00, "output": 24.00},
    "claude-2": {"input": 8.00, "output": 24.00},
    "claude-instant": {"input": 0.80, "output": 2.40},
    "gemini-pro": {"input": 0.50, "output": 1.50},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "palm-2": {"input": 0.00005, "output": 0.0001},
    "llama-2-7b": {"input": 0.10, "output": 0.10},
    "llama-2-13b": {"input": 0.20, "output": 0.20},
    "llama-2-70b": {"input": 0.65, "output": 0.75},
    "llama-3-8b": {"input": 0.05, "output": 0.10},
    "llama-3-70b": {"input": 0.50, "output": 1.00},
    "mistral-large": {"input": 2.00, "output": 6.00},
    "mistral-medium": {"input": 0.27, "output": 0.81},
    "mistral-small": {"input": 0.14, "output": 0.42},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate the USD cost of an LLM API call.

    Args:
        model: Model identifier matching a key in COST_TABLE.
        tokens_in: Number of input/prompt tokens consumed.
        tokens_out: Number of output/completion tokens generated.

    Returns:
        Estimated cost in USD.

    Raises:
        ValueError: If the model is not found in COST_TABLE.
    """
    if model not in COST_TABLE:
        raise ValueError(f"Model '{model}' not found in cost table.")
    pricing = COST_TABLE[model]
    input_cost = (tokens_in / 1_000_000) * pricing["input"]
    output_cost = (tokens_out / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def _extract_provider(url: str) -> str:
    """Extract LLM provider name from a request URL."""
    url_lower = url.lower()
    if "openai" in url_lower:
        return "openai"
    elif "anthropic" in url_lower:
        return "anthropic"
    elif "google" in url_lower or "gemini" in url_lower:
        return "google"
    elif "mistral" in url_lower:
        return "mistral"
    elif "together" in url_lower:
        return "together"
    elif "cohere" in url_lower:
        return "cohere"
    elif "huggingface" in url_lower:
        return "huggingface"
    else:
        return "unknown"


def _extract_model(request_body: Optional[str], response_body: Optional[str]) -> str:
    """Extract model name from a JSON request or response body."""
    bodies = [b for b in [request_body, response_body] if b]
    for body in bodies:
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                for key in ("model", "modelId", "model_id", "engine"):
                    if key in data:
                        return str(data[key])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return "unknown"


def _tok(rs: Optional[str]) -> tuple:
    """Extract (tokens_in, tokens_out) from a JSON response body."""
    if not rs:
        return 0, 0
    try:
        d = json.loads(rs)
        if isinstance(d, dict):
            if "usage" in d:
                u = d["usage"]
                return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            if "usageMetadata" in d:
                u = d["usageMetadata"]
                return u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return 0, 0


@dataclass
class LogEntry:
    """Represents a single LLM API call log entry."""

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
        """Serialize LogEntry to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Deserialize a LogEntry from a plain dictionary."""
        return cls(**data)


class LLMLogger:
    """Store and query LLM API call logs using SQLite or JSONL backends."""

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite") -> None:
        """Initialize LLMLogger.

        Args:
            db_path: File path for the database (SQLite) or log file (JSONL).
                     Use ":memory:" for an in-memory SQLite database.
            backend: Storage backend — ``"sqlite"`` (default) or ``"jsonl"``.

        Raises:
            ValueError: If an unsupported backend is specified.
        """
        self.db_path = db_path
        self.backend = backend
        self.entries: List[LogEntry] = []
        self.conn: Optional[sqlite3.Connection] = None
        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "jsonl":
            pass
        else:
            raise ValueError(f"Unknown backend: '{backend}'. Choose 'sqlite' or 'jsonl'.")

    def _init_sqlite(self) -> None:
        """Create the SQLite database schema if it does not already exist."""
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
        """Persist a single log entry.

        Args:
            entry: The :class:`LogEntry` to store.
        """
        if self.backend == "sqlite":
            self.conn.execute("""
                INSERT OR REPLACE INTO log_entries
                (id, url, method, provider, model, request_body, response_body,
                 status_code, latency_ms, tokens_in, tokens_out, cost_usd, timestamp, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        """Return the total number of stored log entries."""
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
        """Return log entries matching the given filters, newest first.

        Args:
            model: Exact model name to match.
            provider: Exact provider name to match (e.g. ``"openai"``).
            status_code: HTTP status code to match (e.g. ``200``).
            since: ISO-format timestamp lower bound (inclusive).

        Returns:
            List of matching :class:`LogEntry` objects.
        """
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
            rows = self.conn.execute(sql, params).fetchall()
            return [LogEntry(**dict(r)) for r in rows]
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
        """Compute aggregate statistics across all stored entries.

        Returns:
            Dictionary with keys: ``total_calls``, ``total_cost_usd``,
            ``total_tokens_in``, ``total_tokens_out``, ``avg_latency_ms``,
            ``calls_by_model``, ``cost_by_model``.
        """
        entries = self.query()
        if not entries:
            return {
                "total_calls": 0,
                "total_cost_usd": 0.0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "avg_latency_ms": 0.0,
                "calls_by_model": {},
                "cost_by_model": {},
            }
        calls_by_model: Dict[str, int] = {}
        cost_by_model: Dict[str, float] = {}
        for entry in entries:
            calls_by_model[entry.model] = calls_by_model.get(entry.model, 0) + 1
            cost_by_model[entry.model] = cost_by_model.get(entry.model, 0.0) + entry.cost_usd
        return {
            "total_calls": len(entries),
            "total_cost_usd": sum(e.cost_usd for e in entries),
            "total_tokens_in": sum(e.tokens_in for e in entries),
            "total_tokens_out": sum(e.tokens_out for e in entries),
            "avg_latency_ms": sum(e.latency_ms for e in entries) / len(entries),
            "calls_by_model": calls_by_model,
            "cost_by_model": cost_by_model,
        }

    def export_jsonl(self, path: str) -> None:
        """Write all entries to a JSONL file (one JSON object per line).

        Args:
            path: Destination file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.query():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Write all entries to a CSV file.

        Args:
            path: Destination file path.
        """
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
        """Close the database connection (SQLite backend only)."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "LLMLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# urllib monkey-patching
# ---------------------------------------------------------------------------

_original_urlopen = urllib_request.urlopen
_active_logger: Optional[LLMLogger] = None


def _is_llm(url: str, request_body: Optional[str]) -> bool:
    """Return True if the URL or request body looks like an LLM API call."""
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
        if timeout is not None:
            response = _original_urlopen(url, data=data, timeout=timeout, **kwargs)
        else:
            response = _original_urlopen(url, data=data, **kwargs)
        status_code = response.status

        if is_llm:
            raw = response.read()
            response_body = raw.decode("utf-8", errors="ignore")
            saved_headers = response.headers
            response.close()
            # Reconstruct a readable response object so the caller can still use it
            response = urllib.response.addinfourl(
                BytesIO(raw), saved_headers, url_str, status_code
            )

        if is_llm and _active_logger is not None:
            latency_ms = (time.time() - start_time) * 1000
            entry = LogEntry(
                url=url_str, method="POST",
                request_body=request_body, response_body=response_body,
                status_code=status_code, latency_ms=latency_ms,
            )
            _active_logger.record(entry)

        return response

    except Exception as exc:
        if is_llm and _active_logger is not None:
            latency_ms = (time.time() - start_time) * 1000
            entry = LogEntry(
                url=url_str, method="POST",
                request_body=request_body, response_body=response_body,
                status_code=status_code, latency_ms=latency_ms,
                error=str(exc),
            )
            _active_logger.record(entry)
        raise


def patch_urllib(log: Optional[LLMLogger] = None) -> None:
    """Monkey-patch :func:`urllib.request.urlopen` to automatically log LLM calls.

    Args:
        log: Logger instance to record calls into.  If ``None``, calls are
             intercepted but not stored.
    """
    global _active_logger
    _active_logger = log
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore the original :func:`urllib.request.urlopen`."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


@contextmanager
def session(log_file: Optional[str] = None, backend: str = "jsonl", auto_patch: bool = True):
    """Context manager for a scoped LLM API logging session.

    Args:
        log_file: Path to the log file.  Defaults to ``"llm_api.jsonl"`` for
                  JSONL backend or ``":memory:"`` for SQLite.
        backend: Storage backend — ``"sqlite"`` or ``"jsonl"`` (default).
        auto_patch: Whether to automatically monkey-patch
                    :func:`urllib.request.urlopen`.

    Yields:
        An :class:`LLMLogger` instance for the duration of the context.

    Example::

        with session("my_log.jsonl") as log:
            # make LLM API calls here
            urllib.request.urlopen(req)
        # log is flushed to disk on exit
    """
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"
    log = LLMLogger(db_path=log_file, backend=backend)
    if auto_patch:
        patch_urllib(log)
    try:
        yield log
    finally:
        if log_file != ":memory:" and backend == "jsonl":
            log.export_jsonl(log_file)
        if auto_patch:
            unpatch_urllib()
        log.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_jsonl(log_file: str) -> LLMLogger:
    """Load a JSONL log file into an in-memory LLMLogger."""
    log = LLMLogger(db_path=":memory:", backend="jsonl")
    p = Path(log_file)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log.entries.append(LogEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    logger.warning("Skipping malformed JSONL line: %s", exc)
    return log


def main() -> None:
    """Command-line interface for LLM API Logger."""
    parser = argparse.ArgumentParser(
        prog="llm-api-logger",
        description="LLM API Logger — log, query, and export LLM API call records",
    )
    sub = parser.add_subparsers(dest="command", help="available commands")

    # summary
    p_sum = sub.add_parser("summary", help="display aggregate statistics")
    p_sum.add_argument("log_file", nargs="?", default="llm_api.jsonl", help="log file path")

    # query
    p_qry = sub.add_parser("query", help="list and filter log entries")
    p_qry.add_argument("log_file", nargs="?", default="llm_api.jsonl", help="log file path")
    p_qry.add_argument("--model", help="filter by model name")
    p_qry.add_argument("--provider", help="filter by provider name")
    p_qry.add_argument("--limit", type=int, default=10, help="max entries to display (default: 10)")

    # export
    p_exp = sub.add_parser("export", help="export log to CSV or JSONL")
    p_exp.add_argument("log_file", nargs="?", default="llm_api.jsonl", help="source log file")
    p_exp.add_argument("--output", "-o", required=True, help="destination file path")
    p_exp.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv", help="output format")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    log_file: str = args.log_file
    if log_file.endswith(".jsonl"):
        log = _load_jsonl(log_file)
    else:
        log = LLMLogger(db_path=log_file, backend="sqlite")

    try:
        if args.command == "summary":
            s = log.summary()
            print("\n" + "=" * 60)
            print("LLM API CALL SUMMARY")
            print("=" * 60)
            print(f"Total API Calls    : {s['total_calls']}")
            print(f"Total Cost (USD)   : ${s['total_cost_usd']:.6f}")
            print(f"Total Input Tokens : {s['total_tokens_in']:,}")
            print(f"Total Output Tokens: {s['total_tokens_out']:,}")
            print(f"Avg Latency (ms)   : {s['avg_latency_ms']:.2f}")
            if s["calls_by_model"]:
                print("\nBreakdown by Model:")
                for m, cnt in sorted(s["calls_by_model"].items()):
                    cost = s["cost_by_model"].get(m, 0.0)
                    print(f"  {m:<30} {cnt:>5} call(s)  ${cost:>10.6f}")
            print("=" * 60 + "\n")

        elif args.command == "query":
            results = log.query(model=args.model, provider=args.provider)
            print(f"\nFound {len(results)} matching entries\n")
            limit = args.limit
            for entry in results[:limit]:
                err = f" [ERR: {entry.error}]" if entry.error else ""
                print(
                    f"  {entry.timestamp}  {entry.provider:>10}  "
                    f"{entry.model:<25}  ${entry.cost_usd:.6f}{err}"
                )
            if len(results) > limit:
                print(f"  … and {len(results) - limit} more (use --limit to show more)")
            print()

        elif args.command == "export":
            if args.format == "csv":
                log.export_csv(args.output)
            else:
                log.export_jsonl(args.output)
            print(f"Exported {log.count()} entries to '{args.output}' ({args.format.upper()})")
    finally:
        log.close()


# Backwards-compatible aliases
LogRecord = LogEntry
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider
_cli = main


if __name__ == "__main__":
    main()
