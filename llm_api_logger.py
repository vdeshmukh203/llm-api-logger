"""
LLM API Logger - Middleware for logging and analyzing LLM API calls.

Provides:
- ``LogEntry`` dataclass for structured API call tracking
- ``LLMLogger`` with SQLite and JSONL storage backends
- Cost estimation for 25+ LLM models
- Automatic logging via ``urllib.request.urlopen`` monkey-patching
- ``session()`` context manager for scoped logging
- CLI for querying, summarizing, and exporting logs
- GUI launcher via ``llm-api-logger-gui`` entry point
"""

import csv
import json
import logging
import sqlite3
import sys
import argparse
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request
from urllib.response import addinfourl

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table: USD per 1 000 000 tokens (input / output)
# ---------------------------------------------------------------------------
COST_TABLE: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":              {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":         {"input": 0.15,   "output": 0.60},
    "gpt-4-turbo":         {"input": 10.00,  "output": 30.00},
    "gpt-4":               {"input": 30.00,  "output": 60.00},
    "gpt-3.5-turbo":       {"input": 0.50,   "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet":   {"input": 3.00,   "output": 15.00},
    "claude-3-5-haiku":    {"input": 0.80,   "output": 4.00},
    "claude-3-opus":       {"input": 15.00,  "output": 75.00},
    "claude-3-sonnet":     {"input": 3.00,   "output": 15.00},
    "claude-3-haiku":      {"input": 0.25,   "output": 1.25},
    "claude-2.1":          {"input": 8.00,   "output": 24.00},
    "claude-2":            {"input": 8.00,   "output": 24.00},
    "claude-instant":      {"input": 0.80,   "output": 2.40},
    # Google
    "gemini-pro":          {"input": 0.50,   "output": 1.50},
    "gemini-1.5-pro":      {"input": 1.25,   "output": 5.00},
    "gemini-1.5-flash":    {"input": 0.075,  "output": 0.30},
    "gemini-2.0-flash":    {"input": 0.10,   "output": 0.40},
    "palm-2":              {"input": 0.00005, "output": 0.0001},
    # Meta / open-weight (via inference APIs)
    "llama-2-7b":          {"input": 0.10,   "output": 0.10},
    "llama-2-13b":         {"input": 0.20,   "output": 0.20},
    "llama-2-70b":         {"input": 0.65,   "output": 0.75},
    "llama-3-8b":          {"input": 0.05,   "output": 0.10},
    "llama-3-70b":         {"input": 0.50,   "output": 1.00},
    # Mistral
    "mistral-large":       {"input": 2.00,   "output": 6.00},
    "mistral-medium":      {"input": 0.27,   "output": 0.81},
    "mistral-small":       {"input": 0.14,   "output": 0.42},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated USD cost for an API call.

    Parameters
    ----------
    model:
        Model identifier as it appears in ``COST_TABLE``.
    tokens_in:
        Number of prompt / input tokens.
    tokens_out:
        Number of completion / output tokens.

    Raises
    ------
    ValueError
        If *model* is not present in ``COST_TABLE``.
    """
    if model not in COST_TABLE:
        raise ValueError(f"Model '{model}' not found in COST_TABLE.")
    pricing = COST_TABLE[model]
    return (tokens_in / 1_000_000) * pricing["input"] + \
           (tokens_out / 1_000_000) * pricing["output"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_provider(url: str) -> str:
    """Return the LLM provider name inferred from *url*."""
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
    """Return the model name parsed from request or response JSON."""
    for body in [b for b in (request_body, response_body) if b]:
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                for key in ("model", "modelId", "model_id", "engine"):
                    if key in data:
                        return str(data[key])
        except Exception:
            pass
    return "unknown"


def _tok(response_body: Optional[str]) -> tuple:
    """Return ``(tokens_in, tokens_out)`` parsed from a JSON response body."""
    if not response_body:
        return 0, 0
    try:
        d = json.loads(response_body)
        if isinstance(d, dict):
            if "usage" in d:
                u = d["usage"]
                return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            if "usageMetadata" in d:
                u = d["usageMetadata"]
                return u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
    except Exception:
        pass
    return 0, 0


def _is_llm(url: str, request_body: Optional[str]) -> bool:
    """Return ``True`` if the request is likely directed at an LLM API."""
    url_lower = url.lower()
    llm_keywords = (
        "openai", "anthropic", "google", "gemini",
        "mistral", "cohere", "together", "huggingface", "llama",
    )
    if any(kw in url_lower for kw in llm_keywords):
        return True
    if request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict):
                return any(k in data for k in ("model", "engine", "modelId"))
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """A single LLM API call record.

    Fields are automatically populated from *url*, *request_body*, and
    *response_body* when not supplied explicitly.
    """

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
        """Serialize the entry to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Deserialize a ``LogEntry`` from a plain dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def __repr__(self) -> str:
        return (
            f"LogEntry(provider={self.provider!r}, model={self.model!r}, "
            f"status={self.status_code}, tokens_in={self.tokens_in}, "
            f"tokens_out={self.tokens_out}, cost_usd={self.cost_usd:.6f})"
        )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
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
"""

_SQLITE_INSERT = """
    INSERT OR REPLACE INTO log_entries
      (id, url, method, provider, model, request_body, response_body,
       status_code, latency_ms, tokens_in, tokens_out, cost_usd, timestamp, error)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_CSV_FIELDS = (
    "id", "url", "method", "provider", "model",
    "status_code", "latency_ms", "tokens_in", "tokens_out",
    "cost_usd", "timestamp", "error",
)


class LLMLogger:
    """Store and query LLM API call records.

    Parameters
    ----------
    db_path:
        File path for the SQLite database, or ``":memory:"`` for an
        in-memory store.  For the ``"jsonl"`` backend this parameter is
        unused (pass ``":memory:"`` or any dummy string).
    backend:
        ``"sqlite"`` (default) or ``"jsonl"``.
    """

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite") -> None:
        self.db_path = db_path
        self.backend = backend
        self.entries: List[LogEntry] = []
        self.conn: Optional[sqlite3.Connection] = None
        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "jsonl":
            pass
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'sqlite' or 'jsonl'.")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_SQLITE_SCHEMA)
        self.conn.commit()

    def _sqlite_row_to_entry(self, row: sqlite3.Row) -> LogEntry:
        return LogEntry(**dict(row))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, entry: LogEntry) -> None:
        """Persist a :class:`LogEntry`."""
        if self.backend == "sqlite":
            assert self.conn is not None
            self.conn.execute(_SQLITE_INSERT, (
                entry.id, entry.url, entry.method, entry.provider, entry.model,
                entry.request_body, entry.response_body, entry.status_code,
                entry.latency_ms, entry.tokens_in, entry.tokens_out,
                entry.cost_usd, entry.timestamp, entry.error,
            ))
            self.conn.commit()
        else:
            self.entries.append(entry)

    def count(self) -> int:
        """Return the total number of stored entries."""
        if self.backend == "sqlite":
            assert self.conn is not None
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
        """Return entries matching the given filters, newest first.

        Parameters
        ----------
        model:
            Exact model name match.
        provider:
            Exact provider name match (e.g. ``"openai"``).
        status_code:
            HTTP status code filter.
        since:
            ISO-8601 timestamp lower bound (inclusive).
        """
        if self.backend == "sqlite":
            assert self.conn is not None
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
            return [self._sqlite_row_to_entry(r) for r in rows]
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
        """Write all entries to *path* in JSONL format."""
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.query():
                fh.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Write all entries to *path* in CSV format."""
        entries = self.query()
        if not entries:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_CSV_FIELDS))
            writer.writeheader()
            for entry in entries:
                writer.writerow({k: getattr(entry, k) for k in _CSV_FIELDS})

    def close(self) -> None:
        """Release the SQLite connection (no-op for JSONL backend)."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# ---------------------------------------------------------------------------
# urllib monkey-patching
# ---------------------------------------------------------------------------

_original_urlopen = urllib_request.urlopen
_active_logger: Optional[LLMLogger] = None


def _patched_urlopen(url, data=None, timeout=None, **kwargs):
    """Replacement for ``urllib.request.urlopen`` that logs LLM API calls."""
    global _active_logger

    start = time.monotonic()
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    status_code = 200

    # Normalise url and extract method / body from Request objects
    if isinstance(url, urllib_request.Request):
        url_str = url.full_url
        method = url.get_method()
        if data is None and url.data is not None:
            data = url.data
    else:
        url_str = url if isinstance(url, str) else str(url)
        method = "POST" if data is not None else "GET"

    if data is not None:
        if isinstance(data, bytes):
            request_body = data.decode("utf-8", errors="ignore")
        else:
            request_body = str(data)

    is_llm = _is_llm(url_str, request_body)

    try:
        if timeout is not None:
            response = _original_urlopen(url, data=data, timeout=timeout, **kwargs)
        else:
            response = _original_urlopen(url, data=data, **kwargs)

        status_code = response.status

        if is_llm:
            response_data = response.read()
            response_body = response_data.decode("utf-8", errors="ignore")
            headers = response.headers
            response.close()
            # Reconstruct a readable response so callers can still .read() it
            response = addinfourl(BytesIO(response_data), headers, url_str, status_code)

        if is_llm and _active_logger is not None:
            latency_ms = (time.monotonic() - start) * 1000
            _active_logger.record(LogEntry(
                url=url_str, method=method,
                request_body=request_body, response_body=response_body,
                status_code=status_code, latency_ms=latency_ms,
            ))

        return response

    except Exception as exc:
        if is_llm and _active_logger is not None:
            latency_ms = (time.monotonic() - start) * 1000
            _active_logger.record(LogEntry(
                url=url_str, method=method,
                request_body=request_body, response_body=response_body,
                status_code=status_code, latency_ms=latency_ms,
                error=str(exc),
            ))
        raise


def patch_urllib(active_logger: Optional[LLMLogger] = None) -> None:
    """Monkey-patch ``urllib.request.urlopen`` to log LLM API calls.

    Parameters
    ----------
    active_logger:
        Logger instance that receives recorded entries.  If ``None``, calls
        are intercepted but not stored (useful for testing the hook itself).
    """
    global _active_logger
    _active_logger = active_logger
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore ``urllib.request.urlopen`` to its original implementation."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


@contextmanager
def session(
    log_file: Optional[str] = None,
    backend: str = "jsonl",
    auto_patch: bool = True,
):
    """Context manager that captures LLM API calls within a ``with`` block.

    Parameters
    ----------
    log_file:
        Destination file path.  Defaults to ``"llm_api.jsonl"`` for JSONL
        and ``":memory:"`` for SQLite.
    backend:
        ``"jsonl"`` (default) or ``"sqlite"``.
    auto_patch:
        Whether to monkey-patch ``urllib.request.urlopen`` automatically.

    Yields
    ------
    LLMLogger
        The active logger instance.

    Examples
    --------
    >>> with session("run.jsonl") as log:
    ...     # make LLM API calls here
    ...     print(log.count())
    """
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"
    active = LLMLogger(db_path=log_file if backend == "sqlite" else ":memory:", backend=backend)
    if auto_patch:
        patch_urllib(active)
    try:
        yield active
    finally:
        if auto_patch:
            unpatch_urllib()
        if backend == "jsonl" and log_file != ":memory:":
            active.export_jsonl(log_file)
        active.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Command-line interface for LLM API Logger."""
    parser = argparse.ArgumentParser(
        prog="llm-api-logger",
        description="Log and analyse LLM API calls",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # summary
    p_sum = sub.add_parser("summary", help="Print aggregate statistics")
    p_sum.add_argument("log_file", nargs="?", default="llm_api.jsonl", metavar="FILE")

    # query
    p_qry = sub.add_parser("query", help="Filter and list log entries")
    p_qry.add_argument("log_file", nargs="?", default="llm_api.jsonl", metavar="FILE")
    p_qry.add_argument("--model",    help="Exact model name filter")
    p_qry.add_argument("--provider", help="Exact provider name filter")
    p_qry.add_argument("--limit", type=int, default=20, help="Max rows to display (default 20)")

    # export
    p_exp = sub.add_parser("export", help="Export log to CSV or JSONL")
    p_exp.add_argument("log_file", nargs="?", default="llm_api.jsonl", metavar="FILE")
    p_exp.add_argument("--output", "-o", required=True, help="Output file path")
    p_exp.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return

    log_file: str = args.log_file
    backend = "jsonl" if log_file.endswith(".jsonl") else "sqlite"
    log = LLMLogger(db_path=log_file if backend == "sqlite" else ":memory:", backend=backend)

    if backend == "jsonl" and Path(log_file).exists():
        with open(log_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    log.entries.append(LogEntry.from_dict(json.loads(line)))
                except Exception as exc:
                    logger.warning("Skipping malformed line: %s", exc)

    if args.command == "summary":
        s = log.summary()
        sep = "=" * 60
        print(f"\n{sep}\nLLM API CALL SUMMARY\n{sep}")
        print(f"Total API Calls    : {s['total_calls']}")
        print(f"Total Cost (USD)   : ${s['total_cost_usd']:.4f}")
        print(f"Total Input Tokens : {s['total_tokens_in']:,}")
        print(f"Total Output Tokens: {s['total_tokens_out']:,}")
        print(f"Avg Latency (ms)   : {s['avg_latency_ms']:.2f}")
        print("\nCalls by Model:")
        for mdl, cnt in sorted(s["calls_by_model"].items()):
            cost = s["cost_by_model"].get(mdl, 0.0)
            print(f"  {mdl:<35} {cnt:>5} calls  ${cost:>9.4f}")
        print(f"{sep}\n")

    elif args.command == "query":
        results = log.query(model=args.model, provider=args.provider)
        limit = args.limit
        print(f"\nFound {len(results)} entries\n")
        for entry in results[:limit]:
            print(
                f"  {entry.timestamp[:19]}  {entry.provider:>10}  "
                f"{entry.model:<25}  status={entry.status_code}  "
                f"${entry.cost_usd:.6f}"
            )
        if len(results) > limit:
            print(f"  … and {len(results) - limit} more (use --limit to see more)")
        print()

    elif args.command == "export":
        if args.format == "csv":
            log.export_csv(args.output)
        else:
            log.export_jsonl(args.output)
        print(f"Exported {log.count()} entries to {args.output!r} ({args.format.upper()})")


# Entry point alias used by pyproject.toml
_cli = main


# ---------------------------------------------------------------------------
# Backwards-compatible aliases (kept for third-party code relying on old names)
# ---------------------------------------------------------------------------
LogRecord = LogEntry
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider


if __name__ == "__main__":
    main()
