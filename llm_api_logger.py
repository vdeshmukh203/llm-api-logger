"""
LLM API Logger - Structured logging and cost tracking for LLM API calls.

Provides:
- ``LogEntry`` dataclass for structured API call tracking
- ``LLMLogger`` class with SQLite and JSONL backend storage
- Cost estimation for 25+ LLM models across major providers
- ``urllib.request.urlopen`` monkey-patching for automatic, zero-change logging
- ``session`` context manager for scoped logging sessions
- CLI for querying, summarising, and exporting logs
- Optional Tkinter GUI dashboard (``llm-api-logger gui``)
"""

import csv
import io
import json
import logging
import sqlite3
import sys
import argparse
import uuid
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field, fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request
from urllib.response import addinfourl

__version__ = "1.0.0"
__author__ = "Vaibhav Deshmukh"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table – prices in USD per 1 000 000 tokens (as of mid-2025)
# ---------------------------------------------------------------------------

COST_TABLE: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":          {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":     {"input": 0.15,   "output":  0.60},
    "gpt-4-turbo":     {"input": 10.00,  "output": 30.00},
    "gpt-4":           {"input": 30.00,  "output": 60.00},
    "gpt-3.5-turbo":   {"input": 0.50,   "output":  1.50},
    # Anthropic
    "claude-3-5-sonnet": {"input": 3.00,  "output": 15.00},
    "claude-3-opus":     {"input": 15.00, "output": 75.00},
    "claude-3-sonnet":   {"input": 3.00,  "output": 15.00},
    "claude-3-haiku":    {"input": 0.25,  "output":  1.25},
    "claude-2.1":        {"input": 8.00,  "output": 24.00},
    "claude-2":          {"input": 8.00,  "output": 24.00},
    "claude-instant":    {"input": 0.80,  "output":  2.40},
    # Google
    "gemini-pro":        {"input": 0.50,   "output":  1.50},
    "gemini-1.5-pro":    {"input": 1.25,   "output":  5.00},
    "gemini-1.5-flash":  {"input": 0.075,  "output":  0.30},
    "gemini-2.0-flash":  {"input": 0.10,   "output":  0.40},
    "palm-2":            {"input": 0.00005,"output":  0.0001},
    # Meta
    "llama-2-7b":  {"input": 0.10, "output": 0.10},
    "llama-2-13b": {"input": 0.20, "output": 0.20},
    "llama-2-70b": {"input": 0.65, "output": 0.75},
    "llama-3-8b":  {"input": 0.05, "output": 0.10},
    "llama-3-70b": {"input": 0.50, "output": 1.00},
    # Mistral
    "mistral-large":  {"input": 2.00, "output": 6.00},
    "mistral-medium": {"input": 0.27, "output": 0.81},
    "mistral-small":  {"input": 0.14, "output": 0.42},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate the cost in USD for a single LLM API call.

    Parameters
    ----------
    model:
        Model identifier matching a key in :data:`COST_TABLE`.
    tokens_in:
        Number of prompt / input tokens consumed.
    tokens_out:
        Number of completion / output tokens generated.

    Returns
    -------
    float
        Estimated cost in USD.

    Raises
    ------
    ValueError
        If *model* is not present in :data:`COST_TABLE`.
    """
    if model not in COST_TABLE:
        raise ValueError(
            f"Model '{model}' not found in cost table. "
            f"Known models: {sorted(COST_TABLE)}"
        )
    p = COST_TABLE[model]
    return (tokens_in / 1_000_000) * p["input"] + (tokens_out / 1_000_000) * p["output"]


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_provider(url: str) -> str:
    """Infer the LLM provider from a request URL."""
    u = url.lower()
    if "openai" in u:
        return "openai"
    if "anthropic" in u:
        return "anthropic"
    if "google" in u or "gemini" in u:
        return "google"
    if "mistral" in u:
        return "mistral"
    if "together" in u:
        return "together"
    if "cohere" in u:
        return "cohere"
    if "huggingface" in u:
        return "huggingface"
    return "unknown"


def _extract_model(request_body: Optional[str], response_body: Optional[str]) -> str:
    """Extract the model name from a JSON request or response body."""
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


def _extract_tokens(response_body: Optional[str]) -> tuple:
    """Return ``(tokens_in, tokens_out)`` parsed from a JSON response body.

    Handles both OpenAI-style ``usage`` and Google-style ``usageMetadata``
    fields.  Returns ``(0, 0)`` when the body is absent or unparseable.
    """
    if not response_body:
        return 0, 0
    try:
        d = json.loads(response_body)
        if not isinstance(d, dict):
            return 0, 0
        if "usage" in d:
            u = d["usage"]
            return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        if "usageMetadata" in d:
            u = d["usageMetadata"]
            return int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return 0, 0


# Backwards-compatible alias
_tok = _extract_tokens


# ---------------------------------------------------------------------------
# LogEntry dataclass
# ---------------------------------------------------------------------------

_LOG_ENTRY_FIELDS = None  # cached field name set


@dataclass
class LogEntry:
    """A single LLM API call log record.

    Attributes
    ----------
    id:
        UUID4 identifier, auto-generated if omitted.
    url:
        Full request URL.
    method:
        HTTP method (almost always ``"POST"``).
    provider:
        Inferred provider name (openai, anthropic, google, …).
    model:
        Model identifier extracted from request/response JSON.
    request_body:
        Raw request body as a UTF-8 string (may be *None* for GET requests).
    response_body:
        Raw response body as a UTF-8 string.
    status_code:
        HTTP status code of the response.
    latency_ms:
        Round-trip latency in milliseconds.
    tokens_in:
        Input token count extracted from the response usage metadata.
    tokens_out:
        Output token count extracted from the response usage metadata.
    cost_usd:
        Estimated cost in USD based on :data:`COST_TABLE`.
    timestamp:
        ISO-8601 UTC timestamp of the call.
    error:
        Exception message if the request failed; *None* on success.
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
            ti, to = _extract_tokens(self.response_body)
            if ti > 0:
                self.tokens_in = ti
            if to > 0:
                self.tokens_out = to
        if self.tokens_in > 0 and self.tokens_out > 0 and self.cost_usd == 0.0:
            try:
                self.cost_usd = estimate_cost(self.model, self.tokens_in, self.tokens_out)
            except ValueError:
                pass  # unknown model – leave cost at 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entry to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Deserialise a ``LogEntry`` from a dictionary, ignoring unknown keys."""
        global _LOG_ENTRY_FIELDS
        if _LOG_ENTRY_FIELDS is None:
            _LOG_ENTRY_FIELDS = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in _LOG_ENTRY_FIELDS})


# ---------------------------------------------------------------------------
# LLMLogger – storage backend
# ---------------------------------------------------------------------------

class LLMLogger:
    """Stores and queries LLM API call log entries.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file, or ``":memory:"`` for an in-process
        store.  Ignored when *backend* is ``"jsonl"``.
    backend:
        ``"sqlite"`` (default) or ``"jsonl"``.  JSONL entries are kept in
        memory and flushed to *db_path* by the :func:`session` context manager.
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
            raise ValueError(f"Unknown backend '{backend}'. Choose 'sqlite' or 'jsonl'.")

    def _init_sqlite(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS log_entries (
                id          TEXT PRIMARY KEY,
                url         TEXT NOT NULL,
                method      TEXT,
                provider    TEXT,
                model       TEXT,
                request_body  TEXT,
                response_body TEXT,
                status_code   INTEGER,
                latency_ms    REAL,
                tokens_in     INTEGER,
                tokens_out    INTEGER,
                cost_usd      REAL,
                timestamp     TEXT,
                error         TEXT
            )
        """)
        self.conn.commit()

    def record(self, entry: LogEntry) -> None:
        """Persist a :class:`LogEntry` to the configured backend."""
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
        """Return the total number of recorded entries."""
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
        """Return matching log entries, newest first.

        Parameters
        ----------
        model:
            Exact model name filter.
        provider:
            Exact provider name filter.
        status_code:
            Exact HTTP status code filter.
        since:
            ISO-8601 timestamp; only entries at or after this time are returned.
        """
        if self.backend == "sqlite":
            self.conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM log_entries WHERE 1=1"
            params: list = []
            if model:
                sql += " AND model = ?";    params.append(model)
            if provider:
                sql += " AND provider = ?"; params.append(provider)
            if status_code is not None:
                sql += " AND status_code = ?"; params.append(status_code)
            if since:
                sql += " AND timestamp >= ?"; params.append(since)
            sql += " ORDER BY timestamp DESC"
            rows = self.conn.execute(sql, params).fetchall()
            return [LogEntry(**dict(r)) for r in rows]

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
        """Compute aggregate statistics across all recorded entries."""
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
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.query():
                fh.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Write all entries to a CSV file at *path* (excludes raw bodies)."""
        entries = self.query()
        if not entries:
            return
        fieldnames = [
            "id", "url", "method", "provider", "model", "status_code",
            "latency_ms", "tokens_in", "tokens_out", "cost_usd", "timestamp", "error",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                writer.writerow({k: getattr(e, k) for k in fieldnames})

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

_LLM_URL_KEYWORDS = (
    "openai", "anthropic", "google", "gemini",
    "mistral", "cohere", "together", "huggingface", "llama",
)


def _is_llm_request(url: str, request_body: Optional[str]) -> bool:
    """Return True if the request appears to target an LLM API."""
    if any(kw in url.lower() for kw in _LLM_URL_KEYWORDS):
        return True
    if request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict) and any(k in data for k in ("model", "engine", "modelId")):
                return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


_TIMEOUT_UNSET = object()


def _patched_urlopen(url, data=None, timeout=_TIMEOUT_UNSET, **kwargs):
    """Replacement for ``urllib.request.urlopen`` that intercepts LLM calls."""
    start = time.monotonic()
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    status_code = 200

    if data is not None:
        request_body = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else str(data)

    url_str: str = url if isinstance(url, str) else url.full_url
    is_llm = _is_llm_request(url_str, request_body)

    call_kwargs = dict(kwargs)
    if timeout is not _TIMEOUT_UNSET:
        call_kwargs["timeout"] = timeout

    try:
        response = _original_urlopen(url, data=data, **call_kwargs)
        status_code = response.status

        if is_llm:
            raw = response.read()
            response_body = raw.decode("utf-8", errors="ignore")
            headers = response.headers
            response.close()
            # Reconstruct a readable response so callers can still .read() it
            response = addinfourl(io.BytesIO(raw), headers, url_str, status_code)

        if is_llm and _active_logger is not None:
            latency_ms = (time.monotonic() - start) * 1000
            _active_logger.record(LogEntry(
                url=url_str, method="POST",
                request_body=request_body, response_body=response_body,
                status_code=status_code, latency_ms=latency_ms,
            ))
        return response

    except Exception as exc:
        if is_llm and _active_logger is not None:
            latency_ms = (time.monotonic() - start) * 1000
            _active_logger.record(LogEntry(
                url=url_str, method="POST",
                request_body=request_body, response_body=response_body,
                status_code=status_code, latency_ms=latency_ms,
                error=str(exc),
            ))
        raise


def patch_urllib(logger: Optional[LLMLogger] = None) -> None:
    """Replace ``urllib.request.urlopen`` with the logging shim."""
    global _active_logger
    _active_logger = logger
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore the original ``urllib.request.urlopen``."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

@contextmanager
def session(
    log_file: Optional[str] = None,
    backend: str = "jsonl",
    auto_patch: bool = True,
):
    """Context manager that starts a scoped LLM API logging session.

    Parameters
    ----------
    log_file:
        Destination file for persisted logs.  Defaults to
        ``"llm_api.jsonl"`` (JSONL) or ``":memory:"`` (SQLite).
    backend:
        ``"jsonl"`` or ``"sqlite"``.
    auto_patch:
        When *True*, monkey-patches :func:`urllib.request.urlopen` for the
        duration of the ``with`` block.

    Yields
    ------
    LLMLogger
        The active logger instance.
    """
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"
    logger = LLMLogger(db_path=log_file, backend=backend)
    if auto_patch:
        patch_urllib(logger)
    try:
        yield logger
    finally:
        if backend == "jsonl" and log_file != ":memory:":
            logger.export_jsonl(log_file)
        if auto_patch:
            unpatch_urllib()
        logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_logger_from_file(log_file: str) -> LLMLogger:
    """Load a :class:`LLMLogger` from a JSONL or SQLite file."""
    if log_file.endswith(".jsonl"):
        logger = LLMLogger(db_path=log_file, backend="jsonl")
        p = Path(log_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        logger.entries.append(LogEntry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        log.warning("Skipping malformed JSONL line")
    else:
        logger = LLMLogger(db_path=log_file, backend="sqlite")
    return logger


def main() -> None:
    """Entry point for the ``llm-api-logger`` command-line tool."""
    parser = argparse.ArgumentParser(
        description="LLM API Logger – log, query, and export LLM API call records."
    )
    sub = parser.add_subparsers(dest="command")

    # summary
    sp = sub.add_parser("summary", help="Print aggregate statistics.")
    sp.add_argument("log_file", nargs="?", default="llm_api.jsonl")

    # query
    qp = sub.add_parser("query", help="List recent log entries.")
    qp.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    qp.add_argument("--model", help="Filter by model name.")
    qp.add_argument("--provider", help="Filter by provider.")
    qp.add_argument("--limit", type=int, default=20, help="Max rows to display.")

    # export
    ep = sub.add_parser("export", help="Export logs to CSV or JSONL.")
    ep.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    ep.add_argument("--output", "-o", required=True, help="Output file path.")
    ep.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv")

    # gui
    gp = sub.add_parser("gui", help="Launch the Tkinter GUI dashboard.")
    gp.add_argument("log_file", nargs="?", default=None, help="Log file to open on startup.")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "gui":
        from llm_api_logger_gui import launch_gui
        launch_gui(initial_file=args.log_file)
        return

    logger = _load_logger_from_file(args.log_file)

    if args.command == "summary":
        s = logger.summary()
        sep = "=" * 62
        print(f"\n{sep}")
        print("  LLM API CALL SUMMARY")
        print(sep)
        print(f"  Total API Calls  : {s['total_calls']}")
        print(f"  Total Cost (USD) : ${s['total_cost_usd']:.4f}")
        print(f"  Input Tokens     : {s['total_tokens_in']:,}")
        print(f"  Output Tokens    : {s['total_tokens_out']:,}")
        print(f"  Avg Latency (ms) : {s['avg_latency_ms']:.1f}")
        if s["calls_by_model"]:
            print(f"\n  {'Model':<30} {'Calls':>6}  {'Cost':>10}")
            print("  " + "-" * 50)
            for mdl, cnt in sorted(s["calls_by_model"].items()):
                cost = s["cost_by_model"].get(mdl, 0.0)
                print(f"  {mdl:<30} {cnt:>6}  ${cost:>9.4f}")
        print(f"{sep}\n")

    elif args.command == "query":
        results = logger.query(model=args.model, provider=args.provider)
        limit = args.limit
        print(f"\nFound {len(results)} entr{'y' if len(results)==1 else 'ies'}\n")
        print(f"  {'Timestamp':<26} {'Provider':>10}  {'Model':<22} {'Status':>6}  {'Cost':>10}")
        print("  " + "-" * 80)
        for e in results[:limit]:
            print(
                f"  {e.timestamp:<26} {e.provider:>10}  {e.model:<22} "
                f"{e.status_code:>6}  ${e.cost_usd:>9.6f}"
            )
        if len(results) > limit:
            print(f"  … and {len(results) - limit} more (use --limit to show more)")
        print()

    elif args.command == "export":
        if args.format == "csv":
            logger.export_csv(args.output)
        else:
            logger.export_jsonl(args.output)
        print(f"Exported {logger.count()} entries to {args.output} ({args.format.upper()})")


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

LogRecord = LogEntry
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider
_is_llm = _is_llm_request
LoggingSession = session


if __name__ == "__main__":
    main()
