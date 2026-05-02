"""
LLM API Logger — structured logging and cost tracking for LLM API calls.

Provides:
- ``LogEntry`` dataclass with SHA-256 provenance hashing
- ``LLMLogger`` with SQLite and JSONL storage backends
- Cost estimation for 25+ models (OpenAI, Anthropic, Google, Mistral, Meta)
- Transparent ``urllib.request.urlopen`` patching for automatic capture
- ``session()`` context manager for scoped logging
- CLI with ``summary``, ``query``, ``export``, and ``gui`` subcommands
"""

import csv
import hashlib
import json
import logging
import sqlite3
import sys
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib import request as urllib_request
import time
import urllib.response as _urllib_response
import uuid

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table (USD per 1 000 000 tokens)
# ---------------------------------------------------------------------------

COST_TABLE: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":           {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":      {"input": 10.00, "output": 30.00},
    "gpt-4":            {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":    {"input": 0.50,  "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet":{"input": 3.00,  "output": 15.00},
    "claude-3-opus":    {"input": 15.00, "output": 75.00},
    "claude-3-sonnet":  {"input": 3.00,  "output": 15.00},
    "claude-3-haiku":   {"input": 0.25,  "output": 1.25},
    "claude-2.1":       {"input": 8.00,  "output": 24.00},
    "claude-2":         {"input": 8.00,  "output": 24.00},
    "claude-instant":   {"input": 0.80,  "output": 2.40},
    # Google
    "gemini-pro":       {"input": 0.50,   "output": 1.50},
    "gemini-1.5-pro":   {"input": 1.25,   "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075,  "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10,   "output": 0.40},
    "palm-2":           {"input": 0.00005,"output": 0.0001},
    # Meta (via Together / OpenRouter)
    "llama-2-7b":       {"input": 0.10,  "output": 0.10},
    "llama-2-13b":      {"input": 0.20,  "output": 0.20},
    "llama-2-70b":      {"input": 0.65,  "output": 0.75},
    "llama-3-8b":       {"input": 0.05,  "output": 0.10},
    "llama-3-70b":      {"input": 0.50,  "output": 1.00},
    # Mistral
    "mistral-large":    {"input": 2.00,  "output": 6.00},
    "mistral-medium":   {"input": 0.27,  "output": 0.81},
    "mistral-small":    {"input": 0.14,  "output": 0.42},
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated USD cost for one API call.

    Parameters
    ----------
    model:
        Model key exactly as it appears in :data:`COST_TABLE`.
    tokens_in:
        Number of prompt/input tokens consumed.
    tokens_out:
        Number of completion/output tokens generated.

    Returns
    -------
    float
        Estimated cost in US dollars.

    Raises
    ------
    ValueError
        If *model* is not present in :data:`COST_TABLE`.
    """
    if model not in COST_TABLE:
        raise ValueError(
            f"Model '{model}' not found in COST_TABLE. "
            f"Available models: {sorted(COST_TABLE)}"
        )
    pricing = COST_TABLE[model]
    return (tokens_in / 1_000_000) * pricing["input"] + \
           (tokens_out / 1_000_000) * pricing["output"]


def _extract_provider(url: str) -> str:
    """Identify the LLM provider from a request URL.

    Checks for well-known hostnames/substrings in *url* (case-insensitive).
    Returns ``"unknown"`` when no match is found.
    """
    url_lower = url.lower()
    providers = [
        ("openai",      "openai"),
        ("anthropic",   "anthropic"),
        ("google",      "google"),
        ("gemini",      "google"),
        ("mistral",     "mistral"),
        ("together",    "together"),
        ("cohere",      "cohere"),
        ("huggingface", "huggingface"),
    ]
    for keyword, name in providers:
        if keyword in url_lower:
            return name
    return "unknown"


def _extract_model(
    request_body: Optional[str],
    response_body: Optional[str],
) -> str:
    """Extract a model name from JSON request or response bodies.

    Inspects the ``model``, ``modelId``, ``model_id``, and ``engine`` keys of
    the first parseable JSON body.  Returns ``"unknown"`` on failure.
    """
    for body in filter(None, [request_body, response_body]):
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                for key in ("model", "modelId", "model_id", "engine"):
                    if key in data:
                        return str(data[key])
        except (json.JSONDecodeError, ValueError):
            continue
    return "unknown"


def _extract_tokens(response_body: Optional[str]) -> Tuple[int, int]:
    """Parse prompt and completion token counts from a JSON response body.

    Handles both the OpenAI ``usage`` format and the Google ``usageMetadata``
    format.  Returns ``(0, 0)`` when the body is absent or unparseable.
    """
    if not response_body:
        return 0, 0
    try:
        data = json.loads(response_body)
        if not isinstance(data, dict):
            return 0, 0
        if "usage" in data:
            u = data["usage"]
            return (
                int(u.get("prompt_tokens", 0)),
                int(u.get("completion_tokens", 0)),
            )
        if "usageMetadata" in data:
            u = data["usageMetadata"]
            return (
                int(u.get("promptTokenCount", 0)),
                int(u.get("candidatesTokenCount", 0)),
            )
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        pass
    return 0, 0


def _sha256(text: str) -> str:
    """Return the hex-encoded SHA-256 digest of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LogEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """Structured record of a single LLM API HTTP call.

    Fields are populated automatically on construction via :meth:`__post_init__`:
    *provider*, *model*, *tokens_in*, *tokens_out*, *cost_usd*, and
    *content_hash* are derived from *url*, *request_body*, and *response_body*
    when not supplied explicitly.

    Attributes
    ----------
    id:
        UUID4 string uniquely identifying this record.
    url:
        Full request URL.
    method:
        HTTP method (default ``"POST"``).
    provider:
        Inferred provider name (e.g. ``"openai"``, ``"anthropic"``).
    model:
        Model identifier extracted from the request/response body.
    request_body:
        Raw request body as a UTF-8 string, or ``None``.
    response_body:
        Raw response body as a UTF-8 string, or ``None``.
    status_code:
        HTTP status code.
    latency_ms:
        Round-trip latency in milliseconds.
    tokens_in:
        Prompt token count (0 if unavailable).
    tokens_out:
        Completion token count (0 if unavailable).
    cost_usd:
        Estimated cost in US dollars (0.0 if model is not in cost table).
    timestamp:
        UTC ISO-8601 timestamp of when the entry was created.
    error:
        Exception message if the request failed, otherwise ``None``.
    content_hash:
        SHA-256 hex digest of ``request_body + response_body`` for provenance.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    method: str = "POST"
    provider: str = ""
    model: str = ""
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    status_code: int = 200
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None
    content_hash: Optional[str] = None

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
                self.cost_usd = estimate_cost(
                    self.model, self.tokens_in, self.tokens_out
                )
            except ValueError:
                pass
        if self.content_hash is None:
            payload = (self.request_body or "") + (self.response_body or "")
            if payload:
                self.content_hash = _sha256(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain :class:`dict`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Deserialise from a :class:`dict` (e.g. parsed from JSONL)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# LLMLogger: storage backend
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
        error        TEXT,
        content_hash TEXT
    )
"""

_SQLITE_INSERT = """
    INSERT OR REPLACE INTO log_entries
        (id, url, method, provider, model, request_body, response_body,
         status_code, latency_ms, tokens_in, tokens_out, cost_usd,
         timestamp, error, content_hash)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_CSV_FIELDS = [
    "id", "url", "method", "provider", "model", "status_code",
    "latency_ms", "tokens_in", "tokens_out", "cost_usd",
    "timestamp", "error", "content_hash",
]


class LLMLogger:
    """Persist and query :class:`LogEntry` records.

    Parameters
    ----------
    db_path:
        File path for the database.  Use ``":memory:"`` for an in-process
        SQLite database (useful in tests).  For the JSONL backend the path
        is used only by :meth:`export_jsonl`; entries are kept in memory
        during the session.
    backend:
        Storage engine: ``"sqlite"`` (default) or ``"jsonl"``.

    Raises
    ------
    ValueError
        If *backend* is neither ``"sqlite"`` nor ``"jsonl"``.
    """

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite") -> None:
        if backend not in ("sqlite", "jsonl"):
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'sqlite' or 'jsonl'."
            )
        self.db_path = db_path
        self.backend = backend
        self.entries: List[LogEntry] = []
        self.conn: Optional[sqlite3.Connection] = None
        if backend == "sqlite":
            self._init_sqlite()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute(_SQLITE_SCHEMA)
        self.conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> LogEntry:
        return LogEntry(**dict(row))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, entry: LogEntry) -> None:
        """Persist a :class:`LogEntry`."""
        if self.backend == "sqlite":
            assert self.conn is not None
            self.conn.execute(
                _SQLITE_INSERT,
                (
                    entry.id, entry.url, entry.method, entry.provider,
                    entry.model, entry.request_body, entry.response_body,
                    entry.status_code, entry.latency_ms, entry.tokens_in,
                    entry.tokens_out, entry.cost_usd, entry.timestamp,
                    entry.error, entry.content_hash,
                ),
            )
            self.conn.commit()
        else:
            self.entries.append(entry)

    def count(self) -> int:
        """Return the total number of stored entries."""
        if self.backend == "sqlite":
            assert self.conn is not None
            row = self.conn.execute("SELECT COUNT(*) FROM log_entries").fetchone()
            return int(row[0])
        return len(self.entries)

    def query(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[LogEntry]:
        """Retrieve stored entries with optional filters.

        Parameters
        ----------
        model:
            Keep only entries whose ``model`` field equals this value.
        provider:
            Keep only entries whose ``provider`` field equals this value.
        status_code:
            Keep only entries with this HTTP status code.
        since:
            ISO-8601 timestamp lower bound (inclusive).
        until:
            ISO-8601 timestamp upper bound (inclusive).

        Returns
        -------
        List[LogEntry]
            Matching entries ordered by timestamp descending.
        """
        if self.backend == "sqlite":
            assert self.conn is not None
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
            if until:
                sql += " AND timestamp <= ?"
                params.append(until)
            sql += " ORDER BY timestamp DESC"
            rows = self.conn.execute(sql, params).fetchall()
            return [self._row_to_entry(r) for r in rows]
        # JSONL in-memory path
        result = list(self.entries)
        if model:
            result = [e for e in result if e.model == model]
        if provider:
            result = [e for e in result if e.provider == provider]
        if status_code is not None:
            result = [e for e in result if e.status_code == status_code]
        if since:
            result = [e for e in result if e.timestamp >= since]
        if until:
            result = [e for e in result if e.timestamp <= until]
        return sorted(result, key=lambda e: e.timestamp, reverse=True)

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all stored entries.

        Returns
        -------
        dict
            Keys: ``total_calls``, ``total_cost_usd``, ``total_tokens_in``,
            ``total_tokens_out``, ``avg_latency_ms``, ``calls_by_model``,
            ``cost_by_model``.
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

    def export_jsonl(self, path: str) -> None:
        """Write all entries to *path* in JSONL format."""
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.query():
                fh.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Write all entries to *path* in CSV format (bodies excluded)."""
        entries = self.query()
        if not entries:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for entry in entries:
                writer.writerow({k: getattr(entry, k) for k in _CSV_FIELDS})


# ---------------------------------------------------------------------------
# urllib.request monkey-patching
# ---------------------------------------------------------------------------

_original_urlopen = urllib_request.urlopen
_active_logger: Optional[LLMLogger] = None

_LLM_URL_KEYWORDS = frozenset([
    "openai", "anthropic", "google", "gemini",
    "mistral", "cohere", "together", "huggingface", "llama",
])
_LLM_BODY_KEYS = frozenset(["model", "engine", "modelId"])


def _is_llm_request(url: str, request_body: Optional[str]) -> bool:
    """Heuristically determine whether a request targets an LLM API."""
    url_lower = url.lower()
    if any(kw in url_lower for kw in _LLM_URL_KEYWORDS):
        return True
    if request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict) and _LLM_BODY_KEYS.intersection(data):
                return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


def _patched_urlopen(url, data=None, timeout=None, **kwargs):  # type: ignore[no-untyped-def]
    """Drop-in replacement for ``urllib.request.urlopen`` that captures LLM traffic."""
    start = time.monotonic()
    status_code = 0
    request_body: Optional[str] = None
    response_body: Optional[str] = None

    if data is not None:
        request_body = (
            data.decode("utf-8", errors="replace")
            if isinstance(data, (bytes, bytearray))
            else str(data)
        )

    url_str: str = url if isinstance(url, str) else url.full_url
    is_llm = _is_llm_request(url_str, request_body)

    call_kw = dict(kwargs)
    if timeout is not None:
        call_kw["timeout"] = timeout

    try:
        response = _original_urlopen(url, data=data, **call_kw)
        status_code = response.status

        if is_llm:
            raw = response.read()
            response_body = raw.decode("utf-8", errors="replace")
            # Reconstruct a readable response so callers can still read it.
            response = _urllib_response.addinfourl(
                BytesIO(raw), response.headers, url_str, status_code
            )

        if is_llm and _active_logger is not None:
            latency_ms = (time.monotonic() - start) * 1000
            _active_logger.record(
                LogEntry(
                    url=url_str,
                    method="POST",
                    request_body=request_body,
                    response_body=response_body,
                    status_code=status_code,
                    latency_ms=latency_ms,
                )
            )
        return response

    except Exception as exc:
        if is_llm and _active_logger is not None:
            latency_ms = (time.monotonic() - start) * 1000
            _active_logger.record(
                LogEntry(
                    url=url_str,
                    method="POST",
                    request_body=request_body,
                    response_body=response_body,
                    status_code=getattr(exc, "code", 0) or status_code,
                    latency_ms=latency_ms,
                    error=str(exc),
                )
            )
        raise


def patch_urllib(log: Optional[LLMLogger] = None) -> None:
    """Replace ``urllib.request.urlopen`` with the logging wrapper.

    Parameters
    ----------
    log:
        :class:`LLMLogger` instance to receive captured entries.  When
        ``None``, requests are intercepted but not stored.
    """
    global _active_logger
    _active_logger = log
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore the original ``urllib.request.urlopen``."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


@contextmanager
def session(
    log_file: Optional[str] = None,
    backend: str = "jsonl",
    auto_patch: bool = True,
) -> Generator[LLMLogger, None, None]:
    """Context manager that creates a logging session and optionally patches urllib.

    Parameters
    ----------
    log_file:
        Path to write logs.  Defaults to ``"llm_api.jsonl"`` for the JSONL
        backend and ``":memory:"`` for SQLite.
    backend:
        ``"jsonl"`` (default) or ``"sqlite"``.
    auto_patch:
        When ``True`` (default), urllib is patched for the duration of the
        ``with`` block.

    Yields
    ------
    LLMLogger
        The active logger instance.

    Example
    -------
    >>> with session() as log:
    ...     # make LLM API calls — they are captured automatically
    ...     pass
    >>> print(log.summary())
    """
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"
    log = LLMLogger(db_path=log_file, backend=backend)
    if auto_patch:
        patch_urllib(log)
    try:
        yield log
    finally:
        if auto_patch:
            unpatch_urllib()
        if backend == "jsonl" and log_file != ":memory:":
            log.export_jsonl(log_file)


# ---------------------------------------------------------------------------
# Helper: load a log file into an LLMLogger
# ---------------------------------------------------------------------------

def _load_log(log_file: str) -> LLMLogger:
    """Return an :class:`LLMLogger` pre-populated from *log_file*."""
    if log_file.endswith(".jsonl"):
        log = LLMLogger(db_path=log_file, backend="jsonl")
        p = Path(log_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    log.entries.append(LogEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.warning("Skipping malformed line in %s", log_file)
    else:
        log = LLMLogger(db_path=log_file, backend="sqlite")
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_summary(args: argparse.Namespace) -> None:
    log = _load_log(args.log_file)
    s = log.summary()
    sep = "=" * 60
    print(f"\n{sep}")
    print("LLM API CALL SUMMARY")
    print(sep)
    print(f"Total API Calls   : {s['total_calls']}")
    print(f"Total Cost (USD)  : ${s['total_cost_usd']:.4f}")
    print(f"Input Tokens      : {s['total_tokens_in']:,}")
    print(f"Output Tokens     : {s['total_tokens_out']:,}")
    print(f"Avg Latency (ms)  : {s['avg_latency_ms']:.2f}")
    print("\nCalls by Model:")
    for model_name, n in sorted(s["calls_by_model"].items()):
        cost = s["cost_by_model"].get(model_name, 0.0)
        print(f"  {model_name:<32} {n:>5} calls  ${cost:>8.4f}")
    print(f"{sep}\n")


def _cmd_query(args: argparse.Namespace) -> None:
    log = _load_log(args.log_file)
    results = log.query(
        model=args.model,
        provider=args.provider,
        status_code=args.status_code,
        since=args.since,
        until=args.until,
    )
    print(f"\nFound {len(results)} entr{'y' if len(results) == 1 else 'ies'}\n")
    for entry in results[:20]:
        print(
            f"  {entry.timestamp}  {entry.provider:>12}  "
            f"{entry.model:<24}  ${entry.cost_usd:.6f}  {entry.latency_ms:.1f}ms"
        )
    if len(results) > 20:
        print(f"  … and {len(results) - 20} more")
    print()


def _cmd_export(args: argparse.Namespace) -> None:
    log = _load_log(args.log_file)
    if args.format == "csv":
        log.export_csv(args.output)
    else:
        log.export_jsonl(args.output)
    print(f"Exported {log.count()} entries to {args.output} ({args.format.upper()})")


def _cmd_gui(args: argparse.Namespace) -> None:
    """Launch the web dashboard."""
    from llm_api_logger_gui import serve  # type: ignore[import]
    serve(log_file=args.log_file, host=args.host, port=args.port)


def main() -> None:
    """Entry point for the ``llm-api-logger`` command-line tool."""
    parser = argparse.ArgumentParser(
        prog="llm-api-logger",
        description="Log and analyse LLM API calls.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # summary
    p_sum = sub.add_parser("summary", help="Print aggregate statistics")
    p_sum.add_argument("log_file", nargs="?", default="llm_api.jsonl")

    # query
    p_q = sub.add_parser("query", help="List matching log entries")
    p_q.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    p_q.add_argument("--model",       help="Filter by model name")
    p_q.add_argument("--provider",    help="Filter by provider name")
    p_q.add_argument("--status-code", dest="status_code", type=int,
                     help="Filter by HTTP status code")
    p_q.add_argument("--since",  metavar="ISO8601", help="Earliest timestamp (inclusive)")
    p_q.add_argument("--until",  metavar="ISO8601", help="Latest timestamp (inclusive)")

    # export
    p_ex = sub.add_parser("export", help="Export logs to CSV or JSONL")
    p_ex.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    p_ex.add_argument("--output", "-o", required=True, help="Output file path")
    p_ex.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv",
                      help="Output format (default: csv)")

    # gui
    p_gui = sub.add_parser("gui", help="Launch the web dashboard")
    p_gui.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    p_gui.add_argument("--host", default="127.0.0.1")
    p_gui.add_argument("--port", type=int, default=7823)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    dispatch = {
        "summary": _cmd_summary,
        "query":   _cmd_query,
        "export":  _cmd_export,
        "gui":     _cmd_gui,
    }
    dispatch[args.command](args)


# Backwards-compatible aliases kept for downstream code
_cli = main
LogRecord = LogEntry
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider
_tok = _extract_tokens

if __name__ == "__main__":
    main()
