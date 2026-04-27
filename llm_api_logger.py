"""
LLM API Logger - Log and analyze LLM API calls across providers.

Provides:
- LogEntry dataclass with SHA-256 provenance hashes (request + response)
- LLMLogger with SQLite and JSONL backends
- Cost estimation for 25+ models
- urllib.request.urlopen monkey-patching for transparent capture
- session() context manager for scoped logging
- CLI: summary / query / export / gui
"""

import json
import sqlite3
import csv
import hashlib
import argparse
from dataclasses import dataclass, asdict, field, fields as dc_fields
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from pathlib import Path
from urllib import request as urllib_request
from urllib.response import addinfourl
from io import BytesIO
import time
import uuid

__version__ = "0.2.0"

COST_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o":             {"input":  5.00, "output": 15.00},
    "gpt-4o-mini":        {"input":  0.15, "output":  0.60},
    "gpt-4-turbo":        {"input": 10.00, "output": 30.00},
    "gpt-4":              {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":      {"input":  0.50, "output":  1.50},
    "claude-3-5-sonnet":  {"input":  3.00, "output": 15.00},
    "claude-3-opus":      {"input": 15.00, "output": 75.00},
    "claude-3-sonnet":    {"input":  3.00, "output": 15.00},
    "claude-3-haiku":     {"input":  0.25, "output":  1.25},
    "claude-2.1":         {"input":  8.00, "output": 24.00},
    "claude-2":           {"input":  8.00, "output": 24.00},
    "claude-instant":     {"input":  0.80, "output":  2.40},
    "gemini-pro":         {"input":  0.50, "output":  1.50},
    "gemini-1.5-pro":     {"input":  1.25, "output":  5.00},
    "gemini-1.5-flash":   {"input":  0.075,"output":  0.30},
    "gemini-2.0-flash":   {"input":  0.10, "output":  0.40},
    "palm-2":             {"input":  0.00005, "output": 0.0001},
    "llama-2-7b":         {"input":  0.10, "output":  0.10},
    "llama-2-13b":        {"input":  0.20, "output":  0.20},
    "llama-2-70b":        {"input":  0.65, "output":  0.75},
    "llama-3-8b":         {"input":  0.05, "output":  0.10},
    "llama-3-70b":        {"input":  0.50, "output":  1.00},
    "mistral-large":      {"input":  2.00, "output":  6.00},
    "mistral-medium":     {"input":  0.27, "output":  0.81},
    "mistral-small":      {"input":  0.14, "output":  0.42},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated cost in USD for *tokens_in* input and *tokens_out* output tokens."""
    if model not in COST_TABLE:
        raise ValueError(f"Model {model!r} not in COST_TABLE. Add it or supply cost_usd manually.")
    pricing = COST_TABLE[model]
    return (tokens_in / 1_000_000) * pricing["input"] + (tokens_out / 1_000_000) * pricing["output"]


def _extract_provider(url: str) -> str:
    """Identify the LLM provider from the request URL."""
    u = url.lower()
    for keyword, provider in (
        ("anthropic",   "anthropic"),
        ("groq",        "groq"),        # must precede "openai" — Groq uses /openai/ URL paths
        ("openai",      "openai"),
        ("gemini",      "google"),
        ("google",      "google"),
        ("mistral",     "mistral"),
        ("together",    "together"),
        ("cohere",      "cohere"),
        ("huggingface", "huggingface"),
        ("fireworks",   "fireworks"),
        ("perplexity",  "perplexity"),
    ):
        if keyword in u:
            return provider
    return "unknown"


def _extract_model(request_body: Optional[str], response_body: Optional[str]) -> str:
    """Extract the model name from request or response JSON."""
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


def _tok(response_body: Optional[str]) -> tuple:
    """Parse (tokens_in, tokens_out) from a response body.

    Supports OpenAI (prompt_tokens / completion_tokens),
    Anthropic (input_tokens / output_tokens), and
    Google (usageMetadata.promptTokenCount / candidatesTokenCount).
    """
    if not response_body:
        return 0, 0
    try:
        d = json.loads(response_body)
        if not isinstance(d, dict):
            return 0, 0
        if "usage" in d:
            u = d["usage"]
            # Prefer OpenAI keys; fall back to Anthropic keys
            tok_in  = u.get("prompt_tokens")     if "prompt_tokens"     in u else u.get("input_tokens",  0)
            tok_out = u.get("completion_tokens")  if "completion_tokens" in u else u.get("output_tokens", 0)
            return int(tok_in or 0), int(tok_out or 0)
        if "usageMetadata" in d:
            u = d["usageMetadata"]
            return int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
        pass
    return 0, 0


class _ReplayResponse(addinfourl):
    """addinfourl subclass that exposes .status for Python 3.8 compatibility."""

    @property
    def status(self) -> int:  # type: ignore[override]
        return self.code  # type: ignore[return-value]


@dataclass
class LogEntry:
    """A single captured LLM API call with request metadata, token counts, cost, and SHA-256 hashes."""

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
    request_hash: Optional[str] = None
    response_hash: Optional[str] = None

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
        if (self.tokens_in > 0 or self.tokens_out > 0) and self.cost_usd == 0.0:
            try:
                self.cost_usd = estimate_cost(self.model, self.tokens_in, self.tokens_out)
            except ValueError:
                pass
        # SHA-256 provenance hashes for reproducibility verification
        if self.request_hash is None and self.request_body:
            self.request_hash = hashlib.sha256(self.request_body.encode()).hexdigest()
        if self.response_hash is None and self.response_body:
            self.response_hash = hashlib.sha256(self.response_body.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create a LogEntry from a dict, ignoring unknown keys and supplying defaults for missing ones."""
        known = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class LLMLogger:
    """Store and query LLM API call log entries using a SQLite or in-memory JSONL backend."""

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite") -> None:
        self.db_path = db_path
        self.backend = backend
        self.entries: List[LogEntry] = []
        self.conn: Optional[sqlite3.Connection] = None
        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "jsonl":
            pass  # entries list is the store
        else:
            raise ValueError(f"Unknown backend {backend!r}. Choose 'sqlite' or 'jsonl'.")

    def _init_sqlite(self) -> None:
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS log_entries (
                id            TEXT PRIMARY KEY,
                url           TEXT NOT NULL,
                method        TEXT,
                provider      TEXT,
                model         TEXT,
                request_body  TEXT,
                response_body TEXT,
                status_code   INTEGER,
                latency_ms    REAL,
                tokens_in     INTEGER,
                tokens_out    INTEGER,
                cost_usd      REAL,
                timestamp     TEXT,
                error         TEXT,
                request_hash  TEXT,
                response_hash TEXT
            )
        """)
        # Migrate older databases that pre-date the hash columns
        for col_def in ("request_hash TEXT", "response_hash TEXT"):
            try:
                cur.execute(f"ALTER TABLE log_entries ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already present
        self.conn.commit()

    def record(self, entry: LogEntry) -> None:
        """Persist *entry* to the backend store."""
        if self.backend == "sqlite":
            assert self.conn is not None
            self.conn.execute("""
                INSERT OR REPLACE INTO log_entries
                (id, url, method, provider, model, request_body, response_body,
                 status_code, latency_ms, tokens_in, tokens_out, cost_usd,
                 timestamp, error, request_hash, response_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.url, entry.method, entry.provider, entry.model,
                entry.request_body, entry.response_body, entry.status_code,
                entry.latency_ms, entry.tokens_in, entry.tokens_out,
                entry.cost_usd, entry.timestamp, entry.error,
                entry.request_hash, entry.response_hash,
            ))
            self.conn.commit()
        else:
            self.entries.append(entry)

    def count(self) -> int:
        """Return total number of stored entries."""
        if self.backend == "sqlite":
            assert self.conn is not None
            return self.conn.execute("SELECT COUNT(*) FROM log_entries").fetchone()[0]
        return len(self.entries)

    def query(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        since: Optional[str] = None,
    ) -> List[LogEntry]:
        """Return entries newest-first, optionally filtered by model, provider, status, or timestamp."""
        if self.backend == "sqlite":
            assert self.conn is not None
            self.conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM log_entries WHERE 1=1"
            params: List[Any] = []
            if model:
                sql += " AND model = ?";    params.append(model)
            if provider:
                sql += " AND provider = ?"; params.append(provider)
            if status_code is not None:
                sql += " AND status_code = ?"; params.append(status_code)
            if since:
                sql += " AND timestamp >= ?"; params.append(since)
            sql += " ORDER BY timestamp DESC"
            return [LogEntry.from_dict(dict(r)) for r in self.conn.execute(sql, params).fetchall()]

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
                "total_calls": 0, "total_cost_usd": 0.0,
                "total_tokens_in": 0, "total_tokens_out": 0,
                "avg_latency_ms": 0.0, "calls_by_model": {}, "cost_by_model": {},
            }
        calls_by_model: Dict[str, int] = {}
        cost_by_model: Dict[str, float] = {}
        for e in entries:
            calls_by_model[e.model] = calls_by_model.get(e.model, 0) + 1
            cost_by_model[e.model]  = cost_by_model.get(e.model, 0.0) + e.cost_usd
        return {
            "total_calls":      len(entries),
            "total_cost_usd":   sum(e.cost_usd    for e in entries),
            "total_tokens_in":  sum(e.tokens_in   for e in entries),
            "total_tokens_out": sum(e.tokens_out  for e in entries),
            "avg_latency_ms":   sum(e.latency_ms  for e in entries) / len(entries),
            "calls_by_model":   calls_by_model,
            "cost_by_model":    cost_by_model,
        }

    def export_jsonl(self, path: str) -> None:
        """Write all entries to a JSONL file at *path*."""
        with open(path, "w") as fh:
            for entry in self.query():
                fh.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Write all entries to a CSV file at *path* (request/response bodies excluded)."""
        entries = self.query()
        if not entries:
            return
        fieldnames = [
            "id", "url", "method", "provider", "model", "status_code",
            "latency_ms", "tokens_in", "tokens_out", "cost_usd",
            "timestamp", "error", "request_hash", "response_hash",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow({k: getattr(entry, k, None) for k in fieldnames})


# ---------------------------------------------------------------------------
# urllib monkey-patching
# ---------------------------------------------------------------------------

_original_urlopen = urllib_request.urlopen
_active_logger: Optional[LLMLogger] = None

_LLM_KEYWORDS = (
    "openai", "anthropic", "gemini", "google", "mistral",
    "cohere", "together", "huggingface", "groq", "fireworks", "perplexity",
)


def _is_llm(url: str, request_body: Optional[str]) -> bool:
    """Return True when the request looks like an LLM API call."""
    if any(kw in url.lower() for kw in _LLM_KEYWORDS):
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
    """Replacement for urllib.request.urlopen that logs matching LLM API calls."""
    start = time.time()
    request_body: Optional[str] = None

    if data is not None:
        request_body = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else str(data)

    url_str = url if isinstance(url, str) else getattr(url, "full_url", str(url))
    is_llm = _is_llm(url_str, request_body)

    try:
        response = (
            _original_urlopen(url, data=data, timeout=timeout, **kwargs)
            if timeout is not None
            else _original_urlopen(url, data=data, **kwargs)
        )
        status_code = response.status

        if is_llm and _active_logger is not None:
            raw = response.read()
            response_body = raw.decode("utf-8", errors="ignore")
            # Reconstruct response so callers can still read from it
            response = _ReplayResponse(BytesIO(raw), response.headers, url_str, status_code)
            _active_logger.record(LogEntry(
                url=url_str, method="POST",
                request_body=request_body, response_body=response_body,
                status_code=status_code,
                latency_ms=(time.time() - start) * 1000,
            ))

        return response

    except Exception as exc:
        if is_llm and _active_logger is not None:
            _active_logger.record(LogEntry(
                url=url_str, method="POST",
                request_body=request_body, response_body=None,
                status_code=0,
                latency_ms=(time.time() - start) * 1000,
                error=str(exc),
            ))
        raise


def patch_urllib(logger: Optional[LLMLogger] = None) -> None:
    """Monkey-patch urllib.request.urlopen to capture LLM API calls into *logger*."""
    global _active_logger
    _active_logger = logger
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore the original urllib.request.urlopen."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------


def _load_jsonl_into(logger: LLMLogger, path: str) -> None:
    """Append entries from a JSONL file on disk into *logger*."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                logger.record(LogEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass


@contextmanager
def session(log_file: Optional[str] = None, backend: str = "jsonl", auto_patch: bool = True):
    """Context manager that captures LLM API calls and writes them to *log_file*.

    Example::

        with session("run.jsonl") as log:
            # make LLM calls here (via urllib) …
        print(log.summary())
    """
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"
    logger = LLMLogger(db_path=log_file, backend=backend)
    if backend == "jsonl" and log_file != ":memory:" and Path(log_file).exists():
        _load_jsonl_into(logger, log_file)
    if auto_patch:
        patch_urllib(logger)
    try:
        yield logger
    finally:
        if log_file != ":memory:" and backend == "jsonl":
            logger.export_jsonl(log_file)
        if auto_patch:
            unpatch_urllib()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``llm-api-logger`` command."""
    parser = argparse.ArgumentParser(
        description="LLM API Logger – capture and inspect LLM API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("summary", help="Print aggregate statistics")
    sp.add_argument("log_file", nargs="?", default="llm_api.jsonl")

    qp = sub.add_parser("query", help="List log entries")
    qp.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    qp.add_argument("--model",    help="Filter by model name")
    qp.add_argument("--provider", help="Filter by provider")

    ep = sub.add_parser("export", help="Export logs to CSV or JSONL")
    ep.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    ep.add_argument("--output", "-o", required=True, help="Destination file")
    ep.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv")

    gp = sub.add_parser("gui", help="Open the graphical log viewer")
    gp.add_argument("log_file", nargs="?", help="Log file to open on startup")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "gui":
        try:
            from llm_api_logger_gui import launch_gui
        except ImportError:
            print(
                "Error: tkinter is required for the GUI.\n"
                "Install it with your system package manager, e.g.:\n"
                "  sudo apt install python3-tk\n"
                "  brew install python-tk"
            )
            return
        launch_gui(getattr(args, "log_file", None))
        return

    log_file = args.log_file
    backend = "jsonl" if log_file.endswith(".jsonl") else "sqlite"
    logger = LLMLogger(db_path=log_file, backend=backend)
    if backend == "jsonl" and Path(log_file).exists():
        _load_jsonl_into(logger, log_file)

    if args.command == "summary":
        s = logger.summary()
        print("\n" + "=" * 60)
        print("LLM API CALL SUMMARY")
        print("=" * 60)
        print(f"Total API Calls:      {s['total_calls']}")
        print(f"Total Cost (USD):     ${s['total_cost_usd']:.4f}")
        print(f"Total Input Tokens:   {s['total_tokens_in']:,}")
        print(f"Total Output Tokens:  {s['total_tokens_out']:,}")
        print(f"Average Latency (ms): {s['avg_latency_ms']:.2f}")
        if s["calls_by_model"]:
            print("\nCalls by Model:")
            for model, count in sorted(s["calls_by_model"].items()):
                cost = s["cost_by_model"].get(model, 0.0)
                print(f"  {model:<30} {count:>5} calls  ${cost:>8.4f}")
        print("=" * 60 + "\n")

    elif args.command == "query":
        results = logger.query(model=args.model, provider=args.provider)
        print(f"\nFound {len(results)} entries\n")
        for entry in results[:10]:
            print(f"  {entry.timestamp} | {entry.provider:>12} | {entry.model:<25} | ${entry.cost_usd:.6f}")
        if len(results) > 10:
            print(f"  … and {len(results) - 10} more")
        print()

    elif args.command == "export":
        if args.format == "csv":
            logger.export_csv(args.output)
        else:
            logger.export_jsonl(args.output)
        print(f"Exported {logger.count()} entries to {args.output} ({args.format.upper()})")


# Backwards-compatible aliases
LogRecord     = LogEntry
JSONLBackend  = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider
_cli = main


if __name__ == "__main__":
    main()
