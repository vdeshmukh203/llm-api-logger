"""
LLMLogger — urllib-based middleware logger for LLM API calls.

Provides LogEntry, LLMLogger (SQLite/JSONL backends), cost estimation,
urllib monkey-patching, and the LoggingSession context manager.
"""

import json
import sqlite3
import csv
import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime
from io import BytesIO
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path
from urllib import request as urllib_request
from urllib.response import addinfourl
import time
import uuid

__version__ = "0.2.0"

COST_TABLE = {
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
    """Estimate the cost of an LLM API call in USD."""
    if model not in COST_TABLE:
        raise ValueError(f"Model '{model}' not found in cost table.")
    pricing = COST_TABLE[model]
    return (tokens_in / 1_000_000) * pricing["input"] + (tokens_out / 1_000_000) * pricing["output"]


def _extract_provider(url: str) -> str:
    """Extract LLM provider name from a URL string."""
    url_lower = url.lower()
    for keyword, provider in [
        ("openai", "openai"), ("anthropic", "anthropic"),
        ("google", "google"), ("gemini", "google"),
        ("mistral", "mistral"), ("together", "together"),
        ("cohere", "cohere"), ("huggingface", "huggingface"),
    ]:
        if keyword in url_lower:
            return provider
    return "unknown"


def _extract_model(request_body: Optional[str], response_body: Optional[str]) -> str:
    """Extract model name from request or response JSON body."""
    for body in [b for b in [request_body, response_body] if b]:
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
    """Return (tokens_in, tokens_out) from a JSON response body."""
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

    def __post_init__(self):
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
    """Logger for LLM API calls with SQLite or JSONL backends."""

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite"):
        self.db_path = db_path
        self.backend = backend
        self.entries: List[LogEntry] = []
        self.conn = None
        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "jsonl":
            pass
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_sqlite(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS log_entries (
                id TEXT PRIMARY KEY, url TEXT NOT NULL, method TEXT,
                provider TEXT, model TEXT, request_body TEXT, response_body TEXT,
                status_code INTEGER, latency_ms REAL, tokens_in INTEGER,
                tokens_out INTEGER, cost_usd REAL, timestamp TEXT, error TEXT
            )
        """)
        self.conn.commit()

    def record(self, entry: LogEntry) -> None:
        """Persist a LogEntry to the configured backend."""
        if self.backend == "sqlite":
            self.conn.execute("""
                INSERT OR REPLACE INTO log_entries
                (id, url, method, provider, model, request_body, response_body,
                 status_code, latency_ms, tokens_in, tokens_out, cost_usd, timestamp, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry.id, entry.url, entry.method, entry.provider, entry.model,
                  entry.request_body, entry.response_body, entry.status_code,
                  entry.latency_ms, entry.tokens_in, entry.tokens_out,
                  entry.cost_usd, entry.timestamp, entry.error))
            self.conn.commit()
        else:
            self.entries.append(entry)

    def count(self) -> int:
        if self.backend == "sqlite":
            return self.conn.execute("SELECT COUNT(*) FROM log_entries").fetchone()[0]
        return len(self.entries)

    def query(self, model: Optional[str] = None, provider: Optional[str] = None,
              status_code: Optional[int] = None, since: Optional[str] = None) -> List[LogEntry]:
        """Return log entries with optional filtering."""
        if self.backend == "sqlite":
            self.conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM log_entries WHERE 1=1"
            params: list = []
            if model:
                sql += " AND model = ?"; params.append(model)
            if provider:
                sql += " AND provider = ?"; params.append(provider)
            if status_code:
                sql += " AND status_code = ?"; params.append(status_code)
            if since:
                sql += " AND timestamp >= ?"; params.append(since)
            sql += " ORDER BY timestamp DESC"
            return [LogEntry(**dict(r)) for r in self.conn.execute(sql, params).fetchall()]
        entries = self.entries[:]
        if model:
            entries = [e for e in entries if e.model == model]
        if provider:
            entries = [e for e in entries if e.provider == provider]
        if status_code:
            entries = [e for e in entries if e.status_code == status_code]
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)

    def summary(self) -> Dict[str, Any]:
        entries = self.query()
        if not entries:
            return {"total_calls": 0, "total_cost_usd": 0.0, "total_tokens_in": 0,
                    "total_tokens_out": 0, "calls_by_model": {}, "cost_by_model": {},
                    "avg_latency_ms": 0.0}
        calls_by_model: dict = {}
        cost_by_model: dict = {}
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
        with open(path, "w") as f:
            for entry in self.query():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        entries = self.query()
        if not entries:
            return
        fieldnames = ["id", "url", "method", "provider", "model", "status_code",
                      "latency_ms", "tokens_in", "tokens_out", "cost_usd", "timestamp", "error"]
        with open(path, "w", newline="") as f:
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
    """Return True if the URL or request body looks like an LLM API call."""
    url_lower = url.lower()
    llm_keywords = ["openai", "anthropic", "google", "gemini", "mistral",
                    "cohere", "together", "huggingface", "llama"]
    if any(kw in url_lower for kw in llm_keywords):
        return True
    if request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict) and any(k in data for k in ("model", "engine", "modelId")):
                return True
        except Exception:
            pass
    return False


def _patched_urlopen(url, data=None, timeout=None, **kwargs):
    """Drop-in replacement for urllib.request.urlopen that logs LLM calls."""
    start_time = time.time()
    request_body = None
    response_body = None
    status_code = 200
    if data is not None:
        request_body = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else str(data)
    url_str = url if isinstance(url, str) else url.full_url
    is_llm = _is_llm(url_str, request_body)
    try:
        call_kwargs = {"data": data}
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        call_kwargs.update(kwargs)
        response = _original_urlopen(url, **call_kwargs)
        status_code = response.status
        if is_llm:
            response_headers = response.headers
            response_data = response.read()
            response.close()
            response_body = response_data.decode("utf-8", errors="ignore")
            response = addinfourl(BytesIO(response_data), response_headers, url_str, status_code)
        if is_llm and _active_logger:
            latency_ms = (time.time() - start_time) * 1000
            entry = LogEntry(url=url_str, method="POST", request_body=request_body,
                             response_body=response_body, status_code=status_code,
                             latency_ms=latency_ms)
            _active_logger.record(entry)
        return response
    except Exception as e:
        if is_llm and _active_logger:
            latency_ms = (time.time() - start_time) * 1000
            entry = LogEntry(url=url_str, method="POST", request_body=request_body,
                             response_body=response_body, status_code=status_code,
                             latency_ms=latency_ms, error=str(e))
            _active_logger.record(entry)
        raise


def patch_urllib(logger: Optional[LLMLogger] = None) -> None:
    """Patch urllib.request.urlopen to automatically log LLM API calls."""
    global _active_logger
    _active_logger = logger
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore the original urllib.request.urlopen."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


@contextmanager
def session(log_file: Optional[str] = None, backend: str = "jsonl", auto_patch: bool = True):
    """Context manager providing a scoped LLM API logging session."""
    if log_file is None:
        log_file = ":memory:" if backend == "sqlite" else "llm_api.jsonl"
    logger = LLMLogger(db_path=log_file, backend=backend)
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
    """Command-line interface for LLM API Logger."""
    parser = argparse.ArgumentParser(
        description="LLM API Logger — Log and analyse LLM API calls"
    )
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("summary", help="Show summary statistics")
    sp.add_argument("log_file", nargs="?", default="llm_api.jsonl")

    qp = sub.add_parser("query", help="Query log entries")
    qp.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    qp.add_argument("--model")
    qp.add_argument("--provider")

    ep = sub.add_parser("export", help="Export logs")
    ep.add_argument("log_file", nargs="?", default="llm_api.jsonl")
    ep.add_argument("--output", "-o", required=True)
    ep.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv")

    gp = sub.add_parser("gui", help="Open the dashboard GUI")
    gp.add_argument("log_file", nargs="?", default=None)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "gui":
        from .gui import launch
        launch(args.log_file)
        return

    log_file = args.log_file
    backend = "jsonl" if log_file.endswith(".jsonl") else "sqlite"
    logger = LLMLogger(db_path=log_file, backend=backend)
    if backend == "jsonl" and Path(log_file).exists():
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    try:
                        logger.entries.append(LogEntry.from_dict(json.loads(line)))
                    except Exception:
                        pass

    if args.command == "summary":
        s = logger.summary()
        print("\n" + "=" * 60)
        print("LLM API CALL SUMMARY")
        print("=" * 60)
        print(f"Total API Calls   : {s['total_calls']}")
        print(f"Total Cost (USD)  : ${s['total_cost_usd']:.4f}")
        print(f"Total Tokens in   : {s['total_tokens_in']:,}")
        print(f"Total Tokens out  : {s['total_tokens_out']:,}")
        print(f"Avg Latency (ms)  : {s['avg_latency_ms']:.2f}")
        print("\nCalls by Model:")
        for model, count in sorted(s["calls_by_model"].items()):
            cost = s["cost_by_model"].get(model, 0.0)
            print(f"  {model:<30} {count:>5} calls  ${cost:>8.4f}")
        print("=" * 60 + "\n")

    elif args.command == "query":
        results = logger.query(model=args.model, provider=args.provider)
        print(f"\nFound {len(results)} entries\n")
        for entry in results[:10]:
            print(f"  {entry.timestamp} | {entry.provider:>10} | {entry.model:<20} | ${entry.cost_usd:.6f}")
        if len(results) > 10:
            print(f"  … and {len(results) - 10} more")
        print()

    elif args.command == "export":
        if args.format == "csv":
            logger.export_csv(args.output)
            print(f"Exported {logger.count()} entries to {args.output} (CSV)")
        else:
            logger.export_jsonl(args.output)
            print(f"Exported {logger.count()} entries to {args.output} (JSONL)")


# Backwards-compatible aliases
LogRecord = LogEntry
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider
_cli = main
