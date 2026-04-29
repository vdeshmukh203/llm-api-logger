"""
LLM API Logger - Complete implementation for logging and analyzing LLM API calls.

Provides:
- LogEntry dataclass for structured API call tracking
- LLMLogger class with SQLite/JSONL backend storage
- Cost estimation for 10+ LLM models
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
from urllib.response import addinfourl
from urllib.error import URLError
from io import BytesIO
import time
import uuid

__version__ = "1.0.0"

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
    """Estimate the cost of an LLM API call."""
    if model not in COST_TABLE:
        raise ValueError(f"Model '{model}' not found in cost table.")
    pricing = COST_TABLE[model]
    input_cost = (tokens_in / 1_000_000) * pricing["input"]
    output_cost = (tokens_out / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def _extract_provider(url: str) -> str:
    """Extract LLM provider from URL."""
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
    """Extract model name from request or response body."""
    bodies = [b for b in [request_body, response_body] if b]
    for body in bodies:
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                for key in ["model", "modelId", "model_id", "engine"]:
                    if key in data:
                        return str(data[key])
        except Exception:
            pass
    return "unknown"


def _tok(rs: Optional[str]) -> tuple:
    """Extract token counts from response body."""
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
        """Post-initialization processing."""
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
        """Convert LogEntry to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create LogEntry from dictionary."""
        return cls(**data)


class LLMLogger:
    """Main logger class for storing and querying LLM API calls."""

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite"):
        """Initialize LLMLogger."""
        self.db_path = db_path
        self.backend = backend
        self.entries: List[LogEntry] = []
        self.conn = None
        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "jsonl":
            self.entries = []
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_sqlite(self):
        """Initialize SQLite database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute("""
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
        """Record a log entry."""
        if self.backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("""
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
        """Get total number of logged entries."""
        if self.backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM log_entries")
            count = cursor.fetchone()[0]
            return count
        else:
            return len(self.entries)

    def query(self, model: Optional[str] = None, provider: Optional[str] = None,
              status_code: Optional[int] = None, since: Optional[str] = None) -> List[LogEntry]:
        """Query log entries with optional filtering."""
        if self.backend == "sqlite":
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            query = "SELECT * FROM log_entries WHERE 1=1"
            params = []
            if model:
                query += " AND model = ?"
                params.append(model)
            if provider:
                query += " AND provider = ?"
                params.append(provider)
            if status_code:
                query += " AND status_code = ?"
                params.append(status_code)
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            query += " ORDER BY timestamp DESC"
            rows = cursor.execute(query, params).fetchall()
            return [LogEntry(**dict(r)) for r in rows]
        else:
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
        """Get summary statistics of all logged entries."""
        entries = self.query()
        if not entries:
            return {"total_calls": 0, "total_cost_usd": 0.0, "total_tokens_in": 0,
                    "total_tokens_out": 0, "calls_by_model": {}, "cost_by_model": {}, "avg_latency_ms": 0.0}
        summary = {"total_calls": len(entries), "total_cost_usd": sum(e.cost_usd for e in entries),
                   "total_tokens_in": sum(e.tokens_in for e in entries),
                   "total_tokens_out": sum(e.tokens_out for e in entries), "calls_by_model": {},
                   "cost_by_model": {}, "avg_latency_ms": sum(e.latency_ms for e in entries) / len(entries)}
        for entry in entries:
            summary["calls_by_model"][entry.model] = summary["calls_by_model"].get(entry.model, 0) + 1
            summary["cost_by_model"][entry.model] = summary["cost_by_model"].get(entry.model, 0.0) + entry.cost_usd
        return summary

    def export_jsonl(self, path: str) -> None:
        """Export all entries to JSONL file."""
        with open(path, "w") as f:
            for entry in self.query():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def export_csv(self, path: str) -> None:
        """Export all entries to CSV file."""
        entries = self.query()
        if not entries:
            return
        fieldnames = ["id", "url", "method", "provider", "model", "status_code",
                      "latency_ms", "tokens_in", "tokens_out", "cost_usd", "timestamp", "error"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                row = {k: getattr(entry, k) for k in fieldnames}
                writer.writerow(row)


_original_urlopen = urllib_request.urlopen
_active_logger: Optional[LLMLogger] = None


def _is_llm(url: str, request_body: Optional[str]) -> bool:
    """Check if URL is likely an LLM API endpoint."""
    url_lower = url.lower()
    llm_keywords = ["openai", "anthropic", "google", "gemini", "mistral", "cohere", "together", "huggingface", "llama"]
    if any(kw in url_lower for kw in llm_keywords):
        return True
    if request_body:
        try:
            data = json.loads(request_body)
            if isinstance(data, dict):
                if any(k in data for k in ["model", "engine", "modelId"]):
                    return True
        except Exception:
            pass
    return False


def _patched_urlopen(url, data=None, timeout=None, **kwargs):
    """Patched urlopen that logs LLM API calls."""
    start_time = time.time()
    request_body = None
    response_body = None
    status_code = 200
    if data is not None:
        if isinstance(data, bytes):
            request_body = data.decode("utf-8", errors="ignore")
        else:
            request_body = str(data)
    url_str = url if isinstance(url, str) else url.full_url
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
            saved_headers = response.headers
            response.close()
            response = addinfourl(BytesIO(response_data), saved_headers, url_str, status_code)
        if is_llm and _active_logger:
            latency_ms = (time.time() - start_time) * 1000
            entry = LogEntry(url=url_str, method="POST", request_body=request_body, response_body=response_body, status_code=status_code, latency_ms=latency_ms)
            _active_logger.record(entry)
        return response
    except Exception as e:
        if is_llm and _active_logger:
            latency_ms = (time.time() - start_time) * 1000
            entry = LogEntry(url=url_str, method="POST", request_body=request_body, response_body=response_body, status_code=status_code, latency_ms=latency_ms, error=str(e))
            _active_logger.record(entry)
        raise


def patch_urllib(logger: Optional[LLMLogger] = None) -> None:
    """Patch urllib.request.urlopen to automatically log LLM API calls."""
    global _active_logger
    _active_logger = logger
    urllib_request.urlopen = _patched_urlopen


def unpatch_urllib() -> None:
    """Restore original urllib.request.urlopen."""
    global _active_logger
    urllib_request.urlopen = _original_urlopen
    _active_logger = None


@contextmanager
def session(log_file: Optional[str] = None, backend: str = "jsonl", auto_patch: bool = True):
    """Context manager for LLM API logging sessions."""
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


def main():
    """Command-line interface for LLM API Logger."""
    parser = argparse.ArgumentParser(description="LLM API Logger - Log and analyze LLM API calls")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    summary_parser = subparsers.add_parser("summary", help="Show summary statistics")
    summary_parser.add_argument("log_file", nargs="?", default="llm_api.jsonl", help="Log file path")
    query_parser = subparsers.add_parser("query", help="Query log entries")
    query_parser.add_argument("log_file", nargs="?", default="llm_api.jsonl", help="Log file path")
    query_parser.add_argument("--model", help="Filter by model")
    query_parser.add_argument("--provider", help="Filter by provider")
    export_parser = subparsers.add_parser("export", help="Export logs to file")
    export_parser.add_argument("log_file", nargs="?", default="llm_api.jsonl", help="Source log file")
    export_parser.add_argument("--output", "-o", required=True, help="Output file path")
    export_parser.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv", help="Output format")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    log_file = args.log_file
    if log_file.endswith(".jsonl"):
        backend = "jsonl"
    else:
        backend = "sqlite"
    logger = LLMLogger(db_path=log_file, backend=backend)
    if backend == "jsonl" and Path(log_file).exists():
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        entry = LogEntry.from_dict(data)
                        logger.entries.append(entry)
                    except:
                        pass
    if args.command == "summary":
        summary = logger.summary()
        print("\n" + "="*60)
        print("LLM API CALL SUMMARY")
        print("="*60)
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Cost (USD): ${summary['total_cost_usd']:.4f}")
        print(f"Total Input Tokens: {summary['total_tokens_in']:,}")
        print(f"Total Output Tokens: {summary['total_tokens_out']:,}")
        print(f"Average Latency (ms): {summary['avg_latency_ms']:.2f}")
        print("\nCalls by Model:")
        for model, count in sorted(summary["calls_by_model"].items()):
            cost = summary["cost_by_model"].get(model, 0.0)
            print(f"  {model:<30} {count:>5} calls  ${cost:>8.4f}")
        print("="*60 + "\n")
    elif args.command == "query":
        results = logger.query(model=args.model, provider=args.provider)
        print(f"\nFound {len(results)} entries\n")
        for entry in results[:10]:
            print(f"  {entry.timestamp} | {entry.provider:>10} | {entry.model:<20} | ${entry.cost_usd:.6f}")
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")
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


if __name__ == "__main__":
    main()
