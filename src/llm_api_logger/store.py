"""Structured storage for LLM API proxy log records."""

import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid


@dataclass
class LogRecord:
    """A single captured LLM API interaction."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    url: str = ""
    method: str = "POST"
    provider: str = ""
    model: str = ""
    request_headers: Optional[str] = None
    request_body: Optional[str] = None
    response_headers: Optional[str] = None
    response_body: Optional[str] = None
    status_code: int = 200
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    sha256: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LogRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class LogStore:
    """Append-only JSONL store with SHA-256 content hashing for provenance."""

    def __init__(self, path: str = "llm_proxy.jsonl"):
        self.path = path
        self._records: List[LogRecord] = []
        if path != ":memory:":
            self._load_existing()

    # ------------------------------------------------------------------
    def _load_existing(self) -> None:
        p = pathlib.Path(self.path)
        if not p.exists():
            return
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        self._records.append(LogRecord.from_dict(json.loads(line)))
                    except Exception:
                        pass

    def _compute_sha256(self, record: LogRecord) -> str:
        content = (record.request_body or "") + (record.response_body or "")
        return hashlib.sha256(content.encode()).hexdigest()

    # ------------------------------------------------------------------
    def append(self, record: LogRecord) -> None:
        """Persist a record, computing its SHA-256 hash first."""
        record.sha256 = self._compute_sha256(record)
        self._records.append(record)
        if self.path != ":memory:":
            with open(self.path, "a") as fh:
                fh.write(json.dumps(record.to_dict()) + "\n")

    def all(self) -> List[LogRecord]:
        return list(self._records)

    def filter(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        since: Optional[str] = None,
    ) -> List[LogRecord]:
        records = self._records
        if provider:
            records = [r for r in records if r.provider == provider]
        if model:
            records = [r for r in records if r.model == model]
        if status_code is not None:
            records = [r for r in records if r.status_code == status_code]
        if since:
            records = [r for r in records if r.timestamp >= since]
        return sorted(records, key=lambda r: r.timestamp, reverse=True)

    def summary(self) -> dict:
        records = self._records
        if not records:
            return {
                "total_calls": 0,
                "total_cost_usd": 0.0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "avg_latency_ms": 0.0,
                "calls_by_model": {},
                "cost_by_model": {},
            }
        calls_by_model: dict = {}
        cost_by_model: dict = {}
        for r in records:
            calls_by_model[r.model] = calls_by_model.get(r.model, 0) + 1
            cost_by_model[r.model] = cost_by_model.get(r.model, 0.0) + r.cost_usd
        return {
            "total_calls": len(records),
            "total_cost_usd": sum(r.cost_usd for r in records),
            "total_tokens_in": sum(r.tokens_in for r in records),
            "total_tokens_out": sum(r.tokens_out for r in records),
            "avg_latency_ms": sum(r.latency_ms for r in records) / len(records),
            "calls_by_model": calls_by_model,
            "cost_by_model": cost_by_model,
        }

    def __len__(self) -> int:
        return len(self._records)
