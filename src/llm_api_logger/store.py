"""JSONL-backed log store with SHA-256 content provenance."""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LogRecord:
    """A single captured LLM API interaction with provenance hash."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    method: str = "POST"
    url: str = ""
    request_headers: Dict[str, str] = field(default_factory=dict)
    request_body: Optional[str] = None
    response_status: int = 0
    response_headers: Dict[str, str] = field(default_factory=dict)
    response_body: Optional[str] = None
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    # SHA-256 of (url + request_body + response_body) for provenance verification
    sha256: str = ""
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {"url": self.url, "request_body": self.request_body, "response_body": self.response_body},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """Return True if the stored SHA-256 matches the record content."""
        return self.sha256 == self._compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LogStore:
    """Append-only JSONL store for :class:`LogRecord` objects.

    Records are written to disk immediately on :meth:`append` and cached in
    memory so that :meth:`records` is fast.  On initialisation any existing
    file is read into the cache.

    Args:
        path: Path to the JSONL log file.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._records: List[LogRecord] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(LogRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

    def append(self, record: LogRecord) -> None:
        """Append *record* to the in-memory cache and write it to disk."""
        self._records.append(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")

    def records(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[LogRecord]:
        """Return stored records, optionally filtered by *provider* or *model*."""
        result = self._records[:]
        if provider:
            result = [r for r in result if r.provider == provider]
        if model:
            result = [r for r in result if r.model == model]
        return result

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics over all stored records."""
        recs = self._records
        if not recs:
            return {
                "count": 0,
                "total_cost_usd": 0.0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "avg_latency_ms": 0.0,
            }
        return {
            "count": len(recs),
            "total_cost_usd": sum(r.cost_usd for r in recs),
            "total_tokens_in": sum(r.tokens_in for r in recs),
            "total_tokens_out": sum(r.tokens_out for r in recs),
            "avg_latency_ms": sum(r.latency_ms for r in recs) / len(recs),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)
