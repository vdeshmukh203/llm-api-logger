"""Append-only JSONL log store with SHA-256 content-hash provenance."""

import hashlib
import json
from pathlib import Path
from typing import List


class LogStore:
    """Append-only JSONL store; each record is stamped with a SHA-256 hash."""

    def __init__(self, path: str = "llm_api.jsonl"):
        self.path = path
        self._records: List[dict] = []

    def append(self, record: dict) -> str:
        """Persist *record* and return its SHA-256 hex digest."""
        payload = json.dumps(record, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        stamped = {**record, "_sha256": digest}
        self._records.append(stamped)
        if self.path != ":memory:":
            with open(self.path, "a") as fh:
                fh.write(json.dumps(stamped) + "\n")
        return digest

    def load(self) -> List[dict]:
        """Return all records (in-memory cache first, then file)."""
        if self.path == ":memory:":
            return list(self._records)
        records: List[dict] = []
        p = Path(self.path)
        if not p.exists():
            return records
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def verify(self, record: dict) -> bool:
        """Return True if the record's ``_sha256`` field matches its content."""
        stored = record.get("_sha256")
        if not stored:
            return False
        payload = json.dumps({k: v for k, v in record.items() if k != "_sha256"}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest() == stored

    def __len__(self) -> int:
        return len(self.load())
