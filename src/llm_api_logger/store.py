"""
Persistent log store with SHA-256 content hashing.

Each record written by :class:`LogStore` includes a ``content_hash`` field —
the SHA-256 digest of the canonical JSON serialisation of the URL and body
fields.  This allows downstream verification that stored records have not been
modified after capture.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    """Return the hex-encoded SHA-256 digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _provenance_payload(url: str, request_body: Optional[str], response_body: Optional[str]) -> str:
    """Canonical JSON used as input to the SHA-256 hash."""
    return json.dumps(
        {"url": url, "request_body": request_body, "response_body": response_body},
        sort_keys=True,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Record:
    """An immutable, tamper-evident log record for one LLM API call.

    Attributes
    ----------
    content_hash:
        SHA-256 digest of ``{url, request_body, response_body}`` at capture
        time.  Recompute and compare to detect post-hoc modifications.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
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
    content_hash: str = ""
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.content_hash:
            payload = _provenance_payload(self.url, self.request_body, self.response_body)
            self.content_hash = _sha256(payload)

    def verify(self) -> bool:
        """Return True if the stored hash matches a freshly computed digest."""
        payload = _provenance_payload(self.url, self.request_body, self.response_body)
        return self.content_hash == _sha256(payload)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        """Deserialise from a plain dictionary (ignores unknown keys)."""
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS records (
        id           TEXT PRIMARY KEY,
        timestamp    TEXT NOT NULL,
        url          TEXT NOT NULL,
        method       TEXT,
        provider     TEXT,
        model        TEXT,
        request_body  TEXT,
        response_body TEXT,
        status_code  INTEGER,
        latency_ms   REAL,
        tokens_in    INTEGER,
        tokens_out   INTEGER,
        cost_usd     REAL,
        content_hash TEXT,
        error        TEXT
    )
"""

_INSERT_SQL = """
    INSERT OR REPLACE INTO records
    VALUES (:id,:timestamp,:url,:method,:provider,:model,:request_body,
            :response_body,:status_code,:latency_ms,:tokens_in,:tokens_out,
            :cost_usd,:content_hash,:error)
"""


class LogStore:
    """Persistent store for LLM API :class:`Record` objects.

    Supports JSONL (append-only text) and SQLite backends.  Both backends
    preserve the SHA-256 ``content_hash`` field so callers can verify
    provenance via :meth:`Record.verify` or :meth:`LogStore.verify_all`.

    Parameters
    ----------
    path:
        File path for the database or JSONL file.
        Use ``":memory:"`` for a transient SQLite database.
    backend:
        ``"jsonl"`` or ``"sqlite"``.

    Examples
    --------
    >>> store = LogStore("calls.jsonl", backend="jsonl")
    >>> rec = Record(url="https://api.openai.com/v1/chat/completions")
    >>> store.append(rec)
    >>> all_ok = all(store.verify_all().values())
    """

    def __init__(self, path: str = "llm_api.jsonl", backend: str = "jsonl") -> None:
        if backend not in ("jsonl", "sqlite"):
            raise ValueError(f"Unknown backend {backend!r}. Use 'jsonl' or 'sqlite'.")
        self.path = path
        self.backend = backend
        self._conn: Optional[sqlite3.Connection] = None
        if backend == "sqlite":
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute(_SQLITE_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, record: Record) -> str:
        """Persist *record* and return its ``content_hash``.

        Parameters
        ----------
        record:
            The :class:`Record` to store.

        Returns
        -------
        str
            The SHA-256 content hash of the stored record.
        """
        if self.backend == "jsonl":
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        else:
            self._conn.execute(_INSERT_SQL, record.to_dict())
            self._conn.commit()
        return record.content_hash

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, limit: Optional[int] = None) -> List[Record]:
        """Return stored records, most-recent first.

        Parameters
        ----------
        limit:
            Maximum number of records to return.  ``None`` returns all.
        """
        if self.backend == "jsonl":
            records: List[Record] = []
            p = Path(self.path)
            if not p.exists():
                return records
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(Record.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
            records.sort(key=lambda r: r.timestamp, reverse=True)
            return records[:limit] if limit else records
        else:
            sql = "SELECT * FROM records ORDER BY timestamp DESC"
            if limit:
                sql += f" LIMIT {int(limit)}"
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(sql).fetchall()
            return [Record.from_dict(dict(r)) for r in rows]

    def count(self) -> int:
        """Return the total number of stored records."""
        if self.backend == "jsonl":
            p = Path(self.path)
            if not p.exists():
                return 0
            return sum(1 for ln in open(self.path, encoding="utf-8") if ln.strip())
        row = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def verify_all(self) -> Dict[str, bool]:
        """Verify the SHA-256 hash of every stored record.

        Returns
        -------
        dict
            Mapping ``{record_id: is_valid}`` for every record.
        """
        return {r.id: r.verify() for r in self.load()}

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"LogStore(path={self.path!r}, backend={self.backend!r})"

    def __len__(self) -> int:
        return self.count()
