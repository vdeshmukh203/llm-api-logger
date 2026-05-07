"""Tests for llm_api_logger.store — LogStore and SHA-256 provenance."""

import json
import os
import tempfile

from llm_api_logger.store import LogStore, Record, _sha256, _provenance_payload


# ---------------------------------------------------------------------------
# _sha256 / _provenance_payload helpers
# ---------------------------------------------------------------------------

def test_sha256_is_deterministic():
    assert _sha256("hello") == _sha256("hello")


def test_sha256_length():
    assert len(_sha256("data")) == 64


def test_provenance_payload_sorted_keys():
    p1 = _provenance_payload("https://a.com", "req", "resp")
    p2 = _provenance_payload("https://a.com", "req", "resp")
    assert p1 == p2


def test_provenance_payload_none_values():
    payload = _provenance_payload("https://a.com", None, None)
    data = json.loads(payload)
    assert data["request_body"]  is None
    assert data["response_body"] is None


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def test_record_auto_hash():
    r = Record(url="https://api.openai.com/", request_body='{"model":"gpt-4o"}')
    assert len(r.content_hash) == 64


def test_record_verify_passes():
    r = Record(url="https://api.openai.com/", request_body='{"model":"gpt-4o"}')
    assert r.verify() is True


def test_record_verify_fails_after_tampering():
    r = Record(url="https://api.openai.com/", request_body='{"model":"gpt-4o"}')
    r.url = "https://evil.example.com/"  # tamper without rehashing
    assert r.verify() is False


def test_record_roundtrip():
    r = Record(url="https://api.anthropic.com/", model="claude-3-haiku", tokens_in=100)
    restored = Record.from_dict(r.to_dict())
    assert restored.id           == r.id
    assert restored.content_hash == r.content_hash
    assert restored.tokens_in    == 100


def test_record_from_dict_ignores_extra():
    d = Record(url="https://api.openai.com/").to_dict()
    d["future_field"] = "value"
    r = Record.from_dict(d)
    assert r.url == "https://api.openai.com/"


# ---------------------------------------------------------------------------
# LogStore — JSONL backend
# ---------------------------------------------------------------------------

def test_store_jsonl_append_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calls.jsonl")
        store = LogStore(path, backend="jsonl")
        rec = Record(url="https://api.openai.com/", model="gpt-4o")
        h = store.append(rec)
        assert len(h) == 64
        records = store.load()
        assert len(records) == 1
        assert records[0].model == "gpt-4o"


def test_store_jsonl_count():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calls.jsonl")
        store = LogStore(path, backend="jsonl")
        assert store.count() == 0
        store.append(Record(url="https://api.openai.com/"))
        store.append(Record(url="https://api.openai.com/"))
        assert store.count() == 2


def test_store_jsonl_len():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calls.jsonl")
        store = LogStore(path, backend="jsonl")
        store.append(Record(url="https://api.openai.com/"))
        assert len(store) == 1


def test_store_jsonl_verify_all():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calls.jsonl")
        store = LogStore(path, backend="jsonl")
        store.append(Record(url="https://api.openai.com/", request_body='{"model":"gpt-4o"}'))
        results = store.verify_all()
        assert all(results.values())


def test_store_jsonl_load_nonexistent():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "missing.jsonl")
        store = LogStore(path, backend="jsonl")
        assert store.load() == []
        assert store.count() == 0


def test_store_jsonl_limit():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calls.jsonl")
        store = LogStore(path, backend="jsonl")
        for _ in range(5):
            store.append(Record(url="https://api.openai.com/"))
        assert len(store.load(limit=3)) == 3


# ---------------------------------------------------------------------------
# LogStore — SQLite backend
# ---------------------------------------------------------------------------

def test_store_sqlite_memory_append_and_load():
    store = LogStore(":memory:", backend="sqlite")
    rec = Record(url="https://api.anthropic.com/", model="claude-3-haiku")
    store.append(rec)
    records = store.load()
    assert len(records) == 1
    assert records[0].model == "claude-3-haiku"


def test_store_sqlite_count():
    store = LogStore(":memory:", backend="sqlite")
    assert store.count() == 0
    store.append(Record(url="https://api.openai.com/"))
    assert store.count() == 1


def test_store_sqlite_verify_all():
    store = LogStore(":memory:", backend="sqlite")
    store.append(Record(url="https://api.openai.com/", request_body='{"model":"gpt-4o"}'))
    results = store.verify_all()
    assert all(results.values())


def test_store_sqlite_limit():
    store = LogStore(":memory:", backend="sqlite")
    for _ in range(5):
        store.append(Record(url="https://api.openai.com/"))
    assert len(store.load(limit=2)) == 2


def test_store_unknown_backend():
    import pytest
    with pytest.raises(ValueError, match="Unknown backend"):
        LogStore(":memory:", backend="xml")


# ---------------------------------------------------------------------------
# Cross-backend hash consistency
# ---------------------------------------------------------------------------

def test_hash_identical_across_backends():
    rec = Record(
        url="https://api.openai.com/v1/chat/completions",
        request_body='{"model":"gpt-4o"}',
        response_body='{"id":"chatcmpl-123"}',
    )
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_store  = LogStore(os.path.join(tmp, "c.jsonl"), backend="jsonl")
        sqlite_store = LogStore(os.path.join(tmp, "c.db"),    backend="sqlite")
        h1 = jsonl_store.append(rec)
        h2 = sqlite_store.append(rec)
    assert h1 == h2
