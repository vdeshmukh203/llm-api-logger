"""Tests for src/llm_api_logger store and proxy modules."""

import hashlib
import json
import os
import pathlib
import tempfile
import time

import pytest

from llm_api_logger.store import LogRecord, LogStore
from llm_api_logger.proxy import LLMAPIProxy, _detect_provider, _parse_model, _parse_usage


# ---------------------------------------------------------------------------
# LogRecord
# ---------------------------------------------------------------------------

class TestLogRecord:
    def test_defaults(self):
        r = LogRecord()
        assert r.method == "POST"
        assert r.status_code == 200
        assert r.sha256 == ""

    def test_to_dict_roundtrip(self):
        r = LogRecord(url="https://api.openai.com/v1/chat/completions",
                      model="gpt-4o", tokens_in=10, tokens_out=5)
        d = r.to_dict()
        r2 = LogRecord.from_dict(d)
        assert r2.id == r.id
        assert r2.model == "gpt-4o"

    def test_from_dict_ignores_unknown_keys(self):
        r = LogRecord.from_dict({"url": "https://example.com", "future_field": "ignored"})
        assert r.url == "https://example.com"


# ---------------------------------------------------------------------------
# LogStore — in-memory
# ---------------------------------------------------------------------------

class TestLogStoreMemory:
    def setup_method(self):
        self.store = LogStore(":memory:")

    def test_append_and_len(self):
        self.store.append(LogRecord(url="https://api.openai.com"))
        assert len(self.store) == 1

    def test_sha256_computed_on_append(self):
        r = LogRecord(request_body="hello", response_body="world")
        self.store.append(r)
        expected = hashlib.sha256(b"helloworld").hexdigest()
        assert r.sha256 == expected

    def test_sha256_empty_bodies(self):
        r = LogRecord()
        self.store.append(r)
        expected = hashlib.sha256(b"").hexdigest()
        assert r.sha256 == expected

    def test_all_returns_copy(self):
        self.store.append(LogRecord())
        records = self.store.all()
        assert len(records) == 1
        records.clear()
        assert len(self.store) == 1

    def test_filter_by_provider(self):
        self.store.append(LogRecord(provider="openai"))
        self.store.append(LogRecord(provider="anthropic"))
        results = self.store.filter(provider="openai")
        assert len(results) == 1
        assert results[0].provider == "openai"

    def test_filter_by_model(self):
        self.store.append(LogRecord(model="gpt-4o"))
        self.store.append(LogRecord(model="claude-3-haiku"))
        assert len(self.store.filter(model="gpt-4o")) == 1

    def test_filter_by_status_code(self):
        self.store.append(LogRecord(status_code=200))
        self.store.append(LogRecord(status_code=429))
        assert len(self.store.filter(status_code=429)) == 1

    def test_filter_by_since(self):
        r_old = LogRecord(timestamp="2024-01-01T00:00:00")
        r_new = LogRecord(timestamp="2025-06-01T00:00:00")
        self.store.append(r_old)
        self.store.append(r_new)
        results = self.store.filter(since="2025-01-01T00:00:00")
        assert len(results) == 1
        assert results[0].timestamp == "2025-06-01T00:00:00"

    def test_summary_empty(self):
        s = self.store.summary()
        assert s["total_calls"] == 0
        assert s["avg_latency_ms"] == 0.0

    def test_summary_aggregates(self):
        self.store.append(LogRecord(model="gpt-4o", tokens_in=100, tokens_out=50,
                                     cost_usd=0.01, latency_ms=200.0))
        self.store.append(LogRecord(model="gpt-4o", tokens_in=200, tokens_out=100,
                                     cost_usd=0.02, latency_ms=400.0))
        s = self.store.summary()
        assert s["total_calls"] == 2
        assert s["total_tokens_in"] == 300
        assert s["total_tokens_out"] == 150
        assert s["total_cost_usd"] == pytest.approx(0.03)
        assert s["avg_latency_ms"] == pytest.approx(300.0)
        assert s["calls_by_model"]["gpt-4o"] == 2


# ---------------------------------------------------------------------------
# LogStore — file persistence
# ---------------------------------------------------------------------------

class TestLogStorePersistence:
    def test_writes_jsonl_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            store = LogStore(path)
            store.append(LogRecord(url="https://api.openai.com", model="gpt-4o"))
            lines = pathlib.Path(path).read_text().strip().splitlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["model"] == "gpt-4o"
        finally:
            os.unlink(path)

    def test_loads_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            r = LogRecord(url="https://api.openai.com", model="claude-3-haiku")
            f.write(json.dumps(r.to_dict()) + "\n")
            path = f.name
        try:
            store = LogStore(path)
            assert len(store) == 1
            assert store.all()[0].model == "claude-3-haiku"
        finally:
            os.unlink(path)

    def test_appends_across_sessions(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            LogStore(path).append(LogRecord(model="gpt-4o"))
            LogStore(path).append(LogRecord(model="gpt-4o-mini"))
            store = LogStore(path)
            assert len(store) == 2
        finally:
            os.unlink(path)

    def test_skips_malformed_lines(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write("not valid json\n")
            r = LogRecord(model="gpt-4o")
            f.write(json.dumps(r.to_dict()) + "\n")
            path = f.name
        try:
            store = LogStore(path)
            assert len(store) == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Proxy helper functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://api.openai.com/v1/chat/completions", "openai"),
    ("https://api.anthropic.com/v1/messages", "anthropic"),
    ("https://generativelanguage.googleapis.com/v1beta/models", "google"),
    ("https://api.mistral.ai/v1/chat/completions", "mistral"),
    ("https://api.cohere.ai/v1/generate", "cohere"),
    ("https://api.together.xyz/v1/chat/completions", "together"),
    ("https://api-inference.huggingface.co/models", "huggingface"),
    ("https://example.com/api", "unknown"),
])
def test_proxy_detect_provider(url, expected):
    assert _detect_provider(url) == expected


def test_parse_model_from_body():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    assert _parse_model(body) == "gpt-4o"


def test_parse_model_engine_key():
    body = json.dumps({"engine": "davinci"})
    assert _parse_model(body) == "davinci"


def test_parse_model_missing():
    assert _parse_model(None) == ""
    assert _parse_model("{}") == ""


def test_parse_usage_openai():
    body = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    assert _parse_usage(body) == (10, 20)


def test_parse_usage_google():
    body = json.dumps({"usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 25}})
    assert _parse_usage(body) == (15, 25)


def test_parse_usage_empty():
    assert _parse_usage(None) == (0, 0)
    assert _parse_usage("{}") == (0, 0)


# ---------------------------------------------------------------------------
# LLMAPIProxy — lifecycle
# ---------------------------------------------------------------------------

class TestLLMAPIProxy:
    def test_start_stop(self):
        store = LogStore(":memory:")
        proxy = LLMAPIProxy(port=0, store=store)
        # port=0 lets the OS choose a free port
        proxy._server = None  # ensure clean state
        # Use context manager
        with LLMAPIProxy(store=store) as p:
            assert p._server is not None
            assert p._thread is not None
            assert p._thread.is_alive()
        assert p._server is None

    def test_address_property(self):
        proxy = LLMAPIProxy(host="127.0.0.1", port=9999, store=LogStore(":memory:"))
        assert proxy.address == "http://127.0.0.1:9999"

    def test_env_property(self):
        proxy = LLMAPIProxy(host="127.0.0.1", port=9999, store=LogStore(":memory:"))
        env = proxy.env
        assert env["HTTP_PROXY"] == "http://127.0.0.1:9999"
        assert env["HTTPS_PROXY"] == "http://127.0.0.1:9999"

    def test_double_start_is_idempotent(self):
        store = LogStore(":memory:")
        with LLMAPIProxy(store=store) as p:
            first_server = p._server
            p.start()  # second start should be a no-op
            assert p._server is first_server

    def test_stop_when_not_started(self):
        proxy = LLMAPIProxy(store=LogStore(":memory:"))
        proxy.stop()  # should not raise


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------

def test_package_exports():
    from llm_api_logger import LLMAPIProxy, LogRecord, LogStore
    assert callable(LLMAPIProxy)
    assert callable(LogStore)
    assert callable(LogRecord)
