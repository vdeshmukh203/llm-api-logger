"""Tests for llm_api_logger — JOSS-level coverage."""

import json
import sys
import tempfile
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import llm_api_logger as lal


# ---------------------------------------------------------------------------
# Backwards-compatible alias smoke tests (keep originals passing)
# ---------------------------------------------------------------------------

def test_import():
    assert hasattr(lal, "LogRecord")


def test_backends():
    assert hasattr(lal, "JSONLBackend")
    assert hasattr(lal, "SQLiteBackend")
    assert hasattr(lal, "StdoutBackend")


def test_detect_provider():
    assert callable(lal._detect_provider)


def test_extract_model():
    assert callable(lal._extract_model)


# ---------------------------------------------------------------------------
# _extract_provider
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://api.openai.com/v1/chat/completions", "openai"),
    ("https://api.anthropic.com/v1/messages", "anthropic"),
    ("https://generativelanguage.googleapis.com/v1beta/models", "google"),
    ("https://api.mistral.ai/v1/chat/completions", "mistral"),
    ("https://api.cohere.ai/v1/generate", "cohere"),
    ("https://api.together.xyz/v1/chat", "together"),
    ("https://huggingface.co/api/inference", "huggingface"),
    ("https://example.com/api/chat", "unknown"),
])
def test_extract_provider(url, expected):
    assert lal._extract_provider(url) == expected


# ---------------------------------------------------------------------------
# _extract_model
# ---------------------------------------------------------------------------

def test_extract_model_from_request():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    assert lal._extract_model(body, None) == "gpt-4o"


def test_extract_model_from_response():
    resp = json.dumps({"model": "claude-3-5-sonnet", "content": []})
    assert lal._extract_model(None, resp) == "claude-3-5-sonnet"


def test_extract_model_prefers_request():
    req = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({"model": "gpt-4-turbo"})
    assert lal._extract_model(req, resp) == "gpt-4o"


def test_extract_model_unknown():
    assert lal._extract_model(None, None) == "unknown"
    assert lal._extract_model("{not json}", None) == "unknown"


# ---------------------------------------------------------------------------
# _tok — token extraction
# ---------------------------------------------------------------------------

def test_tok_openai_format():
    body = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})
    ti, to = lal._tok(body)
    assert ti == 100
    assert to == 50


def test_tok_anthropic_format():
    body = json.dumps({"usage": {"input_tokens": 200, "output_tokens": 80}})
    ti, to = lal._tok(body)
    assert ti == 200
    assert to == 80


def test_tok_google_format():
    body = json.dumps({"usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 90}})
    ti, to = lal._tok(body)
    assert ti == 300
    assert to == 90


def test_tok_empty():
    assert lal._tok(None) == (0, 0)
    assert lal._tok("") == (0, 0)
    assert lal._tok("not json") == (0, 0)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_known_model():
    cost = lal.estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert abs(cost - (2.50 + 10.00)) < 1e-9


def test_estimate_cost_zero_tokens():
    assert lal.estimate_cost("gpt-4o-mini", 0, 0) == 0.0


def test_estimate_cost_unknown_model():
    with pytest.raises(ValueError, match="not found"):
        lal.estimate_cost("nonexistent-model-xyz", 100, 100)


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

def test_log_entry_defaults():
    e = lal.LogEntry()
    assert e.id
    assert e.timestamp.endswith("Z")
    assert e.provider == "unknown"
    assert e.model == "unknown"


def test_log_entry_provider_derivation():
    e = lal.LogEntry(url="https://api.openai.com/v1/chat")
    assert e.provider == "openai"


def test_log_entry_model_derivation():
    req = json.dumps({"model": "gpt-4o-mini"})
    e = lal.LogEntry(url="https://api.openai.com/v1/chat", request_body=req)
    assert e.model == "gpt-4o-mini"


def test_log_entry_token_derivation_openai():
    resp = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    e = lal.LogEntry(response_body=resp)
    assert e.tokens_in == 10
    assert e.tokens_out == 20


def test_log_entry_token_derivation_anthropic():
    resp = json.dumps({"usage": {"input_tokens": 15, "output_tokens": 30}})
    e = lal.LogEntry(response_body=resp)
    assert e.tokens_in == 15
    assert e.tokens_out == 30


def test_log_entry_cost_derivation():
    req = json.dumps({"model": "gpt-4o-mini"})
    resp = json.dumps({"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}})
    e = lal.LogEntry(
        url="https://api.openai.com/v1/chat",
        request_body=req,
        response_body=resp,
    )
    expected = lal.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(e.cost_usd - expected) < 1e-9


def test_log_entry_round_trip():
    e = lal.LogEntry(url="https://api.openai.com/v1/chat", model="gpt-4o", tokens_in=5, tokens_out=10)
    d = e.to_dict()
    restored = lal.LogEntry.from_dict(d)
    assert restored.id == e.id
    assert restored.model == e.model
    assert restored.tokens_in == e.tokens_in


# ---------------------------------------------------------------------------
# LLMLogger — SQLite backend
# ---------------------------------------------------------------------------

def test_sqlite_record_and_count():
    logger = lal.LLMLogger(backend="sqlite")
    assert logger.count() == 0
    logger.record(lal.LogEntry(url="https://api.openai.com", model="gpt-4o"))
    assert logger.count() == 1


def test_sqlite_query_filter_model():
    logger = lal.LLMLogger(backend="sqlite")
    logger.record(lal.LogEntry(url="https://api.openai.com", model="gpt-4o"))
    logger.record(lal.LogEntry(url="https://api.openai.com", model="gpt-3.5-turbo"))
    results = logger.query(model="gpt-4o")
    assert len(results) == 1
    assert results[0].model == "gpt-4o"


def test_sqlite_query_filter_provider():
    logger = lal.LLMLogger(backend="sqlite")
    logger.record(lal.LogEntry(url="https://api.openai.com", provider="openai"))
    logger.record(lal.LogEntry(url="https://api.anthropic.com", provider="anthropic"))
    assert len(logger.query(provider="openai")) == 1
    assert len(logger.query(provider="anthropic")) == 1


def test_sqlite_query_filter_status():
    logger = lal.LLMLogger(backend="sqlite")
    logger.record(lal.LogEntry(url="https://api.openai.com", status_code=200))
    logger.record(lal.LogEntry(url="https://api.openai.com", status_code=429))
    assert len(logger.query(status_code=429)) == 1


def test_sqlite_summary():
    logger = lal.LLMLogger(backend="sqlite")
    logger.record(lal.LogEntry(model="gpt-4o", tokens_in=100, tokens_out=50, cost_usd=0.001, latency_ms=300))
    logger.record(lal.LogEntry(model="gpt-4o", tokens_in=200, tokens_out=100, cost_usd=0.002, latency_ms=700))
    s = logger.summary()
    assert s["total_calls"] == 2
    assert s["total_tokens_in"] == 300
    assert s["total_tokens_out"] == 150
    assert abs(s["total_cost_usd"] - 0.003) < 1e-9
    assert abs(s["avg_latency_ms"] - 500) < 1e-9
    assert s["calls_by_model"]["gpt-4o"] == 2


def test_sqlite_empty_summary():
    logger = lal.LLMLogger(backend="sqlite")
    s = logger.summary()
    assert s["total_calls"] == 0


# ---------------------------------------------------------------------------
# LLMLogger — JSONL backend
# ---------------------------------------------------------------------------

def test_jsonl_record_and_count():
    logger = lal.LLMLogger(backend="jsonl")
    logger.record(lal.LogEntry(model="claude-3-5-sonnet"))
    assert logger.count() == 1


def test_jsonl_query_filter():
    logger = lal.LLMLogger(backend="jsonl")
    logger.record(lal.LogEntry(model="claude-3-5-sonnet", provider="anthropic"))
    logger.record(lal.LogEntry(model="gpt-4o", provider="openai"))
    assert len(logger.query(provider="anthropic")) == 1


def test_jsonl_export_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(pathlib.Path(tmpdir) / "test.jsonl")
        logger = lal.LLMLogger(backend="jsonl")
        logger.record(lal.LogEntry(model="gpt-4o", tokens_in=5, tokens_out=10))
        logger.record(lal.LogEntry(model="claude-3-5-sonnet", tokens_in=3, tokens_out=6))
        logger.export_jsonl(path)

        # reload
        logger2 = lal.LLMLogger(backend="jsonl")
        with open(path) as f:
            for line in f:
                logger2.entries.append(lal.LogEntry.from_dict(json.loads(line)))
        assert logger2.count() == 2


def test_jsonl_export_append():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(pathlib.Path(tmpdir) / "append.jsonl")
        logger1 = lal.LLMLogger(backend="jsonl")
        logger1.record(lal.LogEntry(model="gpt-4o"))
        logger1.export_jsonl(path)

        logger2 = lal.LLMLogger(backend="jsonl")
        logger2.record(lal.LogEntry(model="gpt-4o-mini"))
        logger2.export_jsonl(path, append=True)

        lines = pathlib.Path(path).read_text().strip().splitlines()
        assert len(lines) == 2


def test_export_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(pathlib.Path(tmpdir) / "out.csv")
        logger = lal.LLMLogger(backend="jsonl")
        logger.record(lal.LogEntry(model="gpt-4o", tokens_in=5, tokens_out=10))
        logger.export_csv(path)
        content = pathlib.Path(path).read_text()
        assert "gpt-4o" in content
        assert "model" in content  # header present


# ---------------------------------------------------------------------------
# LLMLogger — invalid backend
# ---------------------------------------------------------------------------

def test_invalid_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        lal.LLMLogger(backend="redis")


# ---------------------------------------------------------------------------
# session context manager
# ---------------------------------------------------------------------------

def test_session_context_manager_memory():
    with lal.session(backend="sqlite", auto_patch=False) as log:
        log.record(lal.LogEntry(model="gpt-4o"))
    assert log.count() == 1


def test_session_jsonl_appends(tmp_path):
    log_file = str(tmp_path / "session.jsonl")
    with lal.session(log_file, backend="jsonl", auto_patch=False) as log:
        log.record(lal.LogEntry(model="gpt-4o"))
    # second session should append
    with lal.session(log_file, backend="jsonl", auto_patch=False) as log2:
        log2.record(lal.LogEntry(model="gpt-4o-mini"))
    lines = pathlib.Path(log_file).read_text().strip().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# store.LogStore
# ---------------------------------------------------------------------------

def test_store_append_and_persist(tmp_path):
    from src.llm_api_logger.store import LogRecord, LogStore
    path = str(tmp_path / "store.jsonl")
    store = LogStore(path)
    r = LogRecord(url="https://api.openai.com", model="gpt-4o", tokens_in=10, tokens_out=5)
    store.append(r)
    assert len(store) == 1

    store2 = LogStore(path)
    assert len(store2) == 1
    assert store2.records()[0].model == "gpt-4o"


def test_store_sha256_provenance(tmp_path):
    from src.llm_api_logger.store import LogRecord, LogStore
    path = str(tmp_path / "prov.jsonl")
    store = LogStore(path)
    r = LogRecord(url="https://api.openai.com", request_body='{"model":"gpt-4o"}', response_body='{"id":"x"}')
    assert r.verify()
    store.append(r)

    # Reload and verify
    store2 = LogStore(path)
    assert store2.records()[0].verify()


def test_store_summary(tmp_path):
    from src.llm_api_logger.store import LogRecord, LogStore
    path = str(tmp_path / "summary.jsonl")
    store = LogStore(path)
    store.append(LogRecord(tokens_in=100, tokens_out=50, cost_usd=0.001, latency_ms=200))
    store.append(LogRecord(tokens_in=200, tokens_out=100, cost_usd=0.002, latency_ms=400))
    s = store.summary()
    assert s["count"] == 2
    assert s["total_tokens_in"] == 300
    assert abs(s["avg_latency_ms"] - 300.0) < 1e-9


def test_store_filter(tmp_path):
    from src.llm_api_logger.store import LogRecord, LogStore
    path = str(tmp_path / "filter.jsonl")
    store = LogStore(path)
    store.append(LogRecord(provider="openai", model="gpt-4o"))
    store.append(LogRecord(provider="anthropic", model="claude-3-5-sonnet"))
    assert len(store.records(provider="openai")) == 1
    assert len(store.records(model="claude-3-5-sonnet")) == 1


# ---------------------------------------------------------------------------
# _is_llm
# ---------------------------------------------------------------------------

def test_is_llm_by_url():
    assert lal._is_llm("https://api.openai.com/v1/chat", None)
    assert lal._is_llm("https://api.anthropic.com/v1/messages", None)
    assert not lal._is_llm("https://example.com/api", None)


def test_is_llm_by_body():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    assert lal._is_llm("https://example.com", body)


# ---------------------------------------------------------------------------
# COST_TABLE completeness
# ---------------------------------------------------------------------------

def test_cost_table_has_current_models():
    for model in ("gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash", "mistral-large"):
        assert model in lal.COST_TABLE, f"{model} missing from COST_TABLE"


def test_cost_table_positive_prices():
    for model, pricing in lal.COST_TABLE.items():
        assert pricing["input"] >= 0, f"{model} input price is negative"
        assert pricing["output"] >= 0, f"{model} output price is negative"
