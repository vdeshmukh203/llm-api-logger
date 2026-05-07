"""Tests for llm_api_logger — core logger, cost estimation, and JOSS API surface."""

import json
import tempfile
import os

import llm_api_logger as lal


# ---------------------------------------------------------------------------
# Backward-compat aliases (required by JOSS review)
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
# Provider / model extraction
# ---------------------------------------------------------------------------

def test_extract_provider_openai():
    assert lal._extract_provider("https://api.openai.com/v1/chat/completions") == "openai"


def test_extract_provider_anthropic():
    assert lal._extract_provider("https://api.anthropic.com/v1/messages") == "anthropic"


def test_extract_provider_google():
    assert lal._extract_provider("https://generativelanguage.googleapis.com") == "google"


def test_extract_provider_unknown():
    assert lal._extract_provider("https://example.com") == "unknown"


def test_extract_model_from_request():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    assert lal._extract_model(body, None) == "gpt-4o"


def test_extract_model_from_response():
    resp = json.dumps({"model": "claude-3-haiku", "content": []})
    assert lal._extract_model(None, resp) == "claude-3-haiku"


def test_extract_model_prefers_request():
    req  = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({"model": "gpt-4o-mini"})
    assert lal._extract_model(req, resp) == "gpt-4o"


def test_extract_model_unknown():
    assert lal._extract_model(None, None) == "unknown"


def test_extract_model_invalid_json():
    assert lal._extract_model("not-json", "also-not-json") == "unknown"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def test_estimate_cost_gpt4o():
    cost = lal.estimate_cost("gpt-4o", tokens_in=1_000_000, tokens_out=1_000_000)
    assert abs(cost - 20.0) < 1e-9


def test_estimate_cost_zero_tokens():
    cost = lal.estimate_cost("gpt-4o-mini", tokens_in=0, tokens_out=0)
    assert cost == 0.0


def test_estimate_cost_unknown_model():
    import pytest
    with pytest.raises(ValueError, match="not found"):
        lal.estimate_cost("nonexistent-model", 100, 100)


def test_cost_table_structure():
    for model, pricing in lal.COST_TABLE.items():
        assert "input"  in pricing, f"{model} missing 'input'"
        assert "output" in pricing, f"{model} missing 'output'"
        assert pricing["input"]  >= 0
        assert pricing["output"] >= 0


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

def test_log_entry_defaults():
    e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions")
    assert e.provider == "openai"
    assert e.model == "unknown"
    assert e.id != ""
    assert e.timestamp != ""


def test_log_entry_model_extraction():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    e = lal.LogEntry(url="https://api.openai.com/", request_body=body)
    assert e.model == "gpt-4o"


def test_log_entry_token_extraction():
    resp = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
    e = lal.LogEntry(url="https://api.openai.com/", response_body=resp)
    assert e.tokens_in  == 100
    assert e.tokens_out == 50


def test_log_entry_cost_auto():
    resp = json.dumps({
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    })
    e = lal.LogEntry(url="https://api.openai.com/", response_body=resp)
    assert e.cost_usd > 0


def test_log_entry_roundtrip():
    e = lal.LogEntry(url="https://api.openai.com/", status_code=200, latency_ms=42.5)
    restored = lal.LogEntry.from_dict(e.to_dict())
    assert restored.id          == e.id
    assert restored.url         == e.url
    assert restored.latency_ms  == e.latency_ms
    assert restored.status_code == e.status_code


def test_log_entry_from_dict_ignores_extra_keys():
    d = lal.LogEntry(url="https://api.openai.com/").to_dict()
    d["unexpected_future_field"] = "ignored"
    # Should not raise
    e = lal.LogEntry.from_dict(d)
    assert e.url == "https://api.openai.com/"


# ---------------------------------------------------------------------------
# LLMLogger — SQLite backend
# ---------------------------------------------------------------------------

def test_llm_logger_sqlite_record_and_count():
    log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    assert log.count() == 0
    log.record(lal.LogEntry(url="https://api.openai.com/"))
    assert log.count() == 1


def test_llm_logger_sqlite_query_all():
    log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
    log.record(lal.LogEntry(url="https://api.anthropic.com/", model="claude-3-haiku"))
    results = log.query()
    assert len(results) == 2


def test_llm_logger_sqlite_filter_model():
    log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
    log.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o-mini"))
    results = log.query(model="gpt-4o")
    assert len(results) == 1
    assert results[0].model == "gpt-4o"


def test_llm_logger_sqlite_filter_provider():
    log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
    log.record(lal.LogEntry(url="https://api.anthropic.com/", model="claude-3-haiku"))
    results = log.query(provider="anthropic")
    assert len(results) == 1
    assert results[0].provider == "anthropic"


def test_llm_logger_sqlite_filter_status():
    log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    e1 = lal.LogEntry(url="https://api.openai.com/", model="gpt-4o")
    e1.status_code = 200
    e2 = lal.LogEntry(url="https://api.openai.com/", model="gpt-4o")
    e2.status_code = 429
    log.record(e1)
    log.record(e2)
    results = log.query(status_code=429)
    assert len(results) == 1
    assert results[0].status_code == 429


def test_llm_logger_sqlite_summary():
    log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(lal.LogEntry(
        url="https://api.openai.com/", model="gpt-4o",
        tokens_in=100, tokens_out=50, latency_ms=200,
    ))
    stats = log.summary()
    assert stats["total_calls"] == 1
    assert stats["total_tokens_in"]  == 100
    assert stats["total_tokens_out"] == 50
    assert stats["avg_latency_ms"]   == 200.0
    assert "gpt-4o" in stats["calls_by_model"]


def test_llm_logger_empty_summary():
    log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    stats = log.summary()
    assert stats["total_calls"]    == 0
    assert stats["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# LLMLogger — JSONL backend
# ---------------------------------------------------------------------------

def test_llm_logger_jsonl_record_and_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "calls.jsonl")
        log = lal.LLMLogger(db_path=path, backend="jsonl")
        log.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
        assert log.count() == 1
        results = log.query()
        assert len(results) == 1
        assert results[0].model == "gpt-4o"


def test_llm_logger_jsonl_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        src  = os.path.join(tmpdir, "src.jsonl")
        dest = os.path.join(tmpdir, "dest.jsonl")
        log = lal.LLMLogger(db_path=src, backend="jsonl")
        log.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
        log.export_jsonl(dest)
        with open(dest, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["model"] == "gpt-4o"


def test_llm_logger_export_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out.csv")
        log  = lal.LLMLogger(db_path=":memory:", backend="sqlite")
        log.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
        log.export_csv(path)
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["model"] == "gpt-4o"


def test_llm_logger_unknown_backend():
    import pytest
    with pytest.raises(ValueError, match="Unknown backend"):
        lal.LLMLogger(backend="badbackend")


# ---------------------------------------------------------------------------
# Patch / unpatch urllib
# ---------------------------------------------------------------------------

def test_patch_unpatch_urllib():
    import urllib.request as ur
    original = ur.urlopen
    log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    lal.patch_urllib(log)
    assert ur.urlopen is not original
    lal.unpatch_urllib()
    assert ur.urlopen is original


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

def test_session_context_manager():
    with lal.session(backend="sqlite") as log:
        assert isinstance(log, lal.LLMLogger)


def test_session_restores_urlopen():
    import urllib.request as ur
    original = ur.urlopen
    with lal.session(backend="sqlite"):
        pass
    assert ur.urlopen is original
