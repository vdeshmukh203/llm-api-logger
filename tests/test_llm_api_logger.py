"""Functional test suite for llm_api_logger."""

import json
import sys
import tempfile
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import llm_api_logger as lal
from llm_api_logger import (
    LogEntry,
    LLMLogger,
    COST_TABLE,
    estimate_cost,
    patch_urllib,
    unpatch_urllib,
    session,
    _extract_provider,
    _extract_model,
    _tok,
    _is_llm,
)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases (regression)
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


def test_cli_alias():
    assert callable(lal._cli)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_known_model():
    cost = estimate_cost("gpt-4o", tokens_in=1_000_000, tokens_out=1_000_000)
    assert abs(cost - 20.0) < 1e-6


def test_estimate_cost_zero_tokens():
    assert estimate_cost("gpt-4o", 0, 0) == 0.0


def test_estimate_cost_unknown_raises():
    with pytest.raises(ValueError, match="not found in cost table"):
        estimate_cost("nonexistent-model-xyz", 100, 100)


def test_cost_table_structure():
    for model, prices in COST_TABLE.items():
        assert "input" in prices, f"Missing 'input' for {model}"
        assert "output" in prices, f"Missing 'output' for {model}"
        assert prices["input"] >= 0 and prices["output"] >= 0


# ---------------------------------------------------------------------------
# _extract_provider
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://api.openai.com/v1/chat/completions", "openai"),
    ("https://api.anthropic.com/v1/messages", "anthropic"),
    ("https://generativelanguage.googleapis.com/v1/models", "google"),
    ("https://api.mistral.ai/v1/chat/completions", "mistral"),
    ("https://api.cohere.ai/v1/generate", "cohere"),
    ("https://unknown.example.com/api", "unknown"),
])
def test_extract_provider(url, expected):
    assert _extract_provider(url) == expected


# ---------------------------------------------------------------------------
# _extract_model
# ---------------------------------------------------------------------------

def test_extract_model_from_request():
    req = json.dumps({"model": "gpt-4o", "messages": []})
    assert _extract_model(req, None) == "gpt-4o"


def test_extract_model_from_response():
    res = json.dumps({"model": "claude-3-opus", "choices": []})
    assert _extract_model(None, res) == "claude-3-opus"


def test_extract_model_prefers_request():
    req = json.dumps({"model": "gpt-4"})
    res = json.dumps({"model": "gpt-3.5-turbo"})
    assert _extract_model(req, res) == "gpt-4"


def test_extract_model_unknown():
    assert _extract_model(None, None) == "unknown"
    assert _extract_model("{bad json", None) == "unknown"


def test_extract_model_engine_key():
    req = json.dumps({"engine": "davinci", "prompt": "hello"})
    assert _extract_model(req, None) == "davinci"


# ---------------------------------------------------------------------------
# _tok
# ---------------------------------------------------------------------------

def test_tok_openai_format():
    res = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    assert _tok(res) == (10, 20)


def test_tok_google_format():
    res = json.dumps({"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 15}})
    assert _tok(res) == (5, 15)


def test_tok_empty():
    assert _tok(None) == (0, 0)
    assert _tok("") == (0, 0)
    assert _tok("not json") == (0, 0)


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

def test_log_entry_defaults():
    e = LogEntry(url="https://api.openai.com/v1/chat/completions")
    assert e.provider == "openai"
    assert e.id  # non-empty UUID
    assert e.method == "POST"
    assert e.status_code == 200


def test_log_entry_auto_model():
    req = json.dumps({"model": "gpt-4o-mini", "messages": []})
    e = LogEntry(url="https://api.openai.com/v1/chat/completions", request_body=req)
    assert e.model == "gpt-4o-mini"


def test_log_entry_auto_cost():
    res = json.dumps({
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    })
    req = json.dumps({"model": "gpt-4o-mini", "messages": []})
    e = LogEntry(
        url="https://api.openai.com/v1/chat/completions",
        request_body=req, response_body=res,
    )
    assert e.tokens_in == 1_000_000
    assert e.tokens_out == 1_000_000
    expected = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(e.cost_usd - expected) < 1e-9


def test_log_entry_from_dict_roundtrip():
    e = LogEntry(url="https://api.openai.com/v1/chat/completions", model="gpt-4o", latency_ms=123.4)
    d = e.to_dict()
    e2 = LogEntry.from_dict(d)
    assert e.id == e2.id
    assert e.model == e2.model
    assert e.latency_ms == e2.latency_ms


# ---------------------------------------------------------------------------
# LLMLogger — SQLite backend
# ---------------------------------------------------------------------------

def test_llm_logger_sqlite_record_and_count():
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        assert log.count() == 0
        log.record(LogEntry(url="https://api.openai.com/v1/chat/completions", model="gpt-4o"))
        log.record(LogEntry(url="https://api.anthropic.com/v1/messages", model="claude-3-opus"))
        assert log.count() == 2


def test_llm_logger_sqlite_query_filter_model():
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        log.record(LogEntry(url="https://api.openai.com/v1/chat/completions", model="gpt-4o"))
        log.record(LogEntry(url="https://api.openai.com/v1/chat/completions", model="gpt-4o"))
        log.record(LogEntry(url="https://api.anthropic.com/v1/messages", model="claude-3-haiku"))
        results = log.query(model="gpt-4o")
        assert len(results) == 2
        assert all(e.model == "gpt-4o" for e in results)


def test_llm_logger_sqlite_query_filter_provider():
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        log.record(LogEntry(url="https://api.openai.com/v1/chat/completions", model="gpt-4o"))
        log.record(LogEntry(url="https://api.anthropic.com/v1/messages", model="claude-3-opus"))
        results = log.query(provider="anthropic")
        assert len(results) == 1
        assert results[0].provider == "anthropic"


def test_llm_logger_sqlite_query_filter_status():
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o", status_code=200))
        log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o", status_code=429))
        results = log.query(status_code=429)
        assert len(results) == 1
        assert results[0].status_code == 429


def test_llm_logger_sqlite_summary():
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o",
                            tokens_in=100, tokens_out=50, cost_usd=0.001, latency_ms=200))
        log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o",
                            tokens_in=200, tokens_out=100, cost_usd=0.002, latency_ms=400))
        s = log.summary()
        assert s["total_calls"] == 2
        assert s["total_tokens_in"] == 300
        assert s["total_tokens_out"] == 150
        assert abs(s["total_cost_usd"] - 0.003) < 1e-9
        assert abs(s["avg_latency_ms"] - 300.0) < 1e-9
        assert s["calls_by_model"]["gpt-4o"] == 2


def test_llm_logger_empty_summary():
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        s = log.summary()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# LLMLogger — JSONL backend
# ---------------------------------------------------------------------------

def test_llm_logger_jsonl_record_and_count():
    log = LLMLogger(backend="jsonl")
    log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o"))
    assert log.count() == 1


def test_llm_logger_jsonl_query_sorted():
    log = LLMLogger(backend="jsonl")
    log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o",
                        timestamp="2024-01-01T00:00:00"))
    log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o",
                        timestamp="2024-06-01T00:00:00"))
    results = log.query()
    assert results[0].timestamp > results[1].timestamp  # newest first


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_csv(tmp_path):
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o",
                            cost_usd=0.001, latency_ms=300))
        out = str(tmp_path / "test.csv")
        log.export_csv(out)
    import csv
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-4o"


def test_export_jsonl(tmp_path):
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        log.record(LogEntry(url="https://api.openai.com/v1/", model="claude-3-haiku"))
        out = str(tmp_path / "test.jsonl")
        log.export_jsonl(out)
    with open(out) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["model"] == "claude-3-haiku"


def test_export_csv_empty_noop(tmp_path):
    with LLMLogger(db_path=":memory:", backend="sqlite") as log:
        out = str(tmp_path / "empty.csv")
        log.export_csv(out)
    # File should not be created for empty log
    assert not pathlib.Path(out).exists()


# ---------------------------------------------------------------------------
# _is_llm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,body,expected", [
    ("https://api.openai.com/v1/chat/completions", None, True),
    ("https://api.anthropic.com/v1/messages", None, True),
    ("https://example.com/api", json.dumps({"model": "gpt-4o"}), True),
    ("https://example.com/plain", None, False),
    ("https://example.com/plain", json.dumps({"key": "value"}), False),
])
def test_is_llm(url, body, expected):
    assert _is_llm(url, body) == expected


# ---------------------------------------------------------------------------
# session context manager
# ---------------------------------------------------------------------------

def test_session_context_manager_jsonl(tmp_path):
    log_path = str(tmp_path / "session.jsonl")
    with session(log_file=log_path, backend="jsonl", auto_patch=False) as log:
        log.record(LogEntry(url="https://api.openai.com/v1/", model="gpt-4o"))
        log.record(LogEntry(url="https://api.anthropic.com/v1/", model="claude-3-haiku"))
    # File should have been written on exit
    assert pathlib.Path(log_path).exists()
    with open(log_path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# patch/unpatch urllib
# ---------------------------------------------------------------------------

def test_patch_unpatch_restores_original():
    import urllib.request as ur
    orig = ur.urlopen
    patch_urllib()
    assert ur.urlopen is not orig
    unpatch_urllib()
    assert ur.urlopen is orig


# ---------------------------------------------------------------------------
# LLMLogger invalid backend
# ---------------------------------------------------------------------------

def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        LLMLogger(backend="redis")
