"""
Comprehensive test suite for llm_api_logger.

Covers: cost estimation, provider/model extraction, token parsing,
LogEntry dataclass, LLMLogger (sqlite + jsonl backends), query/filter,
summary statistics, CSV/JSONL export, urllib patching, and the session
context manager.
"""

import json
import sys
import pathlib
import pytest
import tempfile
import os

# Allow importing the root-level module when running from the repo
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
from urllib import request as urllib_request

# ---------------------------------------------------------------------------
# Backwards-compatible alias smoke tests (required by existing CI)
# ---------------------------------------------------------------------------

def test_import():
    assert hasattr(lal, "LogRecord")


def test_backends():
    assert hasattr(lal, "JSONLBackend")
    assert hasattr(lal, "SQLiteBackend")
    assert hasattr(lal, "StdoutBackend")


def test_detect_provider():
    assert callable(lal._detect_provider)


def test_extract_model_callable():
    assert callable(lal._extract_model)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_known_model():
    cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == pytest.approx(5.00 + 15.00)


def test_estimate_cost_zero_tokens():
    assert estimate_cost("gpt-4o", 0, 0) == 0.0


def test_estimate_cost_unknown_model_raises():
    with pytest.raises(ValueError, match="not found"):
        estimate_cost("nonexistent-model-xyz", 100, 100)


def test_cost_table_has_expected_models():
    for model in ("gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro", "mistral-large"):
        assert model in COST_TABLE


# ---------------------------------------------------------------------------
# _extract_provider
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://api.openai.com/v1/chat/completions", "openai"),
    ("https://api.anthropic.com/v1/messages", "anthropic"),
    ("https://generativelanguage.googleapis.com/v1/models", "google"),
    ("https://api.mistral.ai/v1/chat/completions", "mistral"),
    ("https://api.together.xyz/v1/chat/completions", "together"),
    ("https://api.cohere.ai/v1/generate", "cohere"),
    ("https://api-inference.huggingface.co/models/gpt2", "huggingface"),
    ("https://example.com/api/v1/call", "unknown"),
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
    resp = json.dumps({"model": "claude-3-haiku", "content": []})
    assert _extract_model(None, resp) == "claude-3-haiku"


def test_extract_model_request_wins_over_response():
    req = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({"model": "gpt-3.5-turbo"})
    assert _extract_model(req, resp) == "gpt-4o"


def test_extract_model_unknown_when_absent():
    assert _extract_model(None, None) == "unknown"
    assert _extract_model("{}", "{}") == "unknown"


def test_extract_model_invalid_json():
    assert _extract_model("not-json", None) == "unknown"


# ---------------------------------------------------------------------------
# _tok  (token extraction)
# ---------------------------------------------------------------------------

def test_tok_openai_format():
    body = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    assert _tok(body) == (10, 20)


def test_tok_anthropic_format():
    body = json.dumps({"usage": {"input_tokens": 15, "output_tokens": 30}})
    assert _tok(body) == (15, 30)


def test_tok_google_format():
    body = json.dumps({"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 25}})
    assert _tok(body) == (5, 25)


def test_tok_empty_response():
    assert _tok(None) == (0, 0)
    assert _tok("") == (0, 0)


def test_tok_invalid_json():
    assert _tok("not-json") == (0, 0)


# ---------------------------------------------------------------------------
# _is_llm
# ---------------------------------------------------------------------------

def test_is_llm_known_provider_url():
    assert _is_llm("https://api.openai.com/v1/chat/completions", None) is True
    assert _is_llm("https://api.anthropic.com/v1/messages", None) is True


def test_is_llm_model_in_body():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    assert _is_llm("https://example.com/api", body) is True


def test_is_llm_non_llm_url():
    assert _is_llm("https://example.com/api/v1/users", None) is False


# ---------------------------------------------------------------------------
# LogEntry dataclass
# ---------------------------------------------------------------------------

def test_log_entry_defaults():
    entry = LogEntry()
    assert entry.id  # non-empty UUID
    assert entry.provider == "unknown"
    assert entry.model == "unknown"
    assert entry.status_code == 200


def test_log_entry_provider_auto_detected():
    entry = LogEntry(url="https://api.openai.com/v1/chat/completions")
    assert entry.provider == "openai"


def test_log_entry_model_auto_detected():
    req = json.dumps({"model": "gpt-4o-mini"})
    entry = LogEntry(url="https://api.openai.com/v1/", request_body=req)
    assert entry.model == "gpt-4o-mini"


def test_log_entry_tokens_auto_extracted():
    resp = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
    entry = LogEntry(url="https://api.openai.com/v1/", response_body=resp)
    assert entry.tokens_in == 100
    assert entry.tokens_out == 50


def test_log_entry_cost_auto_calculated():
    req = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({"model": "gpt-4o", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}})
    entry = LogEntry(url="https://api.openai.com/v1/", request_body=req, response_body=resp)
    assert entry.cost_usd == pytest.approx(20.0)


def test_log_entry_to_dict_roundtrip():
    entry = LogEntry(url="https://api.openai.com/v1/", latency_ms=42.5)
    d = entry.to_dict()
    e2 = LogEntry.from_dict(d)
    assert e2.id == entry.id
    assert e2.latency_ms == entry.latency_ms


def test_log_entry_from_dict_ignores_unknown_keys():
    d = {"url": "https://api.openai.com/v1/", "unknown_future_field": "value"}
    entry = LogEntry.from_dict(d)
    assert entry.url == "https://api.openai.com/v1/"


# ---------------------------------------------------------------------------
# LLMLogger – SQLite backend
# ---------------------------------------------------------------------------

def test_sqlite_backend_record_and_count():
    logger = LLMLogger(db_path=":memory:", backend="sqlite")
    assert logger.count() == 0
    logger.record(LogEntry(url="https://api.openai.com/v1/"))
    assert logger.count() == 1


def test_sqlite_backend_query_all():
    logger = LLMLogger(db_path=":memory:", backend="sqlite")
    logger.record(LogEntry(url="https://api.openai.com/v1/"))
    logger.record(LogEntry(url="https://api.anthropic.com/v1/"))
    assert len(logger.query()) == 2


def test_sqlite_backend_query_filter_model():
    logger = LLMLogger(db_path=":memory:", backend="sqlite")
    logger.record(LogEntry(url="https://api.openai.com/v1/",
                           request_body=json.dumps({"model": "gpt-4o"})))
    logger.record(LogEntry(url="https://api.openai.com/v1/",
                           request_body=json.dumps({"model": "gpt-3.5-turbo"})))
    results = logger.query(model="gpt-4o")
    assert len(results) == 1
    assert results[0].model == "gpt-4o"


def test_sqlite_backend_query_filter_provider():
    logger = LLMLogger(db_path=":memory:", backend="sqlite")
    logger.record(LogEntry(url="https://api.openai.com/v1/"))
    logger.record(LogEntry(url="https://api.anthropic.com/v1/"))
    results = logger.query(provider="openai")
    assert all(e.provider == "openai" for e in results)


def test_sqlite_backend_summary():
    logger = LLMLogger(db_path=":memory:", backend="sqlite")
    logger.record(LogEntry(url="https://api.openai.com/v1/", latency_ms=100.0))
    logger.record(LogEntry(url="https://api.openai.com/v1/", latency_ms=200.0))
    s = logger.summary()
    assert s["total_calls"] == 2
    assert s["avg_latency_ms"] == pytest.approx(150.0)


def test_sqlite_backend_summary_empty():
    logger = LLMLogger(db_path=":memory:", backend="sqlite")
    s = logger.summary()
    assert s["total_calls"] == 0
    assert s["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# LLMLogger – JSONL backend
# ---------------------------------------------------------------------------

def test_jsonl_backend_record_and_count():
    logger = LLMLogger(backend="jsonl")
    logger.record(LogEntry(url="https://api.openai.com/v1/"))
    assert logger.count() == 1


def test_jsonl_backend_query_filter():
    logger = LLMLogger(backend="jsonl")
    logger.record(LogEntry(url="https://api.openai.com/v1/",
                           request_body=json.dumps({"model": "gpt-4o"})))
    logger.record(LogEntry(url="https://api.anthropic.com/v1/",
                           request_body=json.dumps({"model": "claude-3-haiku"})))
    assert len(logger.query(provider="openai")) == 1
    assert len(logger.query(model="claude-3-haiku")) == 1


def test_jsonl_backend_invalid_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        LLMLogger(backend="redis")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_csv(tmp_path):
    logger = LLMLogger(backend="jsonl")
    logger.record(LogEntry(url="https://api.openai.com/v1/", latency_ms=55.0))
    out = tmp_path / "log.csv"
    logger.export_csv(str(out))
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) == 2  # header + 1 data row


def test_export_csv_empty(tmp_path):
    logger = LLMLogger(backend="jsonl")
    out = tmp_path / "empty.csv"
    logger.export_csv(str(out))  # should not raise; file not created
    assert not out.exists()


def test_export_jsonl(tmp_path):
    logger = LLMLogger(backend="jsonl")
    logger.record(LogEntry(url="https://api.openai.com/v1/"))
    logger.record(LogEntry(url="https://api.anthropic.com/v1/"))
    out = tmp_path / "log.jsonl"
    logger.export_jsonl(str(out))
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_export_jsonl_roundtrip(tmp_path):
    logger = LLMLogger(backend="jsonl")
    entry = LogEntry(url="https://api.openai.com/v1/", latency_ms=99.9)
    logger.record(entry)
    out = tmp_path / "log.jsonl"
    logger.export_jsonl(str(out))

    logger2 = LLMLogger(backend="jsonl")
    with open(out) as f:
        for line in f:
            if line.strip():
                logger2.entries.append(LogEntry.from_dict(json.loads(line)))
    results = logger2.query()
    assert len(results) == 1
    assert results[0].latency_ms == pytest.approx(99.9)


# ---------------------------------------------------------------------------
# SQLite persistence on disk
# ---------------------------------------------------------------------------

def test_sqlite_persist_across_instances(tmp_path):
    db = str(tmp_path / "test.db")
    l1 = LLMLogger(db_path=db, backend="sqlite")
    l1.record(LogEntry(url="https://api.openai.com/v1/"))
    l1.conn.close()

    l2 = LLMLogger(db_path=db, backend="sqlite")
    assert l2.count() == 1


# ---------------------------------------------------------------------------
# urllib patching
# ---------------------------------------------------------------------------

def test_patch_and_unpatch_urllib():
    original = urllib_request.urlopen
    logger = LLMLogger(backend="jsonl")
    patch_urllib(logger)
    assert urllib_request.urlopen is not original
    unpatch_urllib()
    assert urllib_request.urlopen is original


def test_patch_idempotent_unpatch():
    unpatch_urllib()  # safe to call even without prior patch
    from urllib import request as ur
    original = ur.urlopen
    patch_urllib(LLMLogger(backend="jsonl"))
    unpatch_urllib()
    assert ur.urlopen is original


# ---------------------------------------------------------------------------
# session() context manager
# ---------------------------------------------------------------------------

def test_session_context_manager_no_file():
    with session(log_file=None, backend="jsonl", auto_patch=False) as logger:
        entry = LogEntry(url="https://api.openai.com/v1/")
        logger.record(entry)
    assert logger.count() == 1


def test_session_context_manager_writes_file(tmp_path):
    log_path = str(tmp_path / "session.jsonl")
    with session(log_file=log_path, backend="jsonl", auto_patch=False) as logger:
        logger.record(LogEntry(url="https://api.openai.com/v1/"))
    assert pathlib.Path(log_path).exists()
    lines = [l for l in pathlib.Path(log_path).read_text().splitlines() if l.strip()]
    assert len(lines) == 1


def test_session_restores_urllib_after_exit():
    from urllib import request as ur
    original = ur.urlopen
    with session(log_file=None, backend="jsonl", auto_patch=True):
        assert ur.urlopen is not original
    assert ur.urlopen is original
