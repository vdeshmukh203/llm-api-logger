"""
Tests for llm_api_logger.

Covers: cost estimation, provider/model extraction, token parsing,
LogEntry auto-population, LLMLogger backends (SQLite + JSONL),
query filtering, summary statistics, CSV/JSONL export, the session()
context manager, and the CLI.
"""

import csv
import json
import pathlib
import sys

import pytest

# Make the repo root importable when running pytest from the project directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import llm_api_logger as lal
from llm_api_logger import (
    COST_TABLE,
    LogEntry,
    LLMLogger,
    _extract_model,
    _extract_provider,
    _is_llm,
    _tok,
    estimate_cost,
    session,
)


# ---------------------------------------------------------------------------
# Backwards-compatible API surface (keep existing tests passing)
# ---------------------------------------------------------------------------

def test_import_logrecord_alias():
    assert hasattr(lal, "LogRecord")


def test_backends_aliases():
    assert hasattr(lal, "JSONLBackend")
    assert hasattr(lal, "SQLiteBackend")
    assert hasattr(lal, "StdoutBackend")


def test_detect_provider_callable():
    assert callable(lal._detect_provider)


def test_extract_model_callable():
    assert callable(lal._extract_model)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def test_estimate_cost_known_model():
    # gpt-4o: $5 input + $15 output per 1M tokens
    cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == pytest.approx(20.0)


def test_estimate_cost_zero_tokens():
    assert estimate_cost("gpt-4o", 0, 0) == pytest.approx(0.0)


def test_estimate_cost_fractional_tokens():
    cost = estimate_cost("gpt-4o-mini", 500_000, 500_000)
    assert cost == pytest.approx((0.15 + 0.60) / 2)


def test_estimate_cost_unknown_model_raises():
    with pytest.raises(ValueError, match="not found"):
        estimate_cost("no-such-model-xyz", 100, 100)


def test_estimate_cost_all_table_entries():
    for model in COST_TABLE:
        cost = estimate_cost(model, 1_000, 1_000)
        assert cost >= 0.0, f"Negative cost for {model}"


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://api.openai.com/v1/chat/completions",                    "openai"),
    ("https://api.anthropic.com/v1/messages",                        "anthropic"),
    ("https://generativelanguage.googleapis.com/v1/models/gemini",   "google"),
    ("https://api.mistral.ai/v1/chat/completions",                   "mistral"),
    ("https://api.together.xyz/v1/completions",                      "together"),
    ("https://api.cohere.ai/v1/generate",                            "cohere"),
    ("https://api-inference.huggingface.co/models/gpt2",             "huggingface"),
    ("https://example.com/api/v1",                                   "unknown"),
])
def test_extract_provider(url, expected):
    assert _extract_provider(url) == expected


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------

def test_extract_model_from_request_body():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    assert _extract_model(body, None) == "gpt-4o"


def test_extract_model_from_response_body():
    resp = json.dumps({"model": "claude-3-5-sonnet", "content": []})
    assert _extract_model(None, resp) == "claude-3-5-sonnet"


def test_extract_model_request_takes_priority():
    req  = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({"model": "gpt-3.5-turbo"})
    assert _extract_model(req, resp) == "gpt-4o"


def test_extract_model_model_id_key():
    body = json.dumps({"modelId": "anthropic.claude-v2"})
    assert _extract_model(body, None) == "anthropic.claude-v2"


def test_extract_model_engine_key():
    body = json.dumps({"engine": "davinci"})
    assert _extract_model(body, None) == "davinci"


def test_extract_model_no_body():
    assert _extract_model(None, None) == "unknown"


def test_extract_model_invalid_json():
    assert _extract_model("{bad json}", "{also bad}") == "unknown"


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def test_tok_openai_format():
    body = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
    assert _tok(body) == (100, 50)


def test_tok_google_format():
    body = json.dumps({"usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 75}})
    assert _tok(body) == (200, 75)


def test_tok_missing_keys_returns_zeros():
    body = json.dumps({"usage": {}})
    assert _tok(body) == (0, 0)


def test_tok_none_input():
    assert _tok(None) == (0, 0)


def test_tok_empty_string():
    assert _tok("") == (0, 0)


def test_tok_invalid_json():
    assert _tok("not valid json") == (0, 0)


# ---------------------------------------------------------------------------
# LogEntry auto-population
# ---------------------------------------------------------------------------

def test_log_entry_auto_provider():
    e = LogEntry(url="https://api.openai.com/v1/chat/completions")
    assert e.provider == "openai"


def test_log_entry_auto_model():
    req = json.dumps({"model": "gpt-4o", "messages": []})
    e = LogEntry(url="https://api.openai.com/v1/chat/completions", request_body=req)
    assert e.model == "gpt-4o"


def test_log_entry_auto_tokens():
    resp = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    e = LogEntry(url="https://api.openai.com/v1/chat/completions", response_body=resp)
    assert e.tokens_in == 10
    assert e.tokens_out == 5


def test_log_entry_auto_cost():
    req  = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}})
    e = LogEntry(
        url="https://api.openai.com/v1/chat/completions",
        request_body=req,
        response_body=resp,
    )
    assert e.cost_usd == pytest.approx(20.0)


def test_log_entry_unknown_model_zero_cost():
    resp = json.dumps({"usage": {"prompt_tokens": 1000, "completion_tokens": 500}})
    e = LogEntry(url="https://example.com/api", response_body=resp)
    assert e.cost_usd == 0.0


def test_log_entry_uuid_id():
    e1 = LogEntry(url="https://api.openai.com/v1/chat/completions")
    e2 = LogEntry(url="https://api.openai.com/v1/chat/completions")
    assert e1.id != e2.id


def test_log_entry_to_dict_roundtrip():
    e = LogEntry(
        url="https://api.openai.com/v1/chat/completions",
        tokens_in=42,
        tokens_out=17,
        latency_ms=123.4,
    )
    e2 = LogEntry.from_dict(e.to_dict())
    assert e.id == e2.id
    assert e.url == e2.url
    assert e.tokens_in == e2.tokens_in
    assert e.tokens_out == e2.tokens_out
    assert e.latency_ms == pytest.approx(e2.latency_ms)


def test_log_entry_from_dict_ignores_unknown_keys():
    data = LogEntry(url="https://api.openai.com/v1/chat/completions").to_dict()
    data["unknown_future_field"] = "value"
    e = LogEntry.from_dict(data)  # must not raise
    assert e.url == "https://api.openai.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# LLMLogger — SQLite backend
# ---------------------------------------------------------------------------

def test_sqlite_record_and_count():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    assert log.count() == 0
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions"))
    assert log.count() == 1


def test_sqlite_query_returns_all_by_default():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    for _ in range(3):
        log.record(LogEntry(url="https://api.openai.com/v1/chat/completions"))
    assert len(log.query()) == 3


def test_sqlite_query_filter_model():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        request_body=json.dumps({"model": "gpt-4o"})))
    log.record(LogEntry(url="https://api.anthropic.com/v1/messages",
                        request_body=json.dumps({"model": "claude-3-haiku"})))
    results = log.query(model="gpt-4o")
    assert len(results) == 1
    assert results[0].model == "gpt-4o"


def test_sqlite_query_filter_provider():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions"))
    log.record(LogEntry(url="https://api.anthropic.com/v1/messages"))
    assert len(log.query(provider="anthropic")) == 1
    assert len(log.query(provider="openai")) == 1


def test_sqlite_query_filter_status_code():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions", status_code=200))
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions", status_code=429))
    assert len(log.query(status_code=429)) == 1


def test_sqlite_query_ordered_newest_first():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        timestamp="2024-01-01T00:00:00"))
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        timestamp="2024-06-01T00:00:00"))
    entries = log.query()
    assert entries[0].timestamp > entries[1].timestamp


def test_sqlite_summary_empty():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    s = log.summary()
    assert s["total_calls"] == 0
    assert s["total_cost_usd"] == 0.0


def test_sqlite_summary_with_entries():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    req  = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
    log.record(LogEntry(
        url="https://api.openai.com/v1/chat/completions",
        request_body=req, response_body=resp, latency_ms=300.0,
    ))
    s = log.summary()
    assert s["total_calls"] == 1
    assert s["total_tokens_in"] == 100
    assert s["total_tokens_out"] == 50
    assert s["avg_latency_ms"] == pytest.approx(300.0)
    assert "gpt-4o" in s["calls_by_model"]


def test_sqlite_export_csv(tmp_path):
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        tokens_in=10, tokens_out=5))
    out = str(tmp_path / "out.csv")
    log.export_csv(out)
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["tokens_in"] == "10"
    assert rows[0]["tokens_out"] == "5"


def test_sqlite_export_csv_empty_noop(tmp_path):
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    out = str(tmp_path / "empty.csv")
    log.export_csv(out)   # should not raise or create file
    assert not pathlib.Path(out).exists()


def test_sqlite_export_jsonl(tmp_path):
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        tokens_in=10, tokens_out=5))
    out = str(tmp_path / "out.jsonl")
    log.export_jsonl(out)
    with open(out) as fh:
        lines = [l for l in fh if l.strip()]
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["tokens_in"] == 10
    assert d["tokens_out"] == 5


def test_sqlite_close():
    log = LLMLogger(db_path=":memory:", backend="sqlite")
    log.close()
    log.close()  # idempotent — must not raise


# ---------------------------------------------------------------------------
# LLMLogger — JSONL backend
# ---------------------------------------------------------------------------

def test_jsonl_record_and_count():
    log = LLMLogger(db_path=":memory:", backend="jsonl")
    assert log.count() == 0
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions"))
    assert log.count() == 1


def test_jsonl_query_filter_provider():
    log = LLMLogger(db_path=":memory:", backend="jsonl")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions"))
    log.record(LogEntry(url="https://api.anthropic.com/v1/messages"))
    assert len(log.query(provider="openai")) == 1


def test_jsonl_query_filter_model():
    log = LLMLogger(db_path=":memory:", backend="jsonl")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        request_body=json.dumps({"model": "gpt-4o"})))
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        request_body=json.dumps({"model": "gpt-4o-mini"})))
    assert len(log.query(model="gpt-4o")) == 1


def test_jsonl_summary():
    log = LLMLogger(db_path=":memory:", backend="jsonl")
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        tokens_in=50, tokens_out=25, latency_ms=100.0))
    log.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                        tokens_in=150, tokens_out=75, latency_ms=200.0))
    s = log.summary()
    assert s["total_calls"] == 2
    assert s["total_tokens_in"] == 200
    assert s["total_tokens_out"] == 100
    assert s["avg_latency_ms"] == pytest.approx(150.0)


def test_jsonl_close_noop():
    log = LLMLogger(db_path=":memory:", backend="jsonl")
    log.close()  # must not raise


def test_jsonl_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        LLMLogger(backend="nosuchbackend")


# ---------------------------------------------------------------------------
# LLM detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,body,expected", [
    ("https://api.openai.com/v1/chat/completions", None, True),
    ("https://api.anthropic.com/v1/messages",      None, True),
    ("https://example.com/api",  json.dumps({"model": "gpt-4o"}), True),
    ("https://example.com/api",  json.dumps({"engine": "davinci"}), True),
    ("https://example.com/api",  None, False),
    ("https://example.com/api",  json.dumps({"foo": "bar"}), False),
])
def test_is_llm(url, body, expected):
    assert _is_llm(url, body) == expected


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

def test_session_yields_llmlogger():
    with session(backend="sqlite") as log:
        assert isinstance(log, LLMLogger)


def test_session_in_memory_records():
    with session(backend="sqlite") as log:
        log.record(LogEntry(url="https://api.openai.com/v1/chat/completions"))
        assert log.count() == 1


def test_session_jsonl_exports_file(tmp_path):
    out = str(tmp_path / "session.jsonl")
    with session(log_file=out, backend="jsonl", auto_patch=False) as log:
        log.record(LogEntry(url="https://api.openai.com/v1/chat/completions"))
    # File should be written on exit
    assert pathlib.Path(out).exists()
    with open(out) as fh:
        lines = [l for l in fh if l.strip()]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_summary_empty(tmp_path, capsys):
    # Write a minimal JSONL file
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    lal.main(["summary", str(jsonl)])
    out = capsys.readouterr().out
    assert "Total API Calls" in out
    assert "0" in out


def test_cli_summary_with_data(tmp_path, capsys):
    jsonl = tmp_path / "log.jsonl"
    entry = LogEntry(url="https://api.openai.com/v1/chat/completions",
                     tokens_in=100, tokens_out=50, latency_ms=200.0)
    jsonl.write_text(json.dumps(entry.to_dict()) + "\n")
    lal.main(["summary", str(jsonl)])
    out = capsys.readouterr().out
    assert "Total API Calls    : 1" in out


def test_cli_query_filter(tmp_path, capsys):
    jsonl = tmp_path / "log.jsonl"
    e1 = LogEntry(url="https://api.openai.com/v1/chat/completions",
                  request_body=json.dumps({"model": "gpt-4o"}))
    e2 = LogEntry(url="https://api.anthropic.com/v1/messages",
                  request_body=json.dumps({"model": "claude-3-haiku"}))
    jsonl.write_text(json.dumps(e1.to_dict()) + "\n" + json.dumps(e2.to_dict()) + "\n")
    lal.main(["query", str(jsonl), "--model", "gpt-4o"])
    out = capsys.readouterr().out
    assert "Found 1 entries" in out


def test_cli_export_csv(tmp_path, capsys):
    jsonl = tmp_path / "log.jsonl"
    entry = LogEntry(url="https://api.openai.com/v1/chat/completions", tokens_in=7)
    jsonl.write_text(json.dumps(entry.to_dict()) + "\n")
    out_csv = str(tmp_path / "out.csv")
    lal.main(["export", str(jsonl), "--output", out_csv, "--format", "csv"])
    assert pathlib.Path(out_csv).exists()
    with open(out_csv) as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["tokens_in"] == "7"


def test_cli_entry_point_alias():
    assert callable(lal._cli)
    assert lal._cli is lal.main


def test_cli_no_command_prints_help(capsys):
    lal.main([])
    out = capsys.readouterr().out
    assert "usage" in out.lower() or "COMMAND" in out or "command" in out.lower()
