"""
Comprehensive test suite for llm_api_logger.

Run with:  pytest tests/ -v
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the root module is importable when running tests from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

import llm_api_logger as lal
from llm_api_logger import (
    COST_TABLE,
    LLMLogger,
    LogEntry,
    _extract_model,
    _extract_provider,
    _tok,
    estimate_cost,
    patch_urllib,
    session,
    unpatch_urllib,
)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(20.0)  # 5.00 input + 15.00 output

    def test_zero_tokens(self):
        assert estimate_cost("gpt-4o", 0, 0) == pytest.approx(0.0)

    def test_only_input_tokens(self):
        cost = estimate_cost("gpt-4o", 1_000_000, 0)
        assert cost == pytest.approx(5.0)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found"):
            estimate_cost("nonexistent-model-xyz", 100, 100)

    def test_prefix_match_versioned_openai(self):
        # "gpt-4o-2024-05-13" should resolve via "gpt-4o" prefix
        cost = estimate_cost("gpt-4o-2024-05-13", 1_000_000, 0)
        assert cost == pytest.approx(5.0)

    def test_prefix_match_versioned_anthropic(self):
        cost = estimate_cost("claude-3-5-sonnet-20241022", 1_000_000, 0)
        assert cost == pytest.approx(3.0)

    def test_all_cost_table_models_compute(self):
        for model in COST_TABLE:
            cost = estimate_cost(model, 500_000, 250_000)
            assert cost >= 0


# ---------------------------------------------------------------------------
# _extract_provider
# ---------------------------------------------------------------------------

class TestExtractProvider:
    def test_openai(self):
        assert _extract_provider("https://api.openai.com/v1/chat/completions") == "openai"

    def test_anthropic(self):
        assert _extract_provider("https://api.anthropic.com/v1/messages") == "anthropic"

    def test_google(self):
        assert _extract_provider("https://generativelanguage.googleapis.com/v1/models/gemini-pro") == "google"

    def test_gemini_in_url(self):
        assert _extract_provider("https://example.com/gemini/v1/generate") == "google"

    def test_mistral(self):
        assert _extract_provider("https://api.mistral.ai/v1/chat/completions") == "mistral"

    def test_together(self):
        assert _extract_provider("https://api.together.xyz/inference") == "together"

    def test_cohere(self):
        assert _extract_provider("https://api.cohere.ai/v1/generate") == "cohere"

    def test_huggingface(self):
        assert _extract_provider("https://api-inference.huggingface.co/models/gpt2") == "huggingface"

    def test_unknown(self):
        assert _extract_provider("https://example.com/api/v1") == "unknown"

    def test_case_insensitive(self):
        assert _extract_provider("https://API.OPENAI.COM/v1/chat") == "openai"


# ---------------------------------------------------------------------------
# _extract_model
# ---------------------------------------------------------------------------

class TestExtractModel:
    def test_from_request_model_field(self):
        req = json.dumps({"model": "gpt-4o", "messages": []})
        assert _extract_model(req, None) == "gpt-4o"

    def test_from_response_model_field(self):
        resp = json.dumps({"id": "msg_01", "model": "claude-3-opus-20240229"})
        assert _extract_model(None, resp) == "claude-3-opus-20240229"

    def test_request_takes_priority(self):
        req = json.dumps({"model": "from-request"})
        resp = json.dumps({"model": "from-response"})
        assert _extract_model(req, resp) == "from-request"

    def test_model_id_key(self):
        req = json.dumps({"modelId": "amazon.titan-tg1-large"})
        assert _extract_model(req, None) == "amazon.titan-tg1-large"

    def test_engine_key(self):
        req = json.dumps({"engine": "davinci-002"})
        assert _extract_model(req, None) == "davinci-002"

    def test_none_inputs(self):
        assert _extract_model(None, None) == "unknown"

    def test_empty_json(self):
        assert _extract_model("{}", "{}") == "unknown"

    def test_invalid_json_ignored(self):
        assert _extract_model("not-json", None) == "unknown"


# ---------------------------------------------------------------------------
# _tok
# ---------------------------------------------------------------------------

class TestTok:
    def test_openai_format(self):
        resp = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        assert _tok(resp) == (100, 50)

    def test_anthropic_format(self):
        resp = json.dumps({"usage": {"input_tokens": 200, "output_tokens": 75}})
        assert _tok(resp) == (200, 75)

    def test_google_format(self):
        resp = json.dumps({"usageMetadata": {"promptTokenCount": 150, "candidatesTokenCount": 60}})
        assert _tok(resp) == (150, 60)

    def test_cohere_format(self):
        resp = json.dumps({"meta": {"billed_units": {"input_tokens": 80, "output_tokens": 30}}})
        assert _tok(resp) == (80, 30)

    def test_none_input(self):
        assert _tok(None) == (0, 0)

    def test_empty_string(self):
        assert _tok("") == (0, 0)

    def test_invalid_json(self):
        assert _tok("not-json") == (0, 0)

    def test_missing_usage_fields(self):
        assert _tok(json.dumps({"usage": {}})) == (0, 0)

    def test_non_dict_response(self):
        assert _tok(json.dumps([1, 2, 3])) == (0, 0)


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

class TestLogEntry:
    def test_auto_provider(self):
        e = LogEntry(url="https://api.openai.com/v1/chat/completions")
        assert e.provider == "openai"

    def test_auto_model_from_request(self):
        req = json.dumps({"model": "gpt-4o-mini"})
        e = LogEntry(url="https://api.openai.com/v1/chat/completions", request_body=req)
        assert e.model == "gpt-4o-mini"

    def test_auto_tokens_and_cost(self):
        req = json.dumps({"model": "gpt-4o"})
        resp = json.dumps({"model": "gpt-4o", "usage": {"prompt_tokens": 1000, "completion_tokens": 500}})
        e = LogEntry(
            url="https://api.openai.com/v1/chat/completions",
            request_body=req,
            response_body=resp,
        )
        assert e.tokens_in == 1000
        assert e.tokens_out == 500
        assert e.cost_usd > 0

    def test_explicit_tokens_not_overridden(self):
        resp = json.dumps({"usage": {"prompt_tokens": 999, "completion_tokens": 999}})
        e = LogEntry(url="u", tokens_in=42, tokens_out=42, response_body=resp)
        assert e.tokens_in == 42
        assert e.tokens_out == 42

    def test_uuid_assigned(self):
        e = LogEntry()
        assert len(e.id) == 36  # UUID4 format

    def test_default_method(self):
        assert LogEntry().method == "POST"

    def test_default_status_code(self):
        assert LogEntry().status_code == 200

    def test_to_dict_roundtrip(self):
        e = LogEntry(url="https://api.openai.com/v1/chat", latency_ms=123.4)
        d = e.to_dict()
        assert isinstance(d, dict)
        restored = LogEntry.from_dict(d)
        assert restored.id == e.id
        assert restored.latency_ms == e.latency_ms
        assert restored.url == e.url

    def test_cost_zero_for_unknown_model(self):
        resp = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        e = LogEntry(url="u", model="totally-unknown-model", response_body=resp)
        assert e.cost_usd == 0.0

    def test_error_field(self):
        e = LogEntry(url="u", error="Connection timeout")
        assert e.error == "Connection timeout"


# ---------------------------------------------------------------------------
# LLMLogger — SQLite backend
# ---------------------------------------------------------------------------

class TestLLMLoggerSQLite:
    def setup_method(self):
        self.logger = LLMLogger(db_path=":memory:", backend="sqlite")

    def test_initial_count(self):
        assert self.logger.count() == 0

    def test_record_and_count(self):
        self.logger.record(LogEntry(url="https://api.openai.com/v1/chat"))
        assert self.logger.count() == 1

    def test_query_all(self):
        self.logger.record(LogEntry(url="u"))
        results = self.logger.query()
        assert len(results) == 1
        assert isinstance(results[0], LogEntry)

    def test_query_filter_model(self):
        self.logger.record(LogEntry(url="u", model="gpt-4o"))
        self.logger.record(LogEntry(url="u", model="claude-3-opus"))
        assert len(self.logger.query(model="gpt-4o")) == 1
        assert len(self.logger.query(model="claude-3-opus")) == 1
        assert len(self.logger.query(model="llama-3-70b")) == 0

    def test_query_filter_provider(self):
        self.logger.record(LogEntry(url="https://api.openai.com/v1"))
        self.logger.record(LogEntry(url="https://api.anthropic.com/v1"))
        assert len(self.logger.query(provider="openai")) == 1
        assert len(self.logger.query(provider="anthropic")) == 1

    def test_query_filter_status_code(self):
        self.logger.record(LogEntry(url="u", status_code=200))
        self.logger.record(LogEntry(url="u", status_code=429))
        assert len(self.logger.query(status_code=200)) == 1
        assert len(self.logger.query(status_code=429)) == 1

    def test_query_newest_first(self):
        self.logger.record(LogEntry(url="u", timestamp="2024-01-01T00:00:00"))
        self.logger.record(LogEntry(url="u", timestamp="2024-06-01T00:00:00"))
        results = self.logger.query()
        assert results[0].timestamp > results[1].timestamp

    def test_summary_empty(self):
        s = self.logger.summary()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0

    def test_summary_aggregation(self):
        self.logger.record(LogEntry(url="u", model="gpt-4o", tokens_in=100, tokens_out=50, cost_usd=0.001, latency_ms=200))
        self.logger.record(LogEntry(url="u", model="gpt-4o", tokens_in=200, tokens_out=100, cost_usd=0.002, latency_ms=400))
        s = self.logger.summary()
        assert s["total_calls"] == 2
        assert s["total_tokens_in"] == 300
        assert s["total_tokens_out"] == 150
        assert s["total_cost_usd"] == pytest.approx(0.003)
        assert s["avg_latency_ms"] == pytest.approx(300.0)
        assert s["calls_by_model"]["gpt-4o"] == 2

    def test_summary_calls_by_provider(self):
        self.logger.record(LogEntry(url="https://api.openai.com/v1"))
        self.logger.record(LogEntry(url="https://api.anthropic.com/v1"))
        s = self.logger.summary()
        assert s["calls_by_provider"]["openai"] == 1
        assert s["calls_by_provider"]["anthropic"] == 1

    def test_export_jsonl(self, tmp_path):
        self.logger.record(LogEntry(url="https://api.openai.com/v1/chat"))
        outfile = str(tmp_path / "out.jsonl")
        self.logger.export_jsonl(outfile)
        lines = [l for l in Path(outfile).read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert "id" in json.loads(lines[0])

    def test_export_csv(self, tmp_path):
        self.logger.record(LogEntry(url="https://api.openai.com/v1/chat"))
        outfile = str(tmp_path / "out.csv")
        self.logger.export_csv(outfile)
        with open(outfile, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert "id" in rows[0]
        assert "model" in rows[0]

    def test_export_csv_empty(self, tmp_path):
        outfile = str(tmp_path / "empty.csv")
        self.logger.export_csv(outfile)
        assert not Path(outfile).exists()


# ---------------------------------------------------------------------------
# LLMLogger — JSONL backend
# ---------------------------------------------------------------------------

class TestLLMLoggerJSONL:
    def setup_method(self):
        self.logger = LLMLogger(db_path=":memory:", backend="jsonl")

    def test_record_and_count(self):
        self.logger.record(LogEntry(url="https://api.anthropic.com/v1/messages"))
        assert self.logger.count() == 1

    def test_query_returns_entry(self):
        entry = LogEntry(url="u", model="claude-3-haiku")
        self.logger.record(entry)
        results = self.logger.query()
        assert len(results) == 1
        assert results[0].id == entry.id

    def test_filter_model(self):
        self.logger.record(LogEntry(url="u", model="a"))
        self.logger.record(LogEntry(url="u", model="b"))
        assert len(self.logger.query(model="a")) == 1

    def test_summary(self):
        self.logger.record(LogEntry(url="u", cost_usd=0.005, latency_ms=100))
        self.logger.record(LogEntry(url="u", cost_usd=0.010, latency_ms=300))
        s = self.logger.summary()
        assert s["total_calls"] == 2
        assert s["total_cost_usd"] == pytest.approx(0.015)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            LLMLogger(backend="redis")


# ---------------------------------------------------------------------------
# session context manager
# ---------------------------------------------------------------------------

class TestSession:
    def test_yields_llmlogger(self):
        with session(backend="sqlite") as log:
            assert isinstance(log, LLMLogger)

    def test_auto_patch_and_unpatch(self):
        import urllib.request as ur
        original = ur.urlopen
        with session(backend="sqlite") as _:
            assert ur.urlopen is not original
        assert ur.urlopen is original

    def test_no_auto_patch(self):
        import urllib.request as ur
        original = ur.urlopen
        with session(backend="sqlite", auto_patch=False) as _:
            assert ur.urlopen is original

    def test_jsonl_session_writes_file(self, tmp_path):
        log_file = str(tmp_path / "run.jsonl")
        with session(log_file=log_file, backend="jsonl") as log:
            log.record(LogEntry(url="https://api.openai.com/v1"))
        assert Path(log_file).exists()
        lines = [l for l in Path(log_file).read_text().splitlines() if l.strip()]
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
# ---------------------------------------------------------------------------

class TestBackwardsCompat:
    def test_log_record_alias(self):
        assert lal.LogRecord is lal.LogEntry

    def test_backend_aliases(self):
        assert lal.JSONLBackend is lal.LLMLogger
        assert lal.SQLiteBackend is lal.LLMLogger
        assert lal.StdoutBackend is lal.LLMLogger

    def test_detect_provider_alias(self):
        assert callable(lal._detect_provider)
        assert lal._detect_provider("https://api.openai.com") == "openai"

    def test_extract_model_callable(self):
        assert callable(lal._extract_model)


# ---------------------------------------------------------------------------
# Module-level smoke test
# ---------------------------------------------------------------------------

def test_module_version():
    assert hasattr(lal, "__version__")
    parts = lal.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cost_table_not_empty():
    assert len(COST_TABLE) >= 10
    for key, val in COST_TABLE.items():
        assert "input" in val and "output" in val
        assert val["input"] >= 0 and val["output"] >= 0
