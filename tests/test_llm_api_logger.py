"""
Comprehensive test suite for llm_api_logger.

Covers: cost estimation, provider/model extraction, token parsing,
LogEntry construction (including SHA-256 hashes), LLMLogger CRUD,
both backends, export functions, the session context manager, and
backwards-compatibility aliases.
"""

import csv
import hashlib
import json
import os
import sys
import tempfile
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import llm_api_logger as lal
from llm_api_logger import (
    COST_TABLE,
    LogEntry,
    LLMLogger,
    _extract_model,
    _extract_provider,
    _load_jsonl_into,
    _tok,
    estimate_cost,
    session,
)


# ===========================================================================
# estimate_cost
# ===========================================================================

class TestEstimateCost:
    def test_known_model_calculates_correctly(self):
        cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(5.00 + 15.00)

    def test_zero_tokens(self):
        assert estimate_cost("gpt-4o-mini", 0, 0) == pytest.approx(0.0)

    def test_only_input_tokens(self):
        cost = estimate_cost("gpt-4o", 1_000_000, 0)
        assert cost == pytest.approx(5.00)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not in COST_TABLE"):
            estimate_cost("totally-unknown-model-xyz", 100, 100)

    def test_all_table_entries_are_valid(self):
        for model in COST_TABLE:
            cost = estimate_cost(model, 1000, 1000)
            assert cost >= 0.0


# ===========================================================================
# _extract_provider
# ===========================================================================

class TestExtractProvider:
    @pytest.mark.parametrize("url,expected", [
        ("https://api.openai.com/v1/chat/completions",                      "openai"),
        ("https://api.anthropic.com/v1/messages",                           "anthropic"),
        ("https://generativelanguage.googleapis.com/v1beta/models/gemini",  "google"),
        ("https://api.mistral.ai/v1/chat/completions",                      "mistral"),
        ("https://api.cohere.ai/v1/generate",                               "cohere"),
        ("https://api.together.xyz/inference",                              "together"),
        ("https://api-inference.huggingface.co/models/gpt2",               "huggingface"),
        ("https://api.groq.com/openai/v1/chat/completions",                 "groq"),
        ("https://api.fireworks.ai/inference/v1/completions",               "fireworks"),
        ("https://api.perplexity.ai/chat/completions",                      "perplexity"),
        ("https://example.com/api/endpoint",                                "unknown"),
    ])
    def test_provider_detection(self, url, expected):
        assert _extract_provider(url) == expected


# ===========================================================================
# _extract_model
# ===========================================================================

class TestExtractModel:
    def test_from_request_body(self):
        req = json.dumps({"model": "gpt-4o", "messages": []})
        assert _extract_model(req, None) == "gpt-4o"

    def test_from_response_body(self):
        resp = json.dumps({"model": "claude-3-haiku", "choices": []})
        assert _extract_model(None, resp) == "claude-3-haiku"

    def test_request_takes_priority_over_response(self):
        req  = json.dumps({"model": "from-request"})
        resp = json.dumps({"model": "from-response"})
        assert _extract_model(req, resp) == "from-request"

    def test_alternative_keys(self):
        for key in ("modelId", "model_id", "engine"):
            body = json.dumps({key: f"model-via-{key}"})
            assert _extract_model(body, None) == f"model-via-{key}"

    def test_no_bodies(self):
        assert _extract_model(None, None) == "unknown"

    def test_invalid_json(self):
        assert _extract_model("not-json", "also-not-json") == "unknown"

    def test_non_dict_json(self):
        assert _extract_model("[1, 2, 3]", None) == "unknown"


# ===========================================================================
# _tok
# ===========================================================================

class TestTok:
    def test_openai_format(self):
        body = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        assert _tok(body) == (100, 50)

    def test_anthropic_format(self):
        body = json.dumps({"usage": {"input_tokens": 200, "output_tokens": 80}})
        assert _tok(body) == (200, 80)

    def test_google_format(self):
        body = json.dumps({"usageMetadata": {"promptTokenCount": 150, "candidatesTokenCount": 60}})
        assert _tok(body) == (150, 60)

    def test_none_input(self):
        assert _tok(None) == (0, 0)

    def test_empty_string(self):
        assert _tok("") == (0, 0)

    def test_invalid_json(self):
        assert _tok("definitely-not-json") == (0, 0)

    def test_missing_usage_key(self):
        body = json.dumps({"choices": []})
        assert _tok(body) == (0, 0)


# ===========================================================================
# LogEntry
# ===========================================================================

class TestLogEntry:
    def test_provider_inferred_from_url(self):
        entry = LogEntry(url="https://api.openai.com/v1/chat/completions")
        assert entry.provider == "openai"

    def test_model_inferred_from_request_body(self):
        req = json.dumps({"model": "gpt-4o", "messages": []})
        entry = LogEntry(url="https://api.openai.com", request_body=req)
        assert entry.model == "gpt-4o"

    def test_tokens_parsed_from_response(self):
        resp = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        entry = LogEntry(url="https://api.openai.com", response_body=resp)
        assert entry.tokens_in == 10
        assert entry.tokens_out == 5

    def test_cost_computed_automatically(self):
        req  = json.dumps({"model": "gpt-4o"})
        resp = json.dumps({"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})
        entry = LogEntry(url="https://api.openai.com", request_body=req, response_body=resp)
        assert entry.cost_usd == pytest.approx(5.00)

    def test_sha256_request_hash(self):
        body = json.dumps({"model": "gpt-4o"})
        entry = LogEntry(url="https://api.openai.com", request_body=body)
        assert entry.request_hash == hashlib.sha256(body.encode()).hexdigest()

    def test_sha256_response_hash(self):
        body = json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        entry = LogEntry(url="https://api.openai.com", response_body=body)
        assert entry.response_hash == hashlib.sha256(body.encode()).hexdigest()

    def test_no_hash_when_no_body(self):
        entry = LogEntry(url="https://api.openai.com")
        assert entry.request_hash is None
        assert entry.response_hash is None

    def test_to_dict_roundtrip(self):
        entry = LogEntry(url="https://api.openai.com", model="gpt-4o", provider="openai")
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["model"] == "gpt-4o"
        assert "request_hash" in d

    def test_from_dict_roundtrip(self):
        entry = LogEntry(url="https://api.openai.com", model="gpt-4o", provider="openai")
        restored = LogEntry.from_dict(entry.to_dict())
        assert restored.id == entry.id
        assert restored.model == entry.model
        assert restored.request_hash == entry.request_hash

    def test_from_dict_ignores_unknown_keys(self):
        data = {"url": "https://api.openai.com", "totally_unknown_field": "value"}
        entry = LogEntry.from_dict(data)
        assert entry.url == "https://api.openai.com"

    def test_from_dict_handles_missing_hash_fields(self):
        # Simulate an older log file without hash columns
        data = {
            "id": "abc", "url": "https://api.openai.com", "method": "POST",
            "provider": "openai", "model": "gpt-4o", "request_body": None,
            "response_body": None, "status_code": 200, "latency_ms": 0.0,
            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "timestamp": "2026-01-01T00:00:00", "error": None,
        }
        entry = LogEntry.from_dict(data)
        assert entry.id == "abc"
        assert entry.request_hash is None


# ===========================================================================
# LLMLogger – SQLite backend
# ===========================================================================

class TestLLMLoggerSQLite:
    def _logger(self) -> LLMLogger:
        return LLMLogger(db_path=":memory:", backend="sqlite")

    def test_empty_count(self):
        assert self._logger().count() == 0

    def test_record_and_count(self):
        logger = self._logger()
        logger.record(LogEntry(url="https://api.openai.com", model="gpt-4o", provider="openai"))
        assert logger.count() == 1

    def test_query_all(self):
        logger = self._logger()
        logger.record(LogEntry(url="https://api.openai.com"))
        logger.record(LogEntry(url="https://api.anthropic.com"))
        assert len(logger.query()) == 2

    def test_query_by_model(self):
        logger = self._logger()
        logger.record(LogEntry(url="https://api.openai.com",   model="gpt-4o",       provider="openai"))
        logger.record(LogEntry(url="https://api.openai.com",   model="gpt-3.5-turbo", provider="openai"))
        results = logger.query(model="gpt-4o")
        assert len(results) == 1
        assert results[0].model == "gpt-4o"

    def test_query_by_provider(self):
        logger = self._logger()
        logger.record(LogEntry(url="https://api.openai.com",   model="gpt-4o",       provider="openai"))
        logger.record(LogEntry(url="https://api.anthropic.com", model="claude-3-haiku", provider="anthropic"))
        results = logger.query(provider="anthropic")
        assert len(results) == 1
        assert results[0].provider == "anthropic"

    def test_query_by_status_code(self):
        logger = self._logger()
        logger.record(LogEntry(url="https://api.openai.com", status_code=200))
        logger.record(LogEntry(url="https://api.openai.com", status_code=429))
        assert len(logger.query(status_code=429)) == 1

    def test_summary_empty(self):
        s = self._logger().summary()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0
        assert s["avg_latency_ms"] == 0.0

    def test_summary_aggregates(self):
        logger = self._logger()
        req  = json.dumps({"model": "gpt-4o"})
        resp = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        logger.record(LogEntry(url="https://api.openai.com/v1/chat/completions",
                               request_body=req, response_body=resp, latency_ms=123.0))
        s = logger.summary()
        assert s["total_calls"] == 1
        assert s["total_tokens_in"] == 100
        assert s["total_tokens_out"] == 50
        assert s["avg_latency_ms"] == pytest.approx(123.0)

    def test_export_jsonl(self):
        logger = self._logger()
        logger.record(LogEntry(url="https://api.openai.com", model="gpt-4o", provider="openai"))
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as fh:
            path = fh.name
        try:
            logger.export_jsonl(path)
            with open(path) as fh:
                lines = [l.strip() for l in fh if l.strip()]
            assert len(lines) == 1
            assert json.loads(lines[0])["model"] == "gpt-4o"
        finally:
            os.unlink(path)

    def test_export_csv(self):
        logger = self._logger()
        logger.record(LogEntry(url="https://api.openai.com", model="gpt-4o", provider="openai"))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
            path = fh.name
        try:
            logger.export_csv(path)
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == 1
            assert rows[0]["model"] == "gpt-4o"
        finally:
            os.unlink(path)

    def test_export_csv_empty_produces_no_file_content(self):
        logger = self._logger()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
            path = fh.name
        try:
            logger.export_csv(path)      # no entries → no-op
            assert os.path.getsize(path) == 0
        finally:
            os.unlink(path)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            LLMLogger(backend="redis")


# ===========================================================================
# LLMLogger – JSONL backend
# ===========================================================================

class TestLLMLoggerJSONL:
    def test_record_and_query(self):
        logger = LLMLogger(backend="jsonl")
        logger.record(LogEntry(url="https://api.openai.com", model="gpt-4o", provider="openai"))
        assert logger.count() == 1
        assert logger.query()[0].model == "gpt-4o"

    def test_filter_by_provider(self):
        logger = LLMLogger(backend="jsonl")
        logger.record(LogEntry(url="https://api.openai.com",    model="gpt-4o",       provider="openai"))
        logger.record(LogEntry(url="https://api.anthropic.com", model="claude-3-haiku", provider="anthropic"))
        assert len(logger.query(provider="openai")) == 1


# ===========================================================================
# _load_jsonl_into
# ===========================================================================

class TestLoadJsonlInto:
    def test_loads_valid_entries(self):
        entry = LogEntry(url="https://api.openai.com", model="gpt-4o", provider="openai")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as fh:
            fh.write(json.dumps(entry.to_dict()) + "\n")
            path = fh.name
        try:
            logger = LLMLogger(db_path=":memory:", backend="sqlite")
            _load_jsonl_into(logger, path)
            assert logger.count() == 1
            assert logger.query()[0].model == "gpt-4o"
        finally:
            os.unlink(path)

    def test_skips_invalid_lines(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as fh:
            fh.write("not-json\n")
            fh.write("\n")
            path = fh.name
        try:
            logger = LLMLogger(db_path=":memory:", backend="sqlite")
            _load_jsonl_into(logger, path)
            assert logger.count() == 0
        finally:
            os.unlink(path)


# ===========================================================================
# session context manager
# ===========================================================================

class TestSession:
    def test_yields_logger(self):
        with session(backend="sqlite", auto_patch=False) as log:
            assert isinstance(log, LLMLogger)

    def test_jsonl_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            path = fh.name
        try:
            with session(log_file=path, backend="jsonl", auto_patch=False) as log:
                log.record(LogEntry(url="https://api.openai.com", model="gpt-4o", provider="openai"))
            with open(path) as fh:
                lines = [l.strip() for l in fh if l.strip()]
            assert len(lines) == 1
            assert json.loads(lines[0])["model"] == "gpt-4o"
        finally:
            os.unlink(path)

    def test_jsonl_appends_on_second_session(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            path = fh.name
        try:
            with session(log_file=path, backend="jsonl", auto_patch=False) as log:
                log.record(LogEntry(url="https://api.openai.com"))
            with session(log_file=path, backend="jsonl", auto_patch=False) as log:
                log.record(LogEntry(url="https://api.anthropic.com"))
            with open(path) as fh:
                lines = [l for l in fh if l.strip()]
            assert len(lines) == 2
        finally:
            os.unlink(path)


# ===========================================================================
# Backwards-compatibility aliases
# ===========================================================================

class TestBackwardsCompat:
    def test_log_record_alias(self):
        assert lal.LogRecord is lal.LogEntry

    def test_backend_aliases(self):
        assert lal.JSONLBackend  is lal.LLMLogger
        assert lal.SQLiteBackend is lal.LLMLogger
        assert lal.StdoutBackend is lal.LLMLogger

    def test_detect_provider_callable(self):
        assert callable(lal._detect_provider)
        assert lal._detect_provider("https://api.openai.com") == "openai"

    def test_extract_model_callable(self):
        assert callable(lal._extract_model)

    def test_cli_alias_callable(self):
        assert callable(lal._cli)
