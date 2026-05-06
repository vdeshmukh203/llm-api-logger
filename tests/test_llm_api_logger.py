"""
Tests for llm_api_logger – targeting JOSS-level coverage.

Run with::

    pytest tests/ -v
"""

import json
import pathlib
import sys
import tempfile
import time
import uuid

import pytest

# Ensure the root module is importable from the test directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import llm_api_logger as lal

# ---------------------------------------------------------------------------
# Backwards-compatibility aliases (regression guard)
# ---------------------------------------------------------------------------

def test_import():
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
# _extract_provider
# ---------------------------------------------------------------------------

class TestExtractProvider:
    def test_openai(self):
        assert lal._extract_provider("https://api.openai.com/v1/chat/completions") == "openai"

    def test_anthropic(self):
        assert lal._extract_provider("https://api.anthropic.com/v1/messages") == "anthropic"

    def test_google(self):
        assert lal._extract_provider("https://generativelanguage.googleapis.com/v1") == "google"

    def test_gemini(self):
        assert lal._extract_provider("https://gemini.google.com/api") == "google"

    def test_mistral(self):
        assert lal._extract_provider("https://api.mistral.ai/v1/chat") == "mistral"

    def test_cohere(self):
        assert lal._extract_provider("https://api.cohere.ai/v1/generate") == "cohere"

    def test_together(self):
        assert lal._extract_provider("https://api.together.xyz/v1") == "together"

    def test_huggingface(self):
        assert lal._extract_provider("https://api-inference.huggingface.co/models/meta-llama") == "huggingface"

    def test_unknown(self):
        assert lal._extract_provider("https://example.com/api") == "unknown"

    def test_case_insensitive(self):
        assert lal._extract_provider("https://API.OPENAI.COM/v1") == "openai"


# ---------------------------------------------------------------------------
# _extract_model
# ---------------------------------------------------------------------------

class TestExtractModel:
    def test_model_field_in_request(self):
        body = json.dumps({"model": "gpt-4o", "messages": []})
        assert lal._extract_model(body, None) == "gpt-4o"

    def test_modelId_field(self):
        body = json.dumps({"modelId": "claude-3-haiku"})
        assert lal._extract_model(body, None) == "claude-3-haiku"

    def test_engine_field(self):
        body = json.dumps({"engine": "text-davinci-003"})
        assert lal._extract_model(body, None) == "text-davinci-003"

    def test_fallback_to_response(self):
        resp = json.dumps({"model": "gpt-4o-mini", "choices": []})
        assert lal._extract_model(None, resp) == "gpt-4o-mini"

    def test_request_takes_precedence_over_response(self):
        req = json.dumps({"model": "gpt-4"})
        resp = json.dumps({"model": "gpt-4-turbo"})
        assert lal._extract_model(req, resp) == "gpt-4"

    def test_unknown_when_no_model(self):
        assert lal._extract_model(None, None) == "unknown"

    def test_invalid_json(self):
        assert lal._extract_model("not-json", None) == "unknown"


# ---------------------------------------------------------------------------
# _extract_tokens
# ---------------------------------------------------------------------------

class TestExtractTokens:
    def test_openai_usage(self):
        body = json.dumps({"usage": {"prompt_tokens": 50, "completion_tokens": 30}})
        assert lal._extract_tokens(body) == (50, 30)

    def test_google_usage(self):
        body = json.dumps({"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 40}})
        assert lal._extract_tokens(body) == (100, 40)

    def test_empty_body(self):
        assert lal._extract_tokens(None) == (0, 0)

    def test_invalid_json(self):
        assert lal._extract_tokens("{bad json") == (0, 0)

    def test_no_usage_field(self):
        body = json.dumps({"choices": []})
        assert lal._extract_tokens(body) == (0, 0)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_gpt4o(self):
        cost = lal.estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert abs(cost - 20.0) < 1e-9

    def test_zero_tokens(self):
        assert lal.estimate_cost("gpt-4o-mini", 0, 0) == 0.0

    def test_known_models_present(self):
        for model in ("gpt-4o", "claude-3-opus", "gemini-1.5-pro", "mistral-large"):
            assert model in lal.COST_TABLE

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found in cost table"):
            lal.estimate_cost("totally-unknown-model", 100, 100)

    def test_fractional_tokens(self):
        # 500K input tokens of gpt-4o = 500_000/1_000_000 * 5.00 = $2.50
        cost = lal.estimate_cost("gpt-4o", 500_000, 0)
        assert abs(cost - 2.50) < 1e-9


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

class TestLogEntry:
    def test_auto_id(self):
        e = lal.LogEntry()
        assert uuid.UUID(e.id)  # valid UUID

    def test_provider_auto_detected(self):
        e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions")
        assert e.provider == "openai"

    def test_model_auto_extracted(self):
        req = json.dumps({"model": "gpt-4o", "messages": []})
        e = lal.LogEntry(url="https://api.openai.com/v1/chat", request_body=req)
        assert e.model == "gpt-4o"

    def test_tokens_auto_extracted(self):
        resp = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        e = lal.LogEntry(response_body=resp)
        assert e.tokens_in == 10
        assert e.tokens_out == 5

    def test_cost_auto_computed(self):
        req = json.dumps({"model": "gpt-4o"})
        resp = json.dumps({"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}})
        e = lal.LogEntry(request_body=req, response_body=resp)
        # 1M input @ $5/M + 1M output @ $15/M = $20
        assert abs(e.cost_usd - 20.0) < 1e-9

    def test_cost_zero_for_unknown_model(self):
        resp = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        e = lal.LogEntry(model="some-future-model", response_body=resp)
        assert e.cost_usd == 0.0

    def test_to_dict_roundtrip(self):
        e = lal.LogEntry(url="https://api.openai.com", model="gpt-4o",
                         tokens_in=10, tokens_out=5)
        d = e.to_dict()
        e2 = lal.LogEntry.from_dict(d)
        assert e.id == e2.id
        assert e.model == e2.model

    def test_from_dict_ignores_extra_keys(self):
        d = lal.LogEntry().to_dict()
        d["extra_future_field"] = "should-be-ignored"
        e = lal.LogEntry.from_dict(d)  # must not raise
        assert not hasattr(e, "extra_future_field")

    def test_error_field(self):
        e = lal.LogEntry(error="connection timeout")
        assert e.error == "connection timeout"


# ---------------------------------------------------------------------------
# LLMLogger – in-memory SQLite
# ---------------------------------------------------------------------------

class TestLLMLoggerSQLite:
    def setup_method(self):
        self.logger = lal.LLMLogger(db_path=":memory:", backend="sqlite")

    def teardown_method(self):
        self.logger.close()

    def _entry(self, **kw) -> lal.LogEntry:
        return lal.LogEntry(url="https://api.openai.com/v1/chat", **kw)

    def test_record_and_count(self):
        self.logger.record(self._entry())
        assert self.logger.count() == 1

    def test_query_all(self):
        for _ in range(3):
            self.logger.record(self._entry())
        assert len(self.logger.query()) == 3

    def test_query_filter_model(self):
        self.logger.record(self._entry(model="gpt-4o"))
        self.logger.record(self._entry(model="gpt-4o-mini"))
        result = self.logger.query(model="gpt-4o")
        assert len(result) == 1
        assert result[0].model == "gpt-4o"

    def test_query_filter_provider(self):
        self.logger.record(self._entry(provider="openai"))
        self.logger.record(lal.LogEntry(url="https://api.anthropic.com/v1/messages",
                                         provider="anthropic"))
        result = self.logger.query(provider="anthropic")
        assert len(result) == 1

    def test_query_filter_status(self):
        self.logger.record(self._entry(status_code=200))
        self.logger.record(self._entry(status_code=400))
        assert len(self.logger.query(status_code=400)) == 1

    def test_query_ordered_newest_first(self):
        e1 = self._entry()
        e1.timestamp = "2024-01-01T00:00:00"
        e2 = self._entry()
        e2.timestamp = "2024-06-01T00:00:00"
        self.logger.record(e1)
        self.logger.record(e2)
        results = self.logger.query()
        assert results[0].timestamp > results[1].timestamp

    def test_summary_empty(self):
        s = self.logger.summary()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0

    def test_summary_aggregates(self):
        self.logger.record(lal.LogEntry(
            url="https://api.openai.com", model="gpt-4o",
            tokens_in=100, tokens_out=50, cost_usd=0.01, latency_ms=200.0
        ))
        self.logger.record(lal.LogEntry(
            url="https://api.openai.com", model="gpt-4o",
            tokens_in=200, tokens_out=100, cost_usd=0.02, latency_ms=400.0
        ))
        s = self.logger.summary()
        assert s["total_calls"] == 2
        assert abs(s["total_cost_usd"] - 0.03) < 1e-9
        assert s["total_tokens_in"] == 300
        assert s["total_tokens_out"] == 150
        assert abs(s["avg_latency_ms"] - 300.0) < 1e-9
        assert s["calls_by_model"]["gpt-4o"] == 2


# ---------------------------------------------------------------------------
# LLMLogger – JSONL backend
# ---------------------------------------------------------------------------

class TestLLMLoggerJSONL:
    def test_record_and_count(self):
        logger = lal.LLMLogger(backend="jsonl")
        logger.record(lal.LogEntry(url="https://api.openai.com"))
        assert logger.count() == 1

    def test_export_import_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name

        try:
            logger = lal.LLMLogger(backend="jsonl")
            for i in range(5):
                logger.record(lal.LogEntry(
                    url="https://api.openai.com", model="gpt-4o",
                    tokens_in=i * 10, tokens_out=i * 5,
                ))
            logger.export_jsonl(path)

            logger2 = lal.LLMLogger(backend="jsonl")
            for line in pathlib.Path(path).read_text().splitlines():
                if line.strip():
                    logger2.entries.append(lal.LogEntry.from_dict(json.loads(line)))

            assert logger2.count() == 5
        finally:
            pathlib.Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_csv_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            logger = lal.LLMLogger(backend="jsonl")
            logger.record(lal.LogEntry(
                url="https://api.openai.com", model="gpt-4o",
                tokens_in=10, tokens_out=5, cost_usd=0.005
            ))
            logger.export_csv(path)
            import csv
            rows = list(csv.DictReader(pathlib.Path(path).open()))
            assert len(rows) == 1
            assert rows[0]["model"] == "gpt-4o"
            assert "cost_usd" in rows[0]
        finally:
            pathlib.Path(path).unlink(missing_ok=True)

    def test_csv_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            logger = lal.LLMLogger(backend="jsonl")
            logger.export_csv(path)  # no entries – should not raise
        finally:
            pathlib.Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# session context manager
# ---------------------------------------------------------------------------

class TestSession:
    def test_session_yields_logger(self):
        with lal.session(backend="sqlite") as lgr:
            assert isinstance(lgr, lal.LLMLogger)

    def test_session_patches_and_unpatches(self):
        original = lal._original_urlopen
        with lal.session(backend="sqlite"):
            assert lal.urllib_request.urlopen is not original
        assert lal.urllib_request.urlopen is original

    def test_session_jsonl_writes_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            with lal.session(log_file=path, backend="jsonl", auto_patch=False) as lgr:
                lgr.record(lal.LogEntry(url="https://api.openai.com", model="gpt-4o"))
            lines = [l for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["model"] == "gpt-4o"
        finally:
            pathlib.Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# LLMAPIProxy (from src package)
# ---------------------------------------------------------------------------

def test_proxy_context_manager():
    # The src package is importable as `src.llm_api_logger` from the project root
    from src.llm_api_logger.proxy import LLMAPIProxy
    original = lal._original_urlopen
    with LLMAPIProxy() as proxy:
        assert lal.urllib_request.urlopen is not original
        assert isinstance(proxy.logger, lal.LLMLogger)
    assert lal.urllib_request.urlopen is original


def test_log_store_alias():
    from src.llm_api_logger.store import LogStore
    assert LogStore is lal.LLMLogger


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_summary(capsys):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        logger = lal.LLMLogger(backend="jsonl")
        logger.record(lal.LogEntry(
            url="https://api.openai.com", model="gpt-4o",
            tokens_in=500, tokens_out=200, cost_usd=0.005, latency_ms=320.0
        ))
        logger.export_jsonl(path)

        sys.argv = ["llm-api-logger", "summary", path]
        lal.main()
        out = capsys.readouterr().out
        assert "gpt-4o" in out
        assert "Total API Calls" in out
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def test_cli_query(capsys):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        logger = lal.LLMLogger(backend="jsonl")
        logger.record(lal.LogEntry(url="https://api.openai.com", model="gpt-4o"))
        logger.export_jsonl(path)

        sys.argv = ["llm-api-logger", "query", path, "--model", "gpt-4o"]
        lal.main()
        out = capsys.readouterr().out
        assert "1 entr" in out
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def test_cli_export_csv(capsys):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as fin, \
         tempfile.NamedTemporaryFile(suffix=".csv",  delete=False, mode="w") as fout:
        in_path  = fin.name
        out_path = fout.name
    try:
        logger = lal.LLMLogger(backend="jsonl")
        logger.record(lal.LogEntry(url="https://api.openai.com", model="gpt-4o"))
        logger.export_jsonl(in_path)

        sys.argv = ["llm-api-logger", "export", in_path, "--output", out_path, "--format", "csv"]
        lal.main()
        out = capsys.readouterr().out
        assert "Exported" in out
        assert pathlib.Path(out_path).stat().st_size > 0
    finally:
        pathlib.Path(in_path).unlink(missing_ok=True)
        pathlib.Path(out_path).unlink(missing_ok=True)
