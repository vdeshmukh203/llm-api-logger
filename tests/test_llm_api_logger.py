"""
Tests for llm_api_logger.

Covers: cost estimation, provider/model extraction, token parsing,
SHA-256 hashing, LogEntry construction and serialisation, LLMLogger
backends (SQLite and JSONL), query filtering, summary statistics,
CSV/JSONL export, and the urllib patching mechanism.
"""

import json
import pathlib
import sys
import tempfile

import pytest

# Ensure the repo root is on the path when tests are run directly.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import llm_api_logger as lal


# ---------------------------------------------------------------------------
# Module-level smoke test
# ---------------------------------------------------------------------------

def test_module_version():
    assert lal.__version__ == "0.1.0"


def test_backwards_compat_aliases():
    assert lal.LogRecord is lal.LogEntry
    assert lal.JSONLBackend is lal.LLMLogger
    assert lal.SQLiteBackend is lal.LLMLogger
    assert lal.StdoutBackend is lal.LLMLogger
    assert callable(lal._detect_provider)
    assert callable(lal._extract_model)
    assert callable(lal._tok)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model_gpt4o(self):
        # 1 M input tokens at $5 + 0.5 M output tokens at $15
        cost = lal.estimate_cost("gpt-4o", 1_000_000, 500_000)
        assert abs(cost - (5.00 + 7.50)) < 1e-9

    def test_known_model_zero_tokens(self):
        assert lal.estimate_cost("gpt-4o-mini", 0, 0) == 0.0

    def test_known_model_claude(self):
        cost = lal.estimate_cost("claude-3-haiku", 100_000, 50_000)
        expected = (100_000 / 1_000_000) * 0.25 + (50_000 / 1_000_000) * 1.25
        assert abs(cost - expected) < 1e-12

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found in COST_TABLE"):
            lal.estimate_cost("no-such-model-xyz", 1000, 500)

    def test_cost_table_keys_are_strings(self):
        for key in lal.COST_TABLE:
            assert isinstance(key, str)

    def test_cost_table_has_input_output(self):
        for model, pricing in lal.COST_TABLE.items():
            assert "input" in pricing, f"Missing 'input' for {model}"
            assert "output" in pricing, f"Missing 'output' for {model}"


# ---------------------------------------------------------------------------
# _extract_provider
# ---------------------------------------------------------------------------

class TestExtractProvider:
    def test_openai(self):
        assert lal._extract_provider("https://api.openai.com/v1/chat/completions") == "openai"

    def test_anthropic(self):
        assert lal._extract_provider("https://api.anthropic.com/v1/messages") == "anthropic"

    def test_google_by_domain(self):
        assert lal._extract_provider("https://generativelanguage.googleapis.com/v1") == "google"

    def test_google_by_gemini(self):
        assert lal._extract_provider("https://gemini.example.com/chat") == "google"

    def test_mistral(self):
        assert lal._extract_provider("https://api.mistral.ai/v1/chat") == "mistral"

    def test_together(self):
        assert lal._extract_provider("https://api.together.xyz/inference") == "together"

    def test_cohere(self):
        assert lal._extract_provider("https://api.cohere.ai/generate") == "cohere"

    def test_huggingface(self):
        assert lal._extract_provider("https://api-inference.huggingface.co/models/x") == "huggingface"

    def test_unknown(self):
        assert lal._extract_provider("https://example.com/api") == "unknown"

    def test_case_insensitive(self):
        assert lal._extract_provider("https://API.OPENAI.COM/v1") == "openai"


# ---------------------------------------------------------------------------
# _extract_model
# ---------------------------------------------------------------------------

class TestExtractModel:
    def _body(self, **kwargs) -> str:
        return json.dumps(kwargs)

    def test_model_key_in_request(self):
        body = self._body(model="gpt-4o", messages=[])
        assert lal._extract_model(body, None) == "gpt-4o"

    def test_engine_key(self):
        body = self._body(engine="davinci")
        assert lal._extract_model(body, None) == "davinci"

    def test_modelId_key(self):
        body = self._body(modelId="claude-3-opus-20240229")
        assert lal._extract_model(body, None) == "claude-3-opus-20240229"

    def test_falls_back_to_response(self):
        req = json.dumps({"messages": []})   # no model key
        resp = json.dumps({"model": "gpt-4o-mini", "choices": []})
        assert lal._extract_model(req, resp) == "gpt-4o-mini"

    def test_both_none(self):
        assert lal._extract_model(None, None) == "unknown"

    def test_invalid_json(self):
        assert lal._extract_model("not-json", "also-not-json") == "unknown"

    def test_empty_strings(self):
        assert lal._extract_model("", "") == "unknown"


# ---------------------------------------------------------------------------
# _extract_tokens
# ---------------------------------------------------------------------------

class TestExtractTokens:
    def test_openai_format(self):
        body = json.dumps({"usage": {"prompt_tokens": 42, "completion_tokens": 7}})
        assert lal._extract_tokens(body) == (42, 7)

    def test_google_format(self):
        body = json.dumps({"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 30}})
        assert lal._extract_tokens(body) == (100, 30)

    def test_none_returns_zeros(self):
        assert lal._extract_tokens(None) == (0, 0)

    def test_empty_string_returns_zeros(self):
        assert lal._extract_tokens("") == (0, 0)

    def test_invalid_json_returns_zeros(self):
        assert lal._extract_tokens("not-json") == (0, 0)

    def test_missing_usage_key(self):
        body = json.dumps({"choices": []})
        assert lal._extract_tokens(body) == (0, 0)

    def test_partial_usage(self):
        body = json.dumps({"usage": {"prompt_tokens": 10}})
        ti, to = lal._extract_tokens(body)
        assert ti == 10
        assert to == 0


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------

def test_sha256_deterministic():
    h1 = lal._sha256("hello world")
    h2 = lal._sha256("hello world")
    assert h1 == h2
    assert len(h1) == 64  # 256 bits → 64 hex chars


def test_sha256_different_inputs():
    assert lal._sha256("a") != lal._sha256("b")


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

class TestLogEntry:
    def _openai_response(self, model="gpt-4o-mini", tin=50, tout=20):
        return json.dumps({
            "model": model,
            "usage": {"prompt_tokens": tin, "completion_tokens": tout},
            "choices": [],
        })

    def test_auto_provider(self):
        e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions")
        assert e.provider == "openai"

    def test_auto_model(self):
        req = json.dumps({"model": "gpt-4o", "messages": []})
        e = lal.LogEntry(url="https://api.openai.com/", request_body=req)
        assert e.model == "gpt-4o"

    def test_auto_tokens(self):
        e = lal.LogEntry(
            url="https://api.openai.com/",
            response_body=self._openai_response(tin=100, tout=40),
        )
        assert e.tokens_in == 100
        assert e.tokens_out == 40

    def test_auto_cost(self):
        resp = self._openai_response(model="gpt-4o-mini", tin=1_000_000, tout=1_000_000)
        e = lal.LogEntry(
            url="https://api.openai.com/",
            request_body=json.dumps({"model": "gpt-4o-mini"}),
            response_body=resp,
        )
        expected = lal.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert abs(e.cost_usd - expected) < 1e-12

    def test_content_hash_set(self):
        e = lal.LogEntry(url="https://api.openai.com/", request_body='{"a":1}', response_body='{"b":2}')
        assert e.content_hash is not None
        assert len(e.content_hash) == 64

    def test_content_hash_reproducible(self):
        kw = dict(url="u", request_body="req", response_body="resp")
        e1 = lal.LogEntry(**kw)
        e2 = lal.LogEntry(**kw)
        assert e1.content_hash == e2.content_hash

    def test_content_hash_none_when_no_bodies(self):
        e = lal.LogEntry(url="https://api.openai.com/")
        assert e.content_hash is None

    def test_to_dict_round_trip(self):
        e = lal.LogEntry(
            url="https://api.openai.com/",
            request_body='{"model":"gpt-4o"}',
            response_body=self._openai_response(),
        )
        d = e.to_dict()
        e2 = lal.LogEntry.from_dict(d)
        assert e2.id == e.id
        assert e2.content_hash == e.content_hash
        assert e2.tokens_in == e.tokens_in

    def test_from_dict_ignores_unknown_keys(self):
        d = lal.LogEntry(url="x").to_dict()
        d["_extra_field"] = "should be ignored"
        # Should not raise
        e = lal.LogEntry.from_dict(d)
        assert e.url == "x"

    def test_explicit_values_not_overwritten(self):
        e = lal.LogEntry(
            url="https://api.openai.com/",
            provider="my-provider",
            model="my-model",
            tokens_in=999,
            tokens_out=111,
        )
        assert e.provider == "my-provider"
        assert e.model == "my-model"
        assert e.tokens_in == 999

    def test_unknown_model_does_not_raise(self):
        resp = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        e = lal.LogEntry(
            url="https://api.openai.com/",
            request_body=json.dumps({"model": "no-such-model"}),
            response_body=resp,
        )
        assert e.cost_usd == 0.0  # graceful fallback


# ---------------------------------------------------------------------------
# LLMLogger — SQLite backend
# ---------------------------------------------------------------------------

class TestLLMLoggerSQLite:
    def _logger(self):
        return lal.LLMLogger(db_path=":memory:", backend="sqlite")

    def _entry(self, model="gpt-4o-mini", cost=0.01):
        return lal.LogEntry(
            url="https://api.openai.com/v1/chat",
            request_body=json.dumps({"model": model}),
            response_body=json.dumps({
                "model": model,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }),
            status_code=200,
            latency_ms=123.4,
        )

    def test_initial_count_zero(self):
        assert self._logger().count() == 0

    def test_record_and_count(self):
        log = self._logger()
        log.record(self._entry())
        log.record(self._entry())
        assert log.count() == 2

    def test_query_all(self):
        log = self._logger()
        log.record(self._entry(model="gpt-4o"))
        log.record(self._entry(model="gpt-4o-mini"))
        assert len(log.query()) == 2

    def test_query_filter_model(self):
        log = self._logger()
        log.record(self._entry(model="gpt-4o"))
        log.record(self._entry(model="gpt-4o-mini"))
        results = log.query(model="gpt-4o")
        assert len(results) == 1
        assert results[0].model == "gpt-4o"

    def test_query_filter_provider(self):
        log = self._logger()
        e = self._entry()
        log.record(e)
        results = log.query(provider="openai")
        assert len(results) == 1

    def test_query_filter_status_code(self):
        log = self._logger()
        e = lal.LogEntry(url="https://api.openai.com/", status_code=429)
        log.record(e)
        log.record(self._entry())
        assert len(log.query(status_code=429)) == 1
        assert len(log.query(status_code=200)) == 1

    def test_query_since_filter(self):
        log = self._logger()
        old = lal.LogEntry(url="https://api.openai.com/", timestamp="2023-01-01T00:00:00")
        new = lal.LogEntry(url="https://api.openai.com/", timestamp="2024-06-01T00:00:00")
        log.record(old)
        log.record(new)
        results = log.query(since="2024-01-01T00:00:00")
        assert len(results) == 1
        assert results[0].timestamp == "2024-06-01T00:00:00"

    def test_query_until_filter(self):
        log = self._logger()
        old = lal.LogEntry(url="https://api.openai.com/", timestamp="2023-01-01T00:00:00")
        new = lal.LogEntry(url="https://api.openai.com/", timestamp="2024-06-01T00:00:00")
        log.record(old)
        log.record(new)
        results = log.query(until="2023-12-31T23:59:59")
        assert len(results) == 1
        assert results[0].timestamp == "2023-01-01T00:00:00"

    def test_summary_empty(self):
        s = self._logger().summary()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0

    def test_summary_aggregates(self):
        log = self._logger()
        log.record(lal.LogEntry(
            url="https://api.openai.com/",
            request_body=json.dumps({"model": "gpt-4o-mini"}),
            response_body=json.dumps({"model": "gpt-4o-mini",
                                      "usage": {"prompt_tokens": 200, "completion_tokens": 100}}),
            latency_ms=50.0,
        ))
        log.record(lal.LogEntry(
            url="https://api.openai.com/",
            request_body=json.dumps({"model": "gpt-4o-mini"}),
            response_body=json.dumps({"model": "gpt-4o-mini",
                                      "usage": {"prompt_tokens": 800, "completion_tokens": 400}}),
            latency_ms=150.0,
        ))
        s = log.summary()
        assert s["total_calls"] == 2
        assert s["total_tokens_in"] == 1000
        assert s["total_tokens_out"] == 500
        assert abs(s["avg_latency_ms"] - 100.0) < 1e-9
        assert "gpt-4o-mini" in s["calls_by_model"]

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            lal.LLMLogger(backend="redis")

    def test_export_csv(self, tmp_path):
        log = self._logger()
        log.record(self._entry())
        out = str(tmp_path / "out.csv")
        log.export_csv(out)
        content = pathlib.Path(out).read_text()
        assert "id" in content
        assert "model" in content
        assert "content_hash" in content

    def test_export_csv_empty_does_not_create_file(self, tmp_path):
        log = self._logger()
        out = str(tmp_path / "empty.csv")
        log.export_csv(out)
        assert not pathlib.Path(out).exists()

    def test_export_jsonl(self, tmp_path):
        log = self._logger()
        log.record(self._entry(model="gpt-4o"))
        out = str(tmp_path / "out.jsonl")
        log.export_jsonl(out)
        lines = pathlib.Path(out).read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["model"] == "gpt-4o"
        assert "content_hash" in record


# ---------------------------------------------------------------------------
# LLMLogger — JSONL (in-memory) backend
# ---------------------------------------------------------------------------

class TestLLMLoggerJSONL:
    def _logger(self):
        return lal.LLMLogger(db_path=":memory:", backend="jsonl")

    def test_record_and_count(self):
        log = self._logger()
        log.record(lal.LogEntry(url="https://api.anthropic.com/"))
        assert log.count() == 1

    def test_query_returns_entries(self):
        log = self._logger()
        log.record(lal.LogEntry(url="https://api.anthropic.com/", model="claude-3-haiku"))
        results = log.query(model="claude-3-haiku")
        assert len(results) == 1

    def test_query_filter_status_code(self):
        log = self._logger()
        log.record(lal.LogEntry(url="https://api.anthropic.com/", status_code=500))
        log.record(lal.LogEntry(url="https://api.anthropic.com/", status_code=200))
        assert len(log.query(status_code=500)) == 1


# ---------------------------------------------------------------------------
# session() context manager
# ---------------------------------------------------------------------------

class TestSession:
    def test_yields_logger(self):
        with lal.session(backend="jsonl") as log:
            assert isinstance(log, lal.LLMLogger)

    def test_auto_patch_restores_original(self):
        import urllib.request as ur
        original = ur.urlopen
        with lal.session(backend="jsonl"):
            pass
        assert ur.urlopen is original

    def test_no_patch_option(self):
        import urllib.request as ur
        original = ur.urlopen
        with lal.session(backend="jsonl", auto_patch=False):
            assert ur.urlopen is original

    def test_jsonl_file_written_on_exit(self, tmp_path):
        log_file = str(tmp_path / "test.jsonl")
        with lal.session(log_file=log_file, backend="jsonl") as log:
            log.record(lal.LogEntry(url="https://api.openai.com/"))
        assert pathlib.Path(log_file).exists()
        lines = pathlib.Path(log_file).read_text().splitlines()
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# urllib patching
# ---------------------------------------------------------------------------

class TestUrllibPatching:
    def setup_method(self):
        lal.unpatch_urllib()   # ensure clean state

    def teardown_method(self):
        lal.unpatch_urllib()

    def test_patch_and_unpatch(self):
        import urllib.request as ur
        orig = ur.urlopen
        lal.patch_urllib()
        assert ur.urlopen is not orig
        lal.unpatch_urllib()
        assert ur.urlopen is orig

    def test_patch_with_logger(self):
        log = lal.LLMLogger(db_path=":memory:", backend="sqlite")
        lal.patch_urllib(log)
        assert lal._active_logger is log
        lal.unpatch_urllib()
        assert lal._active_logger is None


# ---------------------------------------------------------------------------
# _is_llm_request (internal)
# ---------------------------------------------------------------------------

class TestIsLLMRequest:
    def test_openai_url(self):
        assert lal._is_llm_request("https://api.openai.com/v1/chat", None)

    def test_unknown_url_with_model_key(self):
        body = json.dumps({"model": "my-model", "messages": []})
        assert lal._is_llm_request("https://custom.llm.example.com/", body)

    def test_random_url_no_body(self):
        assert not lal._is_llm_request("https://www.example.com/", None)

    def test_bad_json_body_does_not_raise(self):
        assert not lal._is_llm_request("https://www.example.com/", "not-json")
