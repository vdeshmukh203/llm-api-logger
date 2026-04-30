"""Tests for llm_api_logger — top-level module (JOSS level)."""

import json
import os
import pathlib
import tempfile

import pytest
import llm_api_logger as lal


# ---------------------------------------------------------------------------
# Backwards-compatibility shims (still required by original test contract)
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
# estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_known_model():
    cost = lal.estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    # 5.00 (in) + 15.00 (out) = 20.00
    assert abs(cost - 20.0) < 1e-6


def test_estimate_cost_fractional_tokens():
    cost = lal.estimate_cost("gpt-4o-mini", 500, 500)
    assert cost >= 0.0


def test_estimate_cost_unknown_model():
    with pytest.raises(ValueError, match="not found in cost table"):
        lal.estimate_cost("nonexistent-model-xyz", 100, 100)


def test_estimate_cost_zero_tokens():
    cost = lal.estimate_cost("claude-3-haiku", 0, 0)
    assert cost == 0.0


# ---------------------------------------------------------------------------
# _extract_provider
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://api.openai.com/v1/chat/completions", "openai"),
    ("https://api.anthropic.com/v1/messages", "anthropic"),
    ("https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", "google"),
    ("https://api.mistral.ai/v1/chat/completions", "mistral"),
    ("https://api.cohere.ai/v1/generate", "cohere"),
    ("https://api.together.xyz/v1/chat/completions", "together"),
    ("https://api-inference.huggingface.co/models/gpt2", "huggingface"),
    ("https://example.com/api", "unknown"),
])
def test_extract_provider(url, expected):
    assert lal._extract_provider(url) == expected


# ---------------------------------------------------------------------------
# _extract_model
# ---------------------------------------------------------------------------

def test_extract_model_from_request():
    req = json.dumps({"model": "gpt-4o", "messages": []})
    assert lal._extract_model(req, None) == "gpt-4o"


def test_extract_model_from_response():
    resp = json.dumps({"model": "claude-3-opus-20240229", "content": []})
    assert lal._extract_model(None, resp) == "claude-3-opus-20240229"


def test_extract_model_prefers_request():
    req = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({"model": "gpt-3.5-turbo"})
    assert lal._extract_model(req, resp) == "gpt-4o"


def test_extract_model_engine_key():
    req = json.dumps({"engine": "davinci"})
    assert lal._extract_model(req, None) == "davinci"


def test_extract_model_missing():
    assert lal._extract_model(None, None) == "unknown"
    assert lal._extract_model("{}", "{}") == "unknown"


def test_extract_model_bad_json():
    assert lal._extract_model("not json", None) == "unknown"


# ---------------------------------------------------------------------------
# _tok (token extraction)
# ---------------------------------------------------------------------------

def test_tok_openai_usage():
    body = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    ti, to = lal._tok(body)
    assert ti == 10
    assert to == 20


def test_tok_google_usage_metadata():
    body = json.dumps({"usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 25}})
    ti, to = lal._tok(body)
    assert ti == 15
    assert to == 25


def test_tok_empty():
    assert lal._tok(None) == (0, 0)
    assert lal._tok("") == (0, 0)
    assert lal._tok("{}") == (0, 0)


def test_tok_bad_json():
    assert lal._tok("not json") == (0, 0)


# ---------------------------------------------------------------------------
# LogEntry / LogRecord dataclass
# ---------------------------------------------------------------------------

def test_log_entry_defaults():
    e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions")
    assert e.provider == "openai"
    assert e.status_code == 200
    assert e.cost_usd == 0.0


def test_log_entry_auto_provider():
    e = lal.LogEntry(url="https://api.anthropic.com/v1/messages")
    assert e.provider == "anthropic"


def test_log_entry_auto_model_from_request():
    req = json.dumps({"model": "gpt-4o", "messages": []})
    e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions", request_body=req)
    assert e.model == "gpt-4o"


def test_log_entry_auto_tokens_and_cost():
    req = json.dumps({"model": "gpt-4o"})
    resp = json.dumps({
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    })
    e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions",
                     request_body=req, response_body=resp)
    assert e.tokens_in == 1_000_000
    assert e.tokens_out == 1_000_000
    assert abs(e.cost_usd - 20.0) < 1e-6


def test_log_entry_to_dict_roundtrip():
    e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions",
                     model="gpt-4o", tokens_in=10, tokens_out=5)
    d = e.to_dict()
    e2 = lal.LogEntry.from_dict(d)
    assert e2.id == e.id
    assert e2.model == "gpt-4o"
    assert e2.tokens_in == 10


def test_log_entry_unknown_model_no_cost():
    resp = json.dumps({
        "model": "my-private-model",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    })
    e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions",
                     response_body=resp)
    assert e.cost_usd == 0.0  # unknown model → no cost estimation


# ---------------------------------------------------------------------------
# LLMLogger — SQLite backend
# ---------------------------------------------------------------------------

def _make_entry(**kw) -> lal.LogEntry:
    defaults = dict(url="https://api.openai.com/v1/chat/completions",
                    model="gpt-4o", provider="openai",
                    tokens_in=100, tokens_out=50, latency_ms=200.0)
    defaults.update(kw)
    return lal.LogEntry(**defaults)


class TestLLMLoggerSQLite:
    def setup_method(self):
        self.logger = lal.LLMLogger(db_path=":memory:", backend="sqlite")

    def test_record_and_count(self):
        self.logger.record(_make_entry())
        assert self.logger.count() == 1

    def test_record_multiple(self):
        for _ in range(5):
            self.logger.record(_make_entry())
        assert self.logger.count() == 5

    def test_query_all(self):
        self.logger.record(_make_entry())
        entries = self.logger.query()
        assert len(entries) == 1
        assert entries[0].provider == "openai"

    def test_query_filter_model(self):
        self.logger.record(_make_entry(model="gpt-4o"))
        self.logger.record(_make_entry(model="claude-3-haiku", provider="anthropic",
                                       url="https://api.anthropic.com/v1/messages"))
        results = self.logger.query(model="gpt-4o")
        assert all(e.model == "gpt-4o" for e in results)
        assert len(results) == 1

    def test_query_filter_provider(self):
        self.logger.record(_make_entry(provider="openai"))
        self.logger.record(_make_entry(provider="anthropic",
                                       url="https://api.anthropic.com/v1/messages"))
        results = self.logger.query(provider="anthropic")
        assert all(e.provider == "anthropic" for e in results)

    def test_query_filter_status_code(self):
        self.logger.record(_make_entry(status_code=200))
        self.logger.record(_make_entry(status_code=429))
        results = self.logger.query(status_code=429)
        assert len(results) == 1 and results[0].status_code == 429

    def test_summary_empty(self):
        s = self.logger.summary()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0

    def test_summary_values(self):
        req = json.dumps({"model": "gpt-4o"})
        resp = json.dumps({
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        })
        e = lal.LogEntry(url="https://api.openai.com/v1/chat/completions",
                         request_body=req, response_body=resp, latency_ms=100.0)
        self.logger.record(e)
        s = self.logger.summary()
        assert s["total_calls"] == 1
        assert s["calls_by_model"]["gpt-4o"] == 1
        assert s["avg_latency_ms"] == pytest.approx(100.0)

    def test_export_csv(self):
        self.logger.record(_make_entry())
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self.logger.export_csv(path)
            content = pathlib.Path(path).read_text()
            assert "gpt-4o" in content
            assert "openai" in content
        finally:
            os.unlink(path)

    def test_export_jsonl(self):
        self.logger.record(_make_entry())
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            self.logger.export_jsonl(path)
            lines = [l for l in pathlib.Path(path).read_text().splitlines() if l]
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["model"] == "gpt-4o"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# LLMLogger — JSONL backend
# ---------------------------------------------------------------------------

class TestLLMLoggerJSONL:
    def setup_method(self):
        self.logger = lal.LLMLogger(db_path=":memory:", backend="jsonl")

    def test_record_and_count(self):
        self.logger.record(_make_entry())
        assert self.logger.count() == 1

    def test_query_filter(self):
        self.logger.record(_make_entry(model="gpt-4o"))
        self.logger.record(_make_entry(model="gpt-4o-mini"))
        results = self.logger.query(model="gpt-4o")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# LoggingSession context manager
# ---------------------------------------------------------------------------

def test_session_context_manager_creates_logger():
    with lal.session(backend="sqlite") as logger:
        assert isinstance(logger, lal.LLMLogger)


def test_session_yields_working_logger():
    with lal.session(backend="sqlite") as logger:
        logger.record(_make_entry())
        assert logger.count() == 1


# ---------------------------------------------------------------------------
# patch_urllib / unpatch_urllib
# ---------------------------------------------------------------------------

def test_patch_unpatch_urllib():
    from urllib import request as ur
    original = ur.urlopen
    lal.patch_urllib(lal.LLMLogger(backend="sqlite"))
    assert ur.urlopen is not original
    lal.unpatch_urllib()
    assert ur.urlopen is original


# ---------------------------------------------------------------------------
# _is_llm helper
# ---------------------------------------------------------------------------

def test_is_llm_openai_url():
    assert lal._is_llm("https://api.openai.com/v1/chat/completions", None) is True


def test_is_llm_unknown_url_with_model_body():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    assert lal._is_llm("https://example.com/api", body) is True


def test_is_llm_unknown_url_no_body():
    assert lal._is_llm("https://example.com/api", None) is False


# ---------------------------------------------------------------------------
# CLI entry-point alias
# ---------------------------------------------------------------------------

def test_cli_alias_exists():
    assert callable(lal._cli)
    assert lal._cli is lal.main
