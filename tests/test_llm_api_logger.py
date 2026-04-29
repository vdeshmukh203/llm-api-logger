import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import llm_api_logger as lal


# ---------------------------------------------------------------------------
# Import / attribute smoke tests (required by CI)
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
    cost = lal.estimate_cost("gpt-4o", 1_000, 500)
    assert cost > 0


def test_estimate_cost_proportional():
    c1 = lal.estimate_cost("gpt-4o", 1_000, 0)
    c2 = lal.estimate_cost("gpt-4o", 2_000, 0)
    assert abs(c2 - 2 * c1) < 1e-10


def test_estimate_cost_unknown_model():
    try:
        lal.estimate_cost("nonexistent-model-xyz", 100, 100)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# _extract_provider
# ---------------------------------------------------------------------------

def test_extract_provider_openai():
    assert lal._extract_provider("https://api.openai.com/v1/chat/completions") == "openai"


def test_extract_provider_anthropic():
    assert lal._extract_provider("https://api.anthropic.com/v1/messages") == "anthropic"


def test_extract_provider_google():
    assert lal._extract_provider("https://generativelanguage.googleapis.com/v1/models") == "google"


def test_extract_provider_unknown():
    assert lal._extract_provider("https://example.com/api") == "unknown"


# ---------------------------------------------------------------------------
# _extract_model
# ---------------------------------------------------------------------------

def test_extract_model_from_request():
    body = json.dumps({"model": "gpt-4o", "messages": []})
    assert lal._extract_model(body, None) == "gpt-4o"


def test_extract_model_from_response():
    resp = json.dumps({"model": "claude-3-opus", "content": []})
    assert lal._extract_model(None, resp) == "claude-3-opus"


def test_extract_model_fallback():
    assert lal._extract_model(None, None) == "unknown"


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

def test_log_entry_defaults():
    entry = lal.LogEntry(url="https://api.openai.com/v1/chat/completions")
    assert entry.provider == "openai"
    assert entry.id  # non-empty UUID
    assert entry.timestamp  # non-empty ISO string


def test_log_entry_token_extraction():
    resp = json.dumps({
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })
    entry = lal.LogEntry(url="https://api.openai.com/", response_body=resp)
    assert entry.tokens_in == 10
    assert entry.tokens_out == 5


def test_log_entry_cost_auto():
    resp = json.dumps({
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    })
    req = json.dumps({"model": "gpt-4o-mini"})
    entry = lal.LogEntry(url="https://api.openai.com/", request_body=req, response_body=resp)
    assert entry.cost_usd > 0


def test_log_entry_roundtrip():
    entry = lal.LogEntry(
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o",
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.0001,
    )
    restored = lal.LogEntry.from_dict(entry.to_dict())
    assert restored.id == entry.id
    assert restored.model == entry.model
    assert restored.cost_usd == entry.cost_usd


# ---------------------------------------------------------------------------
# LLMLogger – in-memory SQLite
# ---------------------------------------------------------------------------

def test_llmlogger_record_and_count_sqlite():
    logger = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    assert logger.count() == 0
    entry = lal.LogEntry(url="https://api.openai.com/v1/chat/completions", model="gpt-4o")
    logger.record(entry)
    assert logger.count() == 1


def test_llmlogger_query_filter_sqlite():
    logger = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    logger.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
    logger.record(lal.LogEntry(url="https://api.anthropic.com/", model="claude-3-opus"))
    results = logger.query(model="gpt-4o")
    assert len(results) == 1
    assert results[0].model == "gpt-4o"


def test_llmlogger_summary_sqlite():
    logger = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    logger.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o",
                               tokens_in=100, tokens_out=50, latency_ms=120.0))
    s = logger.summary()
    assert s["total_calls"] == 1
    assert s["total_tokens_in"] == 100
    assert s["avg_latency_ms"] == 120.0


# ---------------------------------------------------------------------------
# LLMLogger – JSONL in-memory
# ---------------------------------------------------------------------------

def test_llmlogger_record_and_count_jsonl():
    logger = lal.LLMLogger(db_path=":memory:", backend="jsonl")
    logger.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
    assert logger.count() == 1


def test_llmlogger_export_jsonl():
    logger = lal.LLMLogger(db_path=":memory:", backend="jsonl")
    logger.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o",
                               tokens_in=5, tokens_out=3))
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as fh:
        path = fh.name
    logger.export_jsonl(path)
    lines = [l for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["model"] == "gpt-4o"


def test_llmlogger_export_csv():
    logger = lal.LLMLogger(db_path=":memory:", backend="sqlite")
    logger.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as fh:
        path = fh.name
    logger.export_csv(path)
    content = pathlib.Path(path).read_text()
    assert "gpt-4o" in content
    assert "model" in content  # header row


# ---------------------------------------------------------------------------
# session() context manager
# ---------------------------------------------------------------------------

def test_session_context_manager():
    with lal.session(log_file=":memory:", backend="sqlite", auto_patch=False) as logger:
        logger.record(lal.LogEntry(url="https://api.openai.com/", model="gpt-4o"))
    assert logger.count() == 1


# ---------------------------------------------------------------------------
# LogStore (src package) — loaded via importlib to avoid name collision with
# the top-level llm_api_logger.py module that is already on sys.path.
# ---------------------------------------------------------------------------

def _load_logstore():
    import importlib.util
    store_path = pathlib.Path(__file__).parent.parent / "src" / "llm_api_logger" / "store.py"
    spec = importlib.util.spec_from_file_location("_llm_api_logger_store", store_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LogStore


def test_logstore_append_and_verify():
    LogStore = _load_logstore()
    store = LogStore(":memory:")
    digest = store.append({"model": "gpt-4o", "tokens_in": 10})
    assert len(digest) == 64  # SHA-256 hex
    records = store.load()
    assert len(records) == 1
    assert store.verify(records[0])


def test_logstore_file_roundtrip():
    LogStore = _load_logstore()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
        path = fh.name

    store = LogStore(path)
    store.append({"model": "claude-3-opus", "cost_usd": 0.002})

    store2 = LogStore(path)
    records = store2.load()
    assert len(records) == 1
    assert records[0]["model"] == "claude-3-opus"
    assert store2.verify(records[0])
