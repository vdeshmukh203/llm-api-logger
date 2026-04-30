"""
llm_api_logger: Transparent logging of LLM API traffic.

Two complementary interfaces:

* **Proxy** (``LLMAPIProxy``) — a local HTTP proxy that captures traffic at
  the network layer; route your SDK through it by setting ``HTTP_PROXY``.
* **Middleware** (``LLMLogger``, ``session``) — urllib monkey-patching for
  in-process capture without a separate proxy process.

Quick start (proxy)::

    from llm_api_logger import LLMAPIProxy, LogStore
    store = LogStore("llm_proxy.jsonl")
    with LLMAPIProxy(port=8080, store=store) as proxy:
        import os; os.environ.update(proxy.env)
        # … run your SDK calls …
    print(store.summary())

Quick start (middleware)::

    from llm_api_logger import session
    with session("llm_api.jsonl") as log:
        # … run your urllib-based SDK calls …
    print(log.summary())
"""

__version__ = "0.2.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# Proxy + store layer
from .proxy import LLMAPIProxy
from .store import LogRecord, LogStore

# Middleware layer
from .logger import (
    COST_TABLE,
    LogEntry,
    LLMLogger,
    estimate_cost,
    patch_urllib,
    unpatch_urllib,
    session,
    main,
    _extract_provider,
    _extract_model,
    _is_llm,
    _tok,
    _cli,
)

# Backwards-compatible aliases
LogRecord = LogEntry  # noqa: F811 — intentional override with the richer dataclass
JSONLBackend = LLMLogger
SQLiteBackend = LLMLogger
StdoutBackend = LLMLogger
_detect_provider = _extract_provider

__all__ = [
    # Proxy
    "LLMAPIProxy",
    "LogStore",
    # Middleware
    "LogEntry",
    "LogRecord",
    "LLMLogger",
    "JSONLBackend",
    "SQLiteBackend",
    "StdoutBackend",
    "COST_TABLE",
    "estimate_cost",
    "patch_urllib",
    "unpatch_urllib",
    "session",
    "main",
    "_cli",
    "_detect_provider",
    "_extract_provider",
    "_extract_model",
    "_is_llm",
    "_tok",
]
