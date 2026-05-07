"""
llm_api_logger — transparent HTTP proxy for logging LLM API traffic.

Public API
----------
LLMAPIProxy
    HTTP proxy server that captures LLM calls.
LogStore
    Persistent store with tamper-evident SHA-256 hashing.
LLMLogger
    Convenience logger with SQLite/JSONL backends and cost estimation.
LogEntry
    Dataclass representing a single API call.
"""

__version__ = "1.0.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .proxy import LLMAPIProxy, LLM_HOSTS
from .store import LogStore, Record
from .logger import (
    LLMLogger,
    LogEntry,
    COST_TABLE,
    estimate_cost,
    patch_urllib,
    unpatch_urllib,
    session,
    main,
    _gui_main,
    _extract_provider,
    _extract_model,
    # backward-compat aliases
    LogRecord,
    JSONLBackend,
    SQLiteBackend,
    StdoutBackend,
    _detect_provider,
)

__all__ = [
    # proxy / store
    "LLMAPIProxy",
    "LLM_HOSTS",
    "LogStore",
    "Record",
    # logger
    "LLMLogger",
    "LogEntry",
    "COST_TABLE",
    "estimate_cost",
    "patch_urllib",
    "unpatch_urllib",
    "session",
    "main",
    "_gui_main",
    # private helpers (exposed for tests)
    "_extract_provider",
    "_extract_model",
    "_detect_provider",
    # backward-compat aliases
    "LogRecord",
    "JSONLBackend",
    "SQLiteBackend",
    "StdoutBackend",
]
