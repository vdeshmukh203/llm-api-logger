"""
llm_api_logger package shim.

The canonical implementation lives in the root-level ``llm_api_logger.py``
module (exposed via ``py-modules`` in pyproject.toml).  This package re-exports
its public API so that both ``import llm_api_logger`` and
``from llm_api_logger import LLMLogger`` work regardless of how the package is
installed.
"""

from llm_api_logger import (  # noqa: F401
    __version__,
    COST_TABLE,
    estimate_cost,
    LogEntry,
    LLMLogger,
    session,
    patch_urllib,
    unpatch_urllib,
    _extract_provider,
    _extract_model,
    _tok,
    # backwards-compat aliases
    LogRecord,
    JSONLBackend,
    SQLiteBackend,
    StdoutBackend,
    _detect_provider,
)

__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# Historical names kept for any external code that imported them directly.
# The proxy/store architecture has been superseded by LLMLogger.
LLMAPIProxy = LLMLogger
LogStore = LLMLogger

__all__ = [
    "LLMLogger",
    "LogEntry",
    "session",
    "patch_urllib",
    "unpatch_urllib",
    "estimate_cost",
    "COST_TABLE",
    "LLMAPIProxy",
    "LogStore",
]
