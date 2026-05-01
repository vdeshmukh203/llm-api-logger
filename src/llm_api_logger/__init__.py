"""
llm_api_logger: Middleware HTTP logger for LLM API calls.

This module re-exports the public API from the top-level ``llm_api_logger``
package so that both ``import llm_api_logger`` and
``from llm_api_logger import ...`` work regardless of whether the package is
installed via pip or used directly from the source tree.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from llm_api_logger import (  # noqa: F401
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
    _tok,
    _is_llm,
    # backward-compat aliases
    LogRecord,
    JSONLBackend,
    SQLiteBackend,
    StdoutBackend,
    _detect_provider,
)

__all__ = [
    "COST_TABLE",
    "LogEntry",
    "LLMLogger",
    "estimate_cost",
    "patch_urllib",
    "unpatch_urllib",
    "session",
    "main",
]
