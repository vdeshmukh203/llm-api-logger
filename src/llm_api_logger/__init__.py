"""
llm_api_logger: Transparent logging for LLM API calls with cost estimation.

This package exposes the full public API from the canonical implementation
module (``llm_api_logger.py``), which is registered in ``pyproject.toml``
via ``py-modules = ["llm_api_logger"]``.

Key classes and functions
-------------------------
- :class:`LogEntry`   — dataclass representing a single API call
- :class:`LLMLogger`  — storage backend (SQLite or JSONL)
- :func:`estimate_cost` — USD cost from token counts
- :func:`patch_urllib` / :func:`unpatch_urllib` — automatic interception
- :func:`session`     — context manager for scoped logging
- :func:`main`        — CLI entry point
"""

__version__ = "1.0.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from llm_api_logger import (  # noqa: E402
    LogEntry,
    LLMLogger,
    COST_TABLE,
    estimate_cost,
    patch_urllib,
    unpatch_urllib,
    session,
    main,
    # backwards-compatible aliases
    LogRecord,
    JSONLBackend,
    SQLiteBackend,
    StdoutBackend,
    _detect_provider,
    _extract_model,
)

__all__ = [
    "LogEntry",
    "LLMLogger",
    "COST_TABLE",
    "estimate_cost",
    "patch_urllib",
    "unpatch_urllib",
    "session",
    "main",
    "LogRecord",
    "JSONLBackend",
    "SQLiteBackend",
    "StdoutBackend",
]
