"""
llm_api_logger — re-export the public API from the top-level module.

This package shim ensures that both ``import llm_api_logger`` and
``from llm_api_logger import LLMLogger`` work regardless of whether the
package is installed as a flat module or as a namespace package under ``src/``.
"""

from llm_api_logger import (  # noqa: F401
    __version__,
    __author__,
    __license__,
    COST_TABLE,
    LogEntry,
    LLMLogger,
    estimate_cost,
    patch_urllib,
    unpatch_urllib,
    session,
    _extract_provider,
    _extract_model,
    _extract_tokens,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "COST_TABLE",
    "LogEntry",
    "LLMLogger",
    "estimate_cost",
    "patch_urllib",
    "unpatch_urllib",
    "session",
    "_extract_provider",
    "_extract_model",
    "_extract_tokens",
]
