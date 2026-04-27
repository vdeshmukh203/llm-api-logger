"""
llm_api_logger: re-export of the public API from the root module.

The installable package is the single-file ``llm_api_logger.py`` module in the
project root (configured via ``py-modules`` in pyproject.toml).  This package
stub re-exports the same symbols so that both ``import llm_api_logger`` and
``from llm_api_logger import …`` work regardless of how the source tree is laid
out on sys.path.
"""

__version__ = "0.2.0"
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
    _load_jsonl_into,
    _extract_provider,
    _extract_model,
)

__all__ = [
    "COST_TABLE",
    "LogEntry",
    "LLMLogger",
    "estimate_cost",
    "patch_urllib",
    "unpatch_urllib",
    "session",
    "_load_jsonl_into",
    "_extract_provider",
    "_extract_model",
]
