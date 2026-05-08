"""
llm_api_logger: Transparent middleware logger for LLM API traffic.

Intercepts urllib.request calls to LLM provider APIs (OpenAI, Anthropic,
Google, Mistral, Cohere, etc.), logs request/response pairs with cost and
token metadata to JSONL or SQLite storage, and provides a CLI and context
manager API for scoped logging sessions.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# Re-export the public API from the top-level module so that both
# ``import llm_api_logger`` and ``from llm_api_logger import ...`` work
# regardless of whether the package is installed via the src layout or used
# directly from the repository root.
import sys
import os

# Ensure the repository root is on sys.path when running from the src layout
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from llm_api_logger import (  # noqa: E402  (import after path manipulation)
    LogEntry,
    LLMLogger,
    COST_TABLE,
    estimate_cost,
    patch_urllib,
    unpatch_urllib,
    session,
    main,
    _extract_provider,
    _extract_model,
    _tok,
    _is_llm,
    # backwards-compatible aliases
    LogRecord,
    JSONLBackend,
    SQLiteBackend,
    StdoutBackend,
    _detect_provider,
    _cli,
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
    "_extract_provider",
    "_extract_model",
    "_tok",
    "_is_llm",
    "LogRecord",
    "JSONLBackend",
    "SQLiteBackend",
    "StdoutBackend",
    "_detect_provider",
    "_cli",
]
