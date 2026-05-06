"""
Thin re-export shim so ``from llm_api_logger.store import LogStore`` works.

The canonical implementation lives in the top-level ``llm_api_logger`` module.
``LogStore`` is an alias for :class:`llm_api_logger.LLMLogger`.
"""

import sys
import pathlib

# Ensure the root package is importable when running from the src tree
_root = str(pathlib.Path(__file__).parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_api_logger import LLMLogger as LogStore  # noqa: E402

__all__ = ["LogStore"]
