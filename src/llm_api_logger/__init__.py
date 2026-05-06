"""
llm_api_logger package – re-exports the public API.

The authoritative implementation lives in the top-level ``llm_api_logger``
module (flat-module layout, installed via ``pyproject.toml``).  This package
shim exists so that the source tree can be imported as a package during
development and so that sub-module paths like
``llm_api_logger.proxy.LLMAPIProxy`` remain accessible.
"""

__version__ = "1.0.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .proxy import LLMAPIProxy
from .store import LogStore

__all__ = ["LLMAPIProxy", "LogStore"]
