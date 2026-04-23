"""
llm_api_logger: Transparent HTTP proxy for logging LLM API traffic.

Runs as a local HTTP proxy that intercepts requests to LLM provider APIs
(OpenAI, Anthropic, Cohere, etc.), logs request/response pairs with
SHA-256-linked provenance to JSONL files, and forwards traffic transparently.
Enables passive capture of LLM interactions without modifying application code.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .proxy import LLMAPIProxy
from .store import LogStore

__all__ = ["LLMAPIProxy", "LogStore"]
