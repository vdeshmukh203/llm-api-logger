"""
llm_api_logger: transparent HTTP proxy for logging LLM API traffic.

Runs as a local HTTP proxy that intercepts requests to LLM provider APIs
(OpenAI, Anthropic, Google, Mistral, and any OpenAI-compatible server),
logs request/response pairs with SHA-256-linked provenance to JSONL files,
and forwards traffic transparently.
"""

__version__ = "1.0.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .proxy import LLMAPIProxy
from .store import LogStore, LogRecord

__all__ = ["LLMAPIProxy", "LogStore", "LogRecord"]
