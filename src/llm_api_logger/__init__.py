"""
llm_api_logger: Python middleware for transparent logging of LLM API calls.

Intercepts ``urllib.request.urlopen`` to capture all HTTP traffic to LLM
provider endpoints (OpenAI, Anthropic, Google, Mistral, …) and stores
structured records—including token counts, latency, and cost estimates—to
SQLite or JSONL backends without modifying application code.
"""

__version__ = "1.0.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .proxy import LLMAPIProxy
from .store import LogStore

__all__ = ["LLMAPIProxy", "LogStore"]
