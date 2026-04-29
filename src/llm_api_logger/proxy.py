"""LLMAPIProxy: thin wrapper that combines urllib patching with LogStore persistence."""

from typing import Optional
from .store import LogStore


class LLMAPIProxy:
    """
    Intercepts outbound LLM API calls via urllib monkey-patching and persists
    structured records to a LogStore.

    Usage::

        store = LogStore("run.jsonl")
        with LLMAPIProxy(store=store):
            response = openai_client.chat.completions.create(...)

    All HTTP requests to known LLM endpoints are captured transparently;
    the calling code receives unmodified response objects.
    """

    def __init__(self, store: Optional[LogStore] = None, port: int = 8080):
        self.store = store or LogStore(":memory:")
        self.port = port
        self._logger = None

    def start(self) -> None:
        """Begin intercepting requests."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        import llm_api_logger as _lal

        self._logger = _lal.LLMLogger(db_path=":memory:", backend="sqlite")
        _lal.patch_urllib(self._logger)

    def stop(self) -> None:
        """Stop intercepting and flush captured entries to the store."""
        import llm_api_logger as _lal

        _lal.unpatch_urllib()
        if self._logger is not None:
            for entry in self._logger.query():
                self.store.append(entry.to_dict())
        self._logger = None

    def __enter__(self) -> "LLMAPIProxy":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
