"""
Thin re-export shim so ``from llm_api_logger.proxy import LLMAPIProxy`` works.

``LLMAPIProxy`` wraps :class:`llm_api_logger.LLMLogger` together with the
:func:`llm_api_logger.patch_urllib` / :func:`llm_api_logger.unpatch_urllib`
helpers so that callers get a single proxy-style object.
"""

import sys
import pathlib

_root = str(pathlib.Path(__file__).parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import llm_api_logger as _lal  # noqa: E402


class LLMAPIProxy:
    """High-level proxy object that combines logging and urllib patching.

    Parameters
    ----------
    db_path:
        Path to the SQLite database or JSONL file used for storage.
    backend:
        ``"sqlite"`` or ``"jsonl"``.

    Examples
    --------
    >>> proxy = LLMAPIProxy()
    >>> proxy.start()
    >>> # … make LLM API calls …
    >>> proxy.stop()
    >>> summary = proxy.logger.summary()
    """

    def __init__(self, db_path: str = ":memory:", backend: str = "sqlite") -> None:
        self.logger = _lal.LLMLogger(db_path=db_path, backend=backend)
        self._active = False

    def start(self) -> None:
        """Activate urllib interception."""
        _lal.patch_urllib(self.logger)
        self._active = True

    def stop(self) -> None:
        """Deactivate urllib interception."""
        _lal.unpatch_urllib()
        self._active = False

    def __enter__(self) -> "LLMAPIProxy":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


__all__ = ["LLMAPIProxy"]
