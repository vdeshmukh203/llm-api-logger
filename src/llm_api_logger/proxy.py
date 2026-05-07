"""
LLMAPIProxy — transparent HTTP proxy for logging LLM API traffic.

Usage::

    from llm_api_logger import LLMAPIProxy, LogStore

    store = LogStore("calls.jsonl")
    with LLMAPIProxy(store, port=8080) as proxy:
        # Point your HTTP client at http://localhost:8080
        os.environ["http_proxy"] = "http://localhost:8080"
        # ... make LLM API calls ...

All HTTP requests destined for known LLM provider hostnames are intercepted,
forwarded to the real server, logged to *store*, and returned to the caller
transparently.

Note
----
HTTPS interception requires a MITM certificate.  This module handles plain
HTTP only.  For HTTPS traffic, configure your TLS termination proxy upstream
and point it at this proxy for the HTTP leg.
"""

from __future__ import annotations

import http.server
import threading
import time
import json
import urllib.error
import urllib.request
from typing import ClassVar, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .store import LogStore, Record

# ---------------------------------------------------------------------------
# Known LLM provider hostnames
# ---------------------------------------------------------------------------

LLM_HOSTS: Dict[str, str] = {
    "api.openai.com":                      "openai",
    "api.anthropic.com":                   "anthropic",
    "generativelanguage.googleapis.com":   "google",
    "api.mistral.ai":                      "mistral",
    "api.cohere.ai":                       "cohere",
    "api.together.xyz":                    "together",
    "api-inference.huggingface.co":        "huggingface",
    "openrouter.ai":                       "openrouter",
}

# Headers that must not be forwarded as-is
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    """BaseHTTPRequestHandler subclass that proxies and logs LLM calls."""

    store: ClassVar[Optional["LogStore"]] = None

    # ------------------------------------------------------------------ #
    # HTTP method dispatch
    # ------------------------------------------------------------------ #

    def do_GET(self) -> None:    self._proxy("GET")
    def do_POST(self) -> None:   self._proxy("POST")
    def do_PUT(self) -> None:    self._proxy("PUT")
    def do_DELETE(self) -> None: self._proxy("DELETE")
    def do_PATCH(self) -> None:  self._proxy("PATCH")

    # ------------------------------------------------------------------ #
    # Core proxy logic
    # ------------------------------------------------------------------ #

    def _proxy(self, method: str) -> None:
        host = self.headers.get("Host", "")
        provider = LLM_HOSTS.get(host, "")

        # Read request body
        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length > 0 else b""
        request_body: Optional[str] = body_bytes.decode("utf-8", errors="ignore") or None

        url = f"http://{host}{self.path}"
        req = urllib.request.Request(url, method=method)
        for key, val in self.headers.items():
            if key.lower() not in _HOP_BY_HOP and key.lower() != "host":
                req.add_header(key, val)
        if body_bytes:
            req.data = body_bytes

        start = time.monotonic()
        status = 502
        response_body: Optional[str] = None
        error: Optional[str] = None

        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
                raw = resp.read()
                response_body = raw.decode("utf-8", errors="ignore")
                resp_headers = dict(resp.headers)

            self.send_response(status)
            for key, val in resp_headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, val)
            self.end_headers()
            if raw:
                self.wfile.write(raw)

        except urllib.error.HTTPError as exc:
            status = exc.code
            error = str(exc)
            self.send_error(status, error)
        except Exception as exc:
            error = str(exc)
            self.send_error(502, error)

        if provider and self.store is not None:
            latency_ms = (time.monotonic() - start) * 1000
            self._log(url, method, provider, request_body, response_body, status, latency_ms, error)

    def _log(
        self,
        url: str,
        method: str,
        provider: str,
        request_body: Optional[str],
        response_body: Optional[str],
        status: int,
        latency_ms: float,
        error: Optional[str],
    ) -> None:
        from .store import Record
        model = "unknown"
        for body in filter(None, [request_body, response_body]):
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    for key in ("model", "modelId", "model_id", "engine"):
                        if key in data:
                            model = str(data[key])
                            break
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            if model != "unknown":
                break

        rec = Record(
            url=url,
            method=method,
            provider=provider,
            model=model,
            request_body=request_body,
            response_body=response_body,
            status_code=status,
            latency_ms=latency_ms,
            error=error,
        )
        self.store.append(rec)

    # ------------------------------------------------------------------ #
    # Suppress default access-log output
    # ------------------------------------------------------------------ #

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
        pass


# ---------------------------------------------------------------------------
# Public proxy class
# ---------------------------------------------------------------------------

class LLMAPIProxy:
    """Transparent HTTP proxy that logs LLM API calls to a :class:`~llm_api_logger.store.LogStore`.

    Parameters
    ----------
    store:
        Where captured records are written.  Defaults to an in-memory
        SQLite :class:`~llm_api_logger.store.LogStore`.
    host:
        Interface to bind.  Defaults to ``"localhost"``.
    port:
        TCP port to listen on.  Defaults to ``8080``.

    Examples
    --------
    >>> from llm_api_logger import LLMAPIProxy, LogStore
    >>> store = LogStore(":memory:", "sqlite")
    >>> with LLMAPIProxy(store, port=9090) as proxy:
    ...     # configure HTTP_PROXY=http://localhost:9090 and make calls
    ...     pass
    >>> records = store.load()
    """

    def __init__(
        self,
        store: Optional["LogStore"] = None,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        if store is None:
            from .store import LogStore
            store = LogStore(":memory:", "sqlite")
        self.store = store
        self.host = host
        self.port = port
        self._server: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Bind the socket and start serving in a daemon thread."""
        _ProxyHandler.store = self.store
        self._server = http.server.HTTPServer((self.host, self.port), _ProxyHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name=f"LLMAPIProxy:{self.port}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down the server and join the thread."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "LLMAPIProxy":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def __repr__(self) -> str:  # pragma: no cover
        return f"LLMAPIProxy(host={self.host!r}, port={self.port})"
