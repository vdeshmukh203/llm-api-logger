"""Local HTTP proxy that transparently intercepts and logs LLM API traffic.

Set ``HTTP_PROXY`` / ``HTTPS_PROXY`` to ``http://127.0.0.1:<port>`` (or use
:meth:`LLMAPIProxy.env` to obtain the mapping) to route SDK traffic through
the proxy without modifying application code.
"""

import json
import socket
import ssl
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from .store import LogRecord, LogStore

_LLM_HOSTS = frozenset(
    [
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.mistral.ai",
        "api.cohere.ai",
        "api.together.xyz",
        "api-inference.huggingface.co",
    ]
)

_PROVIDER_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "mistral": "mistral",
    "cohere": "cohere",
    "together": "together",
    "huggingface": "huggingface",
}


def _detect_provider(url: str) -> str:
    url_lower = url.lower()
    for key, provider in _PROVIDER_MAP.items():
        if key in url_lower:
            return provider
    return "unknown"


def _parse_usage(body: Optional[str]) -> tuple:
    """Return (tokens_in, tokens_out) from a JSON response body."""
    if not body:
        return 0, 0
    try:
        d = json.loads(body)
        if "usage" in d:
            u = d["usage"]
            return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        if "usageMetadata" in d:
            u = d["usageMetadata"]
            return u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
    except Exception:
        pass
    return 0, 0


def _parse_model(body: Optional[str]) -> str:
    if not body:
        return ""
    try:
        d = json.loads(body)
        if isinstance(d, dict):
            for key in ("model", "modelId", "model_id", "engine"):
                if key in d:
                    return str(d[key])
    except Exception:
        pass
    return ""


class _ProxyHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 proxy handler that forwards requests and logs LLM traffic."""

    store: LogStore  # injected by LLMAPIProxy

    # ------------------------------------------------------------------
    def _forward(self, method: str) -> None:
        url = self.path
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        req_body = self.rfile.read(content_length) if content_length else b""

        forward_headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "proxy-connection")
        }

        record = LogRecord(
            url=url,
            method=method,
            provider=_detect_provider(url),
            request_headers=json.dumps(dict(self.headers)),
            request_body=req_body.decode("utf-8", errors="replace") if req_body else None,
        )
        record.model = _parse_model(record.request_body)

        start = time.perf_counter()
        try:
            req = urllib.request.Request(
                url,
                data=req_body if req_body else None,
                headers=forward_headers,
                method=method,
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_body = resp.read()
                record.status_code = resp.status
                record.response_headers = json.dumps(dict(resp.headers))
                record.response_body = resp_body.decode("utf-8", errors="replace")
                record.latency_ms = (time.perf_counter() - start) * 1000
                ti, to = _parse_usage(record.response_body)
                record.tokens_in = ti
                record.tokens_out = to
                if not record.model:
                    record.model = _parse_model(record.response_body)
                self.store.append(record)

                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp_body)

        except Exception as exc:
            record.latency_ms = (time.perf_counter() - start) * 1000
            record.error = str(exc)
            self.store.append(record)
            try:
                self.send_error(502, f"Proxy error: {exc}")
            except Exception:
                pass

    def do_POST(self) -> None:
        self._forward("POST")

    def do_GET(self) -> None:
        self._forward("GET")

    def do_PUT(self) -> None:
        self._forward("PUT")

    def do_DELETE(self) -> None:
        self._forward("DELETE")

    def log_message(self, fmt, *args) -> None:  # suppress default stderr output
        pass


class LLMAPIProxy:
    """Local HTTP proxy for transparent LLM API traffic capture.

    Usage::

        store = LogStore("llm_proxy.jsonl")
        proxy = LLMAPIProxy(port=8080, store=store)
        proxy.start()
        # point your SDK: HTTP_PROXY=http://127.0.0.1:8080
        proxy.stop()

    Or as a context manager::

        with LLMAPIProxy(port=8080, store=store) as proxy:
            os.environ.update(proxy.env)
            ...
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        store: Optional[LogStore] = None,
    ):
        self.host = host
        self.port = port
        self.store = store if store is not None else LogStore(":memory:")
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> "LLMAPIProxy":
        """Start the proxy in a daemon background thread."""
        if self._server is not None:
            return self

        store = self.store

        class _Handler(_ProxyHandler):
            pass

        _Handler.store = store  # type: ignore[attr-defined]

        self._server = HTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="llm-api-proxy"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Shut down the proxy server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    @property
    def address(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def env(self) -> dict:
        """Environment variables to route SDK traffic through this proxy."""
        return {"HTTP_PROXY": self.address, "HTTPS_PROXY": self.address}

    def __enter__(self) -> "LLMAPIProxy":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:  # pragma: no cover
        state = "running" if self._server else "stopped"
        return f"LLMAPIProxy({self.address}, {state})"
