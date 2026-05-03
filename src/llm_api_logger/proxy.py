"""Threaded local HTTP proxy that captures LLM API traffic to a LogStore."""

import http.client
import http.server
import json
import threading
import time
import urllib.parse
from typing import Optional

from .store import LogRecord, LogStore

_LLM_HOSTS = frozenset([
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.mistral.ai",
    "api.cohere.ai",
    "api.together.xyz",
])

_PROVIDER_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "gemini": "google",
    "mistral": "mistral",
    "cohere": "cohere",
    "together": "together",
}


def _detect_provider(host: str) -> str:
    host_lower = host.lower()
    for kw, prov in _PROVIDER_MAP.items():
        if kw in host_lower:
            return prov
    return "unknown"


def _extract_model(body: Optional[str]) -> str:
    if not body:
        return "unknown"
    try:
        d = json.loads(body)
        if isinstance(d, dict):
            for key in ("model", "modelId", "model_id", "engine"):
                if key in d:
                    return str(d[key])
    except (json.JSONDecodeError, ValueError):
        pass
    return "unknown"


def _extract_tokens(response_body: Optional[str]):
    """Return (tokens_in, tokens_out) from a JSON response."""
    if not response_body:
        return 0, 0
    try:
        d = json.loads(response_body)
        if isinstance(d, dict):
            if "usage" in d:
                u = d["usage"]
                ti = u.get("prompt_tokens") or u.get("input_tokens", 0)
                to = u.get("completion_tokens") or u.get("output_tokens", 0)
                return int(ti), int(to)
            if "usageMetadata" in d:
                u = d["usageMetadata"]
                return int(u.get("promptTokenCount", 0)), int(u.get("candidatesTokenCount", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return 0, 0


def _is_llm_host(host: str) -> bool:
    return host in _LLM_HOSTS or any(kw in host.lower() for kw in _PROVIDER_MAP)


class LLMAPIProxy:
    """Local HTTP proxy that forwards requests and logs LLM API traffic.

    Configure your client application to use this proxy with::

        export HTTPS_PROXY=http://127.0.0.1:8080

    Then run the proxy::

        from llm_api_logger.store import LogStore
        from llm_api_logger.proxy import LLMAPIProxy

        store = LogStore("run.jsonl")
        with LLMAPIProxy(store) as proxy:
            # application code here

    Args:
        store: :class:`~llm_api_logger.store.LogStore` instance to write records to.
        host: Address to bind the proxy server.
        port: TCP port for the proxy server (default 8080).
    """

    def __init__(self, store: LogStore, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.store = store
        self.host = host
        self.port = port
        self._server: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the proxy server in a daemon background thread."""
        store = self.store

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self._forward("POST")

            def do_GET(self):
                self._forward("GET")

            def do_PUT(self):
                self._forward("PUT")

            def do_PATCH(self):
                self._forward("PATCH")

            def _forward(self, method: str) -> None:
                parsed = urllib.parse.urlparse(self.path)
                host = parsed.hostname or self.headers.get("Host", "").split(":")[0]
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                is_llm = _is_llm_host(host)

                length = int(self.headers.get("Content-Length", 0) or 0)
                req_body_bytes = self.rfile.read(length) if length else b""
                req_body = req_body_bytes.decode("utf-8", errors="ignore") or None

                start = time.monotonic()
                resp_body: Optional[str] = None
                status = 502
                resp_headers: dict = {}

                try:
                    path_qs = parsed.path
                    if parsed.query:
                        path_qs += "?" + parsed.query

                    fwd_headers = {
                        k: v for k, v in self.headers.items()
                        if k.lower() not in ("host", "transfer-encoding", "proxy-connection")
                    }
                    fwd_headers["Host"] = host

                    if parsed.scheme == "https" or port == 443:
                        conn = http.client.HTTPSConnection(host, port, timeout=30)
                    else:
                        conn = http.client.HTTPConnection(host, port, timeout=30)

                    conn.request(method, path_qs, req_body_bytes, fwd_headers)
                    resp = conn.getresponse()
                    status = resp.status
                    resp_data = resp.read()
                    resp_body = resp_data.decode("utf-8", errors="ignore")
                    resp_headers = dict(resp.getheaders())

                    self.send_response(status)
                    for key, val in resp.getheaders():
                        if key.lower() not in ("transfer-encoding", "connection"):
                            self.send_header(key, val)
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    self.wfile.write(resp_data)

                except Exception as exc:
                    self.send_error(502, str(exc))
                    resp_body = None

                finally:
                    latency_ms = (time.monotonic() - start) * 1000
                    if is_llm:
                        ti, to = _extract_tokens(resp_body)
                        record = LogRecord(
                            url=self.path,
                            method=method,
                            request_headers=dict(self.headers),
                            request_body=req_body,
                            response_status=status,
                            response_headers=resp_headers,
                            response_body=resp_body,
                            latency_ms=latency_ms,
                            provider=_detect_provider(host),
                            model=_extract_model(req_body),
                            tokens_in=ti,
                            tokens_out=to,
                        )
                        store.append(record)

            def log_message(self, *args) -> None:  # silence default access log
                pass

        self._server = http.server.HTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shut down the proxy server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        self._thread = None

    def __enter__(self) -> "LLMAPIProxy":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    @property
    def address(self) -> str:
        """Return the proxy URL, e.g. ``http://127.0.0.1:8080``."""
        return f"http://{self.host}:{self.port}"
