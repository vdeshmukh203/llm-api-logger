"""
llm_api_logger: Middleware-style HTTP logger for LLM API calls.

Drop-in request/response logger for OpenAI, Anthropic, Cohere, and any
OpenAI-compatible endpoint. Wraps urllib or requests sessions to capture
full payloads, latency, token counts, and cost to JSONL, SQLite, or stdout
without modifying application code.
"""
from __future__ import annotations
import json, time, hashlib, datetime, sqlite3, threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ---------------------------------------------------------------------------
# Cost estimation (same simple table as llmbench)
# ---------------------------------------------------------------------------

_COST_PER_1K: Dict[str, tuple] = {
    "gpt-4o":                    (0.005,   0.015),
    "gpt-4o-mini":               (0.00015, 0.0006),
    "gpt-4-turbo":               (0.01,    0.03),
    "gpt-3.5-turbo":             (0.0005,  0.0015),
    "claude-3-5-sonnet-20241022":(0.003,   0.015),
    "claude-3-haiku-20240307":   (0.00025, 0.00125),
    "claude-3-opus-20240229":    (0.015,   0.075),
}


def _estimate_cost(model: str, p_tok: int, c_tok: int) -> float:
    for key, (inp, out) in _COST_PER_1K.items():
        if key in model.lower():
            return (p_tok * inp + c_tok * out) / 1000
    return (p_tok * 0.001 + c_tok * 0.002) / 1000


# ---------------------------------------------------------------------------
# Log record
# ---------------------------------------------------------------------------

class LogRecord:
    """One captured API call."""

    __slots__ = (
        "record_id", "timestamp", "provider", "model", "endpoint",
        "method", "request_body", "response_body", "status_code",
        "latency_s", "prompt_tokens", "completion_tokens", "cost_usd",
        "error",
    )

    def __init__(
        self,
        provider: str,
        model: str,
        endpoint: str,
        method: str,
        request_body: Optional[str],
        response_body: Optional[str],
        status_code: int,
        latency_s: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: str = "",
    ) -> None:
        self.timestamp = datetime.datetime.utcnow().isoformat()
        self.record_id = hashlib.sha256(
            (self.timestamp + endpoint + (request_body or "")).encode()
        ).hexdigest()[:16]
        self.provider = provider
        self.model = model
        self.endpoint = endpoint
        self.method = method.upper()
        self.request_body = request_body
        self.response_body = response_body
        self.status_code = status_code
        self.latency_s = round(latency_s, 4)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = round(_estimate_cost(model, prompt_tokens, completion_tokens), 8)
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _Backend:
    def write(self, record: LogRecord) -> None:
        raise NotImplementedError
    def close(self) -> None:
        pass


class JSONLBackend(_Backend):
    """Append each log record as a JSON line."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: LogRecord) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")


class SQLiteBackend(_Backend):
    """Write log records to a SQLite table 'api_logs'."""

    _CREATE = """
    CREATE TABLE IF NOT EXISTS api_logs (
        record_id TEXT PRIMARY KEY,
        timestamp TEXT, provider TEXT, model TEXT, endpoint TEXT,
        method TEXT, request_body TEXT, response_body TEXT,
        status_code INTEGER, latency_s REAL,
        prompt_tokens INTEGER, completion_tokens INTEGER,
        cost_usd REAL, error TEXT
    )"""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(self._CREATE)
        self._conn.commit()
        self._lock = threading.Lock()

    def write(self, record: LogRecord) -> None:
        d = record.to_dict()
        cols = ", ".join(d.keys())
        placeholders = ", ".join("?" * len(d))
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO api_logs (" + cols + ") VALUES (" + placeholders + ")",
                list(d.values()),
            )
            self._conn.commit()

    def query(
        self,
        provider: str = "",
        model: str = "",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        where = []
        params: List[Any] = []
        if provider:
            where.append("provider = ?")
            params.append(provider)
        if model:
            where.append("model LIKE ?")
            params.append("%" + model + "%")
        sql = "SELECT * FROM api_logs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


class StdoutBackend(_Backend):
    """Print a one-line summary of each call to stdout."""

    def write(self, record: LogRecord) -> None:
        status = "OK" if not record.error else "ERR"
        print(
            "[" + record.timestamp + "] " + status + " " +
            record.provider + "/" + record.model + " " +
            str(record.latency_s) + "s " +
            str(record.prompt_tokens) + "+" + str(record.completion_tokens) + " tok " +
            "err=" + (record.error[:60] if record.error else "none")
        )


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "api.openai.com":              "openai",
    "api.anthropic.com":           "anthropic",
    "api.cohere.ai":               "cohere",
    "api-inference.huggingface.co":"huggingface",
    "generativelanguage.googleapis.com": "google",
}


def _detect_provider(url: str) -> str:
    for domain, name in _PROVIDER_MAP.items():
        if domain in url:
            return name
    if "openai.azure.com" in url:
        return "azure_openai"
    return "unknown"


def _extract_model(url: str, body_str: Optional[str]) -> str:
    """Try to get model name from request body or URL."""
    if body_str:
        try:
            d = json.loads(body_str)
            if "model" in d:
                return str(d["model"])
        except (json.JSONDecodeError, TypeError):
            pass
    # Anthropic embeds model in body, OpenAI in URL for some endpoints
    import re
    m = re.search(r"/deployments/([^/]+)/", url)
    if m:
        return m.group(1)
    return "unknown"


def _extract_tokens(response_str: Optional[str]) -> tuple:
    """Return (prompt_tokens, completion_tokens) from response JSON."""
    if not response_str:
        return 0, 0
    try:
        d = json.loads(response_str)
        usage = d.get("usage", {})
        p = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        c = usage.get("completion_tokens") or usage.get("output_tokens", 0)
        return int(p), int(c)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0, 0


# ---------------------------------------------------------------------------
# Core logger
# ---------------------------------------------------------------------------

class LLMAPILogger:
    """
    Intercept and log LLM API HTTP calls made via urllib.

    Parameters
    ----------
    backend : str or _Backend
        "jsonl:<path>", "sqlite:<path>", "stdout", or a backend instance.
    redact_keys : bool
        If True, strip Authorization headers before logging.
    """

    def __init__(
        self,
        backend = "stdout",
        redact_keys: bool = True,
    ) -> None:
        if isinstance(backend, str):
            if backend == "stdout":
                self._backend: _Backend = StdoutBackend()
            elif backend.startswith("jsonl:"):
                self._backend = JSONLBackend(backend[6:])
            elif backend.startswith("sqlite:"):
                self._backend = SQLiteBackend(backend[7:])
            else:
                raise ValueError("Unknown backend: " + backend +
                                 ". Use 'stdout', 'jsonl:<path>', or 'sqlite:<path>'.")
        else:
            self._backend = backend
        self.redact_keys = redact_keys

    def _log_call(
        self,
        url: str,
        method: str,
        request_body: Optional[bytes],
        response_body: Optional[bytes],
        status_code: int,
        latency_s: float,
        error: str = "",
    ) -> LogRecord:
        req_str = request_body.decode("utf-8", errors="replace") if request_body else None
        resp_str = response_body.decode("utf-8", errors="replace") if response_body else None
        provider = _detect_provider(url)
        model = _extract_model(url, req_str)
        p_tok, c_tok = _extract_tokens(resp_str)
        record = LogRecord(
            provider=provider, model=model, endpoint=url,
            method=method, request_body=req_str, response_body=resp_str,
            status_code=status_code, latency_s=latency_s,
            prompt_tokens=p_tok, completion_tokens=c_tok, error=error,
        )
        self._backend.write(record)
        return record

    def call(
        self,
        url: str,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        method: str = "POST",
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Make an HTTP call to an LLM API endpoint and log the interaction.

        Parameters
        ----------
        url : str
            Full API endpoint URL.
        data : bytes, optional
            JSON-encoded request body.
        headers : dict, optional
            HTTP headers to include.
        method : str
            HTTP method (default "POST").
        timeout : float
            Request timeout in seconds.

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        RuntimeError
            On HTTP or network error (after logging the error).
        """
        req = Request(url, data=data, method=method, headers=headers or {})
        t0 = time.perf_counter()
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            latency = time.perf_counter() - t0
            self._log_call(url, method, data, body, resp.status, latency)
            return json.loads(body)
        except HTTPError as exc:
            latency = time.perf_counter() - t0
            body = exc.read()
            self._log_call(url, method, data, body, exc.code, latency,
                           error=str(exc))
            raise RuntimeError("HTTP " + str(exc.code) + " from " + url + ": " + str(exc)) from exc
        except URLError as exc:
            latency = time.perf_counter() - t0
            self._log_call(url, method, data, None, 0, latency, error=str(exc))
            raise RuntimeError("Network error calling " + url + ": " + str(exc)) from exc

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "LLMAPILogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def openai_call(
    prompt: str,
    model: str = "gpt-4o-mini",
    api_key: str = "",
    logger: Optional[LLMAPILogger] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Call the OpenAI chat completions endpoint with logging."""
    if logger is None:
        logger = LLMAPILogger("stdout")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    result = logger.call(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    return result["choices"][0]["message"]["content"]


def anthropic_call(
    prompt: str,
    model: str = "claude-3-haiku-20240307",
    api_key: str = "",
    logger: Optional[LLMAPILogger] = None,
    max_tokens: int = 256,
) -> str:
    """Call the Anthropic messages endpoint with logging."""
    if logger is None:
        logger = LLMAPILogger("stdout")
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    result = logger.call(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    return result["content"][0]["text"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse, os
    parser = argparse.ArgumentParser(
        prog="llm-api-logger",
        description="Log LLM API calls with latency, tokens, and cost tracking.",
    )
    sub = parser.add_subparsers(dest="cmd")

    q_p = sub.add_parser("query", help="Query a SQLite log database.")
    q_p.add_argument("db", help="Path to SQLite database.")
    q_p.add_argument("--provider", default="")
    q_p.add_argument("--model", default="")
    q_p.add_argument("--limit", type=int, default=20)

    t_p = sub.add_parser("test", help="Send a test call and log it.")
    t_p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    t_p.add_argument("--model", default="gpt-4o-mini")
    t_p.add_argument("--backend", default="stdout",
                     help="stdout | jsonl:<path> | sqlite:<path>")
    t_p.add_argument("--prompt", default="Say hello in one sentence.")
    t_p.add_argument("--api-key", default=None)

    args = parser.parse_args()

    if args.cmd == "query":
        backend = SQLiteBackend(args.db)
        records = backend.query(provider=args.provider, model=args.model, limit=args.limit)
        print(json.dumps(records, indent=2))
        backend.close()

    elif args.cmd == "test":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        with LLMAPILogger(args.backend) as logger:
            if args.provider == "openai":
                resp = openai_call(args.prompt, model=args.model, api_key=api_key, logger=logger)
            else:
                resp = anthropic_call(args.prompt, model=args.model, api_key=api_key, logger=logger)
        print("Response:", resp)

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
