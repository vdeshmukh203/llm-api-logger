# llm-api-logger

Lightweight middleware for capturing, storing, and analysing LLM API calls made by Python applications.

[![CI](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

## Features

- **Automatic capture** — monkey-patches `urllib.request.urlopen` so existing code needs no changes.
- **Two storage backends** — SQLite (persistent, queryable) and JSONL (human-readable, append-only).
- **Cost estimation** — built-in pricing table for 30+ models across OpenAI, Anthropic, Google, Mistral, and Meta.
- **Token tracking** — parses `usage` fields from OpenAI, Anthropic, and Google Gemini response formats.
- **SHA-256 provenance** — each record in the proxy-based store carries a content hash for integrity verification.
- **CLI** — `llm-api-logger summary|query|export` for quick inspection from the terminal.
- **Desktop GUI** — Tkinter viewer with filtering, sorting, and CSV/JSONL export.
- **Local HTTP proxy** — optional `LLMAPIProxy` class for capturing traffic below the SDK layer.

## Installation

```bash
pip install llm-api-logger
```

Or from source:

```bash
git clone https://github.com/vdeshmukh203/llm-api-logger
cd llm-api-logger
pip install -e .
```

## Quick start

### Context manager (recommended)

```python
import llm_api_logger as lal

with lal.session("my_run.jsonl") as log:
    # Make LLM API calls as normal — they are captured automatically.
    import urllib.request, json
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-..."},
    )
    urllib.request.urlopen(req)

print(log.summary())
```

### Manual logging

```python
import llm_api_logger as lal

logger = lal.LLMLogger("calls.db", backend="sqlite")

entry = lal.LogEntry(
    url="https://api.anthropic.com/v1/messages",
    model="claude-3-5-sonnet",
    tokens_in=500,
    tokens_out=150,
    latency_ms=820.0,
)
logger.record(entry)

print(logger.summary())
```

### Cost estimation

```python
from llm_api_logger import estimate_cost

cost = estimate_cost("gpt-4o", tokens_in=10_000, tokens_out=2_000)
print(f"Estimated cost: ${cost:.4f}")
```

## CLI

```bash
# Summarise a log file
llm-api-logger summary my_run.jsonl

# List recent calls filtered by model
llm-api-logger query my_run.jsonl --model gpt-4o --limit 20

# Export to CSV
llm-api-logger export my_run.jsonl --output calls.csv --format csv
```

## Desktop GUI

```bash
# Open the GUI with a log file
llm-api-logger-gui my_run.jsonl

# Or from Python
python llm_api_logger_gui.py my_run.jsonl
```

The GUI supports:
- Opening JSONL and SQLite log files
- Filtering by model, provider, and HTTP status code
- Sortable columns
- Inline JSON pretty-printing for request/response bodies
- Export of the current (filtered) view to CSV or JSONL

## Local HTTP proxy (advanced)

For capturing traffic below the SDK layer — useful when applications use `httpx`, `requests`, or other HTTP libraries — run the built-in proxy:

```python
from llm_api_logger.store import LogStore
from llm_api_logger.proxy import LLMAPIProxy

store = LogStore("proxy_log.jsonl")
with LLMAPIProxy(store, port=8080) as proxy:
    print(f"Proxy running at {proxy.address}")
    # Set HTTPS_PROXY=http://127.0.0.1:8080 in your application
    input("Press Enter to stop …")

print(store.summary())
```

Each record includes a SHA-256 hash of `(url, request_body, response_body)` for provenance verification:

```python
for record in store:
    assert record.verify(), f"Integrity check failed for record {record.id}"
```

## Supported models

| Provider  | Models |
|-----------|--------|
| OpenAI    | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o3-mini |
| Anthropic | claude-opus-4, claude-sonnet-4, claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-haiku, claude-2.1 |
| Google    | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, gemini-pro |
| Mistral   | mistral-large, mistral-medium, mistral-small |
| Meta      | llama-3-8b, llama-3-70b, llama-2-7b/13b/70b |

## API reference

### `LogEntry`

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID v4 |
| `url` | str | Request URL |
| `provider` | str | Derived from URL (openai, anthropic, google, …) |
| `model` | str | Extracted from request/response body |
| `tokens_in` | int | Input token count |
| `tokens_out` | int | Output token count |
| `cost_usd` | float | Estimated cost in USD |
| `latency_ms` | float | End-to-end latency in milliseconds |
| `timestamp` | str | ISO-8601 UTC timestamp |
| `error` | str \| None | Exception message if the call failed |

### `LLMLogger`

```python
LLMLogger(db_path=":memory:", backend="sqlite")
```

| Method | Description |
|--------|-------------|
| `record(entry)` | Store a `LogEntry`. |
| `count()` | Number of stored entries. |
| `query(model, provider, status_code, since)` | Filtered list, newest first. |
| `summary()` | Aggregate statistics dict. |
| `export_jsonl(path, append=False)` | Write entries to JSONL. |
| `export_csv(path)` | Write entries to CSV. |

### `session(log_file, backend, auto_patch)`

Context manager that patches `urllib.request.urlopen` for the duration of the block and yields an `LLMLogger`.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## Contributing

Bug reports and pull requests are welcome at <https://github.com/vdeshmukh203/llm-api-logger>.

## Citation

If you use this software in research, please cite the accompanying paper:

```bibtex
@article{deshmukh2026llmapilogger,
  title   = {llm-api-logger: An HTTP proxy for transparent logging of LLM API traffic with structured provenance records},
  author  = {Deshmukh, Vaibhav},
  journal = {Journal of Open Source Software},
  year    = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
