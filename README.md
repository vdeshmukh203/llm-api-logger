# llm-api-logger

[![CI](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

Transparent middleware library for logging, cost-tracking, and analysing LLM API calls.  Works by monkey-patching `urllib.request.urlopen` so every HTTP request to an LLM provider is captured automatically — no SDK changes required.

Supported providers: **OpenAI · Anthropic · Google Gemini · Mistral · Cohere · Together AI · HuggingFace**

---

## Features

- **Zero application changes** — wraps `urllib.request.urlopen`; any SDK that uses urllib is captured automatically
- **Dual storage backends** — SQLite (queryable, persistent) or JSONL (portable, streaming-friendly)
- **Automatic cost estimation** — 30+ models with versioned-name prefix matching (`gpt-4o-2024-05-13` → `gpt-4o`)
- **Multi-provider token extraction** — handles OpenAI, Anthropic, Google Gemini, and Cohere response formats
- **Context manager API** for scoped sessions
- **Web dashboard GUI** — interactive charts, filtering, pagination, and CSV/JSONL export
- **CLI** — `summary`, `query`, `export`, and `gui` sub-commands

---

## Installation

```bash
# Core library (no dependencies beyond the Python standard library)
pip install llm-api-logger

# With the web dashboard
pip install "llm-api-logger[gui]"
```

### From source

```bash
git clone https://github.com/vdeshmukh203/llm-api-logger.git
cd llm-api-logger
pip install -e ".[gui]"
```

---

## Quick Start

### Automatic logging with the context manager

```python
import llm_api_logger
import urllib.request, json

with llm_api_logger.session("run.jsonl") as log:
    # Any urllib-based HTTP call to an LLM API is captured automatically.
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello!"}],
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-..."},
    )
    resp = urllib.request.urlopen(req)
    print(json.loads(resp.read()))

# Entries are persisted to run.jsonl after the context exits.
summary = log.summary()
print(f"Calls: {summary['total_calls']}  Cost: ${summary['total_cost_usd']:.4f}")
```

### Manual patch/unpatch

```python
import llm_api_logger

logger = llm_api_logger.LLMLogger(db_path="logs.db", backend="sqlite")
llm_api_logger.patch_urllib(logger)

# ... make API calls ...

llm_api_logger.unpatch_urllib()
print(logger.summary())
```

---

## API Reference

### `LogEntry`

Dataclass representing a single API call.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | UUID4 assigned automatically |
| `url` | `str` | Full request URL |
| `method` | `str` | HTTP method (default `"POST"`) |
| `provider` | `str` | Auto-detected provider name |
| `model` | `str` | Auto-extracted model identifier |
| `request_body` | `str | None` | Raw JSON request body |
| `response_body` | `str | None` | Raw JSON response body |
| `status_code` | `int` | HTTP status code |
| `latency_ms` | `float` | Wall-clock latency in milliseconds |
| `tokens_in` | `int` | Input token count (auto-extracted) |
| `tokens_out` | `int` | Output token count (auto-extracted) |
| `cost_usd` | `float` | Estimated cost in USD (auto-calculated) |
| `timestamp` | `str` | ISO 8601 UTC timestamp |
| `error` | `str | None` | Exception message if the call failed |

### `LLMLogger`

```python
logger = LLMLogger(db_path=":memory:", backend="sqlite")  # or backend="jsonl"

logger.record(entry)           # Store a LogEntry
logger.count()                 # -> int
logger.query(model=None, provider=None, status_code=None, since=None)  # -> List[LogEntry]
logger.summary()               # -> dict with totals and per-model breakdowns
logger.export_jsonl("out.jsonl")
logger.export_csv("out.csv")
```

### `session()`

```python
with llm_api_logger.session(
    log_file="run.jsonl",   # omit for in-memory SQLite
    backend="jsonl",        # "sqlite" | "jsonl"
    auto_patch=True,        # patch urllib automatically
) as logger:
    ...
```

### `estimate_cost(model, tokens_in, tokens_out)`

Returns the estimated USD cost given a model name and token counts.  Raises `ValueError` for unknown models.

```python
cost = llm_api_logger.estimate_cost("gpt-4o", 1000, 500)  # -> float
```

---

## CLI

```
llm-api-logger <command> [log_file] [options]
```

| Command | Description |
|---------|-------------|
| `summary [file]` | Print aggregate statistics |
| `query [file] --model M --provider P --status 200` | Filter and list entries |
| `export [file] -o out.csv --format csv\|jsonl` | Export to file |
| `gui [file] --host 127.0.0.1 --port 5000` | Launch web dashboard |

```bash
# Show a summary of a JSONL log
llm-api-logger summary run.jsonl

# Filter by provider
llm-api-logger query run.jsonl --provider anthropic

# Export to CSV
llm-api-logger export run.jsonl -o report.csv

# Launch the dashboard
llm-api-logger gui run.jsonl
# or equivalently:
llm-api-logger-gui run.jsonl
```

---

## Web Dashboard

Install Flask and launch:

```bash
pip install "llm-api-logger[gui]"
llm-api-logger-gui run.jsonl         # opens http://127.0.0.1:5000
llm-api-logger-gui run.db --port 8080 --no-browser
```

The dashboard provides:
- **KPI cards** — total calls, cost, average latency, total tokens
- **Bar charts** — calls and cost broken down by model
- **Interactive log table** — filter by model, provider, and status code; paginated 25 at a time
- **Entry detail modal** — formatted request/response JSON
- **CSV and JSONL export** buttons

---

## Cost Table

Pricing is sourced from provider documentation and stored in `COST_TABLE` (USD per million tokens).  Versioned model IDs such as `gpt-4o-2024-05-13` are matched via prefix lookup.  To add a custom model:

```python
import llm_api_logger
llm_api_logger.COST_TABLE["my-custom-model"] = {"input": 1.00, "output": 2.00}
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Contributing

Contributions are welcome.  Please open an issue before submitting a pull request for non-trivial changes.

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit your changes with a clear message
4. Open a pull request against `main`

---

## Citation

If you use this software in academic work, please cite:

```bibtex
@software{deshmukh2024llmapilogger,
  author  = {Deshmukh, Vaibhav},
  title   = {llm-api-logger: Middleware library for transparent logging of LLM API calls},
  year    = {2024},
  url     = {https://github.com/vdeshmukh203/llm-api-logger},
  license = {MIT}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
