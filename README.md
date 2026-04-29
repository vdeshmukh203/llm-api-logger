# llm-api-logger

[![CI](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

Python middleware for transparent logging and cost tracking of LLM API calls.
Works with OpenAI, Anthropic, Google Gemini, Mistral, and any OpenAI-compatible
endpoint — no changes to application code required.

## Installation

```bash
pip install llm-api-logger
```

Or from source:

```bash
git clone https://github.com/vdeshmukh203/llm-api-logger.git
cd llm-api-logger
pip install -e .
```

## Quickstart

```python
import llm_api_logger as lal

with lal.session("experiment_01.jsonl") as logger:
    # Existing LLM SDK calls go here — nothing else changes.
    pass

summary = logger.summary()
print(f"Total cost:   ${summary['total_cost_usd']:.4f}")
print(f"Total calls:  {summary['total_calls']}")
print(f"Input tokens: {summary['total_tokens_in']:,}")
```

## GUI Dashboard

Launch the interactive Tkinter dashboard to browse, filter, and inspect logs:

```bash
python llm_api_logger_gui.py experiment_01.jsonl
```

The dashboard provides:
- Summary bar (total calls, cost, tokens, average latency, error count)
- Sortable, filterable entry table with alternating-row styling
- Detail pane showing full request body and all metadata for the selected row
- Model and provider drop-down filters

## API Reference

### `session(log_file, backend, auto_patch)`

Context manager that creates an `LLMLogger`, patches `urllib.request.urlopen`,
and restores the original on exit.

```python
with lal.session("run.jsonl", backend="jsonl") as logger:
    ...
```

| Parameter    | Default              | Description                                  |
|--------------|----------------------|----------------------------------------------|
| `log_file`   | `"llm_api.jsonl"`    | Storage path; `":memory:"` for in-process    |
| `backend`    | `"jsonl"`            | `"jsonl"` or `"sqlite"`                      |
| `auto_patch` | `True`               | Patch/unpatch `urlopen` automatically        |

### `LLMLogger`

```python
logger = lal.LLMLogger(db_path="logs.db", backend="sqlite")
logger.record(entry)                          # store a LogEntry
entries = logger.query(model="gpt-4o")        # filter by model / provider / status
stats   = logger.summary()                    # aggregate statistics dict
logger.export_csv("report.csv")
logger.export_jsonl("dump.jsonl")
```

### `LogEntry`

Dataclass for a single captured API call. All fields are auto-populated from
the HTTP request/response; only `url` is mandatory when constructing manually.

| Field          | Type            | Description                                        |
|----------------|-----------------|----------------------------------------------------|
| `id`           | `str`           | UUID                                               |
| `url`          | `str`           | Full request URL                                   |
| `provider`     | `str`           | Detected provider (`openai`, `anthropic`, …)       |
| `model`        | `str`           | Model name extracted from request/response JSON    |
| `tokens_in`    | `int`           | Input (prompt) token count                         |
| `tokens_out`   | `int`           | Output (completion) token count                    |
| `cost_usd`     | `float`         | Estimated USD cost from built-in pricing table     |
| `latency_ms`   | `float`         | Wall-clock latency in milliseconds                 |
| `timestamp`    | `str`           | UTC ISO-8601 timestamp                             |
| `status_code`  | `int`           | HTTP response status code                          |
| `error`        | `str \| None`   | Exception message if the call failed               |
| `request_body` | `str \| None`   | Raw JSON request body                              |
| `response_body`| `str \| None`   | Raw JSON response body                             |

### `estimate_cost(model, tokens_in, tokens_out)`

Returns the estimated USD cost for a single call. Raises `ValueError` for
unknown models.

### `patch_urllib(logger)` / `unpatch_urllib()`

Low-level functions for manual control of the monkey-patch.

## CLI

```bash
# Summarise a log file
llm-api-logger summary experiment_01.jsonl

# Query with filters
llm-api-logger query experiment_01.jsonl --model gpt-4o --provider openai

# Export to CSV
llm-api-logger export experiment_01.jsonl --output report.csv --format csv

# Export to JSONL
llm-api-logger export logs.db --output dump.jsonl --format jsonl
```

## Supported Models

Cost estimation is built-in for the following families:

| Family     | Models                                                  |
|------------|---------------------------------------------------------|
| OpenAI     | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| Anthropic  | claude-3-5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-2.1, claude-2, claude-instant |
| Google     | gemini-pro, gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, palm-2 |
| Mistral    | mistral-large, mistral-medium, mistral-small            |
| Meta       | llama-2-7b, llama-2-13b, llama-2-70b, llama-3-8b, llama-3-70b |

## SHA-256 Provenance (`LogStore`)

The `src/llm_api_logger` package provides a `LogStore` class that stamps each
JSONL record with a `_sha256` field for tamper-evident provenance:

```python
from llm_api_logger.store import LogStore

store = LogStore("run.jsonl")
digest = store.append({"model": "gpt-4o", "tokens_in": 42, "cost_usd": 0.001})
assert store.verify(store.load()[-1])
```

## Citation

If you use this software in research, please cite it:

```bibtex
@software{deshmukh2026llmapilogger,
  author  = {Deshmukh, V.A.},
  title   = {llm-api-logger},
  url     = {https://github.com/vdeshmukh203/llm-api-logger},
  year    = {2026},
  license = {MIT}
}
```

## License

[MIT](LICENSE)
