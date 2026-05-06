# llm-api-logger

A lightweight Python library for transparently logging, analysing, and costing
LLM API calls — with zero changes to your application code.

[![CI](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## Overview

`llm-api-logger` intercepts outgoing HTTP requests to LLM provider APIs
(OpenAI, Anthropic, Google, Mistral, …) by monkey-patching
`urllib.request.urlopen`.  Every call is recorded as a structured
[`LogEntry`](#logentry) object containing:

- full request and response payloads
- HTTP status code and round-trip latency
- token counts (parsed from the provider's usage metadata)
- estimated cost in USD (using a built-in pricing table covering 25+ models)

Logs are stored in SQLite (default) or JSONL format and can be exported to
CSV for downstream analysis.  A browser-based dashboard is included for
interactive exploration.

---

## Installation

```bash
pip install llm-api-logger
```

No external runtime dependencies — only the Python standard library is required.

---

## Quick start

### Context manager (recommended)

```python
import llm_api_logger as lal

with lal.session("my_run.jsonl", backend="jsonl") as logger:
    # make LLM API calls here — they are logged automatically
    import urllib.request, json
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}).encode(),
        headers={"Authorization": "Bearer $OPENAI_API_KEY", "Content-Type": "application/json"},
    )
    urllib.request.urlopen(req)

summary = logger.summary()
print(f"Calls: {summary['total_calls']}, Cost: ${summary['total_cost_usd']:.4f}")
```

### Manual patching

```python
import llm_api_logger as lal

logger = lal.LLMLogger(db_path="logs.db", backend="sqlite")
lal.patch_urllib(logger)

# … run your application …

lal.unpatch_urllib()
print(logger.summary())
```

### Proxy object

```python
from src.llm_api_logger.proxy import LLMAPIProxy

with LLMAPIProxy(db_path=":memory:") as proxy:
    # … LLM calls are intercepted and stored in proxy.logger …
    pass
```

---

## CLI reference

```
llm-api-logger <command> [options]
```

### `summary`

Print aggregate statistics for a log file.

```bash
llm-api-logger summary my_run.jsonl
```

```
==============================================================
  LLM API CALL SUMMARY
==============================================================
  Total API Calls  : 42
  Total Cost (USD) : $0.3821
  Input Tokens     : 1,204,300
  Output Tokens    :   312,100
  Avg Latency (ms) : 843.2

  Model                          Calls        Cost
  --------------------------------------------------
  gpt-4o                            30  $   0.3200
  gpt-4o-mini                       12  $   0.0621
==============================================================
```

### `query`

List individual log entries, with optional filtering.

```bash
llm-api-logger query my_run.jsonl --model gpt-4o --limit 5
```

### `export`

Export a log file to CSV or JSONL.

```bash
llm-api-logger export my_run.jsonl --output report.csv --format csv
llm-api-logger export my_run.db   --output report.jsonl --format jsonl
```

### `gui`

Launch the interactive browser dashboard.

```bash
llm-api-logger gui my_run.jsonl
```

The dashboard opens automatically in your default browser.  It provides:

- aggregate summary cards (calls, cost, tokens, latency)
- a filterable, sortable log table
- per-entry request/response inspection with pretty-printed JSON
- CSV export

---

## API reference

### `LogEntry`

```python
@dataclass
class LogEntry:
    id:            str    # UUID4, auto-generated
    url:           str
    method:        str    # typically "POST"
    provider:      str    # auto-detected from URL
    model:         str    # extracted from JSON body
    request_body:  Optional[str]
    response_body: Optional[str]
    status_code:   int
    latency_ms:    float
    tokens_in:     int
    tokens_out:    int
    cost_usd:      float  # auto-computed from COST_TABLE
    timestamp:     str    # ISO-8601 UTC
    error:         Optional[str]
```

`LogEntry` auto-detects `provider`, `model`, token counts, and cost during
`__post_init__`, so in most cases you only need to supply `url`,
`request_body`, `response_body`, `status_code`, and `latency_ms`.

### `LLMLogger`

```python
logger = LLMLogger(db_path=":memory:", backend="sqlite")
# backend="jsonl" keeps entries in memory; flush with logger.export_jsonl(path)

logger.record(entry)          # persist a LogEntry
logger.count()                # total number of entries
logger.query(                 # filter and retrieve entries (newest first)
    model="gpt-4o",
    provider="openai",
    status_code=200,
    since="2025-01-01T00:00:00",
)
logger.summary()              # aggregate dict
logger.export_csv(path)       # write CSV
logger.export_jsonl(path)     # write JSONL
logger.close()                # release SQLite connection
```

### `estimate_cost`

```python
cost = estimate_cost("gpt-4o", tokens_in=1_000, tokens_out=500)
```

Returns USD cost based on `COST_TABLE`.  Raises `ValueError` for unknown
model names.

### `session`

```python
with session(log_file="run.jsonl", backend="jsonl", auto_patch=True) as logger:
    ...
```

`auto_patch=True` (default) monkey-patches `urllib.request.urlopen` for the
duration of the `with` block and restores it on exit.

---

## Supported models and providers

| Provider   | Models                                                      |
|------------|-------------------------------------------------------------|
| OpenAI     | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo    |
| Anthropic  | claude-3-5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-2.x, claude-instant |
| Google     | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, gemini-pro, palm-2 |
| Meta       | llama-2 (7B/13B/70B), llama-3 (8B/70B)                     |
| Mistral    | mistral-large, mistral-medium, mistral-small                |

Cost estimates reflect public list prices as of mid-2025 (USD per million
tokens).  Extend `COST_TABLE` at runtime for additional or custom models.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite exercises provider detection, model extraction, token parsing,
cost estimation, both storage backends, CSV/JSONL export, the session context
manager, the proxy object, and all three CLI subcommands.

---

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/vdeshmukh203/llm-api-logger).

---

## Citation

If you use `llm-api-logger` in published research, please cite the
accompanying JOSS paper (see `paper.md` and `CITATION.cff`).

---

## License

MIT — see [LICENSE](LICENSE).
