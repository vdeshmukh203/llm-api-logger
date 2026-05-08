# llm-api-logger

[![CI](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

Middleware logger for LLM API calls. Intercepts `urllib.request` traffic to
OpenAI, Anthropic, Google, Mistral, Cohere, Together, and HuggingFace
endpoints, recording request/response pairs with token counts, latency, and
cost estimates to SQLite or JSONL storage — with no changes to application code.

---

## Features

- **Automatic interception** of LLM API calls via `urllib.request` monkey-patching
- **Provider detection** for OpenAI, Anthropic, Google Gemini, Mistral, Cohere, Together, HuggingFace
- **Token parsing** for OpenAI, Anthropic, and Google response formats
- **Cost estimation** for 25+ models using per-million-token pricing
- **Dual storage backends**: SQLite (persistent) and JSONL (streaming/files)
- **Query & filter** by model, provider, status code, or time range
- **Export** to CSV or JSONL
- **Context manager API** for scoped, auto-exporting sessions
- **CLI** (`llm-api-logger`) for querying and exporting log files
- **GUI** (`llm-api-logger-gui`) for interactive log exploration

---

## Installation

```bash
pip install llm-api-logger
```

Or install from source:

```bash
git clone https://github.com/vdeshmukh203/llm-api-logger.git
cd llm-api-logger
pip install -e .
```

---

## Quick Start

### Context manager (recommended)

```python
import llm_api_logger as lal

with lal.session("my_session.jsonl") as logger:
    # Any urllib.request calls to LLM APIs are automatically captured
    import urllib.request, json
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}).encode(),
        headers={"Authorization": "Bearer sk-...", "Content-Type": "application/json"},
    )
    urllib.request.urlopen(req)

print(logger.summary())
```

### Manual logging

```python
from llm_api_logger import LLMLogger, LogEntry

logger = LLMLogger(db_path="calls.db", backend="sqlite")
entry = LogEntry(
    url="https://api.openai.com/v1/chat/completions",
    request_body='{"model": "gpt-4o", "messages": [...]}',
    response_body='{"usage": {"prompt_tokens": 100, "completion_tokens": 50}}',
    latency_ms=312.5,
)
logger.record(entry)
print(logger.summary())
```

### Cost estimation

```python
from llm_api_logger import estimate_cost

cost = estimate_cost("gpt-4o", tokens_in=500, tokens_out=200)
print(f"Estimated cost: ${cost:.6f}")
```

---

## Storage Backends

| Backend | `backend=` | Persistence | Best for |
|---------|-----------|-------------|----------|
| SQLite  | `"sqlite"` | Yes (file or `:memory:`) | Long-running sessions, structured queries |
| JSONL   | `"jsonl"`  | Yes (file) or in-memory | Streaming, append-only, simple tooling |

```python
# SQLite on disk
logger = LLMLogger(db_path="logs.db", backend="sqlite")

# JSONL in memory
logger = LLMLogger(backend="jsonl")

# JSONL file (auto-loaded by session())
with lal.session("logs.jsonl", backend="jsonl") as logger:
    ...
```

---

## Querying Logs

```python
# All entries
entries = logger.query()

# Filter by model
entries = logger.query(model="gpt-4o")

# Filter by provider
entries = logger.query(provider="anthropic")

# Filter by HTTP status
errors = logger.query(status_code=500)

# Filter by timestamp (ISO 8601)
recent = logger.query(since="2024-01-01T00:00:00")
```

---

## Summary Statistics

```python
s = logger.summary()
print(f"Total calls:   {s['total_calls']}")
print(f"Total cost:    ${s['total_cost_usd']:.4f}")
print(f"Avg latency:   {s['avg_latency_ms']:.1f} ms")
print(f"Calls by model: {s['calls_by_model']}")
```

---

## Export

```python
logger.export_csv("report.csv")
logger.export_jsonl("archive.jsonl")
```

---

## CLI

```bash
# Show summary of a log file
llm-api-logger summary logs.jsonl

# Query with filters
llm-api-logger query logs.jsonl --model gpt-4o --provider openai

# Export to CSV
llm-api-logger export logs.jsonl --output report.csv --format csv
```

---

## GUI

Launch the interactive log viewer:

```bash
llm-api-logger-gui                  # open file dialog
llm-api-logger-gui logs.jsonl       # open specific file
```

The GUI provides:
- **Summary tab** — total calls, cost, tokens, average latency, per-model breakdown
- **Log Entries tab** — sortable table with model/provider filters and per-entry detail pane
- **Charts tab** — horizontal bar charts for cost, call count, or average latency by model
- File → Export to CSV or JSONL

---

## Supported Models and Pricing

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|--------------------|--------------------|
| gpt-4o | 5.00 | 15.00 |
| gpt-4o-mini | 0.15 | 0.60 |
| gpt-4-turbo | 10.00 | 30.00 |
| claude-3-5-sonnet | 3.00 | 15.00 |
| claude-3-opus | 15.00 | 75.00 |
| claude-3-haiku | 0.25 | 1.25 |
| gemini-1.5-pro | 1.25 | 5.00 |
| gemini-1.5-flash | 0.075 | 0.30 |
| mistral-large | 2.00 | 6.00 |
| llama-3-70b | 0.50 | 1.00 |

Full list of 25+ models available in `llm_api_logger.COST_TABLE`.

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

---

## Citation

If you use llm-api-logger in research, please cite:

```bibtex
@software{deshmukh_llm_api_logger,
  author = {Deshmukh, Vaibhav},
  title  = {{llm-api-logger}: Middleware HTTP logger for LLM API calls},
  url    = {https://github.com/vdeshmukh203/llm-api-logger},
  license = {MIT},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
