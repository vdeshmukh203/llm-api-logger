# llm-api-logger

[![CI](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A lightweight Python library for logging, analysing, and exporting LLM API calls.
Supports OpenAI, Anthropic, Google Gemini, Mistral, Cohere, Together AI, and any
OpenAI-compatible endpoint.

Features:
- **Zero-dependency** — stdlib only (`sqlite3`, `csv`, `json`, `tkinter`)
- **Two backends** — JSONL (append-only) or SQLite
- **SHA-256 provenance hashing** via `LogStore` for tamper-evident records
- **HTTP proxy** (`LLMAPIProxy`) that captures calls without changing application code
- **urllib monkey-patching** (`patch_urllib`) for in-process capture
- **Cost estimation** for 25+ models
- **Tkinter GUI dashboard** — sortable table, filters, export
- **CLI** — `summary`, `query`, `export` sub-commands

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

Requires Python 3.8 or later.  No external runtime dependencies.

---

## Quick Start

### 1. In-process capture (urllib monkey-patch)

```python
import llm_api_logger as lal

with lal.session(log_file="calls.jsonl") as logger:
    # Make LLM API calls as normal — they are logged automatically
    import urllib.request, json
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer YOUR_KEY"},
        method="POST",
    )
    urllib.request.urlopen(req)

print(logger.summary())
```

### 2. HTTP proxy capture (no code changes required)

```bash
# Terminal 1 — start the proxy
python -c "
from llm_api_logger import LLMAPIProxy, LogStore
store = LogStore('calls.jsonl')
with LLMAPIProxy(store, port=8080) as proxy:
    input('Proxy running on :8080 — press Enter to stop')
"

# Terminal 2 — route your LLM SDK through the proxy
export http_proxy=http://localhost:8080
python your_llm_script.py
```

### 3. Direct API

```python
from llm_api_logger import LLMLogger, LogEntry

logger = LLMLogger(db_path=":memory:", backend="sqlite")
logger.record(LogEntry(
    url="https://api.openai.com/v1/chat/completions",
    model="gpt-4o",
    tokens_in=500,
    tokens_out=200,
    latency_ms=834.2,
))
print(logger.summary())
```

---

## CLI

```
llm-api-logger summary  [log_file]              # aggregate statistics
llm-api-logger query    [log_file] [--model M] [--provider P]
llm-api-logger export   [log_file] -o out.csv [--format csv|jsonl]
llm-api-logger gui      [log_file]              # launch Tkinter dashboard
```

Examples:

```bash
llm-api-logger summary calls.jsonl
llm-api-logger query   calls.jsonl --model gpt-4o
llm-api-logger export  calls.jsonl -o report.csv
llm-api-logger gui     calls.jsonl
```

---

## GUI Dashboard

```bash
llm-api-logger-gui [path/to/log.jsonl]
```

The dashboard shows a sortable, filterable table of log entries.
Double-click any row to inspect the full request and response bodies.

**Features:**
- Filter by model, provider, and HTTP status code
- Click column headings to sort
- Export filtered results to CSV or JSONL
- Summary statistics bar (total cost, tokens, average latency)

---

## SHA-256 Provenance (LogStore)

`LogStore` attaches a SHA-256 digest to every record at capture time.
Verify integrity later:

```python
from llm_api_logger import LogStore

store = LogStore("calls.jsonl")
results = store.verify_all()  # {record_id: True/False}
tampered = [rid for rid, ok in results.items() if not ok]
if tampered:
    print(f"WARNING: {len(tampered)} records failed verification")
```

---

## Cost Estimation

```python
from llm_api_logger import estimate_cost, COST_TABLE

cost = estimate_cost("gpt-4o", tokens_in=10_000, tokens_out=500)
print(f"${cost:.4f}")

# List all supported models
for model in sorted(COST_TABLE):
    print(model)
```

Supported models include: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `claude-3-5-sonnet`,
`claude-3-opus`, `claude-3-haiku`, `gemini-1.5-pro`, `gemini-2.0-flash`, `mistral-large`,
and 15+ others.

---

## API Reference

### `LLMLogger`

| Method | Description |
|---|---|
| `record(entry)` | Persist a `LogEntry` |
| `query(model, provider, status_code, since)` | Filtered list of entries |
| `summary()` | Aggregate statistics dict |
| `count()` | Total number of stored entries |
| `export_csv(path)` | Write CSV |
| `export_jsonl(path)` | Write JSONL |

### `LogStore`

| Method | Description |
|---|---|
| `append(record)` | Store a `Record`, return its SHA-256 hash |
| `load(limit)` | Load records, most-recent first |
| `count()` | Total stored records |
| `verify_all()` | Dict of `{id: bool}` hash validity |

### `LLMAPIProxy`

```python
proxy = LLMAPIProxy(store, host="localhost", port=8080)
proxy.start()   # starts background thread
proxy.stop()    # shuts down

# or use as a context manager:
with LLMAPIProxy(store, port=8080) as proxy:
    ...
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Citation

If you use this software in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff) or as:

> Deshmukh, V. A. (2026). *llm-api-logger: An HTTP proxy for transparent logging of
> LLM API traffic with structured provenance records*. Journal of Open Source Software.

---

## License

MIT — see [LICENSE](LICENSE).
