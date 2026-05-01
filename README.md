# llm-api-logger

[![CI](https://github.com/vdeshmukh203/llm-api-logger/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/llm-api-logger/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

Lightweight middleware for transparently logging and analysing LLM API calls (OpenAI, Anthropic, Google, Mistral, Cohere, Together AI, HuggingFace, and any OpenAI-compatible endpoint).

## Statement of Need

Reproducibility in LLM-based research requires capturing not just model outputs but the exact API request context: model version, temperature, system prompt, tool definitions, and latency. Application-level logging frameworks miss provider-side transformations. `llm-api-logger` operates at the HTTP layer — below any SDK abstraction — and records the ground truth of what was sent and received, along with token usage and cost estimates.

## Features

- Zero-dependency — pure Python standard library
- Automatic capture via `urllib.request.urlopen` monkey-patching
- Structured `LogEntry` dataclass with provider, model, tokens, cost, latency, and error fields
- Pluggable storage: **SQLite** (default) or **JSONL**
- Cost estimation for 26 models across OpenAI, Anthropic, Google, Meta, and Mistral
- Context manager (`session()`) for scoped logging
- CLI: `summary`, `query`, and `export` subcommands
- Graphical dashboard (Tkinter): browse, filter, sort, and export logs

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

## Quick Start

### Context Manager (Recommended)

```python
from llm_api_logger import session
import urllib.request, json

with session("my_run.jsonl") as log:
    # Any urllib-based LLM API call is captured automatically
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}).encode(),
        headers={"Authorization": "Bearer sk-...", "Content-Type": "application/json"},
    )
    urllib.request.urlopen(req)

print(f"Logged {log.count()} call(s)")
summary = log.summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
```

### Manual Patching

```python
from llm_api_logger import LLMLogger, patch_urllib, unpatch_urllib

log = LLMLogger(db_path="logs.db", backend="sqlite")
patch_urllib(log)

# ... make LLM API calls ...

unpatch_urllib()
summary = log.summary()
```

### Programmatic API

```python
from llm_api_logger import LLMLogger, LogEntry

log = LLMLogger(db_path=":memory:", backend="sqlite")

# Record a call manually
entry = LogEntry(
    url="https://api.openai.com/v1/chat/completions",
    request_body='{"model": "gpt-4o", "messages": [...]}',
    response_body='{"usage": {"prompt_tokens": 120, "completion_tokens": 45}}',
    latency_ms=312.5,
    status_code=200,
)
log.record(entry)

# Query and summarise
results = log.query(model="gpt-4o", provider="openai")
stats   = log.summary()
print(stats["total_cost_usd"])
```

## CLI

```
Usage: llm-api-logger COMMAND [OPTIONS] [FILE]

Commands:
  summary   Print aggregate statistics
  query     Filter and list log entries
  export    Export logs to CSV or JSONL

  --version  Show version and exit
  --help     Show help and exit
```

**Examples:**

```bash
# Summarise a JSONL log
llm-api-logger summary my_run.jsonl

# Filter by model
llm-api-logger query my_run.jsonl --model gpt-4o --limit 50

# Export to CSV
llm-api-logger export my_run.jsonl --output report.csv --format csv

# Work with SQLite
llm-api-logger summary logs.db
```

## Graphical Dashboard

```bash
# Open the GUI (requires Python's built-in tkinter)
llm-api-logger-gui

# Open a specific file on startup
llm-api-logger-gui my_run.jsonl
```

The dashboard provides:
- Live summary statistics (calls, cost, tokens, latency)
- Sortable, filterable log table
- Request/response body viewer with JSON pretty-printing
- One-click CSV and JSONL export

## Storage Backends

| Backend | Use case | File extension |
|---------|----------|----------------|
| `sqlite` | Default; supports fast filtering and queries | `.db` / `.sqlite` |
| `jsonl` | Streaming append; human-readable; interoperable | `.jsonl` |

```python
# SQLite (persistent)
log = LLMLogger(db_path="logs.db", backend="sqlite")

# JSONL (in-memory, exported on session exit)
with session("run.jsonl", backend="jsonl") as log:
    ...

# In-memory only (testing / ephemeral analysis)
log = LLMLogger(db_path=":memory:", backend="sqlite")
```

## Supported Models & Pricing

Cost estimates are pre-loaded for 26 models (prices in USD per 1M tokens):

| Provider  | Models |
|-----------|--------|
| OpenAI    | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| Anthropic | claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-2, claude-instant |
| Google    | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash, gemini-pro, palm-2 |
| Meta      | llama-3-70b, llama-3-8b, llama-2-70b, llama-2-13b, llama-2-7b |
| Mistral   | mistral-large, mistral-medium, mistral-small |

```python
from llm_api_logger import estimate_cost, COST_TABLE

cost = estimate_cost("gpt-4o", tokens_in=10_000, tokens_out=2_000)
print(f"${cost:.4f}")

# Add a custom model
COST_TABLE["my-model"] = {"input": 1.00, "output": 3.00}
```

## API Reference

### `LogEntry`

```python
@dataclass
class LogEntry:
    id: str              # UUID (auto-generated)
    url: str
    method: str          # HTTP method (default "POST")
    provider: str        # Inferred from URL
    model: str           # Parsed from request/response body
    request_body: str
    response_body: str
    status_code: int     # HTTP status
    latency_ms: float    # Wall-clock request time
    tokens_in: int       # Prompt tokens (parsed from response)
    tokens_out: int      # Completion tokens
    cost_usd: float      # Estimated cost (auto-calculated)
    timestamp: str       # UTC ISO-8601
    error: str           # Exception message if request failed
```

### `LLMLogger`

```python
class LLMLogger:
    def record(entry: LogEntry) -> None
    def count() -> int
    def query(model=None, provider=None, status_code=None, since=None) -> List[LogEntry]
    def summary() -> Dict[str, Any]
    def export_csv(path: str) -> None
    def export_jsonl(path: str) -> None
    def close() -> None
```

### `session()`

```python
@contextmanager
def session(
    log_file: str | None = None,   # Defaults to "llm_api.jsonl" or ":memory:"
    backend: str = "jsonl",        # "jsonl" or "sqlite"
    auto_patch: bool = True,       # Monkey-patch urllib automatically
) -> LLMLogger: ...
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

## Citation

If you use `llm-api-logger` in research, please cite:

```bibtex
@software{deshmukh2024llmapilogger,
  author  = {Deshmukh, Vaibhav},
  title   = {llm-api-logger: An HTTP proxy for transparent logging of LLM API traffic},
  year    = {2024},
  url     = {https://github.com/vdeshmukh203/llm-api-logger},
  license = {MIT}
}
```

## License

[MIT](LICENSE) © Vaibhav Deshmukh
