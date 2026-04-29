---
title: 'llm-api-logger: A Python middleware library for transparent logging and cost tracking of LLM API calls'
tags:
  - Python
  - LLM
  - API logging
  - reproducibility
  - cost tracking
authors:
  - name: Vaibhav Deshmukh
    orcid: 0000-0001-6745-7062
    affiliation: 1
affiliations:
  - name: Independent Researcher, Nagpur, India
    index: 1
date: 29 April 2026
bibliography: paper.bib
---

# Summary

`llm-api-logger` is a Python middleware library that transparently captures and
stores LLM API calls made by Python applications. By replacing
`urllib.request.urlopen`—the underlying HTTP layer used by major LLM client
libraries—with an instrumented wrapper at session start, the library intercepts
all outbound traffic to provider endpoints (OpenAI, Anthropic, Google Gemini,
Mistral, and any OpenAI-compatible server) without requiring any changes to
application code. Each captured interaction is stored as a structured
`LogEntry` record containing the complete request and response payloads,
wall-clock latency, provider-parsed token counts, and per-call cost estimates
derived from a built-in pricing table covering over 20 models across five
provider families. Two pluggable storage backends are provided: SQLite for
structured querying and newline-delimited JSON (JSONL) for stream-friendly
append. A context-manager API enables scoped logging sessions that patch and
restore `urlopen` automatically. A command-line interface supports querying,
filtering, and exporting logs. An interactive Tkinter dashboard is included for
visual exploration of stored records without additional runtime dependencies.

# Statement of Need

Reproducibility in LLM-based research requires capturing not only the final
model output but the exact API request context: model version, sampling
parameters (temperature, top-p), system prompt, tool definitions, and token
usage [@gundersen2018state; @pineau2021improving]. Cost accounting is an
additional practical concern: experiments that iterate over many prompts can
accrue substantial API charges that are difficult to attribute post hoc without
per-call records.

Application-level logging—writing prompts and responses explicitly in user
code—requires invasive instrumentation that must be maintained as the codebase
evolves. SDK-level wrappers tie researchers to a single provider's client
library and must be updated with each SDK release. `llm-api-logger` operates
at the HTTP layer, below any SDK abstraction, so it captures all outbound LLM
traffic uniformly regardless of which provider SDK or raw HTTP client is used,
and without modifying the calling code.

Compared with general-purpose HTTP inspection tools such as mitmproxy
[@mitmproxy], `llm-api-logger` is provider-aware: it extracts token-usage
fields from JSON response bodies according to provider-specific schemas,
computes costs using a bundled pricing table, and stores structured records
optimised for research analysis—including model name, provider, latency, and
estimated USD cost—rather than raw network captures.

# Implementation

The library intercepts HTTP traffic by replacing `urllib.request.urlopen` with
an instrumented wrapper at session start; the original function is restored when
the session closes. The wrapper checks each outbound URL against a set of known
LLM-provider hostnames and, for matching requests, buffers the response body,
parses token-usage fields from the JSON payload according to provider-specific
schemas (OpenAI `usage.prompt_tokens`/`usage.completion_tokens`; Google
`usageMetadata.promptTokenCount`/`usageMetadata.candidatesTokenCount`), and
reconstructs an equivalent response object via `urllib.response.addinfourl` so
that calling code receives unmodified data. All major Python LLM client
libraries (`openai`, `anthropic`, `google-generativeai`) use `urllib` internally
and are therefore captured transparently.

A `LogEntry` dataclass represents each call; its `__post_init__` method derives
provider, model name, token counts, and cost from raw HTTP data so that stored
records are fully self-describing. A companion `LLMLogger` class exposes
`record`, `query`, `summary`, and `export_*` methods over SQLite or JSONL
backends. The `session()` context manager combines logger construction,
patching, and cleanup into a single expression suitable for wrapping experiment
code:

```python
import llm_api_logger as lal

with lal.session("experiment_01.jsonl") as logger:
    response = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )

summary = logger.summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
```

The companion `src/llm_api_logger/` package exposes a `LogStore` class that
extends JSONL persistence with SHA-256 content hashing for tamper-evident
provenance records, and an `LLMAPIProxy` wrapper that couples the patching
mechanism to a `LogStore` instance.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript.
All scientific claims and design decisions are the author's own.

# References
