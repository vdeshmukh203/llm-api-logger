---
title: 'llm-api-logger: A middleware library for transparent logging and cost tracking of LLM API calls'
tags:
  - Python
  - LLM
  - API
  - logging
  - reproducibility
  - cost tracking
authors:
  - name: Vaibhav Deshmukh
    orcid: 0000-0001-6745-7062
    affiliation: 1
affiliations:
  - name: Independent Researcher, Nagpur, India
    index: 1
date: 23 April 2026
bibliography: paper.bib

---

# Summary

`llm-api-logger` is a Python middleware library that transparently intercepts and logs API interactions with large language model (LLM) providers such as OpenAI, Anthropic, Google, Mistral, Cohere, and Together AI.  The library operates by monkey-patching `urllib.request.urlopen`, the underlying HTTP function used by many Python HTTP libraries, so that every outbound LLM API request and its corresponding response are captured without any modification to application code.

Each captured interaction is stored as a structured `LogEntry` record containing the full request and response JSON bodies, HTTP status code, wall-clock latency, automatically extracted token counts, and an estimated cost computed from a built-in pricing table covering more than 30 models.  Entries are persisted to either an SQLite database or a newline-delimited JSON (JSONL) file.  A context manager API enables scoped logging sessions, and a command-line interface (CLI) supports querying, summarising, and exporting log data.  An optional Flask-based web dashboard provides an interactive view of summary statistics and individual log entries.

# Statement of Need

Reproducibility in LLM-based research requires capturing not just the final model output but the complete API request context: the exact model version, temperature and sampling parameters, system prompt, and tool definitions [@gao2023reproducibility].  Without systematic logging, cost and token usage can be difficult to audit, and exact replication of experiments becomes infeasible [@stodden2016enhancing].

`llm-api-logger` addresses this need by inserting a logging layer below SDK abstractions.  Because it intercepts at the `urllib` level, it captures the actual bytes exchanged with the provider rather than relying on SDK-level instrumentation, which may omit retries, provider-side modifications, or intermediary transformations.  The library adds negligible overhead (a single in-process function call and a lightweight database write per request) and requires no changes to application code.  The structured storage format enables downstream cost analysis, reproducibility auditing, and integration with data analysis workflows.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript. All scientific claims and design decisions are the author's own.

# References
