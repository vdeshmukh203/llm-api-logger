---
title: 'llm-api-logger: An HTTP proxy for transparent logging of LLM API traffic with structured provenance records'
tags:
  - Python
  - LLM
  - API
  - logging
  - HTTP
  - reproducibility
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

`llm-api-logger` is a lightweight local HTTP proxy that transparently intercepts and logs all traffic between client applications and LLM API endpoints (OpenAI, Anthropic, Google, and any OpenAI-compatible server). By routing requests through the proxy — a one-line environment variable change — researchers capture complete HTTP request and response payloads including headers, request bodies, response streaming chunks, latency breakdown, and token usage statistics. Each interaction is stored as a structured JSONL record with a SHA-256 content hash [@nist2015sha], enabling provenance verification and exact replay.

# Statement of Need

Reproducibility in LLM-based research requires capturing not just the final model output, but the exact API request context: model version specified in the request header, temperature and sampling parameters, system prompt, tool definitions, and any provider-side modifications such as content filtering [@gao2023reproducibility]. Application-level logging frameworks cannot capture provider-side transformations or accurate latency breakdowns. `llm-api-logger` operates at the HTTP layer, below any SDK abstraction, and thus captures the ground truth of what was sent and received. The proxy introduces negligible latency (< 1 ms local overhead) and requires no changes to application code beyond setting the `HTTPS_PROXY` environment variable [@stodden2016enhancing].

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript. All scientific claims and design decisions are the author's own.

# References
