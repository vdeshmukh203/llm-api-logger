# Changelog

All notable changes to llm-api-logger are documented here.

## [0.1.0] - 2024-01-15

### Added
- Initial release of LLM API Logger
- Middleware hooks for OpenAI, Anthropic, and Cohere Python SDKs
- Request and response capture with wall-clock latency tracking
- Token usage parsing and per-request cost estimation
- Pluggable storage backends: SQLite (default) and newline-delimited JSON
- Configurable sampling rate and PII field masking
- `llmlog` CLI for querying, filtering, and replaying captured logs
- Context manager API for scoped logging sessions
- Unit tests with pytest covering core capture and export paths
- README with quickstart, configuration reference, and backend guide
