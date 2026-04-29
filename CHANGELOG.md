# Changelog

All notable changes to llm-api-logger are documented here.
Versioning follows [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-04-29

### Added
- **GUI dashboard** (`llm_api_logger_gui.py`): zero-dependency Tkinter application
  for interactive browsing, sorting, and filtering of log files (JSONL and SQLite);
  includes summary bar, sortable entry table with error highlighting, detail pane,
  and model/provider filters.
- `src/llm_api_logger/store.py`: `LogStore` class — append-only JSONL store with
  per-record SHA-256 content hashes for tamper-evident provenance.
- `src/llm_api_logger/proxy.py`: `LLMAPIProxy` class — wraps the urllib patching
  mechanism and flushes captured entries to a `LogStore` on context exit.
- `_cli` alias for the `main()` function to preserve the `pyproject.toml` entry
  point reference.

### Fixed
- **`_patched_urlopen`**: replaced the invalid `urllib_request.Response(...)` call
  (no such class) with `urllib.response.addinfourl`, so captured responses are
  correctly reconstructed and returned to callers unmodified.
- **`pyproject.toml`**: corrected `[project.scripts]` entry from
  `llm_api_logger:_cli` (which resolved to nothing) to `llm_api_logger:main`.
- **Bare `except:` clauses**: replaced with `except Exception:` throughout
  `_extract_model`, `_tok`, and `_is_llm` to avoid silently swallowing
  `KeyboardInterrupt` and `SystemExit`.
- **`src/llm_api_logger/__init__.py`**: replaced import of non-existent
  `proxy` / `store` modules with imports from the newly created submodules.

### Changed
- `paper.md`: added *Implementation* section; replaced misattributed
  `@gao2023reproducibility` citation with correct reproducibility references
  (`@gundersen2018state`, `@pineau2021improving`); added comparison to mitmproxy.
- `paper.bib`: added `@software{mitmproxy}` entry.
- `README.md`: complete rewrite with installation, quickstart, full API reference,
  CLI usage, supported models table, and `LogStore` provenance example.
- `CITATION.cff`: updated release date to 2026-04-29.

## [0.1.0] - 2024-01-15

### Added
- `LogEntry` dataclass for structured API call tracking.
- `LLMLogger` class with SQLite and JSONL storage backends.
- Cost estimation for 20+ LLM models (OpenAI, Anthropic, Google, Mistral, Meta).
- `urllib.request.urlopen` monkey-patching for automatic, code-free logging.
- `session()` context manager for scoped logging sessions.
- CLI with `summary`, `query`, and `export` sub-commands.
- Unit tests with pytest.
- GitHub Actions CI workflow.
- JOSS paper draft (`paper.md`, `paper.bib`).
- `CITATION.cff` for software citation.
