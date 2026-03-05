# GigaVector v0.8.2 Release Notes

**Release Date:** 2026-03-05

A patch release fixing OpenAI embedding integration bugs that caused `EmbeddingService.generate()` to fail on valid API responses.

---

## Embedding Service Fixes

- **JSON response parsing** — `parse_openai_embedding_response()` now tolerates whitespace between `"embedding":` and `[` in the OpenAI API response. Previously required `"embedding":[` with no whitespace, causing valid responses to be rejected as unparseable.
- **JSON request escaping** — Added `json_escape_into()` helper that properly escapes `"`, `\`, `\n`, `\r`, `\t` in input text before building the JSON request payload. Previously, text containing quotes or newlines produced malformed JSON requests that the OpenAI API would reject.

## Version Bump

- Updated version to 0.8.2 across `pyproject.toml`, `__init__.py`, and dashboard server.

---

**Full Changelog:** 1 file changed (src/gv_embedding.c), 33 insertions, 9 deletions
