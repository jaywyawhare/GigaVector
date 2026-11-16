# Changelog

All notable changes to GigaVector are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/) once
it reaches 1.0. This file was introduced partway through development; for the full
commit-level history before it, see `git log`.

## [Unreleased]

### Security

- SSO/OIDC now verifies TLS certificates by default. The configuration field was
  inverted to `disable_tls_verify` (default `0` = verify) so that a
  zero-initialized `GV_SSOConfig` is secure by default; verification can only be
  turned off by explicitly opting in (local testing).
- Hardened the gRPC wire decoders against integer overflow: payload-length and
  allocation sizes derived from attacker-controlled `dimension`/`count` fields
  are now computed with overflow checks (`gv_wire_span`), failing closed instead
  of under-allocating and over-reading (matters on ILP32 targets).
- The backup-command sink now validates the local file path (`%f`), not just the
  object name (`%o`), before it is substituted into a shell command.
- Documented that the no-OpenSSL software AES-256-CBC fallback is not
  constant-time and must not be used for attacker-reachable ciphertext; prefer
  AES-256-GCM (OpenSSL).

### Added

- Security policy (`SECURITY.md`), Code of Conduct, `CODEOWNERS`, `.editorconfig`,
  and this changelog.
- libFuzzer harnesses for previously unfuzzed decode surfaces: compression
  decode, backup/snapshot restore, the REST request path, and the value-log
  reopen path.
- Brand/theme assets under `assets/` and a theme-switching README banner.

### Changed

- `src/admin/raft.c` now allocates through the project allocator
  (`gv_alloc`/`gv_free`) instead of raw libc, matching the rest of the tree.
- Documentation corrected to the real C API (`db_*` / `graph_*`, `#include
  "gigavector.h"`), fixed include paths and file-layout references, accurate
  contributor workflow, and consistent server port.

### Fixed

- CI `cppcheck` gate unbroken under newer cppcheck (POSIX `WIFEXITED`/
  `WEXITSTATUS` macros; intentional sentinel/arena `intToPointerCast`).

## [0.8.25]

- Baseline release at the time this changelog was introduced. See `git log` for
  the detailed history of the engine, index algorithms, graph/knowledge-graph
  layer, replication, multimodal support, and the custom slab allocator.
