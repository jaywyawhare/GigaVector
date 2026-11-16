# Security Policy

GigaVector is a database engine written in C. It parses untrusted input on
several surfaces (WAL and snapshot/backup files on disk, the REST and gRPC
network protocols, and compressed payloads), so we take memory-safety and
input-validation reports seriously.

## Supported versions

GigaVector is pre-1.0 and ships from `master`. Security fixes land on `master`
and in the next tagged release. Only the latest release and `master` are
supported; please reproduce against one of those before reporting.

| Version        | Supported |
|----------------|-----------|
| `master`       | ✅        |
| latest release | ✅        |
| older releases | ❌        |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's built-in private vulnerability reporting:

1. Go to the repository's **Security** tab → **Report a vulnerability**
   (Security Advisories → "Report a vulnerability").
2. Describe the issue, affected component, and a reproduction (input bytes,
   build flags, platform). A failing test or a crashing corpus file is ideal.

If you cannot use GitHub advisories, open a minimal issue that says only "security
report — please enable private reporting" without technical details, and a
maintainer will follow up with a private channel.

We aim to acknowledge a report within **5 business days** and to agree on a
disclosure timeline from there. Please give us a reasonable window to ship a
fix before any public disclosure.

## Scope

In scope — memory-safety and correctness bugs reachable from untrusted input:

- Out-of-bounds read/write, use-after-free, double-free, or integer overflow in
  any decoder: WAL replay, snapshot/backup restore, compression/decompression,
  the REST and gRPC request paths, and query parsers (SQL/Cypher/filter).
- Authentication/authorization bypass (JWT/SSO/RBAC), TLS verification flaws,
  and cryptographic mistakes in `src/security/`.
- Path traversal or command injection in file/backup/object-store operations.
- Denial of service via a single crafted input (an unbounded allocation or
  crash from one request/file), as distinct from raw volumetric load.

Out of scope:

- Builds configured with `GV_ALLOW_*`/debug/fuzzing flags
  (`FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION`), which intentionally relax checks.
- Vulnerabilities in third-party dependencies (report those upstream).
- Missing hardening that is not exploitable on its own.

## Hardening notes for operators

- **Build with OpenSSL.** Without it, AES-256-GCM is unavailable and the engine
  falls back to a portable software AES-256-CBC whose block cipher is **not
  constant-time**. Do not use the no-OpenSSL build for attacker-reachable
  encryption. CMake auto-detects OpenSSL; for the Makefile pass
  `OPENSSL_FLAGS=-DGV_HAVE_OPENSSL`.
- **Leave TLS verification on.** SSO/OIDC verifies certificates by default; only
  set `disable_tls_verify` for local testing, never in production.
- The test suite runs under ASan, UBSan, TSan, and Valgrind, and continuously
  fuzzes the decode surfaces in CI; regressions in these areas are gating.
