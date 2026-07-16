# GigaVector — Comprehensive Improvement Report

## Bugs (fix immediately)

### 1. Duplicate declarations in `include/storage/database.h` (lines 412–447)
`db_open_with_ivfsq8_config` and `db_ivfsq8_train` are each declared **twice** back-to-back. Generates compiler warnings and could cause linker issues on stricter toolchains.

### 2. Duplicate include in `include/gigavector.h` (lines 47–48)
`#include "index/ivfsq8.h"` appears twice in a row.

### 3. `src/storage/database.c` also double-includes `ivfsq8.h`
Same duplication inside database.c's own `#include` block.

### 4. `strcpy()` used unsafely in 8 places

| File | Line | Risk |
|------|------|------|
| `src/storage/disk_page_cache.c` | 175 | node key copy, no length check |
| `src/index/diskann.c` | 891, 899 | `data_path` copy, no bound |
| `src/multimodal/embedding.c` | 688, 978, 1124 | building JSON strings |
| `src/multimodal/metadata_index.c` | 133, 134 | key/value copy |

Replace with `snprintf(dst, sizeof(dst), "%s", src)` or `strlcpy`.

---

## Drastic Architectural Changes

### 5. `database.c` is a 5,879-line god object — split it

Contains at least 8 nearly-identical constructor functions (`db_open_with_hnsw_config`,
`db_open_with_ivfpq_config`, etc.) each spanning ~80–150 lines that all do:
`malloc` → zero 30 fields → `pthread_rwlock_init` → `metadata_index_create` → configure
one index-specific struct. This is ~1,500 LOC of copy-paste.

**Fix:** Replace the 8 constructors with one:
```c
GV_Database *db_open_with_config(const char *filepath, size_t dimension,
                                  const GV_IndexConfig *cfg);
```
where `GV_IndexConfig` is a tagged union and a shared `db_init_base()` internal helper
initializes the common 30 fields once.

Then split the remaining file into focused modules:
- `db_core.c` — open/close/add/delete/search
- `db_maintenance.c` — compaction, vacuum, WAL replay (`db_compact_*`, `db_replay_wal`)
- `db_stats.c` — stats, health, latency histograms (`db_get_stats`, `db_init_latency_histogram`)
- `db_resource.c` — memory limits, resource checking (`db_check_resource_limits`)
- `db_io.c` — save/load, mmap, header read/write

### 6. IVF variants share ~70% of their code — extract an IVF base

IVF-Flat (737 LOC), IVF-SQ8 (1,008 LOC), IVF-TurboQuant (901 LOC) = **2,646 LOC** with
structurally identical list management, k-means training, and nprobe probing. The only
difference is the quantization applied to stored vectors.

**Fix:** Extract `src/index/ivf_base.c` with:
- `ivf_train_centroids()` — shared k-means
- `ivf_assign_to_list()` — centroid assignment
- `ivf_probe_lists()` — nprobe scanning

Each variant then only implements `ivf_encode_vector()` and `ivf_decode_distance()`.
Cuts ~1,600 LOC of duplication.

### 7. Python bindings use ctypes — migrate to cffi (ABI mode)

`ctypes` is fragile: field offsets, calling conventions, and struct layouts can silently
diverge from the C side. `cffi` in ABI mode parses actual C headers at import time,
guaranteeing correctness. Alternatively, Cython gives both safety and ~3–5x FFI call
speed improvement for tight loops.

### 8. The REST server appears synchronous — add async I/O

`src/api/server.c` uses `libmicrohttpd` which supports select/epoll. High-concurrency
deployments (many simultaneous search requests) would benefit from an epoll-based event
loop so searches can be parallelized across threads without one blocking another. Consider
pairing with a work-stealing thread pool rather than per-request threads.

---

## Important Improvements

### 9. No lock ordering documentation — deadlock risk

With 2,080 `pthread_mutex/rwlock` calls across dozens of modules (storage, replication,
MVCC, WAL, CDC, etc.) and no documented lock hierarchy, deadlocks are a matter of time.
Add `docs/lock_ordering.md` listing the canonical acquisition order (e.g.,
`db->rwlock` → `db->wal_mutex` → index-internal locks) and annotate critical paths.

### 10. No CI coverage gate

CI runs tests and sanitizers but never enforces a minimum coverage threshold. Add
`make test-coverage` as a CI step with `lcov --fail-under-lines 70` to catch regressions.

### 11. No static analysis in CI

Add a `clang-tidy` step to CI (at minimum `-checks=cert-*,clang-analyzer-*`) — it would
have caught the `strcpy` issues automatically. A single job on Linux with a `.clang-tidy`
config is low cost.

### 12. Benchmark results missing from docs

`docs/performance.md` has tuning guidance but zero actual numbers. Add a results table
showing QPS, recall@10, and p99 latency for each index type at 1M vectors (128-dim) — the
benchmark suite already produces this data, just capture and commit it.

### 13. No CPU batch search

`db_search_batch` is declared in `database.h` but `gpu_batch_search` is the real
implementation. For CPU-path multi-query batching (parallelizing N query vectors across
threads), there is nothing. Adding a proper CPU batch search that spawns
`min(qcount, cpu_cores)` parallel searches would massively improve throughput for
embedding pipelines.

### 14. `db_open_with_*` config-ignored bug (`database.c:1711`)

```c
if (index_type != GV_INDEX_TYPE_HNSW || filepath != NULL) {
    return db_open(filepath, dimension, index_type);
}
```

This falls back to `db_open` (ignoring the config!) when a filepath is provided. Every
config-open function likely has the same bug. Persisted databases are never opened with
their custom config — they silently use defaults.

---

## Quick Wins

### 15. Remove duplicate ivfsq8 declarations from `database.h:435–447`
Two-line fix.

### 16. Remove duplicate includes from `gigavector.h:48` and `database.c`
One-line fix each.

### 17. Add stricter compiler warnings
Add `-Wimplicit-fallthrough` and `-Wformat-truncation` to `Makefile`/`CMakeLists.txt` —
catches string truncation and switch fallthrough bugs at compile time.

### 18. Mark key functions with `warn_unused_result`
Add `__attribute__((warn_unused_result))` to `db_add_vector`, `db_search`, `db_save` —
callers that ignore error returns are silent bugs.

### 19. Add C examples to `docs/api_reference.md`
The docs are Python-heavy. The C API is the primary contract; Python is a wrapper. Add C
code examples alongside each function.

### 20. Expand `main.c` into a proper CLI
`main.c` is the only end-to-end integration driver. Expanding it with `--index`, `--dim`,
`--bench` flags lets users try any index type without writing code.

---

## Priority Order

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | Fix duplicate declarations (#2, #3, #15, #16) | 5 min | Prevents compiler issues |
| 2 | Fix `strcpy` (#4) | 1 hr | Security / safety |
| 3 | Fix `db_open_with_*` config-ignored bug (#14) | 2 hr | Correctness |
| 4 | Split `database.c` (#5) | 2 days | Maintainability |
| 5 | IVF base extraction (#6) | 1–2 days | -1,600 LOC duplication |
| 6 | CPU batch search (#13) | 1 day | Throughput |
| 7 | Lock ordering docs (#9) | half day | Safety |
| 8 | CI: coverage gate + clang-tidy (#10, #11) | 2 hr | Quality gate |
| 9 | cffi migration (#7) | 1–2 days | Binding correctness |
| 10 | Benchmark results in docs (#12) | 2 hr | User trust |
