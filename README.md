# GigaVector

<p align="center">
  <img src="https://raw.githubusercontent.com/jaywyawhare/GigaVector/master/docs/gigavector-logo.png" alt="GigaVector Logo" width="200" />
</p>

<p align="center">
  <a href="https://pepy.tech/projects/gigavector">
    <img src="https://static.pepy.tech/personalized-badge/gigavector?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" />
  </a>
</p>

**GigaVector** is a vector database library written in C with Python bindings. It
combines ANN vector search, a property graph and knowledge graph, a memory layer,
and document ingestion behind one engine, with REST/gRPC servers, replication, and
security built in.

The full, itemized feature list lives in [FEATURE_SPEC.md](FEATURE_SPEC.md) and the
[docs](#documentation); this README is the short tour.

## Index Algorithms

| Index | Type | Training | Best For |
|-------|------|----------|----------|
| KD-Tree | Exact | No | Low-dimensional data (< 20D) |
| HNSW | Approximate | No | General-purpose, high recall |
| IVF-PQ | Approximate | Yes | Large-scale, memory-efficient |
| IVF-Flat | Approximate | Yes | Large-scale, higher accuracy than IVF-PQ |
| IVF-SQ8 | Approximate | Yes | Large-scale IVF with 8-bit scalar quant |
| IVF-TurboQuant | Approximate | Yes | Large-scale IVF with PolarQuant, no codebook training |
| DiskANN | Approximate | Yes | Billion-scale graph on disk with page cache |
| IVFDisk | Approximate | Yes | Larger-than-RAM IVF head + disk posting lists |
| Flat | Exact (brute-force) | No | Small datasets, baseline/ground-truth |
| PQ | Approximate | Yes | Compressed-domain search |
| LSH | Approximate | No | Fast hash-based approximate search |
| Sparse | Exact | No | Sparse vectors (NLP, BoW) |

Call `suggest_index()` / `gv_index_suggest()` for automatic selection; pass
`max_memory_bytes` to prefer on-disk DiskANN or IVFDisk when the dataset exceeds
~70% of RAM (see [larger_than_ram_plan.md](docs/larger_than_ram_plan.md)). Distance
metrics: Euclidean, Cosine, Dot Product, Manhattan, Hamming, all SIMD-optimized
(SSE4.2, AVX2, AVX-512F, FMA).

## Capabilities

- **Search** - k-NN, range, batch, filtered (metadata pre/post), hybrid (BM25 + vector fusion), SQL (`SELECT ... ORDER BY vector_distance(...)`), phased ranking, MMR diversity, grouped, geo-spatial, ColBERT late interaction, recommendations.
- **Storage and durability** - crash-safe WAL with replay and size-based checkpointing, atomic snapshots, point-in-time snapshots, collection versioning, hot/warm/cold tiering, WiscKey value-log.
- **Transactions** - MVCC snapshot isolation, crash-atomic commit (one WAL record per transaction), transactions over REST (`/txn/*`), automatic tombstone GC.
- **Quantization** - PQ/OPQ, scalar (SQ8), TurboQuant/PolarQuant, binary, RaBitQ, 1.5/2/4/8-bit.
- **Distributed** - embedded REST and gRPC servers, TLS, Raft replication (durable log, failover, online resharding), sharding, namespaces/multi-tenancy, per-tenant quotas, streaming ingestion.
- **Security** - API-key and JWT auth, per-key RBAC scopes (read-write / read-only), read-only server mode, OIDC/SSO, AES-wrapped backups.
- **Graph and knowledge graph** - property graph with traversal/analytics, knowledge graph with entity embeddings, a broad Cypher subset, GraphRAG context expansion, graph WAL + transactions.
- **AI integration** - LLM (OpenAI/Anthropic/Gemini), embedding services + auto-embedding, semantic memory layer, ONNX model serving, document ingestion (Markdown/HTML/PDF).
- **Observability** - Prometheus `/metrics`, structured JSON access logs, query tracing, health checks, latency histograms.
- **Acceleration** - SIMD distance kernels, optional CUDA GPU with multi-GPU batch fan-out.

## Build

### Make (default)
```bash
make lib         # static + shared libraries -> build/lib/
make c-test      # run all C tests
make python-test # run Python test suite
```
After changing C code, run `make lib` before using the Python bindings from a local
build: Python loads `build/libGigaVector.so`, which must match `build/lib/libGigaVector.so`.

### CMake
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest
```
Options: `-DBUILD_SHARED_LIBS`, `-DBUILD_TESTS`, `-DBUILD_BENCHMARKS`,
`-DENABLE_SANITIZERS`, `-DENABLE_COVERAGE`, `-DENABLE_NATIVE_OPTIMIZATIONS`.

### Sanitizers and coverage
```bash
make test-asan test-tsan test-ubsan test-valgrind test-coverage   # or: make test-all
```

## Quick Start (Python)

```python
from gigavector import Database, DistanceType, IndexType

with Database.open("example.db", dimension=128, index=IndexType.HNSW) as db:
    db.add_vector([0.1] * 128, metadata={"category": "example"})
    results = db.search([0.1] * 128, k=10, distance=DistanceType.COSINE)
    for hit in results:
        print(f"distance={hit.distance:.4f}")
    db.save("example.db")
```

See the [Usage Guide](docs/usage.md) and [Python Bindings Guide](docs/python_bindings.md)
for index configuration, transactions, graph/KG, geo, recommendations, and more; the
[C API Guide](docs/c_api_guide.md) covers the C contract.

## REST API

```python
from gigavector import Server, ServerConfig

server = Server(db, ServerConfig(port=6969, enable_cors=True))
server.start()
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Database statistics |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/vectors` | Add vector(s) |
| `GET` `PUT` `DELETE` | `/vectors/{id}` | Get / update / delete by index |
| `POST` | `/search`, `/search/range`, `/search/batch` | k-NN, range, batch search |
| `POST` | `/txn/begin`, `/txn/commit`, `/txn/rollback` | Transactions |
| `POST` | `/compact`, `/save` | Compaction, persist to disk |

A built-in dark-theme web dashboard is served at `/dashboard`
(`serve_with_dashboard(db, port=6969)`; pure Python, no libmicrohttpd required).

Client libraries live under `clients/`: Python (CFFI), Go (cgo, embedded),
JavaScript (REST), and a dependency-free Rust REST SDK with connection pooling.

## Environment Variables

```bash
cp .env.example .env   # copy and edit with your keys
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For LLM/embedding tests | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic tests | Anthropic/Claude API key |
| `GOOGLE_API_KEY` | Optional | Google Gemini (`GEMINI_API_KEY` alias accepted) |
| `GV_WAL_DIR` | Optional | Override WAL directory |

## Documentation

- [Usage Guide](docs/usage.md) and [Python Bindings](docs/python_bindings.md)
- [C API Guide](docs/c_api_guide.md) and [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md) and [Performance Tuning](docs/performance.md)
- [Deployment](docs/deployment.md), [Security](docs/security.md), [Troubleshooting](docs/troubleshooting.md)
- [Build and Test](docs/build_and_test.md) and [Contributing](CONTRIBUTING.md)

## License

Licensed under the DBaJ-NC-CFL [License](./LICENCE).
