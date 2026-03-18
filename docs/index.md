# GigaVector

High-performance vector database written in C with Python bindings. Zero external dependencies. One library. Everything you need.

```bash
pip install gigavector
```

---

## Quick Start

```python
from gigavector import Database, DistanceType, IndexType

with Database.open("example.db", dimension=128, index=IndexType.HNSW) as db:
    db.add_vector([0.1] * 128, metadata={"category": "example"})

    results = db.search([0.1] * 128, k=10, distance=DistanceType.COSINE)
    for hit in results:
        print(f"  distance={hit.distance:.4f}")

    db.save("example.db")
```

---

## At a Glance

| | |
|---|---|
| **8** Index Types | HNSW, IVF-PQ, IVF-Flat, DiskANN, Flat, KD-Tree, LSH, Sparse |
| **5** Distance Metrics | Euclidean, Cosine, Dot Product, Manhattan, Hamming |
| **0** Dependencies | Pure C core, no external libraries |
| **SIMD** Optimized | SSE4.2, AVX2, AVX-512F, optional CUDA GPU |

---

## Capabilities

### Search
k-NN, range, batch, filtered, hybrid BM25, scroll, grouped, geo-spatial, ColBERT, recommendations, SQL queries, phased ranking

### Storage
WAL with crash recovery, point-in-time snapshots, mmap I/O, incremental backup, JSON import/export, background compaction

### Data Management
Rich metadata, payload indexing, schema evolution, upsert, TTL, named vectors, collection aliases, CDC streams, time-travel queries

### Distributed
HTTP REST, gRPC, TLS 1.3, hash/range sharding, leader-follower replication, cluster management, multi-tenancy, streaming ingestion

### Graph
Property graph with BFS/DFS/Dijkstra, PageRank, knowledge graph with embeddings, entity resolution, link prediction

### AI Integration
OpenAI, Anthropic, Gemini integrations, auto-embedding, semantic memory, ONNX model serving, agentic interfaces

---

## Index Algorithms

| Index | Type | Training | Best For |
|-------|------|----------|----------|
| **HNSW** | Approximate | No | General-purpose, high recall |
| **IVF-PQ** | Approximate | Yes | Large-scale, memory-efficient |
| **IVF-Flat** | Approximate | Yes | Large-scale, higher accuracy |
| **KD-Tree** | Exact | No | Low-dimensional data (< 20D) |
| **LSH** | Approximate | No | Fast hash-based search |
| **PQ** | Approximate | Yes | Compressed-domain search |
| **Flat** | Exact | No | Small datasets, ground-truth |
| **Sparse** | Exact | No | Sparse vectors (NLP, BoW) |

Use `suggest_index()` for automatic selection based on your data characteristics.

---

## Explore the Docs

- **[Usage Guide](/docs/usage)** Get up and running
- **[Python Bindings](/docs/python_bindings)** Full Python API docs
- **[C API Guide](/docs/c_api_guide)** C usage patterns
- **[API Reference](/docs/api_reference)** Complete reference
- **[Architecture](/docs/architecture)** System design & internals
- **[Performance Tuning](/docs/performance)** Index selection & optimization
- **[Deployment](/docs/deployment)** Production scaling
- **[Security](/docs/security)** Auth, RBAC & best practices
- **[Advanced Features](/docs/advanced_features)** Advanced patterns & examples
