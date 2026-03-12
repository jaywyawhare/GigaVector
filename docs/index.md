---
hide:
  - navigation
  - toc
---

<div class="landing">

# GigaVector

<div class="gv-hero">
  <img src="gigavector-logo.png" alt="GigaVector" width="120">
  <div class="gv-hero-title">GigaVector</div>
  <p class="gv-hero-sub">
    High-performance vector database written in C with Python bindings.
    Zero external dependencies. One library. Everything you need.
  </p>
  <div class="gv-buttons">
    <a href="usage/" class="gv-btn-primary">Get Started</a>
    <a href="https://github.com/jaywyawhare/GigaVector" class="gv-btn-outline">GitHub</a>
  </div>
  <div class="gv-install" markdown>

```bash
pip install gigavector
```

  </div>
</div>

<div class="gv-stats">
  <div class="gv-stat">
    <span class="gv-stat-val">8</span>
    <span class="gv-stat-lbl">Index Types</span>
  </div>
  <div class="gv-stat">
    <span class="gv-stat-val">5</span>
    <span class="gv-stat-lbl">Distance Metrics</span>
  </div>
  <div class="gv-stat">
    <span class="gv-stat-val">0</span>
    <span class="gv-stat-lbl">Dependencies</span>
  </div>
  <div class="gv-stat">
    <span class="gv-stat-val">SIMD</span>
    <span class="gv-stat-lbl">Optimized</span>
  </div>
</div>

<div class="gv-section-title">Quick Start</div>

```python
from gigavector import Database, DistanceType, IndexType

with Database.open("example.db", dimension=128, index=IndexType.HNSW) as db:
    db.add_vector([0.1] * 128, metadata={"category": "example"})

    results = db.search([0.1] * 128, k=10, distance=DistanceType.COSINE)
    for hit in results:
        print(f"  distance={hit.distance:.4f}")

    db.save("example.db")
```

<hr class="gv-sep">

<div class="gv-section-title">Capabilities</div>

<div class="gv-cards">
  <div class="gv-card">
    <div class="gv-card-title">Search</div>
    <p class="gv-card-text">k-NN, range, batch, filtered, hybrid BM25, scroll, grouped, geo-spatial, ColBERT, recommendations, SQL queries, phased ranking</p>
  </div>
  <div class="gv-card">
    <div class="gv-card-title">Storage</div>
    <p class="gv-card-text">WAL, snapshots, mmap I/O, incremental backup, JSON import/export, background compaction</p>
  </div>
  <div class="gv-card">
    <div class="gv-card-title">Data</div>
    <p class="gv-card-text">Rich metadata, payload indexing, schema evolution, upsert, TTL, named vectors, collection aliases, CDC streams, time-travel</p>
  </div>
  <div class="gv-card">
    <div class="gv-card-title">Distributed</div>
    <p class="gv-card-text">HTTP REST, gRPC, TLS, sharding, replication, cluster management, multi-tenancy, streaming ingestion</p>
  </div>
  <div class="gv-card">
    <div class="gv-card-title">Graph</div>
    <p class="gv-card-text">Property graph, knowledge graph with embeddings, BFS/DFS/Dijkstra, PageRank, entity resolution, temporal edges</p>
  </div>
  <div class="gv-card">
    <div class="gv-card-title">AI Integration</div>
    <p class="gv-card-text">OpenAI / Anthropic / Gemini, auto-embedding, semantic memory, ONNX serving, agentic interfaces</p>
  </div>
</div>

<hr class="gv-sep">

<div class="gv-section-title">Index Algorithms</div>

<div class="gv-table" markdown>

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

</div>

Use `suggest_index()` for automatic selection based on your data characteristics.

<hr class="gv-sep">

<div class="gv-section-title">Explore the Docs</div>

<div class="gv-links">
  <a href="usage/" class="gv-link">
    <span class="gv-link-title">Usage Guide</span>
    <span class="gv-link-desc">Get up and running</span>
  </a>
  <a href="python_bindings/" class="gv-link">
    <span class="gv-link-title">Python Bindings</span>
    <span class="gv-link-desc">Full Python API docs</span>
  </a>
  <a href="c_api_guide/" class="gv-link">
    <span class="gv-link-title">C API Guide</span>
    <span class="gv-link-desc">C usage patterns</span>
  </a>
  <a href="api_reference/" class="gv-link">
    <span class="gv-link-title">API Reference</span>
    <span class="gv-link-desc">Complete reference</span>
  </a>
  <a href="architecture/" class="gv-link">
    <span class="gv-link-title">Architecture</span>
    <span class="gv-link-desc">System design &amp; internals</span>
  </a>
  <a href="performance/" class="gv-link">
    <span class="gv-link-title">Performance Tuning</span>
    <span class="gv-link-desc">Index selection &amp; optimization</span>
  </a>
  <a href="deployment/" class="gv-link">
    <span class="gv-link-title">Deployment</span>
    <span class="gv-link-desc">Production scaling</span>
  </a>
  <a href="security/" class="gv-link">
    <span class="gv-link-title">Security</span>
    <span class="gv-link-desc">Auth, RBAC &amp; best practices</span>
  </a>
  <a href="examples/basic_usage/" class="gv-link">
    <span class="gv-link-title">Examples</span>
    <span class="gv-link-desc">Code examples to get started</span>
  </a>
</div>

<div class="gv-footer">
  Licensed under <a href="https://github.com/jaywyawhare/GigaVector/blob/master/LICENCE">DBaJ-NC-CFL</a>
</div>

</div>
