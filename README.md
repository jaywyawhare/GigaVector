# GigaVector

<p align="center">
  <img src="https://raw.githubusercontent.com/jaywyawhare/GigaVector/master/docs/gigavector-logo.png" alt="GigaVector Logo" width="200" />
</p>

<p align="center">
  <a href="https://pepy.tech/projects/gigavector">
    <img src="https://static.pepy.tech/personalized-badge/gigavector?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" />
  </a>
</p>

**GigaVector** is a high-performance, production-ready vector database library written in C with Python bindings, covering indexing, search, storage, graph, networking, security, and AI integration.

---

## Feature Overview

### Index Algorithms (8 types)

| Index | Type | Training | Best For |
|-------|------|----------|----------|
| **KD-Tree** | Exact | No | Low-dimensional data (< 20D) |
| **HNSW** | Approximate | No | General-purpose, high recall |
| **IVF-PQ** | Approximate | Yes | Large-scale, memory-efficient |
| **IVF-Flat** | Approximate | Yes | Large-scale, higher accuracy than IVF-PQ |
| **Flat** | Exact (brute-force) | No | Small datasets, baseline/ground-truth |
| **PQ** | Approximate | Yes | Compressed-domain search |
| **LSH** | Approximate | No | Fast hash-based approximate search |
| **Sparse** | Exact | No | Sparse vectors (NLP, BoW) |

### Distance Metrics (5 types)
Euclidean, Cosine, Dot Product, Manhattan, Hamming -- all with SIMD-optimized implementations (SSE4.2, AVX2, AVX-512F, FMA).

### Search Capabilities
- **k-NN search** with configurable distance metrics
- **Range search** -- find all vectors within a radius
- **Batch search** -- multiple queries in one call
- **Filtered search** -- metadata-based pre/post filtering
- **Dynamic search params** -- per-query ef_search, nprobe, rerank tuning
- **Hybrid search** -- combine vector similarity with BM25 full-text ranking (RRF, weighted, Borda fusion)
- **Scroll/pagination** -- iterate over stored vectors with offset/limit
- **Score threshold filtering** -- return only results above a distance/similarity cutoff
- **Grouped search** -- group results by metadata field with per-group limits
- **Geo-spatial filtering** -- radius and bounding-box queries on lat/lon fields
- **Late interaction / ColBERT** -- multi-vector MaxSim scoring for token-level matching
- **Recommendation API** -- positive/negative example-based recommendations with strategy selection
- **Delete/update by filter** -- bulk delete, metadata update, and count by filter expression
- **MMR diversity reranking** -- maximal marginal relevance for diverse result sets
- **Custom ranking expressions** -- expression parser with decay functions (exp/gauss/linear) and score boosting
- **SQL query interface** -- `SELECT ... WHERE ... ORDER BY vector_distance(...)` SQL syntax
- **Phased ranking pipeline** -- multi-stage ANN → rerank → filter pipeline with per-phase stats
- **Learned sparse index** -- SPLADE-style token-weighted inverted index with WAND acceleration
- **Full-text search** -- Porter stemming, multilingual (6 languages), BlockMax WAND, phrase matching

### Storage and Persistence
- **Write-Ahead Logging (WAL)** -- crash-safe durability with automatic replay
- **Snapshot persistence** -- save/load full database state
- **Point-in-time snapshots** -- create immutable snapshots for historical queries
- **Collection versioning** -- version datasets with diff/compare/rollback
- **Memory-mapped I/O** -- efficient file-backed storage
- **Incremental backup** -- full and incremental backup with compression and CRC verification
- **JSON import/export** -- NDJSON format for interoperability

### Data Management
- **Rich metadata** -- key-value pairs per vector with typed metadata support
- **Payload indexing** -- sorted indexes for int/float/string/bool fields with range queries
- **Schema evolution** -- versioned schemas with validation, diff, and compatibility checking
- **Upsert operations** -- insert-or-update semantics
- **Batch delete** -- delete multiple vectors in one call
- **Vector deduplication** -- LSH-based near-duplicate detection
- **BM25 full-text search** -- TF-IDF style keyword search on text fields
- **TTL (Time-to-Live)** -- automatic expiry of vectors
- **User-defined point IDs** -- string/UUID IDs with bidirectional mapping to internal indices
- **Named vectors** -- multiple named vector fields per point with independent dimensions
- **Collection aliases** -- create, swap, and manage aliases that point to collections
- **Payload compression** -- zlib/LZ4/zstd compression for stored vector payloads
- **JSON path indexing** -- index nested JSON fields with dot-notation, type-aware range queries
- **CDC stream** -- change data capture with ring buffer, polling cursors, and subscriber callbacks
- **Conditional updates** -- CAS-style optimistic concurrency with per-vector versioning
- **Time-travel queries** -- auto-versioned append-only log, query at any version or timestamp
- **Multimodal storage** -- SHA-256 content-addressable blob storage for images, audio, video, documents

### Transactions and Concurrency
- **MVCC transactions** -- snapshot isolation with begin/commit/rollback
- **Thread-safe** -- reader-writer locks for concurrent access
- **Client-side caching** -- LRU/LFU cache with TTL and mutation-based invalidation

### Quantization and Compression
- **Product Quantization (PQ)** -- codebook-based compression
- **Scalar Quantization** -- configurable bit-width reduction
- **Binary Quantization** -- 1-bit compression for HNSW
- **Codebook sharing** -- train once, share PQ codebooks across collections
- **Advanced quantization** -- 1.5-bit (ternary), 2-bit, 4-bit, 8-bit; RaBitQ with Householder rotations; symmetric/asymmetric modes
- **Inline HNSW + incremental rebuild** -- quantized vectors embedded in graph nodes with prefetch, background rebuild

### Distributed Architecture
- **HTTP REST server** -- embedded server with rate limiting, CORS, and API key auth
- **gRPC API** -- binary protocol server with connection pooling and streaming support
- **TLS/HTTPS** -- TLS 1.2/1.3 transport encryption with certificate management
- **Sharding** -- hash/range-based data partitioning
- **Replication** -- leader-follower with automatic failover and election
- **Read replica load balancing** -- round-robin, least-lag, and random routing policies
- **Cluster management** -- multi-node coordination
- **Namespace / multi-tenancy** -- isolated collections within a single instance
- **Configurable consistency** -- eventual, quorum, and strong consistency levels
- **Tenant quotas** -- per-tenant limits on vector count, memory, and QPS
- **Tiered multitenancy** -- shared/dedicated/premium tiers with auto-promote/demote and QPS tracking
- **Embedded / edge mode** -- lightweight in-process database with memory budget, mmap, and quantization

### Security
- **Authentication** -- API key and JWT-based auth
- **RBAC** -- fine-grained role-based access control with per-collection permissions
- **Cryptographic primitives** -- SHA-256, HMAC for secure token handling
- **Enterprise SSO** -- OIDC discovery, JWT validation, SAML XML parsing for enterprise identity providers

### Graph and Knowledge Graph
- **Property graph database** -- nodes with labels/properties, directed weighted edges, hash table storage
- **Graph traversal** -- BFS, DFS, Dijkstra shortest path, all-paths enumeration
- **Graph analytics** -- PageRank (iterative power method), connected components, clustering coefficient, degree centrality
- **Knowledge graph** -- entities with embeddings, SPO triple store with wildcard queries
- **Semantic entity search** -- cosine similarity search over entity embeddings
- **Entity resolution** -- name match + embedding similarity deduplication, entity merging
- **Link prediction** -- embedding similarity + structural patterns (shared neighbors)
- **Hybrid graph+vector search** -- embedding similarity filtered by entity type and predicate
- **Subgraph extraction** -- BFS-based k-hop subgraph with entity and relation IDs
- **Graph persistence** -- binary save/load for both graph DB ("GVGR") and knowledge graph ("GVKG")

### AI Integration
- **LLM support** -- OpenAI, Anthropic, Google Gemini (chat completions, streaming)
- **Embedding services** -- OpenAI, Google, HuggingFace embedding APIs with caching
- **Auto-embedding** -- server-side text-to-vector with configurable providers and batching
- **Semantic memory layer** -- extract, store, consolidate memories from conversations
- **Context graphs** -- entity-relationship extraction for context-aware retrieval
- **Importance scoring** -- rank memories by relevance and recency
- **ONNX model serving** -- load ONNX models for inference, reranking, and embedding in the search pipeline
- **Agentic interfaces** -- LLM-powered natural language query, data transformation, and personalization agents
- **MUVERA encoding** -- compress ColBERT multi-vectors into single dense vectors via random projections
- **Integrated inference** -- text-in/results-out API that combines auto-embedding with vector search

### Observability and Operations
- **Query optimizer** -- cost-based strategy selection (exact scan vs index vs oversample+filter)
- **Query tracing** -- span-level timing for search pipeline profiling
- **Bloom filter indexes** -- probabilistic skip indexes for fast set membership
- **Index migration** -- background thread rebuilds index while old one continues serving
- **DiskANN** -- on-disk approximate nearest neighbor index with Vamana graph
- **Async vacuum** -- background compaction with configurable thresholds and scheduling
- **Webhooks** -- event-driven notifications for insert/delete/update operations
- **GPU acceleration** -- CUDA-based distance computation and batch search (optional)
- **Database statistics** -- insert/query counts, latency tracking

---

## Build

### Make (default)
```bash
make lib        # static + shared libraries -> build/lib/
make c-test     # run all C tests (21 test suites)
make python-test # run Python test suite
```

### CMake
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest
```

**CMake Options:**
- `-DBUILD_SHARED_LIBS=ON/OFF` -- shared library (default: ON)
- `-DBUILD_TESTS=ON/OFF` -- test executables (default: ON)
- `-DBUILD_BENCHMARKS=ON/OFF` -- benchmark executables (default: ON)
- `-DENABLE_SANITIZERS=ON/OFF` -- ASAN, TSAN, UBSAN (default: OFF)
- `-DENABLE_COVERAGE=ON/OFF` -- code coverage (default: OFF)

### Sanitizer and Coverage Testing
```bash
make test-asan      # AddressSanitizer
make test-tsan      # ThreadSanitizer
make test-ubsan     # UndefinedBehaviorSanitizer
make test-valgrind  # Valgrind memory check
make test-coverage  # gcov/lcov coverage report
make test-all       # run everything
```

---

## Python Bindings

### Install
```bash
pip install gigavector        # from PyPI
cd python && pip install .    # from source
```

### Quick Start
```python
from gigavector import Database, DistanceType, IndexType

# Open / create a database
with Database.open("example.db", dimension=128, index=IndexType.HNSW) as db:
    # Add vectors with metadata
    db.add_vector([0.1] * 128, metadata={"category": "example"})

    # Search
    results = db.search([0.1] * 128, k=10, distance=DistanceType.COSINE)
    for hit in results:
        print(f"  index={hit.index}, distance={hit.distance:.4f}")

    # Save to disk
    db.save("example.db")
```

### Index Types
```python
# Flat (brute-force exact search)
db = Database.open(None, dimension=128, index=IndexType.FLAT)

# HNSW with custom config
from gigavector import HNSWConfig
db = Database.open(None, dimension=128, index=IndexType.HNSW,
                   hnsw_config=HNSWConfig(M=32, efConstruction=200, efSearch=100))

# IVF-PQ (requires training)
db = Database.open(None, dimension=128, index=IndexType.IVFPQ)
db.train_ivfpq(training_vectors)

# IVF-Flat (requires training)
from gigavector import IVFFlatConfig
db = Database.open(None, dimension=128, index=IndexType.IVFFLAT,
                   ivfflat_config=IVFFlatConfig(nlist=64, nprobe=8))
db.train_ivfflat(training_vectors)

# LSH (no training needed)
from gigavector import LSHConfig
db = Database.open(None, dimension=128, index=IndexType.LSH,
                   lsh_config=LSHConfig(num_tables=8, num_hash_bits=16))
```

### Advanced Features
```python
from gigavector import (
    SearchParams, BloomFilter, Cache, CacheConfig,
    Schema, SchemaFieldType, MVCCManager, QueryOptimizer,
    PayloadIndex, FieldType, DedupIndex, MultiVecIndex,
    SnapshotManager, VersionManager, Codebook, QueryTrace,
)

# Dynamic search parameters
results = db.search_with_params([0.1] * 128, k=10,
    distance=DistanceType.COSINE,
    params=SearchParams(ef_search=200, nprobe=16))

# Bloom filter for fast membership checks
bf = BloomFilter(expected_items=10000, fp_rate=0.01)
bf.add_string("hello")
assert "hello" in bf

# Client-side result caching
cache = Cache(CacheConfig(max_entries=1024, ttl_seconds=30))

# Schema validation
schema = Schema(version=1)
schema.add_field("name", SchemaFieldType.STRING, required=True)
schema.add_field("score", SchemaFieldType.FLOAT)
assert schema.validate({"name": "test", "score": "0.95"})

# MVCC transactions
mvcc = MVCCManager(dimension=128)
with mvcc.begin() as txn:
    txn.add_vector([0.1] * 128)
    txn.add_vector([0.2] * 128)
    # auto-commits on exit, or auto-rolls-back on exception

# Query optimizer
opt = QueryOptimizer()
plan = opt.plan(k=10, has_filter=True, filter_selectivity=0.05)
print(f"Strategy: {plan.strategy.name}, ef_search={plan.ef_search}")

# Payload indexing
idx = PayloadIndex()
idx.add_field("category", FieldType.STRING)
idx.insert_string(0, "category", "science")

# Vector deduplication
dedup = DedupIndex(dimension=128)
dedup.insert([0.1] * 128)
is_duplicate = dedup.check([0.1] * 128)

# Multi-vector documents
mv = MultiVecIndex(dimension=128)
mv.add_document(doc_id=1, chunks=[[0.1]*128, [0.2]*128, [0.3]*128])
results = mv.search([0.15]*128, k=5)

# Point-in-time snapshots
snap_mgr = SnapshotManager(max_snapshots=10)

# Collection versioning
ver_mgr = VersionManager(max_versions=20)

# Codebook sharing (train once, reuse)
cb = Codebook(dimension=128, m=8, nbits=8)
cb.train(training_data)
cb.save("shared_codebook.bin")

# Query tracing
with QueryTrace() as trace:
    trace.span_start("search")
    results = db.search([0.1]*128, k=10)
    trace.span_end()
```

### New Features (v0.8)
```python
from gigavector import (
    PointIDMap, NamedVectorStore, VectorFieldConfig,
    GeoIndex, GeoPoint, GroupedSearch, GroupSearchConfig,
    DiskANNIndex, DiskANNConfig, Recommender, RecommendConfig,
    AliasManager, VacuumManager, ConsistencyLevel, ConsistencyManager,
    search_with_threshold, delete_by_filter, count_by_filter,
)

# User-defined string/UUID point IDs
id_map = PointIDMap()
id_map["doc-abc-123"] = 0
id_map["doc-def-456"] = 1
print(id_map["doc-abc-123"])  # 0

# Named vectors (multiple vector fields per point)
store = NamedVectorStore()
store.add_field(VectorFieldConfig(name="title", dimension=128))
store.add_field(VectorFieldConfig(name="content", dimension=256))
store.insert("title", 0, [0.1] * 128)
store.insert("content", 0, [0.2] * 256)

# Score threshold filtering
results = search_with_threshold(db, [0.1]*128, k=10, threshold=0.5)

# Delete/update by filter
deleted = delete_by_filter(db, 'category == "old"')
count = count_by_filter(db, 'status == "active"')

# Geo-spatial search
geo = GeoIndex()
geo.add(0, GeoPoint(lat=40.7128, lon=-74.0060))
nearby = geo.search_radius(GeoPoint(lat=40.71, lon=-74.01), radius_km=1.0, limit=10)

# Grouped search
gs = GroupedSearch(db)
groups = gs.search([0.1]*128, group_by="category",
                   config=GroupSearchConfig(group_size=3, num_groups=5))

# DiskANN on-disk index
disk_idx = DiskANNIndex(DiskANNConfig(dimension=128, max_degree=64, search_list_size=128))

# Recommendation
rec = Recommender(db)
results = rec.recommend(positive_ids=[0, 1], negative_ids=[5],
                        config=RecommendConfig(limit=10))

# Collection aliases
aliases = AliasManager()
aliases.create("production", "vectors_v2")
aliases.swap("production", "vectors_v3")

# Vacuum / compaction
vacuum = VacuumManager(db)
vacuum.run()

# Consistency levels
cm = ConsistencyManager()
cm.set_level(ConsistencyLevel.QUORUM)
```

### Graph Database and Knowledge Graph
```python
from gigavector import GraphDB, GraphDBConfig, KnowledgeGraph, KGConfig

# Property graph database
g = GraphDB(GraphDBConfig(node_bucket_count=4096))
alice = g.add_node("Person")
bob = g.add_node("Person")
g.set_node_prop(alice, "name", "Alice")
g.set_node_prop(bob, "name", "Bob")
e = g.add_edge(alice, bob, "KNOWS", weight=1.0)

# Traversal and analytics
visited = g.bfs(alice, max_depth=3)
path = g.shortest_path(alice, bob)
pr = g.pagerank(alice, iterations=20, damping=0.85)
cc = g.clustering_coefficient(alice)
components = g.connected_components()

# Persistence
g.save("social.gvgr")
g2 = GraphDB.load("social.gvgr")

# Knowledge graph with embeddings
kg = KnowledgeGraph(KGConfig(embedding_dimension=128))
e1 = kg.add_entity("Alice", "Person", embedding=[0.1] * 128)
e2 = kg.add_entity("Anthropic", "Company", embedding=[0.2] * 128)
kg.add_relation(e1, "works_at", e2, weight=1.0)

# SPO triple queries (None = wildcard)
triples = kg.query_triples(predicate="works_at")

# Semantic search over entity embeddings
results = kg.search_similar([0.15] * 128, k=5)

# Hybrid search (vector + type/predicate filters)
results = kg.hybrid_search([0.1] * 128, entity_type="Person",
                            predicate_filter="works_at", k=10)

# Entity resolution and link prediction
resolved_id = kg.resolve_entity("Alice Smith", "Person", embedding=[0.1] * 128)
predictions = kg.predict_links(e1, k=5)
subgraph = kg.extract_subgraph(center=e1, radius=2)

# Persistence
kg.save("knowledge.gvkg")
kg2 = KnowledgeGraph.load("knowledge.gvkg")
```

### JSON Import/Export
```python
db.export_json("vectors.ndjson")
db.import_json("vectors.ndjson")
```

### Upsert and Batch Operations
```python
db.upsert(index=0, data=[0.5] * 128, metadata={"updated": "true"})
deleted = db.delete_vectors([0, 1, 2])
entries = db.scroll(offset=0, limit=100)
```

---

## REST API

GigaVector includes an embedded HTTP server for remote access.

```python
from gigavector import Server, ServerConfig

config = ServerConfig(port=8080, thread_pool_size=4, enable_cors=True,
                      max_requests_per_second=100.0)
server = Server(db, config)
server.start()
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Database statistics |
| `POST` | `/vectors` | Add vector(s) |
| `GET` | `/vectors/{id}` | Get vector by index |
| `PUT` | `/vectors/{id}` | Update vector |
| `DELETE` | `/vectors/{id}` | Delete vector |
| `POST` | `/search` | k-NN search |
| `POST` | `/search/range` | Range search |
| `POST` | `/search/batch` | Batch search |
| `POST` | `/compact` | Trigger compaction |
| `POST` | `/save` | Save database to disk |

---

## Environment Variables

```bash
cp .env.example .env   # copy and edit with your keys
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For LLM/embedding tests | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic tests | Anthropic/Claude API key |
| `GOOGLE_API_KEY` | Optional | Google Gemini/embeddings |
| `GV_WAL_DIR` | Optional | Override WAL directory |

---

## Documentation

- [Usage Guide](docs/usage.md) -- comprehensive usage guide
- [Build and Test Guide](docs/build_and_test.md) -- build instructions and testing
- [Python Bindings Guide](docs/python_bindings.md) -- Python API documentation
- [C API Guide](docs/c_api_guide.md) -- C API usage patterns
- [API Reference](docs/api_reference.md) -- complete API reference
- [Architecture](docs/architecture.md) -- system design and internals
- [Deployment Guide](docs/deployment.md) -- production deployment and scaling
- [Security Guide](docs/security.md) -- security best practices
- [Performance Tuning](docs/performance.md) -- index selection and optimization
- [Troubleshooting](docs/troubleshooting.md) -- common issues and solutions
- [Advanced Features](docs/examples/advanced_features.md) -- advanced patterns
- [Contributing](CONTRIBUTING.md) -- how to contribute

---

## License

This project is licensed under the DBaJ-NC-CFL [License](./LICENCE).
