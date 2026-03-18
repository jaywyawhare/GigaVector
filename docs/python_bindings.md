# Python Bindings

## Overview

GigaVector's Python bindings use **CFFI (C Foreign Function Interface)** to call the underlying C library. This provides:

- **High Performance:** Direct calls to optimized C code
- **Type Safety:** Automatic type conversion between Python and C
- **Memory Safety:** Automatic memory management for Python objects
- **Easy Integration:** Pythonic API that hides C complexity

> **FFI Note:** The bindings are defined in `_ffi.py` (CFFI declarations and library loading) and wrapped with Pythonic classes in `_core.py`. The C library manages vector data; Python objects are lightweight wrappers. See the source for details.

## Basic Usage

```python
from gigavector import Database, IndexType, DistanceType

# Open database
db = Database.open("example.db", dimension=128, index=IndexType.HNSW)

# Add vector
db.add_vector([0.1] * 128, metadata={"id": "1"})

# Search
results = db.search([0.1] * 128, k=10, distance=DistanceType.EUCLIDEAN)

# Close
db.close()
```

### Context Manager (Recommended)

```python
# Automatically closes database
with Database.open("example.db", dimension=128) as db:
    db.add_vector([0.1] * 128)
    results = db.search([0.1] * 128, k=10)
    # Database closed automatically on exit
```

### Working with Metadata

```python
# Add vector with metadata
db.add_vector(
    [0.1] * 128,
    metadata={
        "id": "12345",
        "category": "electronics",
        "price": "99.99"
    }
)

# Search with filter
results = db.search(
    query, k=10,
    filter_metadata=("category", "electronics")
)
```

### Batch Operations

```python
# Batch insertion (more efficient)
vectors = [[random.random() for _ in range(128)] for _ in range(1000)]
db.add_vectors(vectors)

# Batch search
queries = [[random.random() for _ in range(128)] for _ in range(10)]
all_results = db.search_batch(queries, k=5)
```

### Configuration

```python
from gigavector import HNSWConfig, IVFPQConfig

# HNSW with custom parameters
hnsw_config = HNSWConfig(
    M=32,                    # More connections
    ef_construction=200,     # Construction quality
    ef_search=50,           # Search quality
    use_binary_quant=True,   # Enable quantization
    quant_rerank=20          # Rerank candidates
)

db = Database.open(
    "db.db", dimension=128, index=IndexType.HNSW,
    hnsw_config=hnsw_config
)

# IVFPQ configuration
ivfpq_config = IVFPQConfig(
    nlist=256,
    m=16,
    nprobe=16,
    default_rerank=32
)

db = Database.open(
    None, dimension=128, index=IndexType.IVFPQ,
    ivfpq_config=ivfpq_config
)

# Must train before use
training_data = [[random.random() for _ in range(128)] for _ in range(1000)]
db.train_ivfpq(training_data)
```

## Best Practices

1. **Use context managers** to avoid resource leaks.
2. **Use batch operations** (`add_vectors`, `search_batch`) to minimize FFI overhead.
3. **Handle errors** with `try`/`except` -- the library raises `RuntimeError` for database errors and `ValueError` for invalid input (e.g., wrong dimensions).
4. **Monitor resources** -- call `db.get_memory_usage()` and `db.get_resource_limits()` for large datasets.
5. **Use appropriate data types** -- vectors must be lists of floats; metadata values must be strings. NumPy arrays work after calling `.tolist()`.

## Available Modules

The Python bindings expose the following module groups. See [usage.md](usage.md) and [c_api_guide.md](c_api_guide.md) for full examples.

- **Core:** `Database`, `Vector`, `SearchHit`, `IndexType`, `DistanceType` -- database operations and search.
- **Configuration:** `HNSWConfig`, `IVFPQConfig`, `ScalarQuantConfig` -- index tuning parameters.
- **LLM Integration:** `LLM`, `LLMConfig`, `EmbeddingService` -- embedding generation and LLM helpers.
- **Memory:** `MemoryLayer`, `ContextGraph` -- conversational memory and context tracking.
- **GPU:** `GPUContext`, `GPUIndex`, `GPUConfig` -- GPU-accelerated indexing and search.
- **Server:** `Server`, `ServerConfig`, `serve_with_dashboard` -- HTTP REST server and web dashboard.
- **Search:** `BM25Index`, `HybridSearcher`, `HybridConfig`, `FusionType` -- full-text and hybrid search.
- **Storage:** `NamespaceManager`, `TTLManager`, `ShardManager` -- multi-tenancy, expiration, and sharding.
- **High Availability:** `ReplicationManager`, `Cluster`, `ClusterConfig` -- replication and cluster management.
- **Security:** `AuthManager`, `AuthConfig` -- API key and JWT authentication.
- **Backup:** `backup_create`, `backup_restore`, `backup_verify`, `BackupOptions` -- backup and restore.
- **Graph:** `GraphDB`, `GraphDBConfig`, `GraphPath` -- property graph with traversal and analytics.
- **Knowledge Graph:** `KnowledgeGraph`, `KGConfig`, `KGTriple` -- entity/relation store with semantic search and link prediction.

---

For more information, see:
- [Usage Guide](usage.md) for general usage patterns
- [C API Guide](c_api_guide.md) for understanding the underlying C API
- [Performance Tuning Guide](performance.md) for optimization tips
