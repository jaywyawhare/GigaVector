# GigaVector

<p align="center">
  <img src="https://raw.githubusercontent.com/jaywyawhare/GigaVector/master/gigavector-logo.png" alt="GigaVector Logo" width="200" />
</p>

<p align="center">
  <a href="https://pepy.tech/projects/gigavector">
    <img src="https://static.pepy.tech/personalized-badge/gigavector?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" />
  </a>
</p>

A high-performance vector database library for Python. GigaVector provides efficient similarity search with support for multiple index types, metadata filtering, and persistent storage.

## Features

**Core Database:**
- Multiple index types: KD-tree, HNSW, and IVFPQ
- Distance metrics: Euclidean and Cosine similarity
- Rich metadata support with key-value pairs
- Metadata filtering in search queries
- Persistent storage with snapshot and WAL (Write-Ahead Log)
- Batch operations for vector insertion and search
- Thread-safe operations

**Advanced Features:**
- GPU acceleration with CUDA support
- HTTP REST API server
- BM25 full-text search
- Hybrid search (vector + text fusion)
- Backup and restore with compression
- TTL (Time-to-Live) for automatic data expiration

**Enterprise Features:**
- Multi-tenancy with namespaces
- Sharding for horizontal scaling
- Replication for high availability
- Cluster management
- API key and JWT authentication

## Installation

Install from PyPI:

```bash
pip install gigavector
```

The package includes pre-built native libraries for supported platforms. No external dependencies required.

## Quick Start

```python
from gigavector import Database, DistanceType, IndexType

# Create an in-memory database
with Database.open(None, dimension=128, index=IndexType.HNSW) as db:
    # Add vectors with metadata
    db.add_vector([0.1] * 128, metadata={"id": "vec1", "category": "A"})
    db.add_vector([0.2] * 128, metadata={"id": "vec2", "category": "B"})
    
    # Search for similar vectors
    hits = db.search([0.1] * 128, k=5, distance=DistanceType.EUCLIDEAN)
    for hit in hits:
        print(f"Distance: {hit.distance}, Metadata: {hit.vector.metadata}")
```

## API Reference

### Database

The main class for vector database operations.

#### `Database.open(path, dimension, index=IndexType.KDTREE)`

Create or open a database instance.

**Parameters:**
- `path` (str | None): File path for persistent storage. Use `None` for in-memory database.
- `dimension` (int): Vector dimension (must be consistent for all vectors).
- `index` (IndexType): Index type to use. Defaults to `IndexType.KDTREE`.

**Returns:** `Database` instance

**Example:**
```python
# In-memory database
db = Database.open(None, dimension=128, index=IndexType.HNSW)

# Persistent database
db = Database.open("vectors.db", dimension=128, index=IndexType.KDTREE)
```

#### `add_vector(vector, metadata=None)`

Add a single vector to the database.

**Parameters:**
- `vector` (Sequence[float]): Vector data as a sequence of floats. Length must match database dimension.
- `metadata` (dict[str, str] | None): Optional dictionary of key-value metadata pairs.

**Raises:**
- `ValueError`: If vector dimension doesn't match database dimension.
- `RuntimeError`: If insertion fails.

**Example:**
```python
# Vector without metadata
db.add_vector([1.0, 2.0, 3.0])

# Vector with single metadata entry
db.add_vector([1.0, 2.0, 3.0], metadata={"id": "123"})

# Vector with multiple metadata entries
db.add_vector([1.0, 2.0, 3.0], metadata={
    "id": "123",
    "category": "electronics",
    "price": "99.99"
})
```

#### `add_vectors(vectors)`

Add multiple vectors to the database in batch. Vectors added via this method cannot include metadata.

**Parameters:**
- `vectors` (Iterable[Sequence[float]]): Iterable of vectors. All vectors must have the same dimension.

**Raises:**
- `ValueError`: If vectors have inconsistent dimensions.
- `RuntimeError`: If batch insertion fails.

**Example:**
```python
vectors = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]
db.add_vectors(vectors)
```

#### `search(query, k, distance=DistanceType.EUCLIDEAN, filter_metadata=None)`

Search for k nearest neighbors to a query vector.

**Parameters:**
- `query` (Sequence[float]): Query vector. Length must match database dimension.
- `k` (int): Number of nearest neighbors to return.
- `distance` (DistanceType): Distance metric to use. Defaults to `DistanceType.EUCLIDEAN`.
- `filter_metadata` (tuple[str, str] | None): Optional metadata filter as (key, value) tuple. Only vectors matching the filter are considered.

**Returns:** `list[SearchHit]` - List of search results, ordered by distance (ascending).

**Raises:**
- `ValueError`: If query dimension doesn't match database dimension.
- `RuntimeError`: If search fails.

**Example:**
```python
# Basic search
hits = db.search([1.0, 2.0, 3.0], k=5, distance=DistanceType.EUCLIDEAN)

# Search with metadata filter
hits = db.search(
    [1.0, 2.0, 3.0],
    k=5,
    distance=DistanceType.EUCLIDEAN,
    filter_metadata=("category", "electronics")
)
```

#### `search_batch(queries, k, distance=DistanceType.EUCLIDEAN)`

Search for k nearest neighbors for multiple query vectors in batch.

**Parameters:**
- `queries` (Iterable[Sequence[float]]): Iterable of query vectors.
- `k` (int): Number of nearest neighbors to return per query.
- `distance` (DistanceType): Distance metric to use. Defaults to `DistanceType.EUCLIDEAN`.

**Returns:** `list[list[SearchHit]]` - List of search result lists, one per query.

**Raises:**
- `ValueError`: If any query dimension doesn't match database dimension.
- `RuntimeError`: If batch search fails.

**Example:**
```python
queries = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]
results = db.search_batch(queries, k=5)
for i, hits in enumerate(results):
    print(f"Query {i}: {len(hits)} results")
```

#### `save(path=None)`

Persist the database to a binary snapshot file. If a file path was provided when opening the database, writes to that path. Otherwise, use the provided path.

**Parameters:**
- `path` (str | None): Optional file path. If None and database was opened with a path, uses that path.

**Raises:**
- `RuntimeError`: If save operation fails.

**Example:**
```python
# Save to the path used when opening
db.save()

# Save to a different path
db.save("backup.db")
```

#### `train_ivfpq(data)`

Train the IVFPQ index with training vectors. Only applicable when using `IndexType.IVFPQ`.

**Parameters:**
- `data` (Sequence[Sequence[float]]): Training vectors. All vectors must match the database dimension.

**Raises:**
- `ValueError`: If training data is empty or dimensions don't match.
- `RuntimeError`: If training fails.

**Example:**
```python
# Train with at least 256 vectors (recommended)
train_data = [[(i % 10) / 10.0 for _ in range(128)] for i in range(256)]
db.train_ivfpq(train_data)
```

#### `close()`

Close the database and release resources. Automatically called when using the context manager.

**Example:**
```python
db = Database.open(None, dimension=128)
# ... use database ...
db.close()
```

### IndexType

Enumeration of available index types.

- `IndexType.KDTREE`: KD-tree index. Good for low to medium dimensional data.
- `IndexType.HNSW`: Hierarchical Navigable Small World graph. Good for high-dimensional data with fast approximate search.
- `IndexType.IVFPQ`: Inverted File with Product Quantization. Memory-efficient for large-scale datasets. Requires training before use.

### DistanceType

Enumeration of distance metrics.

- `DistanceType.EUCLIDEAN`: Euclidean (L2) distance.
- `DistanceType.COSINE`: Cosine similarity distance.

### Vector

Data class representing a vector with metadata.

**Attributes:**
- `data` (list[float]): Vector data.
- `metadata` (dict[str, str]): Dictionary of metadata key-value pairs.

### SearchHit

Data class representing a search result.

**Attributes:**
- `distance` (float): Distance from the query vector.
- `vector` (Vector): The matched vector with its metadata.

## Usage Examples

### Persistent Storage with WAL

```python
from gigavector import Database, IndexType, DistanceType

# Create a persistent database
with Database.open("vectors.db", dimension=128, index=IndexType.KDTREE) as db:
    db.add_vector([0.1] * 128, metadata={"id": "1", "tag": "A"})
    db.add_vector([0.2] * 128, metadata={"id": "2", "tag": "B"})
    db.save()  # Create snapshot

# Reopen - WAL automatically replays any uncommitted changes
with Database.open("vectors.db", dimension=128, index=IndexType.KDTREE) as db:
    hits = db.search([0.1] * 128, k=5)
    # All vectors are restored, including metadata
```

### IVFPQ Index with Training

```python
from gigavector import Database, IndexType, DistanceType
import random

# Create IVFPQ database
db = Database.open(None, dimension=64, index=IndexType.IVFPQ)

# Generate training data (at least 256 vectors recommended)
train_data = [
    [random.random() for _ in range(64)]
    for _ in range(256)
]
db.train_ivfpq(train_data)

# Add vectors
with db:
    for i in range(1000):
        vec = [random.random() for _ in range(64)]
        db.add_vector(vec, metadata={"id": str(i)})
    
    # Search
    query = [random.random() for _ in range(64)]
    hits = db.search(query, k=10, distance=DistanceType.EUCLIDEAN)
```

### Metadata Filtering

```python
from gigavector import Database, IndexType, DistanceType

with Database.open(None, dimension=128, index=IndexType.HNSW) as db:
    # Add vectors with different categories
    db.add_vector([0.1] * 128, metadata={"category": "A", "price": "10"})
    db.add_vector([0.2] * 128, metadata={"category": "B", "price": "20"})
    db.add_vector([0.15] * 128, metadata={"category": "A", "price": "15"})
    
    # Search only in category A
    hits = db.search(
        [0.1] * 128,
        k=10,
        distance=DistanceType.EUCLIDEAN,
        filter_metadata=("category", "A")
    )
    # Returns only vectors with category="A"
```

### Batch Operations

```python
from gigavector import Database, IndexType, DistanceType

with Database.open(None, dimension=128, index=IndexType.KDTREE) as db:
    # Batch insert vectors (without metadata)
    vectors = [[i * 0.01] * 128 for i in range(1000)]
    db.add_vectors(vectors)
    
    # Batch search
    queries = [[i * 0.01] * 128 for i in range(10)]
    results = db.search_batch(queries, k=5)
    for i, hits in enumerate(results):
        print(f"Query {i}: {len(hits)} results")
```

## Advanced Features

### GPU Acceleration

```python
from gigavector import gpu_available, gpu_device_count, gpu_get_device_info, GPUIndex, GPUConfig

# Check GPU availability
if gpu_available():
    print(f"GPU devices: {gpu_device_count()}")
    info = gpu_get_device_info(0)
    print(f"Device 0: {info.name}, {info.total_memory // 1024**2} MB")

    # Create GPU-accelerated index
    config = GPUConfig(device_id=0, use_float16=True)
    gpu_index = GPUIndex(dimension=128, config=config)
    gpu_index.add_vectors(vectors)
    results = gpu_index.search(query, k=10)
```

### HTTP REST Server

```python
from gigavector import Database, Server, ServerConfig, IndexType

# Create database and server
db = Database.open(None, dimension=128, index=IndexType.HNSW)
config = ServerConfig(port=8080, enable_cors=True)

with Server(db, config) as server:
    server.start()
    print("Server running on http://localhost:8080")
    # Server handles REST API requests:
    # GET  /health - Health check
    # POST /vectors - Add vector
    # POST /search - Search vectors
    # GET  /stats - Server statistics
```

### BM25 Full-Text Search

```python
from gigavector import BM25Index, BM25Config

# Create BM25 index for text search
config = BM25Config(k1=1.2, b=0.75)
bm25 = BM25Index(config)

# Add documents
bm25.add_document(0, "Machine learning for vector databases")
bm25.add_document(1, "Neural networks and deep learning")
bm25.add_document(2, "Vector similarity search algorithms")

# Search
results = bm25.search("vector search", k=10)
for r in results:
    print(f"Doc {r.doc_id}: score={r.score:.4f}")

bm25.close()
```

### Hybrid Search (Vector + Text)

```python
from gigavector import Database, BM25Index, HybridSearcher, HybridConfig, IndexType

db = Database.open(None, dimension=128, index=IndexType.HNSW)
bm25 = BM25Index()

# Add vectors and corresponding documents
for i, (vec, text) in enumerate(zip(vectors, documents)):
    db.add_vector(vec, metadata={"id": str(i)})
    bm25.add_document(i, text)

# Create hybrid searcher
config = HybridConfig(vector_weight=0.7, text_weight=0.3)
hybrid = HybridSearcher(db, bm25, config)

# Search with both vector and text
results = hybrid.search(query_vector, "search query", k=10)
for r in results:
    print(f"Index {r.vector_index}: combined={r.combined_score:.4f}")

hybrid.close()
```

### Namespaces (Multi-Tenancy)

```python
from gigavector import NamespaceManager, NamespaceConfig

# Create namespace manager
ns_mgr = NamespaceManager("/path/to/data")

# Create isolated namespaces for different tenants
config = NamespaceConfig(name="tenant_a", dimension=128)
tenant_a = ns_mgr.create(config)

config = NamespaceConfig(name="tenant_b", dimension=128)
tenant_b = ns_mgr.create(config)

# Each namespace is isolated
tenant_a.add_vector([0.1] * 128)
tenant_b.add_vector([0.2] * 128)

print(f"Tenant A vectors: {tenant_a.count}")
print(f"Tenant B vectors: {tenant_b.count}")

ns_mgr.close()
```

### TTL (Time-to-Live)

```python
from gigavector import TTLManager, TTLConfig

# Create TTL manager for automatic expiration
config = TTLConfig(
    default_ttl_seconds=3600,  # 1 hour default
    cleanup_interval_seconds=60
)
ttl = TTLManager(config)

# Set TTL for vectors
ttl.set_ttl(vector_index=0, ttl_seconds=1800)  # 30 minutes

# Get stats
stats = ttl.get_stats()
print(f"Vectors with TTL: {stats.total_vectors_with_ttl}")
print(f"Expired: {stats.total_expired}")

ttl.close()
```

### Authentication

```python
from gigavector import AuthManager, AuthConfig, AuthType

# Create auth manager with API key authentication
config = AuthConfig(auth_type=AuthType.API_KEY)
auth = AuthManager(config)

# Generate API key
key, key_id = auth.generate_api_key("My Application")
print(f"API Key: {key}")
print(f"Key ID: {key_id}")

# Authenticate requests
result, identity = auth.authenticate(key)
if result == AuthResult.SUCCESS:
    print(f"Authenticated: {identity.key_id}")

auth.close()
```

### Backup and Restore

```python
from gigavector import (
    Database, backup_create, backup_restore, backup_verify,
    BackupOptions, RestoreOptions, BackupCompression
)

# Create backup
options = BackupOptions(
    compression=BackupCompression.ZSTD,
    include_metadata=True
)
result = backup_create(db, "backup.gvb", options)
print(f"Backup created: {result.vectors_backed_up} vectors")

# Verify backup
if backup_verify("backup.gvb"):
    print("Backup is valid")

# Restore to new database
restore_opts = RestoreOptions(verify_checksums=True)
restored_db = backup_restore("backup.gvb", "restored.db", restore_opts)
```

## Requirements

- Python 3.9 or higher
- cffi >= 1.16
- CUDA toolkit (optional, for GPU acceleration)

## License

Licensed under the DBaJ-NC-CFL License. See [LICENCE.md](LICENCE.md) for details.

## Links

- GitHub Repository: https://github.com/jaywyawhare/GigaVector
