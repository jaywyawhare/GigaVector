# Python Bindings Guide

This guide explains how GigaVector's Python bindings work and how to use them effectively.

## Table of Contents

1. [Overview](#overview)
2. [How the FFI Works](#how-the-ffi-works)
3. [Using Python Bindings](#using-python-bindings)
4. [Best Practices](#best-practices)
5. [Extending the Bindings](#extending-the-bindings)
6. [Troubleshooting](#troubleshooting)

## Overview

GigaVector's Python bindings use **CFFI (C Foreign Function Interface)** to call the underlying C library. This provides:

- **High Performance:** Direct calls to optimized C code
- **Type Safety:** Automatic type conversion between Python and C
- **Memory Safety:** Automatic memory management for Python objects
- **Easy Integration:** Pythonic API that hides C complexity

## How the FFI Works

### Architecture

```
Python Code
    ↓
gigavector._core.py (Python wrapper classes)
    ↓
gigavector._ffi.py (CFFI interface)
    ↓
libGigaVector.so (C shared library)
    ↓
C Implementation
```

### CFFI Interface Definition

The FFI interface is defined in `_ffi.py` using C declarations:

```python
from cffi import FFI

ffi = FFI()

# Define C types and functions
ffi.cdef("""
    typedef struct {
        size_t dimension;
        float *data;
        GV_Metadata *metadata;
    } GV_Vector;
    
    GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type);
    int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);
    // ... more declarations
""")

# Load the shared library
lib = ffi.dlopen("libGigaVector.so")
```

### Type Conversions

CFFI automatically handles conversions:

| C Type | Python Type | Conversion |
|--------|-------------|------------|
| `float[]` | `list[float]` | `ffi.new("float[]", [1.0, 2.0, 3.0])` |
| `const char*` | `str` | `"text".encode()` |
| `GV_Database*` | Opaque pointer | Managed by wrapper |
| `int` | `int` | Automatic |
| `size_t` | `int` | Automatic |

### Memory Management

**Important:** The C library manages vector data. Python objects are wrappers:

```python
# Python creates a buffer
buf = ffi.new("float[]", [0.1, 0.2, 0.3])

# C library takes ownership of the data
lib.gv_db_add_vector(db, buf, 3)

# Python buffer is automatically freed when out of scope
# But the C library keeps a copy of the data
```

## Using Python Bindings

### Basic Usage

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

### 1. Always Use Context Managers

```python
# Good
with Database.open("db.db", 128) as db:
    # Use database
    pass

# Bad - may leak resources
db = Database.open("db.db", 128)
# Forgot to close!
```

### 2. Check Vector Dimensions

```python
# The library checks dimensions automatically
try:
    db.add_vector([0.1] * 64)  # Wrong dimension
except ValueError as e:
    print(f"Error: {e}")  # "expected vector of dim 128, got 64"
```

### 3. Handle Errors Properly

```python
try:
    db = Database.open("db.db", 128)
    db.add_vector(vector)
except RuntimeError as e:
    print(f"Database error: {e}")
    # Handle error appropriately
except ValueError as e:
    print(f"Invalid input: {e}")
    # Handle validation error
```

### 4. Use Batch Operations

```python
# Good - efficient
db.add_vectors(vectors)
results = db.search_batch(queries, k=10)

# Less efficient - many FFI calls
for vec in vectors:
    db.add_vector(vec)
```

### 5. Pre-allocate Lists for Results

```python
# Results are returned as Python lists
# No need to pre-allocate, but be aware of memory usage
results = db.search(query, k=1000)  # Large k uses more memory
```

### 6. Monitor Resource Usage

```python
# Check memory usage
memory_bytes = db.get_memory_usage()
print(f"Memory: {memory_bytes / 1024 / 1024:.2f} MB")

# Check resource limits
limits = db.get_resource_limits()
print(f"Max memory: {limits['max_memory_bytes']}")
print(f"Max vectors: {limits['max_vectors']}")
```

### 7. Use Appropriate Data Types

```python
# Good - list of floats
vector = [0.1, 0.2, 0.3, ...]

# Also works - numpy array
import numpy as np
vector = np.array([0.1, 0.2, 0.3, ...], dtype=np.float32)
db.add_vector(vector.tolist())  # Convert to list

# Good - Python dict for metadata
metadata = {"key": "value"}

# Bad - complex types in metadata
metadata = {"key": [1, 2, 3]}  # Only strings supported
```

## Extending the Bindings

### Understanding the FFI Layer

The FFI layer (`_ffi.py`) defines the C interface:

```python
# C type definitions
ffi.cdef("""
    typedef struct {
        size_t dimension;
        float *data;
    } GV_Vector;
    
    int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);
""")

# Load library
lib = ffi.dlopen("libGigaVector.so")
```

### Adding New Functions

To add a new function:

1. **Add C declaration to `_ffi.py`:**
```python
ffi.cdef("""
    int gv_db_new_function(GV_Database *db, const float *data);
""")
```

2. **Add Python wrapper in `_core.py`:**
```python
class Database:
    def new_function(self, data: Sequence[float]) -> int:
        """New function wrapper."""
        buf = ffi.new("float[]", list(data))
        rc = lib.gv_db_new_function(self._db, buf)
        if rc != 0:
            raise RuntimeError("gv_db_new_function failed")
        return rc
```

### Type Conversion Helpers

Common conversion patterns:

```python
# Python list to C float array
buf = ffi.new("float[]", list(vector))

# Python string to C string
c_str = key.encode("utf-8")

# Python dict to C metadata arrays
keys = [ffi.new("char[]", k.encode()) for k in metadata.keys()]
values = [ffi.new("char[]", v.encode()) for v in metadata.values()]
keys_c = ffi.new("const char * []", keys)
values_c = ffi.new("const char * []", values)

# C struct to Python dict
def _metadata_to_dict(meta_ptr):
    result = {}
    cur = meta_ptr
    while cur != ffi.NULL:
        key = ffi.string(cur.key).decode("utf-8")
        value = ffi.string(cur.value).decode("utf-8")
        result[key] = value
        cur = cur.next
    return result
```

### Error Handling Pattern

```python
def safe_call(func, *args):
    """Wrapper for C function calls with error handling."""
    rc = func(*args)
    if rc != 0:
        raise RuntimeError(f"{func.__name__} failed with code {rc}")
    return rc

# Usage
safe_call(lib.gv_db_add_vector, db, buf, dimension)
```

## Troubleshooting

### Common Issues

**1. "libGigaVector.so not found"**

**Cause:** Shared library not found

**Solutions:**
```python
# Check library path
from pathlib import Path
import gigavector._ffi as ffi_module
print(ffi_module._load_lib.__doc__)  # Shows search paths

# Build library first
# cd /path/to/GigaVector
# make lib
```

**2. "FFI type mismatch" errors**

**Cause:** C declaration doesn't match actual C function

**Solution:** Ensure `_ffi.py` declarations match C headers exactly

**3. Memory leaks**

**Cause:** Not closing databases or holding references

**Solution:**
```python
# Always use context managers
with Database.open("db.db", 128) as db:
    # Use database
    pass

# Or explicitly close
db = Database.open("db.db", 128)
try:
    # Use database
    pass
finally:
    db.close()
```

**4. Slow performance**

**Cause:** Too many individual FFI calls

**Solution:**
```python
# Use batch operations
db.add_vectors(vectors)  # One FFI call
# Instead of:
for vec in vectors:
    db.add_vector(vec)  # Many FFI calls
```

**5. Type errors**

**Cause:** Wrong Python types passed to C functions

**Solution:**
```python
# Ensure vectors are lists of floats
vector = [float(x) for x in vector]  # Convert to floats

# Ensure metadata values are strings
metadata = {k: str(v) for k, v in metadata.items()}
```

### Debugging Tips

**Enable FFI debugging:**
```python
import cffi
cffi.verifier.set_verbose(True)
```

**Check library loading:**
```python
from gigavector import _ffi
print(f"Library loaded: {_ffi.lib}")
```

**Inspect C objects:**
```python
# Database handle is an opaque pointer
db = Database.open("db.db", 128)
print(f"Database handle: {db._db}")  # C pointer (opaque)
```

**Monitor FFI calls:**
```python
import time

# Measure FFI overhead
start = time.time()
for _ in range(1000):
    db.add_vector(vector)
ffi_time = time.time() - start
print(f"FFI overhead: {ffi_time:.4f}s")
```

### Performance Optimization

**1. Minimize FFI Calls**
```python
# Good - single FFI call
db.add_vectors(vectors)

# Bad - many FFI calls
for vec in vectors:
    db.add_vector(vec)
```

**2. Pre-allocate Buffers (Advanced)**
```python
# For very high-performance scenarios
import cffi
ffi = cffi.FFI()

# Pre-allocate buffer
buf = ffi.new("float[]", dimension)
# Reuse buffer for multiple operations
```

**3. Use Appropriate Batch Sizes**
```python
# Optimal batch size depends on your use case
# Typically 100-1000 vectors per batch
batch_size = 500
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    db.add_vectors(batch)
```

## Advanced Modules

The Python bindings include several advanced modules for enterprise features.

### GPU Acceleration

```python
from gigavector import (
    gpu_available, gpu_device_count, gpu_get_device_info,
    GPUContext, GPUIndex, GPUConfig, GPUSearchParams
)

# Check GPU availability
if gpu_available():
    count = gpu_device_count()
    for i in range(count):
        info = gpu_get_device_info(i)
        print(f"GPU {i}: {info.name}, {info.total_memory // 1024**2} MB")

    # Create GPU index
    config = GPUConfig(device_id=0, use_float16=True)
    index = GPUIndex(dimension=128, config=config)

    # Add vectors and search
    index.add_vectors(vectors)
    results = index.search(query, k=10)
```

### HTTP REST Server

```python
from gigavector import Database, Server, ServerConfig, IndexType

db = Database.open(None, dimension=128, index=IndexType.HNSW)

# Configure server
config = ServerConfig(
    port=6969,
    bind_address="0.0.0.0",
    thread_pool_size=4,
    max_connections=100,
    enable_cors=True,
    cors_origins="*",
    enable_logging=True
)

# Start server
server = Server(db, config)
server.start()

# Check stats
stats = server.get_stats()
print(f"Total requests: {stats.total_requests}")

# Stop server
server.stop()
```

### Web Dashboard

Launch the built-in web dashboard with a single call.  The dashboard is a
pure-Python HTTP server (uses `http.server` from stdlib) — no libmicrohttpd
or any other C dependency is needed:

```python
from gigavector import Database, IndexType, serve_with_dashboard

db = Database.open(None, dimension=128, index=IndexType.HNSW)
server = serve_with_dashboard(db, port=6969)
# Open http://localhost:6969/dashboard
server.stop()
```

Or use the `DashboardServer` class directly:

```python
from gigavector.dashboard.server import DashboardServer

server = DashboardServer(db, port=6969)
server.start()
# ...
server.stop()
```

### BM25 Full-Text Search

```python
from gigavector import BM25Index, BM25Config

# Create index with custom parameters
config = BM25Config(k1=1.2, b=0.75)
index = BM25Index(config)

# Add documents
index.add_document(0, "First document about vectors")
index.add_document(1, "Second document about search")

# Search
results = index.search("vector search", k=5)
for r in results:
    print(f"Doc {r.doc_id}: {r.score}")

# Get statistics
stats = index.get_stats()
print(f"Documents: {stats.document_count}, Terms: {stats.term_count}")
```

### Hybrid Search

Combines vector similarity search with BM25 text search.

```python
from gigavector import (
    Database, BM25Index, HybridSearcher,
    HybridConfig, FusionType, IndexType
)

db = Database.open(None, dimension=128, index=IndexType.HNSW)
bm25 = BM25Index()

# Add data
for i, (vec, text) in enumerate(zip(vectors, documents)):
    db.add_vector(vec)
    bm25.add_document(i, text)

# Configure hybrid search
config = HybridConfig(
    fusion_type=FusionType.RRF,  # Reciprocal Rank Fusion
    vector_weight=0.6,
    text_weight=0.4,
    rrf_k=60
)

hybrid = HybridSearcher(db, bm25, config)

# Search with both vector and text
results = hybrid.search(query_vector, "query text", k=10)
for r in results:
    print(f"Index {r.vector_index}: combined={r.combined_score:.4f}")

# Can also do vector-only or text-only through hybrid
vector_results = hybrid.search_vector_only(query_vector, k=10)
text_results = hybrid.search_text_only("query text", k=10)
```

### Namespace Management (Multi-Tenancy)

```python
from gigavector import NamespaceManager, NamespaceConfig, NSIndexType

# Create manager (path for persistence, None for in-memory)
mgr = NamespaceManager("/data/namespaces")

# Create namespaces
config = NamespaceConfig(
    name="tenant_1",
    dimension=128,
    index_type=NSIndexType.HNSW,
    max_vectors=1000000
)
ns1 = mgr.create(config)

# Use namespace
ns1.add_vector([0.1] * 128)
results = ns1.search([0.1] * 128, k=5)
print(f"Namespace count: {ns1.count}")

# List all namespaces
names = mgr.list_namespaces()
print(f"Namespaces: {names}")

# Delete namespace
mgr.delete("tenant_1")
mgr.close()
```

### TTL (Time-to-Live)

```python
from gigavector import TTLManager, TTLConfig

config = TTLConfig(
    default_ttl_seconds=3600,      # 1 hour
    cleanup_interval_seconds=60,   # Check every minute
    lazy_expiration=True,
    max_expired_per_cleanup=1000
)

ttl = TTLManager(config)

# Set TTL for specific vectors
ttl.set_ttl(0, 1800)  # Vector 0 expires in 30 minutes
ttl.set_ttl(1, 7200)  # Vector 1 expires in 2 hours

# Check if expired
if ttl.is_expired(0):
    print("Vector 0 has expired")

# Get remaining time
remaining = ttl.get_remaining_ttl(1)
print(f"Vector 1 expires in {remaining} seconds")

# Remove TTL
ttl.remove_ttl(1)

# Manual cleanup
expired = ttl.cleanup_expired()
print(f"Cleaned up {expired} expired vectors")

# Background cleanup (requires database)
ttl.start_background_cleanup(db)
ttl.stop_background_cleanup()

ttl.close()
```

### Authentication

```python
from gigavector import AuthManager, AuthConfig, AuthType, JWTConfig

# API Key authentication
config = AuthConfig(auth_type=AuthType.API_KEY)
auth = AuthManager(config)

# Generate key
key, key_id = auth.generate_api_key("My App", expires_at=0)  # 0 = never expires

# Authenticate
result, identity = auth.authenticate(key)
if result == AuthResult.SUCCESS:
    print(f"Authenticated as: {identity.key_id}")

# List keys
keys = auth.list_api_keys()
for k in keys:
    print(f"Key: {k.key_id}, Description: {k.description}")

# Revoke key
auth.revoke_api_key(key_id)

auth.close()
```

### Backup and Restore

```python
from gigavector import (
    backup_create, backup_restore, backup_verify,
    backup_read_header, backup_restore_to_db,
    BackupOptions, RestoreOptions, BackupCompression
)

# Create backup
options = BackupOptions(
    compression=BackupCompression.ZSTD,
    include_metadata=True,
    include_index=True
)
result = backup_create(db, "backup.gvb", options)
print(f"Backed up {result.vectors_backed_up} vectors")
print(f"Size: {result.compressed_size} bytes")

# Read backup header
header = backup_read_header("backup.gvb")
print(f"Version: {header.version}")
print(f"Vectors: {header.vector_count}")
print(f"Dimension: {header.dimension}")

# Verify backup integrity
is_valid = backup_verify("backup.gvb")

# Restore to file
restore_opts = RestoreOptions(verify_checksums=True)
new_db = backup_restore("backup.gvb", "restored.db", restore_opts)

# Or restore to existing database
backup_restore_to_db("backup.gvb", existing_db, restore_opts)
```

### Sharding

```python
from gigavector import ShardManager, ShardConfig, ShardStrategy

config = ShardConfig(
    num_shards=4,
    strategy=ShardStrategy.HASH,
    data_path="/data/shards"
)

mgr = ShardManager(config)

# Add vector (automatically routed to correct shard)
mgr.add_vector([0.1] * 128)

# Search across all shards
results = mgr.search([0.1] * 128, k=10)

# Get shard info
for shard in mgr.list_shards():
    print(f"Shard {shard.id}: {shard.vector_count} vectors")

mgr.close()
```

### Replication

```python
from gigavector import (
    ReplicationManager, ReplicationConfig, ReplicationRole
)

# Leader configuration
config = ReplicationConfig(
    node_id="node-1",
    listen_address="0.0.0.0:5000",
    leader_address=None  # This is the leader
)

repl = ReplicationManager(db, config)
repl.start()

# Check role
role = repl.get_role()
print(f"Role: {role.name}")  # LEADER or FOLLOWER

# Get replication stats
stats = repl.get_stats()
print(f"Replicas: {stats.replica_count}")
print(f"Sync lag: {stats.sync_lag}")

repl.stop()
repl.close()
```

### Cluster Management

```python
from gigavector import Cluster, ClusterConfig, NodeRole

config = ClusterConfig(
    node_id="node-1",
    listen_address="0.0.0.0:6000",
    seed_nodes="node-2:6000,node-3:6000",
    role=NodeRole.DATA,
    heartbeat_interval_ms=1000
)

cluster = Cluster(config)
cluster.start()

# Get cluster info
local = cluster.get_local_node()
print(f"Local node: {local.node_id}, State: {local.state.name}")

# List all nodes
nodes = cluster.list_nodes()
for node in nodes:
    print(f"Node {node.node_id}: {node.state.name}")

cluster.stop()
cluster.close()
```

## Graph Database and Knowledge Graph

### Graph Database

```python
from gigavector import GraphDB, GraphDBConfig

# Create a graph database
g = GraphDB(GraphDBConfig(node_bucket_count=4096))

# Add nodes with labels and properties
alice = g.add_node("Person")
bob = g.add_node("Person")
g.set_node_prop(alice, "name", "Alice")
g.set_node_prop(bob, "name", "Bob")

# Add edges with labels and weights
g.add_edge(alice, bob, "KNOWS", weight=1.0)

# Traversal
visited = g.bfs(alice, max_depth=3)        # breadth-first search
visited = g.dfs(alice, max_depth=3)        # depth-first search
path = g.shortest_path(alice, bob)          # Dijkstra
print(f"Path: {path.node_ids}, weight: {path.total_weight}")

# Analytics
pr = g.pagerank(alice, iterations=20, damping=0.85)
cc = g.clustering_coefficient(alice)
components = g.connected_components()
degree = g.degree(alice)

# Persistence
g.save("social.gvgr")
g2 = GraphDB.load("social.gvgr")
```

### Knowledge Graph

```python
from gigavector import KnowledgeGraph, KGConfig

# Create with embedding support
kg = KnowledgeGraph(KGConfig(embedding_dimension=128))

# Add entities with optional embeddings
alice = kg.add_entity("Alice", "Person", embedding=[0.1] * 128)
bob = kg.add_entity("Bob", "Person", embedding=[0.2] * 128)
company = kg.add_entity("Anthropic", "Company", embedding=[0.3] * 128)

# Add relations (SPO triples)
kg.add_relation(alice, "works_at", company, weight=1.0)
kg.add_relation(bob, "works_at", company, weight=0.9)
kg.add_relation(alice, "knows", bob, weight=0.8)

# Query triples (None = wildcard)
triples = kg.query_triples(predicate="works_at")
for t in triples:
    print(f"{t.subject_name} --{t.predicate}--> {t.object_name}")

# Semantic search over entity embeddings
results = kg.search_similar([0.15] * 128, k=5)

# Hybrid search (vector + type/predicate filters)
results = kg.hybrid_search([0.1] * 128, entity_type="Person",
                            predicate_filter="works_at", k=10)

# Entity resolution (find or create)
resolved = kg.resolve_entity("Alice Smith", "Person", embedding=[0.1] * 128)

# Link prediction
predictions = kg.predict_links(alice, k=5)

# Graph traversal
neighbors = kg.get_neighbors(alice)
path = kg.shortest_path(alice, company)
subgraph = kg.extract_subgraph(center=alice, radius=2)

# Analytics
stats = kg.get_stats()
centrality = kg.entity_centrality(alice)

# Persistence
kg.save("knowledge.gvkg")
kg2 = KnowledgeGraph.load("knowledge.gvkg")
```

---

## Summary

- GigaVector Python bindings use CFFI for high-performance C integration
- Always use context managers for resource management
- Use batch operations to minimize FFI overhead
- Handle errors appropriately with try/except
- Monitor resource usage for large datasets
- The C library manages vector data; Python provides convenient wrappers

**Available Modules:**
- Core: Database, Vector, SearchHit, IndexType, DistanceType
- Configuration: HNSWConfig, IVFPQConfig, ScalarQuantConfig
- LLM Integration: LLM, LLMConfig, EmbeddingService
- Memory: MemoryLayer, ContextGraph
- GPU: GPUContext, GPUIndex, GPUConfig
- Server: Server, ServerConfig, serve_with_dashboard
- Search: BM25Index, HybridSearcher
- Storage: NamespaceManager, TTLManager, ShardManager
- High Availability: ReplicationManager, Cluster
- Security: AuthManager, backup_create, backup_restore
- Graph: GraphDB, GraphDBConfig, GraphPath
- Knowledge Graph: KnowledgeGraph, KGConfig, KGTriple, KGSearchResult, KGSubgraph, KGStats

For more information, see:
- [Usage Guide](usage.md) for general usage patterns
- [C API Guide](c_api_guide.md) for understanding the underlying C API
- [Performance Tuning Guide](performance.md) for optimization tips

