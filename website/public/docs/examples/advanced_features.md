# Advanced Features Examples

This guide demonstrates advanced features and optimization techniques for GigaVector.

## Table of Contents

1. [Advanced Index Configuration](#advanced-index-configuration)
2. [Range Search](#range-search)
3. [Sparse Vectors](#sparse-vectors)
4. [Performance Optimization](#performance-optimization)
5. [Concurrent Operations](#concurrent-operations)
6. [Monitoring and Observability](#monitoring-and-observability)

## Advanced Index Configuration

### Fine-Tuning HNSW

```python
from gigavector import Database, IndexType, HNSWConfig

# High-quality HNSW configuration
high_quality_config = HNSWConfig(
    M=48,                    # More connections for better recall
    ef_construction=400,     # Higher construction quality
    ef_search=100,           # Higher search quality
    use_binary_quant=True,   # Enable binary quantization
    quant_rerank=50,         # Rerank top 50 candidates
    use_acorn=True,          # Enable ACORN for filtered search
    acorn_hops=2             # 2-hop exploration
)

with Database.open(
    "high_quality.db", dimension=128, index=IndexType.HNSW,
    hnsw_config=high_quality_config
) as db:
    # Add vectors
    for i in range(1000000):
        vector = [random.random() for _ in range(128)]
        db.add_vector(vector)
    
    # Search with high recall
    hits = db.search(query, k=10)
```

### Optimized IVFPQ Configuration

```python
from gigavector import Database, IndexType, IVFPQConfig, ScalarQuantConfig

# Memory-optimized IVFPQ
memory_optimized_config = IVFPQConfig(
    nlist=1024,              # More centroids for large datasets
    m=32,                    # More subquantizers for accuracy
    nbits=8,                 # Standard quantization
    nprobe=64,               # Probe more lists for recall
    train_iters=20,          # More training iterations
    default_rerank=64,       # Larger rerank pool
    use_scalar_quant=True,   # Enable scalar quantization
    scalar_quant_config=ScalarQuantConfig(
        bits=8,              # 8-bit scalar quantization
        per_dimension=False  # Global quantization
    ),
    oversampling_factor=2.0  # 2x oversampling for recall
)

with Database.open(
    None, dimension=256, index=IndexType.IVFPQ,
    ivfpq_config=memory_optimized_config
) as db:
    # Train with large dataset
    training_data = [
        [random.random() for _ in range(256)]
        for _ in range(10000)
    ]
    db.train_ivfpq(training_data)
    
    # Add millions of vectors
    for i in range(10000000):
        vector = [random.random() for _ in range(256)]
        db.add_vector(vector)
```

## Range Search

Range search finds all vectors within a specified distance threshold.

### C API

```c
// Range search: find all vectors within radius 0.5
float radius = 0.5f;
GV_SearchResult results[1000];  // Pre-allocated results array
size_t max_results = 1000;

int found = gv_db_range_search(
    db, query, 128, radius, results, max_results,
    GV_DISTANCE_EUCLIDEAN
);

printf("Found %d vectors within radius %f\n", found, radius);
for (int i = 0; i < found; i++) {
    printf("  Distance: %f, Index: %zu\n",
           results[i].distance, results[i].vector_index);
}
```

### Python API

```python
# Range search
radius = 0.5
hits = db.range_search(query, radius=radius, distance=DistanceType.EUCLIDEAN)

print(f"Found {len(hits)} vectors within radius {radius}")
for hit in hits:
    print(f"  Distance: {hit.distance}")
```

### Filtered Range Search

```python
# Range search with metadata filter
hits = db.range_search(
    query, radius=0.5, distance=DistanceType.EUCLIDEAN,
    filter_metadata=("category", "electronics")
)
```

## Sparse Vectors

GigaVector supports sparse vectors for efficient storage of high-dimensional sparse data.

### C API

```c
// Create sparse vector
GV_SparseVector *sparse = gv_sparse_vector_create(1000);  // 1000 dimensions

// Set non-zero values
gv_sparse_vector_set(sparse, 10, 0.5f);   // dimension 10 = 0.5
gv_sparse_vector_set(sparse, 50, 0.3f);   // dimension 50 = 0.3
gv_sparse_vector_set(sparse, 100, 0.8f);   // dimension 100 = 0.8

// Add to sparse index
GV_Database *db = gv_db_open("sparse.db", 1000, GV_INDEX_TYPE_SPARSE);
gv_db_add_sparse_vector(db, sparse);

// Search
GV_SparseVector *query_sparse = gv_sparse_vector_create(1000);
gv_sparse_vector_set(query_sparse, 10, 0.6f);
gv_sparse_vector_set(query_sparse, 50, 0.4f);

GV_SearchResult results[10];
int found = gv_db_search_sparse(db, query_sparse, 10, results, GV_DISTANCE_EUCLIDEAN);

gv_sparse_vector_destroy(sparse);
gv_sparse_vector_destroy(query_sparse);
```

### Python API

```python
from gigavector import Database, IndexType

# Create sparse database
with Database.open("sparse.db", dimension=1000, index=IndexType.SPARSE) as db:
    # Create sparse vector using indices and values
    indices = [10, 50, 100]   # non-zero dimensions
    values = [0.5, 0.3, 0.8]  # corresponding values

    db.add_sparse_vector(indices, values)

    # Search with sparse query
    query_indices = [10, 50]
    query_values = [0.6, 0.4]
    hits = db.search_sparse(query_indices, query_values, k=10)
```

## Graph Database and Knowledge Graph

GigaVector includes a full property graph database and a knowledge graph layer with vector embeddings.

### Building a Social Graph

```c
#include "gigavector/gv_graph_db.h"

GV_GraphDB *g = gv_graph_create(NULL);

// Create nodes
uint64_t alice = gv_graph_add_node(g, "Person");
uint64_t bob = gv_graph_add_node(g, "Person");
uint64_t charlie = gv_graph_add_node(g, "Person");
gv_graph_set_node_prop(g, alice, "name", "Alice");
gv_graph_set_node_prop(g, bob, "name", "Bob");
gv_graph_set_node_prop(g, charlie, "name", "Charlie");

// Create relationships
gv_graph_add_edge(g, alice, bob, "KNOWS", 1.0f);
gv_graph_add_edge(g, bob, charlie, "KNOWS", 1.0f);
gv_graph_add_edge(g, alice, charlie, "FRIENDS", 0.5f);

// Find shortest path
GV_GraphPath path;
if (gv_graph_shortest_path(g, alice, charlie, &path) == 0) {
    printf("Path length: %zu, weight: %.2f\n", path.length, path.total_weight);
    gv_graph_free_path(&path);
}

// Compute PageRank
float pr = gv_graph_pagerank(g, alice, 20, 0.85f);
printf("Alice PageRank: %.4f\n", pr);

// Connected components
uint64_t comps[3];
int num_comps = gv_graph_connected_components(g, comps, 3);
printf("Components: %d\n", num_comps);

gv_graph_save(g, "social.gvgr");
gv_graph_destroy(g);
```

### Knowledge Graph with Semantic Search

```python
from gigavector import KnowledgeGraph, KGConfig

kg = KnowledgeGraph(KGConfig(embedding_dimension=128))

# Build knowledge base
alice = kg.add_entity("Alice", "Person", embedding=[0.1] * 128)
bob = kg.add_entity("Bob", "Person", embedding=[0.2] * 128)
company = kg.add_entity("Anthropic", "Company", embedding=[0.5] * 128)

kg.add_relation(alice, "works_at", company)
kg.add_relation(bob, "works_at", company)
kg.add_relation(alice, "manages", bob)

# SPO triple queries
triples = kg.query_triples(predicate="works_at")
print(f"Found {len(triples)} 'works_at' triples")

# Semantic search
results = kg.search_similar([0.15] * 128, k=3)
for r in results:
    print(f"  {r.name} ({r.type}): similarity={r.similarity:.3f}")

# Hybrid search (vector + graph filters)
results = kg.hybrid_search(
    [0.1] * 128, entity_type="Person", predicate_filter="works_at", k=5
)

# Entity resolution (deduplicate)
resolved = kg.resolve_entity("Alice Smith", "Person", embedding=[0.1] * 128)
print(f"Resolved to entity {resolved}")

# Link prediction
predictions = kg.predict_links(alice, k=3)
for p in predictions:
    print(f"  Predicted: {p.entity_a} -> {p.entity_b} ({p.confidence:.3f})")

# Subgraph extraction
subgraph = kg.extract_subgraph(center=alice, radius=2)
print(f"Subgraph: {len(subgraph.entity_ids)} entities, {len(subgraph.relation_ids)} relations")

# Persistence
kg.save("knowledge.gvkg")
kg2 = KnowledgeGraph.load("knowledge.gvkg")
```

---

## Performance Optimization

### Batch Operations

```python
# Efficient batch insertion
def batch_insert(db, vectors, batch_size=1000):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        for vec in batch:
            db.add_vector(vec)
        # Optional: save periodically
        if i % 10000 == 0:
            db.save("checkpoint.db")

with Database.open("batch.db", dimension=128) as db:
    vectors = [[random.random() for _ in range(128)] for _ in range(100000)]
    batch_insert(db, vectors)
```

### Pre-allocating Results

```c
// Pre-allocate results array to avoid repeated allocations
GV_SearchResult *results = malloc(1000 * sizeof(GV_SearchResult));
if (!results) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
}

// Reuse for multiple searches
for (int i = 0; i < 100; i++) {
    int found = gv_db_search(db, queries[i], 128, 10, results, GV_DISTANCE_EUCLIDEAN);
    // Process results...
}

free(results);
```

### Using Binary Quantization

```python
# Enable binary quantization for 2-3x speedup
hnsw_config = HNSWConfig(
    use_binary_quant=True,
    quant_rerank=20  # Rerank top 20 candidates with exact distance
)

with Database.open(
    "fast.db", dimension=128, index=IndexType.HNSW,
    hnsw_config=hnsw_config
) as db:
    # Fast searches with binary quantization
    hits = db.search(query, k=10)
```

### Memory Optimization with Scalar Quantization

```python
# Enable scalar quantization for memory reduction
ivfpq_config = IVFPQConfig(
    use_scalar_quant=True,
    scalar_quant_config=ScalarQuantConfig(
        bits=8,              # 8-bit quantization (2x memory reduction)
        per_dimension=False  # Global quantization
    )
)

with Database.open(
    None, dimension=256, index=IndexType.IVFPQ,
    ivfpq_config=ivfpq_config
) as db:
    # Memory-efficient storage
    db.train_ivfpq(training_data)
    for i in range(10000000):
        db.add_vector(vector)
```

## Concurrent Operations

### Thread-Safe Usage

```python
import threading
from gigavector import Database

# Each thread should use its own database handle or synchronize access
def worker_thread(db_path, thread_id, vectors):
    with Database.open(db_path, dimension=128) as db:
        for vec in vectors:
            db.add_vector(vec, metadata={"thread": str(thread_id)})

# Create multiple threads
threads = []
for i in range(4):
    t = threading.Thread(
        target=worker_thread,
        args=("concurrent.db", i, vectors_per_thread[i])
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

### Read-Only Access

```c
// Open database in read-only mode for concurrent reads
GV_Database *db = gv_db_open("readonly.db", 128, GV_INDEX_TYPE_KDTREE);
// Multiple threads can safely read from the same database handle
// (assuming proper synchronization if needed)
```

## Monitoring and Observability

### Getting Detailed Statistics

```python
# Get basic statistics
stats = db.get_stats()
print(f"Total inserts: {stats.total_inserts}")
print(f"Total queries: {stats.total_queries}")

# Get memory usage
memory_bytes = db.get_memory_usage()
print(f"Memory usage: {memory_bytes / 1024 / 1024:.2f} MB")

# Get detailed statistics (includes latency histograms, QPS, etc.)
detailed = db.get_detailed_stats()
print(f"Queries per second: {detailed['queries_per_second']:.2f}")
```

### Performance Monitoring

```python
import time

# Measure search latency
start = time.time()
hits = db.search(query, k=10)
latency = (time.time() - start) * 1000  # milliseconds
print(f"Search latency: {latency:.2f} ms")

# Measure insertion throughput
start = time.time()
for i in range(1000):
    db.add_vector(vector)
throughput = 1000 / (time.time() - start)
print(f"Insertion throughput: {throughput:.0f} vectors/sec")
```

### Health Checks

```python
# Check database health
try:
    health_status = db.health_check()
    if health_status == 0:
        print("Database health: OK")
    elif health_status == -1:
        print("Warning: Database is degraded")
    else:
        print("Warning: Database is unhealthy")

    # Check memory usage
    memory_bytes = db.get_memory_usage()
    if memory_bytes > 10 * 1024 * 1024 * 1024:  # 10GB
        print("Warning: High memory usage")
except Exception as e:
    print(f"Database health check failed: {e}")
```

### Custom Monitoring

```c
// Monitor search performance
#include <time.h>

struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);

int found = gv_db_search(db, query, 128, k, results, GV_DISTANCE_EUCLIDEAN);

clock_gettime(CLOCK_MONOTONIC, &end);
double latency_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1000000.0;

printf("Search latency: %.2f ms, Found: %d\n", latency_ms, found);
```

## Advanced Patterns

### Incremental Index Building

```python
# Build index incrementally with periodic saves
with Database.open("incremental.db", dimension=128) as db:
    for batch_num in range(100):
        # Add batch of vectors
        for i in range(1000):
            vector = generate_vector()
            db.add_vector(vector)
        
        # Save checkpoint every 10 batches
        if batch_num % 10 == 0:
            db.save(f"checkpoint_{batch_num}.db")
            print(f"Checkpoint saved at batch {batch_num}")
```

### Multi-Index Strategy

```python
# Use different indexes for different use cases
# Fast approximate search with HNSW
hnsw_db = Database.open("hnsw.db", dimension=128, index=IndexType.HNSW)

# Exact search with KD-Tree for small subsets
kdtree_db = Database.open("kdtree.db", dimension=128, index=IndexType.KDTREE)

# Memory-efficient storage with IVFPQ
ivfpq_db = Database.open("ivfpq.db", dimension=128, index=IndexType.IVFPQ)
```

### Custom Distance Metrics

```python
# Use appropriate distance metric for your data
# Euclidean for general purpose
hits_euclidean = db.search(query, k=10, distance=DistanceType.EUCLIDEAN)

# Cosine for normalized vectors
hits_cosine = db.search(query, k=10, distance=DistanceType.COSINE)
```

These advanced features enable you to optimize GigaVector for your specific use case. For more details on performance tuning, see the [Performance Tuning Guide](../performance.md).

