# GigaVector Usage Guide

This guide provides comprehensive instructions for using GigaVector in your applications, covering both Python and C APIs.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Choosing the Right Index](#choosing-the-right-index)
3. [Python Usage](#python-usage)
4. [C API Usage](#c-api-usage)
5. [HTTP REST Server](#http-rest-server)
6. [CLI Tools](#cli-tools)
7. [Common Patterns](#common-patterns)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

**Python (from PyPI):**
```bash
pip install gigavector
```

**Python (from source):**
```bash
cd python
pip install .
```

**C Library (from source):**
```bash
# Using Make
make lib

# Using CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Quick Start Example

**Python:**
```python
from gigavector import Database, DistanceType, IndexType

# Create a database
with Database.open("my_db.db", dimension=128, index=IndexType.HNSW) as db:
    # Add a vector
    db.add_vector([0.1] * 128, metadata={"id": "1", "category": "A"})
    
    # Search
    results = db.search([0.1] * 128, k=5, distance=DistanceType.EUCLIDEAN)
    for hit in results:
        print(f"Distance: {hit.distance}, Metadata: {hit.vector.metadata}")
```

**C:**
```c
#include "gigavector/gigavector.h"

// Create database
GV_Database *db = gv_db_open("my_db.db", 128, GV_INDEX_TYPE_HNSW);

// Add vector
float data[128];
for (int i = 0; i < 128; i++) data[i] = 0.1f;
gv_db_add_vector_with_metadata(db, data, 128, "id", "1");

// Search
float query[128];
for (int i = 0; i < 128; i++) query[i] = 0.1f;
GV_SearchResult results[5];
int found = gv_db_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);

// Process results
for (int i = 0; i < found; i++) {
    printf("Distance: %f\n", results[i].distance);
}

gv_db_close(db);
```

## Choosing the Right Index

GigaVector supports multiple index types. Choose based on your requirements:

| Index Type | Best For | Dataset Size | Dimensions | Search Speed | Memory |
|------------|----------|--------------|------------|--------------|--------|
| **KD-Tree** | Exact search, small datasets | < 1M | < 100 | Fast (small) | Low |
| **HNSW** | General purpose, large datasets | 1K - 1B+ | Any | Very Fast | Medium |
| **IVFPQ** | Very large datasets, memory constrained | 100K+ | 64+ | Fast | Very Low |
| **Sparse** | Sparse vectors, text embeddings | Any | High (sparse) | Fast | Low |

**Quick Decision Guide:**
- **Small dataset (< 100K vectors) and need exact results?** → KD-Tree
- **Large dataset and need fast approximate search?** → HNSW
- **Very large dataset and memory is critical?** → IVFPQ
- **Sparse vectors (most dimensions are zero)?** → Sparse Index

See the [Performance Tuning Guide](performance.md) for detailed recommendations.

## Python Usage

### Creating a Database

```python
from gigavector import Database, IndexType

# In-memory database
db = Database.open(None, dimension=128, index=IndexType.HNSW)

# Persistent database
db = Database.open("data.db", dimension=128, index=IndexType.HNSW)

# Auto-select index type
db = Database.open_auto("data.db", dimension=128, expected_count=1000000)

# Memory-mapped read-only database
db = Database.open_mmap("data.db", dimension=128)
```

### Adding Vectors

```python
# Simple vector
db.add_vector([0.1, 0.2, 0.3, ...])

# With metadata
db.add_vector(
    [0.1, 0.2, 0.3, ...],
    metadata={"id": "123", "category": "electronics", "price": "99.99"}
)

# Batch insertion
vectors = [[random.random() for _ in range(128)] for _ in range(1000)]
db.add_vectors(vectors)
```

### Searching

```python
from gigavector import DistanceType

# Basic search
results = db.search([0.1] * 128, k=10, distance=DistanceType.EUCLIDEAN)

# Filtered search
results = db.search(
    [0.1] * 128, k=10,
    filter_metadata=("category", "electronics")
)

# Advanced filter expression
results = db.search_with_filter_expr(
    [0.1] * 128, k=10,
    filter_expr='category == "electronics" AND price >= "50"'
)

# Range search (find all within radius)
results = db.range_search(
    [0.1] * 128, radius=0.5, max_results=100
)

# Batch search
queries = [[random.random() for _ in range(128)] for _ in range(10)]
all_results = db.search_batch(queries, k=5)
```

### Managing Data

```python
# Update a vector
db.update_vector(0, [0.2, 0.3, 0.4, ...])

# Update metadata
db.update_metadata(0, {"price": "149.99", "updated": "true"})

# Delete a vector
db.delete_vector(0)

# Save database
db.save("backup.db")
```

### Configuration

```python
from gigavector import HNSWConfig, IVFPQConfig

# HNSW with custom config
hnsw_config = HNSWConfig(
    M=32,
    ef_construction=200,
    ef_search=50,
    use_binary_quant=True
)
db = Database.open("db.db", 128, IndexType.HNSW, hnsw_config=hnsw_config)

# IVFPQ with custom config
ivfpq_config = IVFPQConfig(
    nlist=256,
    m=16,
    nprobe=16,
    default_rerank=32
)
db = Database.open(None, 128, IndexType.IVFPQ, ivfpq_config=ivfpq_config)
db.train_ivfpq(training_data)  # Must train before use
```

### Resource Management

```python
# Context manager (recommended)
with Database.open("db.db", 128) as db:
    db.add_vector([0.1] * 128)
    # Automatically closed on exit

# Manual management
db = Database.open("db.db", 128)
try:
    # Use database
    pass
finally:
    db.close()

# Resource limits
db.set_resource_limits(
    max_memory_bytes=1024 * 1024 * 1024,  # 1GB
    max_vectors=1000000,
    max_concurrent_operations=100
)
```

### Monitoring and Statistics

```python
# Basic statistics
stats = db.get_stats()
print(f"Total inserts: {stats.total_inserts}")
print(f"Total queries: {stats.total_queries}")

# Detailed statistics
detailed = db.get_detailed_stats()
print(f"QPS: {detailed['queries_per_second']}")
print(f"Memory: {detailed['memory']['total_bytes'] / 1024 / 1024:.2f} MB")
print(f"Recall: {detailed['recall']['avg_recall']:.2%}")

# Health check
health = db.health_check()
if health == 0:
    print("Database is healthy")
elif health == -1:
    print("Database is degraded")
else:
    print("Database is unhealthy")
```

## C API Usage

### Basic Operations

```c
#include "gigavector/gigavector.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Create database
    GV_Database *db = gv_db_open("example.db", 128, GV_INDEX_TYPE_HNSW);
    if (!db) {
        fprintf(stderr, "Failed to create database\n");
        return 1;
    }
    
    // Add vector
    float data[128];
    for (int i = 0; i < 128; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
    
    int rc = gv_db_add_vector_with_metadata(
        db, data, 128, "id", "1"
    );
    if (rc != 0) {
        fprintf(stderr, "Failed to add vector\n");
        gv_db_close(db);
        return 1;
    }
    
    // Search
    float query[128];
    for (int i = 0; i < 128; i++) {
        query[i] = (float)rand() / RAND_MAX;
    }
    
    GV_SearchResult results[10];
    int found = gv_db_search(
        db, query, 10, results, GV_DISTANCE_EUCLIDEAN
    );
    
    if (found > 0) {
        printf("Found %d results:\n", found);
        for (int i = 0; i < found; i++) {
            printf("  Distance: %f\n", results[i].distance);
        }
    }
    
    // Save and close
    gv_db_save(db, "example.db");
    gv_db_close(db);
    return 0;
}
```

### Error Handling

```c
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
if (!db) {
    // Handle error - check errno or log message
    perror("gv_db_open");
    return 1;
}

int rc = gv_db_add_vector(db, data, 128);
if (rc != 0) {
    // Handle error
    fprintf(stderr, "Failed to add vector\n");
    gv_db_close(db);
    return 1;
}
```

### Memory Management

```c
// Vectors are managed by the database
// You don't need to free vectors returned from search results
// The database owns all vector data

// Always close the database when done
gv_db_close(db);
```

### Configuration

```c
// HNSW configuration
GV_HNSWConfig hnsw_config = {
    .M = 32,
    .efConstruction = 200,
    .efSearch = 50,
    .use_binary_quant = 1,
    .quant_rerank = 20
};

GV_Database *db = gv_db_open_with_hnsw_config(
    "db.db", 128, GV_INDEX_TYPE_HNSW, &hnsw_config
);

// IVFPQ configuration
GV_IVFPQConfig ivfpq_config = {
    .nlist = 256,
    .m = 16,
    .nbits = 8,
    .nprobe = 16,
    .default_rerank = 32
};

GV_Database *db_ivfpq = gv_db_open_with_ivfpq_config(
    "ivfpq.db", 128, GV_INDEX_TYPE_IVFPQ, &ivfpq_config
);

// Train IVFPQ
float training_data[1000 * 128];
// ... populate training data ...
gv_db_ivfpq_train(db_ivfpq, training_data, 1000, 128);
```

## HTTP REST Server

Start GigaVector as an HTTP server for language-agnostic access.

### Starting the Server

**C:**
```c
#include "gigavector/gv_server.h"

GV_Database *db = gv_db_open("vectors.db", 128, GV_INDEX_TYPE_HNSW);

GV_ServerConfig config;
gv_server_config_init(&config);
config.port = 8080;
config.enable_cors = 1;

GV_Server *server = gv_server_create(db, &config);
gv_server_start(server);

// Server runs until stopped
gv_server_stop(server);
gv_server_destroy(server);
gv_db_close(db);
```

### API Examples

**Health check:**
```bash
curl http://localhost:8080/health
```

**Add vector:**
```bash
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "metadata": {"id": "1"}}'
```

**Search:**
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "k": 10}'
```

**Get stats:**
```bash
curl http://localhost:8080/stats
```

---

## CLI Tools

GigaVector includes command-line tools for database management.

### gvbackup - Create backups

```bash
# Basic backup
gvbackup mydb.db backup.gvb

# Compressed backup
gvbackup --compress mydb.db backup.gvb.gz

# Include WAL
gvbackup --include-wal mydb.db backup.gvb
```

### gvrestore - Restore from backup

```bash
# Restore to new database
gvrestore backup.gvb restored.db

# Restore with verification
gvrestore --verify backup.gvb restored.db
```

### gvinspect - Inspect database

```bash
# Show database info
gvinspect mydb.db

# Output:
# Database: mydb.db
# Version: 0.7.0
# Vectors: 1,234,567
# Dimension: 128
# Index: HNSW (M=16, ef=200)
# Size: 156.2 MB
```

---

## Common Patterns

### Pattern 1: Building an Index Incrementally

```python
# Python
with Database.open("index.db", dimension=128) as db:
    for batch in data_batches:
        for vector in batch:
            db.add_vector(vector)
        # Periodic saves
        if batch_num % 100 == 0:
            db.save("checkpoint.db")
```

### Pattern 2: Filtered Search

```python
# Search for products in a specific category
results = db.search(
    query_vector, k=20,
    filter_metadata=("category", "electronics")
)

# Complex filtering
results = db.search_with_filter_expr(
    query_vector, k=20,
    filter_expr='category == "electronics" AND price >= "100" AND rating >= "4.0"'
)
```

### Pattern 3: Batch Processing

```python
# Process queries in batches
query_batches = [queries[i:i+100] for i in range(0, len(queries), 100)]
all_results = []

for batch in query_batches:
    results = db.search_batch(batch, k=10)
    all_results.extend(results)
```

### Pattern 4: Monitoring Performance

```python
import time

# Measure search latency
start = time.time()
results = db.search(query, k=10)
latency_ms = (time.time() - start) * 1000
print(f"Search latency: {latency_ms:.2f} ms")

# Record metrics
db.record_latency(int(latency_ms * 1000), is_insert=False)
```

## Best Practices

### 1. Choose the Right Index Type

- Use KD-Tree for small datasets requiring exact search
- Use HNSW for general-purpose large-scale search
- Use IVFPQ for very large datasets with memory constraints
- Use Sparse index for sparse vectors

### 2. Dimension Consistency

**Always ensure vector dimensions match:**
```python
# Good
db = Database.open("db.db", dimension=128)
db.add_vector([0.1] * 128)  # Correct dimension

# Bad
db.add_vector([0.1] * 64)  # Wrong dimension - will raise ValueError
```

### 3. Resource Management

**Always use context managers or try/finally:**
```python
# Good
with Database.open("db.db", 128) as db:
    # Use database
    pass

# Also good
db = Database.open("db.db", 128)
try:
    # Use database
    pass
finally:
    db.close()
```

### 4. Batch Operations

**Use batch operations when possible:**
```python
# Good - batch insertion
db.add_vectors(vectors)

# Good - batch search
results = db.search_batch(queries, k=10)

# Less efficient - individual operations
for vec in vectors:
    db.add_vector(vec)
```

### 5. Metadata Usage

**Use metadata for filtering and organization:**
```python
# Good - structured metadata
db.add_vector(
    vector,
    metadata={
        "id": "12345",
        "category": "electronics",
        "price": "99.99",
        "rating": "4.5"
    }
)

# Then filter efficiently
results = db.search(
    query, k=10,
    filter_metadata=("category", "electronics")
)
```

### 6. Persistence Strategy

**Save periodically for large datasets:**
```python
with Database.open("db.db", 128) as db:
    for i, vector in enumerate(vectors):
        db.add_vector(vector)
        
        # Save every 10K vectors
        if i % 10000 == 0:
            db.save("db.db")
            print(f"Saved checkpoint at {i} vectors")
```

### 7. Error Handling

**Always check return values and handle errors:**
```python
# Python
try:
    db = Database.open("db.db", 128)
    db.add_vector(vector)
except RuntimeError as e:
    print(f"Error: {e}")
    # Handle error
```

```c
// C
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
if (!db) {
    // Handle error
    return 1;
}

int rc = gv_db_add_vector(db, data, 128);
if (rc != 0) {
    // Handle error
    gv_db_close(db);
    return 1;
}
```

## Troubleshooting

### Common Issues

**1. "Vector dimension mismatch" error**
- **Cause:** Vector dimension doesn't match database dimension
- **Solution:** Ensure all vectors have the same dimension as specified when creating the database

**2. "Database open failed" error**
- **Cause:** Invalid path, permissions, or corrupted database
- **Solution:** Check file permissions, ensure directory exists, verify database file integrity

**3. Low search recall**
- **Cause:** Index parameters not tuned for your data
- **Solution:** Increase `efSearch` (HNSW) or `nprobe` (IVFPQ), see [Performance Guide](performance.md)

**4. High memory usage**
- **Cause:** Large dataset with exact storage
- **Solution:** Use IVFPQ with scalar quantization, or reduce HNSW `M` parameter

**5. Slow search performance**
- **Cause:** Index not optimized, too many results, or wrong index type
- **Solution:** 
  - Enable binary quantization (HNSW)
  - Reduce `k` if possible
  - Consider IVFPQ for very large datasets
  - Check SIMD optimizations are enabled

### Getting Help

- Check the [Performance Tuning Guide](performance.md) for optimization tips
- Review [Basic Usage Examples](examples/basic_usage.md) for common patterns
- See [Advanced Features](examples/advanced_features.md) for complex scenarios
- Check API documentation for detailed function signatures

### Debugging Tips

**Enable detailed logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Monitor database health:**
```python
health = db.health_check()
if health != 0:
    stats = db.get_detailed_stats()
    print(f"Health status: {health}")
    print(f"Memory usage: {stats['memory']['total_bytes']}")
    print(f"Deleted ratio: {stats['deleted_ratio']}")
```

**Check resource limits:**
```python
limits = db.get_resource_limits()
print(f"Memory limit: {limits['max_memory_bytes']}")
print(f"Vector limit: {limits['max_vectors']}")
```

For more information, see the [Python Bindings Guide](python_bindings.md) and [C API Guide](c_api_guide.md).

