# Basic Usage Examples

This guide demonstrates basic usage patterns for GigaVector in both C and Python.

## Table of Contents

1. [C API Examples](#c-api-examples)
2. [Python API Examples](#python-api-examples)
3. [Common Patterns](#common-patterns)

## C API Examples

### Creating a Database

```c
#include "gigavector/gigavector.h"

// Create a new database with KD-Tree index
GV_Database *db = gv_db_open("example.db", 128, GV_INDEX_TYPE_KDTREE);
if (!db) {
    fprintf(stderr, "Failed to create database\n");
    return 1;
}
```

### Adding Vectors

```c
// Create a vector
float data[128];
for (int i = 0; i < 128; i++) {
    data[i] = (float)rand() / RAND_MAX;
}

// Add vector without metadata
size_t index = gv_db_add_vector(db, data, 128);
if (index == (size_t)-1) {
    fprintf(stderr, "Failed to add vector\n");
}

// Add vector with metadata
GV_Metadata *metadata = gv_metadata_create();
gv_metadata_set(metadata, "category", "electronics");
gv_metadata_set(metadata, "price", "99.99");

index = gv_db_add_vector_with_metadata(db, data, 128, metadata);
gv_metadata_destroy(metadata);
```

### Searching for Nearest Neighbors

```c
// Prepare query vector
float query[128];
// ... populate query ...

// Search for 10 nearest neighbors
size_t k = 10;
GV_SearchResult results[10];
int found = gv_db_search(db, query, 128, k, results, GV_DISTANCE_EUCLIDEAN);

if (found > 0) {
    printf("Found %d neighbors:\n", found);
    for (int i = 0; i < found; i++) {
        printf("  Distance: %f, Index: %zu\n", 
               results[i].distance, results[i].vector_index);
    }
}
```

### Filtered Search

```c
// Search with metadata filter
int found = gv_db_search_filtered(
    db, query, 128, k, results, GV_DISTANCE_EUCLIDEAN,
    "category", "electronics"
);

printf("Found %d electronics items\n", found);
```

### Saving and Loading

```c
// Save database to disk
if (gv_db_save(db, "example.db") != 0) {
    fprintf(stderr, "Failed to save database\n");
}

// Close database
gv_db_close(db);

// Reopen database (WAL is automatically replayed)
db = gv_db_open("example.db", 128, GV_INDEX_TYPE_KDTREE);
```

### Using Different Index Types

```c
// HNSW index
GV_Database *db_hnsw = gv_db_open("hnsw.db", 128, GV_INDEX_TYPE_HNSW);

// IVFPQ index (requires training)
GV_Database *db_ivfpq = gv_db_open("ivfpq.db", 128, GV_INDEX_TYPE_IVFPQ);

// Training data
float training_data[256 * 128];  // 256 vectors of dimension 128
// ... populate training data ...

gv_db_train_ivfpq(db_ivfpq, training_data, 256);
```

## Python API Examples

### Basic Database Operations

```python
from gigavector import Database, DistanceType, IndexType

# Create a new database
with Database.open("example.db", dimension=128, index=IndexType.KDTREE) as db:
    # Add a vector
    vector = [0.1, 0.2, 0.3] * 42 + [0.4, 0.5]  # 128 dimensions
    db.add_vector(vector)
    
    # Add vector with metadata
    db.add_vector(
        vector,
        metadata={"category": "electronics", "price": "99.99"}
    )
    
    # Search for nearest neighbors
    query = [0.15, 0.25, 0.35] * 42 + [0.45, 0.55]
    hits = db.search(query, k=10, distance=DistanceType.EUCLIDEAN)
    
    for hit in hits:
        print(f"Distance: {hit.distance}")
        print(f"Metadata: {hit.vector.metadata}")
```

### Filtered Search

```python
# Search with metadata filter
hits = db.search(
    query, k=10, distance=DistanceType.EUCLIDEAN,
    filter_metadata=("category", "electronics")
)

print(f"Found {len(hits)} electronics items")
```

### Using HNSW Index

```python
from gigavector import HNSWConfig

# Configure HNSW
hnsw_config = HNSWConfig(
    M=32,                    # More connections for better quality
    ef_construction=200,     # Higher quality construction
    ef_search=50,            # Search quality
    use_binary_quant=True,   # Enable binary quantization
    quant_rerank=20          # Rerank top 20 candidates
)

with Database.open(
    "hnsw.db", dimension=128, index=IndexType.HNSW,
    hnsw_config=hnsw_config
) as db:
    # Add vectors
    for i in range(10000):
        vector = [random.random() for _ in range(128)]
        db.add_vector(vector, metadata={"id": str(i)})
    
    # Search
    hits = db.search(query, k=10)
```

### Using IVFPQ Index

```python
from gigavector import IVFPQConfig, ScalarQuantConfig

# Configure IVFPQ
ivfpq_config = IVFPQConfig(
    nlist=256,              # Number of coarse centroids
    m=16,                   # Number of subquantizers
    nbits=8,                # Bits per code
    nprobe=16,              # Lists to probe
    default_rerank=32,      # Rerank pool size
    use_scalar_quant=True,  # Enable scalar quantization
    scalar_quant_config=ScalarQuantConfig(bits=8)
)

with Database.open(
    None, dimension=128, index=IndexType.IVFPQ,
    ivfpq_config=ivfpq_config
) as db:
    # Prepare training data
    training_data = [
        [random.random() for _ in range(128)]
        for _ in range(1000)
    ]
    
    # Train IVFPQ
    db.train_ivfpq(training_data)
    
    # Add vectors
    for i in range(100000):
        vector = [random.random() for _ in range(128)]
        db.add_vector(vector)
    
    # Search
    hits = db.search(query, k=10)
```

### Persistence

```python
# Save database
with Database.open("example.db", dimension=128) as db:
    # ... add vectors ...
    db.save("example.db")  # Create snapshot

# Reopen (WAL is automatically replayed)
with Database.open("example.db", dimension=128) as db:
    # Database is restored with all vectors
    hits = db.search(query, k=10)
```

### Auto Index Selection

```python
# Let GigaVector choose the best index type
with Database.open_auto(
    "auto.db", dimension=128, expected_count=1000000
) as db:
    # Database automatically uses HNSW for large datasets
    db.add_vector(vector)
    hits = db.search(query, k=10)
```

## Common Patterns

### Batch Insertion

```python
# Efficient batch insertion
vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
metadata_list = [{"id": str(i)} for i in range(10000)]

with Database.open("batch.db", dimension=128) as db:
    for vec, meta in zip(vectors, metadata_list):
        db.add_vector(vec, metadata=meta)
```

### Distance Metrics

```python
# Euclidean distance (default)
hits = db.search(query, k=10, distance=DistanceType.EUCLIDEAN)

# Cosine distance
hits = db.search(query, k=10, distance=DistanceType.COSINE)
```

### Getting Statistics

```python
stats = db.get_stats()
print(f"Total inserts: {stats.total_inserts}")
print(f"Total queries: {stats.total_queries}")

# For memory usage
memory_bytes = db.get_memory_usage()
print(f"Memory usage: {memory_bytes / 1024 / 1024:.2f} MB")
```

### Error Handling

```python
try:
    with Database.open("example.db", dimension=128) as db:
        db.add_vector(vector)
        hits = db.search(query, k=10)
except RuntimeError as e:
    print(f"Error: {e}")
```

### Working with Metadata

```python
# Add vector with multiple metadata fields
db.add_vector(
    vector,
    metadata={
        "category": "electronics",
        "brand": "Example",
        "price": "99.99",
        "rating": "4.5"
    }
)

# Search with metadata filter
hits = db.search(
    query, k=10,
    filter_metadata=("category", "electronics")
)

# Access metadata from results
for hit in hits:
    metadata = hit.vector.metadata
    print(f"Category: {metadata.get('category')}")
    print(f"Price: {metadata.get('price')}")
```

These examples demonstrate the core functionality of GigaVector. For advanced features and optimization, see the [Advanced Features Guide](advanced_features.md) and [Performance Tuning Guide](../performance.md).

