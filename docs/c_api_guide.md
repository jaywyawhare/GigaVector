# C API Guide

## Getting Started

### Including Headers

```c
#include "gigavector/gigavector.h"
```

This includes all necessary headers:
- `gv_types.h` - Type definitions
- `gv_database.h` - Database API
- `gv_vector.h` - Vector operations
- `gv_metadata.h` - Metadata management
- `gv_distance.h` - Distance metrics
- Index-specific headers (HNSW, IVFPQ, KD-Tree)

### Linking

**Using Make:**
```bash
gcc -o myapp myapp.c -Lbuild/lib -lGigaVector -lm -pthread
```

**Using CMake:**
```cmake
target_link_libraries(myapp GigaVector)
```

**Runtime:**
```bash
export LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH
./myapp
```

## Core Concepts

The database is represented by an opaque `GV_Database *` handle. Vectors are plain `float` arrays. Always close handles with `gv_db_close()` when done.

```c
typedef enum {
    GV_INDEX_TYPE_KDTREE = 0,
    GV_INDEX_TYPE_HNSW = 1,
    GV_INDEX_TYPE_IVFPQ = 2,
    GV_INDEX_TYPE_SPARSE = 3
} GV_IndexType;

typedef enum {
    GV_DISTANCE_EUCLIDEAN = 0,
    GV_DISTANCE_COSINE = 1,
    GV_DISTANCE_DOT_PRODUCT = 2,
    GV_DISTANCE_MANHATTAN = 3
} GV_DistanceType;
```

## Database Operations

### Creating a Database

```c
// Basic creation
GV_Database *db = gv_db_open("example.db", 128, GV_INDEX_TYPE_HNSW);
if (!db) {
    fprintf(stderr, "Failed to create database\n");
    return 1;
}

// With HNSW configuration
GV_HNSWConfig hnsw_config = {
    .M = 32,
    .efConstruction = 200,
    .efSearch = 50,
    .use_binary_quant = 1,
    .quant_rerank = 20
};
GV_Database *db = gv_db_open_with_hnsw_config(
    "hnsw.db", 128, GV_INDEX_TYPE_HNSW, &hnsw_config
);

// With IVFPQ configuration
GV_IVFPQConfig ivfpq_config = {
    .nlist = 256,
    .m = 16,
    .nbits = 8,
    .nprobe = 16,
    .default_rerank = 32
};
GV_Database *db = gv_db_open_with_ivfpq_config(
    "ivfpq.db", 128, GV_INDEX_TYPE_IVFPQ, &ivfpq_config
);

// Memory-mapped read-only
GV_Database *db = gv_db_open_mmap("readonly.db", 128, GV_INDEX_TYPE_KDTREE);
```

### Adding Vectors

```c
// Simple vector
float data[128];
// ... populate data ...
int rc = gv_db_add_vector(db, data, 128);

// With single metadata entry
rc = gv_db_add_vector_with_metadata(
    db, data, 128, "id", "12345"
);

// With multiple metadata entries
const char *keys[] = {"id", "category", "price"};
const char *values[] = {"12345", "electronics", "99.99"};
rc = gv_db_add_vector_with_rich_metadata(
    db, data, 128, keys, values, 3
);

// Batch insertion
float batch_data[1000 * 128];  // 1000 vectors of 128 dimensions
// ... populate batch_data ...
rc = gv_db_add_vectors(db, batch_data, 1000, 128);
```

### Searching

```c
// Basic search
float query[128];
GV_SearchResult results[10];
int found = gv_db_search(
    db, query, 10, results, GV_DISTANCE_EUCLIDEAN
);

for (int i = 0; i < found; i++) {
    printf("Distance: %f\n", results[i].distance);
}

// Filtered search
found = gv_db_search_filtered(
    db, query, 10, results, GV_DISTANCE_EUCLIDEAN,
    "category", "electronics"
);

// Advanced filter expression
found = gv_db_search_with_filter_expr(
    db, query, 10, results, GV_DISTANCE_EUCLIDEAN,
    "category == \"electronics\" AND price >= \"50\""
);

// Range search
found = gv_db_range_search(
    db, query, 0.5f, results, 100, GV_DISTANCE_EUCLIDEAN
);

// Batch search
float queries[10 * 128];  // 10 queries
GV_SearchResult batch_results[10 * 5];  // 10 queries, k=5
int total_found = gv_db_search_batch(
    db, queries, 10, 5, batch_results, GV_DISTANCE_EUCLIDEAN
);
```

### Updating and Deleting

```c
// Update vector data
float new_data[128];
int rc = gv_db_update_vector(db, 0, new_data, 128);

// Update metadata
const char *keys[] = {"price", "updated"};
const char *values[] = {"149.99", "true"};
rc = gv_db_update_vector_metadata(db, 0, keys, values, 2);

// Delete vector
rc = gv_db_delete_vector_by_index(db, 0);
```

### Saving and Loading

```c
int rc = gv_db_save(db, "backup.db");
gv_db_close(db);

// Reopen (WAL is automatically replayed)
db = gv_db_open("backup.db", 128, GV_INDEX_TYPE_HNSW);
```

### IVFPQ Training

```c
GV_Database *db = gv_db_open("ivfpq.db", 128, GV_INDEX_TYPE_IVFPQ);

float training_data[1000 * 128];
// ... populate training_data ...

int rc = gv_db_ivfpq_train(db, training_data, 1000, 128);
if (rc != 0) {
    fprintf(stderr, "Training failed\n");
}

// Now can add vectors
float vector[128];
gv_db_add_vector(db, vector, 128);
```

## Advanced Usage

### Configuration Structures

```c
// HNSW configuration
GV_HNSWConfig hnsw_config = {
    .M = 32,                    // Connections per node
    .efConstruction = 200,      // Construction quality
    .efSearch = 50,             // Search quality
    .maxLevel = 0,              // Auto-calculate (0)
    .use_binary_quant = 1,      // Enable quantization
    .quant_rerank = 20,         // Rerank candidates
    .use_acorn = 1,             // Enable ACORN
    .acorn_hops = 2             // ACORN depth
};

// IVFPQ configuration
GV_ScalarQuantConfig sq_config = {
    .bits = 8,
    .per_dimension = 0
};

GV_IVFPQConfig ivfpq_config = {
    .nlist = 256,
    .m = 16,
    .nbits = 8,
    .nprobe = 16,
    .train_iters = 20,
    .default_rerank = 32,
    .use_cosine = 0,
    .use_scalar_quant = 1,
    .scalar_quant_config = sq_config,
    .oversampling_factor = 2.0f
};
```

### Resource Limits

```c
GV_ResourceLimits limits = {
    .max_memory_bytes = 1024 * 1024 * 1024,  // 1GB
    .max_vectors = 1000000,
    .max_concurrent_operations = 100
};

int rc = gv_db_set_resource_limits(db, &limits);

// Get current limits
GV_ResourceLimits current;
gv_db_get_resource_limits(db, &current);
printf("Max memory: %zu bytes\n", current.max_memory_bytes);
```

### Statistics

```c
GV_DBStats stats;
gv_db_get_stats(db, &stats);
printf("Total inserts: %lu\n", stats.total_inserts);
printf("Total queries: %lu\n", stats.total_queries);

size_t memory = gv_db_get_memory_usage(db);
printf("Memory usage: %.2f MB\n", memory / 1024.0 / 1024.0);
```

### Compaction

```c
// Start background compaction
int rc = gv_db_start_background_compaction(db);

// Configure compaction
gv_db_set_compaction_interval(db, 300);  // 5 minutes
gv_db_set_wal_compaction_threshold(db, 10 * 1024 * 1024);  // 10MB
gv_db_set_deleted_ratio_threshold(db, 0.1);  // 10%

// Manual compaction
rc = gv_db_compact(db);

// Stop background compaction
gv_db_stop_background_compaction(db);
```

## Tips

- Most functions return `0` on success, `-1` on error, `NULL` for pointer returns on error
- Input vectors and metadata strings are copied on add; safe to free after the call
- Search result vectors are owned by the database; do not free them
- Use batch operations (`gv_db_add_vectors`, `gv_db_search_batch`) for throughput
- Pre-allocate `GV_SearchResult` buffers and reuse across searches
- Graph and knowledge graph APIs are documented in [api_reference.md](api_reference.md)
- Thread safety details are documented in [api_reference.md](api_reference.md)

For more information, see:
- [Usage Guide](usage.md) for general usage patterns
- [Python Bindings Guide](python_bindings.md) for Python integration
- [Performance Tuning Guide](performance.md) for optimization tips
