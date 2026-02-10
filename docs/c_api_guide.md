# C API Guide

This guide explains how to use GigaVector's C API effectively and correctly.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Database Operations](#database-operations)
4. [Memory Management](#memory-management)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)
7. [Advanced Usage](#advanced-usage)
8. [Thread Safety](#thread-safety)

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

### Database Handle

The database is represented by an opaque pointer:

```c
GV_Database *db = gv_db_open("example.db", 128, GV_INDEX_TYPE_HNSW);
// Use database...
gv_db_close(db);
```

**Important:** Always close the database when done to free resources.

### Vector Data

Vectors are passed as arrays of `float`:

```c
float vector[128];
for (int i = 0; i < 128; i++) {
    vector[i] = (float)rand() / RAND_MAX;
}

gv_db_add_vector(db, vector, 128);
```

### Index Types

```c
typedef enum {
    GV_INDEX_TYPE_KDTREE = 0,
    GV_INDEX_TYPE_HNSW = 1,
    GV_INDEX_TYPE_IVFPQ = 2,
    GV_INDEX_TYPE_SPARSE = 3
} GV_IndexType;
```

### Distance Types

```c
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
if (rc != 0) {
    fprintf(stderr, "Failed to add vector\n");
}

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
// ... populate query ...
GV_SearchResult results[10];
int found = gv_db_search(
    db, query, 10, results, GV_DISTANCE_EUCLIDEAN
);

if (found > 0) {
    for (int i = 0; i < found; i++) {
        printf("Distance: %f\n", results[i].distance);
        // Access vector via results[i].vector
    }
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
// ... populate new_data ...
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
// Save database
int rc = gv_db_save(db, "backup.db");
if (rc != 0) {
    fprintf(stderr, "Failed to save database\n");
}

// Close database (automatically saves WAL if enabled)
gv_db_close(db);

// Reopen (WAL is automatically replayed)
db = gv_db_open("backup.db", 128, GV_INDEX_TYPE_HNSW);
```

### IVFPQ Training

```c
// Create IVFPQ database
GV_Database *db = gv_db_open("ivfpq.db", 128, GV_INDEX_TYPE_IVFPQ);

// Prepare training data
float training_data[1000 * 128];  // 1000 training vectors
// ... populate training_data ...

// Train
int rc = gv_db_ivfpq_train(db, training_data, 1000, 128);
if (rc != 0) {
    fprintf(stderr, "Training failed\n");
}

// Now can add vectors
float vector[128];
gv_db_add_vector(db, vector, 128);
```

## Memory Management

### Ownership Rules

**Vectors:**
- Vectors passed to `gv_db_add_vector()` are copied internally
- You can free your vector data after calling add functions
- Vectors returned in search results are owned by the database
- Do not free vectors from search results

```c
// Safe: data is copied
float *data = malloc(128 * sizeof(float));
// ... populate data ...
gv_db_add_vector(db, data, 128);
free(data);  // Safe to free - database has a copy

// Search results - do not free
GV_SearchResult results[10];
int found = gv_db_search(db, query, 10, results, GV_DISTANCE_EUCLIDEAN);
// results[i].vector is owned by database - do not free
```

### Database Handle

- Database handle is owned by you
- Always call `gv_db_close()` when done
- Do not use database handle after closing

```c
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
// Use database...
gv_db_close(db);
// db is now invalid - do not use
```

### Metadata Strings

- Metadata strings are copied internally
- You can free your strings after calling add functions

```c
char *key = strdup("id");
char *value = strdup("12345");
gv_db_add_vector_with_metadata(db, data, 128, key, value);
free(key);    // Safe to free
free(value);  // Safe to free
```

### Pre-allocated Buffers

For performance, pre-allocate result buffers:

```c
// Pre-allocate once
GV_SearchResult *results = malloc(1000 * sizeof(GV_SearchResult));
if (!results) {
    // Handle error
}

// Reuse for multiple searches
for (int i = 0; i < 100; i++) {
    int found = gv_db_search(db, queries[i], 10, results, GV_DISTANCE_EUCLIDEAN);
    // Process results...
}

free(results);
```

## Error Handling

### Return Values

Most functions return:
- `0` on success
- `-1` on error
- `NULL` for pointer returns on error

```c
// Check return values
int rc = gv_db_add_vector(db, data, 128);
if (rc != 0) {
    fprintf(stderr, "Failed to add vector\n");
    // Handle error
}

// Check pointer returns
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
if (!db) {
    fprintf(stderr, "Failed to open database\n");
    // Handle error - check errno if needed
    return 1;
}
```

### Error Patterns

```c
// Pattern 1: Check and return
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
if (!db) {
    perror("gv_db_open");
    return 1;
}

// Pattern 2: Check and cleanup
int rc = gv_db_add_vector(db, data, 128);
if (rc != 0) {
    fprintf(stderr, "Failed to add vector\n");
    gv_db_close(db);
    return 1;
}

// Pattern 3: Check search results
int found = gv_db_search(db, query, 10, results, GV_DISTANCE_EUCLIDEAN);
if (found < 0) {
    fprintf(stderr, "Search failed\n");
    gv_db_close(db);
    return 1;
}
// found >= 0 means success (0 to k results found)
```

## Best Practices

### 1. Always Check Return Values

```c
// Good
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
if (!db) {
    // Handle error
    return 1;
}

// Bad - may crash if db is NULL
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
gv_db_add_vector(db, data, 128);  // Crash if db is NULL!
```

### 2. Always Close Databases

```c
// Good
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
if (!db) return 1;

// Use database...
gv_db_close(db);

// Also good - with cleanup
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
if (!db) return 1;

int rc = gv_db_add_vector(db, data, 128);
if (rc != 0) {
    gv_db_close(db);  // Cleanup on error
    return 1;
}

gv_db_close(db);
```

### 3. Use Consistent Dimensions

```c
// Good - consistent dimension
GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
float vector1[128];
float vector2[128];
gv_db_add_vector(db, vector1, 128);
gv_db_add_vector(db, vector2, 128);

// Bad - dimension mismatch
float vector3[64];  // Wrong dimension!
gv_db_add_vector(db, vector3, 64);  // Error!
```

### 4. Pre-allocate Result Buffers

```c
// Good - pre-allocated
GV_SearchResult *results = malloc(100 * sizeof(GV_SearchResult));
for (int i = 0; i < 100; i++) {
    int found = gv_db_search(db, queries[i], 10, results, GV_DISTANCE_EUCLIDEAN);
    // Process results...
}
free(results);

// Less efficient - stack allocation for many searches
for (int i = 0; i < 100; i++) {
    GV_SearchResult results[10];  // Allocated on stack each iteration
    int found = gv_db_search(db, queries[i], 10, results, GV_DISTANCE_EUCLIDEAN);
}
```

### 5. Use Batch Operations

```c
// Good - batch insertion
float batch[1000 * 128];
// ... populate batch ...
gv_db_add_vectors(db, batch, 1000, 128);

// Less efficient - individual insertions
for (int i = 0; i < 1000; i++) {
    gv_db_add_vector(db, vectors[i], 128);
}
```

### 6. Handle Search Results Correctly

```c
// Good - check found count
GV_SearchResult results[10];
int found = gv_db_search(db, query, 10, results, GV_DISTANCE_EUCLIDEAN);
if (found < 0) {
    // Error
} else if (found == 0) {
    // No results
} else {
    // Process found results
    for (int i = 0; i < found; i++) {
        printf("Distance: %f\n", results[i].distance);
    }
}

// Bad - assuming k results
for (int i = 0; i < 10; i++) {  // May access invalid results!
    printf("Distance: %f\n", results[i].distance);
}
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
if (rc != 0) {
    fprintf(stderr, "Failed to set resource limits\n");
}

// Get current limits
GV_ResourceLimits current;
gv_db_get_resource_limits(db, &current);
printf("Max memory: %zu bytes\n", current.max_memory_bytes);
```

### Statistics

```c
// Basic statistics
GV_DBStats stats;
gv_db_get_stats(db, &stats);
printf("Total inserts: %lu\n", stats.total_inserts);
printf("Total queries: %lu\n", stats.total_queries);

// Memory usage
size_t memory = gv_db_get_memory_usage(db);
printf("Memory usage: %.2f MB\n", memory / 1024.0 / 1024.0);
```

### Compaction

```c
// Start background compaction
int rc = gv_db_start_background_compaction(db);
if (rc != 0) {
    fprintf(stderr, "Failed to start compaction\n");
}

// Configure compaction
gv_db_set_compaction_interval(db, 300);  // 5 minutes
gv_db_set_wal_compaction_threshold(db, 10 * 1024 * 1024);  // 10MB
gv_db_set_deleted_ratio_threshold(db, 0.1);  // 10%

// Manual compaction
rc = gv_db_compact(db);
if (rc != 0) {
    fprintf(stderr, "Compaction failed\n");
}

// Stop background compaction
gv_db_stop_background_compaction(db);
```

## Graph Database

GigaVector provides a property graph database layer for storing and querying graph-structured data.

### Creating a Graph

```c
#include "gigavector/gv_graph_db.h"

// Create with default config
GV_GraphDB *g = gv_graph_create(NULL);

// Or with custom config
GV_GraphDBConfig cfg;
gv_graph_config_init(&cfg);
cfg.node_bucket_count = 8192;
cfg.enforce_referential_integrity = 1;
GV_GraphDB *g = gv_graph_create(&cfg);
```

### Nodes and Edges

```c
// Add nodes
uint64_t alice = gv_graph_add_node(g, "Person");
uint64_t bob = gv_graph_add_node(g, "Person");
gv_graph_set_node_prop(g, alice, "name", "Alice");

// Add edges
uint64_t edge = gv_graph_add_edge(g, alice, bob, "KNOWS", 1.0f);
gv_graph_set_edge_prop(g, edge, "since", "2024");

// Query neighbors
uint64_t neighbors[64];
int count = gv_graph_get_neighbors(g, alice, neighbors, 64);
```

### Traversal and Analytics

```c
// BFS / DFS
uint64_t visited[1024];
int n = gv_graph_bfs(g, alice, 3, visited, 1024);

// Shortest path (Dijkstra)
GV_GraphPath path;
if (gv_graph_shortest_path(g, alice, bob, &path) == 0) {
    printf("Path weight: %f, length: %zu\n", path.total_weight, path.length);
    gv_graph_free_path(&path);
}

// PageRank
float pr = gv_graph_pagerank(g, alice, 20, 0.85f);

// Clustering coefficient
float cc = gv_graph_clustering_coefficient(g, alice);

// Save / Load
gv_graph_save(g, "social.gvgr");
GV_GraphDB *g2 = gv_graph_load("social.gvgr");

gv_graph_destroy(g);
```

## Knowledge Graph

The knowledge graph layer combines graph structure with vector embeddings.

### Creating a Knowledge Graph

```c
#include "gigavector/gv_knowledge_graph.h"

GV_KGConfig cfg;
gv_kg_config_init(&cfg);
cfg.embedding_dimension = 128;
GV_KnowledgeGraph *kg = gv_kg_create(&cfg);
```

### Entities and Relations

```c
// Add entities with embeddings
float emb_alice[128] = { /* ... */ };
uint64_t alice = gv_kg_add_entity(kg, "Alice", "Person", emb_alice, 128);
uint64_t company = gv_kg_add_entity(kg, "Anthropic", "Company", NULL, 0);

// Add relations (SPO triples)
uint64_t rel = gv_kg_add_relation(kg, alice, "works_at", company, 1.0f);

// Query triples (NULL = wildcard)
GV_KGTriple triples[64];
int n = gv_kg_query_triples(kg, NULL, "works_at", NULL, triples, 64);
gv_kg_free_triples(triples, n);
```

### Semantic Search and Link Prediction

```c
// Search by embedding similarity
float query[128] = { /* ... */ };
GV_KGSearchResult results[10];
int n = gv_kg_search_similar(kg, query, 128, 10, results);
gv_kg_free_search_results(results, n);

// Predict missing links
GV_KGLinkPrediction predictions[5];
int np = gv_kg_predict_links(kg, alice, 5, predictions);

// Entity resolution
int resolved_id = gv_kg_resolve_entity(kg, "Alice Smith", "Person", emb_alice, 128);

// Save / Load
gv_kg_save(kg, "knowledge.gvkg");
GV_KnowledgeGraph *kg2 = gv_kg_load("knowledge.gvkg");

gv_kg_destroy(kg);
```

---

## Thread Safety

### Database Handle

- Each thread should use its own database handle, OR
- Synchronize access to a shared database handle

```c
// Option 1: Separate handles (recommended)
void *thread_func(void *arg) {
    GV_Database *db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);
    // Use database in this thread...
    gv_db_close(db);
    return NULL;
}

// Option 2: Shared handle with synchronization
pthread_mutex_t db_mutex = PTHREAD_MUTEX_INITIALIZER;
GV_Database *shared_db = gv_db_open("db.db", 128, GV_INDEX_TYPE_HNSW);

void *thread_func(void *arg) {
    pthread_mutex_lock(&db_mutex);
    // Use shared_db...
    pthread_mutex_unlock(&db_mutex);
    return NULL;
}
```

### Search Operations

Search operations are generally thread-safe when using separate database handles or proper synchronization.

## Summary

- Always check return values and handle errors
- Always close database handles when done
- Use consistent vector dimensions
- Pre-allocate result buffers for performance
- Use batch operations when possible
- Understand memory ownership rules
- Use appropriate synchronization for multi-threaded access

For more information, see:
- [Usage Guide](usage.md) for general usage patterns
- [Python Bindings Guide](python_bindings.md) for Python integration
- [Performance Tuning Guide](performance.md) for optimization tips

