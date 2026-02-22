# GigaVector API Reference

Complete reference documentation for the GigaVector C API.

## Table of Contents

1. [Database Operations](#database-operations)
2. [Vector Operations](#vector-operations)
3. [Search Operations](#search-operations)
4. [Metadata Management](#metadata-management)
5. [Memory Layer](#memory-layer)
6. [LLM Integration](#llm-integration)
7. [Embedding Services](#embedding-services)
8. [Context Graph](#context-graph)
9. [Index Configuration](#index-configuration)
10. [Statistics and Monitoring](#statistics-and-monitoring)
11. [Error Handling](#error-handling)
12. [Resource Management](#resource-management)
13. [Authentication](#authentication)

---

## Database Operations

### Opening and Closing Databases

#### `gv_db_open`

```c
GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type);
```

Creates or opens a vector database.

**Parameters:**
- `filepath`: Path to database file (NULL for in-memory only)
- `dimension`: Vector dimensionality (must be consistent)
- `index_type`: Index algorithm (`GV_INDEX_TYPE_KDTREE`, `GV_INDEX_TYPE_HNSW`, `GV_INDEX_TYPE_IVFPQ`, `GV_INDEX_TYPE_SPARSE`, `GV_INDEX_TYPE_FLAT`, `GV_INDEX_TYPE_IVFFLAT`, `GV_INDEX_TYPE_PQ`, `GV_INDEX_TYPE_LSH`)

**Returns:** Database handle or NULL on error

**Example:**
```c
GV_Database *db = gv_db_open("vectors.db", 128, GV_INDEX_TYPE_HNSW);
if (db == NULL) {
    fprintf(stderr, "Failed to open database\n");
    return 1;
}
```

#### `gv_db_open_with_hnsw_config`

```c
GV_Database *gv_db_open_with_hnsw_config(const char *filepath, size_t dimension,
                                         GV_IndexType index_type, const GV_HNSWConfig *hnsw_config);
```

Opens database with custom HNSW parameters.

**Parameters:**
- `hnsw_config`: HNSW configuration structure (NULL for defaults)

**HNSW Configuration:**
```c
typedef struct {
    int M;              // Number of bi-directional links (default: 16)
    int ef_construction; // Size of candidate list during construction (default: 200)
    int ef_search;      // Size of candidate list during search (default: 50)
    float level_mult;   // Level multiplier (default: 1.0)
} GV_HNSWConfig;
```

#### `gv_db_open_with_ivfpq_config`

```c
GV_Database *gv_db_open_with_ivfpq_config(const char *filepath, size_t dimension,
                                          GV_IndexType index_type, const GV_IVFPQConfig *ivfpq_config);
```

Opens database with custom IVFPQ parameters.

**IVFPQ Configuration:**
```c
typedef struct {
    size_t n_clusters;      // Number of clusters (default: 256)
    size_t n_subvectors;   // Number of subvectors (default: 8)
    size_t n_bits;          // Bits per subvector (default: 8)
} GV_IVFPQConfig;
```

#### `gv_db_close`

```c
void gv_db_close(GV_Database *db);
```

Closes database and releases all resources. Safe to call with NULL.

#### `gv_free`

```c
void gv_free(void *ptr);
```

Frees memory allocated by GigaVector (e.g. strings returned by `gv_schema_to_json`, `gv_json_export`, etc.). Safe to call with NULL.

**Example:**
```c
char *json = gv_schema_to_json(schema);
// ... use json ...
gv_free(json);
```

---

## Vector Operations

### Adding Vectors

#### `gv_db_add_vector`

```c
int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);
```

Adds a dense vector to the database.

**Returns:** 0 on success, -1 on error

#### `gv_db_add_vector_with_metadata`

```c
int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                   const char *metadata_key, const char *metadata_value);
```

Adds vector with single metadata key-value pair.

#### `gv_db_add_vector_with_rich_metadata`

```c
int gv_db_add_vector_with_rich_metadata(GV_Database *db, const float *data, size_t dimension,
                                       const char *const *metadata_keys,
                                       const char *const *metadata_values,
                                       size_t metadata_count);
```

Adds vector with multiple metadata entries.

**Example:**
```c
const char *keys[] = {"category", "author", "date"};
const char *values[] = {"article", "John Doe", "2024-01-15"};
int result = gv_db_add_vector_with_rich_metadata(db, vector, 128, keys, values, 3);
```

#### `gv_db_add_sparse_vector`

```c
int gv_db_add_sparse_vector(GV_Database *db, const uint32_t *indices, const float *values,
                           size_t nnz, size_t dimension,
                           const char *metadata_key, const char *metadata_value);
```

Adds a sparse vector (only for `GV_INDEX_TYPE_SPARSE`).

**Parameters:**
- `indices`: Array of dimension indices
- `values`: Array of non-zero values
- `nnz`: Number of non-zero elements

---

## Search Operations

### Nearest Neighbor Search

#### `gv_db_search`

```c
int gv_db_search(const GV_Database *db, const float *query, size_t k,
                GV_DistanceType distance_type, GV_SearchResult *results);
```

Performs k-nearest neighbor search.

**Parameters:**
- `query`: Query vector
- `k`: Number of results to return
- `distance_type`: `GV_DISTANCE_EUCLIDEAN` or `GV_DISTANCE_COSINE`
- `results`: Output array (must be pre-allocated for k elements)

**Returns:** Number of results found (0 to k)

**Example:**
```c
GV_SearchResult results[10];
int count = gv_db_search(db, query_vector, 10, GV_DISTANCE_COSINE, results);
for (int i = 0; i < count; i++) {
    printf("Distance: %f, ID: %zu\n", results[i].distance, results[i].id);
}
```

#### `gv_db_search_with_filter`

```c
int gv_db_search_with_filter(const GV_Database *db, const float *query, size_t k,
                             GV_DistanceType distance_type, const GV_Filter *filter,
                             GV_SearchResult *results);
```

Search with metadata filtering.

**Filter Example:**
```c
GV_Filter filter;
filter.operation = GV_FILTER_EQUALS;
filter.key = "category";
filter.value = "article";
int count = gv_db_search_with_filter(db, query, 10, GV_DISTANCE_COSINE, &filter, results);
```

### Range Search

#### `gv_db_range_search`

```c
int gv_db_range_search(const GV_Database *db, const float *query, float radius,
                      GV_DistanceType distance_type, GV_SearchResult *results, size_t max_results);
```

Finds all vectors within specified radius.

**Parameters:**
- `radius`: Maximum distance threshold
- `max_results`: Maximum number of results to return

---

## Metadata Management

### Filtering

GigaVector supports complex metadata filtering:

```c
typedef enum {
    GV_FILTER_EQUALS,
    GV_FILTER_NOT_EQUALS,
    GV_FILTER_GREATER_THAN,
    GV_FILTER_LESS_THAN,
    GV_FILTER_IN,
    GV_FILTER_AND,
    GV_FILTER_OR
} GV_FilterOperation;

typedef struct GV_Filter {
    GV_FilterOperation operation;
    const char *key;
    const char *value;
    struct GV_Filter *left;
    struct GV_Filter *right;
} GV_Filter;
```

**Complex Filter Example:**
```c
GV_Filter filter1 = {GV_FILTER_EQUALS, "category", "article", NULL, NULL};
GV_Filter filter2 = {GV_FILTER_GREATER_THAN, "date", "2024-01-01", NULL, NULL};
GV_Filter and_filter = {GV_FILTER_AND, NULL, NULL, &filter1, &filter2};
```

---

## Memory Layer

### Memory Extraction

#### `gv_memory_extract_candidates_from_conversation`

```c
int gv_memory_extract_candidates_from_conversation(const char *conversation,
                                                   const char *conversation_id,
                                                   int is_agent_memory,
                                                   GV_MemoryCandidate *candidates,
                                                   size_t max_candidates,
                                                   size_t *actual_count);
```

Extracts memory candidates from conversation text using heuristic patterns.

#### `gv_memory_extract_candidates_from_conversation_llm`

```c
int gv_memory_extract_candidates_from_conversation_llm(GV_LLM *llm,
                                                      const char *conversation,
                                                      const char *conversation_id,
                                                      int is_agent_memory,
                                                      const char *custom_prompt,
                                                      GV_MemoryCandidate *candidates,
                                                      size_t max_candidates,
                                                      size_t *actual_count);
```

Extracts memories using LLM (more accurate, requires API key).

### Memory Storage

#### `gv_memory_store`

```c
int gv_memory_store(GV_Database *db, const GV_MemoryCandidate *candidate,
                   const char *memory_id, double importance_score);
```

Stores a memory in the database.

### Memory Retrieval

#### `gv_memory_search`

```c
int gv_memory_search(GV_Database *db, const char *query, size_t k,
                    const GV_MemorySearchOptions *options,
                    GV_MemoryResult *results);
```

Searches memories with semantic similarity and temporal weighting.

**Search Options:**
```c
typedef struct {
    float temporal_weight;      // 0.0=semantic only, 1.0=recency only
    float importance_weight;     // Weight for importance (default: 0.4)
    int include_linked;          // Include linked memories
    float link_boost;            // Score boost for linked memories
    time_t min_timestamp;        // Minimum creation time
    time_t max_timestamp;        // Maximum creation time
} GV_MemorySearchOptions;
```

### Memory Consolidation

#### `gv_memory_consolidate`

```c
int gv_memory_consolidate(GV_Database *db, double similarity_threshold,
                         GV_ConsolidationStrategy strategy);
```

Consolidates similar or redundant memories.

**Strategies:**
- `GV_CONSOLIDATION_MERGE`: Merge similar memories
- `GV_CONSOLIDATION_UPDATE`: Update existing with new info
- `GV_CONSOLIDATION_LINK`: Create relationship link
- `GV_CONSOLIDATION_ARCHIVE`: Archive redundant memory

---

## LLM Integration

### LLM Configuration

```c
typedef struct {
    GV_LLMProvider provider;     // OPENAI, ANTHROPIC, GOOGLE, CUSTOM
    char *api_key;               // API key for authentication
    char *model;                 // Model name (e.g., "gpt-4o-mini")
    char *base_url;              // Base URL (NULL for default)
    double temperature;          // 0.0-2.0
    int max_tokens;              // Maximum tokens in response
    int timeout_seconds;         // Request timeout
    char *custom_prompt;         // Custom extraction prompt
} GV_LLMConfig;
```

### Creating LLM Instance

#### `gv_llm_create`

```c
GV_LLM *gv_llm_create(const GV_LLMConfig *config);
```

**Example:**
```c
GV_LLMConfig config = {
    .provider = GV_LLM_PROVIDER_OPENAI,
    .api_key = getenv("OPENAI_API_KEY"),
    .model = "gpt-4o-mini",
    .temperature = 0.7,
    .max_tokens = 1000,
    .timeout_seconds = 30
};
GV_LLM *llm = gv_llm_create(&config);
```

### Generating Responses

#### `gv_llm_generate_response`

```c
int gv_llm_generate_response(GV_LLM *llm, const GV_LLMMessage *messages,
                             size_t message_count, const char *response_format,
                             GV_LLMResponse *response);
```

**Example:**
```c
GV_LLMMessage messages[] = {
    {.role = "user", .content = "Extract key facts from: User loves Python"}
};
GV_LLMResponse response;
int result = gv_llm_generate_response(llm, messages, 1, "json_object", &response);
if (result == GV_LLM_SUCCESS) {
    printf("Response: %s\n", response.content);
    gv_llm_response_free(&response);
}
```

### Error Handling

```c
const char *error_msg = gv_llm_get_last_error(llm);
const char *error_desc = gv_llm_error_string(error_code);
```

---

## Embedding Services

### Embedding Configuration

```c
typedef struct {
    GV_EmbeddingProvider provider;  // OPENAI, GOOGLE, HUGGINGFACE, CUSTOM
    char *api_key;                  // API key
    char *model;                    // Model name
    char *base_url;                 // Base URL (NULL for default)
    size_t dimension;               // Output dimension
} GV_EmbeddingConfig;
```

### Creating Embedding Service

#### `gv_embedding_create`

```c
GV_EmbeddingService *gv_embedding_create(const GV_EmbeddingConfig *config);
```

### Generating Embeddings

#### `gv_embedding_embed`

```c
int gv_embedding_embed(GV_EmbeddingService *service, const char *text, float *output);
```

#### `gv_embedding_embed_batch`

```c
int gv_embedding_embed_batch(GV_EmbeddingService *service, const char **texts,
                             size_t text_count, float **outputs);
```

---

## Context Graph

### Building Context Graph

#### `gv_context_graph_build`

```c
GV_ContextGraph *gv_context_graph_build(GV_Database *db, const char **entity_names,
                                       size_t entity_count);
```

Builds entity-relationship graph from database metadata.

### Querying Context

#### `gv_context_graph_get_related`

```c
int gv_context_graph_get_related(const GV_ContextGraph *graph, const char *entity_name,
                                char **related_entities, size_t max_related);
```

---

## Graph Database

### Configuration

#### `gv_graph_config_init`

```c
void gv_graph_config_init(GV_GraphDBConfig *config);
```

Initialize a graph database configuration with default values (node_bucket_count=4096, edge_bucket_count=8192, enforce_referential_integrity=1).

### Creating and Destroying

#### `gv_graph_create`

```c
GV_GraphDB *gv_graph_create(const GV_GraphDBConfig *config);
```

Create a new graph database. Pass NULL for default configuration. Returns NULL on error.

#### `gv_graph_destroy`

```c
void gv_graph_destroy(GV_GraphDB *g);
```

### Node Operations

#### `gv_graph_add_node`

```c
uint64_t gv_graph_add_node(GV_GraphDB *g, const char *label);
```

Add a node with the given label. Returns node_id (>0) on success, 0 on error.

#### `gv_graph_remove_node`

```c
int gv_graph_remove_node(GV_GraphDB *g, uint64_t node_id);
```

Remove a node and cascade-delete all incident edges. Returns 0 on success.

#### `gv_graph_get_node`

```c
const GV_GraphNode *gv_graph_get_node(const GV_GraphDB *g, uint64_t node_id);
```

Look up a node by ID. Returns pointer valid until next mutation, or NULL.

#### `gv_graph_set_node_prop` / `gv_graph_get_node_prop`

```c
int gv_graph_set_node_prop(GV_GraphDB *g, uint64_t node_id, const char *key, const char *value);
const char *gv_graph_get_node_prop(const GV_GraphDB *g, uint64_t node_id, const char *key);
```

#### `gv_graph_find_nodes_by_label`

```c
int gv_graph_find_nodes_by_label(const GV_GraphDB *g, const char *label,
                                 uint64_t *out_ids, size_t max_count);
```

### Edge Operations

#### `gv_graph_add_edge`

```c
uint64_t gv_graph_add_edge(GV_GraphDB *g, uint64_t source, uint64_t target,
                           const char *label, float weight);
```

Add a directed weighted edge. Returns edge_id (>0) on success, 0 on error.

#### `gv_graph_remove_edge`

```c
int gv_graph_remove_edge(GV_GraphDB *g, uint64_t edge_id);
```

#### `gv_graph_get_edges_out` / `gv_graph_get_edges_in` / `gv_graph_get_neighbors`

```c
int gv_graph_get_edges_out(const GV_GraphDB *g, uint64_t node_id, uint64_t *out_ids, size_t max_count);
int gv_graph_get_edges_in(const GV_GraphDB *g, uint64_t node_id, uint64_t *out_ids, size_t max_count);
int gv_graph_get_neighbors(const GV_GraphDB *g, uint64_t node_id, uint64_t *out_ids, size_t max_count);
```

### Traversal

#### `gv_graph_bfs` / `gv_graph_dfs`

```c
int gv_graph_bfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count);
int gv_graph_dfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count);
```

#### `gv_graph_shortest_path`

```c
int gv_graph_shortest_path(const GV_GraphDB *g, uint64_t from, uint64_t to, GV_GraphPath *path);
```

Dijkstra's algorithm. Caller must free result with `gv_graph_free_path()`.

#### `gv_graph_all_paths`

```c
int gv_graph_all_paths(const GV_GraphDB *g, uint64_t from, uint64_t to,
                       size_t max_depth, GV_GraphPath *paths, size_t max_paths);
```

### Analytics

#### `gv_graph_pagerank`

```c
float gv_graph_pagerank(const GV_GraphDB *g, uint64_t node_id, size_t iterations, float damping);
```

Iterative power method PageRank. Typical values: iterations=20, damping=0.85.

#### `gv_graph_connected_components`

```c
int gv_graph_connected_components(const GV_GraphDB *g, uint64_t *component_ids, size_t max_count);
```

Returns number of distinct components. Array indexed by enumeration order.

#### `gv_graph_clustering_coefficient`

```c
float gv_graph_clustering_coefficient(const GV_GraphDB *g, uint64_t node_id);
```

Local clustering coefficient in [0.0, 1.0].

### Persistence

#### `gv_graph_save` / `gv_graph_load`

```c
int gv_graph_save(const GV_GraphDB *g, const char *path);
GV_GraphDB *gv_graph_load(const char *path);
```

Binary format with magic "GVGR".

---

## Knowledge Graph

### Configuration

#### `gv_kg_config_init`

```c
void gv_kg_config_init(GV_KGConfig *config);
```

Initialize with defaults: entity_bucket_count=4096, relation_bucket_count=8192, embedding_dimension=128, similarity_threshold=0.7, link_prediction_threshold=0.8, max_entities=1000000.

### Creating and Destroying

```c
GV_KnowledgeGraph *gv_kg_create(const GV_KGConfig *config);
void gv_kg_destroy(GV_KnowledgeGraph *kg);
```

### Entity Operations

#### `gv_kg_add_entity`

```c
uint64_t gv_kg_add_entity(GV_KnowledgeGraph *kg, const char *name, const char *type,
                           const float *embedding, size_t dimension);
```

Add an entity with optional embedding. Returns entity_id (>0) on success.

#### `gv_kg_remove_entity`

```c
int gv_kg_remove_entity(GV_KnowledgeGraph *kg, uint64_t entity_id);
```

Cascade-deletes all relations involving this entity.

#### `gv_kg_find_entities_by_type` / `gv_kg_find_entities_by_name`

```c
int gv_kg_find_entities_by_type(const GV_KnowledgeGraph *kg, const char *type,
                                 uint64_t *out_ids, size_t max_count);
int gv_kg_find_entities_by_name(const GV_KnowledgeGraph *kg, const char *name,
                                 uint64_t *out_ids, size_t max_count);
```

### Relation (Triple) Operations

#### `gv_kg_add_relation`

```c
uint64_t gv_kg_add_relation(GV_KnowledgeGraph *kg, uint64_t subject,
                             const char *predicate, uint64_t object, float weight);
```

Creates a (subject, predicate, object) triple. Both entities must exist.

### SPO Triple Queries

#### `gv_kg_query_triples`

```c
int gv_kg_query_triples(const GV_KnowledgeGraph *kg, const uint64_t *subject,
                         const char *predicate, const uint64_t *object,
                         GV_KGTriple *out, size_t max_count);
```

Pass NULL for any parameter to treat it as a wildcard. Free results with `gv_kg_free_triples()`.

### Semantic Search

#### `gv_kg_search_similar`

```c
int gv_kg_search_similar(const GV_KnowledgeGraph *kg, const float *query_embedding,
                          size_t dimension, size_t k, GV_KGSearchResult *results);
```

Cosine similarity search over entity embeddings. Free results with `gv_kg_free_search_results()`.

#### `gv_kg_hybrid_search`

```c
int gv_kg_hybrid_search(const GV_KnowledgeGraph *kg, const float *query_embedding,
                         size_t dimension, const char *entity_type,
                         const char *predicate_filter, size_t k, GV_KGSearchResult *results);
```

Embedding similarity filtered by entity type and/or predicate.

### Entity Resolution

#### `gv_kg_resolve_entity`

```c
int gv_kg_resolve_entity(GV_KnowledgeGraph *kg, const char *name, const char *type,
                          const float *embedding, size_t dimension);
```

Find existing match or create new entity. Returns entity_id.

#### `gv_kg_merge_entities`

```c
int gv_kg_merge_entities(GV_KnowledgeGraph *kg, uint64_t keep_id, uint64_t merge_id);
```

### Link Prediction

#### `gv_kg_predict_links`

```c
int gv_kg_predict_links(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                         size_t k, GV_KGLinkPrediction *results);
```

Predict missing links using embedding similarity and structural patterns.

### Traversal and Subgraph

```c
int gv_kg_get_neighbors(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                         uint64_t *out_ids, size_t max_count);
int gv_kg_traverse(const GV_KnowledgeGraph *kg, uint64_t start,
                    size_t max_depth, uint64_t *out_ids, size_t max_count);
int gv_kg_shortest_path(const GV_KnowledgeGraph *kg, uint64_t from,
                         uint64_t to, uint64_t *path_ids, size_t max_len);
int gv_kg_extract_subgraph(const GV_KnowledgeGraph *kg, uint64_t center,
                            size_t radius, GV_KGSubgraph *subgraph);
```

### Analytics

```c
int gv_kg_get_stats(const GV_KnowledgeGraph *kg, GV_KGStats *stats);
float gv_kg_entity_centrality(const GV_KnowledgeGraph *kg, uint64_t entity_id);
int gv_kg_get_entity_types(const GV_KnowledgeGraph *kg, char **out_types, size_t max_count);
int gv_kg_get_predicates(const GV_KnowledgeGraph *kg, char **out_predicates, size_t max_count);
```

### Persistence

```c
int gv_kg_save(const GV_KnowledgeGraph *kg, const char *path);
GV_KnowledgeGraph *gv_kg_load(const char *path);
```

Binary format with magic "GVKG".

---

## Index Configuration

### Index Selection

#### `gv_index_suggest`

```c
GV_IndexType gv_index_suggest(size_t dimension, size_t expected_count);
```

Suggests optimal index type based on dimension and expected size.

**Heuristics:**
- Small datasets (≤20k) and low dimensions (≤64): KDTREE
- Very large datasets (≥500k) and high dimensions (≥128): IVFPQ
- Otherwise: HNSW

### Cosine Normalization

#### `gv_db_set_cosine_normalized`

```c
void gv_db_set_cosine_normalized(GV_Database *db, int enabled);
```

Enables L2 normalization for cosine distance optimization.

---

## Statistics and Monitoring

### Basic Statistics

#### `gv_db_get_stats`

```c
void gv_db_get_stats(const GV_Database *db, GV_DBStats *out);
```

**Statistics Structure:**
```c
typedef struct {
    size_t vector_count;
    size_t dimension;
    size_t memory_bytes;
    uint64_t total_inserts;
    uint64_t total_queries;
    double avg_query_latency_ms;
} GV_DBStats;
```

### Detailed Statistics

#### `gv_db_get_detailed_stats`

```c
int gv_db_get_detailed_stats(const GV_Database *db, GV_DetailedStats *out);
void gv_db_free_detailed_stats(GV_DetailedStats *stats);
```

Returns comprehensive database metrics including latency histograms, throughput, memory breakdown, recall metrics, and health indicators.

**Detailed Statistics Structure:**
```c
typedef struct {
    GV_DBStats basic_stats;                // Basic aggregated statistics
    GV_LatencyHistogram insert_latency;    // Insert operation latency histogram
    GV_LatencyHistogram search_latency;    // Search operation latency histogram
    double queries_per_second;             // Current QPS
    double inserts_per_second;             // Current IPS
    GV_MemoryBreakdown memory;             // Memory breakdown (SoA, index, metadata, WAL)
    GV_RecallMetrics recall;               // Recall metrics for approximate search
    int health_status;                     // 0=healthy, -1=degraded, -2=unhealthy
    size_t deleted_vector_count;           // Number of deleted vectors
    double deleted_ratio;                  // Ratio of deleted vectors (0.0--1.0)
} GV_DetailedStats;
```

**Example:**
```c
GV_DetailedStats stats;
if (gv_db_get_detailed_stats(db, &stats) == 0) {
    printf("QPS: %.1f, Memory: %zu bytes\n",
           stats.queries_per_second, stats.memory.total_bytes);
    printf("Health: %d, Recall: %.2f\n",
           stats.health_status, stats.recall.avg_recall);
    gv_db_free_detailed_stats(&stats);
}
```

---

## Error Handling

### Error Codes

All functions return:
- `0` or positive value: Success
- `-1`: Generic error
- Specific error codes: See function documentation

### Error Messages

Many functions set error messages accessible via:
```c
const char *error = gv_llm_get_last_error(llm);
```

### Best Practices

1. **Always check return values**
2. **Free resources in reverse order of allocation**
3. **Use NULL checks before dereferencing**
4. **Handle errors gracefully**

**Example:**
```c
GV_Database *db = gv_db_open("db.gvdb", 128, GV_INDEX_TYPE_HNSW);
if (db == NULL) {
    fprintf(stderr, "Failed to open database\n");
    return 1;
}

int result = gv_db_add_vector(db, vector, 128);
if (result != 0) {
    fprintf(stderr, "Failed to add vector\n");
    gv_db_close(db);
    return 1;
}

// ... use database ...

gv_db_close(db);
```

---

## HTTP REST Server

### Server Configuration

```c
typedef struct {
    int port;                    // Default: 8080
    int thread_pool_size;        // Default: 4
    int max_connections;         // Default: 100
    int request_timeout_ms;      // Default: 30000
    size_t max_request_body_bytes; // Default: 10MB
    int enable_cors;             // Default: 0
    int enable_logging;          // Default: 1
    const char *api_key;         // Optional API key auth
} GV_ServerConfig;
```

### Starting the Server

```c
GV_ServerConfig config;
gv_server_config_init(&config);
config.port = 9090;
config.enable_cors = 1;

GV_Server *server = gv_server_create(db, &config);
gv_server_start(server);

// ... server runs ...

gv_server_stop(server);
gv_server_destroy(server);
```

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Database statistics |
| `/vectors` | POST | Add vector |
| `/vectors/:id` | GET | Get vector by ID |
| `/vectors/:id` | DELETE | Delete vector |
| `/search` | POST | Vector search |
| `/namespaces` | GET/POST | Manage namespaces |

---

## Hybrid Search (BM25 + Vector)

Combines text-based BM25 ranking with vector similarity.

### Tokenizer

```c
GV_Tokenizer *tokenizer = gv_tokenizer_create(GV_TOKENIZER_WHITESPACE);
GV_TokenList *tokens = gv_tokenizer_tokenize(tokenizer, "hello world");
gv_token_list_free(tokens);
gv_tokenizer_destroy(tokenizer);
```

### BM25 Search

```c
GV_BM25Config config = {
    .k1 = 1.2,
    .b = 0.75
};
GV_BM25Index *index = gv_bm25_create(&config);
gv_bm25_add_document(index, doc_id, "document text content");
gv_bm25_search(index, "query terms", results, max_results);
```

### Hybrid Search

```c
GV_HybridSearchConfig config = {
    .vector_weight = 0.7,
    .bm25_weight = 0.3,
    .fusion_method = GV_FUSION_RRF  // Reciprocal Rank Fusion
};
int count = gv_hybrid_search(db, bm25_index, query_vector, "query text",
                             k, &config, results);
```

---

## Namespaces

Isolate data into separate logical collections.

```c
// Create namespace
gv_namespace_create(db, "project_a");

// Add vector to namespace
gv_namespace_add_vector(db, "project_a", vector, dim, metadata);

// Search within namespace
gv_namespace_search(db, "project_a", query, k, distance_type, results);

// List namespaces
char **names;
size_t count;
gv_namespace_list(db, &names, &count);

// Delete namespace
gv_namespace_delete(db, "project_a");
```

---

## TTL (Time-To-Live)

Automatic expiration of vectors.

```c
// Set TTL on vector (seconds)
gv_ttl_set(db, vector_id, 3600);  // Expires in 1 hour

// Get remaining TTL
int64_t remaining = gv_ttl_get(db, vector_id);

// Remove TTL
gv_ttl_remove(db, vector_id);

// Run expiration (call periodically or use background thread)
size_t expired = gv_ttl_expire(db);
```

---

## Backup and Restore

### CLI Tools

```bash
# Backup database
gvbackup mydb.db backup.gvb

# Restore database
gvrestore backup.gvb restored.db

# Inspect database
gvinspect mydb.db
```

### C API

```c
// Create backup
GV_BackupConfig config = {
    .compress = 1,
    .include_wal = 1
};
gv_backup_create(db, "backup.gvb", &config);

// Restore backup
GV_Database *restored = gv_backup_restore("backup.gvb", "restored.db");
```

---

## Authentication

### API Key Auth

```c
GV_AuthConfig auth_config = {
    .type = GV_AUTH_API_KEY,
    .api_key = "your-secret-key"
};
gv_server_set_auth(server, &auth_config);
```

### Authorization (Fine-Grained)

GigaVector provides a fine-grained authorization system with permission flags and resource-level access control.

#### Permission Flags

```c
typedef enum {
    GV_PERM_NONE   = 0,   // No permissions
    GV_PERM_READ   = 1,   // Read vectors / search
    GV_PERM_WRITE  = 2,   // Add / update vectors
    GV_PERM_DELETE = 4,   // Delete vectors
    GV_PERM_ADMIN  = 8,   // Manage users / namespaces
    GV_PERM_ALL    = 15   // All permissions
} GV_Permission;

typedef enum {
    GV_RESOURCE_GLOBAL    = 0,   // Database level
    GV_RESOURCE_NAMESPACE = 1,   // Specific namespace
    GV_RESOURCE_VECTOR    = 2    // Specific vector
} GV_ResourceType;
```

#### Lifecycle

```c
GV_AuthzManager *gv_authz_create(void);
void gv_authz_destroy(GV_AuthzManager *authz);
int gv_authz_init_builtin_roles(GV_AuthzManager *authz);  // creates "admin", "reader", "writer"
```

#### Role Management

```c
int gv_authz_define_role(GV_AuthzManager *authz, const char *name,
                          uint32_t permissions, const char **namespaces,
                          size_t namespace_count);
int gv_authz_remove_role(GV_AuthzManager *authz, const char *name);
int gv_authz_get_role(GV_AuthzManager *authz, const char *name, GV_Role *role);
int gv_authz_list_roles(GV_AuthzManager *authz, GV_Role **roles, size_t *count);
```

#### User-Role Assignment

```c
int gv_authz_assign_role(GV_AuthzManager *authz, const char *subject, const char *role_name);
int gv_authz_revoke_role(GV_AuthzManager *authz, const char *subject, const char *role_name);
```

#### Authorization Checks

```c
int gv_authz_check(GV_AuthzManager *authz, const GV_Identity *identity,
                    GV_Permission permission, GV_ResourceType resource_type,
                    const char *resource_name, GV_AuthzResult *result);

// Convenience helpers (return 1 if allowed, 0 if denied)
int gv_authz_can_read(GV_AuthzManager *authz, const GV_Identity *identity, const char *ns);
int gv_authz_can_write(GV_AuthzManager *authz, const GV_Identity *identity, const char *ns);
int gv_authz_can_delete(GV_AuthzManager *authz, const GV_Identity *identity, const char *ns);
int gv_authz_is_admin(GV_AuthzManager *authz, const GV_Identity *identity);
```

**Example:**
```c
GV_AuthzManager *authz = gv_authz_create();
gv_authz_init_builtin_roles(authz);

// Define a custom role scoped to one namespace
const char *ns[] = {"production"};
gv_authz_define_role(authz, "prod_reader", GV_PERM_READ, ns, 1);

// Assign role to a user
gv_authz_assign_role(authz, "user_123", "prod_reader");

// Check access
GV_AuthzResult result;
gv_authz_check(authz, identity, GV_PERM_READ, GV_RESOURCE_NAMESPACE, "production", &result);
if (result.allowed) { /* proceed */ }

gv_authz_destroy(authz);
```

---

## GPU Acceleration

Requires CUDA. Enabled automatically when available.

```c
// Check GPU availability
if (gv_gpu_available()) {
    GV_GPUInfo info;
    gv_gpu_get_info(&info);
    printf("GPU: %s, Memory: %zu MB\n", info.name, info.memory_mb);
}

// Enable GPU for database
gv_db_enable_gpu(db, 0);  // Device 0

// GPU-accelerated search
gv_db_search_gpu(db, query, k, distance_type, results);

// Batch search on GPU
gv_db_search_batch_gpu(db, queries, num_queries, k, distance_type, all_results);
```

---

## Streaming

For large result sets or continuous queries.

```c
// Create stream
GV_Stream *stream = gv_stream_search(db, query, GV_DISTANCE_COSINE);

// Read results incrementally
GV_SearchResult result;
while (gv_stream_next(stream, &result) == 0) {
    process_result(&result);
}

gv_stream_close(stream);
```

---

## Sharding and Clustering

### Sharding

```c
GV_ShardConfig config = {
    .num_shards = 4,
    .strategy = GV_SHARD_HASH  // or GV_SHARD_RANGE
};
GV_ShardedDB *sdb = gv_shard_create(config);

// Add shard
gv_shard_add(sdb, "shard_0", "host1:8080");

// Operations route automatically
gv_shard_add_vector(sdb, vector, dim, metadata);
gv_shard_search(sdb, query, k, results);
```

### Replication

```c
GV_ReplicationConfig config = {
    .mode = GV_REPL_ASYNC,
    .replicas = 2
};
gv_replication_enable(db, &config);
gv_replication_add_replica(db, "replica1:8080");
```

---

## Thread Safety

GigaVector databases are thread-safe for concurrent reads. Writes require external synchronization.

**Safe:**
- Multiple threads reading simultaneously
- Multiple threads calling `gv_db_search()` concurrently

**Requires Synchronization:**
- Concurrent writes (`gv_db_add_vector()`)
- Mixed read/write operations

**Example with Mutex:**
```c
pthread_mutex_t db_mutex = PTHREAD_MUTEX_INITIALIZER;

// Thread 1
pthread_mutex_lock(&db_mutex);
gv_db_add_vector(db, vector1, 128);
pthread_mutex_unlock(&db_mutex);

// Thread 2
pthread_mutex_lock(&db_mutex);
gv_db_add_vector(db, vector2, 128);
pthread_mutex_unlock(&db_mutex);
```

---

## Resource Management

### Resource Limits

```c
typedef struct {
    size_t max_memory_bytes;           // Maximum memory usage (0 = unlimited)
    size_t max_vectors;                // Maximum number of vectors (0 = unlimited)
    size_t max_concurrent_operations;  // Maximum concurrent operations (0 = unlimited)
} GV_ResourceLimits;
```

#### `gv_db_set_resource_limits`

```c
int gv_db_set_resource_limits(GV_Database *db, const GV_ResourceLimits *limits);
```

Applies resource constraints to the database. Inserts and operations will be rejected when limits are exceeded.

**Returns:** 0 on success, -1 on error

#### `gv_db_get_resource_limits`

```c
void gv_db_get_resource_limits(const GV_Database *db, GV_ResourceLimits *limits);
```

#### `gv_db_get_memory_usage`

```c
size_t gv_db_get_memory_usage(const GV_Database *db);
```

Returns estimated memory usage in bytes.

#### `gv_db_get_concurrent_operations`

```c
size_t gv_db_get_concurrent_operations(const GV_Database *db);
```

Returns the number of currently active operations.

**Example:**
```c
GV_ResourceLimits limits = {
    .max_memory_bytes = 1024 * 1024 * 1024,  // 1GB
    .max_vectors = 1000000,
    .max_concurrent_operations = 100
};
gv_db_set_resource_limits(db, &limits);

size_t mem = gv_db_get_memory_usage(db);
printf("Memory usage: %zu bytes\n", mem);
```

### Health Check

#### `gv_db_health_check`

```c
int gv_db_health_check(const GV_Database *db);
```

Checks database integrity, index consistency, and resource usage.

**Returns:**
- `0` -- healthy
- `-1` -- degraded (e.g., high deleted ratio, approaching resource limits)
- `-2` -- unhealthy (e.g., corrupt index, exceeded limits)

### Memory-Mapped Loading

#### `gv_db_open_mmap`

```c
GV_Database *gv_db_open_mmap(const char *filepath, size_t dimension, GV_IndexType index_type);
```

Opens a database by memory-mapping an existing snapshot file. The database is **read-only**: WAL is disabled and modifications are not persisted. Useful for fast startup and sharing data across processes.

**Returns:** Database handle or NULL on error

**Example:**
```c
GV_Database *db = gv_db_open_mmap("snapshot.db", 128, GV_INDEX_TYPE_HNSW);
// Search works normally
int count = gv_db_search(db, query, 10, GV_DISTANCE_COSINE, results);
// Writes are not persisted
gv_db_close(db);
```

### Compaction Control

#### `gv_db_compact`

```c
int gv_db_compact(GV_Database *db);
```

Triggers immediate compaction -- reclaims space from deleted vectors and compacts the WAL.

**Returns:** 0 on success, -1 on error

#### `gv_db_start_background_compaction` / `gv_db_stop_background_compaction`

```c
int gv_db_start_background_compaction(GV_Database *db);
void gv_db_stop_background_compaction(GV_Database *db);
```

Starts or stops a background thread that runs compaction periodically.

#### `gv_db_set_compaction_interval`

```c
void gv_db_set_compaction_interval(GV_Database *db, size_t interval_sec);
```

Sets the interval (in seconds) between background compaction runs. Default: 300.

#### `gv_db_set_wal_compaction_threshold`

```c
void gv_db_set_wal_compaction_threshold(GV_Database *db, size_t threshold_bytes);
```

Sets the WAL size threshold that triggers compaction. Default: 10 MB.

#### `gv_db_set_deleted_ratio_threshold`

```c
void gv_db_set_deleted_ratio_threshold(GV_Database *db, double ratio);
```

Sets the ratio of deleted vectors (0.0--1.0) that triggers compaction. Default: 0.1.

### Exact Search Control

#### `gv_db_set_exact_search_threshold`

```c
void gv_db_set_exact_search_threshold(GV_Database *db, size_t threshold);
```

Sets the maximum collection size below which search falls back to brute-force exact scan (bypassing the index). Useful for small collections where index overhead is wasteful.

#### `gv_db_set_force_exact_search`

```c
void gv_db_set_force_exact_search(GV_Database *db, int enabled);
```

When enabled (non-zero), all searches use brute-force exact scan regardless of collection size.

### IVF-PQ Per-Query Tuning

#### `gv_db_search_ivfpq_opts`

```c
int gv_db_search_ivfpq_opts(const GV_Database *db, const float *query_data, size_t k,
                            GV_SearchResult *results, GV_DistanceType distance_type,
                            size_t nprobe_override, size_t rerank_top);
```

Performs IVF-PQ search with per-query overrides for nprobe and rerank pool size.

**Parameters:**
- `nprobe_override`: Number of inverted lists to probe (0 uses default from config)
- `rerank_top`: Number of candidates to rerank with full-precision vectors (0 to disable)

**Returns:** Number of results found

### Persistence

#### `gv_db_save`

```c
int gv_db_save(const GV_Database *db, const char *filepath);
```

Saves database snapshot. WAL is automatically replayed on next open.

---

## Performance Tips

1. **Choose the right index**: Use `gv_index_suggest()` for guidance
2. **Enable cosine normalization**: For cosine distance queries
3. **Batch operations**: Use batch embedding APIs when available
4. **Monitor statistics**: Use `gv_db_get_detailed_stats()` for optimization
5. **Tune HNSW parameters**: Adjust M and ef_construction for your workload

---

For more examples, see [Basic Usage Examples](examples/basic_usage.md) and [Advanced Features](examples/advanced_features.md).

