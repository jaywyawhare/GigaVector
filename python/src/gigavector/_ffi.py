"""GigaVector CFFI bindings for the C library.

This module provides low-level bindings to the libGigaVector shared library
using CFFI (C Foreign Function Interface). The bindings expose all core
functionality including:

- Vector database operations (create, add, search, delete)
- Multiple index types (KD-Tree, HNSW, IVFPQ, Sparse)
- Distance metrics (Euclidean, Cosine, Dot Product, Manhattan)
- LLM integration for memory extraction
- Embedding service for text-to-vector conversion
- Context graph for entity/relationship extraction
- Memory layer for semantic memory storage

Type Aliases:
    CData: CFFI pointer type (cffi.FFI.CData)
    GV_Database: Opaque database handle
    GV_LLM: Opaque LLM client handle
    GV_EmbeddingService: Opaque embedding service handle
    GV_ContextGraph: Opaque context graph handle
    GV_MemoryLayer: Memory layer structure

Note:
    This module is internal. Users should use the high-level classes in
    _core.py (Database, LLM, EmbeddingService, ContextGraph, MemoryLayer).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from cffi import FFI

if TYPE_CHECKING:
    from cffi import FFI as FFIType

ffi: FFIType = FFI()

# =============================================================================
# C Type Definitions
# =============================================================================
# Keep this cdef in sync with the C headers (include/gigavector.h).
# The definitions below map directly to the C structures and functions
# exposed by libGigaVector.so.
ffi.cdef(
    """
typedef long time_t;  // Define time_t for FFI
typedef enum { GV_INDEX_TYPE_KDTREE = 0, GV_INDEX_TYPE_HNSW = 1, GV_INDEX_TYPE_IVFPQ = 2, GV_INDEX_TYPE_SPARSE = 3, GV_INDEX_TYPE_FLAT = 4, GV_INDEX_TYPE_IVFFLAT = 5, GV_INDEX_TYPE_PQ = 6, GV_INDEX_TYPE_LSH = 7 } GV_IndexType;
typedef enum { GV_DISTANCE_EUCLIDEAN = 0, GV_DISTANCE_COSINE = 1, GV_DISTANCE_DOT_PRODUCT = 2, GV_DISTANCE_MANHATTAN = 3, GV_DISTANCE_HAMMING = 4 } GV_DistanceType;

typedef struct {
    uint32_t index;
    float value;
} GV_SparseEntry;

typedef struct GV_SparseVector {
    size_t dimension;
    size_t nnz;
    GV_SparseEntry *entries;
    void *metadata; /* GV_Metadata* */
} GV_SparseVector;

typedef struct {
    size_t M;
    size_t efConstruction;
    size_t efSearch;
    size_t maxLevel;
    int use_binary_quant;
    size_t quant_rerank;
    int use_acorn;
    size_t acorn_hops;
} GV_HNSWConfig;

typedef struct {
    uint8_t bits;
    int per_dimension;
} GV_ScalarQuantConfig;

typedef struct {
    size_t nlist;
    size_t m;
    uint8_t nbits;
    size_t nprobe;
    size_t train_iters;
    size_t default_rerank;
    int use_cosine;
    int use_scalar_quant;
    GV_ScalarQuantConfig scalar_quant_config;
    float oversampling_factor;
} GV_IVFPQConfig;

typedef struct {
    size_t nlist;
    size_t nprobe;
    size_t train_iters;
    int use_cosine;
} GV_IVFFlatConfig;

typedef struct {
    size_t m;
    uint8_t nbits;
    size_t train_iters;
} GV_PQConfig;

typedef struct {
    size_t num_tables;
    size_t num_hash_bits;
    uint64_t seed;
} GV_LSHConfig;

typedef struct GV_Metadata {
    char *key;
    char *value;
    struct GV_Metadata *next;
} GV_Metadata;

typedef struct {
    size_t dimension;
    float *data;
    GV_Metadata *metadata;
} GV_Vector;

typedef struct GV_KDNode {
    GV_Vector *point;
    size_t axis;
    struct GV_KDNode *left;
    struct GV_KDNode *right;
} GV_KDNode;

typedef struct GV_WAL GV_WAL;

typedef struct GV_Database {
    size_t dimension;
    GV_IndexType index_type;
    GV_KDNode *root;
    void *hnsw_index;
    char *filepath;
    char *wal_path;
    GV_WAL *wal;
    int wal_replaying;
    void *rwlock;  // pthread_rwlock_t - opaque for FFI
    void *wal_mutex;  // pthread_mutex_t - opaque for FFI
    size_t count;
} GV_Database;

typedef struct {
    uint64_t total_inserts;
    uint64_t total_queries;
    uint64_t total_range_queries;
    uint64_t total_wal_records;
} GV_DBStats;

typedef struct {
    const GV_Vector *vector;
    const GV_SparseVector *sparse_vector;
    int is_sparse;
    float distance;
} GV_SearchResult;

GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type);
GV_Database *gv_db_open_with_hnsw_config(const char *filepath, size_t dimension, GV_IndexType index_type, const GV_HNSWConfig *hnsw_config);
GV_Database *gv_db_open_with_ivfpq_config(const char *filepath, size_t dimension, GV_IndexType index_type, const GV_IVFPQConfig *ivfpq_config);
GV_Database *gv_db_open_with_ivfflat_config(const char *filepath, size_t dimension, GV_IndexType index_type, const GV_IVFFlatConfig *config);
GV_Database *gv_db_open_with_pq_config(const char *filepath, size_t dimension, GV_IndexType index_type, const GV_PQConfig *config);
GV_Database *gv_db_open_with_lsh_config(const char *filepath, size_t dimension, GV_IndexType index_type, const GV_LSHConfig *config);
GV_Database *gv_db_open_from_memory(const void *data, size_t size,
                                    size_t dimension, GV_IndexType index_type);
GV_Database *gv_db_open_mmap(const char *filepath, size_t dimension, GV_IndexType index_type);
GV_IndexType gv_index_suggest(size_t dimension, size_t expected_count);
void gv_db_get_stats(const GV_Database *db, GV_DBStats *out);
void gv_db_set_cosine_normalized(GV_Database *db, int enabled);
void gv_db_close(GV_Database *db);

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);
int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                    const char *metadata_key, const char *metadata_value);
int gv_db_add_vector_with_rich_metadata(GV_Database *db, const float *data, size_t dimension,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count);
int gv_db_delete_vector_by_index(GV_Database *db, size_t vector_index);
int gv_db_update_vector(GV_Database *db, size_t vector_index, const float *new_data, size_t dimension);
int gv_db_update_vector_metadata(GV_Database *db, size_t vector_index,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count);
int gv_db_save(const GV_Database *db, const char *filepath);
int gv_db_ivfpq_train(GV_Database *db, const float *data, size_t count, size_t dimension);
int gv_db_ivfflat_train(GV_Database *db, const float *data, size_t count, size_t dimension);
int gv_db_pq_train(GV_Database *db, const float *data, size_t count, size_t dimension);
int gv_db_add_vectors(GV_Database *db, const float *data, size_t count, size_t dimension);
int gv_db_add_vectors_with_metadata(GV_Database *db, const float *data,
                                    const char *const *keys, const char *const *values,
                                    size_t count, size_t dimension);

int gv_db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type);
int gv_db_search_filtered(const GV_Database *db, const float *query_data, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type,
                          const char *filter_key, const char *filter_value);
int gv_db_search_batch(const GV_Database *db, const float *queries, size_t qcount, size_t k,
                       GV_SearchResult *results, GV_DistanceType distance_type);
int gv_db_search_with_filter_expr(const GV_Database *db, const float *query_data, size_t k,
                                   GV_SearchResult *results, GV_DistanceType distance_type,
                                   const char *filter_expr);
int gv_db_search_ivfpq_opts(const GV_Database *db, const float *query_data, size_t k,
                  GV_SearchResult *results, GV_DistanceType distance_type,
                  size_t nprobe_override, size_t rerank_top);
void gv_db_set_exact_search_threshold(GV_Database *db, size_t threshold);
void gv_db_set_force_exact_search(GV_Database *db, int enabled);
int gv_db_add_sparse_vector(GV_Database *db, const uint32_t *indices, const float *values,
                            size_t nnz, size_t dimension,
                            const char *metadata_key, const char *metadata_value);
int gv_db_search_sparse(const GV_Database *db, const uint32_t *indices, const float *values,
                        size_t nnz, size_t k, GV_SearchResult *results, GV_DistanceType distance_type);
int gv_db_range_search(const GV_Database *db, const float *query_data, float radius,
                       GV_SearchResult *results, size_t max_results, GV_DistanceType distance_type);
int gv_db_range_search_filtered(const GV_Database *db, const float *query_data, float radius,
                                 GV_SearchResult *results, size_t max_results,
                                 GV_DistanceType distance_type,
                                 const char *filter_key, const char *filter_value);

// Vector creation and metadata management
GV_Vector *gv_vector_create_from_data(size_t dimension, const float *data);
int gv_vector_set_metadata(GV_Vector *vector, const char *key, const char *value);
void gv_vector_destroy(GV_Vector *vector);

// Index insertion functions
int gv_kdtree_insert(GV_KDNode **root, GV_Vector *point, size_t depth);
int gv_hnsw_insert(void *index, GV_Vector *vector);
int gv_ivfpq_insert(void *index, GV_Vector *vector);

// WAL functions
int gv_wal_append_insert(GV_WAL *wal, const float *data, size_t dimension,
                         const char *metadata_key, const char *metadata_value);
int gv_wal_append_insert_rich(GV_WAL *wal, const float *data, size_t dimension,
                              const char *const *metadata_keys, const char *const *metadata_values,
                              size_t metadata_count);
int gv_wal_truncate(GV_WAL *wal);

// Resource limits
typedef struct {
    size_t max_memory_bytes;
    size_t max_vectors;
    size_t max_concurrent_operations;
} GV_ResourceLimits;

int gv_db_set_resource_limits(GV_Database *db, const GV_ResourceLimits *limits);
void gv_db_get_resource_limits(const GV_Database *db, GV_ResourceLimits *limits);
size_t gv_db_get_memory_usage(const GV_Database *db);
size_t gv_db_get_concurrent_operations(const GV_Database *db);

// Compaction functions
int gv_db_start_background_compaction(GV_Database *db);
void gv_db_stop_background_compaction(GV_Database *db);
int gv_db_compact(GV_Database *db);
void gv_db_set_compaction_interval(GV_Database *db, size_t interval_sec);
void gv_db_set_wal_compaction_threshold(GV_Database *db, size_t threshold_bytes);
void gv_db_set_deleted_ratio_threshold(GV_Database *db, double ratio);

// Observability structures
typedef struct {
    uint64_t *buckets;
    size_t bucket_count;
    double *bucket_boundaries;
    uint64_t total_samples;
    uint64_t sum_latency_us;
} GV_LatencyHistogram;

typedef struct {
    size_t soa_storage_bytes;
    size_t index_bytes;
    size_t metadata_index_bytes;
    size_t wal_bytes;
    size_t total_bytes;
} GV_MemoryBreakdown;

typedef struct {
    uint64_t total_queries;
    double avg_recall;
    double min_recall;
    double max_recall;
} GV_RecallMetrics;

typedef struct {
    GV_DBStats basic_stats;
    GV_LatencyHistogram insert_latency;
    GV_LatencyHistogram search_latency;
    double queries_per_second;
    double inserts_per_second;
    uint64_t last_qps_update_time;
    GV_MemoryBreakdown memory;
    GV_RecallMetrics recall;
    int health_status;
    size_t deleted_vector_count;
    double deleted_ratio;
} GV_DetailedStats;

// Observability functions
int gv_db_get_detailed_stats(const GV_Database *db, GV_DetailedStats *out);
void gv_db_free_detailed_stats(GV_DetailedStats *stats);
int gv_db_health_check(const GV_Database *db);
void gv_db_record_latency(GV_Database *db, uint64_t latency_us, int is_insert);
void gv_db_record_recall(GV_Database *db, double recall);

// LLM types
typedef enum { GV_LLM_PROVIDER_OPENAI = 0, GV_LLM_PROVIDER_ANTHROPIC = 1, GV_LLM_PROVIDER_GOOGLE = 2, GV_LLM_PROVIDER_CUSTOM = 3 } GV_LLMProvider;

typedef struct {
    GV_LLMProvider provider;
    char *api_key;
    char *model;
    char *base_url;
    double temperature;
    int max_tokens;
    int timeout_seconds;
    char *custom_prompt;
} GV_LLMConfig;

typedef struct {
    char *role;
    char *content;
} GV_LLMMessage;

typedef struct {
    char *content;
    int finish_reason;
    int token_count;
} GV_LLMResponse;

typedef struct GV_LLM GV_LLM;

// LLM functions
GV_LLM *gv_llm_create(const GV_LLMConfig *config);
void gv_llm_destroy(GV_LLM *llm);
int gv_llm_generate_response(GV_LLM *llm, const GV_LLMMessage *messages, size_t message_count, const char *response_format, GV_LLMResponse *response);
void gv_llm_response_free(GV_LLMResponse *response);
void gv_llm_message_free(GV_LLMMessage *message);
void gv_llm_messages_free(GV_LLMMessage *messages, size_t count);

// Embedding service types
typedef enum { GV_EMBEDDING_PROVIDER_OPENAI = 0, GV_EMBEDDING_PROVIDER_HUGGINGFACE = 1, GV_EMBEDDING_PROVIDER_CUSTOM = 2, GV_EMBEDDING_PROVIDER_NONE = 3 } GV_EmbeddingProvider;

typedef struct {
    GV_EmbeddingProvider provider;
    char *api_key;
    char *model;
    char *base_url;
    size_t embedding_dimension;
    size_t batch_size;
    int enable_cache;
    size_t cache_size;
    int timeout_seconds;
    char *huggingface_model_path;
} GV_EmbeddingConfig;

typedef struct GV_EmbeddingService GV_EmbeddingService;
typedef struct GV_EmbeddingCache GV_EmbeddingCache;

// Embedding service functions
GV_EmbeddingService *gv_embedding_service_create(const GV_EmbeddingConfig *config);
void gv_embedding_service_destroy(GV_EmbeddingService *service);
int gv_embedding_generate(GV_EmbeddingService *service, const char *text, size_t *embedding_dim, float **embedding);
int gv_embedding_generate_batch(GV_EmbeddingService *service, const char **texts, size_t text_count, size_t **embedding_dims, float ***embeddings);
GV_EmbeddingConfig gv_embedding_config_default(void);
void gv_embedding_config_free(GV_EmbeddingConfig *config);
GV_EmbeddingCache *gv_embedding_cache_create(size_t max_size);
void gv_embedding_cache_destroy(GV_EmbeddingCache *cache);
int gv_embedding_cache_get(GV_EmbeddingCache *cache, const char *text, size_t *embedding_dim, const float **embedding);
int gv_embedding_cache_put(GV_EmbeddingCache *cache, const char *text, size_t embedding_dim, const float *embedding);
void gv_embedding_cache_clear(GV_EmbeddingCache *cache);
void gv_embedding_cache_stats(GV_EmbeddingCache *cache, size_t *size, uint64_t *hits, uint64_t *misses);
const char *gv_embedding_get_last_error(GV_EmbeddingService *service);

// Context graph types
typedef enum { GV_ENTITY_TYPE_PERSON = 0, GV_ENTITY_TYPE_ORGANIZATION = 1, GV_ENTITY_TYPE_LOCATION = 2, GV_ENTITY_TYPE_EVENT = 3, GV_ENTITY_TYPE_OBJECT = 4, GV_ENTITY_TYPE_CONCEPT = 5, GV_ENTITY_TYPE_USER = 6 } GV_EntityType;

typedef struct {
    char *entity_id;
    char *name;
    GV_EntityType entity_type;
    float *embedding;
    size_t embedding_dim;
    time_t created;
    time_t updated;
    uint64_t mentions;
    char *user_id;
    char *agent_id;
    char *run_id;
} GV_GraphEntity;

typedef struct {
    char *relationship_id;
    char *source_entity_id;
    char *destination_entity_id;
    char *relationship_type;
    time_t created;
    time_t updated;
    uint64_t mentions;
} GV_GraphRelationship;

typedef struct {
    char *source_name;
    char *relationship_type;
    char *destination_name;
    float similarity;
} GV_GraphQueryResult;

typedef struct GV_ContextGraph GV_ContextGraph;

typedef float *(*GV_EmbeddingCallback)(const char *text, size_t *embedding_dim, void *user_data);

typedef struct {
    void *llm;
    double similarity_threshold;
    int enable_entity_extraction;
    int enable_relationship_extraction;
    size_t max_traversal_depth;
    size_t max_results;
    GV_EmbeddingCallback embedding_callback;
    void *embedding_user_data;
    size_t embedding_dimension;
} GV_ContextGraphConfig;

// Context graph functions
GV_ContextGraph *gv_context_graph_create(const GV_ContextGraphConfig *config);
void gv_context_graph_destroy(GV_ContextGraph *graph);
int gv_context_graph_extract(GV_ContextGraph *graph, const char *text, const char *user_id, const char *agent_id, const char *run_id, GV_GraphEntity **entities, size_t *entity_count, GV_GraphRelationship **relationships, size_t *relationship_count);
int gv_context_graph_add_entities(GV_ContextGraph *graph, const GV_GraphEntity *entities, size_t entity_count);
int gv_context_graph_add_relationships(GV_ContextGraph *graph, const GV_GraphRelationship *relationships, size_t relationship_count);
int gv_context_graph_search(GV_ContextGraph *graph, const float *query_embedding, size_t embedding_dim, const char *user_id, const char *agent_id, const char *run_id, GV_GraphQueryResult *results, size_t max_results);
int gv_context_graph_get_related(GV_ContextGraph *graph, const char *entity_id, size_t max_depth, GV_GraphQueryResult *results, size_t max_results);
int gv_context_graph_delete_entities(GV_ContextGraph *graph, const char **entity_ids, size_t entity_count);
int gv_context_graph_delete_relationships(GV_ContextGraph *graph, const char **relationship_ids, size_t relationship_count);
void gv_graph_entity_free(GV_GraphEntity *entity);
void gv_graph_relationship_free(GV_GraphRelationship *relationship);
void gv_graph_query_result_free(GV_GraphQueryResult *result);
GV_ContextGraphConfig gv_context_graph_config_default(void);

// Memory layer types
typedef enum { GV_MEMORY_TYPE_FACT = 0, GV_MEMORY_TYPE_PREFERENCE = 1, GV_MEMORY_TYPE_RELATIONSHIP = 2, GV_MEMORY_TYPE_EVENT = 3 } GV_MemoryType;
typedef enum { GV_CONSOLIDATION_MERGE = 0, GV_CONSOLIDATION_UPDATE = 1, GV_CONSOLIDATION_LINK = 2, GV_CONSOLIDATION_ARCHIVE = 3 } GV_ConsolidationStrategy;

typedef struct {
    char *memory_id;
    GV_MemoryType memory_type;
    char *source;
    time_t timestamp;
    double importance_score;
    char *extraction_metadata;
    char **related_memory_ids;
    size_t related_count;
    int consolidated;
} GV_MemoryMetadata;

typedef struct {
    char *memory_id;
    char *content;
    float relevance_score;
    float distance;
    GV_MemoryMetadata *metadata;
    GV_MemoryMetadata **related;
    size_t related_count;
} GV_MemoryResult;

typedef struct {
    double extraction_threshold;
    double consolidation_threshold;
    GV_ConsolidationStrategy default_strategy;
    int enable_temporal_weighting;
    int enable_relationship_retrieval;
    size_t max_related_memories;
    void *llm_config;
    int use_llm_extraction;
    int use_llm_consolidation;
} GV_MemoryLayerConfig;

typedef struct GV_MemoryLayer {
    GV_Database *db;
    GV_MemoryLayerConfig config;
    uint64_t next_memory_id;
    void *mutex;
} GV_MemoryLayer;

// Memory layer functions
GV_MemoryLayerConfig gv_memory_layer_config_default(void);
GV_MemoryLayer *gv_memory_layer_create(GV_Database *db, const GV_MemoryLayerConfig *config);
void gv_memory_layer_destroy(GV_MemoryLayer *layer);
char *gv_memory_add(GV_MemoryLayer *layer, const char *content, const float *embedding, GV_MemoryMetadata *metadata);
char **gv_memory_extract_from_conversation(GV_MemoryLayer *layer, const char *conversation, const char *conversation_id, float **embeddings, size_t *memory_count);
char **gv_memory_extract_from_text(GV_MemoryLayer *layer, const char *text, const char *source, float **embeddings, size_t *memory_count);
int gv_memory_extract_candidates_from_conversation_llm(GV_LLM *llm, const char *conversation, const char *conversation_id, int is_agent_memory, const char *custom_prompt, void *candidates, size_t max_candidates, size_t *actual_count);
const char *gv_llm_get_last_error(GV_LLM *llm);
const char *gv_llm_error_string(int error_code);
typedef enum { GV_LLM_SUCCESS = 0, GV_LLM_ERROR_NULL_POINTER = -1, GV_LLM_ERROR_INVALID_CONFIG = -2, GV_LLM_ERROR_INVALID_API_KEY = -3, GV_LLM_ERROR_INVALID_URL = -4, GV_LLM_ERROR_MEMORY_ALLOCATION = -5, GV_LLM_ERROR_CURL_INIT = -6, GV_LLM_ERROR_NETWORK = -7, GV_LLM_ERROR_TIMEOUT = -8, GV_LLM_ERROR_RESPONSE_TOO_LARGE = -9, GV_LLM_ERROR_PARSE_FAILED = -10, GV_LLM_ERROR_INVALID_RESPONSE = -11, GV_LLM_ERROR_CUSTOM_URL_REQUIRED = -12 } GV_LLMError;
int gv_memory_consolidate(GV_MemoryLayer *layer, double threshold, int strategy);
int gv_memory_search(GV_MemoryLayer *layer, const float *query_embedding, size_t k, GV_MemoryResult *results, GV_DistanceType distance_type);
int gv_memory_search_filtered(GV_MemoryLayer *layer, const float *query_embedding, size_t k, GV_MemoryResult *results, GV_DistanceType distance_type, int memory_type, const char *source, time_t min_timestamp, time_t max_timestamp);
int gv_memory_get_related(GV_MemoryLayer *layer, const char *memory_id, size_t k, GV_MemoryResult *results);
int gv_memory_get(GV_MemoryLayer *layer, const char *memory_id, GV_MemoryResult *result);
int gv_memory_update(GV_MemoryLayer *layer, const char *memory_id, const float *new_embedding, GV_MemoryMetadata *new_metadata);
int gv_memory_delete(GV_MemoryLayer *layer, const char *memory_id);
void gv_memory_result_free(GV_MemoryResult *result);
void gv_memory_metadata_free(GV_MemoryMetadata *metadata);

// ============================================================================
// Database Accessor Functions
// ============================================================================
size_t gv_database_count(const GV_Database *db);
size_t gv_database_dimension(const GV_Database *db);
const float *gv_database_get_vector(const GV_Database *db, size_t index);

// ============================================================================
// Upsert, Batch Delete, Scroll, Search Params, JSON Import/Export
// ============================================================================
int gv_db_upsert(GV_Database *db, size_t vector_index, const float *data, size_t dimension);
int gv_db_upsert_with_metadata(GV_Database *db, size_t vector_index,
                                const float *data, size_t dimension,
                                const char *const *metadata_keys,
                                const char *const *metadata_values,
                                size_t metadata_count);
int gv_db_delete_vectors(GV_Database *db, const size_t *indices, size_t count);

typedef struct {
    size_t index;
    const float *data;
    size_t dimension;
    GV_Metadata *metadata;
} GV_ScrollResult;

int gv_db_scroll(const GV_Database *db, size_t offset, size_t limit,
                 GV_ScrollResult *results);

typedef struct {
    size_t ef_search;
    size_t nprobe;
    size_t rerank_top;
} GV_SearchParams;

int gv_db_search_with_params(const GV_Database *db, const float *query_data, size_t k,
                              GV_SearchResult *results, GV_DistanceType distance_type,
                              const GV_SearchParams *params);

int gv_db_export_json(const GV_Database *db, const char *filepath);
int gv_db_import_json(GV_Database *db, const char *filepath);

// ============================================================================
// GPU Acceleration
// ============================================================================
typedef struct {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
} GV_GPUDeviceInfo;

typedef struct {
    size_t initial_size;
    size_t max_size;
    int allow_growth;
} GV_GPUMemoryConfig;

typedef struct {
    int device_id;
    size_t max_vectors_per_batch;
    size_t max_query_batch_size;
    int enable_tensor_cores;
    int enable_async_transfers;
    int stream_count;
    GV_GPUMemoryConfig memory;
} GV_GPUConfig;

typedef enum { GV_GPU_EUCLIDEAN = 0, GV_GPU_COSINE = 1, GV_GPU_DOT_PRODUCT = 2, GV_GPU_MANHATTAN = 3 } GV_GPUDistanceMetric;

typedef struct {
    GV_GPUDistanceMetric metric;
    size_t k;
    float radius;
    int use_precomputed_norms;
} GV_GPUSearchParams;

typedef struct {
    uint64_t total_searches;
    uint64_t total_vectors_processed;
    uint64_t total_distance_computations;
    double total_gpu_time_ms;
    double total_transfer_time_ms;
    double avg_search_time_ms;
    size_t peak_memory_usage;
    size_t current_memory_usage;
} GV_GPUStats;

typedef struct GV_GPUContext GV_GPUContext;
typedef struct GV_GPUIndex GV_GPUIndex;

int gv_gpu_available(void);
int gv_gpu_device_count(void);
int gv_gpu_get_device_info(int device_id, GV_GPUDeviceInfo *info);
void gv_gpu_config_init(GV_GPUConfig *config);
GV_GPUContext *gv_gpu_create(const GV_GPUConfig *config);
void gv_gpu_destroy(GV_GPUContext *ctx);
int gv_gpu_synchronize(GV_GPUContext *ctx);
GV_GPUIndex *gv_gpu_index_create(GV_GPUContext *ctx, const float *vectors, size_t count, size_t dimension);
GV_GPUIndex *gv_gpu_index_from_db(GV_GPUContext *ctx, GV_Database *db);
int gv_gpu_index_add(GV_GPUIndex *index, const float *vectors, size_t count);
int gv_gpu_index_remove(GV_GPUIndex *index, const size_t *indices, size_t count);
int gv_gpu_index_update(GV_GPUIndex *index, const size_t *indices, const float *vectors, size_t count);
int gv_gpu_index_info(GV_GPUIndex *index, size_t *count, size_t *dimension, size_t *memory_usage);
void gv_gpu_index_destroy(GV_GPUIndex *index);
int gv_gpu_compute_distances(GV_GPUContext *ctx, const float *queries, size_t num_queries, const float *database, size_t num_vectors, size_t dimension, GV_GPUDistanceMetric metric, float *distances);
int gv_gpu_index_compute_distances(GV_GPUIndex *index, const float *queries, size_t num_queries, GV_GPUDistanceMetric metric, float *distances);
int gv_gpu_knn_search(GV_GPUContext *ctx, const float *queries, size_t num_queries, const float *database, size_t num_vectors, size_t dimension, const GV_GPUSearchParams *params, size_t *indices, float *distances);
int gv_gpu_index_knn_search(GV_GPUIndex *index, const float *queries, size_t num_queries, const GV_GPUSearchParams *params, size_t *indices, float *distances);
int gv_gpu_index_search(GV_GPUIndex *index, const float *query, const GV_GPUSearchParams *params, size_t *indices, float *distances);
int gv_gpu_batch_add(GV_GPUContext *ctx, GV_Database *db, const float *vectors, size_t count);
int gv_gpu_batch_search(GV_GPUContext *ctx, GV_Database *db, const float *queries, size_t num_queries, size_t k, size_t *indices, float *distances);
int gv_gpu_get_stats(GV_GPUContext *ctx, GV_GPUStats *stats);
int gv_gpu_reset_stats(GV_GPUContext *ctx);
const char *gv_gpu_get_error(GV_GPUContext *ctx);

// ============================================================================
// HTTP Server & REST API
// ============================================================================
typedef enum { GV_SERVER_OK = 0, GV_SERVER_ERROR_NULL_POINTER = -1, GV_SERVER_ERROR_INVALID_CONFIG = -2, GV_SERVER_ERROR_ALREADY_RUNNING = -3, GV_SERVER_ERROR_NOT_RUNNING = -4, GV_SERVER_ERROR_START_FAILED = -5, GV_SERVER_ERROR_MEMORY = -6, GV_SERVER_ERROR_BIND_FAILED = -7 } GV_ServerError;

typedef struct {
    uint16_t port;
    const char *bind_address;
    size_t thread_pool_size;
    size_t max_connections;
    size_t request_timeout_ms;
    size_t max_request_body_bytes;
    int enable_cors;
    const char *cors_origins;
    int enable_logging;
    const char *api_key;
    double max_requests_per_second;
    size_t rate_limit_burst;
} GV_ServerConfig;

typedef struct {
    uint64_t total_requests;
    uint64_t active_connections;
    uint64_t requests_per_second;
    uint64_t total_bytes_sent;
    uint64_t total_bytes_received;
    uint64_t error_count;
} GV_ServerStats;

typedef struct GV_Server GV_Server;

void gv_server_config_init(GV_ServerConfig *config);
GV_Server *gv_server_create(GV_Database *db, const GV_ServerConfig *config);
int gv_server_start(GV_Server *server);
int gv_server_stop(GV_Server *server);
void gv_server_destroy(GV_Server *server);
int gv_server_is_running(const GV_Server *server);
int gv_server_get_stats(const GV_Server *server, GV_ServerStats *stats);
uint16_t gv_server_get_port(const GV_Server *server);
const char *gv_server_error_string(int error);

// ============================================================================
// Backup & Restore
// ============================================================================
typedef enum { GV_BACKUP_COMPRESS_NONE = 0, GV_BACKUP_COMPRESS_ZLIB = 1, GV_BACKUP_COMPRESS_LZ4 = 2 } GV_BackupCompression;

typedef struct {
    GV_BackupCompression compression;
    int include_wal;
    int include_metadata;
    int verify_after;
    const char *encryption_key;
} GV_BackupOptions;

typedef struct {
    uint32_t version;
    uint32_t flags;
    uint64_t created_at;
    uint64_t vector_count;
    uint32_t dimension;
    uint32_t index_type;
    uint64_t original_size;
    uint64_t compressed_size;
    char checksum[65];
} GV_BackupHeader;

typedef struct {
    int overwrite;
    int verify_checksum;
    const char *decryption_key;
} GV_RestoreOptions;

typedef struct {
    int success;
    char *error_message;
    uint64_t bytes_processed;
    uint64_t vectors_processed;
    double elapsed_seconds;
} GV_BackupResult;

typedef void (*GV_BackupProgressCallback)(size_t current, size_t total, void *user_data);

void gv_backup_options_init(GV_BackupOptions *options);
void gv_restore_options_init(GV_RestoreOptions *options);
GV_BackupResult *gv_backup_create(GV_Database *db, const char *backup_path, const GV_BackupOptions *options, GV_BackupProgressCallback progress, void *user_data);
GV_BackupResult *gv_backup_create_from_file(const char *db_path, const char *backup_path, const GV_BackupOptions *options, GV_BackupProgressCallback progress, void *user_data);
void gv_backup_result_free(GV_BackupResult *result);
GV_BackupResult *gv_backup_restore(const char *backup_path, const char *db_path, const GV_RestoreOptions *options, GV_BackupProgressCallback progress, void *user_data);
GV_BackupResult *gv_backup_restore_to_db(const char *backup_path, const GV_RestoreOptions *options, GV_Database **db);
int gv_backup_read_header(const char *backup_path, GV_BackupHeader *header);
GV_BackupResult *gv_backup_verify(const char *backup_path, const char *decryption_key);
int gv_backup_get_info(const char *backup_path, char *info_buf, size_t buf_size);
GV_BackupResult *gv_backup_create_incremental(GV_Database *db, const char *backup_path, const char *base_backup_path, const GV_BackupOptions *options);
GV_BackupResult *gv_backup_merge(const char *base_backup_path, const char **incremental_paths, size_t incremental_count, const char *output_path);
int gv_backup_compute_checksum(const char *backup_path, char *checksum_out);
const char *gv_backup_compression_string(GV_BackupCompression compression);

// ============================================================================
// Shard Management
// ============================================================================
typedef enum { GV_SHARD_ACTIVE = 0, GV_SHARD_READONLY = 1, GV_SHARD_MIGRATING = 2, GV_SHARD_OFFLINE = 3 } GV_ShardState;
typedef enum { GV_SHARD_HASH = 0, GV_SHARD_RANGE = 1, GV_SHARD_CONSISTENT = 2 } GV_ShardStrategy;

typedef struct {
    uint32_t shard_id;
    char *node_address;
    GV_ShardState state;
    uint64_t vector_count;
    uint64_t capacity;
    uint32_t replica_count;
    uint64_t last_heartbeat;
} GV_ShardInfo;

typedef struct {
    uint32_t shard_count;
    uint32_t virtual_nodes;
    GV_ShardStrategy strategy;
    uint32_t replication_factor;
} GV_ShardConfig;

typedef struct GV_ShardManager GV_ShardManager;

void gv_shard_config_init(GV_ShardConfig *config);
GV_ShardManager *gv_shard_manager_create(const GV_ShardConfig *config);
void gv_shard_manager_destroy(GV_ShardManager *mgr);
int gv_shard_add(GV_ShardManager *mgr, uint32_t shard_id, const char *node_address);
int gv_shard_remove(GV_ShardManager *mgr, uint32_t shard_id);
int gv_shard_for_vector(GV_ShardManager *mgr, uint64_t vector_id);
int gv_shard_for_key(GV_ShardManager *mgr, const void *key, size_t key_len);
int gv_shard_get_info(GV_ShardManager *mgr, uint32_t shard_id, GV_ShardInfo *info);
int gv_shard_list(GV_ShardManager *mgr, GV_ShardInfo **shards, size_t *count);
void gv_shard_free_list(GV_ShardInfo *shards, size_t count);
int gv_shard_set_state(GV_ShardManager *mgr, uint32_t shard_id, GV_ShardState state);
int gv_shard_rebalance_start(GV_ShardManager *mgr);
int gv_shard_rebalance_status(GV_ShardManager *mgr, double *progress);
int gv_shard_rebalance_cancel(GV_ShardManager *mgr);
int gv_shard_attach_local(GV_ShardManager *mgr, uint32_t shard_id, GV_Database *db);
GV_Database *gv_shard_get_local_db(GV_ShardManager *mgr, uint32_t shard_id);

// ============================================================================
// Replication
// ============================================================================
typedef enum { GV_REPL_LEADER = 0, GV_REPL_FOLLOWER = 1, GV_REPL_CANDIDATE = 2 } GV_ReplicationRole;
typedef enum { GV_REPL_SYNCING = 0, GV_REPL_STREAMING = 1, GV_REPL_LAGGING = 2, GV_REPL_DISCONNECTED = 3 } GV_ReplicationState;

typedef struct {
    const char *node_id;
    const char *listen_address;
    const char *leader_address;
    uint32_t sync_interval_ms;
    uint32_t election_timeout_ms;
    uint32_t heartbeat_interval_ms;
    size_t max_lag_entries;
} GV_ReplicationConfig;

typedef struct {
    char *node_id;
    char *address;
    GV_ReplicationRole role;
    GV_ReplicationState state;
    uint64_t last_wal_position;
    uint64_t lag_entries;
    uint64_t last_heartbeat;
} GV_ReplicaInfo;

typedef struct {
    GV_ReplicationRole role;
    uint64_t term;
    char *leader_id;
    size_t follower_count;
    uint64_t wal_position;
    uint64_t commit_position;
    uint64_t bytes_replicated;
} GV_ReplicationStats;

typedef struct GV_ReplicationManager GV_ReplicationManager;

void gv_replication_config_init(GV_ReplicationConfig *config);
GV_ReplicationManager *gv_replication_create(GV_Database *db, const GV_ReplicationConfig *config);
void gv_replication_destroy(GV_ReplicationManager *mgr);
int gv_replication_start(GV_ReplicationManager *mgr);
int gv_replication_stop(GV_ReplicationManager *mgr);
GV_ReplicationRole gv_replication_get_role(GV_ReplicationManager *mgr);
int gv_replication_step_down(GV_ReplicationManager *mgr);
int gv_replication_request_leadership(GV_ReplicationManager *mgr);
int gv_replication_add_follower(GV_ReplicationManager *mgr, const char *node_id, const char *address);
int gv_replication_remove_follower(GV_ReplicationManager *mgr, const char *node_id);
int gv_replication_list_replicas(GV_ReplicationManager *mgr, GV_ReplicaInfo **replicas, size_t *count);
void gv_replication_free_replicas(GV_ReplicaInfo *replicas, size_t count);
int gv_replication_sync_commit(GV_ReplicationManager *mgr, uint32_t timeout_ms);
int64_t gv_replication_get_lag(GV_ReplicationManager *mgr);
int gv_replication_wait_sync(GV_ReplicationManager *mgr, size_t max_lag, uint32_t timeout_ms);
int gv_replication_get_stats(GV_ReplicationManager *mgr, GV_ReplicationStats *stats);
void gv_replication_free_stats(GV_ReplicationStats *stats);
int gv_replication_is_healthy(GV_ReplicationManager *mgr);

// ============================================================================
// Cluster Management
// ============================================================================
typedef enum { GV_NODE_COORDINATOR = 0, GV_NODE_DATA = 1, GV_NODE_QUERY = 2 } GV_NodeRole;
typedef enum { GV_NODE_JOINING = 0, GV_NODE_ACTIVE = 1, GV_NODE_LEAVING = 2, GV_NODE_DEAD = 3 } GV_NodeState;

typedef struct {
    char *node_id;
    char *address;
    GV_NodeRole role;
    GV_NodeState state;
    uint32_t *shard_ids;
    size_t shard_count;
    uint64_t last_heartbeat;
    double load;
} GV_NodeInfo;

typedef struct {
    const char *node_id;
    const char *listen_address;
    const char *seed_nodes;
    GV_NodeRole role;
    uint32_t heartbeat_interval_ms;
    uint32_t failure_timeout_ms;
} GV_ClusterConfig;

typedef struct {
    size_t total_nodes;
    size_t active_nodes;
    size_t total_shards;
    uint64_t total_vectors;
    double avg_load;
} GV_ClusterStats;

typedef struct GV_Cluster GV_Cluster;

void gv_cluster_config_init(GV_ClusterConfig *config);
GV_Cluster *gv_cluster_create(const GV_ClusterConfig *config);
void gv_cluster_destroy(GV_Cluster *cluster);
int gv_cluster_start(GV_Cluster *cluster);
int gv_cluster_stop(GV_Cluster *cluster);
int gv_cluster_get_local_node(GV_Cluster *cluster, GV_NodeInfo *info);
int gv_cluster_get_node(GV_Cluster *cluster, const char *node_id, GV_NodeInfo *info);
int gv_cluster_list_nodes(GV_Cluster *cluster, GV_NodeInfo **nodes, size_t *count);
void gv_cluster_free_node_info(GV_NodeInfo *info);
void gv_cluster_free_node_list(GV_NodeInfo *nodes, size_t count);
int gv_cluster_get_stats(GV_Cluster *cluster, GV_ClusterStats *stats);
GV_ShardManager *gv_cluster_get_shard_manager(GV_Cluster *cluster);
int gv_cluster_is_healthy(GV_Cluster *cluster);
int gv_cluster_wait_ready(GV_Cluster *cluster, uint32_t timeout_ms);

// ============================================================================
// Namespace / Multi-tenancy
// ============================================================================
typedef enum { GV_NS_INDEX_KDTREE = 0, GV_NS_INDEX_HNSW = 1, GV_NS_INDEX_IVFPQ = 2, GV_NS_INDEX_SPARSE = 3 } GV_NSIndexType;

typedef struct {
    const char *name;
    size_t dimension;
    GV_NSIndexType index_type;
    size_t max_vectors;
    size_t max_memory_bytes;
} GV_NamespaceConfig;

typedef struct {
    char *name;
    size_t dimension;
    GV_NSIndexType index_type;
    size_t vector_count;
    size_t memory_bytes;
    uint64_t created_at;
    uint64_t last_modified;
} GV_NamespaceInfo;

typedef struct GV_Namespace GV_Namespace;
typedef struct GV_NamespaceManager GV_NamespaceManager;

void gv_namespace_config_init(GV_NamespaceConfig *config);
GV_NamespaceManager *gv_namespace_manager_create(const char *base_path);
void gv_namespace_manager_destroy(GV_NamespaceManager *mgr);
GV_Namespace *gv_namespace_create(GV_NamespaceManager *mgr, const GV_NamespaceConfig *config);
GV_Namespace *gv_namespace_get(GV_NamespaceManager *mgr, const char *name);
int gv_namespace_delete(GV_NamespaceManager *mgr, const char *name);
int gv_namespace_list(GV_NamespaceManager *mgr, char ***names, size_t *count);
int gv_namespace_get_info(const GV_Namespace *ns, GV_NamespaceInfo *info);
void gv_namespace_free_info(GV_NamespaceInfo *info);
int gv_namespace_exists(GV_NamespaceManager *mgr, const char *name);
int gv_namespace_add_vector(GV_Namespace *ns, const float *data, size_t dimension);
int gv_namespace_add_vector_with_metadata(GV_Namespace *ns, const float *data, size_t dimension, const char *const *keys, const char *const *values, size_t meta_count);
int gv_namespace_search(const GV_Namespace *ns, const float *query, size_t k, GV_SearchResult *results, GV_DistanceType distance_type);
int gv_namespace_search_filtered(const GV_Namespace *ns, const float *query, size_t k, GV_SearchResult *results, GV_DistanceType distance_type, const char *filter_key, const char *filter_value);
int gv_namespace_delete_vector(GV_Namespace *ns, size_t vector_index);
size_t gv_namespace_count(const GV_Namespace *ns);
int gv_namespace_save(GV_Namespace *ns);
int gv_namespace_manager_save_all(GV_NamespaceManager *mgr);
int gv_namespace_manager_load_all(GV_NamespaceManager *mgr);
GV_Database *gv_namespace_get_db(GV_Namespace *ns);

// ============================================================================
// TTL (Time-to-Live)
// ============================================================================
typedef struct {
    uint64_t default_ttl_seconds;
    uint64_t cleanup_interval_seconds;
    int lazy_expiration;
    size_t max_expired_per_cleanup;
} GV_TTLConfig;

typedef struct {
    uint64_t total_vectors_with_ttl;
    uint64_t total_expired;
    uint64_t next_expiration_time;
    uint64_t last_cleanup_time;
} GV_TTLStats;

typedef struct GV_TTLManager GV_TTLManager;

void gv_ttl_config_init(GV_TTLConfig *config);
GV_TTLManager *gv_ttl_create(const GV_TTLConfig *config);
void gv_ttl_destroy(GV_TTLManager *mgr);
int gv_ttl_set(GV_TTLManager *mgr, size_t vector_index, uint64_t ttl_seconds);
int gv_ttl_set_absolute(GV_TTLManager *mgr, size_t vector_index, uint64_t expire_at_unix);
int gv_ttl_get(const GV_TTLManager *mgr, size_t vector_index, uint64_t *expire_at);
int gv_ttl_remove(GV_TTLManager *mgr, size_t vector_index);
int gv_ttl_is_expired(const GV_TTLManager *mgr, size_t vector_index);
int gv_ttl_get_remaining(const GV_TTLManager *mgr, size_t vector_index, uint64_t *remaining_seconds);
int gv_ttl_cleanup_expired(GV_TTLManager *mgr, GV_Database *db);
int gv_ttl_start_background_cleanup(GV_TTLManager *mgr, GV_Database *db);
void gv_ttl_stop_background_cleanup(GV_TTLManager *mgr);
int gv_ttl_is_background_cleanup_running(const GV_TTLManager *mgr);
int gv_ttl_get_stats(const GV_TTLManager *mgr, GV_TTLStats *stats);
int gv_ttl_set_bulk(GV_TTLManager *mgr, const size_t *indices, size_t count, uint64_t ttl_seconds);
int gv_ttl_get_expiring_before(const GV_TTLManager *mgr, uint64_t before_unix, size_t *indices, size_t max_indices);

// ============================================================================
// BM25 Full-text Search
// ============================================================================
typedef struct {
    int type;
    int lowercase;
    int remove_punctuation;
    const char *stopwords;
    int stem;
    int ngram_min;
    int ngram_max;
} GV_TokenizerConfig;

typedef struct {
    double k1;
    double b;
    GV_TokenizerConfig tokenizer;
} GV_BM25Config;

typedef struct {
    size_t doc_id;
    double score;
} GV_BM25Result;

typedef struct {
    size_t total_documents;
    size_t total_terms;
    size_t total_postings;
    double avg_document_length;
    size_t memory_bytes;
} GV_BM25Stats;

typedef struct GV_BM25Index GV_BM25Index;

void gv_bm25_config_init(GV_BM25Config *config);
GV_BM25Index *gv_bm25_create(const GV_BM25Config *config);
void gv_bm25_destroy(GV_BM25Index *index);
int gv_bm25_add_document(GV_BM25Index *index, size_t doc_id, const char *text);
int gv_bm25_add_document_terms(GV_BM25Index *index, size_t doc_id, const char **terms, size_t term_count);
int gv_bm25_remove_document(GV_BM25Index *index, size_t doc_id);
int gv_bm25_update_document(GV_BM25Index *index, size_t doc_id, const char *text);
int gv_bm25_search(GV_BM25Index *index, const char *query, size_t k, GV_BM25Result *results);
int gv_bm25_search_terms(GV_BM25Index *index, const char **terms, size_t term_count, size_t k, GV_BM25Result *results);
int gv_bm25_score_document(GV_BM25Index *index, size_t doc_id, const char *query, double *score);
int gv_bm25_get_stats(const GV_BM25Index *index, GV_BM25Stats *stats);
size_t gv_bm25_get_doc_freq(const GV_BM25Index *index, const char *term);
int gv_bm25_has_document(const GV_BM25Index *index, size_t doc_id);
int gv_bm25_save(const GV_BM25Index *index, const char *filepath);
GV_BM25Index *gv_bm25_load(const char *filepath);

// ============================================================================
// Hybrid Search
// ============================================================================
typedef enum { GV_FUSION_LINEAR = 0, GV_FUSION_RRF = 1, GV_FUSION_WEIGHTED_RRF = 2 } GV_FusionType;

typedef struct {
    GV_FusionType fusion_type;
    double vector_weight;
    double text_weight;
    double rrf_k;
    GV_DistanceType distance_type;
    size_t prefetch_k;
} GV_HybridConfig;

typedef struct {
    size_t vector_index;
    double combined_score;
    double vector_score;
    double text_score;
    size_t vector_rank;
    size_t text_rank;
} GV_HybridResult;

typedef struct {
    size_t vector_candidates;
    size_t text_candidates;
    size_t unique_candidates;
    double vector_search_time_ms;
    double text_search_time_ms;
    double fusion_time_ms;
    double total_time_ms;
} GV_HybridStats;

typedef struct GV_HybridSearcher GV_HybridSearcher;

void gv_hybrid_config_init(GV_HybridConfig *config);
GV_HybridSearcher *gv_hybrid_create(GV_Database *db, GV_BM25Index *bm25, const GV_HybridConfig *config);
void gv_hybrid_destroy(GV_HybridSearcher *searcher);
int gv_hybrid_search(GV_HybridSearcher *searcher, const float *query_vector, const char *query_text, size_t k, GV_HybridResult *results);
int gv_hybrid_search_with_stats(GV_HybridSearcher *searcher, const float *query_vector, const char *query_text, size_t k, GV_HybridResult *results, GV_HybridStats *stats);
int gv_hybrid_search_vector_only(GV_HybridSearcher *searcher, const float *query_vector, size_t k, GV_HybridResult *results);
int gv_hybrid_search_text_only(GV_HybridSearcher *searcher, const char *query_text, size_t k, GV_HybridResult *results);
int gv_hybrid_set_config(GV_HybridSearcher *searcher, const GV_HybridConfig *config);
int gv_hybrid_get_config(const GV_HybridSearcher *searcher, GV_HybridConfig *config);
int gv_hybrid_set_weights(GV_HybridSearcher *searcher, double vector_weight, double text_weight);
double gv_hybrid_linear_fusion(double vector_score, double text_score, double vector_weight, double text_weight);
double gv_hybrid_rrf_fusion(size_t vector_rank, size_t text_rank, double k);
double gv_hybrid_normalize_score(double score, double min_score, double max_score);

// ============================================================================
// Authentication
// ============================================================================
typedef enum { GV_AUTH_NONE = 0, GV_AUTH_API_KEY = 1, GV_AUTH_JWT = 2 } GV_AuthType;
typedef enum { GV_AUTH_SUCCESS = 0, GV_AUTH_INVALID_KEY = 1, GV_AUTH_EXPIRED = 2, GV_AUTH_INVALID_SIGNATURE = 3, GV_AUTH_INVALID_FORMAT = 4, GV_AUTH_MISSING = 5 } GV_AuthResult;

typedef struct {
    char *key_id;
    char *key_hash;
    char *description;
    uint64_t created_at;
    uint64_t expires_at;
    int enabled;
} GV_APIKey;

typedef struct {
    const char *secret;
    size_t secret_len;
    const char *issuer;
    const char *audience;
    uint64_t clock_skew_seconds;
} GV_JWTConfig;

typedef struct {
    GV_AuthType type;
    GV_JWTConfig jwt;
} GV_AuthConfig;

typedef struct {
    char *subject;
    char *key_id;
    uint64_t auth_time;
    uint64_t expires_at;
    void *claims;
} GV_Identity;

typedef struct GV_AuthManager GV_AuthManager;

void gv_auth_config_init(GV_AuthConfig *config);
GV_AuthManager *gv_auth_create(const GV_AuthConfig *config);
void gv_auth_destroy(GV_AuthManager *auth);
int gv_auth_generate_api_key(GV_AuthManager *auth, const char *description, uint64_t expires_at, char *key_out, char *key_id_out);
int gv_auth_add_api_key(GV_AuthManager *auth, const char *key_id, const char *key_hash, const char *description, uint64_t expires_at);
int gv_auth_revoke_api_key(GV_AuthManager *auth, const char *key_id);
int gv_auth_list_api_keys(GV_AuthManager *auth, GV_APIKey **keys, size_t *count);
void gv_auth_free_api_keys(GV_APIKey *keys, size_t count);
GV_AuthResult gv_auth_verify_api_key(GV_AuthManager *auth, const char *api_key, GV_Identity *identity);
GV_AuthResult gv_auth_verify_jwt(GV_AuthManager *auth, const char *token, GV_Identity *identity);
GV_AuthResult gv_auth_authenticate(GV_AuthManager *auth, const char *credential, GV_Identity *identity);
void gv_auth_free_identity(GV_Identity *identity);
int gv_auth_generate_jwt(GV_AuthManager *auth, const char *subject, uint64_t expires_in, char *token_out, size_t token_size);
const char *gv_auth_result_string(GV_AuthResult result);
int gv_auth_sha256(const void *data, size_t len, unsigned char *hash_out);
void gv_auth_to_hex(const unsigned char *hash, size_t hash_len, char *hex_out);

// ============================================================================
// Multi-Vector / Chunked Documents
// ============================================================================
typedef enum { GV_DOC_AGG_MAX_SIM = 0, GV_DOC_AGG_AVG_SIM = 1, GV_DOC_AGG_SUM_SIM = 2 } GV_DocAggregation;

typedef struct {
    size_t max_chunks_per_doc;
    GV_DocAggregation aggregation;
} GV_MultiVecConfig;

typedef struct {
    uint64_t doc_id;
    float score;
    size_t num_chunks;
    size_t best_chunk_index;
} GV_DocSearchResult;

void *gv_multivec_create(size_t dimension, const GV_MultiVecConfig *config);
void gv_multivec_destroy(void *index);
int gv_multivec_add_document(void *index, uint64_t doc_id, const float *chunks, size_t num_chunks, size_t dimension);
int gv_multivec_delete_document(void *index, uint64_t doc_id);
int gv_multivec_search(void *index, const float *query, size_t k, GV_DocSearchResult *results, int distance_type);
size_t gv_multivec_count_documents(const void *index);
size_t gv_multivec_count_chunks(const void *index);

// ============================================================================
// Point-in-Time Snapshots
// ============================================================================
typedef struct {
    uint64_t snapshot_id;
    uint64_t timestamp_us;
    size_t vector_count;
    char label[64];
} GV_SnapshotInfo;

typedef struct GV_SnapshotManager GV_SnapshotManager;
typedef struct GV_Snapshot GV_Snapshot;

GV_SnapshotManager *gv_snapshot_manager_create(size_t max_snapshots);
void gv_snapshot_manager_destroy(GV_SnapshotManager *mgr);
uint64_t gv_snapshot_create(GV_SnapshotManager *mgr, size_t vector_count, const float *vector_data, size_t dimension, const char *label);
GV_Snapshot *gv_snapshot_open(GV_SnapshotManager *mgr, uint64_t snapshot_id);
void gv_snapshot_close(GV_Snapshot *snap);
size_t gv_snapshot_count(const GV_Snapshot *snap);
const float *gv_snapshot_get_vector(const GV_Snapshot *snap, size_t index);
size_t gv_snapshot_dimension(const GV_Snapshot *snap);
int gv_snapshot_list(const GV_SnapshotManager *mgr, GV_SnapshotInfo *infos, size_t max_infos);
int gv_snapshot_delete(GV_SnapshotManager *mgr, uint64_t snapshot_id);

// ============================================================================
// MVCC Transactions
// ============================================================================
typedef enum { GV_TXN_ACTIVE = 0, GV_TXN_COMMITTED = 1, GV_TXN_ABORTED = 2 } GV_TxnStatus;

typedef struct GV_MVCCManager GV_MVCCManager;
typedef struct GV_Transaction GV_Transaction;

typedef struct {
    size_t vector_index;
    uint64_t create_txn;
    uint64_t delete_txn;
    float *data;
    size_t dimension;
} GV_MVCCVersion;

GV_MVCCManager *gv_mvcc_create(size_t dimension);
void gv_mvcc_destroy(GV_MVCCManager *mgr);
GV_Transaction *gv_txn_begin(GV_MVCCManager *mgr);
int gv_txn_commit(GV_Transaction *txn);
int gv_txn_rollback(GV_Transaction *txn);
uint64_t gv_txn_id(const GV_Transaction *txn);
GV_TxnStatus gv_txn_status(const GV_Transaction *txn);
int gv_txn_add_vector(GV_Transaction *txn, const float *data, size_t dimension);
int gv_txn_delete_vector(GV_Transaction *txn, size_t vector_index);
int gv_txn_get_vector(const GV_Transaction *txn, size_t vector_index, float *out);
size_t gv_txn_count(const GV_Transaction *txn);
int gv_mvcc_gc(GV_MVCCManager *mgr);
size_t gv_mvcc_version_count(const GV_MVCCManager *mgr);
size_t gv_mvcc_active_txn_count(const GV_MVCCManager *mgr);

// ============================================================================
// Query Optimizer
// ============================================================================
typedef enum { GV_PLAN_EXACT_SCAN = 0, GV_PLAN_INDEX_SEARCH = 1, GV_PLAN_OVERSAMPLE_FILTER = 2 } GV_PlanStrategy;

typedef struct {
    GV_PlanStrategy strategy;
    size_t ef_search;
    size_t nprobe;
    size_t rerank_top;
    double estimated_cost;
    double estimated_recall;
    int use_metadata_index;
    size_t oversample_k;
    char explanation[256];
} GV_QueryPlan;

typedef struct {
    size_t total_vectors;
    size_t dimension;
    int index_type;
    double deleted_ratio;
    double avg_vectors_per_filter_match;
    size_t last_search_latency_us;
} GV_CollectionStats;

typedef struct GV_QueryOptimizer GV_QueryOptimizer;

GV_QueryOptimizer *gv_optimizer_create(void);
void gv_optimizer_destroy(GV_QueryOptimizer *opt);
void gv_optimizer_update_stats(GV_QueryOptimizer *opt, const GV_CollectionStats *stats);
int gv_optimizer_plan(const GV_QueryOptimizer *opt, size_t k, int has_filter, double filter_selectivity, GV_QueryPlan *plan);
void gv_optimizer_record_result(GV_QueryOptimizer *opt, const GV_QueryPlan *plan, uint64_t actual_latency_us, double actual_recall);
size_t gv_optimizer_recommend_ef_search(const GV_QueryOptimizer *opt, size_t k);
size_t gv_optimizer_recommend_nprobe(const GV_QueryOptimizer *opt, size_t k);

// ============================================================================
// Payload Indexing
// ============================================================================
typedef enum { GV_FIELD_INT = 0, GV_FIELD_FLOAT = 1, GV_FIELD_STRING = 2, GV_FIELD_BOOL = 3 } GV_FieldType;
typedef enum { GV_PAYLOAD_OP_EQ = 0, GV_PAYLOAD_OP_NE = 1, GV_PAYLOAD_OP_GT = 2, GV_PAYLOAD_OP_GE = 3, GV_PAYLOAD_OP_LT = 4, GV_PAYLOAD_OP_LE = 5, GV_PAYLOAD_OP_CONTAINS = 6, GV_PAYLOAD_OP_PREFIX = 7 } GV_PayloadOp;

typedef struct {
    const char *field_name;
    GV_PayloadOp op;
    int64_t int_val;
    double float_val;
    const char *string_val;
    int bool_val;
    GV_FieldType field_type;
} GV_PayloadQuery;

typedef struct GV_PayloadIndex GV_PayloadIndex;

GV_PayloadIndex *gv_payload_index_create(void);
void gv_payload_index_destroy(GV_PayloadIndex *idx);
int gv_payload_index_add_field(GV_PayloadIndex *idx, const char *name, GV_FieldType type);
int gv_payload_index_remove_field(GV_PayloadIndex *idx, const char *name);
int gv_payload_index_field_count(const GV_PayloadIndex *idx);
int gv_payload_index_insert_int(GV_PayloadIndex *idx, size_t vector_id, const char *field, int64_t value);
int gv_payload_index_insert_float(GV_PayloadIndex *idx, size_t vector_id, const char *field, double value);
int gv_payload_index_insert_string(GV_PayloadIndex *idx, size_t vector_id, const char *field, const char *value);
int gv_payload_index_insert_bool(GV_PayloadIndex *idx, size_t vector_id, const char *field, int value);
int gv_payload_index_remove(GV_PayloadIndex *idx, size_t vector_id);
int gv_payload_index_query(const GV_PayloadIndex *idx, const GV_PayloadQuery *query, size_t *result_ids, size_t max_results);
int gv_payload_index_query_multi(const GV_PayloadIndex *idx, const GV_PayloadQuery *queries, size_t query_count, size_t *result_ids, size_t max_results);
size_t gv_payload_index_total_entries(const GV_PayloadIndex *idx);

// ============================================================================
// Vector Deduplication
// ============================================================================
typedef struct {
    float epsilon;
    size_t num_hash_tables;
    size_t hash_bits;
    uint64_t seed;
} GV_DedupConfig;

typedef struct {
    size_t original_index;
    size_t duplicate_index;
    float distance;
} GV_DedupResult;

typedef struct GV_DedupIndex GV_DedupIndex;

GV_DedupIndex *gv_dedup_create(size_t dimension, const GV_DedupConfig *config);
void gv_dedup_destroy(GV_DedupIndex *dedup);
int gv_dedup_check(GV_DedupIndex *dedup, const float *data, size_t dimension);
int gv_dedup_insert(GV_DedupIndex *dedup, const float *data, size_t dimension);
int gv_dedup_scan(GV_DedupIndex *dedup, GV_DedupResult *results, size_t max_results);
size_t gv_dedup_count(const GV_DedupIndex *dedup);
void gv_dedup_clear(GV_DedupIndex *dedup);

// ============================================================================
// Auto Index Migration
// ============================================================================
typedef enum { GV_MIGRATION_PENDING = 0, GV_MIGRATION_RUNNING = 1, GV_MIGRATION_COMPLETED = 2, GV_MIGRATION_FAILED = 3, GV_MIGRATION_CANCELLED = 4 } GV_MigrationStatus;

typedef struct {
    GV_MigrationStatus status;
    double progress;
    size_t vectors_migrated;
    size_t total_vectors;
    uint64_t start_time_us;
    uint64_t elapsed_us;
    char error_message[256];
} GV_MigrationInfo;

typedef struct GV_Migration GV_Migration;

GV_Migration *gv_migration_start(const float *source_data, size_t count, size_t dimension, int new_index_type, const void *new_index_config);
int gv_migration_get_info(const GV_Migration *mig, GV_MigrationInfo *info);
int gv_migration_wait(GV_Migration *mig);
int gv_migration_cancel(GV_Migration *mig);
void *gv_migration_take_index(GV_Migration *mig);
void gv_migration_destroy(GV_Migration *mig);

// ============================================================================
// Collection Versioning
// ============================================================================
typedef struct {
    uint64_t version_id;
    uint64_t timestamp_us;
    size_t vector_count;
    size_t dimension;
    char label[128];
    size_t data_size_bytes;
} GV_VersionInfo;

typedef struct GV_VersionManager GV_VersionManager;

GV_VersionManager *gv_version_manager_create(size_t max_versions);
void gv_version_manager_destroy(GV_VersionManager *mgr);
uint64_t gv_version_create(GV_VersionManager *mgr, const float *data, size_t count, size_t dimension, const char *label);
int gv_version_list(const GV_VersionManager *mgr, GV_VersionInfo *infos, size_t max_infos);
int gv_version_count(const GV_VersionManager *mgr);
int gv_version_get_info(const GV_VersionManager *mgr, uint64_t version_id, GV_VersionInfo *info);
float *gv_version_get_data(const GV_VersionManager *mgr, uint64_t version_id, size_t *count_out, size_t *dimension_out);
int gv_version_delete(GV_VersionManager *mgr, uint64_t version_id);
int gv_version_compare(const GV_VersionManager *mgr, uint64_t v1, uint64_t v2, size_t *added, size_t *removed, size_t *modified);

// ============================================================================
// Read Replica Load Balancing
// ============================================================================
typedef enum { GV_READ_LEADER_ONLY = 0, GV_READ_ROUND_ROBIN = 1, GV_READ_LEAST_LAG = 2, GV_READ_RANDOM = 3 } GV_ReadPolicy;

int gv_replication_set_read_policy(GV_ReplicationManager *mgr, GV_ReadPolicy policy);
GV_ReadPolicy gv_replication_get_read_policy(GV_ReplicationManager *mgr);
GV_Database *gv_replication_route_read(GV_ReplicationManager *mgr);
int gv_replication_set_max_read_lag(GV_ReplicationManager *mgr, uint64_t max_lag);
int gv_replication_register_follower_db(GV_ReplicationManager *mgr, const char *node_id, GV_Database *db);

// ============================================================================
// Bloom Filter
// ============================================================================
typedef struct GV_BloomFilter GV_BloomFilter;

GV_BloomFilter *gv_bloom_create(size_t expected_items, double fp_rate);
void gv_bloom_destroy(GV_BloomFilter *bf);
int gv_bloom_add(GV_BloomFilter *bf, const void *data, size_t len);
int gv_bloom_add_string(GV_BloomFilter *bf, const char *str);
int gv_bloom_check(const GV_BloomFilter *bf, const void *data, size_t len);
int gv_bloom_check_string(const GV_BloomFilter *bf, const char *str);
size_t gv_bloom_count(const GV_BloomFilter *bf);
double gv_bloom_fp_rate(const GV_BloomFilter *bf);
void gv_bloom_clear(GV_BloomFilter *bf);

// ============================================================================
// Query Tracing
// ============================================================================
typedef struct {
    const char *name;
    uint64_t start_us;
    uint64_t duration_us;
    const char *metadata;
} GV_TraceSpan;

typedef struct {
    uint64_t trace_id;
    uint64_t total_duration_us;
    GV_TraceSpan *spans;
    size_t span_count;
    size_t span_capacity;
    int active;
} GV_QueryTrace;

GV_QueryTrace *gv_trace_begin(void);
void gv_trace_end(GV_QueryTrace *trace);
void gv_trace_destroy(GV_QueryTrace *trace);
void gv_trace_span_start(GV_QueryTrace *trace, const char *name);
void gv_trace_span_end(GV_QueryTrace *trace);
void gv_trace_span_add(GV_QueryTrace *trace, const char *name, uint64_t duration_us);
void gv_trace_set_metadata(GV_QueryTrace *trace, const char *metadata);
char *gv_trace_to_json(const GV_QueryTrace *trace);
uint64_t gv_trace_get_time_us(void);

// ============================================================================
// Client-Side Caching
// ============================================================================
typedef enum { GV_CACHE_LRU = 0, GV_CACHE_LFU = 1 } GV_CachePolicy;

typedef struct {
    size_t max_entries;
    size_t max_memory_bytes;
    uint32_t ttl_seconds;
    uint64_t invalidate_after_mutations;
    GV_CachePolicy policy;
} GV_CacheConfig;

typedef struct {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t invalidations;
    size_t current_entries;
    size_t current_memory;
    double hit_rate;
} GV_CacheStats;

typedef struct {
    size_t *indices;
    float *distances;
    size_t count;
} GV_CachedResult;

typedef struct GV_Cache GV_Cache;

void gv_cache_config_init(GV_CacheConfig *config);
GV_Cache *gv_cache_create(const GV_CacheConfig *config);
void gv_cache_destroy(GV_Cache *cache);
int gv_cache_lookup(GV_Cache *cache, const float *query_data, size_t dimension, size_t k, int distance_type, GV_CachedResult *result);
int gv_cache_store(GV_Cache *cache, const float *query_data, size_t dimension, size_t k, int distance_type, const size_t *indices, const float *distances, size_t count);
void gv_cache_notify_mutation(GV_Cache *cache);
void gv_cache_invalidate_all(GV_Cache *cache);
void gv_cache_free_result(GV_CachedResult *result);
int gv_cache_get_stats(const GV_Cache *cache, GV_CacheStats *stats);
void gv_cache_reset_stats(GV_Cache *cache);

// ============================================================================
// Schema Evolution
// ============================================================================
typedef enum { GV_SCHEMA_STRING = 0, GV_SCHEMA_INT = 1, GV_SCHEMA_FLOAT = 2, GV_SCHEMA_BOOL = 3 } GV_SchemaFieldType;

typedef struct {
    char name[64];
    GV_SchemaFieldType type;
    int required;
    char default_value[256];
} GV_SchemaField;

typedef struct {
    uint32_t version;
    GV_SchemaField *fields;
    size_t field_count;
    size_t field_capacity;
} GV_Schema;

typedef struct {
    char name[64];
    int added;
    int removed;
    int type_changed;
    GV_SchemaFieldType old_type;
    GV_SchemaFieldType new_type;
} GV_SchemaDiff;

GV_Schema *gv_schema_create(uint32_t version);
void gv_schema_destroy(GV_Schema *schema);
GV_Schema *gv_schema_copy(const GV_Schema *schema);
int gv_schema_add_field(GV_Schema *schema, const char *name, GV_SchemaFieldType type, int required, const char *default_value);
int gv_schema_remove_field(GV_Schema *schema, const char *name);
int gv_schema_has_field(const GV_Schema *schema, const char *name);
const GV_SchemaField *gv_schema_get_field(const GV_Schema *schema, const char *name);
size_t gv_schema_field_count(const GV_Schema *schema);
int gv_schema_validate(const GV_Schema *schema, const char *const *keys, const char *const *values, size_t count);
int gv_schema_diff(const GV_Schema *old_schema, const GV_Schema *new_schema, GV_SchemaDiff *diffs, size_t max_diffs);
int gv_schema_is_compatible(const GV_Schema *old_schema, const GV_Schema *new_schema);
char *gv_schema_to_json(const GV_Schema *schema);

// ============================================================================
// Codebook Sharing
// ============================================================================
typedef struct {
    size_t dimension;
    size_t m;
    size_t ksub;
    uint8_t nbits;
    size_t dsub;
    float *centroids;
    int trained;
} GV_Codebook;

GV_Codebook *gv_codebook_create(size_t dimension, size_t m, uint8_t nbits);
void gv_codebook_destroy(GV_Codebook *cb);
int gv_codebook_train(GV_Codebook *cb, const float *data, size_t count, size_t train_iters);
int gv_codebook_encode(const GV_Codebook *cb, const float *vector, uint8_t *codes);
int gv_codebook_decode(const GV_Codebook *cb, const uint8_t *codes, float *output);
float gv_codebook_distance_adc(const GV_Codebook *cb, const float *query, const uint8_t *codes);
int gv_codebook_save(const GV_Codebook *cb, const char *filepath);
GV_Codebook *gv_codebook_load(const char *filepath);
GV_Codebook *gv_codebook_copy(const GV_Codebook *cb);

/* --- Point IDs --- */

typedef struct GV_PointIDMap GV_PointIDMap;

GV_PointIDMap *gv_point_id_create(size_t initial_capacity);
void gv_point_id_destroy(GV_PointIDMap *map);

int gv_point_id_set(GV_PointIDMap *map, const char *string_id, size_t internal_index);
int gv_point_id_get(const GV_PointIDMap *map, const char *string_id, size_t *out_index);
int gv_point_id_remove(GV_PointIDMap *map, const char *string_id);
int gv_point_id_has(const GV_PointIDMap *map, const char *string_id);

const char *gv_point_id_reverse_lookup(const GV_PointIDMap *map, size_t internal_index);

int gv_point_id_generate_uuid(char *buf, size_t buf_size);

size_t gv_point_id_count(const GV_PointIDMap *map);

int gv_point_id_iterate(const GV_PointIDMap *map, void *callback, void *ctx);

int gv_point_id_save(const GV_PointIDMap *map, const char *filepath);
GV_PointIDMap *gv_point_id_load(const char *filepath);

/* --- TLS/HTTPS --- */

typedef enum {
    GV_TLS_1_2 = 0,
    GV_TLS_1_3 = 1
} GV_TLSVersion;

typedef struct {
    const char *cert_file;
    const char *key_file;
    const char *ca_file;
    GV_TLSVersion min_version;
    const char *cipher_list;
    int verify_client;
} GV_TLSConfig;

typedef struct GV_TLSContext GV_TLSContext;

void gv_tls_config_init(GV_TLSConfig *config);
GV_TLSContext *gv_tls_create(const GV_TLSConfig *config);
void gv_tls_destroy(GV_TLSContext *ctx);

int gv_tls_is_available(void);
const char *gv_tls_version_string(const GV_TLSContext *ctx);

int gv_tls_accept(GV_TLSContext *ctx, int client_fd, void **tls_conn);
int gv_tls_read(void *tls_conn, void *buf, size_t len);
int gv_tls_write(void *tls_conn, const void *buf, size_t len);
void gv_tls_close_conn(void *tls_conn);

int gv_tls_get_peer_cn(void *tls_conn, char *buf, size_t buf_size);
int gv_tls_cert_days_remaining(const GV_TLSContext *ctx);

/* --- Score Threshold --- */

typedef struct {
    size_t index;
    float distance;
} GV_ThresholdResult;

int gv_db_search_with_threshold(const void *db, const float *query_data, size_t k,
                                 int distance_type, float score_threshold,
                                 GV_ThresholdResult *results);

size_t gv_threshold_filter(GV_ThresholdResult *results, size_t count,
                            float threshold, int distance_type);

int gv_threshold_passes(float distance, float threshold, int distance_type);

/* --- Named Vectors --- */

typedef struct GV_NamedVectorStore GV_NamedVectorStore;

typedef struct {
    const char *name;
    size_t dimension;
    int distance_type;
} GV_VectorFieldConfig;

typedef struct {
    const char *field_name;
    const float *data;
    size_t dimension;
} GV_NamedVector;

typedef struct {
    size_t point_index;
    float distance;
    const char *field_name;
} GV_NamedSearchResult;

GV_NamedVectorStore *gv_named_vectors_create(void);
void gv_named_vectors_destroy(GV_NamedVectorStore *store);

int gv_named_vectors_add_field(GV_NamedVectorStore *store, const GV_VectorFieldConfig *config);
int gv_named_vectors_remove_field(GV_NamedVectorStore *store, const char *name);
size_t gv_named_vectors_field_count(const GV_NamedVectorStore *store);
int gv_named_vectors_get_field(const GV_NamedVectorStore *store, const char *name, GV_VectorFieldConfig *out);

int gv_named_vectors_insert(GV_NamedVectorStore *store, size_t point_id,
                             const GV_NamedVector *vectors, size_t vector_count);
int gv_named_vectors_update(GV_NamedVectorStore *store, size_t point_id,
                             const GV_NamedVector *vectors, size_t vector_count);
int gv_named_vectors_delete(GV_NamedVectorStore *store, size_t point_id);

int gv_named_vectors_search(const GV_NamedVectorStore *store, const char *field_name,
                             const float *query, size_t k, GV_NamedSearchResult *results);

const float *gv_named_vectors_get(const GV_NamedVectorStore *store, size_t point_id, const char *field_name);

size_t gv_named_vectors_count(const GV_NamedVectorStore *store);

int gv_named_vectors_save(const GV_NamedVectorStore *store, const char *filepath);
GV_NamedVectorStore *gv_named_vectors_load(const char *filepath);

/* --- Filter Ops --- */

int gv_db_delete_by_filter(GV_Database *db, const char *filter_expr);

int gv_db_update_by_filter(GV_Database *db, const char *filter_expr,
                            const float *new_data, size_t dimension);

int gv_db_update_metadata_by_filter(GV_Database *db, const char *filter_expr,
                                     const char **metadata_keys,
                                     const char **metadata_values,
                                     size_t metadata_count);

int gv_db_count_by_filter(const GV_Database *db, const char *filter_expr);

int gv_db_find_by_filter(const GV_Database *db, const char *filter_expr,
                          size_t *out_indices, size_t max_count);

/* --- gRPC Binary Protocol --- */

typedef enum {
    GV_GRPC_OK = 0,
    GV_GRPC_ERROR_NULL = -1,
    GV_GRPC_ERROR_CONFIG = -2,
    GV_GRPC_ERROR_RUNNING = -3,
    GV_GRPC_ERROR_NOT_RUNNING = -4,
    GV_GRPC_ERROR_START = -5,
    GV_GRPC_ERROR_MEMORY = -6,
    GV_GRPC_ERROR_BIND = -7
} GV_GrpcError;

typedef enum {
    GV_MSG_ADD_VECTOR = 1,
    GV_MSG_SEARCH = 2,
    GV_MSG_DELETE = 3,
    GV_MSG_UPDATE = 4,
    GV_MSG_GET = 5,
    GV_MSG_BATCH_ADD = 6,
    GV_MSG_BATCH_SEARCH = 7,
    GV_MSG_STATS = 8,
    GV_MSG_HEALTH = 9,
    GV_MSG_SAVE = 10,
    GV_MSG_RESPONSE = 128
} GV_GrpcMsgType;

typedef struct {
    uint16_t port;
    const char *bind_address;
    size_t max_connections;
    size_t max_message_bytes;
    size_t thread_pool_size;
    int enable_compression;
} GV_GrpcConfig;

typedef struct {
    uint32_t length;
    uint8_t msg_type;
    uint32_t request_id;
    uint8_t *payload;
    size_t payload_len;
} GV_GrpcMessage;

typedef struct {
    uint64_t total_requests;
    uint64_t active_connections;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t errors;
    double avg_latency_us;
} GV_GrpcStats;

typedef struct GV_GrpcServer GV_GrpcServer;

void gv_grpc_config_init(GV_GrpcConfig *config);
GV_GrpcServer *gv_grpc_create(GV_Database *db, const GV_GrpcConfig *config);
int gv_grpc_start(GV_GrpcServer *server);
int gv_grpc_stop(GV_GrpcServer *server);
void gv_grpc_destroy(GV_GrpcServer *server);
int gv_grpc_is_running(const GV_GrpcServer *server);

int gv_grpc_get_stats(const GV_GrpcServer *server, GV_GrpcStats *stats);
const char *gv_grpc_error_string(int error);

int gv_grpc_encode_search_request(const float *query, size_t dimension, size_t k,
                                   int distance_type, uint8_t *buf, size_t buf_size, size_t *out_len);
int gv_grpc_decode_search_request(const uint8_t *buf, size_t len,
                                   float **query, size_t *dimension, size_t *k, int *distance_type);
int gv_grpc_encode_add_request(const float *data, size_t dimension,
                                uint8_t *buf, size_t buf_size, size_t *out_len);

/* --- Auto Embed --- */

typedef enum {
    GV_EMBED_PROVIDER_OPENAI = 0,
    GV_EMBED_PROVIDER_GOOGLE = 1,
    GV_EMBED_PROVIDER_HUGGINGFACE = 2,
    GV_EMBED_PROVIDER_CUSTOM = 3
} GV_AutoEmbedProvider;

typedef struct {
    GV_AutoEmbedProvider provider;
    const char *api_key;
    const char *model_name;
    const char *base_url;
    size_t dimension;
    int cache_embeddings;
    size_t max_cache_entries;
    size_t max_text_length;
    int batch_size;
} GV_AutoEmbedConfig;

typedef struct GV_AutoEmbedder GV_AutoEmbedder;

typedef struct {
    uint64_t total_embeddings;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t api_calls;
    uint64_t api_errors;
    double avg_latency_ms;
} GV_AutoEmbedStats;

void gv_auto_embed_config_init(GV_AutoEmbedConfig *config);
GV_AutoEmbedder *gv_auto_embed_create(const GV_AutoEmbedConfig *config);
void gv_auto_embed_destroy(GV_AutoEmbedder *embedder);

int gv_auto_embed_add_text(GV_AutoEmbedder *embedder, GV_Database *db,
                            const char *text, const char *metadata_key, const char *metadata_value);

int gv_auto_embed_search_text(GV_AutoEmbedder *embedder, const GV_Database *db,
                               const char *text, size_t k, int distance_type,
                               size_t *out_indices, float *out_distances, size_t *out_count);

int gv_auto_embed_add_texts(GV_AutoEmbedder *embedder, GV_Database *db,
                             const char **texts, size_t count,
                             const char **metadata_keys, const char **metadata_values);

float *gv_auto_embed_text(GV_AutoEmbedder *embedder, const char *text, size_t *out_dimension);

int gv_auto_embed_get_stats(const GV_AutoEmbedder *embedder, GV_AutoEmbedStats *stats);
void gv_auto_embed_clear_cache(GV_AutoEmbedder *embedder);

/* --- DiskANN On-Disk Index --- */

typedef struct {
    size_t max_degree;
    float alpha;
    size_t build_beam_width;
    size_t search_beam_width;
    size_t pq_dim;
    const char *data_path;
    size_t cache_size_mb;
    size_t sector_size;
} GV_DiskANNConfig;

typedef struct GV_DiskANNIndex GV_DiskANNIndex;

typedef struct {
    size_t index;
    float distance;
} GV_DiskANNResult;

typedef struct {
    size_t total_vectors;
    size_t graph_edges;
    size_t cache_hits;
    size_t cache_misses;
    size_t disk_reads;
    double avg_search_latency_us;
    size_t memory_usage_bytes;
    size_t disk_usage_bytes;
} GV_DiskANNStats;

void gv_diskann_config_init(GV_DiskANNConfig *config);
GV_DiskANNIndex *gv_diskann_create(size_t dimension, const GV_DiskANNConfig *config);
void gv_diskann_destroy(GV_DiskANNIndex *index);

int gv_diskann_build(GV_DiskANNIndex *index, const float *data, size_t count, size_t dimension);
int gv_diskann_insert(GV_DiskANNIndex *index, const float *data, size_t dimension);

int gv_diskann_search(const GV_DiskANNIndex *index, const float *query, size_t dimension,
                       size_t k, GV_DiskANNResult *results);

int gv_diskann_delete(GV_DiskANNIndex *index, size_t vector_index);

int gv_diskann_get_stats(const GV_DiskANNIndex *index, GV_DiskANNStats *stats);

int gv_diskann_save(const GV_DiskANNIndex *index, const char *filepath);
GV_DiskANNIndex *gv_diskann_load(const char *filepath, const GV_DiskANNConfig *config);

size_t gv_diskann_count(const GV_DiskANNIndex *index);

/* --- Group Search --- */

typedef struct {
    size_t index;
    float distance;
} GV_GroupHit;

typedef struct {
    char *group_value;
    GV_GroupHit *hits;
    size_t hit_count;
} GV_SearchGroup;

typedef struct {
    GV_SearchGroup *groups;
    size_t group_count;
    size_t total_hits;
} GV_GroupedResult;

typedef struct {
    const char *group_by;
    size_t group_limit;
    size_t hits_per_group;
    int distance_type;
    size_t oversample;
} GV_GroupSearchConfig;

void gv_group_search_config_init(GV_GroupSearchConfig *config);

int gv_group_search(const GV_Database *db, const float *query, size_t dimension,
                     const GV_GroupSearchConfig *config, GV_GroupedResult *result);

void gv_group_search_free_result(GV_GroupedResult *result);

/* --- Geo-Spatial Filtering --- */

typedef struct { double lat; double lng; } GV_GeoPoint;
typedef struct { GV_GeoPoint min; GV_GeoPoint max; } GV_GeoBBox;

typedef struct GV_GeoIndex GV_GeoIndex;

typedef struct {
    size_t point_index;
    double lat;
    double lng;
    double distance_km;
} GV_GeoResult;

GV_GeoIndex *gv_geo_create(void);
void gv_geo_destroy(GV_GeoIndex *index);

int gv_geo_insert(GV_GeoIndex *index, size_t point_index, double lat, double lng);
int gv_geo_update(GV_GeoIndex *index, size_t point_index, double lat, double lng);
int gv_geo_remove(GV_GeoIndex *index, size_t point_index);

int gv_geo_radius_search(const GV_GeoIndex *index, double lat, double lng,
                          double radius_km, GV_GeoResult *results, size_t max_results);

int gv_geo_bbox_search(const GV_GeoIndex *index, const GV_GeoBBox *bbox,
                        GV_GeoResult *results, size_t max_results);

int gv_geo_get_candidates(const GV_GeoIndex *index, double lat, double lng,
                           double radius_km, size_t *out_indices, size_t max_count);

double gv_geo_distance_km(double lat1, double lng1, double lat2, double lng2);

size_t gv_geo_count(const GV_GeoIndex *index);

int gv_geo_save(const GV_GeoIndex *index, const char *filepath);
GV_GeoIndex *gv_geo_load(const char *filepath);

/* --- ColBERT Late Interaction (MaxSim) --- */

typedef struct GV_LateInteractionIndex GV_LateInteractionIndex;

typedef struct {
    size_t token_dimension;
    size_t max_doc_tokens;
    size_t max_query_tokens;
    size_t candidate_pool;
} GV_LateInteractionConfig;

typedef struct {
    size_t doc_index;
    float score;
} GV_LateInteractionResult;

typedef struct {
    size_t total_documents;
    size_t total_tokens_stored;
    size_t memory_bytes;
} GV_LateInteractionStats;

void gv_late_interaction_config_init(GV_LateInteractionConfig *config);
GV_LateInteractionIndex *gv_late_interaction_create(const GV_LateInteractionConfig *config);
void gv_late_interaction_destroy(GV_LateInteractionIndex *index);

int gv_late_interaction_add_doc(GV_LateInteractionIndex *index,
                                 const float *token_embeddings, size_t num_tokens);

int gv_late_interaction_search(const GV_LateInteractionIndex *index,
                                const float *query_tokens, size_t num_query_tokens,
                                size_t k, GV_LateInteractionResult *results);

int gv_late_interaction_delete(GV_LateInteractionIndex *index, size_t doc_index);

int gv_late_interaction_get_stats(const GV_LateInteractionIndex *index, GV_LateInteractionStats *stats);
size_t gv_late_interaction_count(const GV_LateInteractionIndex *index);

int gv_late_interaction_save(const GV_LateInteractionIndex *index, const char *filepath);
GV_LateInteractionIndex *gv_late_interaction_load(const char *filepath);

/* --- Recommendation API --- */

typedef struct {
    float positive_weight;
    float negative_weight;
    int distance_type;
    size_t oversample;
    int exclude_input;
} GV_RecommendConfig;

typedef struct {
    size_t index;
    float score;
} GV_RecommendResult;

void gv_recommend_config_init(GV_RecommendConfig *config);

int gv_recommend_by_id(const GV_Database *db,
                        const size_t *positive_ids, size_t positive_count,
                        const size_t *negative_ids, size_t negative_count,
                        size_t k, const GV_RecommendConfig *config,
                        GV_RecommendResult *results);

int gv_recommend_by_vector(const GV_Database *db,
                            const float *positive_vectors, size_t positive_count,
                            const float *negative_vectors, size_t negative_count,
                            size_t dimension, size_t k, const GV_RecommendConfig *config,
                            GV_RecommendResult *results);

int gv_recommend_discover(const GV_Database *db,
                           const float *target, const float *context,
                           size_t dimension, size_t k, const GV_RecommendConfig *config,
                           GV_RecommendResult *results);

/* --- Collection Aliases --- */

typedef struct GV_AliasManager GV_AliasManager;

typedef struct {
    char *alias_name;
    char *collection_name;
    uint64_t created_at;
    uint64_t updated_at;
} GV_AliasInfo;

GV_AliasManager *gv_alias_manager_create(void);
void gv_alias_manager_destroy(GV_AliasManager *mgr);

int gv_alias_create(GV_AliasManager *mgr, const char *alias_name, const char *collection_name);
int gv_alias_update(GV_AliasManager *mgr, const char *alias_name, const char *new_collection_name);
int gv_alias_delete(GV_AliasManager *mgr, const char *alias_name);
int gv_alias_exists(const GV_AliasManager *mgr, const char *alias_name);

int gv_alias_swap(GV_AliasManager *mgr, const char *alias_a, const char *alias_b);

const char *gv_alias_resolve(const GV_AliasManager *mgr, const char *alias_name);

int gv_alias_list(const GV_AliasManager *mgr, GV_AliasInfo **out_list, size_t *out_count);
void gv_alias_free_list(GV_AliasInfo *list, size_t count);

int gv_alias_get_info(const GV_AliasManager *mgr, const char *alias_name, GV_AliasInfo *info);

size_t gv_alias_count(const GV_AliasManager *mgr);

int gv_alias_save(const GV_AliasManager *mgr, const char *filepath);
GV_AliasManager *gv_alias_load(const char *filepath);

/* --- Async Vacuum --- */

typedef enum {
    GV_VACUUM_IDLE = 0,
    GV_VACUUM_RUNNING = 1,
    GV_VACUUM_COMPLETED = 2,
    GV_VACUUM_FAILED = 3
} GV_VacuumState;

typedef struct {
    size_t min_deleted_count;
    double min_fragmentation_ratio;
    size_t batch_size;
    int priority;
    size_t interval_sec;
} GV_VacuumConfig;

typedef struct {
    GV_VacuumState state;
    size_t vectors_compacted;
    size_t bytes_reclaimed;
    double fragmentation_before;
    double fragmentation_after;
    uint64_t started_at;
    uint64_t completed_at;
    uint64_t duration_ms;
    size_t total_runs;
} GV_VacuumStats;

typedef struct GV_VacuumManager GV_VacuumManager;

void gv_vacuum_config_init(GV_VacuumConfig *config);
GV_VacuumManager *gv_vacuum_create(GV_Database *db, const GV_VacuumConfig *config);
void gv_vacuum_destroy(GV_VacuumManager *mgr);

int gv_vacuum_run(GV_VacuumManager *mgr);

int gv_vacuum_start_auto(GV_VacuumManager *mgr);
int gv_vacuum_stop_auto(GV_VacuumManager *mgr);

double gv_vacuum_get_fragmentation(const GV_VacuumManager *mgr);

int gv_vacuum_get_stats(const GV_VacuumManager *mgr, GV_VacuumStats *stats);
GV_VacuumState gv_vacuum_get_state(const GV_VacuumManager *mgr);

/* --- Consistency Levels --- */

typedef enum {
    GV_CONSISTENCY_STRONG = 0,
    GV_CONSISTENCY_EVENTUAL = 1,
    GV_CONSISTENCY_BOUNDED_STALENESS = 2,
    GV_CONSISTENCY_SESSION = 3
} GV_ConsistencyLevel;

typedef struct {
    GV_ConsistencyLevel level;
    uint64_t max_staleness_ms;
    uint64_t session_token;
} GV_ConsistencyConfig;

typedef struct GV_ConsistencyManager GV_ConsistencyManager;

GV_ConsistencyManager *gv_consistency_create(GV_ConsistencyLevel default_level);
void gv_consistency_destroy(GV_ConsistencyManager *mgr);

int gv_consistency_set_default(GV_ConsistencyManager *mgr, GV_ConsistencyLevel level);
GV_ConsistencyLevel gv_consistency_get_default(const GV_ConsistencyManager *mgr);

int gv_consistency_check(const GV_ConsistencyManager *mgr, const GV_ConsistencyConfig *config,
                          uint64_t replica_lag_ms, uint64_t replica_position);

uint64_t gv_consistency_new_session(GV_ConsistencyManager *mgr);
int gv_consistency_update_session(GV_ConsistencyManager *mgr, uint64_t session_token, uint64_t write_position);
uint64_t gv_consistency_get_session_position(const GV_ConsistencyManager *mgr, uint64_t session_token);

void gv_consistency_config_init(GV_ConsistencyConfig *config);
GV_ConsistencyConfig gv_consistency_strong(void);
GV_ConsistencyConfig gv_consistency_eventual(void);
GV_ConsistencyConfig gv_consistency_bounded(uint64_t max_staleness_ms);
GV_ConsistencyConfig gv_consistency_session(uint64_t token);

/* --- Tenant Quotas --- */

typedef struct {
    size_t max_vectors;
    size_t max_memory_bytes;
    double max_qps;
    double max_ips;
    size_t max_storage_bytes;
    size_t max_collections;
} GV_QuotaConfig;

typedef struct {
    size_t current_vectors;
    size_t current_memory_bytes;
    double current_qps;
    double current_ips;
    size_t current_storage_bytes;
    size_t current_collections;
    uint64_t total_throttled;
    uint64_t total_rejected;
} GV_QuotaUsage;

typedef enum {
    GV_QUOTA_OK = 0,
    GV_QUOTA_THROTTLED = 1,
    GV_QUOTA_EXCEEDED = 2,
    GV_QUOTA_ERROR = -1
} GV_QuotaResult;

typedef struct GV_QuotaManager GV_QuotaManager;

GV_QuotaManager *gv_quota_create(void);
void gv_quota_destroy(GV_QuotaManager *mgr);

int gv_quota_set(GV_QuotaManager *mgr, const char *tenant_id, const GV_QuotaConfig *config);
int gv_quota_get(const GV_QuotaManager *mgr, const char *tenant_id, GV_QuotaConfig *config);
int gv_quota_remove(GV_QuotaManager *mgr, const char *tenant_id);

GV_QuotaResult gv_quota_check_insert(GV_QuotaManager *mgr, const char *tenant_id, size_t vector_count);
GV_QuotaResult gv_quota_check_query(GV_QuotaManager *mgr, const char *tenant_id);

int gv_quota_record_insert(GV_QuotaManager *mgr, const char *tenant_id, size_t count, size_t bytes);
int gv_quota_record_query(GV_QuotaManager *mgr, const char *tenant_id);
int gv_quota_record_delete(GV_QuotaManager *mgr, const char *tenant_id, size_t count, size_t bytes);

int gv_quota_get_usage(const GV_QuotaManager *mgr, const char *tenant_id, GV_QuotaUsage *usage);

int gv_quota_reset_usage(GV_QuotaManager *mgr, const char *tenant_id);

void gv_quota_config_init(GV_QuotaConfig *config);

/* --- Payload Compression --- */

typedef enum {
    GV_COMPRESS_NONE = 0,
    GV_COMPRESS_LZ4 = 1,
    GV_COMPRESS_ZSTD = 2,
    GV_COMPRESS_SNAPPY = 3
} GV_CompressionType;

typedef struct {
    GV_CompressionType type;
    int level;
    size_t min_size;
} GV_CompressionConfig;

typedef struct {
    uint64_t total_compressed;
    uint64_t total_decompressed;
    uint64_t bytes_in;
    uint64_t bytes_out;
    double avg_ratio;
} GV_CompressionStats;

typedef struct GV_Compressor GV_Compressor;

void gv_compression_config_init(GV_CompressionConfig *config);
GV_Compressor *gv_compression_create(const GV_CompressionConfig *config);
void gv_compression_destroy(GV_Compressor *comp);

size_t gv_compress(GV_Compressor *comp, const void *input, size_t input_len,
                    void *output, size_t output_capacity);

size_t gv_decompress(GV_Compressor *comp, const void *input, size_t input_len,
                      void *output, size_t output_capacity);

size_t gv_compress_bound(const GV_Compressor *comp, size_t input_len);

int gv_compression_get_stats(const GV_Compressor *comp, GV_CompressionStats *stats);

/* --- Webhooks / Change Streams --- */

typedef enum {
    GV_EVENT_INSERT = 1,
    GV_EVENT_UPDATE = 2,
    GV_EVENT_DELETE = 4,
    GV_EVENT_ALL = 7
} GV_EventType;

typedef struct {
    GV_EventType event_type;
    size_t vector_index;
    uint64_t timestamp;
    const char *collection;
} GV_Event;

typedef struct {
    char *url;
    GV_EventType event_mask;
    char *secret;
    int max_retries;
    int timeout_ms;
    int active;
} GV_WebhookConfig;

typedef struct GV_WebhookManager GV_WebhookManager;

GV_WebhookManager *gv_webhook_create(void);
void gv_webhook_destroy(GV_WebhookManager *mgr);

int gv_webhook_register(GV_WebhookManager *mgr, const char *webhook_id, const GV_WebhookConfig *config);
int gv_webhook_unregister(GV_WebhookManager *mgr, const char *webhook_id);
int gv_webhook_pause(GV_WebhookManager *mgr, const char *webhook_id);
int gv_webhook_resume(GV_WebhookManager *mgr, const char *webhook_id);

int gv_webhook_list(const GV_WebhookManager *mgr, char ***out_ids, size_t *out_count);
void gv_webhook_free_list(char **ids, size_t count);

int gv_webhook_fire(GV_WebhookManager *mgr, const GV_Event *event);

int gv_webhook_subscribe(GV_WebhookManager *mgr, GV_EventType mask, void *cb, void *user_data);
int gv_webhook_unsubscribe(GV_WebhookManager *mgr, void *cb);

typedef struct {
    uint64_t events_fired;
    uint64_t webhooks_delivered;
    uint64_t webhooks_failed;
    uint64_t callbacks_invoked;
} GV_WebhookStats;

int gv_webhook_get_stats(const GV_WebhookManager *mgr, GV_WebhookStats *stats);

/* --- RBAC Refinements --- */

typedef enum {
    GV_PERM_READ = 1,
    GV_PERM_WRITE = 2,
    GV_PERM_DELETE = 4,
    GV_PERM_ADMIN = 8,
    GV_PERM_ALL = 15
} GV_Permission;

typedef struct {
    char *resource;
    uint32_t permissions;
} GV_RBACRule;

typedef struct {
    char *role_name;
    GV_RBACRule *rules;
    size_t rule_count;
    int inherits_from;
} GV_Role;

typedef struct {
    char *user_id;
    char **role_names;
    size_t role_count;
} GV_UserRoles;

typedef struct GV_RBACManager GV_RBACManager;

GV_RBACManager *gv_rbac_create(void);
void gv_rbac_destroy(GV_RBACManager *mgr);

int gv_rbac_create_role(GV_RBACManager *mgr, const char *role_name);
int gv_rbac_delete_role(GV_RBACManager *mgr, const char *role_name);
int gv_rbac_add_rule(GV_RBACManager *mgr, const char *role_name,
                      const char *resource, uint32_t permissions);
int gv_rbac_remove_rule(GV_RBACManager *mgr, const char *role_name, const char *resource);
int gv_rbac_set_inheritance(GV_RBACManager *mgr, const char *role_name, const char *parent_role);

int gv_rbac_assign_role(GV_RBACManager *mgr, const char *user_id, const char *role_name);
int gv_rbac_revoke_role(GV_RBACManager *mgr, const char *user_id, const char *role_name);
int gv_rbac_get_user_roles(const GV_RBACManager *mgr, const char *user_id,
                            char ***out_roles, size_t *out_count);

int gv_rbac_check(const GV_RBACManager *mgr, const char *user_id,
                   const char *resource, GV_Permission required);

int gv_rbac_list_roles(const GV_RBACManager *mgr, char ***out_names, size_t *out_count);
void gv_rbac_free_string_list(char **list, size_t count);

int gv_rbac_init_defaults(GV_RBACManager *mgr);

int gv_rbac_save(const GV_RBACManager *mgr, const char *filepath);
GV_RBACManager *gv_rbac_load(const char *filepath);
"""
)


def _load_lib() -> "FFIType.CData":
    """Load the GigaVector shared library.

    Searches for libGigaVector.so in the following locations (in order):
    1. build/libGigaVector.so (development build)
    2. build/lib/libGigaVector.so (alternative build location)
    3. Packaged alongside this module

    Returns:
        CFFI library handle for calling C functions.

    Raises:
        FileNotFoundError: If libGigaVector.so is not found in any location.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent.parent  # .../GigaVector
    # Prefer freshly built library, fall back to packaged copy
    candidate_paths = [
        repo_root / "build" / "libGigaVector.so",  # Check build/ first
        repo_root / "build" / "lib" / "libGigaVector.so",  # Then build/lib/
        here / "libGigaVector.so",
    ]
    for lib_path in candidate_paths:
        if lib_path.exists():
            return ffi.dlopen(os.fspath(lib_path))
    raise FileNotFoundError(f"libGigaVector.so not found in {candidate_paths}")


#: Library handle for the GigaVector C library.
#: Use this to call C functions directly (advanced usage).
lib: "FFIType.CData" = _load_lib()

