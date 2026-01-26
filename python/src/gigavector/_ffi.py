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
typedef enum { GV_INDEX_TYPE_KDTREE = 0, GV_INDEX_TYPE_HNSW = 1, GV_INDEX_TYPE_IVFPQ = 2, GV_INDEX_TYPE_SPARSE = 3 } GV_IndexType;
typedef enum { GV_DISTANCE_EUCLIDEAN = 0, GV_DISTANCE_COSINE = 1, GV_DISTANCE_DOT_PRODUCT = 2, GV_DISTANCE_MANHATTAN = 3 } GV_DistanceType;

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

