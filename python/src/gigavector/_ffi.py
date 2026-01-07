from __future__ import annotations

import os
from pathlib import Path

from cffi import FFI

ffi = FFI()

# Keep this cdef in sync with the C headers.
ffi.cdef(
    """
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
"""
)


def _load_lib():
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent.parent  # .../GigaVector
    # Prefer freshly built library, fall back to packaged copy
    candidate_paths = [
        repo_root / "build" / "lib" / "libGigaVector.so",
        here / "libGigaVector.so",
    ]
    for lib_path in candidate_paths:
        if lib_path.exists():
            return ffi.dlopen(os.fspath(lib_path))
    raise FileNotFoundError(f"libGigaVector.so not found in {candidate_paths}")


lib = _load_lib()

