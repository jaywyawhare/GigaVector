#ifndef GIGAVECTOR_GV_DATABASE_H
#define GIGAVECTOR_GV_DATABASE_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#include "gv_types.h"
#include "gv_kdtree.h"
#include "gv_wal.h"
#include "gv_hnsw.h"
#include "gv_ivfpq.h"
#include "gv_flat.h"
#include "gv_ivfflat.h"
#include "gv_pq.h"
#include "gv_lsh.h"
#include "gv_filter.h"
#include "gv_sparse_index.h"
#include "gv_sparse_vector.h"
#include "gv_soa_storage.h"
#include "gv_metadata_index.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Index type enumeration.
 */
typedef enum {
    GV_INDEX_TYPE_KDTREE  = 0,
    GV_INDEX_TYPE_HNSW    = 1,
    GV_INDEX_TYPE_IVFPQ   = 2,
    GV_INDEX_TYPE_SPARSE  = 3,
    GV_INDEX_TYPE_FLAT    = 4,
    GV_INDEX_TYPE_IVFFLAT = 5,
    GV_INDEX_TYPE_PQ      = 6,
    GV_INDEX_TYPE_LSH     = 7
} GV_IndexType;

/**
 * @brief Resource limits configuration for a database.
 */
typedef struct {
    size_t max_memory_bytes;           /**< Maximum memory usage in bytes (0 = unlimited). */
    size_t max_vectors;                /**< Maximum number of vectors (0 = unlimited). */
    size_t max_concurrent_operations;  /**< Maximum concurrent operations (0 = unlimited). */
} GV_ResourceLimits;

/**
 * @brief Latency histogram for operation timing.
 */
typedef struct {
    uint64_t *buckets;             /**< Array of bucket counts. */
    size_t bucket_count;           /**< Number of buckets. */
    double *bucket_boundaries;     /**< Upper boundaries for each bucket (in microseconds). */
    uint64_t total_samples;        /**< Total number of samples. */
    uint64_t sum_latency_us;       /**< Sum of all latencies in microseconds. */
} GV_LatencyHistogram;

/**
 * @brief Memory usage breakdown.
 */
typedef struct {
    size_t soa_storage_bytes;      /**< Memory used by SoA storage. */
    size_t index_bytes;            /**< Memory used by index structures. */
    size_t metadata_index_bytes;   /**< Memory used by metadata index. */
    size_t wal_bytes;              /**< Memory used by WAL. */
    size_t total_bytes;            /**< Total estimated memory usage. */
} GV_MemoryBreakdown;

/**
 * @brief Recall metrics for approximate search.
 */
typedef struct {
    uint64_t total_queries;        /**< Total queries used for recall calculation. */
    double avg_recall;             /**< Average recall (0.0 to 1.0). */
    double min_recall;             /**< Minimum recall observed. */
    double max_recall;             /**< Maximum recall observed. */
} GV_RecallMetrics;

/**
 * @brief Represents an in-memory vector database.
 */
typedef struct GV_Database {
    size_t dimension;
    GV_IndexType index_type;
    GV_KDNode *root;
    void *hnsw_index;
    char *filepath;
    char *wal_path;
    GV_WAL *wal;
    int wal_replaying;
    pthread_rwlock_t rwlock;
    pthread_mutex_t wal_mutex;
    size_t count;
    size_t exact_search_threshold; /**< Max collection size to use brute-force exact search. */
    int force_exact_search;        /**< Force exact search even when above threshold. */
    GV_SparseIndex *sparse_index;  /**< Sparse inverted index when index_type == GV_INDEX_TYPE_SPARSE. */
    GV_SoAStorage *soa_storage;    /**< Structure-of-Arrays storage for dense vectors (KD-tree, HNSW). */
    uint64_t total_inserts;        /**< Total successful vector insertions (dense + sparse). */
    uint64_t total_queries;        /**< Total k-NN / filtered / batch queries. */
    uint64_t total_range_queries;  /**< Total range-search calls. */
    uint64_t total_wal_records;    /**< Total WAL records appended. */
    int cosine_normalized;         /**< If non-zero, stored dense vectors are L2-normalized. */
    GV_MetadataIndex *metadata_index; /**< Inverted index for fast metadata filtering. */
    /* Background compaction */
    pthread_t compaction_thread;   /**< Background compaction thread handle. */
    int compaction_running;        /**< 1 if compaction thread is running, 0 otherwise. */
    pthread_mutex_t compaction_mutex; /**< Mutex for compaction thread control. */
    pthread_cond_t compaction_cond;    /**< Condition variable for compaction thread. */
    size_t compaction_interval_sec;    /**< Compaction interval in seconds (default: 300). */
    size_t wal_compaction_threshold;   /**< WAL size threshold for compaction in bytes (default: 10MB). */
    double deleted_ratio_threshold;    /**< Ratio of deleted vectors to trigger compaction (default: 0.1). */
    /* Resource limits */
    GV_ResourceLimits resource_limits; /**< Resource limits configuration. */
    size_t current_memory_bytes;       /**< Current estimated memory usage in bytes. */
    size_t current_concurrent_ops;     /**< Current number of concurrent operations. */
    pthread_mutex_t resource_mutex;     /**< Mutex for resource tracking. */
    /* Observability */
    GV_LatencyHistogram insert_latency_hist; /**< Insert operation latency histogram. */
    GV_LatencyHistogram search_latency_hist; /**< Search operation latency histogram. */
    uint64_t last_qps_update_time_us;  /**< Last QPS calculation time (microseconds). */
    uint64_t last_ips_update_time_us;  /**< Last IPS calculation time (microseconds). */
    uint64_t first_insert_time_us;     /**< Time of first insert (microseconds) - preserved for precise IPS calculation. */
    uint64_t query_count_since_update;  /**< Query count since last QPS update. */
    uint64_t insert_count_since_update; /**< Insert count since last IPS update. */
    double current_qps;                /**< Current queries per second. */
    double current_ips;                 /**< Current inserts per second. */
    GV_RecallMetrics recall_metrics;   /**< Recall metrics for approximate search. */
    pthread_mutex_t observability_mutex; /**< Mutex for observability data. */
} GV_Database;

/**
 * @brief Aggregated runtime statistics for a database.
 */
typedef struct {
    uint64_t total_inserts;        /**< Total successful vector insertions (dense + sparse). */
    uint64_t total_queries;        /**< Total k-NN / filtered / batch queries. */
    uint64_t total_range_queries;  /**< Total range-search calls. */
    uint64_t total_wal_records;    /**< Total WAL records appended. */
} GV_DBStats;

/**
 * @brief Detailed statistics for a database.
 */
typedef struct {
    /* Basic stats */
    GV_DBStats basic_stats;        /**< Basic aggregated statistics. */
    
    /* Latency histograms */
    GV_LatencyHistogram insert_latency;    /**< Insert operation latency histogram. */
    GV_LatencyHistogram search_latency;    /**< Search operation latency histogram. */
    
    /* QPS tracking */
    double queries_per_second;     /**< Current queries per second. */
    double inserts_per_second;     /**< Current inserts per second. */
    uint64_t last_qps_update_time; /**< Last QPS calculation time (microseconds since epoch). */
    
    /* Memory breakdown */
    GV_MemoryBreakdown memory;      /**< Memory usage breakdown. */
    
    /* Recall metrics */
    GV_RecallMetrics recall;       /**< Recall metrics for approximate search. */
    
    /* Health indicators */
    int health_status;             /**< Health status: 0 = healthy, -1 = degraded, -2 = unhealthy. */
    size_t deleted_vector_count;  /**< Number of deleted vectors. */
    double deleted_ratio;          /**< Ratio of deleted vectors (0.0 to 1.0). */
} GV_DetailedStats;

/**
 * @brief Open an in-memory database, optionally loading from a file.
 *
 * If @p filepath points to an existing file, the database is loaded from it.
 * If the file does not exist, a new empty database is created.
 *
 * @param filepath Optional file path string to associate with the database.
 * @param dimension Expected dimensionality; if loading, it must match the file.
 * @param index_type Type of index to use (KD-tree or HNSW).
 * @return Allocated database instance or NULL on invalid arguments or failure.
 */
GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type);

/**
 * @brief Open an in-memory database with HNSW configuration.
 *
 * Similar to gv_db_open, but allows specifying HNSW configuration parameters.
 * Only applicable when index_type is GV_INDEX_TYPE_HNSW.
 *
 * @param filepath Optional file path string to associate with the database.
 * @param dimension Expected dimensionality; if loading, it must match the file.
 * @param index_type Type of index to use (must be GV_INDEX_TYPE_HNSW for config to apply).
 * @param hnsw_config HNSW configuration; NULL to use defaults. Ignored if index_type is not HNSW.
 * @return Allocated database instance or NULL on invalid arguments or failure.
 */
GV_Database *gv_db_open_with_hnsw_config(const char *filepath, size_t dimension, 
                                          GV_IndexType index_type, const GV_HNSWConfig *hnsw_config);

/**
 * @brief Open an in-memory database with IVFPQ configuration.
 *
 * Similar to gv_db_open, but allows specifying IVFPQ configuration parameters.
 * Only applicable when index_type is GV_INDEX_TYPE_IVFPQ.
 *
 * @param filepath Optional file path string to associate with the database.
 * @param dimension Expected dimensionality; if loading, it must match the file.
 * @param index_type Type of index to use (must be GV_INDEX_TYPE_IVFPQ for config to apply).
 * @param ivfpq_config IVFPQ configuration; NULL to use defaults. Ignored if index_type is not IVFPQ.
 * @return Allocated database instance or NULL on invalid arguments or failure.
 */
GV_Database *gv_db_open_with_ivfpq_config(const char *filepath, size_t dimension, 
                                           GV_IndexType index_type, const GV_IVFPQConfig *ivfpq_config);

/**
 * @brief Suggest an index type based on dimension and expected collection size.
 *
 * Heuristic:
 *  - Small datasets (<= 20k) and low/moderate dimensions (<= 64): KDTREE.
 *  - Very large datasets (>= 500k) and high dimensions (>= 128): IVFPQ.
 *  - Otherwise: HNSW.
 *
 * @param dimension Vector dimensionality.
 * @param expected_count Estimated number of vectors to be stored.
 * @return Suggested GV_IndexType.
 */
GV_IndexType gv_index_suggest(size_t dimension, size_t expected_count);

/**
 * @brief Retrieve current runtime statistics for a database.
 *
 * @param db Database handle; must be non-NULL.
 * @param out Output pointer; must be non-NULL. Filled on success.
 */
void gv_db_get_stats(const GV_Database *db, GV_DBStats *out);

/**
 * @brief Enable or disable cosine pre-normalization for dense vectors.
 *
 * When enabled, all subsequently inserted dense vectors are L2-normalized
 * to unit length. For cosine distance, callers may safely treat it as
 * negative dot product on these normalized vectors.
 *
 * @param db Database handle; must be non-NULL.
 * @param enabled Non-zero to enable normalization, zero to disable.
 */
void gv_db_set_cosine_normalized(GV_Database *db, int enabled);

/**
 * @brief Open a database from an in-memory snapshot.
 *
 * The snapshot must contain a full GVDB binary image as produced by
 * gv_db_save(), including header, index data, and trailing checksum.
 * This function is read-only: WAL is disabled and modifications are
 * not persisted back to the snapshot.
 *
 * A common pattern is to memory map a snapshot file with gv_mmap_open_readonly()
 * and then call this function with the mapped pointer and size.
 *
 * @param data Pointer to contiguous snapshot bytes.
 * @param size Size of the snapshot in bytes.
 * @param dimension Expected dimensionality; if non-zero, must match snapshot.
 * @param index_type Index type stored in the snapshot.
 * @return Allocated database instance or NULL on invalid arguments or failure.
 */
GV_Database *gv_db_open_from_memory(const void *data, size_t size,
                                    size_t dimension, GV_IndexType index_type);

/**
 * @brief Open a database by memory-mapping an existing snapshot file.
 *
 * This is a convenience wrapper around gv_mmap_open_readonly() and
 * gv_db_open_from_memory(). The mapping is owned by the database and
 * released when gv_db_close() is called. The resulting database is
 * read-only: WAL is disabled and modifications are not persisted.
 *
 * @param filepath Path to a GVDB snapshot file produced by gv_db_save().
 * @param dimension Expected dimensionality; if non-zero, must match snapshot.
 * @param index_type Expected index type stored in the snapshot.
 * @return Allocated database instance or NULL on error.
 */
GV_Database *gv_db_open_mmap(const char *filepath, size_t dimension, GV_IndexType index_type);

/**
 * @brief Release all resources held by the database, including its K-D tree.
 *
 * Safe to call with NULL; no action is taken.
 *
 * @param db Database instance to destroy.
 */
void gv_db_close(GV_Database *db);

/**
 * @brief Add a vector to the database by copying user data into the tree.
 *
 * @param db Target database; must be non-NULL.
 * @param data Pointer to an array of @p dimension floats.
 * @param dimension Number of components provided in @p data; must equal db->dimension.
 * @return 0 on success, -1 on invalid arguments or allocation failure.
 */
int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);

/**
 * @brief Add a vector with metadata to the database.
 *
 * @param db Target database; must be non-NULL.
 * @param data Pointer to an array of @p dimension floats.
 * @param dimension Number of components provided in @p data; must equal db->dimension.
 * @param metadata_key Optional metadata key; NULL to skip.
 * @param metadata_value Optional metadata value; NULL if key is NULL.
 * @return 0 on success, -1 on invalid arguments or allocation failure.
 */
int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                    const char *metadata_key, const char *metadata_value);
/**
 * @brief Train IVF-PQ index with provided training data.
 *
 * @param db Database; must be IVF-PQ type.
 * @param data Contiguous floats of size count * dimension.
 * @param count Number of training vectors.
 * @param dimension Vector dimension; must match db->dimension.
 * @return 0 on success, -1 on invalid args or training failure.
 */
int gv_db_ivfpq_train(GV_Database *db, const float *data, size_t count, size_t dimension);

/**
 * @brief Open an in-memory database with IVF-Flat configuration.
 *
 * @param filepath Optional file path string to associate with the database.
 * @param dimension Expected dimensionality.
 * @param index_type Must be GV_INDEX_TYPE_IVFFLAT for config to apply.
 * @param config IVF-Flat configuration; NULL to use defaults.
 * @return Allocated database instance or NULL on failure.
 */
GV_Database *gv_db_open_with_ivfflat_config(const char *filepath, size_t dimension,
                                             GV_IndexType index_type, const GV_IVFFlatConfig *config);

/**
 * @brief Open an in-memory database with PQ configuration.
 *
 * @param filepath Optional file path string to associate with the database.
 * @param dimension Expected dimensionality.
 * @param index_type Must be GV_INDEX_TYPE_PQ for config to apply.
 * @param config PQ configuration; NULL to use defaults.
 * @return Allocated database instance or NULL on failure.
 */
GV_Database *gv_db_open_with_pq_config(const char *filepath, size_t dimension,
                                        GV_IndexType index_type, const GV_PQConfig *config);

/**
 * @brief Open an in-memory database with LSH configuration.
 *
 * @param filepath Optional file path string to associate with the database.
 * @param dimension Expected dimensionality.
 * @param index_type Must be GV_INDEX_TYPE_LSH for config to apply.
 * @param config LSH configuration; NULL to use defaults.
 * @return Allocated database instance or NULL on failure.
 */
GV_Database *gv_db_open_with_lsh_config(const char *filepath, size_t dimension,
                                         GV_IndexType index_type, const GV_LSHConfig *config);

/**
 * @brief Train IVF-Flat index with provided training data.
 *
 * @param db Database; must be IVF-Flat type.
 * @param data Contiguous floats of size count * dimension.
 * @param count Number of training vectors.
 * @param dimension Vector dimension; must match db->dimension.
 * @return 0 on success, -1 on error.
 */
int gv_db_ivfflat_train(GV_Database *db, const float *data, size_t count, size_t dimension);

/**
 * @brief Train PQ index with provided training data.
 *
 * @param db Database; must be PQ type.
 * @param data Contiguous floats of size count * dimension.
 * @param count Number of training vectors.
 * @param dimension Vector dimension; must match db->dimension.
 * @return 0 on success, -1 on error.
 */
int gv_db_pq_train(GV_Database *db, const float *data, size_t count, size_t dimension);

/**
 * @brief Add a vector with multiple metadata entries to the database.
 *
 * @param db Target database; must be non-NULL.
 * @param data Pointer to an array of @p dimension floats.
 * @param dimension Number of components provided in @p data; must equal db->dimension.
 * @param metadata_keys Array of metadata keys; NULL if count is 0.
 * @param metadata_values Array of metadata values; NULL if count is 0.
 * @param metadata_count Number of metadata entries.
 * @return 0 on success, -1 on invalid arguments or allocation failure.
 */
int gv_db_add_vector_with_rich_metadata(GV_Database *db, const float *data, size_t dimension,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count);

/**
 * @brief Add a sparse vector to the database (sparse index only).
 *
 * @param db Target database; must be GV_INDEX_TYPE_SPARSE.
 * @param indices Dimension indices array of length nnz.
 * @param values Values array of length nnz.
 * @param nnz Number of non-zero entries.
 * @param dimension Vector dimension; must match db->dimension.
 * @param metadata_key Optional metadata key; NULL to skip.
 * @param metadata_value Optional metadata value; NULL if key is NULL.
 * @return 0 on success, -1 on error.
 */
int gv_db_add_sparse_vector(GV_Database *db, const uint32_t *indices, const float *values,
                            size_t nnz, size_t dimension,
                            const char *metadata_key, const char *metadata_value);

/**
 * @brief Delete a vector from the database by its index (insertion order).
 *
 * @param db Target database; must be non-NULL.
 * @param vector_index Index of the vector to delete (0-based insertion order).
 * @return 0 on success, -1 on invalid arguments or vector not found.
 */
int gv_db_delete_vector_by_index(GV_Database *db, size_t vector_index);

/**
 * @brief Update a vector in the database by its index (insertion order).
 *
 * @param db Target database; must be non-NULL.
 * @param vector_index Index of the vector to update (0-based insertion order).
 * @param new_data Pointer to an array of @p dimension floats.
 * @param dimension Number of components provided in @p new_data; must equal db->dimension.
 * @return 0 on success, -1 on invalid arguments or vector not found.
 */
int gv_db_update_vector(GV_Database *db, size_t vector_index, const float *new_data, size_t dimension);

/**
 * @brief Update metadata for a vector in the database by its index.
 *
 * @param db Target database; must be non-NULL.
 * @param vector_index Index of the vector to update (0-based insertion order).
 * @param metadata_keys Array of metadata keys; NULL if count is 0.
 * @param metadata_values Array of metadata values; NULL if count is 0.
 * @param metadata_count Number of metadata entries.
 * @return 0 on success, -1 on invalid arguments or vector not found.
 */
int gv_db_update_vector_metadata(GV_Database *db, size_t vector_index,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count);

/**
 * @brief Save the database (tree and vectors) to a binary file.
 *
 * @param db Database to persist; must be non-NULL.
 * @param filepath Output file path; if NULL, uses db->filepath.
 * @return 0 on success, -1 on invalid arguments or I/O failures.
 */
int gv_db_save(const GV_Database *db, const char *filepath);

/**
 * @brief Search for k nearest neighbors to a query vector.
 *
 * @param db Database to search; must be non-NULL.
 * @param query_data Query vector data array.
 * @param k Number of nearest neighbors to find.
 * @param results Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Search for k nearest neighbors with metadata filtering.
 *
 * Only vectors matching the metadata filter (key-value pair) are considered.
 *
 * @param db Database to search; must be non-NULL.
 * @param query_data Query vector data array.
 * @param k Number of nearest neighbors to find.
 * @param results Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @param filter_key Metadata key to filter by; NULL to disable filtering.
 * @param filter_value Metadata value to match; NULL if filter_key is NULL.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_db_search_filtered(const GV_Database *db, const float *query_data, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type,
                          const char *filter_key, const char *filter_value);

/**
 * @brief Range search: find all vectors within a distance threshold.
 *
 * @param db Database to search; must be non-NULL.
 * @param query_data Query vector data array.
 * @param radius Maximum distance threshold (inclusive).
 * @param results Output array to store results; must be pre-allocated with sufficient capacity.
 * @param max_results Maximum number of results to return (capacity of results array).
 * @param distance_type Distance metric to use.
 * @return Number of vectors found within radius (0 to max_results), or -1 on error.
 *         If more vectors are found than max_results, returns max_results.
 */
int gv_db_range_search(const GV_Database *db, const float *query_data, float radius,
                       GV_SearchResult *results, size_t max_results, GV_DistanceType distance_type);

/**
 * @brief Range search with metadata filtering.
 *
 * Only vectors matching the metadata filter and within the distance threshold are returned.
 *
 * @param db Database to search; must be non-NULL.
 * @param query_data Query vector data array.
 * @param radius Maximum distance threshold (inclusive).
 * @param results Output array to store results; must be pre-allocated with sufficient capacity.
 * @param max_results Maximum number of results to return (capacity of results array).
 * @param distance_type Distance metric to use.
 * @param filter_key Metadata key to filter by; NULL to disable filtering.
 * @param filter_value Metadata value to match; NULL if filter_key is NULL.
 * @return Number of vectors found within radius (0 to max_results), or -1 on error.
 */
int gv_db_range_search_filtered(const GV_Database *db, const float *query_data, float radius,
                                 GV_SearchResult *results, size_t max_results,
                                 GV_DistanceType distance_type,
                          const char *filter_key, const char *filter_value);

/**
 * @brief IVF-PQ search with per-query overrides.
 *
 * @param db Database to search; must be IVF-PQ.
 * @param query_data Query vector data array.
 * @param k Number of nearest neighbors to find.
 * @param results Output array of at least k elements.
 * @param distance_type Distance metric to use.
 * @param nprobe_override If >0, overrides configured nprobe (capped at nlist).
 * @param rerank_top If >0, reranks this many top PQ hits with exact L2.
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_db_search_ivfpq_opts(const GV_Database *db, const float *query_data, size_t k,
                            GV_SearchResult *results, GV_DistanceType distance_type,
                            size_t nprobe_override, size_t rerank_top);

/**
 * @brief Batch-insert multiple vectors from a contiguous float buffer.
 *
 * Layout: data is count * dimension floats, contiguous. Each vector has no
 * metadata. Equivalent to calling gv_db_add_vector repeatedly but with less
 * per-call overhead.
 *
 * @param db Target database; must be non-NULL.
 * @param data Pointer to contiguous floats of size count * dimension.
 * @param count Number of vectors.
 * @param dimension Vector dimensionality (must match db->dimension).
 * @return 0 on success, -1 on error (no partial rollback).
 */
int gv_db_add_vectors(GV_Database *db, const float *data, size_t count, size_t dimension);

/**
 * @brief Batch-insert vectors with optional metadata (one key/value per vector).
 *
 * @param db Target database; must be non-NULL.
 * @param data Contiguous floats of size count * dimension.
 * @param keys Optional array of metadata keys (can be NULL).
 * @param values Optional array of metadata values (can be NULL if keys is NULL).
 * @param count Number of vectors.
 * @param dimension Vector dimensionality (must match db->dimension).
 * @return 0 on success, -1 on error (no partial rollback).
 */
int gv_db_add_vectors_with_metadata(GV_Database *db, const float *data,
                                    const char *const *keys, const char *const *values,
                                    size_t count, size_t dimension);

/**
 * @brief Batch search for multiple queries.
 *
 * @param db Database to search; must be non-NULL.
 * @param queries Contiguous float array of size qcount * dimension.
 * @param qcount Number of queries.
 * @param k Number of neighbors per query.
 * @param results Output array sized qcount * k. Results for query i start at
 *                results[i * k].
 * @param distance_type Distance metric to use.
 * @return Total results written (qcount * k) on success, or -1 on error.
 */
int gv_db_search_batch(const GV_Database *db, const float *queries, size_t qcount, size_t k,
                       GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Search with an advanced metadata filter expression.
 *
 * The filter expression is parsed and evaluated against vector metadata.
 * All index types are supported via the database abstraction. Internally,
 * the implementation may oversample and apply post-filtering to satisfy
 * the requested @p k matches.
 *
 * @param db Database to search; must be non-NULL.
 * @param query_data Query vector data array.
 * @param k Number of nearest neighbors to return after filtering.
 * @param results Output array of at least @p k elements.
 * @param distance_type Distance metric to use.
 * @param filter_expr Null-terminated filter expression string; see gv_filter_parse().
 * @return Number of neighbors found (0 to k), or -1 on error.
 */
int gv_db_search_with_filter_expr(const GV_Database *db, const float *query_data, size_t k,
                                  GV_SearchResult *results, GV_DistanceType distance_type,
                                  const char *filter_expr);

/**
 * @brief Configure the exact search threshold for a database.
 *
 * When the number of stored vectors is less than or equal to this threshold,
 * the database may choose a brute-force exact search path instead of using
 * the index. A threshold of 0 disables automatic exact search fallback.
 *
 * @param db Database handle; must be non-NULL.
 * @param threshold New threshold value.
 */
void gv_db_set_exact_search_threshold(GV_Database *db, size_t threshold);

/**
 * @brief Force or disable exact search regardless of collection size.
 *
 * When enabled, the database prefers brute-force exact search over indexed
 * search for supported index types. This is primarily intended for testing
 * and benchmarking exact-search behavior.
 *
 * @param db Database handle; must be non-NULL.
 * @param enabled Non-zero to force exact search, zero to use automatic logic.
 */
void gv_db_set_force_exact_search(GV_Database *db, int enabled);

/**
 * @brief Search sparse index with a sparse query.
 *
 * @param db Database; must be GV_INDEX_TYPE_SPARSE.
 * @param indices Query indices array.
 * @param values Query values array.
 * @param nnz Number of entries in query.
 * @param k Number of results.
 * @param results Output array of at least k elements.
 * @param distance_type GV_DISTANCE_DOT_PRODUCT or GV_DISTANCE_COSINE.
 * @return Number of results (0..k) or -1 on error.
 */
int gv_db_search_sparse(const GV_Database *db, const uint32_t *indices, const float *values,
                        size_t nnz, size_t k, GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Enable or reconfigure WAL for a database.
 *
 * If a WAL is already open, it is closed and replaced. Passing NULL disables
 * WAL logging.
 *
 * @param db Database handle; must be non-NULL.
 * @param wal_path Filesystem path for the WAL; NULL to disable WAL.
 * @return 0 on success, -1 on invalid arguments or I/O failure.
 */
int gv_db_set_wal(GV_Database *db, const char *wal_path);

/**
 * @brief Disable WAL for the database, closing any open WAL handle.
 *
 * @param db Database handle; must be non-NULL.
 */
void gv_db_disable_wal(GV_Database *db);

/**
 * @brief Dump the current WAL contents in human-readable form.
 *
 * @param db Database handle; must be non-NULL and have WAL enabled.
 * @param out Output stream (e.g., stdout); must be non-NULL.
 * @return 0 on success, -1 on error or if WAL is disabled.
 */
int gv_db_wal_dump(const GV_Database *db, FILE *out);

/**
 * @brief Start background compaction thread.
 *
 * The compaction thread periodically:
 * - Removes deleted vectors from SoA storage
 * - Rebuilds indexes to remove gaps
 * - Compacts WAL when it grows too large
 *
 * @param db Database instance; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_db_start_background_compaction(GV_Database *db);

/**
 * @brief Stop background compaction thread.
 *
 * This function waits for the compaction thread to finish its current operation
 * and then stops it gracefully.
 *
 * @param db Database instance; must be non-NULL.
 */
void gv_db_stop_background_compaction(GV_Database *db);

/**
 * @brief Manually trigger compaction (runs synchronously).
 *
 * This function performs the same compaction operations as the background thread
 * but runs synchronously in the current thread.
 *
 * @param db Database instance; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_db_compact(GV_Database *db);

/**
 * @brief Set compaction interval in seconds.
 *
 * @param db Database instance; must be non-NULL.
 * @param interval_sec Compaction interval in seconds (default: 300).
 */
void gv_db_set_compaction_interval(GV_Database *db, size_t interval_sec);

/**
 * @brief Set WAL compaction threshold in bytes.
 *
 * @param db Database instance; must be non-NULL.
 * @param threshold_bytes WAL size threshold for compaction (default: 10MB).
 */
void gv_db_set_wal_compaction_threshold(GV_Database *db, size_t threshold_bytes);

/**
 * @brief Set deleted vector ratio threshold for triggering compaction.
 *
 * Compaction is triggered when the ratio of deleted vectors exceeds this threshold.
 *
 * @param db Database instance; must be non-NULL.
 * @param ratio Threshold ratio (0.0 to 1.0, default: 0.1).
 */
void gv_db_set_deleted_ratio_threshold(GV_Database *db, double ratio);

/**
 * @brief Set resource limits for the database.
 *
 * @param db Database instance; must be non-NULL.
 * @param limits Resource limits structure; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_db_set_resource_limits(GV_Database *db, const GV_ResourceLimits *limits);

/**
 * @brief Get current resource limits.
 *
 * @param db Database instance; must be non-NULL.
 * @param limits Output structure to fill; must be non-NULL.
 */
void gv_db_get_resource_limits(const GV_Database *db, GV_ResourceLimits *limits);

/**
 * @brief Get current estimated memory usage in bytes.
 *
 * @param db Database instance; must be non-NULL.
 * @return Current memory usage in bytes.
 */
size_t gv_db_get_memory_usage(const GV_Database *db);

/**
 * @brief Get current number of concurrent operations.
 *
 * @param db Database instance; must be non-NULL.
 * @return Current number of concurrent operations.
 */
size_t gv_db_get_concurrent_operations(const GV_Database *db);

/**
 * @brief Get detailed statistics for the database.
 *
 * @param db Database instance; must be non-NULL.
 * @param out Output structure to fill; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_db_get_detailed_stats(const GV_Database *db, GV_DetailedStats *out);

/**
 * @brief Free resources allocated by gv_db_get_detailed_stats().
 *
 * @param stats Detailed stats structure to free.
 */
void gv_db_free_detailed_stats(GV_DetailedStats *stats);

/**
 * @brief Perform health check on the database.
 *
 * Checks database integrity, index consistency, and resource usage.
 *
 * @param db Database instance; must be non-NULL.
 * @return 0 if healthy, -1 if degraded, -2 if unhealthy.
 */
int gv_db_health_check(const GV_Database *db);

/**
 * @brief Record latency for an operation.
 *
 * @param db Database instance; must be non-NULL.
 * @param latency_us Latency in microseconds.
 * @param is_insert 1 for insert operations, 0 for search operations.
 */
void gv_db_record_latency(GV_Database *db, uint64_t latency_us, int is_insert);

/**
 * @brief Record recall for a search operation.
 *
 * @param db Database instance; must be non-NULL.
 * @param recall Recall value (0.0 to 1.0).
 */
void gv_db_record_recall(GV_Database *db, double recall);

/**
 * @brief Get the number of vectors in the database.
 *
 * @param db Database instance; must be non-NULL.
 * @return Number of vectors stored.
 */
size_t gv_database_count(const GV_Database *db);

/**
 * @brief Get the dimension of vectors in the database.
 *
 * @param db Database instance; must be non-NULL.
 * @return Vector dimension.
 */
size_t gv_database_dimension(const GV_Database *db);

/**
 * @brief Get a pointer to the vector data at the given index.
 *
 * @param db Database instance; must be non-NULL.
 * @param index Vector index; must be < count.
 * @return Pointer to vector data, or NULL on invalid index.
 */
const float *gv_database_get_vector(const GV_Database *db, size_t index);

#ifdef __cplusplus
}
#endif

#endif
