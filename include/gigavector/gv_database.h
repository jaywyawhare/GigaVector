#ifndef GIGAVECTOR_GV_DATABASE_H
#define GIGAVECTOR_GV_DATABASE_H

#include <stddef.h>
#include <pthread.h>

#include "gv_types.h"
#include "gv_kdtree.h"
#include "gv_wal.h"
#include "gv_hnsw.h"
#include "gv_ivfpq.h"
#include "gv_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Index type enumeration.
 */
typedef enum {
    GV_INDEX_TYPE_KDTREE = 0,
    GV_INDEX_TYPE_HNSW = 1,
    GV_INDEX_TYPE_IVFPQ = 2
} GV_IndexType;

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
} GV_Database;

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

#ifdef __cplusplus
}
#endif

#endif
