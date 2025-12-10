#ifndef GIGAVECTOR_GV_DATABASE_H
#define GIGAVECTOR_GV_DATABASE_H

#include <stddef.h>
#include <pthread.h>

#include "gv_types.h"
#include "gv_kdtree.h"
#include "gv_wal.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Index type enumeration.
 */
typedef enum {
    GV_INDEX_TYPE_KDTREE = 0,
    GV_INDEX_TYPE_HNSW = 1
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
