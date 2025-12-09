#ifndef GIGAVECTOR_GV_DATABASE_H
#define GIGAVECTOR_GV_DATABASE_H

#include <stddef.h>

#include "gv_kdtree.h"
#include "gv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents an in-memory vector database.
 */
typedef struct GV_Database {
    size_t dimension;
    GV_KDNode *root;
    char *filepath;
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
 * @return Allocated database instance or NULL on invalid arguments or failure.
 */
GV_Database *gv_db_open(const char *filepath, size_t dimension);

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

#ifdef __cplusplus
}
#endif

#endif
