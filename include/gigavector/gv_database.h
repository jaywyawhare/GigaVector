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
 * @brief Create a new in-memory database with a target vector dimension.
 *
 * @param filepath Optional file path string to associate with the database.
 * @param dimension Dimensionality all vectors in this database must share.
 * @return Allocated database instance or NULL on invalid arguments or allocation failure.
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

#ifdef __cplusplus
}
#endif

#endif
