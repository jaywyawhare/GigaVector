#ifndef GIGAVECTOR_GV_SQL_H
#define GIGAVECTOR_GV_SQL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_sql.h
 * @brief SQL-like query interface for GigaVector.
 *
 * Parse and execute SQL-like queries against the vector database.
 * Supports SELECT with WHERE, ORDER BY, LIMIT, and ANN (approximate
 * nearest neighbor) operators, as well as DELETE and UPDATE statements.
 *
 * Supported SQL syntax:
 *
 *   -- Vector search
 *   SELECT * FROM vectors ANN(query=[0.1,0.2,...], k=10, metric=cosine)
 *   SELECT * FROM vectors ANN(query=[0.1,0.2,...], k=10) WHERE category = 'science'
 *
 *   -- Metadata queries
 *   SELECT * FROM vectors WHERE score > 0.5 AND category = 'tech' LIMIT 100
 *
 *   -- Count
 *   SELECT COUNT(*) FROM vectors WHERE status = 'active'
 *
 *   -- Delete
 *   DELETE FROM vectors WHERE category = 'old'
 *
 *   -- Update metadata
 *   UPDATE vectors SET status = 'archived' WHERE score < 0.1
 */

/**
 * @brief Result set returned by a SQL query execution.
 */
typedef struct {
    size_t *indices;          /**< Array of matching vector indices (row_count elements). */
    float *distances;         /**< Array of distances for ANN queries (row_count elements); NULL for non-ANN. */
    char **metadata_jsons;    /**< Array of JSON-serialized metadata strings (row_count elements). */
    size_t row_count;         /**< Number of result rows. */
    size_t column_count;      /**< Number of columns in the result set. */
    char **column_names;      /**< Array of column name strings (column_count elements). */
} GV_SQLResult;

/**
 * @brief Opaque SQL engine handle.
 *
 * Thread-safe: all operations on a single engine instance are serialized
 * with an internal mutex.
 */
typedef struct GV_SQLEngine GV_SQLEngine;

/**
 * @brief Create a SQL engine bound to a database.
 *
 * @param db Database handle (GV_Database *); must be non-NULL and remain valid
 *           for the lifetime of the engine.
 * @return Allocated engine instance or NULL on failure.
 */
GV_SQLEngine *gv_sql_create(void *db);

/**
 * @brief Destroy a SQL engine and release all resources.
 *
 * Safe to call with NULL; no action is taken.
 *
 * @param eng Engine instance to destroy.
 */
void gv_sql_destroy(GV_SQLEngine *eng);

/**
 * @brief Execute a SQL query string against the database.
 *
 * On success the caller owns @p result and must free it with
 * gv_sql_free_result().  On error the result is left untouched and
 * a human-readable message is available via gv_sql_last_error().
 *
 * @param eng    Engine instance; must be non-NULL.
 * @param query  Null-terminated SQL query string; must be non-NULL.
 * @param result Output result set; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_sql_execute(GV_SQLEngine *eng, const char *query, GV_SQLResult *result);

/**
 * @brief Free all resources held by a result set.
 *
 * Safe to call with NULL or an already-freed result.  After this call
 * the struct is zeroed.
 *
 * @param result Result set to free.
 */
void gv_sql_free_result(GV_SQLResult *result);

/**
 * @brief Retrieve the last error message from the engine.
 *
 * The returned pointer is valid until the next call to gv_sql_execute()
 * or gv_sql_explain() on the same engine.
 *
 * @param eng Engine instance; must be non-NULL.
 * @return Null-terminated error string, or empty string if no error.
 */
const char *gv_sql_last_error(const GV_SQLEngine *eng);

/**
 * @brief Produce an execution plan for a query without running it.
 *
 * Writes a human-readable plan string into @p plan (including the chosen
 * index type, estimated row count, and filter strategy).
 *
 * @param eng       Engine instance; must be non-NULL.
 * @param query     Null-terminated SQL query string; must be non-NULL.
 * @param plan      Output buffer for the plan string.
 * @param plan_size Size of @p plan in bytes.
 * @return 0 on success, -1 on parse/explain error.
 */
int gv_sql_explain(GV_SQLEngine *eng, const char *query, char *plan, size_t plan_size);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SQL_H */
