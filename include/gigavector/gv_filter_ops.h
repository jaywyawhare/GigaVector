#ifndef GIGAVECTOR_GV_FILTER_OPS_H
#define GIGAVECTOR_GV_FILTER_OPS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Database GV_Database;

/**
 * Delete all vectors matching a filter expression.
 * Uses the same filter syntax as gv_filter_parse() from gv_filter.h.
 *
 * @param db Database to modify.
 * @param filter_expr Filter expression string.
 * @return Number of vectors deleted, or -1 on error.
 */
int gv_db_delete_by_filter(GV_Database *db, const char *filter_expr);

/**
 * Update vector data for all vectors matching a filter expression.
 *
 * @param db Database to modify.
 * @param filter_expr Filter expression string.
 * @param new_data New vector data to set for matching vectors.
 * @param dimension Vector dimension.
 * @return Number of vectors updated, or -1 on error.
 */
int gv_db_update_by_filter(GV_Database *db, const char *filter_expr,
                            const float *new_data, size_t dimension);

/**
 * Update metadata for all vectors matching a filter expression.
 *
 * @param db Database to modify.
 * @param filter_expr Filter expression string.
 * @param metadata_keys Array of metadata keys to set.
 * @param metadata_values Array of metadata values to set.
 * @param metadata_count Number of metadata entries.
 * @return Number of vectors updated, or -1 on error.
 */
int gv_db_update_metadata_by_filter(GV_Database *db, const char *filter_expr,
                                     const char *const *metadata_keys,
                                     const char *const *metadata_values,
                                     size_t metadata_count);

/**
 * Count vectors matching a filter expression (without modifying).
 *
 * @param db Database to scan.
 * @param filter_expr Filter expression string.
 * @return Number of matching vectors, or -1 on error.
 */
int gv_db_count_by_filter(const GV_Database *db, const char *filter_expr);

/**
 * Get indices of vectors matching a filter expression.
 *
 * @param db Database to scan.
 * @param filter_expr Filter expression string.
 * @param out_indices Output array (caller allocates).
 * @param max_count Maximum indices to return.
 * @return Number of matching indices, or -1 on error.
 */
int gv_db_find_by_filter(const GV_Database *db, const char *filter_expr,
                          size_t *out_indices, size_t max_count);

#ifdef __cplusplus
}
#endif
#endif
