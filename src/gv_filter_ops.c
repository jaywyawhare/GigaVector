/*
 * gv_filter_ops.c -- Bulk delete/update operations driven by filter expressions.
 *
 * Every public function follows the same pattern:
 *   1. Parse the filter expression with gv_filter_parse().
 *   2. Iterate over all vectors in the SoA storage.
 *   3. Skip deleted vectors.
 *   4. Build a lightweight GV_Vector view for the filter evaluator.
 *   5. Evaluate the filter; on match, perform the requested operation.
 *   6. Clean up and return the match count (or -1 on error).
 */

#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_filter_ops.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_filter.h"
#include "gigavector/gv_types.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_soa_storage.h"

/* Internal helper: collect all non-deleted indices that match a       */
/* compiled filter.  Returns match count, or -1 on error.  When       */
/* out_indices is NULL the function just counts.                       */
static int gv_filter_ops_collect_matches(const GV_Database *db,
                                         const GV_Filter *filter,
                                         size_t *out_indices,
                                         size_t max_count)
{
    if (!db || !filter) {
        return -1;
    }

    const GV_SoAStorage *storage = db->soa_storage;
    if (!storage) {
        return -1;
    }

    size_t total = gv_soa_storage_count(storage);
    int matched = 0;

    for (size_t i = 0; i < total; i++) {
        /* Skip deleted vectors. */
        if (gv_soa_storage_is_deleted(storage, i) == 1) {
            continue;
        }

        /* Build a temporary GV_Vector view for the filter evaluator. */
        GV_Vector view;
        if (gv_soa_storage_get_vector_view(storage, i, &view) != 0) {
            continue;
        }

        int result = gv_filter_eval(filter, &view);
        if (result == 1) {
            if (out_indices != NULL && (size_t)matched < max_count) {
                out_indices[matched] = i;
            }
            matched++;
        }
    }

    return matched;
}

/* Public API                                                         */

int gv_db_delete_by_filter(GV_Database *db, const char *filter_expr)
{
    if (!db || !filter_expr) {
        return -1;
    }

    GV_Filter *filter = gv_filter_parse(filter_expr);
    if (!filter) {
        return -1;
    }

    /* First pass: count matches so we can allocate an index buffer. */
    int count = gv_filter_ops_collect_matches(db, filter, NULL, 0);
    if (count <= 0) {
        gv_filter_destroy(filter);
        return count; /* 0 matches or error */
    }

    size_t *indices = (size_t *)malloc(sizeof(size_t) * (size_t)count);
    if (!indices) {
        gv_filter_destroy(filter);
        return -1;
    }

    /* Second pass: collect the actual indices. */
    int collected = gv_filter_ops_collect_matches(db, filter, indices, (size_t)count);
    gv_filter_destroy(filter);

    if (collected < 0) {
        free(indices);
        return -1;
    }

    /*
     * Delete in reverse order so that earlier indices remain valid when
     * the underlying storage merely marks entries as deleted (which is
     * the case for SoA storage).  This also guards against any future
     * implementation that compacts on delete.
     */
    int deleted = 0;
    for (int i = collected - 1; i >= 0; i--) {
        if (gv_db_delete_vector_by_index(db, indices[i]) == 0) {
            deleted++;
        }
    }

    free(indices);
    return deleted;
}

int gv_db_update_by_filter(GV_Database *db, const char *filter_expr,
                            const float *new_data, size_t dimension)
{
    if (!db || !filter_expr || !new_data || dimension == 0) {
        return -1;
    }

    GV_Filter *filter = gv_filter_parse(filter_expr);
    if (!filter) {
        return -1;
    }

    /* First pass: count. */
    int count = gv_filter_ops_collect_matches(db, filter, NULL, 0);
    if (count <= 0) {
        gv_filter_destroy(filter);
        return count;
    }

    size_t *indices = (size_t *)malloc(sizeof(size_t) * (size_t)count);
    if (!indices) {
        gv_filter_destroy(filter);
        return -1;
    }

    /* Second pass: collect indices. */
    int collected = gv_filter_ops_collect_matches(db, filter, indices, (size_t)count);
    gv_filter_destroy(filter);

    if (collected < 0) {
        free(indices);
        return -1;
    }

    int updated = 0;
    for (int i = 0; i < collected; i++) {
        if (gv_db_update_vector(db, indices[i], new_data, dimension) == 0) {
            updated++;
        }
    }

    free(indices);
    return updated;
}

int gv_db_update_metadata_by_filter(GV_Database *db, const char *filter_expr,
                                     const char *const *metadata_keys,
                                     const char *const *metadata_values,
                                     size_t metadata_count)
{
    if (!db || !filter_expr) {
        return -1;
    }
    if (metadata_count > 0 && (!metadata_keys || !metadata_values)) {
        return -1;
    }

    GV_Filter *filter = gv_filter_parse(filter_expr);
    if (!filter) {
        return -1;
    }

    /* First pass: count. */
    int count = gv_filter_ops_collect_matches(db, filter, NULL, 0);
    if (count <= 0) {
        gv_filter_destroy(filter);
        return count;
    }

    size_t *indices = (size_t *)malloc(sizeof(size_t) * (size_t)count);
    if (!indices) {
        gv_filter_destroy(filter);
        return -1;
    }

    /* Second pass: collect indices. */
    int collected = gv_filter_ops_collect_matches(db, filter, indices, (size_t)count);
    gv_filter_destroy(filter);

    if (collected < 0) {
        free(indices);
        return -1;
    }

    int updated = 0;
    for (int i = 0; i < collected; i++) {
        if (gv_db_update_vector_metadata(db, indices[i],
                                          metadata_keys, metadata_values,
                                          metadata_count) == 0) {
            updated++;
        }
    }

    free(indices);
    return updated;
}

int gv_db_count_by_filter(const GV_Database *db, const char *filter_expr)
{
    if (!db || !filter_expr) {
        return -1;
    }

    GV_Filter *filter = gv_filter_parse(filter_expr);
    if (!filter) {
        return -1;
    }

    int count = gv_filter_ops_collect_matches(db, filter, NULL, 0);
    gv_filter_destroy(filter);
    return count;
}

int gv_db_find_by_filter(const GV_Database *db, const char *filter_expr,
                          size_t *out_indices, size_t max_count)
{
    if (!db || !filter_expr || !out_indices || max_count == 0) {
        return -1;
    }

    GV_Filter *filter = gv_filter_parse(filter_expr);
    if (!filter) {
        return -1;
    }

    int found = gv_filter_ops_collect_matches(db, filter, out_indices, max_count);
    gv_filter_destroy(filter);

    /* Cap the return value at max_count since the collect helper may have
     * counted more matches than the output buffer can hold. */
    if (found > 0 && (size_t)found > max_count) {
        return (int)max_count;
    }

    return found;
}
