/**
 * @file scroll_cursor.c
 * @brief Streaming result cursor for keyset-based pagination over GigaVector databases.
 *
 * Implementation notes
 * --------------------
 * The cursor anchors on the raw soa_storage array index so that new inserts
 * appended after cursor creation fall *above* the snapshot_count horizon and
 * are therefore never returned.  Deleted slots are detected with
 * soa_storage_is_deleted() and silently skipped.  No copy of vector data is
 * made: GV_SearchResult.vector is set to NULL and the caller uses the id
 * field together with GV_ScrollResult-style direct data access when needed.
 * Distance is set to 0.0 because a scroll cursor is not a similarity search.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "storage/scroll_cursor.h"
#include "storage/soa_storage.h"
#include "storage/db_internal.h"
#include "core/types.h"

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

static uint64_t sc_now_us(void) {
    return db_get_time_us();
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

int gv_db_cursor_open(GV_Database *db, const float *query,
                      size_t page_size, GV_ScrollCursor *out)
{
    (void)query; /* reserved for future search-guided cursors */

    if (db == NULL || out == NULL || page_size == 0) {
        return -1;
    }

    pthread_rwlock_rdlock(&db->rwlock);
    uint64_t snap = (db->soa_storage != NULL) ? (uint64_t)db->soa_storage->count : 0u;
    pthread_rwlock_unlock(&db->rwlock);

    out->last_id        = 0;
    out->created_at_us  = sc_now_us();
    out->page_size      = page_size;
    out->exhausted      = 0;
    out->snapshot_count = snap;

    return 0;
}

int gv_db_cursor_next(GV_Database *db, GV_ScrollCursor *cursor,
                      GV_SearchResult *out, size_t *n)
{
    if (db == NULL || cursor == NULL || out == NULL || n == NULL) {
        return -1;
    }

    *n = 0;

    if (cursor->exhausted) {
        return 0;
    }

    pthread_rwlock_rdlock(&db->rwlock);

    GV_SoAStorage *soa = db->soa_storage;

    if (soa == NULL) {
        pthread_rwlock_unlock(&db->rwlock);
        cursor->exhausted = 1;
        return 0;
    }

    /* Honour the snapshot horizon captured at open time so that inserts
     * after cursor creation are not visible.  The current live count may
     * exceed snapshot_count; we stop at the lower of the two. */
    size_t total = soa->count;
    if (cursor->snapshot_count < (uint64_t)total) {
        total = (size_t)cursor->snapshot_count;
    }

    size_t found = 0;
    size_t i     = (size_t)cursor->last_id;

    while (i < total && found < cursor->page_size) {
        if (soa_storage_is_deleted(soa, i) == 1) {
            i++;
            continue;
        }

        /* Fill a lightweight result: no heap allocation, caller owns array. */
        out[found].id            = i;
        out[found].distance      = 0.0f;
        out[found].is_sparse     = 0;
        out[found].sparse_vector = NULL;
        out[found].vector        = NULL; /* Caller may use id to look up data. */

        found++;
        i++;
    }

    cursor->last_id = (uint64_t)i;
    *n = found;

    if (i >= total) {
        cursor->exhausted = 1;
    }

    pthread_rwlock_unlock(&db->rwlock);
    return 0;
}

void gv_db_cursor_close(GV_ScrollCursor *cursor)
{
    if (cursor == NULL) {
        return;
    }
    memset(cursor, 0, sizeof(*cursor));
}
