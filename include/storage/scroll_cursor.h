#ifndef GIGAVECTOR_SCROLL_CURSOR_H
#define GIGAVECTOR_SCROLL_CURSOR_H

#include <stddef.h>
#include <stdint.h>

#include "core/types.h"
#include "storage/database.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Stateful scroll cursor for keyset-based pagination over the database.
 *
 * The cursor anchors on the internal insertion index (last_id) to provide
 * stable iteration: new inserts appended after cursor creation do not disturb
 * pages already yielded.  Deletes within the already-visited range are
 * silently skipped.
 */
typedef struct {
    uint64_t last_id;        /**< Keyset anchor: soa_storage index to resume from. */
    uint64_t created_at_us;  /**< Cursor creation timestamp (microseconds since epoch). */
    size_t   page_size;      /**< Maximum results per cursor_next() call. */
    int      exhausted;      /**< Non-zero once all vectors have been returned. */
    uint64_t snapshot_count; /**< soa_storage count captured at cursor open (upper bound). */
} GV_ScrollCursor;

/**
 * @brief Open a scroll cursor positioned at the beginning of the database.
 *
 * The @p query parameter is reserved for future search-guided cursors; pass
 * NULL to scroll all live vectors in insertion order.
 *
 * @param db         Database to scroll; must be non-NULL.
 * @param query      Optional query vector (currently unused, pass NULL).
 * @param page_size  Number of results to return per cursor_next() call; must be > 0.
 * @param out        Output cursor to initialise; must be non-NULL.
 * @return 0 on success, -1 on invalid arguments.
 */
int gv_db_cursor_open(GV_Database *db, const float *query,
                      size_t page_size, GV_ScrollCursor *out);

/**
 * @brief Advance the cursor and fill the next page of results.
 *
 * Callers must pre-allocate @p out with at least cursor->page_size elements.
 * On return, @p *n is set to the number of results written (0 means
 * cursor is exhausted).
 *
 * @param db     Database; must be non-NULL.
 * @param cursor Cursor state; must be non-NULL and previously opened.
 * @param out    Output array of GV_SearchResult; must hold >= cursor->page_size.
 * @param n      Set to the count of results written; must be non-NULL.
 * @return 0 on success (including exhaustion), -1 on error.
 */
int gv_db_cursor_next(GV_Database *db, GV_ScrollCursor *cursor,
                      GV_SearchResult *out, size_t *n);

/**
 * @brief Close and reset a scroll cursor.
 *
 * After this call the cursor is reset to a safe zero state.  The cursor
 * object itself is caller-owned; this function does not free it.
 *
 * @param cursor Cursor to close; safe to call with NULL.
 */
void gv_db_cursor_close(GV_ScrollCursor *cursor);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_SCROLL_CURSOR_H */
