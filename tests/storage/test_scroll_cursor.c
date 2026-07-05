/**
 * @file test_scroll_cursor.c
 * @brief Tests for the GV_ScrollCursor keyset-based pagination API.
 *
 * Test plan
 * ---------
 * 1. Insert 500 vectors, open a cursor with page_size=50.
 * 2. Call cursor_next() 10 times; verify each page returns exactly 50
 *    unique vectors and the combined total equals 500.
 * 3. Insert 100 more vectors mid-iteration and confirm no duplicates
 *    appear in subsequent pages (snapshot isolation).
 * 4. Verify the cursor reports exhausted after all 500 original vectors.
 * 5. Verify cursor_close() resets the state safely.
 * 6. Verify cursor_open() with invalid args returns -1.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/database.h"
#include "storage/scroll_cursor.h"
#include "core/types.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
            return -1; \
        } \
    } while (0)

#define DIM       4
#define N_INIT  500
#define PAGE_SZ  50

/* Build a trivial distinct vector: vec[i] = (float)id for all dims. */
static void make_vec(float *buf, size_t id) {
    for (int d = 0; d < DIM; d++) {
        buf[d] = (float)id;
    }
}

/* -------------------------------------------------------------------------
 * Test: basic scroll exhausts all N_INIT vectors with no duplicates
 * ---------------------------------------------------------------------- */
static int test_full_scroll(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    float vec[DIM];
    for (size_t i = 0; i < N_INIT; i++) {
        make_vec(vec, i);
        ASSERT(db_add_vector(db, vec, DIM) == 0, "db_add_vector");
    }
    ASSERT((size_t)db->count == N_INIT, "count after inserts");

    GV_ScrollCursor cursor;
    ASSERT(gv_db_cursor_open(db, NULL, PAGE_SZ, &cursor) == 0, "cursor_open");
    ASSERT(cursor.exhausted == 0, "not exhausted initially");
    ASSERT(cursor.page_size == PAGE_SZ, "page_size stored");
    ASSERT(cursor.snapshot_count == N_INIT, "snapshot_count == N_INIT");

    /* Track seen IDs to detect duplicates. */
    int seen[N_INIT];
    memset(seen, 0, sizeof(seen));

    GV_SearchResult *page = (GV_SearchResult *)malloc(PAGE_SZ * sizeof(GV_SearchResult));
    ASSERT(page != NULL, "alloc page buffer");

    size_t total = 0;
    int pages = 0;

    while (!cursor.exhausted) {
        size_t n = 0;
        ASSERT(gv_db_cursor_next(db, &cursor, page, &n) == 0, "cursor_next");
        ASSERT(n <= PAGE_SZ, "page count <= page_size");

        for (size_t j = 0; j < n; j++) {
            size_t id = page[j].id;
            ASSERT(id < N_INIT, "id in valid range");
            ASSERT(seen[id] == 0, "no duplicate id");
            seen[id] = 1;
        }

        total += n;
        pages++;

        /* Safety: prevent infinite loop in buggy implementations. */
        ASSERT(pages <= N_INIT + 1, "too many pages");
    }

    ASSERT(total == N_INIT, "total == N_INIT");
    ASSERT(pages == N_INIT / PAGE_SZ, "correct page count");

    /* Calling cursor_next after exhaustion should return 0 results safely. */
    size_t n_after = 0;
    ASSERT(gv_db_cursor_next(db, &cursor, page, &n_after) == 0, "cursor_next after exhaustion");
    ASSERT(n_after == 0, "zero results after exhaustion");

    free(page);
    gv_db_cursor_close(&cursor);
    ASSERT(cursor.page_size == 0, "cursor zeroed after close");

    db_close(db);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: inserts after cursor_open do not appear in pages (snapshot isolation)
 * ---------------------------------------------------------------------- */
static int test_snapshot_isolation(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    float vec[DIM];
    for (size_t i = 0; i < N_INIT; i++) {
        make_vec(vec, i);
        ASSERT(db_add_vector(db, vec, DIM) == 0, "insert pre-cursor");
    }

    GV_ScrollCursor cursor;
    ASSERT(gv_db_cursor_open(db, NULL, PAGE_SZ, &cursor) == 0, "cursor_open");

    /* Insert extra vectors AFTER cursor is opened. */
    for (size_t i = N_INIT; i < N_INIT + 100; i++) {
        make_vec(vec, i);
        ASSERT(db_add_vector(db, vec, DIM) == 0, "insert post-cursor");
    }

    /* Drain the cursor; ids must all be < N_INIT (pre-snapshot). */
    GV_SearchResult *page = (GV_SearchResult *)malloc(PAGE_SZ * sizeof(GV_SearchResult));
    ASSERT(page != NULL, "alloc page buffer");

    size_t total = 0;
    int seen[N_INIT];
    memset(seen, 0, sizeof(seen));

    while (!cursor.exhausted) {
        size_t n = 0;
        ASSERT(gv_db_cursor_next(db, &cursor, page, &n) == 0, "cursor_next");
        for (size_t j = 0; j < n; j++) {
            size_t id = page[j].id;
            /* Must be within the original snapshot. */
            ASSERT(id < N_INIT, "snapshot isolation: id < N_INIT");
            ASSERT(seen[id] == 0, "no duplicate id");
            seen[id] = 1;
        }
        total += n;
    }

    ASSERT(total == N_INIT, "snapshot total == N_INIT");

    free(page);
    gv_db_cursor_close(&cursor);
    db_close(db);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: invalid argument handling
 * ---------------------------------------------------------------------- */
static int test_invalid_args(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    GV_ScrollCursor cursor;

    ASSERT(gv_db_cursor_open(NULL, NULL, PAGE_SZ, &cursor) == -1, "NULL db");
    ASSERT(gv_db_cursor_open(db, NULL, 0, &cursor) == -1, "zero page_size");
    ASSERT(gv_db_cursor_open(db, NULL, PAGE_SZ, NULL) == -1, "NULL out");

    ASSERT(gv_db_cursor_open(db, NULL, PAGE_SZ, &cursor) == 0, "valid open");

    GV_SearchResult res;
    size_t n = 0;
    ASSERT(gv_db_cursor_next(NULL, &cursor, &res, &n) == -1, "NULL db");
    ASSERT(gv_db_cursor_next(db, NULL, &res, &n) == -1, "NULL cursor");
    ASSERT(gv_db_cursor_next(db, &cursor, NULL, &n) == -1, "NULL out");
    ASSERT(gv_db_cursor_next(db, &cursor, &res, NULL) == -1, "NULL n");

    gv_db_cursor_close(NULL); /* must not crash */
    gv_db_cursor_close(&cursor);

    db_close(db);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: empty database returns zero results on first cursor_next
 * ---------------------------------------------------------------------- */
static int test_empty_db(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    GV_ScrollCursor cursor;
    ASSERT(gv_db_cursor_open(db, NULL, PAGE_SZ, &cursor) == 0, "cursor_open on empty db");

    GV_SearchResult page[PAGE_SZ];
    size_t n = 999;
    ASSERT(gv_db_cursor_next(db, &cursor, page, &n) == 0, "cursor_next on empty db");
    ASSERT(n == 0, "zero results on empty db");
    ASSERT(cursor.exhausted != 0, "exhausted on empty db");

    gv_db_cursor_close(&cursor);
    db_close(db);
    return 0;
}

/* -------------------------------------------------------------------------
 * Test: generation counter increments on mutations
 * ---------------------------------------------------------------------- */
static int test_generation_counter(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");
    ASSERT(db->generation == 0, "initial generation == 0");

    float vec[DIM];
    make_vec(vec, 0);
    ASSERT(db_add_vector(db, vec, DIM) == 0, "insert");
    ASSERT(db->generation == 1, "generation after insert");

    ASSERT(db_delete_vector_by_index(db, 0) == 0, "delete");
    ASSERT(db->generation == 2, "generation after delete");

    db_close(db);
    return 0;
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */
int main(void) {
    int failed = 0;

#define RUN(fn) \
    do { \
        printf("  %-45s", #fn); \
        if ((fn)() == 0) { \
            printf("PASS\n"); \
        } else { \
            printf("FAIL\n"); \
            failed++; \
        } \
    } while (0)

    printf("scroll_cursor tests\n");
    RUN(test_full_scroll);
    RUN(test_snapshot_isolation);
    RUN(test_invalid_args);
    RUN(test_empty_db);
    RUN(test_generation_counter);

    if (failed == 0) {
        printf("All tests passed.\n");
        return 0;
    } else {
        printf("%d test(s) FAILED.\n", failed);
        return 1;
    }
}
