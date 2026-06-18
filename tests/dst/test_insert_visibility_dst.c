/**
 * @file test_insert_visibility_dst.c
 * @brief DST oracle: inserts are visible to search immediately after WAL append.
 *
 * Invariant: after db_add_vector (WAL fsync), the next oversampled db_search
 * must include the inserted vector in its candidate set.
 */

#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "storage/database.h"
#include "core/types.h"
#include "../test_tmp.h"
#include "dst_harness.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

#define DIM 8

static int results_contain_id(const GV_SearchResult *results, int n, size_t target_id) {
    for (int i = 0; i < n; ++i) {
        if (results[i].id == target_id) {
            return 1;
        }
    }
    return 0;
}

static int run_visibility_oracle(GV_Database *db, GV_IndexType index_type, GV_DstRng *rng,
                                 size_t iters) {
    (void)index_type;

    for (size_t i = 0; i < iters; ++i) {
        float vec[DIM];
        for (size_t d = 0; d < DIM; ++d) {
            vec[d] = gv_dst_rng_float(rng);
        }

        size_t inserted_id = db->count;
        ASSERT(db_add_vector(db, vec, DIM) == 0, "insert vector");

        size_t oversample_k = (inserted_id + 1u) * 10u;
        if (oversample_k < 10u) {
            oversample_k = 10u;
        }
        if (oversample_k > db->count) {
            oversample_k = db->count;
        }
        if (oversample_k == 0) {
            continue;
        }

        GV_SearchResult *hits = (GV_SearchResult *)gv_calloc(oversample_k, sizeof(GV_SearchResult));
        ASSERT(hits != NULL, "alloc search buffer");

        int found = db_search(db, vec, oversample_k, hits, GV_DISTANCE_EUCLIDEAN);
        ASSERT(found > 0, "search returns candidates after insert");
        ASSERT(results_contain_id(hits, found, inserted_id),
               "inserted vector visible in oversampled search");

        gv_free(hits);
    }
    return 0;
}

static int test_insert_visibility_flat_memory(void) {
    GV_DstRng rng = gv_dst_rng_seed(gv_dst_seed_from_env());
    size_t iters = gv_dst_iters_from_env(25);

    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "open FLAT db");
    int rc = run_visibility_oracle(db, GV_INDEX_TYPE_FLAT, &rng, iters);
    db_close(db);
    return rc;
}

static int test_insert_visibility_hnsw_wal(void) {
    char db_path[512];
    if (gv_test_make_temp_path(db_path, sizeof(db_path), "gv_vis_dst", ".db") != 0) {
        return 0;
    }

    GV_DstRng rng = gv_dst_rng_seed(gv_dst_seed_from_env() ^ 0x9e3779b97f4a7c15ULL);
    size_t iters = gv_dst_iters_from_env(25);

    GV_Database *db = db_open(db_path, DIM, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "open HNSW db with WAL");
    ASSERT(db->wal != NULL, "WAL enabled for file-backed db");

    int rc = run_visibility_oracle(db, GV_INDEX_TYPE_HNSW, &rng, iters);
    db_close(db);
    return rc;
}

int main(void) {
    uint64_t seed = gv_dst_seed_from_env();
    size_t iters = gv_dst_iters_from_env(25);
    fprintf(stderr, "DST insert visibility: seed=%llu iters=%zu\n",
            (unsigned long long)seed, iters);

    if (test_insert_visibility_flat_memory() != 0) return 1;
    if (test_insert_visibility_hnsw_wal() != 0) return 1;
    return 0;
}
