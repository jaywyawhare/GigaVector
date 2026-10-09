/*
 * Cross-index non-euclidean metric coverage.
 *
 * The HNSW cosine/dot stack-overflow (fixed in the hnsw_dist_batch4 work)
 * shipped because no test ever built a non-euclidean index at a scale where
 * nodes accumulate many neighbours. This sweeps every in-memory index type
 * that supports an unkeyed build (no explicit train step) under COSINE and
 * DOT_PRODUCT at 400 vectors, so a future metric-specific memory bug in any of
 * them is caught by the ASan CI job instead of in production. IVF/PQ types are
 * excluded here: they require an explicit train() before search (db_search
 * returns -1 otherwise), which the e2e profiler and their own tests cover.
 */
#include <stdio.h>
#include <stdlib.h>

#include "gigavector.h"
#include "storage/database.h"

#define ASSERT(cond, msg)                       \
    do {                                        \
        if (!(cond)) {                          \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;                          \
        }                                       \
    } while (0)

static int sweep_one(const char *name, GV_IndexType type) {
    const GV_DistanceType metrics[2] = { GV_DISTANCE_COSINE, GV_DISTANCE_DOT_PRODUCT };
    const size_t dim = 48, n = 400;

    for (int mi = 0; mi < 2; ++mi) {
        GV_Database *db = db_open(NULL, dim, type);
        if (db == NULL) {
            return 0; /* type unavailable in this build - skip, not a failure */
        }

        float v[48];
        srand(7);
        for (size_t k = 0; k < n; ++k) {
            for (size_t d = 0; d < dim; ++d) v[d] = (float)rand() / (float)RAND_MAX - 0.5f;
            (void)db_add_vector(db, v, dim);
        }

        GV_SearchResult res[10];
        int got = 0;
        for (size_t q = 0; q < 20; ++q) {
            for (size_t d = 0; d < dim; ++d) v[d] = (float)rand() / (float)RAND_MAX - 0.5f;
            int g = db_search(db, v, 10, res, metrics[mi]);
            if (g > 0) { got += g; gv_search_results_free(res, (size_t)g); }
        }
        ASSERT(got > 0, name); /* every swept type must return results */

        db_close(db);
    }
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= sweep_one("kdtree non-euclidean", GV_INDEX_TYPE_KDTREE);
    rc |= sweep_one("hnsw non-euclidean",   GV_INDEX_TYPE_HNSW);
    rc |= sweep_one("flat non-euclidean",   GV_INDEX_TYPE_FLAT);
    rc |= sweep_one("lsh non-euclidean",    GV_INDEX_TYPE_LSH);
    rc |= sweep_one("rabitq non-euclidean", GV_INDEX_TYPE_RABITQ);
    if (rc == 0) printf("test_index_metrics: all non-euclidean sweeps passed\n");
    return rc;
}
