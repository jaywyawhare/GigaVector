/**
 * test_ivf_retrain.c
 *
 * Tests for incremental IVF retraining.
 *
 * Test plan:
 *  1. Insert 1000 vectors into an IVFFLAT database.
 *  2. Enable retrain config.
 *  3. Insert 500 more vectors.
 *  4. Manually trigger retrain via gv_db_trigger_retrain().
 *  5. Verify retrain completes (retrain_running == 0).
 *  6. Run a search and verify results still come back correctly.
 *  7. Test drift detection: verify ivf_retrain_check_drift() returns a
 *     positive finite value for a trained index.
 *  8. Test gv_db_retrain_status() API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gigavector.h"

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return -1; \
        } \
    } while (0)

/* Generate a pseudo-random float in [0, 1) */
static float rand_float(unsigned int *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / (float)(1u << 24);
}

/* ------------------------------------------------------------------ */

static int test_retrain_basic(void) {
    const size_t dim = 16;
    const size_t nlist = 4;
    const size_t ntrain = 200;   /* training vectors */
    const size_t ninsert = 1000; /* first batch */
    const size_t nmore = 500;    /* second batch */

    GV_IVFFlatConfig cfg = {
        .nlist       = nlist,
        .nprobe      = nlist,  /* probe all lists for exact recall */
        .train_iters = 10,
        .use_cosine  = 0,
    };

    GV_Database *db = db_open_with_ivfflat_config(NULL, dim,
                                                   GV_INDEX_TYPE_IVFFLAT, &cfg);
    ASSERT(db != NULL);

    /* Train the index */
    unsigned int seed = 42;
    float *train_data = (float *)malloc(ntrain * dim * sizeof(float));
    ASSERT(train_data != NULL);
    for (size_t i = 0; i < ntrain * dim; i++) {
        train_data[i] = rand_float(&seed);
    }
    int rc = db_ivfflat_train(db, train_data, ntrain, dim);
    ASSERT(rc == 0);
    free(train_data);

    /* Insert first batch */
    for (size_t i = 0; i < ninsert; i++) {
        float vec[16];
        for (size_t d = 0; d < dim; d++) {
            vec[d] = rand_float(&seed);
        }
        rc = db_add_vector(db, vec, dim);
        ASSERT(rc == 0);
    }
    ASSERT(db->count == ninsert);

    /* Enable retrain config (low threshold so test is fast) */
    gv_db_set_retrain_config(db, 0.15f, 100 /* low min_new_vectors for test */);
    ASSERT(db->retrain_enabled == 1);

    /* Insert second batch */
    for (size_t i = 0; i < nmore; i++) {
        float vec[16];
        for (size_t d = 0; d < dim; d++) {
            /* Shift distribution to create drift */
            vec[d] = rand_float(&seed) + 2.0f;
        }
        rc = db_add_vector(db, vec, dim);
        ASSERT(rc == 0);
    }
    ASSERT(db->count == ninsert + nmore);

    /* Check drift is computable */
    float drift = ivf_retrain_check_drift(db);
    ASSERT(drift >= 0.0f);
    ASSERT(isfinite(drift));

    /* Manually trigger synchronous retrain */
    rc = gv_db_trigger_retrain(db);
    ASSERT(rc == 0);

    /* Verify retrain completed */
    int is_running = 1;
    float last_drift = 0.0f;
    gv_db_retrain_status(db, &is_running, &last_drift);
    ASSERT(is_running == 0);

    /* The insert counter should have been reset to 0 by the retrain */
    ASSERT(db->inserts_since_retrain == 0);

    /* Search should still return results */
    float query[16];
    for (size_t d = 0; d < dim; d++) {
        query[d] = rand_float(&seed);
    }

    GV_SearchResult results[10];
    memset(results, 0, sizeof(results));
    int found = db_search(db, query, 10, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0);
    ASSERT(found <= 10);

    /* Results should have valid distances */
    for (int i = 0; i < found; i++) {
        ASSERT(results[i].distance >= 0.0f);
        ASSERT(isfinite(results[i].distance));
    }

    /* Cleanup */
    for (int i = 0; i < found; i++) {
        if (results[i].vector) vector_destroy(results[i].vector);
    }

    db_close(db);
    return 0;
}

/* ------------------------------------------------------------------ */

static int test_retrain_status_api(void) {
    const size_t dim = 8;
    GV_IVFFlatConfig cfg = {
        .nlist       = 2,
        .nprobe      = 2,
        .train_iters = 5,
        .use_cosine  = 0,
    };

    GV_Database *db = db_open_with_ivfflat_config(NULL, dim,
                                                   GV_INDEX_TYPE_IVFFLAT, &cfg);
    ASSERT(db != NULL);

    /* Before any configuration: default values */
    int is_running = -1;
    float last_drift = -1.0f;
    gv_db_retrain_status(db, &is_running, &last_drift);
    ASSERT(is_running == 0);
    ASSERT(last_drift >= 0.0f); /* default 1.0 */

    /* Train and insert minimum data for retrain */
    unsigned int seed = 12345;
    float train_data[20 * 8];
    for (size_t i = 0; i < 20 * 8; i++) {
        train_data[i] = rand_float(&seed);
    }
    ASSERT(db_ivfflat_train(db, train_data, 20, dim) == 0);

    for (size_t i = 0; i < 10; i++) {
        float vec[8];
        for (size_t d = 0; d < dim; d++) vec[d] = rand_float(&seed);
        ASSERT(db_add_vector(db, vec, dim) == 0);
    }

    /* Trigger retrain synchronously */
    gv_db_set_retrain_config(db, 0.0f, 1);
    ASSERT(gv_db_trigger_retrain(db) == 0);

    gv_db_retrain_status(db, &is_running, &last_drift);
    ASSERT(is_running == 0);

    db_close(db);
    return 0;
}

/* ------------------------------------------------------------------ */

static int test_retrain_non_ivf_type(void) {
    /* For non-IVF types, drift check should return -1.0 (no-op) */
    GV_Database *db = db_open(NULL, 8, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL);

    float drift = ivf_retrain_check_drift(db);
    ASSERT(drift < 0.0f); /* -1.0f expected */

    /* trigger should fail gracefully: for non-IVF index types the retrain
     * switch falls through to the default case which returns -1 (no-op). */
    int rc = gv_db_trigger_retrain(db);
    ASSERT(rc == -1);

    db_close(db);
    return 0;
}

/* ------------------------------------------------------------------ */

int main(void) {
    int rc = 0;
    rc |= test_retrain_basic();
    rc |= test_retrain_status_api();
    rc |= test_retrain_non_ivf_type();
    if (rc == 0) {
        printf("All ivf_retrain tests PASSED.\n");
    }
    return rc;
}
