#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_phased_ranking.h"
#include "gigavector/gv_database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 4

/* Helper: create and populate a test database */
static GV_Database *create_test_db(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;

    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.9f, 0.1f, 0.0f, 0.0f};
    float v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v4[] = {0.5f, 0.5f, 0.5f, 0.5f};

    gv_db_add_vector(db, v0, DIM);
    gv_db_add_vector(db, v1, DIM);
    gv_db_add_vector(db, v2, DIM);
    gv_db_add_vector(db, v3, DIM);
    gv_db_add_vector(db, v4, DIM);

    return db;
}

/* Custom rerank callback for testing */
static float test_rerank_callback(size_t index, float current_score,
                                   const void *user_data) {
    (void)index;
    const float *bonus = (const float *)user_data;
    return current_score + (*bonus);
}

/* --- Test: create and destroy pipeline --- */
static int test_create_destroy(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open should succeed");

    GV_Pipeline *pipe = gv_pipeline_create(db);
    ASSERT(pipe != NULL, "gv_pipeline_create should return non-NULL");
    ASSERT(gv_pipeline_phase_count(pipe) == 0, "new pipeline should have 0 phases");

    gv_pipeline_destroy(pipe);
    /* NULL destroy should be safe */
    gv_pipeline_destroy(NULL);

    gv_db_close(db);
    return 0;
}

/* --- Test: add ANN phase --- */
static int test_add_ann_phase(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open should succeed");

    GV_Pipeline *pipe = gv_pipeline_create(db);
    ASSERT(pipe != NULL, "pipeline create should succeed");

    GV_PhaseConfig ann;
    memset(&ann, 0, sizeof(ann));
    ann.type = GV_PHASE_ANN;
    ann.output_k = 100;
    ann.params.ann.distance_type = 1;  /* GV_DISTANCE_COSINE */
    ann.params.ann.ef_search = 0;      /* use default */

    int idx = gv_pipeline_add_phase(pipe, &ann);
    ASSERT(idx >= 0, "add_phase ANN should succeed");
    ASSERT(gv_pipeline_phase_count(pipe) == 1, "phase count should be 1");

    gv_pipeline_destroy(pipe);
    gv_db_close(db);
    return 0;
}

/* --- Test: add multiple phases --- */
static int test_multi_phase(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open should succeed");

    GV_Pipeline *pipe = gv_pipeline_create(db);
    ASSERT(pipe != NULL, "pipeline create should succeed");

    /* Phase 1: ANN */
    GV_PhaseConfig ann;
    memset(&ann, 0, sizeof(ann));
    ann.type = GV_PHASE_ANN;
    ann.output_k = 50;
    ann.params.ann.distance_type = 1;
    int rc = gv_pipeline_add_phase(pipe, &ann);
    ASSERT(rc >= 0, "add ANN phase should succeed");

    /* Phase 2: MMR rerank */
    GV_PhaseConfig mmr;
    memset(&mmr, 0, sizeof(mmr));
    mmr.type = GV_PHASE_RERANK_MMR;
    mmr.output_k = 20;
    mmr.params.mmr.lambda = 0.7f;
    rc = gv_pipeline_add_phase(pipe, &mmr);
    ASSERT(rc >= 0, "add MMR phase should succeed");

    ASSERT(gv_pipeline_phase_count(pipe) == 2, "phase count should be 2");

    gv_pipeline_destroy(pipe);
    gv_db_close(db);
    return 0;
}

/* --- Test: clear phases --- */
static int test_clear_phases(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open should succeed");

    GV_Pipeline *pipe = gv_pipeline_create(db);
    ASSERT(pipe != NULL, "pipeline create should succeed");

    GV_PhaseConfig ann;
    memset(&ann, 0, sizeof(ann));
    ann.type = GV_PHASE_ANN;
    ann.output_k = 50;
    gv_pipeline_add_phase(pipe, &ann);
    ASSERT(gv_pipeline_phase_count(pipe) == 1, "phase count should be 1 before clear");

    gv_pipeline_clear_phases(pipe);
    ASSERT(gv_pipeline_phase_count(pipe) == 0, "phase count should be 0 after clear");

    gv_pipeline_destroy(pipe);
    gv_db_close(db);
    return 0;
}

/* --- Test: execute pipeline --- */
static int test_execute(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_Pipeline *pipe = gv_pipeline_create(db);
    ASSERT(pipe != NULL, "pipeline create should succeed");

    GV_PhaseConfig ann;
    memset(&ann, 0, sizeof(ann));
    ann.type = GV_PHASE_ANN;
    ann.output_k = 10;
    ann.params.ann.distance_type = 0;  /* GV_DISTANCE_EUCLIDEAN */
    gv_pipeline_add_phase(pipe, &ann);

    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_PhasedResult results[5];
    int n = gv_pipeline_execute(pipe, query, DIM, 5, results);
    ASSERT(n >= 1, "execute should return at least 1 result");
    ASSERT(n <= 5, "execute should return at most 5 results");

    /* Results should have valid indices */
    for (int i = 0; i < n; i++) {
        ASSERT(results[i].index < 5, "result index should be within DB range");
        ASSERT(results[i].phase_reached >= 0, "phase_reached should be >= 0");
    }

    gv_pipeline_destroy(pipe);
    gv_db_close(db);
    return 0;
}

/* --- Test: execute with callback rerank --- */
static int test_execute_callback(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_Pipeline *pipe = gv_pipeline_create(db);
    ASSERT(pipe != NULL, "pipeline create should succeed");

    /* Phase 1: ANN */
    GV_PhaseConfig ann;
    memset(&ann, 0, sizeof(ann));
    ann.type = GV_PHASE_ANN;
    ann.output_k = 10;
    ann.params.ann.distance_type = 0;
    gv_pipeline_add_phase(pipe, &ann);

    /* Phase 2: Callback rerank */
    float bonus = 100.0f;
    GV_PhaseConfig cb;
    memset(&cb, 0, sizeof(cb));
    cb.type = GV_PHASE_RERANK_CALLBACK;
    cb.output_k = 3;
    cb.params.callback.fn = test_rerank_callback;
    cb.params.callback.data = &bonus;
    gv_pipeline_add_phase(pipe, &cb);

    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_PhasedResult results[3];
    int n = gv_pipeline_execute(pipe, query, DIM, 3, results);
    ASSERT(n >= 1, "execute with callback should return results");

    gv_pipeline_destroy(pipe);
    gv_db_close(db);
    return 0;
}

/* --- Test: pipeline stats --- */
static int test_stats(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_Pipeline *pipe = gv_pipeline_create(db);
    ASSERT(pipe != NULL, "pipeline create should succeed");

    GV_PhaseConfig ann;
    memset(&ann, 0, sizeof(ann));
    ann.type = GV_PHASE_ANN;
    ann.output_k = 10;
    ann.params.ann.distance_type = 0;
    gv_pipeline_add_phase(pipe, &ann);

    /* Execute first to populate stats */
    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_PhasedResult results[5];
    gv_pipeline_execute(pipe, query, DIM, 5, results);

    GV_PipelineStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = gv_pipeline_get_stats(pipe, &stats);
    ASSERT(rc == 0, "get_stats should succeed");
    ASSERT(stats.phase_count == 1, "stats should report 1 phase");
    ASSERT(stats.total_latency_ms >= 0.0, "total latency should be non-negative");

    gv_pipeline_free_stats(&stats);

    gv_pipeline_destroy(pipe);
    gv_db_close(db);
    return 0;
}

/* --- Test: free_stats with NULL fields --- */
static int test_free_stats_null(void) {
    GV_PipelineStats stats;
    memset(&stats, 0, sizeof(stats));
    /* Should be safe with NULL internal arrays */
    gv_pipeline_free_stats(&stats);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing pipeline create/destroy...", test_create_destroy},
        {"Testing pipeline add ANN phase...", test_add_ann_phase},
        {"Testing pipeline multi-phase...", test_multi_phase},
        {"Testing pipeline clear phases...", test_clear_phases},
        {"Testing pipeline execute...", test_execute},
        {"Testing pipeline execute callback...", test_execute_callback},
        {"Testing pipeline stats...", test_stats},
        {"Testing pipeline free_stats null...", test_free_stats_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("%s", tests[i].name);
        if (tests[i].fn() == 0) { printf(" [OK]\n"); passed++; }
        else { printf(" [FAIL]\n"); }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
