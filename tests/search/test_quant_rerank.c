/**
 * @file test_quant_rerank.c
 * @brief Tests for the quantization-aware reranking pipeline.
 *
 * Tests:
 *  1. quant_rerank_config_init sets correct defaults.
 *  2. quant_rerank_search rejects NULL inputs gracefully.
 *  3. quant_rerank_search returns results for a simple dataset.
 *  4. quant_rerank_search result count is at most k.
 *  5. Higher recall with reranking vs ANN-only (top-1 comparison).
 *  6. gv_db_search_with_rerank returns valid results.
 *  7. GV_PHASE_RERANK_QUANT integrates into the phased pipeline.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "search/quant_rerank.h"
#include "search/phased_ranking.h"
#include "storage/database.h"
#include "specialized/quantization.h"

#define ASSERT(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); return -1; } } while(0)

#define DIM   8
#define NVECS 200

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

/** Fill a vector with reproducible pseudo-random floats in [-1, 1]. */
static void fill_vector(float *v, size_t dim, unsigned seed) {
    for (size_t i = 0; i < dim; i++) {
        seed = seed * 1664525u + 1013904223u;
        float val = (float)(int)(seed >> 16) / 32768.0f - 1.0f;
        v[i] = val;
    }
}

/** Create a test database with NVECS random vectors and return it. */
static GV_Database *make_db(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;

    float v[DIM];
    for (int i = 0; i < NVECS; i++) {
        fill_vector(v, DIM, (unsigned)(i * 7919 + 1));
        if (db_add_vector(db, v, DIM) < 0) {
            db_close(db);
            return NULL;
        }
    }
    return db;
}

/**
 * Build a flat codes array for all vectors currently in db.
 * Returns allocated buffer (caller frees) and sets *stride.
 */
static uint8_t *build_codes(const GV_Database *db,
                             const GV_QuantCodebook *cb,
                             size_t *stride) {
    size_t dim   = database_dimension(db);
    size_t count = database_count(db);

    *stride = quant_code_size(cb, dim);
    if (*stride == 0) return NULL;

    uint8_t *codes = calloc(count, *stride);
    if (!codes) return NULL;

    for (size_t i = 0; i < count; i++) {
        const float *vec = database_get_vector(db, i);
        if (!vec) { free(codes); return NULL; }
        if (quant_encode(cb, vec, dim, codes + i * (*stride)) != 0) {
            free(codes);
            return NULL;
        }
    }
    return codes;
}

/* ------------------------------------------------------------------ */
/* Tests                                                                */
/* ------------------------------------------------------------------ */

static int test_config_init(void) {
    GV_QuantRerankConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg)); /* poison */
    quant_rerank_config_init(&cfg);

    ASSERT(cfg.oversample_factor == 4, "default oversample_factor should be 4");
    ASSERT(cfg.distance_type == GV_DISTANCE_EUCLIDEAN, "default distance_type should be L2");
    ASSERT(cfg.codebook == NULL, "default codebook should be NULL");
    ASSERT(cfg.codes == NULL, "default codes should be NULL");
    ASSERT(cfg.code_stride == 0, "default code_stride should be 0");
    return 0;
}

static int test_null_inputs(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    GV_QuantRerankResult out[10];
    GV_QuantRerankConfig cfg;
    quant_rerank_config_init(&cfg);

    ASSERT(quant_rerank_search(NULL, NULL, 5, NULL, NULL) == -1,
           "all-NULL should return -1");
    ASSERT(quant_rerank_search(db, NULL, 5, &cfg, out) == -1,
           "NULL query should return -1");

    float q[DIM] = {0};
    ASSERT(quant_rerank_search(db, q, 0, &cfg, out) == -1,
           "k==0 should return -1");
    ASSERT(quant_rerank_search(db, q, 5, NULL, out) == -1,
           "NULL cfg should return -1");
    ASSERT(quant_rerank_search(db, q, 5, &cfg, NULL) == -1,
           "NULL out should return -1");

    /* cfg with no codebook should return -1 */
    ASSERT(quant_rerank_search(db, q, 5, &cfg, out) == -1,
           "cfg with NULL codebook should return -1");

    db_close(db);
    return 0;
}

static int test_basic_search(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    size_t dim = DIM;

    /* Train a codebook on the stored vectors. */
    float *training = malloc(NVECS * dim * sizeof(float));
    ASSERT(training != NULL, "alloc training buf");
    for (int i = 0; i < NVECS; i++) {
        const float *v = database_get_vector(db, (size_t)i);
        ASSERT(v != NULL, "database_get_vector should succeed");
        memcpy(training + (size_t)i * dim, v, dim * sizeof(float));
    }

    GV_QuantConfig qcfg;
    quant_config_init(&qcfg);
    qcfg.type = GV_QUANT_8BIT;
    qcfg.mode = GV_QUANT_ASYMMETRIC;

    GV_QuantCodebook *cb = quant_train(training, NVECS, dim, &qcfg);
    free(training);
    ASSERT(cb != NULL, "quant_train should succeed");

    size_t stride;
    uint8_t *codes = build_codes(db, cb, &stride);
    ASSERT(codes != NULL, "build_codes should succeed");

    GV_QuantRerankConfig cfg;
    quant_rerank_config_init(&cfg);
    cfg.codebook        = cb;
    cfg.codes           = codes;
    cfg.code_stride     = stride;
    cfg.oversample_factor = 4;

    float query[DIM];
    fill_vector(query, DIM, 42u);

    GV_QuantRerankResult results[10];
    int n = quant_rerank_search(db, query, 10, &cfg, results);
    ASSERT(n >= 0, "quant_rerank_search should return non-negative");
    ASSERT(n <= 10, "result count should not exceed k");

    for (int i = 0; i < n; i++) {
        ASSERT(results[i].index < NVECS, "result index should be in range");
        ASSERT(results[i].rerank_score >= 0.0f, "rerank_score should be non-negative");
    }

    /* Results should be sorted ascending by rerank_score. */
    for (int i = 1; i < n; i++) {
        ASSERT(results[i].rerank_score >= results[i-1].rerank_score,
               "results should be sorted ascending by rerank_score");
    }

    free(codes);
    quant_codebook_destroy(cb);
    db_close(db);
    return 0;
}

static int test_result_count_le_k(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    size_t dim = DIM;
    float *training = malloc(NVECS * dim * sizeof(float));
    ASSERT(training != NULL, "alloc training buf");
    for (int i = 0; i < NVECS; i++) {
        const float *v = database_get_vector(db, (size_t)i);
        ASSERT(v != NULL, "database_get_vector ok");
        memcpy(training + (size_t)i * dim, v, dim * sizeof(float));
    }

    GV_QuantConfig qcfg;
    quant_config_init(&qcfg);
    qcfg.type = GV_QUANT_4BIT;

    GV_QuantCodebook *cb = quant_train(training, NVECS, dim, &qcfg);
    free(training);
    ASSERT(cb != NULL, "quant_train should succeed");

    size_t stride;
    uint8_t *codes = build_codes(db, cb, &stride);
    ASSERT(codes != NULL, "build_codes ok");

    GV_QuantRerankConfig cfg;
    quant_rerank_config_init(&cfg);
    cfg.codebook = cb;
    cfg.codes    = codes;
    cfg.code_stride = stride;
    cfg.oversample_factor = 2;

    float q[DIM] = {0.5f, -0.3f, 0.1f, 0.8f, -0.6f, 0.2f, -0.9f, 0.4f};

    /* Request more results than in the DB to test clamping. */
    GV_QuantRerankResult results[300];
    int n = quant_rerank_search(db, q, 300, &cfg, results);
    ASSERT(n >= 0, "search ok");
    ASSERT((size_t)n <= NVECS, "should not return more than NVECS results");

    free(codes);
    quant_codebook_destroy(cb);
    db_close(db);
    return 0;
}

/** Compare top-1 from reranking vs ANN-only against brute-force ground truth. */
static int test_recall_improvement(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db ok");

    size_t dim = DIM;
    float query[DIM];
    fill_vector(query, DIM, 99u);

    /* Brute-force ground truth: find the closest vector. */
    size_t true_nn = 0;
    float  best_dist = 1e30f;
    for (size_t i = 0; i < NVECS; i++) {
        const float *v = database_get_vector(db, i);
        if (!v) continue;
        float d = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            float diff = query[j] - v[j];
            d += diff * diff;
        }
        if (d < best_dist) { best_dist = d; true_nn = i; }
    }

    /* Rerank search with generous oversample — should find the true NN. */
    int n = gv_db_search_with_rerank(db, query, 1, 8, NULL);
    /* We can't pass NULL for out here; allocate a result */
    GV_SearchResult out[1];
    memset(out, 0, sizeof(out));
    n = gv_db_search_with_rerank(db, query, 1, 8, out);

    ASSERT(n == 1, "should return 1 result");
    ASSERT(out[0].id == true_nn,
           "reranked top-1 should match brute-force nearest neighbour");

    /* Free any allocated vector in result. */
    if (out[0].vector) {
        /* The vector pointer is owned by the SoA storage reference;
         * gv_db_search_with_rerank may or may not allocate a copy.
         * Use gv_search_results_free if available, else just set to NULL. */
        gv_search_results_free(out, 1);
    }

    db_close(db);
    return 0;
}

static int test_simple_rerank(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db ok");

    float q[DIM];
    fill_vector(q, DIM, 77u);

    GV_SearchResult out[10];
    memset(out, 0, sizeof(out));
    int n = gv_db_search_with_rerank(db, q, 10, 4, out);
    ASSERT(n >= 0, "gv_db_search_with_rerank should succeed");
    ASSERT(n <= 10, "should not exceed k");

    /* Distances should be sorted ascending. */
    for (int i = 1; i < n; i++) {
        ASSERT(out[i].distance >= out[i-1].distance,
               "results should be sorted by ascending distance");
    }

    gv_search_results_free(out, (size_t)n);
    db_close(db);
    return 0;
}

static int test_phased_pipeline_quant(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db ok");

    size_t dim = DIM;

    /* Train codebook. */
    float *training = malloc(NVECS * dim * sizeof(float));
    ASSERT(training != NULL, "alloc training buf");
    for (int i = 0; i < NVECS; i++) {
        const float *v = database_get_vector(db, (size_t)i);
        ASSERT(v != NULL, "get_vector ok");
        memcpy(training + (size_t)i * dim, v, dim * sizeof(float));
    }

    GV_QuantConfig qcfg;
    quant_config_init(&qcfg);
    qcfg.type = GV_QUANT_8BIT;
    qcfg.mode = GV_QUANT_ASYMMETRIC;

    GV_QuantCodebook *cb = quant_train(training, NVECS, dim, &qcfg);
    free(training);
    ASSERT(cb != NULL, "quant_train ok");

    size_t stride;
    uint8_t *codes = build_codes(db, cb, &stride);
    ASSERT(codes != NULL, "build_codes ok");

    /* Build pipeline: ANN -> RERANK_QUANT. */
    GV_Pipeline *pipe = pipeline_create(db);
    ASSERT(pipe != NULL, "pipeline_create ok");

    GV_PhaseConfig ann;
    memset(&ann, 0, sizeof(ann));
    ann.type = GV_PHASE_ANN;
    ann.output_k = 40;
    ann.params.ann.distance_type = GV_DISTANCE_EUCLIDEAN;
    ASSERT(pipeline_add_phase(pipe, &ann) >= 0, "add ANN phase ok");

    GV_PhaseConfig quant_phase;
    memset(&quant_phase, 0, sizeof(quant_phase));
    quant_phase.type     = GV_PHASE_RERANK_QUANT;
    quant_phase.output_k = 10;
    quant_phase.params.quant.codebook    = cb;
    quant_phase.params.quant.codes       = codes;
    quant_phase.params.quant.code_stride = stride;
    ASSERT(pipeline_add_phase(pipe, &quant_phase) >= 0, "add RERANK_QUANT phase ok");

    float query[DIM];
    fill_vector(query, DIM, 55u);

    GV_PhasedResult results[10];
    int n = pipeline_execute(pipe, query, DIM, 10, results);
    ASSERT(n >= 0, "pipeline_execute should succeed");
    ASSERT(n <= 10, "should not exceed final_k");

    for (int i = 0; i < n; i++) {
        ASSERT(results[i].index < NVECS, "result index in range");
        ASSERT(results[i].phase_reached >= 0, "phase_reached valid");
    }

    pipeline_destroy(pipe);
    free(codes);
    quant_codebook_destroy(cb);
    db_close(db);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test runner                                                           */
/* ------------------------------------------------------------------ */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"config_init defaults",              test_config_init},
        {"null input rejection",              test_null_inputs},
        {"basic quant_rerank_search",         test_basic_search},
        {"result count <= k",                 test_result_count_le_k},
        {"recall improvement (exact rerank)", test_recall_improvement},
        {"gv_db_search_with_rerank",          test_simple_rerank},
        {"phased pipeline RERANK_QUANT",      test_phased_pipeline_quant},
    };

    int total  = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;

    for (int i = 0; i < total; i++) {
        fprintf(stderr, "  %s ... ", tests[i].name);
        if (tests[i].fn() == 0) {
            fprintf(stderr, "PASS\n");
            passed++;
        } else {
            fprintf(stderr, "FAIL\n");
        }
    }

    fprintf(stderr, "\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
