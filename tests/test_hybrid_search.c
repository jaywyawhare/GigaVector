#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_hybrid_search.h"
#include "gigavector/gv_bm25.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"

#define DIM 4

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Helper: create a database with a few vectors and a BM25 index with matching docs. */
static GV_Database *make_db(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;
    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v2[] = {0.0f, 0.0f, 1.0f, 0.0f};
    gv_db_add_vector(db, v0, DIM);
    gv_db_add_vector(db, v1, DIM);
    gv_db_add_vector(db, v2, DIM);
    return db;
}

static GV_BM25Index *make_bm25(void) {
    GV_BM25Index *bm = gv_bm25_create(NULL);
    if (!bm) return NULL;
    gv_bm25_add_document(bm, 0, "alpha bravo charlie");
    gv_bm25_add_document(bm, 1, "delta echo foxtrot");
    gv_bm25_add_document(bm, 2, "alpha delta gamma");
    return bm;
}

/* ---------- test_config_defaults ---------- */
static int test_config_defaults(void) {
    GV_HybridConfig cfg;
    gv_hybrid_config_init(&cfg);
    ASSERT(cfg.fusion_type == GV_FUSION_LINEAR, "default fusion should be LINEAR");
    ASSERT(fabs(cfg.vector_weight - 0.5) < 0.01, "default vector_weight should be 0.5");
    ASSERT(fabs(cfg.text_weight - 0.5) < 0.01, "default text_weight should be 0.5");
    ASSERT(fabs(cfg.rrf_k - 60.0) < 0.01, "default rrf_k should be 60");
    return 0;
}

/* ---------- test_create_destroy ---------- */
static int test_create_destroy(void) {
    GV_Database *db = make_db();
    GV_BM25Index *bm = make_bm25();
    ASSERT(db != NULL && bm != NULL, "setup should succeed");

    GV_HybridSearcher *hs = gv_hybrid_create(db, bm, NULL);
    ASSERT(hs != NULL, "gv_hybrid_create should succeed");

    gv_hybrid_destroy(hs);
    gv_hybrid_destroy(NULL); /* NULL should be safe */
    gv_bm25_destroy(bm);
    gv_db_close(db);
    return 0;
}

/* ---------- test_linear_fusion_util ---------- */
static int test_linear_fusion_util(void) {
    double score = gv_hybrid_linear_fusion(0.8, 0.6, 0.7, 0.3);
    /* 0.7*0.8 + 0.3*0.6 = 0.56 + 0.18 = 0.74 */
    ASSERT(fabs(score - 0.74) < 0.001, "linear fusion should compute correctly");
    return 0;
}

/* ---------- test_rrf_fusion_util ---------- */
static int test_rrf_fusion_util(void) {
    /* RRF: 1/(k+rank_v) + 1/(k+rank_t) */
    double score = gv_hybrid_rrf_fusion(1, 2, 60.0);
    double expected = 1.0 / (60.0 + 1.0) + 1.0 / (60.0 + 2.0);
    ASSERT(fabs(score - expected) < 0.0001, "RRF fusion should compute correctly");

    /* rank 0 means not found -- only one term contributes */
    double score2 = gv_hybrid_rrf_fusion(3, 0, 60.0);
    double expected2 = 1.0 / (60.0 + 3.0);
    ASSERT(fabs(score2 - expected2) < 0.0001, "RRF with one missing rank should work");
    return 0;
}

/* ---------- test_normalize_score ---------- */
static int test_normalize_score(void) {
    double n = gv_hybrid_normalize_score(5.0, 2.0, 10.0);
    /* (5-2)/(10-2) = 3/8 = 0.375 */
    ASSERT(fabs(n - 0.375) < 0.001, "normalize should map to [0,1]");

    /* When min==max, should return 0 or handle gracefully */
    double n2 = gv_hybrid_normalize_score(5.0, 5.0, 5.0);
    ASSERT(n2 >= 0.0 && n2 <= 1.0, "normalize with equal min/max should be in [0,1]");
    return 0;
}

/* ---------- test_set_weights ---------- */
static int test_set_weights(void) {
    GV_Database *db = make_db();
    GV_BM25Index *bm = make_bm25();
    GV_HybridSearcher *hs = gv_hybrid_create(db, bm, NULL);
    ASSERT(hs != NULL, "create should succeed");

    int rc = gv_hybrid_set_weights(hs, 0.8, 0.2);
    ASSERT(rc == 0, "set_weights should succeed");

    GV_HybridConfig cfg;
    rc = gv_hybrid_get_config(hs, &cfg);
    ASSERT(rc == 0, "get_config should succeed");
    /* Weights are normalized, so 0.8/(0.8+0.2) = 0.8, 0.2/(0.8+0.2) = 0.2 */
    ASSERT(fabs(cfg.vector_weight - 0.8) < 0.01, "vector_weight should be 0.8");
    ASSERT(fabs(cfg.text_weight - 0.2) < 0.01, "text_weight should be 0.2");

    gv_hybrid_destroy(hs);
    gv_bm25_destroy(bm);
    gv_db_close(db);
    return 0;
}

/* ---------- test_hybrid_search_basic ---------- */
static int test_hybrid_search_basic(void) {
    GV_Database *db = make_db();
    GV_BM25Index *bm = make_bm25();
    GV_HybridSearcher *hs = gv_hybrid_create(db, bm, NULL);
    ASSERT(hs != NULL, "create should succeed");

    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_HybridResult results[3];
    int n = gv_hybrid_search(hs, query, "alpha", 3, results);
    ASSERT(n >= 0, "hybrid search should not fail");

    gv_hybrid_destroy(hs);
    gv_bm25_destroy(bm);
    gv_db_close(db);
    return 0;
}

/* ---------- test_set_config ---------- */
static int test_set_config(void) {
    GV_Database *db = make_db();
    GV_BM25Index *bm = make_bm25();
    GV_HybridSearcher *hs = gv_hybrid_create(db, bm, NULL);
    ASSERT(hs != NULL, "create should succeed");

    GV_HybridConfig cfg;
    gv_hybrid_config_init(&cfg);
    cfg.fusion_type = GV_FUSION_RRF;
    cfg.rrf_k = 30.0;
    int rc = gv_hybrid_set_config(hs, &cfg);
    ASSERT(rc == 0, "set_config should succeed");

    GV_HybridConfig out;
    rc = gv_hybrid_get_config(hs, &out);
    ASSERT(rc == 0, "get_config should succeed");
    ASSERT(out.fusion_type == GV_FUSION_RRF, "fusion type should be RRF");

    gv_hybrid_destroy(hs);
    gv_bm25_destroy(bm);
    gv_db_close(db);
    return 0;
}

/* ========== test runner ========== */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing hybrid config defaults...",      test_config_defaults},
        {"Testing hybrid create/destroy...",       test_create_destroy},
        {"Testing hybrid linear fusion util...",   test_linear_fusion_util},
        {"Testing hybrid RRF fusion util...",      test_rrf_fusion_util},
        {"Testing hybrid normalize score...",      test_normalize_score},
        {"Testing hybrid set weights...",          test_set_weights},
        {"Testing hybrid search basic...",         test_hybrid_search_basic},
        {"Testing hybrid set config...",           test_set_config},
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
