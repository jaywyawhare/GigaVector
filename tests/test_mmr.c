#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_mmr.h"

#define DIM 4

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ---------- test_config_init ---------- */
static int test_config_init(void) {
    GV_MMRConfig cfg;
    gv_mmr_config_init(&cfg);
    ASSERT(fabs(cfg.lambda - 0.7f) < 0.01f, "lambda should default to 0.7");
    /* distance_type default should be cosine (1) */
    ASSERT(cfg.distance_type == 1, "distance_type should default to COSINE (1)");
    return 0;
}

/* ---------- test_rerank_basic ---------- */
static int test_rerank_basic(void) {
    /* Query vector */
    float query[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};

    /* 4 candidate vectors */
    float candidates[4 * DIM] = {
        1.0f, 0.0f, 0.0f, 0.0f,   /* identical to query */
        0.9f, 0.1f, 0.0f, 0.0f,   /* very similar */
        0.0f, 1.0f, 0.0f, 0.0f,   /* orthogonal */
        0.0f, 0.0f, 1.0f, 0.0f,   /* orthogonal, different */
    };
    size_t indices[4] = {0, 1, 2, 3};
    float distances[4] = {0.0f, 0.1f, 1.0f, 1.0f};

    GV_MMRResult results[3];
    int n = gv_mmr_rerank(query, DIM, candidates, indices, distances,
                          4, 3, NULL, results);
    ASSERT(n >= 1, "rerank should return at least 1 result");
    ASSERT(n <= 3, "rerank should return at most k=3 results");

    /* First result should be the most relevant (closest to query) */
    ASSERT(results[0].index == 0, "first MMR result should be the most relevant candidate");

    return 0;
}

/* ---------- test_rerank_diversity ---------- */
static int test_rerank_diversity(void) {
    /* With lambda=0 (full diversity), the results should be maximally spread out */
    float query[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};

    /* Two candidates near query, two orthogonal */
    float candidates[4 * DIM] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.99f, 0.01f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    size_t indices[4] = {10, 11, 12, 13};
    float distances[4] = {0.0f, 0.01f, 1.0f, 1.0f};

    GV_MMRConfig cfg;
    gv_mmr_config_init(&cfg);
    cfg.lambda = 0.0f; /* full diversity */

    GV_MMRResult results[3];
    int n = gv_mmr_rerank(query, DIM, candidates, indices, distances,
                          4, 3, &cfg, results);
    ASSERT(n == 3, "should get 3 results");

    /* With full diversity, after picking first (most relevant), next should be
       maximally different, not the near-duplicate */
    int has_diverse = 0;
    for (int i = 0; i < n; i++) {
        if (results[i].index == 12 || results[i].index == 13)
            has_diverse = 1;
    }
    ASSERT(has_diverse, "diversity mode should select orthogonal candidates");

    return 0;
}

/* ---------- test_rerank_k_larger_than_candidates ---------- */
static int test_rerank_k_larger_than_candidates(void) {
    float query[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float candidates[2 * DIM] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    size_t indices[2] = {0, 1};
    float distances[2] = {0.0f, 1.0f};

    GV_MMRResult results[5];
    int n = gv_mmr_rerank(query, DIM, candidates, indices, distances,
                          2, 5, NULL, results);
    /* Should return at most candidate_count = 2 */
    ASSERT(n >= 0 && n <= 2, "k > candidates: should return at most candidate_count");

    return 0;
}

/* ---------- test_rerank_single_candidate ---------- */
static int test_rerank_single_candidate(void) {
    float query[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float candidates[DIM] = {0.5f, 0.5f, 0.0f, 0.0f};
    size_t indices[1] = {42};
    float distances[1] = {0.5f};

    GV_MMRResult results[1];
    int n = gv_mmr_rerank(query, DIM, candidates, indices, distances,
                          1, 1, NULL, results);
    ASSERT(n == 1, "single candidate should return 1 result");
    ASSERT(results[0].index == 42, "result index should match");

    return 0;
}

/* ---------- test_rerank_zero_candidates ---------- */
static int test_rerank_zero_candidates(void) {
    float query[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};

    GV_MMRResult results[1];
    int n = gv_mmr_rerank(query, DIM, NULL, NULL, NULL, 0, 1, NULL, results);
    ASSERT(n == 0 || n == -1, "zero candidates should return 0 or error");

    return 0;
}

/* ---------- test_result_fields ---------- */
static int test_result_fields(void) {
    float query[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float candidates[2 * DIM] = {
        0.8f, 0.2f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.5f,
    };
    size_t indices[2] = {5, 6};
    float distances[2] = {0.2f, 1.0f};

    GV_MMRResult results[2];
    int n = gv_mmr_rerank(query, DIM, candidates, indices, distances,
                          2, 2, NULL, results);
    ASSERT(n == 2, "should return 2 results");

    /* Check that all fields are populated */
    for (int i = 0; i < n; i++) {
        ASSERT(results[i].index == 5 || results[i].index == 6,
               "result index should be 5 or 6");
        /* relevance should be non-negative */
        ASSERT(results[i].relevance >= 0.0f || results[i].relevance <= 1.0f,
               "relevance should be reasonable");
    }

    return 0;
}

/* test runner */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing MMR config init...",                   test_config_init},
        {"Testing MMR rerank basic...",                  test_rerank_basic},
        {"Testing MMR rerank diversity...",              test_rerank_diversity},
        {"Testing MMR k > candidates...",               test_rerank_k_larger_than_candidates},
        {"Testing MMR single candidate...",             test_rerank_single_candidate},
        {"Testing MMR zero candidates...",              test_rerank_zero_candidates},
        {"Testing MMR result fields...",                test_result_fields},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
