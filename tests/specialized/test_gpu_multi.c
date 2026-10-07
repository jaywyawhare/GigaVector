/* Multi-device (multi-GPU) context: query fan-out across N contexts must
 * return results identical to a single-context batch search. On a CPU-only
 * build the contexts fall back to CPU but the batch is still partitioned and
 * run in parallel, so the correctness contract is the same. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "specialized/gpu.h"
#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 4u
#define NVEC 16u
#define NQ 7u
#define K 3u

static int test_multi_create_destroy(void) {
    GV_GPUConfig cfg;
    gpu_config_init(&cfg);

    GV_GPUMultiContext *m = gpu_multi_create(&cfg, 3);
    ASSERT(m != NULL, "gpu_multi_create(3) should succeed (CPU fallback contexts)");
    ASSERT(gpu_multi_device_count(m) == 3, "device count should equal requested 3");
    gpu_multi_destroy(m);

    /* 0 clamps up to 1; NULL base_config uses defaults. */
    m = gpu_multi_create(NULL, 0);
    ASSERT(m != NULL, "gpu_multi_create(NULL, 0) should succeed");
    ASSERT(gpu_multi_device_count(m) == 1, "0 devices clamps to 1");
    gpu_multi_destroy(m);

    gpu_multi_destroy(NULL); /* must not crash */
    ASSERT(gpu_multi_device_count(NULL) == 0, "device count of NULL is 0");
    return 0;
}

static int test_multi_matches_single(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "open db");

    float vecs[NVEC][DIM];
    for (unsigned i = 0; i < NVEC; i++) {
        for (unsigned d = 0; d < DIM; d++) vecs[i][d] = (float)((i * 7u + d * 3u) % 11u);
        ASSERT(db_add_vector(db, vecs[i], DIM) == 0, "add vector");
    }

    float queries[NQ][DIM];
    for (unsigned q = 0; q < NQ; q++) {
        for (unsigned d = 0; d < DIM; d++) queries[q][d] = (float)((q * 5u + d) % 11u);
    }

    GV_GPUConfig cfg;
    gpu_config_init(&cfg);

    /* Reference: single context over the whole batch. */
    GV_GPUContext *single = gpu_create(&cfg);
    ASSERT(single != NULL, "create single context");
    size_t s_idx[NQ * K];
    float  s_dist[NQ * K];
    ASSERT(gpu_batch_search(single, db, &queries[0][0], NQ, K, s_idx, s_dist) == 0,
           "single-context batch search");
    gpu_destroy(single);

    /* Multi-context fan-out (more devices than some shards exercise the
     * uneven-split path: 7 queries over 4 contexts -> 2,2,2,1). */
    GV_GPUMultiContext *m = gpu_multi_create(&cfg, 4);
    ASSERT(m != NULL, "create multi context");
    size_t m_idx[NQ * K];
    float  m_dist[NQ * K];
    ASSERT(gpu_multi_batch_search(m, db, &queries[0][0], NQ, K, m_idx, m_dist) == 0,
           "multi-context batch search");
    gpu_multi_destroy(m);

    for (unsigned i = 0; i < NQ * K; i++) {
        ASSERT(m_idx[i] == s_idx[i], "multi indices match single");
    }

    /* Invalid-argument contract. */
    GV_GPUMultiContext *m2 = gpu_multi_create(&cfg, 2);
    ASSERT(m2 != NULL, "create multi context for arg checks");
    ASSERT(gpu_multi_batch_search(NULL, db, &queries[0][0], NQ, K, m_idx, m_dist) == -1,
           "NULL mctx rejected");
    ASSERT(gpu_multi_batch_search(m2, NULL, &queries[0][0], NQ, K, m_idx, m_dist) == -1,
           "NULL db rejected");
    ASSERT(gpu_multi_batch_search(m2, db, &queries[0][0], 0, K, m_idx, m_dist) == -1,
           "zero queries rejected");
    ASSERT(gpu_multi_batch_search(m2, db, &queries[0][0], NQ, 0, m_idx, m_dist) == -1,
           "zero k rejected");
    gpu_multi_destroy(m2);

    db_close(db);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing multi_create_destroy...", test_multi_create_destroy},
        {"Testing multi_matches_single...", test_multi_matches_single},
    };
    int n = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("  %s ", tests[i].name);
        if (tests[i].fn() == 0) { printf("OK\n"); passed++; }
        else { printf("FAILED\n"); }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
