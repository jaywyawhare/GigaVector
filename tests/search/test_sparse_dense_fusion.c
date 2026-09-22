#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "search/sparse_dense_fusion.h"
#include "core/memory.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_defaults(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);

    ASSERT(cfg.type == GV_SPARSE_DENSE_FUSION_RRF,
           "default type should be RRF");
    ASSERT(fabs(cfg.sparse_weight - 0.5f) < 0.01f,
           "default sparse_weight should be 0.5");
    ASSERT(fabs(cfg.dense_weight - 0.5f) < 0.01f,
           "default dense_weight should be 0.5");
    ASSERT(cfg.rrf_k == 60,
           "default rrf_k should be 60");

    /* NULL config is safe */
    sparse_dense_fusion_config_init(NULL);

    return 0;
}

static int test_fuse_rrf(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);
    cfg.type = GV_SPARSE_DENSE_FUSION_RRF;

    /* Sparse results: doc 0, 1, 2 (score desc) */
    GV_SearchResult sparse[3];
    memset(sparse, 0, sizeof(sparse));
    sparse[0].id = 0; sparse[0].distance = 0.9f;
    sparse[1].id = 1; sparse[1].distance = 0.7f;
    sparse[2].id = 2; sparse[2].distance = 0.5f;

    /* Dense results: doc 1, 3, 0 (score desc) — note overlap on ids 0 and 1 */
    GV_SearchResult dense[3];
    memset(dense, 0, sizeof(dense));
    dense[0].id = 1; dense[0].distance = 0.8f;
    dense[1].id = 3; dense[1].distance = 0.6f;
    dense[2].id = 0; dense[2].distance = 0.4f;

    GV_SearchResult fused[10];
    int n = sparse_dense_fuse(sparse, 3, dense, 3, fused, 10, &cfg);
    ASSERT(n == 4, "should produce 4 unique docs (0,1,2,3)");

    /* Doc 1 appears in both: rank 2 in sparse, rank 1 in dense.
     * RRF score = 1/(60+2) + 1/(60+1) ≈ 0.0325
     * Doc 0: rank 1 sparse, rank 3 dense => 1/61 + 1/63 ≈ 0.0321
     * Doc 1 should rank higher than doc 0 */
    ASSERT(fused[0].id == 1, "doc 1 (in both lists, high dense rank) should be first");

    /* Verify all unique ids are present */
    int found_ids[4] = {0};
    for (int i = 0; i < n; i++) {
        if (fused[i].id < 4) found_ids[fused[i].id] = 1;
    }
    for (int i = 0; i < 4; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "doc %d should be in fused results", i);
        ASSERT(found_ids[i], msg);
    }

    return 0;
}

static int test_fuse_weighted_rrf(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);
    cfg.type = GV_SPARSE_DENSE_FUSION_WEIGHTED_RRF;
    cfg.sparse_weight = 0.8f;
    cfg.dense_weight  = 0.2f;

    GV_SearchResult sparse[2];
    memset(sparse, 0, sizeof(sparse));
    sparse[0].id = 0; sparse[0].distance = 0.9f;
    sparse[1].id = 1; sparse[1].distance = 0.7f;

    GV_SearchResult dense[2];
    memset(dense, 0, sizeof(dense));
    dense[0].id = 1; dense[0].distance = 0.8f;
    dense[1].id = 2; dense[1].distance = 0.6f;

    GV_SearchResult fused[10];
    int n = sparse_dense_fuse(sparse, 2, dense, 2, fused, 10, &cfg);
    ASSERT(n == 3, "should produce 3 unique docs");

    /* With heavy sparse weight (0.8), sparse-only doc 0 should beat dense-only doc 2 */
    ASSERT(fused[0].id == 0 || fused[0].id == 1,
           "top result should be doc 0 or 1");

    /* Doc 2 is only in dense with low weight => should be last */
    ASSERT(fused[n - 1].id == 2,
           "dense-only doc should rank last with low dense weight");

    return 0;
}

static int test_fuse_linear(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);
    cfg.type = GV_SPARSE_DENSE_FUSION_LINEAR;
    cfg.sparse_weight = 0.5f;
    cfg.dense_weight  = 0.5f;

    /* Only sparse */
    GV_SearchResult sparse[3];
    memset(sparse, 0, sizeof(sparse));
    sparse[0].id = 0; sparse[0].distance = 1.0f;
    sparse[1].id = 1; sparse[1].distance = 0.5f;
    sparse[2].id = 2; sparse[2].distance = 0.0f;

    /* Empty dense */
    GV_SearchResult fused[10];
    int n = sparse_dense_fuse(sparse, 3, NULL, 0, fused, 10, &cfg);
    ASSERT(n == 3, "should return all 3 sparse results when dense is empty");

    /* Scores should be normalised within sparse only */
    ASSERT(fused[0].id == 0, "highest sparse score doc should be first");

    return 0;
}

static int test_fuse_convex(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);
    cfg.type = GV_SPARSE_DENSE_FUSION_CONVEX;
    cfg.dense_weight = 1.0f;  /* alpha = 1.0 => dense only */

    GV_SearchResult sparse[2];
    memset(sparse, 0, sizeof(sparse));
    sparse[0].id = 0; sparse[0].distance = 0.9f;
    sparse[1].id = 1; sparse[1].distance = 0.1f;

    GV_SearchResult dense[2];
    memset(dense, 0, sizeof(dense));
    dense[0].id = 2; dense[0].distance = 0.5f;
    dense[1].id = 3; dense[1].distance = 0.4f;

    GV_SearchResult fused[10];
    int n = sparse_dense_fuse(sparse, 2, dense, 2, fused, 10, &cfg);
    ASSERT(n == 4, "should return all 4 docs");

    /* With alpha=1.0, only dense scores matter => doc 2 (highest dense) first */
    ASSERT(fused[0].id == 2, "with alpha=1.0 (dense only), dense top doc should be first");

    return 0;
}

static int test_deduplication(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);
    cfg.type = GV_SPARSE_DENSE_FUSION_RRF;

    /* Both lists contain exactly the same docs */
    GV_SearchResult sparse[3];
    memset(sparse, 0, sizeof(sparse));
    sparse[0].id = 0; sparse[0].distance = 0.9f;
    sparse[1].id = 1; sparse[1].distance = 0.8f;
    sparse[2].id = 2; sparse[2].distance = 0.7f;

    GV_SearchResult dense[3];
    memset(dense, 0, sizeof(dense));
    dense[0].id = 0; dense[0].distance = 0.6f;
    dense[1].id = 1; dense[1].distance = 0.5f;
    dense[2].id = 2; dense[2].distance = 0.4f;

    GV_SearchResult fused[10];
    int n = sparse_dense_fuse(sparse, 3, dense, 3, fused, 10, &cfg);
    ASSERT(n == 3, "fully overlapping lists should produce exactly 3 unique results");

    /* Each doc appears in both, so all should have two RRF contributions */
    for (int i = 0; i < n; i++) {
        ASSERT(fused[i].distance > 0.0f, "all fused scores should be positive");
    }

    return 0;
}

static int test_weight_effects(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);
    cfg.type = GV_SPARSE_DENSE_FUSION_WEIGHTED_RRF;

    GV_SearchResult sparse[1];
    memset(sparse, 0, sizeof(sparse));
    sparse[0].id = 0; sparse[0].distance = 0.9f;

    GV_SearchResult dense[1];
    memset(dense, 0, sizeof(dense));
    dense[0].id = 1; dense[0].distance = 0.9f;

    /* Run with equal weights */
    cfg.sparse_weight = 0.5f;
    cfg.dense_weight  = 0.5f;
    GV_SearchResult fused_eq[10];
    int n_eq = sparse_dense_fuse(sparse, 1, dense, 1, fused_eq, 10, &cfg);
    ASSERT(n_eq == 2, "should have 2 results");

    /* Run with heavy sparse weight */
    cfg.sparse_weight = 0.9f;
    cfg.dense_weight  = 0.1f;
    GV_SearchResult fused_sp[10];
    int n_sp = sparse_dense_fuse(sparse, 1, dense, 1, fused_sp, 10, &cfg);
    ASSERT(n_sp == 2, "should have 2 results");

    /* Doc 0 is sparse-only, doc 1 is dense-only.
     * With higher sparse weight, sparse-only doc should score relatively better. */
    float score_sp_doc0 = -1;
    float score_eq_doc0 = -1;
    for (int i = 0; i < n_eq; i++) {
        if (fused_eq[i].id == 0) score_eq_doc0 = fused_eq[i].distance;
    }
    for (int i = 0; i < n_sp; i++) {
        if (fused_sp[i].id == 0) score_sp_doc0 = fused_sp[i].distance;
    }

    ASSERT(score_sp_doc0 > score_eq_doc0,
           "sparse-only doc should score higher with increased sparse weight");

    return 0;
}

static int test_max_fused_limit(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);

    GV_SearchResult sparse[5];
    memset(sparse, 0, sizeof(sparse));
    for (int i = 0; i < 5; i++) {
        sparse[i].id = (size_t)i;
        sparse[i].distance = 0.9f - (float)i * 0.1f;
    }

    GV_SearchResult fused[3]; /* capacity limited to 3 */
    int n = sparse_dense_fuse(sparse, 5, NULL, 0, fused, 3, &cfg);
    ASSERT(n == 3, "should respect max_fused capacity limit");

    return 0;
}

static int test_single_source_only(void) {
    GV_SparseDenseFusionConfig cfg;
    sparse_dense_fusion_config_init(&cfg);
    cfg.type = GV_SPARSE_DENSE_FUSION_LINEAR;

    /* Only dense, no sparse */
    GV_SearchResult dense[3];
    memset(dense, 0, sizeof(dense));
    dense[0].id = 0; dense[0].distance = 0.9f;
    dense[1].id = 1; dense[1].distance = 0.5f;
    dense[2].id = 2; dense[2].distance = 0.1f;

    GV_SearchResult fused[10];
    int n = sparse_dense_fuse(NULL, 0, dense, 3, fused, 10, &cfg);
    ASSERT(n == 3, "should return all dense results when sparse is empty");
    ASSERT(fused[0].id == 0, "highest dense score should be first");

    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing sparse-dense fusion config defaults...",    test_config_defaults},
        {"Testing sparse-dense fusion RRF...",                test_fuse_rrf},
        {"Testing sparse-dense fusion weighted RRF...",       test_fuse_weighted_rrf},
        {"Testing sparse-dense fusion linear...",             test_fuse_linear},
        {"Testing sparse-dense fusion convex...",             test_fuse_convex},
        {"Testing sparse-dense deduplication...",             test_deduplication},
        {"Testing sparse-dense weight effects...",            test_weight_effects},
        {"Testing sparse-dense max fused limit...",           test_max_fused_limit},
        {"Testing sparse-dense single source only...",        test_single_source_only},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("%s ", tests[i].name);
        if (tests[i].fn() == 0) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
        }
    }
    return passed == n ? 0 : 1;
}
