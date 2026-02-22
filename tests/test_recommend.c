#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_recommend.h"
#include "gigavector/gv_database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 4

/* Helper: create a database and populate it with test vectors */
static GV_Database *create_test_db(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;

    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.9f, 0.1f, 0.0f, 0.0f};
    float v2[] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v3[] = {0.0f, 0.0f, 0.9f, 0.1f};
    float v4[] = {0.5f, 0.5f, 0.5f, 0.5f};

    gv_db_add_vector(db, v0, DIM);
    gv_db_add_vector(db, v1, DIM);
    gv_db_add_vector(db, v2, DIM);
    gv_db_add_vector(db, v3, DIM);
    gv_db_add_vector(db, v4, DIM);

    return db;
}

/* --- Test: config init defaults --- */
static int test_config_init(void) {
    GV_RecommendConfig config;
    memset(&config, 0xFF, sizeof(config));
    gv_recommend_config_init(&config);

    ASSERT(config.positive_weight > 0.0f, "positive_weight should be positive");
    ASSERT(config.negative_weight >= 0.0f, "negative_weight should be non-negative");
    ASSERT(config.oversample >= 1, "oversample should be at least 1");
    ASSERT(config.exclude_input == 1, "exclude_input should default to 1");

    return 0;
}

/* --- Test: recommend by ID with positives only --- */
static int test_recommend_by_id_positive(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_RecommendConfig config;
    gv_recommend_config_init(&config);

    /* Recommend based on v0 (index 0) as positive. Should find v1 similar. */
    size_t positive_ids[] = {0};
    GV_RecommendResult results[3];
    int n = gv_recommend_by_id(db, positive_ids, 1, NULL, 0, 3, &config, results);
    ASSERT(n >= 1, "recommend_by_id should return at least 1 result");

    gv_db_close(db);
    return 0;
}

/* --- Test: recommend by ID with positives and negatives --- */
static int test_recommend_by_id_pos_neg(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_RecommendConfig config;
    gv_recommend_config_init(&config);

    /* Positive: v0 (index 0), Negative: v2 (index 2) */
    size_t positive_ids[] = {0};
    size_t negative_ids[] = {2};
    GV_RecommendResult results[3];
    int n = gv_recommend_by_id(db, positive_ids, 1, negative_ids, 1, 3, &config, results);
    ASSERT(n >= 1, "recommend_by_id with negatives should return results");

    gv_db_close(db);
    return 0;
}

/* --- Test: recommend by vector --- */
static int test_recommend_by_vector(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_RecommendConfig config;
    gv_recommend_config_init(&config);

    float positive_vecs[] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_RecommendResult results[3];
    int n = gv_recommend_by_vector(db, positive_vecs, 1, NULL, 0,
                                    DIM, 3, &config, results);
    ASSERT(n >= 1, "recommend_by_vector should return at least 1 result");

    gv_db_close(db);
    return 0;
}

/* --- Test: recommend by vector with negatives --- */
static int test_recommend_by_vector_neg(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_RecommendConfig config;
    gv_recommend_config_init(&config);

    float positive_vecs[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float negative_vecs[] = {0.0f, 0.0f, 1.0f, 0.0f};
    GV_RecommendResult results[3];
    int n = gv_recommend_by_vector(db, positive_vecs, 1, negative_vecs, 1,
                                    DIM, 3, &config, results);
    ASSERT(n >= 1, "recommend_by_vector with negatives should return results");

    gv_db_close(db);
    return 0;
}

/* --- Test: discover --- */
static int test_discover(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_RecommendConfig config;
    gv_recommend_config_init(&config);

    float target[]  = {1.0f, 0.0f, 0.0f, 0.0f};
    float context[] = {0.0f, 0.0f, 1.0f, 0.0f};
    GV_RecommendResult results[3];
    int n = gv_recommend_discover(db, target, context, DIM, 3, &config, results);
    ASSERT(n >= 1, "discover should return at least 1 result");

    gv_db_close(db);
    return 0;
}

/* --- Test: recommend with k larger than database --- */
static int test_recommend_k_larger(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "create_test_db should succeed");

    GV_RecommendConfig config;
    gv_recommend_config_init(&config);

    size_t positive_ids[] = {0};
    GV_RecommendResult results[100];
    int n = gv_recommend_by_id(db, positive_ids, 1, NULL, 0, 100, &config, results);
    /* Should return at most the number of vectors in DB (minus excluded input) */
    ASSERT(n >= 0, "recommend with large k should not error");
    ASSERT((size_t)n <= 5, "should not return more results than vectors in DB");

    gv_db_close(db);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing recommend config init...", test_config_init},
        {"Testing recommend by ID (positives)...", test_recommend_by_id_positive},
        {"Testing recommend by ID (pos+neg)...", test_recommend_by_id_pos_neg},
        {"Testing recommend by vector...", test_recommend_by_vector},
        {"Testing recommend by vector (neg)...", test_recommend_by_vector_neg},
        {"Testing recommend discover...", test_discover},
        {"Testing recommend k larger than DB...", test_recommend_k_larger},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
