#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_optimizer.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_optimizer_create_destroy(void) {
    GV_QueryOptimizer *opt = gv_optimizer_create();
    ASSERT(opt != NULL, "optimizer creation should succeed");

    gv_optimizer_destroy(opt);

    /* Destroying NULL should be safe */
    gv_optimizer_destroy(NULL);
    return 0;
}

static int test_optimizer_update_stats(void) {
    GV_QueryOptimizer *opt = gv_optimizer_create();
    ASSERT(opt != NULL, "optimizer creation");

    GV_CollectionStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_vectors = 100000;
    stats.dimension = 128;
    stats.index_type = GV_PLAN_INDEX_SEARCH;
    stats.deleted_ratio = 0.05;
    stats.avg_vectors_per_filter_match = 500.0;
    stats.last_search_latency_us = 2000;

    gv_optimizer_update_stats(opt, &stats);

    /* No crash means success; stats are internal state */
    gv_optimizer_destroy(opt);
    return 0;
}

static int test_optimizer_plan_no_filter(void) {
    GV_QueryOptimizer *opt = gv_optimizer_create();
    ASSERT(opt != NULL, "optimizer creation");

    GV_CollectionStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_vectors = 50000;
    stats.dimension = 256;
    stats.index_type = GV_PLAN_INDEX_SEARCH;
    stats.deleted_ratio = 0.01;

    gv_optimizer_update_stats(opt, &stats);

    GV_QueryPlan plan;
    memset(&plan, 0, sizeof(plan));
    int rc = gv_optimizer_plan(opt, 10, 0, 1.0, &plan);
    ASSERT(rc == 0, "plan generation should succeed");
    ASSERT(plan.estimated_recall >= 0.0 && plan.estimated_recall <= 1.0,
           "estimated recall should be in [0,1]");
    ASSERT(plan.estimated_cost >= 0.0, "estimated cost should be non-negative");

    gv_optimizer_destroy(opt);
    return 0;
}

static int test_optimizer_plan_with_filter(void) {
    GV_QueryOptimizer *opt = gv_optimizer_create();
    ASSERT(opt != NULL, "optimizer creation");

    GV_CollectionStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_vectors = 100000;
    stats.dimension = 128;
    stats.index_type = GV_PLAN_INDEX_SEARCH;
    stats.deleted_ratio = 0.02;
    stats.avg_vectors_per_filter_match = 100.0;

    gv_optimizer_update_stats(opt, &stats);

    GV_QueryPlan plan;
    memset(&plan, 0, sizeof(plan));
    int rc = gv_optimizer_plan(opt, 10, 1, 0.01, &plan);
    ASSERT(rc == 0, "plan generation with filter should succeed");
    ASSERT(plan.strategy >= GV_PLAN_EXACT_SCAN && plan.strategy <= GV_PLAN_OVERSAMPLE_FILTER,
           "strategy should be a valid enum value");

    gv_optimizer_destroy(opt);
    return 0;
}

static int test_optimizer_plan_explanation(void) {
    GV_QueryOptimizer *opt = gv_optimizer_create();
    ASSERT(opt != NULL, "optimizer creation");

    GV_CollectionStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_vectors = 10000;
    stats.dimension = 64;
    stats.index_type = GV_PLAN_INDEX_SEARCH;

    gv_optimizer_update_stats(opt, &stats);

    GV_QueryPlan plan;
    memset(&plan, 0, sizeof(plan));
    int rc = gv_optimizer_plan(opt, 5, 0, 1.0, &plan);
    ASSERT(rc == 0, "plan generation should succeed");
    ASSERT(strlen(plan.explanation) > 0, "plan explanation should not be empty");

    gv_optimizer_destroy(opt);
    return 0;
}

static int test_optimizer_recommend_ef_search(void) {
    GV_QueryOptimizer *opt = gv_optimizer_create();
    ASSERT(opt != NULL, "optimizer creation");

    GV_CollectionStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_vectors = 50000;
    stats.dimension = 128;
    stats.index_type = GV_PLAN_INDEX_SEARCH;

    gv_optimizer_update_stats(opt, &stats);

    size_t ef = gv_optimizer_recommend_ef_search(opt, 10);
    ASSERT(ef >= 10, "recommended ef_search should be >= k");

    size_t ef2 = gv_optimizer_recommend_ef_search(opt, 100);
    ASSERT(ef2 >= 100, "recommended ef_search should be >= k for larger k");

    gv_optimizer_destroy(opt);
    return 0;
}

static int test_optimizer_recommend_nprobe(void) {
    GV_QueryOptimizer *opt = gv_optimizer_create();
    ASSERT(opt != NULL, "optimizer creation");

    GV_CollectionStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_vectors = 100000;
    stats.dimension = 128;

    gv_optimizer_update_stats(opt, &stats);

    size_t nprobe = gv_optimizer_recommend_nprobe(opt, 10);
    ASSERT(nprobe >= 1, "recommended nprobe should be >= 1");

    gv_optimizer_destroy(opt);
    return 0;
}

static int test_optimizer_record_result(void) {
    GV_QueryOptimizer *opt = gv_optimizer_create();
    ASSERT(opt != NULL, "optimizer creation");

    GV_CollectionStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.total_vectors = 10000;
    stats.dimension = 64;
    stats.index_type = GV_PLAN_INDEX_SEARCH;

    gv_optimizer_update_stats(opt, &stats);

    GV_QueryPlan plan;
    memset(&plan, 0, sizeof(plan));
    int rc = gv_optimizer_plan(opt, 10, 0, 1.0, &plan);
    ASSERT(rc == 0, "plan generation should succeed");

    /* Record a result for learning */
    gv_optimizer_record_result(opt, &plan, 1500, 0.95);

    /* No crash means success; recording is internal */
    gv_optimizer_destroy(opt);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing optimizer create/destroy...", test_optimizer_create_destroy},
        {"Testing optimizer update stats...", test_optimizer_update_stats},
        {"Testing optimizer plan (no filter)...", test_optimizer_plan_no_filter},
        {"Testing optimizer plan (with filter)...", test_optimizer_plan_with_filter},
        {"Testing optimizer plan explanation...", test_optimizer_plan_explanation},
        {"Testing optimizer recommend ef_search...", test_optimizer_recommend_ef_search},
        {"Testing optimizer recommend nprobe...", test_optimizer_recommend_nprobe},
        {"Testing optimizer record result...", test_optimizer_record_result},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
