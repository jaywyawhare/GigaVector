#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_ttl.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_init(void) {
    GV_TTLConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_ttl_config_init(&cfg);
    ASSERT(cfg.default_ttl_seconds == 0,       "default_ttl_seconds should be 0");
    ASSERT(cfg.cleanup_interval_seconds == 60,  "cleanup_interval_seconds should be 60");
    ASSERT(cfg.lazy_expiration == 1,            "lazy_expiration should be 1");
    ASSERT(cfg.max_expired_per_cleanup == 1000, "max_expired_per_cleanup should be 1000");
    return 0;
}

static int test_create_destroy(void) {
    /* NULL config => defaults */
    GV_TTLManager *mgr = gv_ttl_create(NULL);
    ASSERT(mgr != NULL, "gv_ttl_create(NULL) should succeed");
    gv_ttl_destroy(mgr);

    /* Explicit config */
    GV_TTLConfig cfg;
    gv_ttl_config_init(&cfg);
    cfg.default_ttl_seconds = 120;
    mgr = gv_ttl_create(&cfg);
    ASSERT(mgr != NULL, "gv_ttl_create with config should succeed");
    gv_ttl_destroy(mgr);

    /* Destroy NULL is safe */
    gv_ttl_destroy(NULL);
    return 0;
}

static int test_set_and_get(void) {
    GV_TTLManager *mgr = gv_ttl_create(NULL);
    ASSERT(mgr != NULL, "create");

    /* Set TTL for index 5 to 3600 seconds */
    ASSERT(gv_ttl_set(mgr, 5, 3600) == 0, "set ttl");

    uint64_t expire_at = 0;
    ASSERT(gv_ttl_get(mgr, 5, &expire_at) == 0, "get ttl");
    ASSERT(expire_at > 0, "expire_at should be non-zero after set");

    /* Index without TTL */
    uint64_t no_ttl = 999;
    ASSERT(gv_ttl_get(mgr, 99, &no_ttl) == 0, "get ttl on unset index");
    ASSERT(no_ttl == 0, "expire_at should be 0 for unset index");

    gv_ttl_destroy(mgr);
    return 0;
}

static int test_set_absolute(void) {
    GV_TTLManager *mgr = gv_ttl_create(NULL);
    ASSERT(mgr != NULL, "create");

    uint64_t future = 9999999999ULL; /* far in the future */
    ASSERT(gv_ttl_set_absolute(mgr, 0, future) == 0, "set_absolute");

    uint64_t expire_at = 0;
    ASSERT(gv_ttl_get(mgr, 0, &expire_at) == 0, "get after set_absolute");
    ASSERT(expire_at == future, "expire_at should match the absolute time");

    /* Remove by setting 0 */
    ASSERT(gv_ttl_set_absolute(mgr, 0, 0) == 0, "set_absolute 0 to remove");
    ASSERT(gv_ttl_get(mgr, 0, &expire_at) == 0, "get after remove");
    ASSERT(expire_at == 0, "expire_at should be 0 after removal");

    gv_ttl_destroy(mgr);
    return 0;
}

static int test_remove(void) {
    GV_TTLManager *mgr = gv_ttl_create(NULL);
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_ttl_set(mgr, 10, 600) == 0, "set ttl");
    ASSERT(gv_ttl_remove(mgr, 10) == 0,    "remove ttl");

    uint64_t expire_at = 999;
    ASSERT(gv_ttl_get(mgr, 10, &expire_at) == 0, "get after remove");
    ASSERT(expire_at == 0, "expire_at should be 0 after remove");

    gv_ttl_destroy(mgr);
    return 0;
}

static int test_is_expired(void) {
    GV_TTLManager *mgr = gv_ttl_create(NULL);
    ASSERT(mgr != NULL, "create");

    /* Set expiration in the past */
    ASSERT(gv_ttl_set_absolute(mgr, 1, 1) == 0, "set to past timestamp");
    int expired = gv_ttl_is_expired(mgr, 1);
    ASSERT(expired == 1, "vector with past timestamp should be expired");

    /* Set expiration far in the future */
    ASSERT(gv_ttl_set_absolute(mgr, 2, 9999999999ULL) == 0, "set to future timestamp");
    expired = gv_ttl_is_expired(mgr, 2);
    ASSERT(expired == 0, "vector with future timestamp should not be expired");

    /* No TTL */
    expired = gv_ttl_is_expired(mgr, 42);
    ASSERT(expired == 0, "vector without TTL should not be expired");

    gv_ttl_destroy(mgr);
    return 0;
}

static int test_get_remaining(void) {
    GV_TTLManager *mgr = gv_ttl_create(NULL);
    ASSERT(mgr != NULL, "create");

    /* Set TTL far in the future => remaining should be positive */
    ASSERT(gv_ttl_set(mgr, 0, 7200) == 0, "set 2h ttl");
    uint64_t rem = 0;
    ASSERT(gv_ttl_get_remaining(mgr, 0, &rem) == 0, "get_remaining");
    ASSERT(rem > 0, "remaining should be positive for future TTL");

    /* Set TTL in the past */
    ASSERT(gv_ttl_set_absolute(mgr, 1, 1) == 0, "set past");
    ASSERT(gv_ttl_get_remaining(mgr, 1, &rem) == 0, "get_remaining past");
    ASSERT(rem == 0, "remaining should be 0 for expired");

    gv_ttl_destroy(mgr);
    return 0;
}

static int test_bulk_and_stats(void) {
    GV_TTLManager *mgr = gv_ttl_create(NULL);
    ASSERT(mgr != NULL, "create");

    size_t indices[] = {0, 1, 2, 3, 4};
    int set = gv_ttl_set_bulk(mgr, indices, 5, 1800);
    ASSERT(set == 5, "set_bulk should return 5");

    GV_TTLStats stats;
    memset(&stats, 0, sizeof(stats));
    ASSERT(gv_ttl_get_stats(mgr, &stats) == 0, "get_stats");
    ASSERT(stats.total_vectors_with_ttl == 5, "should have 5 vectors with ttl");

    gv_ttl_destroy(mgr);
    return 0;
}

/* main */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config_init...",       test_config_init},
        {"Testing create_destroy...",    test_create_destroy},
        {"Testing set_and_get...",       test_set_and_get},
        {"Testing set_absolute...",      test_set_absolute},
        {"Testing remove...",            test_remove},
        {"Testing is_expired...",        test_is_expired},
        {"Testing get_remaining...",     test_get_remaining},
        {"Testing bulk_and_stats...",    test_bulk_and_stats},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
