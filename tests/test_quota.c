#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_quota.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_init(void) {
    GV_QuotaConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_quota_config_init(&cfg);
    ASSERT(cfg.max_vectors == 0,       "max_vectors should be 0 (unlimited)");
    ASSERT(cfg.max_memory_bytes == 0,  "max_memory_bytes should be 0");
    ASSERT(cfg.max_qps == 0.0,        "max_qps should be 0");
    ASSERT(cfg.max_ips == 0.0,        "max_ips should be 0");
    ASSERT(cfg.max_storage_bytes == 0, "max_storage_bytes should be 0");
    ASSERT(cfg.max_collections == 0,   "max_collections should be 0");
    return 0;
}

static int test_create_destroy(void) {
    GV_QuotaManager *mgr = gv_quota_create();
    ASSERT(mgr != NULL, "gv_quota_create should succeed");
    gv_quota_destroy(mgr);

    gv_quota_destroy(NULL);
    return 0;
}

static int test_set_get_remove(void) {
    GV_QuotaManager *mgr = gv_quota_create();
    ASSERT(mgr != NULL, "create");

    GV_QuotaConfig cfg;
    gv_quota_config_init(&cfg);
    cfg.max_vectors = 1000;
    cfg.max_memory_bytes = 1024 * 1024;

    ASSERT(gv_quota_set(mgr, "tenant_1", &cfg) == 0, "set quota");

    GV_QuotaConfig out;
    memset(&out, 0, sizeof(out));
    ASSERT(gv_quota_get(mgr, "tenant_1", &out) == 0, "get quota");
    ASSERT(out.max_vectors == 1000, "max_vectors should be 1000");
    ASSERT(out.max_memory_bytes == 1024 * 1024, "max_memory_bytes should match");

    ASSERT(gv_quota_remove(mgr, "tenant_1") == 0, "remove quota");
    ASSERT(gv_quota_get(mgr, "tenant_1", &out) != 0, "get after remove should fail");

    gv_quota_destroy(mgr);
    return 0;
}

static int test_check_insert_under_limit(void) {
    GV_QuotaManager *mgr = gv_quota_create();
    ASSERT(mgr != NULL, "create");

    GV_QuotaConfig cfg;
    gv_quota_config_init(&cfg);
    cfg.max_vectors = 100;
    ASSERT(gv_quota_set(mgr, "t1", &cfg) == 0, "set quota");

    GV_QuotaResult result = gv_quota_check_insert(mgr, "t1", 10);
    ASSERT(result == GV_QUOTA_OK, "insert of 10 should be OK under limit of 100");

    gv_quota_destroy(mgr);
    return 0;
}

static int test_check_insert_over_limit(void) {
    GV_QuotaManager *mgr = gv_quota_create();
    ASSERT(mgr != NULL, "create");

    GV_QuotaConfig cfg;
    gv_quota_config_init(&cfg);
    cfg.max_vectors = 5;
    ASSERT(gv_quota_set(mgr, "t2", &cfg) == 0, "set quota");

    /* Record existing usage to fill up */
    ASSERT(gv_quota_record_insert(mgr, "t2", 5, 500) == 0, "record 5 inserts");

    GV_QuotaResult result = gv_quota_check_insert(mgr, "t2", 1);
    ASSERT(result == GV_QUOTA_EXCEEDED || result == GV_QUOTA_THROTTLED,
           "insert beyond limit should be throttled or exceeded");

    gv_quota_destroy(mgr);
    return 0;
}

static int test_record_and_usage(void) {
    GV_QuotaManager *mgr = gv_quota_create();
    ASSERT(mgr != NULL, "create");

    GV_QuotaConfig cfg;
    gv_quota_config_init(&cfg);
    cfg.max_vectors = 1000;
    ASSERT(gv_quota_set(mgr, "t3", &cfg) == 0, "set quota");

    ASSERT(gv_quota_record_insert(mgr, "t3", 10, 4096) == 0, "record inserts");
    ASSERT(gv_quota_record_query(mgr, "t3") == 0, "record query");

    GV_QuotaUsage usage;
    memset(&usage, 0, sizeof(usage));
    ASSERT(gv_quota_get_usage(mgr, "t3", &usage) == 0, "get_usage");
    ASSERT(usage.current_vectors == 10, "vectors should be 10");
    ASSERT(usage.current_memory_bytes == 4096, "memory should be 4096");

    /* Record deletion */
    ASSERT(gv_quota_record_delete(mgr, "t3", 3, 1024) == 0, "record delete");
    ASSERT(gv_quota_get_usage(mgr, "t3", &usage) == 0, "get_usage after delete");
    ASSERT(usage.current_vectors == 7, "vectors should be 7 after delete");

    gv_quota_destroy(mgr);
    return 0;
}

static int test_reset_usage(void) {
    GV_QuotaManager *mgr = gv_quota_create();
    ASSERT(mgr != NULL, "create");

    GV_QuotaConfig cfg;
    gv_quota_config_init(&cfg);
    ASSERT(gv_quota_set(mgr, "t4", &cfg) == 0, "set quota");
    ASSERT(gv_quota_record_insert(mgr, "t4", 50, 8192) == 0, "record inserts");

    ASSERT(gv_quota_reset_usage(mgr, "t4") == 0, "reset_usage");

    GV_QuotaUsage usage;
    memset(&usage, 0, sizeof(usage));
    ASSERT(gv_quota_get_usage(mgr, "t4", &usage) == 0, "get_usage after reset");
    ASSERT(usage.current_vectors == 0, "vectors should be 0 after reset");
    ASSERT(usage.current_memory_bytes == 0, "memory should be 0 after reset");

    gv_quota_destroy(mgr);
    return 0;
}

static int test_check_query(void) {
    GV_QuotaManager *mgr = gv_quota_create();
    ASSERT(mgr != NULL, "create");

    GV_QuotaConfig cfg;
    gv_quota_config_init(&cfg);
    cfg.max_qps = 0.0; /* unlimited */
    ASSERT(gv_quota_set(mgr, "t5", &cfg) == 0, "set quota");

    GV_QuotaResult result = gv_quota_check_query(mgr, "t5");
    ASSERT(result == GV_QUOTA_OK, "query with unlimited qps should be OK");

    gv_quota_destroy(mgr);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config_init...",             test_config_init},
        {"Testing create_destroy...",          test_create_destroy},
        {"Testing set_get_remove...",          test_set_get_remove},
        {"Testing check_insert_under_limit..", test_check_insert_under_limit},
        {"Testing check_insert_over_limit..",  test_check_insert_over_limit},
        {"Testing record_and_usage...",        test_record_and_usage},
        {"Testing reset_usage...",             test_reset_usage},
        {"Testing check_query...",             test_check_query},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
