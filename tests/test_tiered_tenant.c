#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_tiered_tenant.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_init(void) {
    GV_TieredTenantConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_tiered_config_init(&cfg);
    ASSERT(cfg.thresholds.shared_max_vectors == 10000,     "shared_max_vectors default");
    ASSERT(cfg.thresholds.dedicated_max_vectors == 1000000, "dedicated_max_vectors default");
    ASSERT(cfg.thresholds.shared_max_memory_mb == 64,       "shared_max_memory_mb default");
    ASSERT(cfg.thresholds.dedicated_max_memory_mb == 1024,  "dedicated_max_memory_mb default");
    ASSERT(cfg.auto_promote == 1,                           "auto_promote default");
    ASSERT(cfg.auto_demote == 0,                            "auto_demote default");
    ASSERT(cfg.max_shared_tenants == 1000,                  "max_shared_tenants default");
    ASSERT(cfg.max_total_tenants == 10000,                  "max_total_tenants default");
    return 0;
}

static int test_create_destroy(void) {
    GV_TieredManager *mgr = gv_tiered_create(NULL);
    ASSERT(mgr != NULL, "gv_tiered_create(NULL) should succeed");
    gv_tiered_destroy(mgr);

    GV_TieredTenantConfig cfg;
    gv_tiered_config_init(&cfg);
    mgr = gv_tiered_create(&cfg);
    ASSERT(mgr != NULL, "gv_tiered_create with config should succeed");
    gv_tiered_destroy(mgr);

    /* Destroy NULL is safe */
    gv_tiered_destroy(NULL);
    return 0;
}

static int test_add_and_get_info(void) {
    GV_TieredManager *mgr = gv_tiered_create(NULL);
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_tiered_add_tenant(mgr, "small_co", GV_TIER_SHARED) == 0, "add tenant");

    GV_TenantInfo info;
    memset(&info, 0, sizeof(info));
    ASSERT(gv_tiered_get_info(mgr, "small_co", &info) == 0, "get_info");
    ASSERT(info.tier == GV_TIER_SHARED, "tier should be SHARED");
    ASSERT(info.vector_count == 0, "initial vector count should be 0");

    /* Unknown tenant */
    ASSERT(gv_tiered_get_info(mgr, "ghost", &info) == -1, "get_info for unknown should fail");

    gv_tiered_destroy(mgr);
    return 0;
}

static int test_remove_tenant(void) {
    GV_TieredManager *mgr = gv_tiered_create(NULL);
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_tiered_add_tenant(mgr, "rm_me", GV_TIER_SHARED) == 0, "add");
    ASSERT(gv_tiered_tenant_count(mgr) == 1, "count should be 1");

    ASSERT(gv_tiered_remove_tenant(mgr, "rm_me") == 0, "remove");
    ASSERT(gv_tiered_tenant_count(mgr) == 0, "count should be 0 after remove");

    /* Remove unknown => error */
    ASSERT(gv_tiered_remove_tenant(mgr, "rm_me") == -1, "double remove should fail");

    gv_tiered_destroy(mgr);
    return 0;
}

static int test_promote(void) {
    GV_TieredManager *mgr = gv_tiered_create(NULL);
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_tiered_add_tenant(mgr, "growing", GV_TIER_SHARED) == 0, "add");
    ASSERT(gv_tiered_promote(mgr, "growing", GV_TIER_DEDICATED) == 0, "promote");

    GV_TenantInfo info;
    memset(&info, 0, sizeof(info));
    ASSERT(gv_tiered_get_info(mgr, "growing", &info) == 0, "get_info");
    ASSERT(info.tier == GV_TIER_DEDICATED, "tier should be DEDICATED after promotion");

    ASSERT(gv_tiered_promote(mgr, "growing", GV_TIER_PREMIUM) == 0, "promote to premium");
    ASSERT(gv_tiered_get_info(mgr, "growing", &info) == 0, "get_info");
    ASSERT(info.tier == GV_TIER_PREMIUM, "tier should be PREMIUM");

    gv_tiered_destroy(mgr);
    return 0;
}

static int test_record_usage_and_auto_promote(void) {
    GV_TieredTenantConfig cfg;
    gv_tiered_config_init(&cfg);
    cfg.thresholds.shared_max_vectors = 5; /* low threshold for testing */
    cfg.auto_promote = 1;

    GV_TieredManager *mgr = gv_tiered_create(&cfg);
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_tiered_add_tenant(mgr, "burst", GV_TIER_SHARED) == 0, "add");
    ASSERT(gv_tiered_record_usage(mgr, "burst", 10, 4096) == 0, "record usage");

    int promoted = gv_tiered_check_promote(mgr);
    ASSERT(promoted >= 0, "check_promote should not error");

    /* Tenant should have been auto-promoted past shared */
    GV_TenantInfo info;
    memset(&info, 0, sizeof(info));
    ASSERT(gv_tiered_get_info(mgr, "burst", &info) == 0, "get_info");
    /* With 10 vectors exceeding shared_max_vectors=5, expect promotion */
    ASSERT(info.tier >= GV_TIER_DEDICATED,
           "tenant exceeding shared threshold should be promoted");

    gv_tiered_destroy(mgr);
    return 0;
}

static int test_list_tenants(void) {
    GV_TieredManager *mgr = gv_tiered_create(NULL);
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_tiered_add_tenant(mgr, "s1", GV_TIER_SHARED) == 0, "add s1");
    ASSERT(gv_tiered_add_tenant(mgr, "s2", GV_TIER_SHARED) == 0, "add s2");
    ASSERT(gv_tiered_add_tenant(mgr, "d1", GV_TIER_DEDICATED) == 0, "add d1");

    ASSERT(gv_tiered_tenant_count(mgr) == 3, "total count should be 3");

    GV_TenantInfo shared_list[10];
    int shared_count = gv_tiered_list_tenants(mgr, GV_TIER_SHARED, shared_list, 10);
    ASSERT(shared_count == 2, "should have 2 shared tenants");

    GV_TenantInfo ded_list[10];
    int ded_count = gv_tiered_list_tenants(mgr, GV_TIER_DEDICATED, ded_list, 10);
    ASSERT(ded_count == 1, "should have 1 dedicated tenant");

    gv_tiered_destroy(mgr);
    return 0;
}

static int test_save_load(void) {
    const char *path = "tmp_tiered_test.bin";
    remove(path);

    GV_TieredManager *mgr = gv_tiered_create(NULL);
    ASSERT(mgr != NULL, "create");
    ASSERT(gv_tiered_add_tenant(mgr, "persist_t", GV_TIER_DEDICATED) == 0, "add");
    ASSERT(gv_tiered_save(mgr, path) == 0, "save");
    gv_tiered_destroy(mgr);

    GV_TieredManager *loaded = gv_tiered_load(path);
    ASSERT(loaded != NULL, "load should succeed");
    ASSERT(gv_tiered_tenant_count(loaded) == 1, "loaded count should be 1");

    GV_TenantInfo info;
    memset(&info, 0, sizeof(info));
    ASSERT(gv_tiered_get_info(loaded, "persist_t", &info) == 0, "get_info after load");
    ASSERT(info.tier == GV_TIER_DEDICATED, "tier should be DEDICATED after load");

    gv_tiered_destroy(loaded);
    remove(path);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config_init...",            test_config_init},
        {"Testing create_destroy...",         test_create_destroy},
        {"Testing add_and_get_info...",       test_add_and_get_info},
        {"Testing remove_tenant...",          test_remove_tenant},
        {"Testing promote...",                test_promote},
        {"Testing record_usage_auto_promote..",test_record_usage_and_auto_promote},
        {"Testing list_tenants...",           test_list_tenants},
        {"Testing save_load...",              test_save_load},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
