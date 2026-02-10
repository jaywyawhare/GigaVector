#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_timetravel.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ------------------------------------------------------------------ */
static int test_config_init(void) {
    GV_TimeTravelConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_tt_config_init(&cfg);
    ASSERT(cfg.max_versions == 1000, "max_versions default should be 1000");
    ASSERT(cfg.max_storage_mb == 512, "max_storage_mb default should be 512");
    ASSERT(cfg.auto_gc == 1,          "auto_gc default should be 1");
    ASSERT(cfg.gc_keep_count == 100,  "gc_keep_count default should be 100");
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_create_destroy(void) {
    GV_TimeTravelManager *mgr = gv_tt_create(NULL);
    ASSERT(mgr != NULL, "gv_tt_create(NULL) should succeed");
    gv_tt_destroy(mgr);

    GV_TimeTravelConfig cfg;
    gv_tt_config_init(&cfg);
    cfg.max_versions = 500;
    mgr = gv_tt_create(&cfg);
    ASSERT(mgr != NULL, "gv_tt_create with config should succeed");
    gv_tt_destroy(mgr);

    /* Destroy NULL is safe */
    gv_tt_destroy(NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_record_insert(void) {
    GV_TimeTravelManager *mgr = gv_tt_create(NULL);
    ASSERT(mgr != NULL, "create");

    float vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t v1 = gv_tt_record_insert(mgr, 0, vec, 4);
    ASSERT(v1 > 0, "record_insert should return non-zero version");

    float vec2[] = {5.0f, 6.0f, 7.0f, 8.0f};
    uint64_t v2 = gv_tt_record_insert(mgr, 1, vec2, 4);
    ASSERT(v2 > v1, "second insert version should be greater than first");

    ASSERT(gv_tt_current_version(mgr) == v2, "current_version should be latest");

    gv_tt_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_record_update(void) {
    GV_TimeTravelManager *mgr = gv_tt_create(NULL);
    ASSERT(mgr != NULL, "create");

    float old_vec[] = {1.0f, 0.0f};
    float new_vec[] = {0.0f, 1.0f};

    /* Insert first so the index exists conceptually */
    uint64_t v1 = gv_tt_record_insert(mgr, 0, old_vec, 2);
    ASSERT(v1 > 0, "insert");

    uint64_t v2 = gv_tt_record_update(mgr, 0, old_vec, new_vec, 2);
    ASSERT(v2 > v1, "update version should be greater");

    gv_tt_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_record_delete(void) {
    GV_TimeTravelManager *mgr = gv_tt_create(NULL);
    ASSERT(mgr != NULL, "create");

    float vec[] = {3.0f, 4.0f};
    uint64_t v1 = gv_tt_record_insert(mgr, 0, vec, 2);
    ASSERT(v1 > 0, "insert");

    uint64_t v2 = gv_tt_record_delete(mgr, 0, vec, 2);
    ASSERT(v2 > v1, "delete version should be greater");

    gv_tt_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_query_at_version(void) {
    GV_TimeTravelManager *mgr = gv_tt_create(NULL);
    ASSERT(mgr != NULL, "create");

    float vec1[] = {1.0f, 2.0f};
    uint64_t v1 = gv_tt_record_insert(mgr, 0, vec1, 2);
    ASSERT(v1 > 0, "insert v1");

    float vec2[] = {10.0f, 20.0f};
    uint64_t v2 = gv_tt_record_update(mgr, 0, vec1, vec2, 2);
    ASSERT(v2 > 0, "update to v2");

    /* Query at the latest version should give updated vector */
    float out[2] = {0};
    int found = gv_tt_query_at_version(mgr, v2, 0, out, 2);
    ASSERT(found == 1, "should find vector at v2");
    ASSERT(out[0] == 10.0f && out[1] == 20.0f, "data at v2 should be updated values");

    /* Query at the original version should give original vector */
    float out_old[2] = {0};
    found = gv_tt_query_at_version(mgr, v1, 0, out_old, 2);
    ASSERT(found == 1, "should find vector at v1");
    ASSERT(out_old[0] == 1.0f && out_old[1] == 2.0f, "data at v1 should be original values");

    gv_tt_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_list_versions(void) {
    GV_TimeTravelManager *mgr = gv_tt_create(NULL);
    ASSERT(mgr != NULL, "create");

    float vec[] = {0.5f};
    gv_tt_record_insert(mgr, 0, vec, 1);
    gv_tt_record_insert(mgr, 1, vec, 1);
    gv_tt_record_insert(mgr, 2, vec, 1);

    GV_VersionEntry entries[10];
    memset(entries, 0, sizeof(entries));
    int count = gv_tt_list_versions(mgr, entries, 10);
    ASSERT(count == 3, "should list 3 versions");

    /* Versions should be ordered oldest to newest */
    ASSERT(entries[0].version_id < entries[1].version_id, "v0 < v1");
    ASSERT(entries[1].version_id < entries[2].version_id, "v1 < v2");

    gv_tt_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_gc(void) {
    GV_TimeTravelConfig cfg;
    gv_tt_config_init(&cfg);
    cfg.max_versions = 5;
    cfg.gc_keep_count = 2;
    cfg.auto_gc = 0; /* manual GC for test control */

    GV_TimeTravelManager *mgr = gv_tt_create(&cfg);
    ASSERT(mgr != NULL, "create");

    float vec[] = {1.0f};
    for (int i = 0; i < 10; i++) {
        uint64_t v = gv_tt_record_insert(mgr, (size_t)i, vec, 1);
        ASSERT(v > 0, "insert");
    }

    int removed = gv_tt_gc(mgr);
    ASSERT(removed >= 0, "gc should not error");

    /* After GC, at most max_versions entries remain (but at least gc_keep_count) */
    GV_VersionEntry entries[20];
    int remaining = gv_tt_list_versions(mgr, entries, 20);
    ASSERT(remaining >= 0, "list_versions after gc");
    ASSERT((size_t)remaining <= cfg.max_versions, "remaining should be <= max_versions");

    gv_tt_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_save_load(void) {
    const char *path = "tmp_timetravel_test.bin";
    remove(path);

    GV_TimeTravelManager *mgr = gv_tt_create(NULL);
    ASSERT(mgr != NULL, "create");

    float vec[] = {1.0f, 2.0f};
    gv_tt_record_insert(mgr, 0, vec, 2);
    gv_tt_record_insert(mgr, 1, vec, 2);
    uint64_t saved_version = gv_tt_current_version(mgr);

    ASSERT(gv_tt_save(mgr, path) == 0, "save");
    gv_tt_destroy(mgr);

    GV_TimeTravelManager *loaded = gv_tt_load(path);
    ASSERT(loaded != NULL, "load should succeed");
    ASSERT(gv_tt_current_version(loaded) == saved_version,
           "loaded current_version should match saved");

    GV_VersionEntry entries[10];
    int count = gv_tt_list_versions(loaded, entries, 10);
    ASSERT(count == 2, "loaded should have 2 versions");

    gv_tt_destroy(loaded);
    remove(path);
    return 0;
}

/* ================================================================== */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config_init...",       test_config_init},
        {"Testing create_destroy...",    test_create_destroy},
        {"Testing record_insert...",     test_record_insert},
        {"Testing record_update...",     test_record_update},
        {"Testing record_delete...",     test_record_delete},
        {"Testing query_at_version...",  test_query_at_version},
        {"Testing list_versions...",     test_list_versions},
        {"Testing gc...",                test_gc},
        {"Testing save_load...",         test_save_load},
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
