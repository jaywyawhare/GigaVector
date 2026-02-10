#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_alias.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ------------------------------------------------------------------ */
static int test_create_destroy(void) {
    GV_AliasManager *mgr = gv_alias_manager_create();
    ASSERT(mgr != NULL, "gv_alias_manager_create should succeed");
    gv_alias_manager_destroy(mgr);

    /* Destroy NULL is safe */
    gv_alias_manager_destroy(NULL);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_create_and_resolve(void) {
    GV_AliasManager *mgr = gv_alias_manager_create();
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_alias_create(mgr, "prod", "collection_v2") == 0, "create alias");

    const char *resolved = gv_alias_resolve(mgr, "prod");
    ASSERT(resolved != NULL, "resolve should succeed");
    ASSERT(strcmp(resolved, "collection_v2") == 0, "resolved should match collection_v2");

    /* Non-existent alias */
    ASSERT(gv_alias_resolve(mgr, "unknown") == NULL, "resolve unknown should return NULL");

    gv_alias_manager_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_update(void) {
    GV_AliasManager *mgr = gv_alias_manager_create();
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_alias_create(mgr, "live", "v1") == 0, "create alias");
    ASSERT(gv_alias_update(mgr, "live", "v2") == 0, "update alias");

    const char *resolved = gv_alias_resolve(mgr, "live");
    ASSERT(resolved != NULL, "resolve after update");
    ASSERT(strcmp(resolved, "v2") == 0, "should resolve to v2 after update");

    gv_alias_manager_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_delete_and_exists(void) {
    GV_AliasManager *mgr = gv_alias_manager_create();
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_alias_create(mgr, "tmp", "c1") == 0, "create alias");
    ASSERT(gv_alias_exists(mgr, "tmp") == 1, "should exist");

    ASSERT(gv_alias_delete(mgr, "tmp") == 0, "delete alias");
    ASSERT(gv_alias_exists(mgr, "tmp") == 0, "should not exist after delete");

    gv_alias_manager_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_swap(void) {
    GV_AliasManager *mgr = gv_alias_manager_create();
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_alias_create(mgr, "blue", "collection_a") == 0, "create blue");
    ASSERT(gv_alias_create(mgr, "green", "collection_b") == 0, "create green");

    ASSERT(gv_alias_swap(mgr, "blue", "green") == 0, "swap");

    const char *blue_target = gv_alias_resolve(mgr, "blue");
    const char *green_target = gv_alias_resolve(mgr, "green");
    ASSERT(blue_target != NULL && strcmp(blue_target, "collection_b") == 0,
           "blue should now point to collection_b");
    ASSERT(green_target != NULL && strcmp(green_target, "collection_a") == 0,
           "green should now point to collection_a");

    gv_alias_manager_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_count_and_list(void) {
    GV_AliasManager *mgr = gv_alias_manager_create();
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_alias_count(mgr) == 0, "initial count should be 0");

    ASSERT(gv_alias_create(mgr, "a1", "c1") == 0, "create a1");
    ASSERT(gv_alias_create(mgr, "a2", "c2") == 0, "create a2");
    ASSERT(gv_alias_create(mgr, "a3", "c3") == 0, "create a3");
    ASSERT(gv_alias_count(mgr) == 3, "count should be 3");

    GV_AliasInfo *list = NULL;
    size_t count = 0;
    ASSERT(gv_alias_list(mgr, &list, &count) == 0, "list");
    ASSERT(count == 3, "list count should be 3");
    gv_alias_free_list(list, count);

    gv_alias_manager_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_get_info(void) {
    GV_AliasManager *mgr = gv_alias_manager_create();
    ASSERT(mgr != NULL, "create");

    ASSERT(gv_alias_create(mgr, "info_alias", "target_col") == 0, "create alias");

    GV_AliasInfo info;
    memset(&info, 0, sizeof(info));
    ASSERT(gv_alias_get_info(mgr, "info_alias", &info) == 0, "get_info");
    ASSERT(info.alias_name != NULL, "alias_name should be set");
    ASSERT(info.collection_name != NULL, "collection_name should be set");
    ASSERT(strcmp(info.collection_name, "target_col") == 0,
           "collection_name should match");

    /* Note: info fields are owned by the manager; we do not free them here */
    gv_alias_manager_destroy(mgr);
    return 0;
}

/* ------------------------------------------------------------------ */
static int test_save_load(void) {
    const char *path = "tmp_alias_test.bin";
    remove(path);

    GV_AliasManager *mgr = gv_alias_manager_create();
    ASSERT(mgr != NULL, "create");
    ASSERT(gv_alias_create(mgr, "saved", "my_collection") == 0, "create alias");
    ASSERT(gv_alias_save(mgr, path) == 0, "save");
    gv_alias_manager_destroy(mgr);

    GV_AliasManager *loaded = gv_alias_load(path);
    ASSERT(loaded != NULL, "load should succeed");
    ASSERT(gv_alias_exists(loaded, "saved") == 1, "alias should exist after load");

    const char *resolved = gv_alias_resolve(loaded, "saved");
    ASSERT(resolved != NULL && strcmp(resolved, "my_collection") == 0,
           "resolved target should match after load");

    gv_alias_manager_destroy(loaded);
    remove(path);
    return 0;
}

/* ================================================================== */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing create_destroy...",    test_create_destroy},
        {"Testing create_and_resolve..", test_create_and_resolve},
        {"Testing update...",            test_update},
        {"Testing delete_and_exists...", test_delete_and_exists},
        {"Testing swap...",              test_swap},
        {"Testing count_and_list...",    test_count_and_list},
        {"Testing get_info...",          test_get_info},
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
