#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "admin/versioning.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_manager_create_destroy(void) {
    GV_VersionManager *mgr = version_manager_create(10);
    ASSERT(mgr != NULL, "version_manager_create returned NULL");
    ASSERT(version_count(mgr) == 0, "initial version count should be 0");
    version_manager_destroy(mgr);
    return 0;
}

static int test_create_version(void) {
    GV_VersionManager *mgr = version_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f};
    uint64_t vid = version_create(mgr, data, 2, 4, "v1");
    ASSERT(vid > 0, "version_create should return nonzero id");
    ASSERT(version_count(mgr) == 1, "version count should be 1");

    version_manager_destroy(mgr);
    return 0;
}

static int test_get_info(void) {
    GV_VersionManager *mgr = version_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t vid = version_create(mgr, data, 1, 4, "info-test");
    ASSERT(vid > 0, "create version");

    GV_VersionInfo info;
    int rc = version_get_info(mgr, vid, &info);
    ASSERT(rc == 0, "get_info should succeed");
    ASSERT(info.version_id == vid, "info version_id matches");
    ASSERT(info.vector_count == 1, "info vector_count == 1");
    ASSERT(info.dimension == 4, "info dimension == 4");
    ASSERT(strcmp(info.label, "info-test") == 0, "label matches");

    version_manager_destroy(mgr);
    return 0;
}

static int test_get_data(void) {
    GV_VersionManager *mgr = version_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float data[8] = {10.0f, 20.0f, 30.0f, 40.0f,
                      50.0f, 60.0f, 70.0f, 80.0f};
    uint64_t vid = version_create(mgr, data, 2, 4, "data-test");
    ASSERT(vid > 0, "create version");

    size_t count_out = 0, dim_out = 0;
    float *retrieved = version_get_data(mgr, vid, &count_out, &dim_out);
    ASSERT(retrieved != NULL, "get_data should return non-NULL");
    ASSERT(count_out == 2, "count_out == 2");
    ASSERT(dim_out == 4, "dim_out == 4");
    ASSERT(retrieved[0] == 10.0f && retrieved[7] == 80.0f, "data matches");

    free(retrieved);
    version_manager_destroy(mgr);
    return 0;
}

static int test_delete_version(void) {
    GV_VersionManager *mgr = version_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t vid = version_create(mgr, data, 1, 4, "del-test");
    ASSERT(version_count(mgr) == 1, "count should be 1");

    int rc = version_delete(mgr, vid);
    ASSERT(rc == 0, "delete should succeed");
    ASSERT(version_count(mgr) == 0, "count should be 0 after delete");

    rc = version_delete(mgr, vid);
    ASSERT(rc != 0, "double delete should fail");

    version_manager_destroy(mgr);
    return 0;
}

static int test_list_versions(void) {
    GV_VersionManager *mgr = version_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float d1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float d2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float d3[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    version_create(mgr, d1, 1, 4, "v-a");
    version_create(mgr, d2, 1, 4, "v-b");
    version_create(mgr, d3, 1, 4, "v-c");

    GV_VersionInfo infos[10];
    int listed = version_list(mgr, infos, 10);
    ASSERT(listed == 3, "should list 3 versions");

    version_manager_destroy(mgr);
    return 0;
}

static int test_compare_versions(void) {
    GV_VersionManager *mgr = version_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float d1[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f};
    float d2[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                     9.0f, 9.0f, 9.0f, 9.0f,
                     0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t v1 = version_create(mgr, d1, 2, 4, "cmp-v1");
    uint64_t v2 = version_create(mgr, d2, 3, 4, "cmp-v2");
    ASSERT(v1 > 0 && v2 > 0, "create two versions");

    size_t added = 0, removed = 0, modified = 0;
    int rc = version_compare(mgr, v1, v2, &added, &removed, &modified);
    ASSERT(rc >= 0, "compare should not return error");

    version_manager_destroy(mgr);
    return 0;
}

static int test_save_load(void) {
    GV_VersionManager *mgr = version_manager_create(10);
    ASSERT(mgr != NULL, "create manager");

    float data[4] = {3.14f, 2.71f, 1.41f, 1.73f};
    uint64_t vid = version_create(mgr, data, 1, 4, "save-test");
    ASSERT(vid > 0, "create version");

    FILE *tmp = tmpfile();
    ASSERT(tmp != NULL, "tmpfile() failed");

    int rc = version_save(mgr, tmp);
    ASSERT(rc == 0, "save should succeed");
    rewind(tmp);

    GV_VersionManager *loaded = NULL;
    rc = version_load(&loaded, tmp);
    ASSERT(rc == 0, "load should succeed");
    ASSERT(loaded != NULL, "loaded manager should be non-NULL");
    ASSERT(version_count(loaded) == 1, "loaded version count == 1");

    size_t cnt = 0, dim = 0;
    float *d = version_get_data(loaded, vid, &cnt, &dim);
    ASSERT(d != NULL, "loaded data not NULL");
    ASSERT(cnt == 1 && dim == 4, "loaded count/dim correct");
    ASSERT(d[0] == 3.14f, "loaded data matches");

    free(d);
    version_manager_destroy(loaded);
    version_manager_destroy(mgr);
    fclose(tmp);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing manager create/destroy...", test_manager_create_destroy},
        {"Testing create version...", test_create_version},
        {"Testing get info...", test_get_info},
        {"Testing get data...", test_get_data},
        {"Testing delete version...", test_delete_version},
        {"Testing list versions...", test_list_versions},
        {"Testing compare versions...", test_compare_versions},
        {"Testing save/load...", test_save_load},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
