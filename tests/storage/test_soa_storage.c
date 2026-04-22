#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/soa_storage.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_soa_storage_create(void) {
    GV_SoAStorage *s = soa_storage_create(4, 0);
    ASSERT(s != NULL, "create soa storage");
    ASSERT(s->dimension == 4, "dimension matches");
    ASSERT(s->count == 0, "initial count is 0");
    soa_storage_destroy(s);
    return 0;
}

static int test_soa_storage_create_zero_dim(void) {
    GV_SoAStorage *s = soa_storage_create(0, 0);
    ASSERT(s == NULL, "zero dimension fails");
    return 0;
}

static int test_soa_storage_add(void) {
    GV_SoAStorage *s = soa_storage_create(4, 0);
    ASSERT(s != NULL, "create storage");
    float data[4] = {1,2,3,4};
    size_t idx = soa_storage_add(s, data, NULL);
    ASSERT(idx == 0, "add returns index 0");
    ASSERT(soa_storage_count(s) == 1, "count is 1");
    soa_storage_destroy(s);
    return 0;
}

static int test_soa_storage_add_null(void) {
    GV_SoAStorage *s = soa_storage_create(4, 0);
    size_t idx = soa_storage_add(s, NULL, NULL);
    ASSERT(idx == (size_t)-1, "add null data fails");
    soa_storage_destroy(s);
    return 0;
}

static int test_soa_storage_get_data(void) {
    GV_SoAStorage *s = soa_storage_create(2, 0);
    float d1[2] = {1.5f, 2.5f};
    soa_storage_add(s, d1, NULL);
    const float *ret = soa_storage_get_data(s, 0);
    ASSERT(ret != NULL, "get_data returns");
    ASSERT(ret[0] == 1.5f, "first element matches");
    ASSERT(ret[1] == 2.5f, "second element matches");
    soa_storage_destroy(s);
    return 0;
}

static int test_soa_storage_get_data_oob(void) {
    GV_SoAStorage *s = soa_storage_create(2, 0);
    const float *ret = soa_storage_get_data(s, 100);
    ASSERT(ret == NULL, "oob returns null");
    soa_storage_destroy(s);
    return 0;
}

static int test_soa_storage_dimension(void) {
    GV_SoAStorage *s = soa_storage_create(8, 0);
    ASSERT(soa_storage_dimension(s) == 8, "dimension matches");
    soa_storage_destroy(s);
    return 0;
}

static int test_soa_storage_count(void) {
    GV_SoAStorage *s = soa_storage_create(2, 0);
    ASSERT(soa_storage_count(s) == 0, "initial count 0");
    float d[2] = {1,2};
    soa_storage_add(s, d, NULL);
    ASSERT(soa_storage_count(s) == 1, "count after add");
    soa_storage_destroy(s);
    return 0;
}

static int test_soa_storage_mark_deleted(void) {
    GV_SoAStorage *s = soa_storage_create(2, 0);
    float d[2] = {1,2};
    soa_storage_add(s, d, NULL);
    int rc = soa_storage_mark_deleted(s, 0);
    ASSERT(rc == 0, "mark_deleted succeeds");
    ASSERT(soa_storage_is_deleted(s, 0) == 1, "is_deleted returns true");
    soa_storage_destroy(s);
    return 0;
}

static int test_soa_storage_update_data(void) {
    GV_SoAStorage *s = soa_storage_create(2, 0);
    float d[2] = {1,2};
    soa_storage_add(s, d, NULL);
    float newd[2] = {9,9};
    int rc = soa_storage_update_data(s, 0, newd);
    ASSERT(rc == 0, "update_data succeeds");
    const float *ret = soa_storage_get_data(s, 0);
    ASSERT(ret[0] == 9 && ret[1] == 9, "data updated");
    soa_storage_destroy(s);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"soa_storage_create", test_soa_storage_create},
        {"soa_storage_create_zero_dim", test_soa_storage_create_zero_dim},
        {"soa_storage_add", test_soa_storage_add},
        {"soa_storage_add_null", test_soa_storage_add_null},
        {"soa_storage_get_data", test_soa_storage_get_data},
        {"soa_storage_get_data_oob", test_soa_storage_get_data_oob},
        {"soa_storage_dimension", test_soa_storage_dimension},
        {"soa_storage_count", test_soa_storage_count},
        {"soa_storage_mark_deleted", test_soa_storage_mark_deleted},
        {"soa_storage_update_data", test_soa_storage_update_data},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) passed++;
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}