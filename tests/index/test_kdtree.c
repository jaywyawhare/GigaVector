#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "index/kdtree.h"
#include "storage/soa_storage.h"
#include "schema/vector.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_kdtree_insert_null(void) {
    GV_SoAStorage *storage = soa_storage_create(4, 0);
    ASSERT(storage != NULL, "create storage");
    int rc = kdtree_insert(NULL, storage, 0, 0);
    ASSERT(rc == -1, "null root fails");
    soa_storage_destroy(storage);
    return 0;
}

static int test_kdtree_insert_oob(void) {
    GV_SoAStorage *storage = soa_storage_create(4, 0);
    ASSERT(storage != NULL, "create storage");
    float data[4] = {1,2,3,4};
    soa_storage_add(storage, data, NULL);
    GV_KDNode *root = NULL;
    int rc = kdtree_insert(&root, storage, 10, 0);
    ASSERT(rc == -1, "out of bounds index fails");
    soa_storage_destroy(storage);
    return 0;
}

static int test_kdtree_insert_basic(void) {
    GV_SoAStorage *storage = soa_storage_create(4, 0);
    ASSERT(storage != NULL, "create storage");
    float d1[4] = {1,2,3,4};
    float d2[4] = {4,3,2,1};
    soa_storage_add(storage, d1, NULL);
    soa_storage_add(storage, d2, NULL);
    GV_KDNode *root = NULL;
    int rc = kdtree_insert(&root, storage, 0, 0);
    ASSERT(rc == 0, "insert first");
    kdtree_destroy_recursive(root);
    soa_storage_destroy(storage);
    return 0;
}

static int test_kdtree_destroy_null(void) {
    kdtree_destroy_recursive(NULL);
    return 0;
}

static int test_kdtree_destroy_recursive(void) {
    GV_SoAStorage *storage = soa_storage_create(2, 0);
    float d1[2] = {1,2};
    float d2[2] = {3,4};
    soa_storage_add(storage, d1, NULL);
    soa_storage_add(storage, d2, NULL);
    GV_KDNode *root = NULL;
    kdtree_insert(&root, storage, 0, 0);
    kdtree_insert(&root, storage, 1, 0);
    kdtree_destroy_recursive(root);
    soa_storage_destroy(storage);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"kdtree_insert_null", test_kdtree_insert_null},
        {"kdtree_insert_oob", test_kdtree_insert_oob},
        {"kdtree_insert_basic", test_kdtree_insert_basic},
        {"kdtree_destroy_null", test_kdtree_destroy_null},
        {"kdtree_destroy_recursive", test_kdtree_destroy_recursive},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) passed++;
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}