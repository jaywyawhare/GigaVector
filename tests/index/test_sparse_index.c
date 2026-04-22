#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "index/sparse_index.h"
#include "storage/sparse_vector.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_sparse_index_create(void) {
    GV_SparseIndex *idx = sparse_index_create(100);
    ASSERT(idx != NULL, "create sparse index");
    sparse_index_destroy(idx);
    return 0;
}

static int test_sparse_index_create_zero_dim(void) {
    GV_SparseIndex *idx = sparse_index_create(0);
    ASSERT(idx == NULL, "zero dimension fails");
    return 0;
}

static int test_sparse_index_destroy_null(void) {
    sparse_index_destroy(NULL);
    return 0;
}

static int test_sparse_index_add_null(void) {
    GV_SparseIndex *idx = sparse_index_create(10);
    ASSERT(idx != NULL, "create index");
    int rc = sparse_index_add(idx, NULL);
    ASSERT(rc == -1, "add null vector fails");
    sparse_index_destroy(idx);
    return 0;
}

static int test_sparse_index_add(void) {
    GV_SparseIndex *idx = sparse_index_create(10);
    ASSERT(idx != NULL, "create index");
    uint32_t idx_arr[2] = {1, 5};
    float val_arr[2] = {1.0f, 2.0f};
    GV_SparseVector *sv = sparse_vector_create(10, idx_arr, val_arr, 2);
    ASSERT(sv != NULL, "create sparse vector");
    int rc = sparse_index_add(idx, sv);
    ASSERT(rc == 0, "add succeeds");
    sparse_index_destroy(idx);
    return 0;
}

static int test_sparse_index_add_dimension_mismatch(void) {
    GV_SparseIndex *idx = sparse_index_create(10);
    ASSERT(idx != NULL, "create index");
    uint32_t idx_arr[2] = {1, 5};
    float val_arr[2] = {1.0f, 2.0f};
    GV_SparseVector *sv = sparse_vector_create(100, idx_arr, val_arr, 2);
    int rc = sparse_index_add(idx, sv);
    ASSERT(rc == -1, "dimension mismatch fails");
    sparse_vector_destroy(sv);
    sparse_index_destroy(idx);
    return 0;
}

static int test_sparse_index_delete(void) {
    GV_SparseIndex *idx = sparse_index_create(10);
    ASSERT(idx != NULL, "create index");
    uint32_t idx_arr[2] = {1, 5};
    float val_arr[2] = {1.0f, 2.0f};
    GV_SparseVector *sv = sparse_vector_create(10, idx_arr, val_arr, 2);
    sparse_index_add(idx, sv);
    int rc = sparse_index_delete(idx, 0);
    ASSERT(rc == 0, "delete succeeds");
    sparse_index_destroy(idx);
    return 0;
}

static int test_sparse_index_delete_oob(void) {
    GV_SparseIndex *idx = sparse_index_create(10);
    int rc = sparse_index_delete(idx, 100);
    ASSERT(rc == -1, "delete oob fails");
    sparse_index_destroy(idx);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"sparse_index_create", test_sparse_index_create},
        {"sparse_index_create_zero_dim", test_sparse_index_create_zero_dim},
        {"sparse_index_destroy_null", test_sparse_index_destroy_null},
        {"sparse_index_add_null", test_sparse_index_add_null},
        {"sparse_index_add", test_sparse_index_add},
        {"sparse_index_add_dimension_mismatch", test_sparse_index_add_dimension_mismatch},
        {"sparse_index_delete", test_sparse_index_delete},
        {"sparse_index_delete_oob", test_sparse_index_delete_oob},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) passed++;
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}