#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/sparse_vector.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_sparse_vector_create(void) {
    uint32_t indices[2] = {1, 5};
    float values[2] = {1.5f, 2.5f};
    GV_SparseVector *sv = sparse_vector_create(10, indices, values, 2);
    ASSERT(sv != NULL, "create sparse vector");
    ASSERT(sv->dimension == 10, "dimension matches");
    ASSERT(sv->nnz == 2, "nnz matches");
    sparse_vector_destroy(sv);
    return 0;
}

static int test_sparse_vector_create_zero_dim(void) {
    uint32_t indices[2] = {1, 5};
    float values[2] = {1.5f, 2.5f};
    GV_SparseVector *sv = sparse_vector_create(0, indices, values, 2);
    ASSERT(sv == NULL, "zero dimension fails");
    return 0;
}

static int test_sparse_vector_create_null_arrays(void) {
    GV_SparseVector *sv = sparse_vector_create(10, NULL, NULL, 0);
    ASSERT(sv != NULL, "null arrays with nnz=0 is ok");
    sparse_vector_destroy(sv);
    return 0;
}

static int test_sparse_vector_create_null_with_nnz(void) {
    GV_SparseVector *sv = sparse_vector_create(10, NULL, NULL, 5);
    ASSERT(sv == NULL, "null arrays with nnz>0 fails");
    return 0;
}

static int test_sparse_vector_destroy_null(void) {
    sparse_vector_destroy(NULL);
    return 0;
}

static int test_sparse_vector_destroy(void) {
    uint32_t indices[2] = {1, 5};
    float values[2] = {1.5f, 2.5f};
    GV_SparseVector *sv = sparse_vector_create(10, indices, values, 2);
    sparse_vector_destroy(sv);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"sparse_vector_create", test_sparse_vector_create},
        {"sparse_vector_create_zero_dim", test_sparse_vector_create_zero_dim},
        {"sparse_vector_create_null_arrays", test_sparse_vector_create_null_arrays},
        {"sparse_vector_create_null_with_nnz", test_sparse_vector_create_null_with_nnz},
        {"sparse_vector_destroy_null", test_sparse_vector_destroy_null},
        {"sparse_vector_destroy", test_sparse_vector_destroy},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) passed++;
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}