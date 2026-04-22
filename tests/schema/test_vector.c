#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "schema/vector.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_vector_create(void) {
    GV_Vector *v = vector_create(4);
    ASSERT(v != NULL, "create vector");
    ASSERT(v->dimension == 4, "dimension matches");
    ASSERT(v->data != NULL, "data is allocated");
    ASSERT(v->metadata == NULL, "no initial metadata");
    vector_destroy(v);
    return 0;
}

static int test_vector_create_zero_dim(void) {
    GV_Vector *v = vector_create(0);
    ASSERT(v == NULL, "zero dimension fails");
    return 0;
}

static int test_vector_create_from_data(void) {
    float data[4] = {1, 2, 3, 4};
    GV_Vector *v = vector_create_from_data(4, data);
    ASSERT(v != NULL, "create from data");
    ASSERT(v->data[0] == 1 && v->data[3] == 4, "data copied");
    vector_destroy(v);
    return 0;
}

static int test_vector_create_from_data_null(void) {
    GV_Vector *v = vector_create_from_data(0, NULL);
    ASSERT(v == NULL, "zero dim fails");
    v = vector_create_from_data(4, NULL);
    ASSERT(v == NULL, "null data fails");
    return 0;
}

static int test_vector_set_get(void) {
    GV_Vector *v = vector_create(4);
    ASSERT(v != NULL, "create vector");
    int rc = vector_set(v, 0, 42.0f);
    ASSERT(rc == 0, "set succeeds");
    float val = 0;
    rc = vector_get(v, 0, &val);
    ASSERT(rc == 0, "get succeeds");
    ASSERT(val == 42.0f, "value matches");
    rc = vector_set(v, 10, 1.0f);
    ASSERT(rc == -1, "set oob fails");
    vector_destroy(v);
    return 0;
}

static int test_vector_clear(void) {
    GV_Vector *v = vector_create_from_data(3, (float[]){5, 6, 7});
    int rc = vector_clear(v);
    ASSERT(rc == 0, "clear succeeds");
    ASSERT(v->data[0] == 0 && v->data[1] == 0 && v->data[2] == 0, "all zeros");
    vector_destroy(v);
    return 0;
}

static int test_vector_destroy_null(void) {
    vector_destroy(NULL);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"vector_create", test_vector_create},
        {"vector_create_zero_dim", test_vector_create_zero_dim},
        {"vector_create_from_data", test_vector_create_from_data},
        {"vector_create_from_data_null", test_vector_create_from_data_null},
        {"vector_set_get", test_vector_set_get},
        {"vector_clear", test_vector_clear},
        {"vector_destroy_null", test_vector_destroy_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) passed++;
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}