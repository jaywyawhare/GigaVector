/* tests/test_metadata_index.c — In-depth tests for the metadata index */
#include <stdio.h>
#include <string.h>
#include "gigavector/gv_metadata_index.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while (0)

static int test_create_destroy(void) {
    GV_MetadataIndex *idx = gv_metadata_index_create();
    ASSERT(idx != NULL, "create metadata index");
    gv_metadata_index_destroy(idx);
    return 0;
}

static int test_add_and_query(void) {
    GV_MetadataIndex *idx = gv_metadata_index_create();
    ASSERT(idx != NULL, "create");

    /* Add multiple vectors with same key-value */
    ASSERT(gv_metadata_index_add(idx, "color", "red", 0) == 0, "add 0 red");
    ASSERT(gv_metadata_index_add(idx, "color", "red", 5) == 0, "add 5 red");
    ASSERT(gv_metadata_index_add(idx, "color", "blue", 1) == 0, "add 1 blue");
    ASSERT(gv_metadata_index_add(idx, "color", "red", 10) == 0, "add 10 red");
    ASSERT(gv_metadata_index_add(idx, "shape", "circle", 0) == 0, "add 0 circle");

    /* Count */
    ASSERT(gv_metadata_index_count(idx, "color", "red") == 3, "3 red vectors");
    ASSERT(gv_metadata_index_count(idx, "color", "blue") == 1, "1 blue vector");
    ASSERT(gv_metadata_index_count(idx, "shape", "circle") == 1, "1 circle vector");
    ASSERT(gv_metadata_index_count(idx, "color", "green") == 0, "0 green");

    /* Query */
    size_t out[10];
    int n = gv_metadata_index_query(idx, "color", "red", out, 10);
    ASSERT(n == 3, "query returns 3");
    /* Check all expected indices are present */
    int found0 = 0, found5 = 0, found10 = 0;
    for (int i = 0; i < n; i++) {
        if (out[i] == 0) found0 = 1;
        if (out[i] == 5) found5 = 1;
        if (out[i] == 10) found10 = 1;
    }
    ASSERT(found0 && found5 && found10, "all red indices present");

    gv_metadata_index_destroy(idx);
    return 0;
}

static int test_remove_specific(void) {
    GV_MetadataIndex *idx = gv_metadata_index_create();
    gv_metadata_index_add(idx, "tag", "a", 1);
    gv_metadata_index_add(idx, "tag", "a", 2);
    gv_metadata_index_add(idx, "tag", "a", 3);

    ASSERT(gv_metadata_index_count(idx, "tag", "a") == 3, "3 before remove");
    gv_metadata_index_remove(idx, "tag", "a", 2);
    ASSERT(gv_metadata_index_count(idx, "tag", "a") == 2, "2 after remove");

    size_t out[10];
    int n = gv_metadata_index_query(idx, "tag", "a", out, 10);
    ASSERT(n == 2, "query returns 2");
    for (int i = 0; i < n; i++) {
        ASSERT(out[i] != 2, "removed index should not appear");
    }

    gv_metadata_index_destroy(idx);
    return 0;
}

static int test_remove_vector(void) {
    GV_MetadataIndex *idx = gv_metadata_index_create();
    gv_metadata_index_add(idx, "color", "red", 7);
    gv_metadata_index_add(idx, "shape", "square", 7);
    gv_metadata_index_add(idx, "color", "red", 8);

    gv_metadata_index_remove_vector(idx, 7);
    ASSERT(gv_metadata_index_count(idx, "color", "red") == 1, "1 red after remove_vector");
    ASSERT(gv_metadata_index_count(idx, "shape", "square") == 0, "0 square after remove_vector");

    gv_metadata_index_destroy(idx);
    return 0;
}

static int test_query_max_limit(void) {
    GV_MetadataIndex *idx = gv_metadata_index_create();
    for (size_t i = 0; i < 20; i++) {
        gv_metadata_index_add(idx, "k", "v", i);
    }
    ASSERT(gv_metadata_index_count(idx, "k", "v") == 20, "20 total");

    size_t out[5];
    int n = gv_metadata_index_query(idx, "k", "v", out, 5);
    ASSERT(n == 5, "capped at max_indices=5");

    gv_metadata_index_destroy(idx);
    return 0;
}

static int test_nonexistent_query(void) {
    GV_MetadataIndex *idx = gv_metadata_index_create();
    size_t out[10];
    int n = gv_metadata_index_query(idx, "nope", "nada", out, 10);
    ASSERT(n == 0, "empty query returns 0");
    gv_metadata_index_destroy(idx);
    return 0;
}

int main(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"create_destroy", test_create_destroy},
        {"add_and_query", test_add_and_query},
        {"remove_specific", test_remove_specific},
        {"remove_vector", test_remove_vector},
        {"query_max_limit", test_query_max_limit},
        {"nonexistent_query", test_nonexistent_query},
    };
    size_t n = sizeof(tests) / sizeof(tests[0]);
    for (size_t i = 0; i < n; i++) {
        printf("  %s ... ", tests[i].name);
        if (tests[i].fn() == 0) printf("OK\n"); else { printf("FAILED\n"); failures++; }
    }
    if (failures) { fprintf(stderr, "%d test(s) failed\n", failures); return 1; }
    printf("All metadata_index tests passed (%zu tests)\n", n);
    return 0;
}
