#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "index/exact_search.h"
#include "search/distance.h"
#include "schema/vector.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)
#define DIM 4

static int test_exact_knn_null_vectors(void) {
    GV_Vector query = { .dimension = DIM, .data = (float[]){1,0,0,0}, .metadata = NULL };
    GV_SearchResult results[3] = {0};
    int rc = exact_knn_search_vectors(NULL, 10, &query, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(rc == -1, "null vectors with positive count fails");
    return 0;
}

static int test_exact_knn_null_query(void) {
    GV_Vector *vectors[2] = { NULL };
    GV_SearchResult results[3] = {0};
    int rc = exact_knn_search_vectors(vectors, 2, NULL, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(rc == -1, "null query fails");
    return 0;
}

static int test_exact_knn_zero_k(void) {
    GV_Vector query = { .dimension = DIM, .data = (float[]){1,0,0,0}, .metadata = NULL };
    GV_SearchResult results[3] = {0};
    int rc = exact_knn_search_vectors(NULL, 0, &query, 0, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(rc == -1, "zero k fails");
    return 0;
}

static int test_exact_knn_empty_database(void) {
    GV_Vector query = { .dimension = DIM, .data = (float[]){1,0,0,0}, .metadata = NULL };
    GV_Vector *vectors[1] = { &query };
    GV_SearchResult results[3] = {0};
    int rc = exact_knn_search_vectors(vectors, 0, &query, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(rc == 0, "empty database returns 0");
    return 0;
}

static int test_exact_knn_basic(void) {
    GV_Vector v1 = { .dimension = DIM, .data = (float[]){0,0,0,0}, .metadata = NULL };
    GV_Vector v2 = { .dimension = DIM, .data = (float[]){1,0,0,0}, .metadata = NULL };
    GV_Vector v3 = { .dimension = DIM, .data = (float[]){0,1,0,0}, .metadata = NULL };
    GV_Vector *vectors[3] = { &v1, &v2, &v3 };
    GV_Vector query = { .dimension = DIM, .data = (float[]){0.9f,0,0,0}, .metadata = NULL };
    GV_SearchResult results[3] = {0};
    int rc = exact_knn_search_vectors(vectors, 3, &query, 2, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(rc > 0, "search returns results");
    ASSERT(results[0].distance <= results[1].distance, "results are sorted");
    ASSERT(results[0].id < 3, "result id is valid");
    return 0;
}

static int test_exact_knn_cosine(void) {
    GV_Vector v1 = { .dimension = DIM, .data = (float[]){1,0,0,0}, .metadata = NULL };
    GV_Vector v2 = { .dimension = DIM, .data = (float[]){0,1,0,0}, .metadata = NULL };
    GV_Vector *vectors[2] = { &v1, &v2 };
    GV_Vector query = { .dimension = DIM, .data = (float[]){1,0,0,0}, .metadata = NULL };
    GV_SearchResult results[2] = {0};
    int rc = exact_knn_search_vectors(vectors, 2, &query, 2, results, GV_DISTANCE_COSINE);
    ASSERT(rc > 0, "cosine search returns results");
    ASSERT(results[0].id < 2, "result id is valid");
    return 0;
}

static int test_exact_knn_k_limited(void) {
    GV_Vector v1 = { .dimension = DIM, .data = (float[]){1,0,0,0}, .metadata = NULL };
    GV_Vector *vectors[1] = { &v1 };
    GV_Vector query = { .dimension = DIM, .data = (float[]){1,0,0,0}, .metadata = NULL };
    GV_SearchResult results[5] = {0};
    int rc = exact_knn_search_vectors(vectors, 1, &query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(rc == 1, "returns at most count vectors");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"exact_knn_null_vectors", test_exact_knn_null_vectors},
        {"exact_knn_null_query", test_exact_knn_null_query},
        {"exact_knn_zero_k", test_exact_knn_zero_k},
        {"exact_knn_empty_database", test_exact_knn_empty_database},
        {"exact_knn_basic", test_exact_knn_basic},
        {"exact_knn_cosine", test_exact_knn_cosine},
        {"exact_knn_k_limited", test_exact_knn_k_limited},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) passed++;
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}