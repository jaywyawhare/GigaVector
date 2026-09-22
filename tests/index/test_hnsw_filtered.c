#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "index/hnsw.h"
#include "index/hnsw_filtered.h"
#include "core/id_bitmap.h"
#include "core/memory.h"
#include "storage/soa_storage.h"
#include "schema/vector.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static float vectors[100][32];

static void generate_vectors(int n, int dim) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) {
            vectors[i][j] = (float)(i * dim + j) * 0.01f;
        }
    }
}

static int test_filtered_search_basic(void) {
    const int N = 100, dim = 32;
    generate_vectors(N, dim);

    GV_HNSWConfig cfg = { .M = 8, .efConstruction = 64, .efSearch = 32 };
    void *index = gv_hnsw_create(dim, &cfg, NULL);
    ASSERT(index != NULL, "create index");

    for (int i = 0; i < N; ++i)
        ASSERT(gv_hnsw_insert_raw(index, vectors[i], dim) == 0, "insert");

    /* Build allowed set: IDs 0..49 */
    GV_IdBitmap *allowed = gv_id_bitmap_create();
    ASSERT(allowed != NULL, "create bitmap");
    for (int i = 0; i < 50; ++i)
        gv_id_bitmap_add(allowed, (uint64_t)i);

    float query[32];
    for (int j = 0; j < dim; ++j) query[j] = vectors[0][j];

    GV_HNSWFilteredConfig fc = { .k = 10, .distance_type = GV_DISTANCE_EUCLIDEAN, .ef_search = 32, .max_candidates = 0 };
    GV_SearchResult results[10];
    int n = gv_hnsw_search_filtered(index, query, dim, 10, allowed, results, &fc);
    ASSERT(n > 0, "filtered search should return results");
    ASSERT(n <= 10, "result count <= k");

    for (int i = 0; i < n; ++i) {
        ASSERT(results[i].id < 50, "all results must be in allowed set");
        if (results[i].vector) vector_destroy((GV_Vector *)results[i].vector);
    }

    gv_id_bitmap_free(allowed);
    gv_hnsw_destroy(index);
    return 0;
}

static int test_filtered_search_vs_unfiltered(void) {
    const int N = 100, dim = 32;
    generate_vectors(N, dim);

    GV_HNSWConfig cfg = { .M = 8, .efConstruction = 64, .efSearch = 32 };
    void *index = gv_hnsw_create(dim, &cfg, NULL);
    ASSERT(index != NULL, "create index");

    for (int i = 0; i < N; ++i)
        ASSERT(gv_hnsw_insert_raw(index, vectors[i], dim) == 0, "insert");

    float query[32];
    for (int j = 0; j < dim; ++j) query[j] = vectors[50][j];

    /* Unfiltered search */
    GV_SearchResult unfiltered[10];
    GV_Vector qv = { .data = query, .dimension = dim, .metadata = NULL };
    int n_all = gv_hnsw_search(index, &qv, 10, unfiltered, GV_DISTANCE_EUCLIDEAN, NULL, NULL);

    /* Filtered search: only IDs 60..99 */
    GV_IdBitmap *allowed = gv_id_bitmap_create();
    for (int i = 60; i < 100; ++i)
        gv_id_bitmap_add(allowed, (uint64_t)i);

    GV_HNSWFilteredConfig fc = { .k = 10, .distance_type = GV_DISTANCE_EUCLIDEAN, .ef_search = 32, .max_candidates = 0 };
    GV_SearchResult filtered[10];
    int n_filt = gv_hnsw_search_filtered(index, query, dim, 10, allowed, filtered, &fc);

    ASSERT(n_filt > 0, "filtered search should return results");
    for (int i = 0; i < n_filt; ++i) {
        ASSERT(filtered[i].id >= 60 && filtered[i].id < 100,
               "filtered results must be in [60,100)");
        if (filtered[i].vector) vector_destroy((GV_Vector *)filtered[i].vector);
    }

    /* The unfiltered results may differ because they can include IDs outside the allowed set */
    for (int i = 0; i < n_all; ++i) {
        if (unfiltered[i].vector) vector_destroy((GV_Vector *)unfiltered[i].vector);
    }

    gv_id_bitmap_free(allowed);
    gv_hnsw_destroy(index);
    return 0;
}

static int test_range_search_filtered(void) {
    const int N = 50, dim = 8;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dim; ++j)
            vectors[i][j] = (float)i + (float)j * 0.1f;
    }

    GV_HNSWConfig cfg = { .M = 8, .efConstruction = 64, .efSearch = 32 };
    void *index = gv_hnsw_create(dim, &cfg, NULL);
    ASSERT(index != NULL, "create index");

    for (int i = 0; i < N; ++i)
        ASSERT(gv_hnsw_insert_raw(index, vectors[i], dim) == 0, "insert");

    GV_IdBitmap *allowed = gv_id_bitmap_create();
    for (int i = 0; i < 25; ++i)
        gv_id_bitmap_add(allowed, (uint64_t)i);

    float query[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    GV_SearchResult results[25];
    int n = gv_hnsw_range_search_filtered(index, query, dim, 5.0f, allowed, results, 25, NULL);
    ASSERT(n >= 0, "range search completed");

    for (int i = 0; i < n; ++i) {
        ASSERT(results[i].id < 25, "range results in allowed set");
        ASSERT(results[i].distance <= 5.0f, "distance within radius");
        if (results[i].vector) vector_destroy((GV_Vector *)results[i].vector);
    }

    gv_id_bitmap_free(allowed);
    gv_hnsw_destroy(index);
    return 0;
}

static int test_no_allowed_nodes(void) {
    const int N = 10, dim = 8;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dim; ++j) vectors[i][j] = (float)(i * dim + j);
    }

    GV_HNSWConfig cfg = { .M = 8, .efConstruction = 64, .efSearch = 16 };
    void *index = gv_hnsw_create(dim, &cfg, NULL);
    ASSERT(index != NULL, "create index");
    for (int i = 0; i < N; ++i)
        ASSERT(gv_hnsw_insert_raw(index, vectors[i], dim) == 0, "insert");

    /* Empty bitmap: no nodes allowed */
    GV_IdBitmap *allowed = gv_id_bitmap_create();
    float query[8] = {0};
    GV_HNSWFilteredConfig fc = { .k = 5, .distance_type = GV_DISTANCE_EUCLIDEAN, .ef_search = 16, .max_candidates = 0 };
    GV_SearchResult results[5];
    int n = gv_hnsw_search_filtered(index, query, dim, 5, allowed, results, &fc);
    ASSERT(n == 0, "empty bitmap should return 0 results");

    gv_id_bitmap_free(allowed);
    gv_hnsw_destroy(index);
    return 0;
}

static int test_null_bitmap_no_filter(void) {
    const int N = 20, dim = 8;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dim; ++j) vectors[i][j] = (float)(i * dim + j);
    }

    GV_HNSWConfig cfg = { .M = 8, .efConstruction = 64, .efSearch = 32 };
    void *index = gv_hnsw_create(dim, &cfg, NULL);
    ASSERT(index != NULL, "create index");
    for (int i = 0; i < N; ++i)
        ASSERT(gv_hnsw_insert_raw(index, vectors[i], dim) == 0, "insert");

    float query[8] = {0};
    GV_HNSWFilteredConfig fc = { .k = 5, .distance_type = GV_DISTANCE_EUCLIDEAN, .ef_search = 32, .max_candidates = 0 };
    GV_SearchResult results[5];
    /* NULL bitmap = no filter, equivalent to unfiltered search */
    int n = gv_hnsw_search_filtered(index, query, dim, 5, NULL, results, &fc);
    ASSERT(n == 5, "null bitmap should return k results");

    for (int i = 0; i < n; ++i) {
        if (results[i].vector) vector_destroy((GV_Vector *)results[i].vector);
    }
    gv_hnsw_destroy(index);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_filtered_search_basic();
    rc |= test_filtered_search_vs_unfiltered();
    rc |= test_range_search_filtered();
    rc |= test_no_allowed_nodes();
    rc |= test_null_bitmap_no_filter();
    if (rc == 0) printf("All HNSW filtered tests passed.\n");
    return rc;
}
