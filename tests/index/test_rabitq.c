/**
 * test_rabitq.c — direct tests for the RaBitQ index (src/index/rabitq.c):
 * create/destroy, insert/search round-trip, self-query recall on a small
 * synthetic set, delete, update, and DB-level integration.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector.h"
#include "index/rabitq.h"

#define ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return -1; } } while (0)

/* Deterministic pseudo-random float in [0, 1). */
static float rf(unsigned int *s) {
    *s = *s * 1664525u + 1013904223u;
    return (float)(*s >> 8) / (float)(1u << 24);
}

static int test_rabitq_create_destroy(void) {
    const size_t dim = 16;
    GV_RaBitQConfig cfg = { .seed = 42, .rerank_factor = 0 };
    GV_SoAStorage *storage = soa_storage_create(dim, 0);
    ASSERT(storage != NULL);

    void *index = rabitq_create(dim, &cfg, storage);
    ASSERT(index != NULL);
    ASSERT(rabitq_count(index) == 0);

    rabitq_destroy(index);
    soa_storage_destroy(storage);
    return 0;
}

static int test_rabitq_insert_search_roundtrip(void) {
    const size_t dim = 16;
    const int n = 64;
    GV_RaBitQConfig cfg = { .seed = 42, .rerank_factor = 0 };
    GV_SoAStorage *storage = soa_storage_create(dim, 0);
    ASSERT(storage != NULL);
    void *index = rabitq_create(dim, &cfg, storage);
    ASSERT(index != NULL);

    unsigned int s = 7u;
    float first[16];
    for (int i = 0; i < n; i++) {
        float data[16];
        for (size_t j = 0; j < dim; j++) data[j] = rf(&s);
        if (i == 0) memcpy(first, data, sizeof(first));
        GV_Vector *vec = vector_create_from_data(dim, data);
        ASSERT(vec != NULL);
        ASSERT(rabitq_insert(index, vec) == 0);
        /* rabitq_insert takes ownership of vec */
    }
    ASSERT(rabitq_count(index) == (size_t)n);

    /* Query with an exact copy of the first inserted vector: it should be
     * findable in the top results (encode/decode/rerank round-trip). */
    GV_Vector *query = vector_create_from_data(dim, first);
    ASSERT(query != NULL);
    GV_SearchResult results[10];
    memset(results, 0, sizeof(results));
    int count = rabitq_search(index, query, 10, results, GV_DISTANCE_COSINE, NULL, NULL);
    ASSERT(count > 0);

    int found = 0;
    for (int i = 0; i < count; i++) {
        if (results[i].id == 0) found = 1;
    }
    ASSERT(found); /* the exact self-match must appear in the top-k */

    for (int i = 0; i < count; i++) vector_destroy((GV_Vector *)results[i].vector);
    vector_destroy(query);
    rabitq_destroy(index);
    soa_storage_destroy(storage);
    return 0;
}

static int test_rabitq_recall(void) {
    const size_t dim = 32;
    const int n = 200;
    GV_RaBitQConfig cfg = { .seed = 123, .rerank_factor = 8 };
    GV_SoAStorage *storage = soa_storage_create(dim, 0);
    ASSERT(storage != NULL);
    void *index = rabitq_create(dim, &cfg, storage);
    ASSERT(index != NULL);

    /* Keep copies of the inserted vectors so we can self-query each. */
    float (*vecs)[32] = malloc((size_t)n * sizeof(*vecs));
    ASSERT(vecs != NULL);

    unsigned int s = 99u;
    for (int i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) vecs[i][j] = rf(&s);
        GV_Vector *vec = vector_create_from_data(dim, vecs[i]);
        ASSERT(vec != NULL);
        ASSERT(rabitq_insert(index, vec) == 0);
    }

    /* Self-query recall@10: querying an inserted vector should return itself
     * in the top-10 for the large majority of points. */
    int hits = 0, trials = 40;
    for (int t = 0; t < trials; t++) {
        int target = (t * 5) % n;
        GV_Vector *q = vector_create_from_data(dim, vecs[target]);
        ASSERT(q != NULL);
        GV_SearchResult results[10];
        memset(results, 0, sizeof(results));
        int count = rabitq_search(index, q, 10, results, GV_DISTANCE_COSINE, NULL, NULL);
        for (int i = 0; i < count; i++) {
            if (results[i].id == (size_t)target) { hits++; break; }
        }
        for (int i = 0; i < count; i++) vector_destroy((GV_Vector *)results[i].vector);
        vector_destroy(q);
    }

    /* RaBitQ with exact rerank should recall self-matches very reliably. */
    ASSERT(hits >= (trials * 8) / 10); /* >= 80% recall@10 */

    free(vecs);
    rabitq_destroy(index);
    soa_storage_destroy(storage);
    return 0;
}

static int test_rabitq_delete_update(void) {
    const size_t dim = 16;
    GV_RaBitQConfig cfg = { .seed = 42, .rerank_factor = 0 };
    GV_SoAStorage *storage = soa_storage_create(dim, 0);
    ASSERT(storage != NULL);
    void *index = rabitq_create(dim, &cfg, storage);
    ASSERT(index != NULL);

    unsigned int s = 3u;
    for (int i = 0; i < 8; i++) {
        float data[16];
        for (size_t j = 0; j < dim; j++) data[j] = rf(&s);
        GV_Vector *vec = vector_create_from_data(dim, data);
        ASSERT(vec != NULL);
        ASSERT(rabitq_insert(index, vec) == 0);
    }
    ASSERT(rabitq_count(index) == 8);

    ASSERT(rabitq_delete(index, 2) == 0);

    float nd[16];
    for (size_t j = 0; j < dim; j++) nd[j] = (float)j;
    ASSERT(rabitq_update(index, 0, nd, dim) == 0);

    rabitq_destroy(index);
    soa_storage_destroy(storage);
    return 0;
}

static int test_rabitq_db_integration(void) {
    const size_t dim = 16;
    GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_RABITQ);
    if (db == NULL) return 0; /* type may be unavailable in some builds */

    unsigned int s = 55u;
    for (int i = 0; i < 32; i++) {
        float data[16];
        for (size_t j = 0; j < dim; j++) data[j] = rf(&s);
        ASSERT(db_add_vector(db, data, dim) == 0);
    }

    float query[16];
    for (size_t j = 0; j < dim; j++) query[j] = (float)j / 16.0f;
    GV_SearchResult results[5];
    memset(results, 0, sizeof(results));
    int count = db_search(db, query, 5, results, GV_DISTANCE_COSINE);
    ASSERT(count > 0);

    gv_search_results_free(results, (size_t)count);
    db_close(db);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_rabitq_create_destroy();
    rc |= test_rabitq_insert_search_roundtrip();
    rc |= test_rabitq_recall();
    rc |= test_rabitq_delete_update();
    rc |= test_rabitq_db_integration();
    if (rc == 0) printf("All rabitq tests PASSED.\n");
    return rc != 0;
}
