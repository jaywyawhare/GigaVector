#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "multimodal/multivec.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 4

static int test_create_destroy(void) {
    void *idx = multivec_create(DIM, NULL);
    ASSERT(idx != NULL, "multivec index creation with defaults");
    multivec_destroy(idx);
    multivec_destroy(NULL);
    return 0;
}

static int test_create_with_config(void) {
    GV_MultiVecConfig cfg;
    cfg.max_chunks_per_doc = 32;
    cfg.aggregation = GV_DOC_AGG_AVG_SIM;

    void *idx = multivec_create(DIM, &cfg);
    ASSERT(idx != NULL, "multivec index creation with custom config");
    multivec_destroy(idx);
    return 0;
}

static int test_add_document(void) {
    void *idx = multivec_create(DIM, NULL);
    ASSERT(idx != NULL, "index creation");

    float chunks[3 * DIM] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };

    int rc = multivec_add_document(idx, 100, chunks, 3, DIM);
    ASSERT(rc == 0, "add document with 3 chunks");
    ASSERT(multivec_count_documents(idx) == 1, "document count should be 1");
    ASSERT(multivec_count_chunks(idx) == 3, "chunk count should be 3");

    multivec_destroy(idx);
    return 0;
}

static int test_add_multiple_documents(void) {
    void *idx = multivec_create(DIM, NULL);
    ASSERT(idx != NULL, "index creation");

    float doc1[2 * DIM] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    float doc2[1 * DIM] = {
        0.5f, 0.5f, 0.0f, 0.0f,
    };

    ASSERT(multivec_add_document(idx, 1, doc1, 2, DIM) == 0, "add doc1");
    ASSERT(multivec_add_document(idx, 2, doc2, 1, DIM) == 0, "add doc2");
    ASSERT(multivec_count_documents(idx) == 2, "document count should be 2");
    ASSERT(multivec_count_chunks(idx) == 3, "total chunk count should be 3");

    multivec_destroy(idx);
    return 0;
}

static int test_delete_document(void) {
    void *idx = multivec_create(DIM, NULL);
    ASSERT(idx != NULL, "index creation");

    float chunks[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    multivec_add_document(idx, 10, chunks, 1, DIM);
    multivec_add_document(idx, 20, chunks, 1, DIM);
    ASSERT(multivec_count_documents(idx) == 2, "2 documents before delete");

    int rc = multivec_delete_document(idx, 10);
    ASSERT(rc == 0, "delete document 10");
    ASSERT(multivec_count_documents(idx) == 1, "1 document after delete");

    rc = multivec_delete_document(idx, 999);
    ASSERT(rc == -1, "delete non-existent document should fail");

    multivec_destroy(idx);
    return 0;
}

static int test_search(void) {
    void *idx = multivec_create(DIM, NULL);
    ASSERT(idx != NULL, "index creation");

    float doc1[2 * DIM] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.9f, 0.1f, 0.0f, 0.0f,
    };
    float doc2[1 * DIM] = {0.0f, 1.0f, 0.0f, 0.0f};
    float doc3[1 * DIM] = {0.0f, 0.0f, 1.0f, 0.0f};

    multivec_add_document(idx, 1, doc1, 2, DIM);
    multivec_add_document(idx, 2, doc2, 1, DIM);
    multivec_add_document(idx, 3, doc3, 1, DIM);

    float query[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_DocSearchResult results[3];
    memset(results, 0, sizeof(results));

    int found = multivec_search(idx, query, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found > 0, "search should return at least one result");
    ASSERT(results[0].doc_id == 1, "closest document should be doc_id=1");

    multivec_destroy(idx);
    return 0;
}

static int test_save_load(void) {
    const char *tmppath = "/tmp/test_multivec.bin";
    void *idx = multivec_create(DIM, NULL);
    ASSERT(idx != NULL, "index creation");

    float chunks[DIM] = {1.0f, 2.0f, 3.0f, 4.0f};
    multivec_add_document(idx, 42, chunks, 1, DIM);

    FILE *fout = fopen(tmppath, "wb");
    ASSERT(fout != NULL, "open file for writing");
    int rc = multivec_save(idx, fout);
    fclose(fout);
    ASSERT(rc == 0, "save multivec index");

    multivec_destroy(idx);

    void *loaded = NULL;
    FILE *fin = fopen(tmppath, "rb");
    ASSERT(fin != NULL, "open file for reading");
    rc = multivec_load(&loaded, fin, DIM);
    fclose(fin);
    ASSERT(rc == 0, "load multivec index");
    ASSERT(loaded != NULL, "loaded index should not be NULL");
    ASSERT(multivec_count_documents(loaded) == 1, "loaded index should have 1 document");

    multivec_destroy(loaded);
    unlink(tmppath);
    return 0;
}

static int test_aggregation_modes(void) {
    GV_MultiVecConfig cfg;
    cfg.max_chunks_per_doc = 256;

    GV_DocAggregation modes[] = {GV_DOC_AGG_MAX_SIM, GV_DOC_AGG_AVG_SIM, GV_DOC_AGG_SUM_SIM};
    for (int i = 0; i < 3; i++) {
        cfg.aggregation = modes[i];
        void *idx = multivec_create(DIM, &cfg);
        ASSERT(idx != NULL, "create with aggregation mode");
        multivec_destroy(idx);
    }
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing create/destroy...",          test_create_destroy},
        {"Testing create with config...",      test_create_with_config},
        {"Testing add document...",            test_add_document},
        {"Testing add multiple documents...",  test_add_multiple_documents},
        {"Testing delete document...",         test_delete_document},
        {"Testing search...",                  test_search},
        {"Testing save/load...",               test_save_load},
        {"Testing aggregation modes...",       test_aggregation_modes},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
