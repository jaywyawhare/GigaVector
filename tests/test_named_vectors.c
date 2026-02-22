#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_named_vectors.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_named_vectors_create_destroy(void) {
    GV_NamedVectorStore *store = gv_named_vectors_create();
    ASSERT(store != NULL, "named vector store creation");

    gv_named_vectors_destroy(store);

    /* Destroy NULL should be safe */
    gv_named_vectors_destroy(NULL);
    return 0;
}

static int test_named_vectors_add_field(void) {
    GV_NamedVectorStore *store = gv_named_vectors_create();
    ASSERT(store != NULL, "store creation");

    GV_VectorFieldConfig cfg1 = { .name = "title", .dimension = 4, .distance_type = 0 };
    GV_VectorFieldConfig cfg2 = { .name = "content", .dimension = 4, .distance_type = 0 };

    ASSERT(gv_named_vectors_add_field(store, &cfg1) == 0, "add field 'title'");
    ASSERT(gv_named_vectors_add_field(store, &cfg2) == 0, "add field 'content'");
    ASSERT(gv_named_vectors_field_count(store) == 2, "field count is 2");

    /* Retrieve field config */
    GV_VectorFieldConfig out;
    ASSERT(gv_named_vectors_get_field(store, "title", &out) == 0, "get field 'title'");
    ASSERT(out.dimension == 4, "title dimension is 4");

    gv_named_vectors_destroy(store);
    return 0;
}

static int test_named_vectors_remove_field(void) {
    GV_NamedVectorStore *store = gv_named_vectors_create();
    ASSERT(store != NULL, "store creation");

    GV_VectorFieldConfig cfg = { .name = "temp", .dimension = 4, .distance_type = 0 };
    gv_named_vectors_add_field(store, &cfg);
    ASSERT(gv_named_vectors_field_count(store) == 1, "field count is 1");

    ASSERT(gv_named_vectors_remove_field(store, "temp") == 0, "remove field 'temp'");
    ASSERT(gv_named_vectors_field_count(store) == 0, "field count is 0 after removal");

    /* Removing nonexistent field */
    ASSERT(gv_named_vectors_remove_field(store, "nonexistent") == -1,
           "remove nonexistent field returns -1");

    gv_named_vectors_destroy(store);
    return 0;
}

static int test_named_vectors_insert_and_get(void) {
    GV_NamedVectorStore *store = gv_named_vectors_create();
    ASSERT(store != NULL, "store creation");

    GV_VectorFieldConfig cfg_title = { .name = "title", .dimension = 4, .distance_type = 0 };
    GV_VectorFieldConfig cfg_body = { .name = "body", .dimension = 4, .distance_type = 0 };
    gv_named_vectors_add_field(store, &cfg_title);
    gv_named_vectors_add_field(store, &cfg_body);

    float title_data[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float body_data[4] = {0.0f, 1.0f, 0.0f, 0.0f};

    GV_NamedVector vectors[2] = {
        { .field_name = "title", .data = title_data, .dimension = 4 },
        { .field_name = "body",  .data = body_data,  .dimension = 4 },
    };

    ASSERT(gv_named_vectors_insert(store, 0, vectors, 2) == 0, "insert point 0");
    ASSERT(gv_named_vectors_count(store) == 1, "count is 1 after insert");

    /* Retrieve the title vector for point 0 */
    const float *retrieved = gv_named_vectors_get(store, 0, "title");
    ASSERT(retrieved != NULL, "get title vector for point 0");
    ASSERT(retrieved[0] == 1.0f, "title[0] == 1.0");

    /* Retrieve the body vector for point 0 */
    retrieved = gv_named_vectors_get(store, 0, "body");
    ASSERT(retrieved != NULL, "get body vector for point 0");
    ASSERT(retrieved[1] == 1.0f, "body[1] == 1.0");

    gv_named_vectors_destroy(store);
    return 0;
}

static int test_named_vectors_update(void) {
    GV_NamedVectorStore *store = gv_named_vectors_create();
    ASSERT(store != NULL, "store creation");

    GV_VectorFieldConfig cfg = { .name = "embed", .dimension = 4, .distance_type = 0 };
    gv_named_vectors_add_field(store, &cfg);

    float data_v1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_NamedVector vec = { .field_name = "embed", .data = data_v1, .dimension = 4 };
    gv_named_vectors_insert(store, 0, &vec, 1);

    /* Update the vector */
    float data_v2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    GV_NamedVector updated = { .field_name = "embed", .data = data_v2, .dimension = 4 };
    ASSERT(gv_named_vectors_update(store, 0, &updated, 1) == 0, "update point 0");

    const float *retrieved = gv_named_vectors_get(store, 0, "embed");
    ASSERT(retrieved != NULL, "get after update");
    ASSERT(retrieved[0] == 5.0f, "updated embed[0] == 5.0");
    ASSERT(retrieved[3] == 8.0f, "updated embed[3] == 8.0");

    gv_named_vectors_destroy(store);
    return 0;
}

static int test_named_vectors_delete(void) {
    GV_NamedVectorStore *store = gv_named_vectors_create();
    ASSERT(store != NULL, "store creation");

    GV_VectorFieldConfig cfg = { .name = "data", .dimension = 4, .distance_type = 0 };
    gv_named_vectors_add_field(store, &cfg);

    float d1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float d2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    GV_NamedVector v1 = { .field_name = "data", .data = d1, .dimension = 4 };
    GV_NamedVector v2 = { .field_name = "data", .data = d2, .dimension = 4 };

    gv_named_vectors_insert(store, 0, &v1, 1);
    gv_named_vectors_insert(store, 1, &v2, 1);
    ASSERT(gv_named_vectors_count(store) == 2, "count is 2");

    ASSERT(gv_named_vectors_delete(store, 0) == 0, "delete point 0");

    /* Deleted point should not be retrievable */
    const float *gone = gv_named_vectors_get(store, 0, "data");
    ASSERT(gone == NULL, "deleted point returns NULL");

    gv_named_vectors_destroy(store);
    return 0;
}

static int test_named_vectors_search(void) {
    GV_NamedVectorStore *store = gv_named_vectors_create();
    ASSERT(store != NULL, "store creation");

    GV_VectorFieldConfig cfg = { .name = "embed", .dimension = 4, .distance_type = 0 };
    gv_named_vectors_add_field(store, &cfg);

    float d0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float d1[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float d2[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    GV_NamedVector v0 = { .field_name = "embed", .data = d0, .dimension = 4 };
    GV_NamedVector v1 = { .field_name = "embed", .data = d1, .dimension = 4 };
    GV_NamedVector v2 = { .field_name = "embed", .data = d2, .dimension = 4 };
    gv_named_vectors_insert(store, 0, &v0, 1);
    gv_named_vectors_insert(store, 1, &v1, 1);
    gv_named_vectors_insert(store, 2, &v2, 1);

    float query[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_NamedSearchResult results[3];
    int n = gv_named_vectors_search(store, "embed", query, 2, results);
    ASSERT(n >= 1, "search returned at least 1 result");

    /* The nearest result should be point 0 (exact match) */
    ASSERT(results[0].distance < 1e-5f, "nearest result has near-zero distance");

    gv_named_vectors_destroy(store);
    return 0;
}

static int test_named_vectors_save_load(void) {
    const char *path = "/tmp/test_named_vectors.bin";
    GV_NamedVectorStore *store = gv_named_vectors_create();
    ASSERT(store != NULL, "store creation");

    GV_VectorFieldConfig cfg = { .name = "vec", .dimension = 4, .distance_type = 0 };
    gv_named_vectors_add_field(store, &cfg);

    float d[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_NamedVector v = { .field_name = "vec", .data = d, .dimension = 4 };
    gv_named_vectors_insert(store, 0, &v, 1);

    ASSERT(gv_named_vectors_save(store, path) == 0, "save named vectors");
    gv_named_vectors_destroy(store);

    GV_NamedVectorStore *loaded = gv_named_vectors_load(path);
    ASSERT(loaded != NULL, "load named vectors");
    ASSERT(gv_named_vectors_field_count(loaded) == 1, "loaded field count is 1");
    ASSERT(gv_named_vectors_count(loaded) == 1, "loaded point count is 1");

    const float *r = gv_named_vectors_get(loaded, 0, "vec");
    ASSERT(r != NULL, "get vector from loaded store");
    ASSERT(r[0] == 1.0f && r[3] == 4.0f, "loaded vector data correct");

    gv_named_vectors_destroy(loaded);
    remove(path);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing named_vectors create/destroy...", test_named_vectors_create_destroy},
        {"Testing named_vectors add field...", test_named_vectors_add_field},
        {"Testing named_vectors remove field...", test_named_vectors_remove_field},
        {"Testing named_vectors insert and get...", test_named_vectors_insert_and_get},
        {"Testing named_vectors update...", test_named_vectors_update},
        {"Testing named_vectors delete...", test_named_vectors_delete},
        {"Testing named_vectors search...", test_named_vectors_search},
        {"Testing named_vectors save/load...", test_named_vectors_save_load},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
