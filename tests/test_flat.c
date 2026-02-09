#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "gigavector/gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_flat_create_destroy(void) {
    GV_SoAStorage *storage = gv_soa_storage_create(4, 0);
    ASSERT(storage != NULL, "soa storage creation");

    void *index = gv_flat_create(4, NULL, storage);
    ASSERT(index != NULL, "flat index creation");

    gv_flat_destroy(index);
    gv_soa_storage_destroy(storage);
    return 0;
}

static int test_flat_insert_search(void) {
    GV_SoAStorage *storage = gv_soa_storage_create(4, 0);
    ASSERT(storage != NULL, "soa storage creation");

    void *index = gv_flat_create(4, NULL, storage);
    ASSERT(index != NULL, "flat index creation");

    float vectors[5][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f}
    };

    for (int i = 0; i < 5; i++) {
        GV_Vector *v = gv_vector_create_from_data(4, vectors[i]);
        ASSERT(v != NULL, "vector creation");
        ASSERT(gv_flat_insert(index, v) == 0, "flat insert");
    }

    ASSERT(gv_flat_count(index) == 5, "flat count after insert");

    float query[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_Vector *qv = gv_vector_create_from_data(4, query);
    ASSERT(qv != NULL, "query vector creation");

    GV_SearchResult results[3];
    int n = gv_flat_search(index, qv, 3, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(n > 0, "flat search returned results");

    /* Verify results are sorted by distance (ascending) */
    for (int i = 1; i < n; i++) {
        ASSERT(results[i].distance >= results[i - 1].distance,
               "results sorted by distance");
    }

    gv_vector_destroy(qv);
    gv_flat_destroy(index);
    gv_soa_storage_destroy(storage);
    return 0;
}

static int test_flat_exact_results(void) {
    GV_SoAStorage *storage = gv_soa_storage_create(4, 0);
    ASSERT(storage != NULL, "soa storage creation");

    void *index = gv_flat_create(4, NULL, storage);
    ASSERT(index != NULL, "flat index creation");

    float v1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float v3[4] = {9.0f, 10.0f, 11.0f, 12.0f};

    GV_Vector *vec1 = gv_vector_create_from_data(4, v1);
    GV_Vector *vec2 = gv_vector_create_from_data(4, v2);
    GV_Vector *vec3 = gv_vector_create_from_data(4, v3);
    ASSERT(vec1 != NULL && vec2 != NULL && vec3 != NULL, "vector creation");

    ASSERT(gv_flat_insert(index, vec1) == 0, "insert vec1");
    ASSERT(gv_flat_insert(index, vec2) == 0, "insert vec2");
    ASSERT(gv_flat_insert(index, vec3) == 0, "insert vec3");

    /* Search with an exact match to v1 */
    GV_Vector *qv = gv_vector_create_from_data(4, v1);
    ASSERT(qv != NULL, "query vector creation");

    GV_SearchResult results[3];
    int n = gv_flat_search(index, qv, 3, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(n > 0, "flat search returned results");

    /* The closest result should have distance ~0 (exact match) */
    ASSERT(results[0].distance < 1e-5f, "exact match has near-zero distance");

    gv_vector_destroy(qv);
    gv_flat_destroy(index);
    gv_soa_storage_destroy(storage);
    return 0;
}

static int test_flat_range_search(void) {
    GV_SoAStorage *storage = gv_soa_storage_create(4, 0);
    ASSERT(storage != NULL, "soa storage creation");

    void *index = gv_flat_create(4, NULL, storage);
    ASSERT(index != NULL, "flat index creation");

    float v1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v3[4] = {2.0f, 0.0f, 0.0f, 0.0f};
    float v4[4] = {10.0f, 0.0f, 0.0f, 0.0f};

    GV_Vector *vec1 = gv_vector_create_from_data(4, v1);
    GV_Vector *vec2 = gv_vector_create_from_data(4, v2);
    GV_Vector *vec3 = gv_vector_create_from_data(4, v3);
    GV_Vector *vec4 = gv_vector_create_from_data(4, v4);
    ASSERT(vec1 && vec2 && vec3 && vec4, "vector creation");

    ASSERT(gv_flat_insert(index, vec1) == 0, "insert vec1");
    ASSERT(gv_flat_insert(index, vec2) == 0, "insert vec2");
    ASSERT(gv_flat_insert(index, vec3) == 0, "insert vec3");
    ASSERT(gv_flat_insert(index, vec4) == 0, "insert vec4");

    /* Query at origin, radius 2.5 should find v1(dist=0), v2(dist=1), v3(dist=2) */
    float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GV_Vector *qv = gv_vector_create_from_data(4, query);
    ASSERT(qv != NULL, "query vector creation");

    GV_SearchResult results[10];
    int n = gv_flat_range_search(index, qv, 2.5f, results, 10, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(n >= 0, "range search did not fail");

    /* All returned results must be within the radius */
    for (int i = 0; i < n; i++) {
        ASSERT(results[i].distance <= 2.5f, "result within radius");
    }

    gv_vector_destroy(qv);
    gv_flat_destroy(index);
    gv_soa_storage_destroy(storage);
    return 0;
}

static int test_flat_delete(void) {
    GV_SoAStorage *storage = gv_soa_storage_create(4, 0);
    ASSERT(storage != NULL, "soa storage creation");

    void *index = gv_flat_create(4, NULL, storage);
    ASSERT(index != NULL, "flat index creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[4] = {0.0f, 0.0f, 1.0f, 0.0f};

    GV_Vector *vec1 = gv_vector_create_from_data(4, v1);
    GV_Vector *vec2 = gv_vector_create_from_data(4, v2);
    GV_Vector *vec3 = gv_vector_create_from_data(4, v3);
    ASSERT(vec1 && vec2 && vec3, "vector creation");

    ASSERT(gv_flat_insert(index, vec1) == 0, "insert vec1");
    ASSERT(gv_flat_insert(index, vec2) == 0, "insert vec2");
    ASSERT(gv_flat_insert(index, vec3) == 0, "insert vec3");

    size_t count_before = gv_flat_count(index);
    ASSERT(count_before == 3, "count before delete");

    /* Delete the second vector (index 1) */
    ASSERT(gv_flat_delete(index, 1) == 0, "delete vector at index 1");

    /* Search for the deleted vector; it should not appear as nearest */
    GV_Vector *qv = gv_vector_create_from_data(4, v2);
    ASSERT(qv != NULL, "query vector creation");

    GV_SearchResult results[3];
    int n = gv_flat_search(index, qv, 3, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);

    /* The deleted vector should not be returned, so the exact match (distance ~0)
     * should not appear in results */
    int found_deleted = 0;
    for (int i = 0; i < n; i++) {
        if (results[i].distance < 1e-5f) {
            found_deleted = 1;
        }
    }
    ASSERT(!found_deleted, "deleted vector not returned in search");

    gv_vector_destroy(qv);
    gv_flat_destroy(index);
    gv_soa_storage_destroy(storage);
    return 0;
}

static int test_flat_update(void) {
    GV_SoAStorage *storage = gv_soa_storage_create(4, 0);
    ASSERT(storage != NULL, "soa storage creation");

    void *index = gv_flat_create(4, NULL, storage);
    ASSERT(index != NULL, "flat index creation");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};

    GV_Vector *vec1 = gv_vector_create_from_data(4, v1);
    GV_Vector *vec2 = gv_vector_create_from_data(4, v2);
    ASSERT(vec1 && vec2, "vector creation");

    ASSERT(gv_flat_insert(index, vec1) == 0, "insert vec1");
    ASSERT(gv_flat_insert(index, vec2) == 0, "insert vec2");

    /* Search for v1 before update */
    GV_Vector *qv = gv_vector_create_from_data(4, v1);
    ASSERT(qv != NULL, "query vector creation");

    GV_SearchResult results[2];
    int n = gv_flat_search(index, qv, 1, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(n == 1, "search found 1 result");
    float dist_before = results[0].distance;
    ASSERT(dist_before < 1e-5f, "exact match before update");

    /* Update vector 0 to a distant location */
    float new_data[4] = {100.0f, 100.0f, 100.0f, 100.0f};
    ASSERT(gv_flat_update(index, 0, new_data, 4) == 0, "update vector 0");

    /* Search again for original v1 -- the closest should now be v2 (farther) */
    n = gv_flat_search(index, qv, 1, results, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
    ASSERT(n == 1, "search found 1 result after update");
    ASSERT(results[0].distance > 1e-5f, "updated vector no longer matches original query");

    gv_vector_destroy(qv);
    gv_flat_destroy(index);
    gv_soa_storage_destroy(storage);
    return 0;
}

static int test_flat_save_load(void) {
    const char *path = "test_flat_save.db";
    remove(path);

    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open with flat index");

    float v1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v2[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float v3[4] = {9.0f, 10.0f, 11.0f, 12.0f};

    ASSERT(gv_db_add_vector(db, v1, 4) == 0, "add vector 1");
    ASSERT(gv_db_add_vector(db, v2, 4) == 0, "add vector 2");
    ASSERT(gv_db_add_vector(db, v3, 4) == 0, "add vector 3");

    /* Save to file */
    ASSERT(gv_db_save(db, path) == 0, "save database to file");
    gv_db_close(db);

    /* Reopen from file */
    GV_Database *db2 = gv_db_open(path, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db2 != NULL, "reopen database from file");

    /* Search should still work after reload */
    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_SearchResult results[3];
    int n = gv_db_search(db2, query, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n > 0, "search returned results after reload");
    ASSERT(results[0].distance < 1e-5f, "exact match found after reload");

    gv_db_close(db2);
    unlink(path);
    return 0;
}

static int test_flat_metadata_filter(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open with flat index");

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v4[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    ASSERT(gv_db_add_vector_with_metadata(db, v1, 4, "category", "A") == 0, "add vector with category A");
    ASSERT(gv_db_add_vector_with_metadata(db, v2, 4, "category", "B") == 0, "add vector with category B");
    ASSERT(gv_db_add_vector_with_metadata(db, v3, 4, "category", "A") == 0, "add vector with category A");
    ASSERT(gv_db_add_vector_with_metadata(db, v4, 4, "category", "B") == 0, "add vector with category B");

    /* Search with filter for category A */
    float query[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_SearchResult results[4];
    int n = gv_db_search_filtered(db, query, 4, results, GV_DISTANCE_EUCLIDEAN, "category", "A");
    ASSERT(n > 0, "filtered search returned results");

    /* All returned results should belong to category A; since flat is exact,
     * the count should be at most 2 (only vectors with category A) */
    ASSERT(n <= 2, "filtered search returned at most 2 category-A results");

    gv_db_close(db);
    return 0;
}

int main(void) {
    int rc = 0;
    printf("Running Flat index tests...\n");
    rc |= test_flat_create_destroy();
    rc |= test_flat_insert_search();
    rc |= test_flat_exact_results();
    rc |= test_flat_range_search();
    rc |= test_flat_delete();
    rc |= test_flat_update();
    rc |= test_flat_save_load();
    rc |= test_flat_metadata_filter();
    if (rc == 0) {
        printf("All Flat index tests passed\n");
    }
    return rc;
}
