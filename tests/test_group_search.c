#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_group_search.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"

#define DIM 4

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Helper: create a flat DB with vectors assigned to groups via metadata */
static GV_Database *make_db(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    if (!db) return NULL;

    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.9f, 0.1f, 0.0f, 0.0f};
    float v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[] = {0.0f, 0.9f, 0.1f, 0.0f};
    float v4[] = {0.0f, 0.0f, 1.0f, 0.0f};
    float v5[] = {0.0f, 0.0f, 0.9f, 0.1f};

    /* Assign group metadata via "category" key */
    gv_db_add_vector_with_metadata(db, v0, DIM, "category", "electronics");
    gv_db_add_vector_with_metadata(db, v1, DIM, "category", "electronics");
    gv_db_add_vector_with_metadata(db, v2, DIM, "category", "books");
    gv_db_add_vector_with_metadata(db, v3, DIM, "category", "books");
    gv_db_add_vector_with_metadata(db, v4, DIM, "category", "clothing");
    gv_db_add_vector_with_metadata(db, v5, DIM, "category", "clothing");

    return db;
}

/* ---------- test_config_init ---------- */
static int test_config_init(void) {
    GV_GroupSearchConfig cfg;
    gv_group_search_config_init(&cfg);
    ASSERT(cfg.group_limit == 10, "default group_limit should be 10");
    ASSERT(cfg.hits_per_group == 3, "default hits_per_group should be 3");
    ASSERT(cfg.group_by == NULL, "default group_by should be NULL");
    return 0;
}

/* ---------- test_group_search_basic ---------- */
static int test_group_search_basic(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    float query[] = {0.5f, 0.5f, 0.0f, 0.0f};

    GV_GroupSearchConfig cfg;
    gv_group_search_config_init(&cfg);
    cfg.group_by = "category";
    cfg.group_limit = 5;
    cfg.hits_per_group = 2;
    cfg.distance_type = GV_DISTANCE_EUCLIDEAN;

    GV_GroupedResult result;
    memset(&result, 0, sizeof(result));
    int rc = gv_group_search(db, query, DIM, &cfg, &result);
    ASSERT(rc == 0, "group_search should succeed");

    /* We have 3 groups: electronics, books, clothing */
    ASSERT(result.group_count > 0, "should find at least 1 group");
    ASSERT(result.group_count <= 3, "should have at most 3 groups");

    /* Check that groups have valid data */
    for (size_t i = 0; i < result.group_count; i++) {
        ASSERT(result.groups[i].group_value != NULL, "group_value should not be NULL");
        ASSERT(result.groups[i].hit_count > 0, "each group should have at least 1 hit");
        ASSERT(result.groups[i].hit_count <= 2, "each group should have at most hits_per_group hits");
    }

    gv_group_search_free_result(&result);
    gv_db_close(db);
    return 0;
}

/* ---------- test_group_search_free_result ---------- */
static int test_group_search_free_result(void) {
    /* Freeing a zeroed result should not crash */
    GV_GroupedResult result;
    memset(&result, 0, sizeof(result));
    gv_group_search_free_result(&result);
    return 0;
}

/* ---------- test_group_search_single_group ---------- */
static int test_group_search_single_group(void) {
    GV_Database *db = gv_db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db open should succeed");

    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.9f, 0.1f, 0.0f, 0.0f};
    float v2[] = {0.8f, 0.2f, 0.0f, 0.0f};

    gv_db_add_vector_with_metadata(db, v0, DIM, "type", "same");
    gv_db_add_vector_with_metadata(db, v1, DIM, "type", "same");
    gv_db_add_vector_with_metadata(db, v2, DIM, "type", "same");

    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};

    GV_GroupSearchConfig cfg;
    gv_group_search_config_init(&cfg);
    cfg.group_by = "type";
    cfg.group_limit = 10;
    cfg.hits_per_group = 2;
    cfg.distance_type = GV_DISTANCE_EUCLIDEAN;

    GV_GroupedResult result;
    memset(&result, 0, sizeof(result));
    int rc = gv_group_search(db, query, DIM, &cfg, &result);
    ASSERT(rc == 0, "group_search should succeed");
    ASSERT(result.group_count == 1, "should have exactly 1 group");

    if (result.group_count == 1) {
        ASSERT(strcmp(result.groups[0].group_value, "same") == 0,
               "group value should be 'same'");
        ASSERT(result.groups[0].hit_count <= 2,
               "hits should respect hits_per_group");
    }

    gv_group_search_free_result(&result);
    gv_db_close(db);
    return 0;
}

/* ---------- test_group_search_limit ---------- */
static int test_group_search_limit(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    float query[] = {0.5f, 0.5f, 0.5f, 0.0f};

    GV_GroupSearchConfig cfg;
    gv_group_search_config_init(&cfg);
    cfg.group_by = "category";
    cfg.group_limit = 2; /* Only want top 2 groups */
    cfg.hits_per_group = 1;
    cfg.distance_type = GV_DISTANCE_EUCLIDEAN;

    GV_GroupedResult result;
    memset(&result, 0, sizeof(result));
    int rc = gv_group_search(db, query, DIM, &cfg, &result);
    ASSERT(rc == 0, "group_search should succeed");
    ASSERT(result.group_count <= 2, "should return at most group_limit groups");

    gv_group_search_free_result(&result);
    gv_db_close(db);
    return 0;
}

/* ---------- test_group_search_hits_sorted ---------- */
static int test_group_search_hits_sorted(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};

    GV_GroupSearchConfig cfg;
    gv_group_search_config_init(&cfg);
    cfg.group_by = "category";
    cfg.group_limit = 3;
    cfg.hits_per_group = 2;
    cfg.distance_type = GV_DISTANCE_EUCLIDEAN;

    GV_GroupedResult result;
    memset(&result, 0, sizeof(result));
    int rc = gv_group_search(db, query, DIM, &cfg, &result);
    ASSERT(rc == 0, "group_search should succeed");

    /* Within each group, hits should be sorted by distance ascending */
    for (size_t g = 0; g < result.group_count; g++) {
        for (size_t h = 1; h < result.groups[g].hit_count; h++) {
            ASSERT(result.groups[g].hits[h].distance >= result.groups[g].hits[h - 1].distance,
                   "hits within a group should be sorted by distance");
        }
    }

    gv_group_search_free_result(&result);
    gv_db_close(db);
    return 0;
}

/* test runner */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing group search config init...",       test_config_init},
        {"Testing group search basic...",             test_group_search_basic},
        {"Testing group search free result...",       test_group_search_free_result},
        {"Testing group search single group...",      test_group_search_single_group},
        {"Testing group search limit...",             test_group_search_limit},
        {"Testing group search hits sorted...",       test_group_search_hits_sorted},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
