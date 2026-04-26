#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "api/python_compat.h"
#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_open_close(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "gv_db_open should return non-NULL");
    gv_db_close(db);
    return 0;
}

static int test_null_safety(void) {
    gv_db_close(NULL);
    return 0;
}

static int test_add_and_search(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create db");

    float vec[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    int rc = gv_db_add_vector(db, vec, 4);
    ASSERT(rc == 0, "gv_db_add_vector should succeed");

    GV_SearchResult results[1];
    int found = gv_db_search(db, vec, 1, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(found == 1, "should find 1 result");
    ASSERT(results[0].distance >= 0.0f, "euclidean distance should be non-negative");

    gv_db_close(db);
    return 0;
}

static int test_add_with_metadata(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create db");

    float vec[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    int rc = gv_db_add_vector_with_metadata(db, vec, 4, "key", "value");
    ASSERT(rc == 0, "add with metadata should succeed");

    gv_db_close(db);
    return 0;
}

int main(void) {
    int failed = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"open_close",        test_open_close},
        {"null_safety",       test_null_safety},
        {"add_and_search",    test_add_and_search},
        {"add_with_metadata", test_add_with_metadata},
    };
    size_t n = sizeof(tests) / sizeof(tests[0]);
    for (size_t i = 0; i < n; i++) {
        if (tests[i].fn() != 0) {
            fprintf(stderr, "FAILED: %s\n", tests[i].name);
            failed++;
        } else {
            printf("PASS: %s\n", tests[i].name);
        }
    }
    return failed ? 1 : 0;
}
