#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "gigavector/gv_json_index.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define TMP_INDEX_PATH "/tmp/gv_test_json_index.bin"

/* ── Test: create and destroy ──────────────────────────────────────────── */
static int test_create_destroy(void) {
    GV_JSONPathIndex *idx = gv_json_index_create();
    ASSERT(idx != NULL, "json index creation");
    gv_json_index_destroy(idx);
    /* NULL safety */
    gv_json_index_destroy(NULL);
    return 0;
}

/* ── Test: add and remove path ─────────────────────────────────────────── */
static int test_add_remove_path(void) {
    GV_JSONPathIndex *idx = gv_json_index_create();
    ASSERT(idx != NULL, "index creation");

    GV_JSONPathConfig pcfg;
    pcfg.path = "name";
    pcfg.type = GV_JP_STRING;
    int rc = gv_json_index_add_path(idx, &pcfg);
    ASSERT(rc == 0, "add path 'name'");

    pcfg.path = "age";
    pcfg.type = GV_JP_INT;
    rc = gv_json_index_add_path(idx, &pcfg);
    ASSERT(rc == 0, "add path 'age'");

    rc = gv_json_index_remove_path(idx, "name");
    ASSERT(rc == 0, "remove path 'name'");

    /* removing non-existent path should fail */
    rc = gv_json_index_remove_path(idx, "nonexistent");
    ASSERT(rc == -1, "remove non-existent path should return -1");

    gv_json_index_destroy(idx);
    return 0;
}

/* ── Test: insert and lookup string ────────────────────────────────────── */
static int test_insert_lookup_string(void) {
    GV_JSONPathIndex *idx = gv_json_index_create();
    ASSERT(idx != NULL, "index creation");

    GV_JSONPathConfig pcfg;
    pcfg.path = "city";
    pcfg.type = GV_JP_STRING;
    gv_json_index_add_path(idx, &pcfg);

    const char *json1 = "{\"city\": \"Seattle\"}";
    const char *json2 = "{\"city\": \"Portland\"}";
    const char *json3 = "{\"city\": \"Seattle\"}";

    int rc = gv_json_index_insert(idx, 0, json1);
    ASSERT(rc == 0, "insert json1 at vector_index 0");
    rc = gv_json_index_insert(idx, 1, json2);
    ASSERT(rc == 0, "insert json2 at vector_index 1");
    rc = gv_json_index_insert(idx, 2, json3);
    ASSERT(rc == 0, "insert json3 at vector_index 2");

    size_t out[10];
    int found = gv_json_index_lookup_string(idx, "city", "Seattle", out, 10);
    ASSERT(found == 2, "lookup 'Seattle' should return 2 results");

    found = gv_json_index_lookup_string(idx, "city", "Portland", out, 10);
    ASSERT(found == 1, "lookup 'Portland' should return 1 result");
    ASSERT(out[0] == 1, "Portland result should be vector_index 1");

    gv_json_index_destroy(idx);
    return 0;
}

/* ── Test: insert and lookup int range ─────────────────────────────────── */
static int test_insert_lookup_int_range(void) {
    GV_JSONPathIndex *idx = gv_json_index_create();
    ASSERT(idx != NULL, "index creation");

    GV_JSONPathConfig pcfg;
    pcfg.path = "score";
    pcfg.type = GV_JP_INT;
    gv_json_index_add_path(idx, &pcfg);

    gv_json_index_insert(idx, 0, "{\"score\": 10}");
    gv_json_index_insert(idx, 1, "{\"score\": 50}");
    gv_json_index_insert(idx, 2, "{\"score\": 90}");
    gv_json_index_insert(idx, 3, "{\"score\": 30}");

    size_t out[10];
    int found = gv_json_index_lookup_int_range(idx, "score", 20, 60, out, 10);
    ASSERT(found == 2, "int range [20,60] should match 2 entries (30 and 50)");

    found = gv_json_index_lookup_int_range(idx, "score", 0, 100, out, 10);
    ASSERT(found == 4, "int range [0,100] should match all 4 entries");

    gv_json_index_destroy(idx);
    return 0;
}

/* ── Test: remove entries by vector index ──────────────────────────────── */
static int test_remove_entries(void) {
    GV_JSONPathIndex *idx = gv_json_index_create();
    ASSERT(idx != NULL, "index creation");

    GV_JSONPathConfig pcfg;
    pcfg.path = "tag";
    pcfg.type = GV_JP_STRING;
    gv_json_index_add_path(idx, &pcfg);

    gv_json_index_insert(idx, 0, "{\"tag\": \"alpha\"}");
    gv_json_index_insert(idx, 1, "{\"tag\": \"beta\"}");

    ASSERT(gv_json_index_count(idx, "tag") == 2, "count should be 2 before remove");

    int rc = gv_json_index_remove(idx, 0);
    ASSERT(rc == 0, "remove vector_index 0");
    ASSERT(gv_json_index_count(idx, "tag") == 1, "count should be 1 after remove");

    gv_json_index_destroy(idx);
    return 0;
}

/* ── Test: count ───────────────────────────────────────────────────────── */
static int test_count(void) {
    GV_JSONPathIndex *idx = gv_json_index_create();
    ASSERT(idx != NULL, "index creation");

    /* Count on unregistered path should be 0 */
    size_t c = gv_json_index_count(idx, "no_such_path");
    ASSERT(c == 0, "count on unregistered path should be 0");

    GV_JSONPathConfig pcfg;
    pcfg.path = "x";
    pcfg.type = GV_JP_FLOAT;
    gv_json_index_add_path(idx, &pcfg);

    ASSERT(gv_json_index_count(idx, "x") == 0, "count on empty path should be 0");

    gv_json_index_insert(idx, 0, "{\"x\": 1.5}");
    ASSERT(gv_json_index_count(idx, "x") == 1, "count after one insert should be 1");

    gv_json_index_destroy(idx);
    return 0;
}

/* ── Test: save and load ───────────────────────────────────────────────── */
static int test_save_load(void) {
    GV_JSONPathIndex *idx = gv_json_index_create();
    ASSERT(idx != NULL, "index creation");

    GV_JSONPathConfig pcfg;
    pcfg.path = "name";
    pcfg.type = GV_JP_STRING;
    gv_json_index_add_path(idx, &pcfg);

    gv_json_index_insert(idx, 0, "{\"name\": \"alice\"}");
    gv_json_index_insert(idx, 1, "{\"name\": \"bob\"}");

    int rc = gv_json_index_save(idx, TMP_INDEX_PATH);
    ASSERT(rc == 0, "save index");
    gv_json_index_destroy(idx);

    /* Load */
    GV_JSONPathIndex *loaded = gv_json_index_load(TMP_INDEX_PATH);
    ASSERT(loaded != NULL, "load index");
    ASSERT(gv_json_index_count(loaded, "name") == 2, "loaded index should have 2 entries");

    size_t out[10];
    int found = gv_json_index_lookup_string(loaded, "name", "alice", out, 10);
    ASSERT(found == 1, "lookup 'alice' in loaded index should return 1");

    gv_json_index_destroy(loaded);
    unlink(TMP_INDEX_PATH);
    return 0;
}

/* ── Test: float range lookup ──────────────────────────────────────────── */
static int test_float_range_lookup(void) {
    GV_JSONPathIndex *idx = gv_json_index_create();
    ASSERT(idx != NULL, "index creation");

    GV_JSONPathConfig pcfg;
    pcfg.path = "weight";
    pcfg.type = GV_JP_FLOAT;
    gv_json_index_add_path(idx, &pcfg);

    gv_json_index_insert(idx, 0, "{\"weight\": 1.5}");
    gv_json_index_insert(idx, 1, "{\"weight\": 3.7}");
    gv_json_index_insert(idx, 2, "{\"weight\": 5.2}");

    size_t out[10];
    int found = gv_json_index_lookup_float_range(idx, "weight", 2.0, 4.0, out, 10);
    ASSERT(found == 1, "float range [2.0,4.0] should match 1 entry (3.7)");
    ASSERT(out[0] == 1, "matched entry should be vector_index 1");

    gv_json_index_destroy(idx);
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing create/destroy...",          test_create_destroy},
        {"Testing add/remove path...",         test_add_remove_path},
        {"Testing insert/lookup string...",    test_insert_lookup_string},
        {"Testing insert/lookup int range...", test_insert_lookup_int_range},
        {"Testing remove entries...",          test_remove_entries},
        {"Testing count...",                   test_count},
        {"Testing save/load...",               test_save_load},
        {"Testing float range lookup...",      test_float_range_lookup},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
