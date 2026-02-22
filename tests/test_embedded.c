#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "gigavector/gv_embedded.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 4
#define TMP_SAVE_PATH "/tmp/gv_test_embedded.bin"

/* ── Test: config init ─────────────────────────────────────────────────── */
static int test_config_init(void) {
    GV_EmbeddedConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_embedded_config_init(&cfg);

    ASSERT(cfg.dimension == 0, "default dimension should be 0 (caller sets it)");
    ASSERT(cfg.index_type == GV_EMBEDDED_INDEX_FLAT, "default index_type should be FLAT");
    ASSERT(cfg.max_vectors == 0, "default max_vectors should be 0 (unlimited)");
    ASSERT(cfg.memory_limit_mb == 64, "default memory_limit_mb should be 64");
    ASSERT(cfg.mmap_storage == 0, "default mmap_storage should be 0");
    ASSERT(cfg.storage_path == NULL, "default storage_path should be NULL");
    ASSERT(cfg.quantize == 0, "default quantize should be 0");
    return 0;
}

/* ── Test: open and close ──────────────────────────────────────────────── */
static int test_open_close(void) {
    GV_EmbeddedConfig cfg;
    gv_embedded_config_init(&cfg);
    cfg.dimension = DIM;

    GV_EmbeddedDB *db = gv_embedded_open(&cfg);
    ASSERT(db != NULL, "embedded db open");
    ASSERT(gv_embedded_count(db) == 0, "empty db should have count 0");

    gv_embedded_close(db);
    /* NULL safety */
    gv_embedded_close(NULL);
    return 0;
}

/* ── Test: add and count ───────────────────────────────────────────────── */
static int test_add_count(void) {
    GV_EmbeddedConfig cfg;
    gv_embedded_config_init(&cfg);
    cfg.dimension = DIM;

    GV_EmbeddedDB *db = gv_embedded_open(&cfg);
    ASSERT(db != NULL, "db open");

    float v1[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[DIM] = {0.0f, 1.0f, 0.0f, 0.0f};

    int idx1 = gv_embedded_add(db, v1);
    ASSERT(idx1 >= 0, "add v1 should succeed");
    int idx2 = gv_embedded_add(db, v2);
    ASSERT(idx2 >= 0, "add v2 should succeed");
    ASSERT(gv_embedded_count(db) == 2, "count should be 2");

    gv_embedded_close(db);
    return 0;
}

/* ── Test: add with explicit ID and get ────────────────────────────────── */
static int test_add_with_id_and_get(void) {
    GV_EmbeddedConfig cfg;
    gv_embedded_config_init(&cfg);
    cfg.dimension = DIM;

    GV_EmbeddedDB *db = gv_embedded_open(&cfg);
    ASSERT(db != NULL, "db open");

    float vec[DIM] = {3.0f, 1.4f, 1.5f, 9.2f};
    int rc = gv_embedded_add_with_id(db, 5, vec);
    ASSERT(rc == 0, "add with id=5 should succeed");

    float out[DIM] = {0};
    rc = gv_embedded_get(db, 5, out);
    ASSERT(rc == 0, "get id=5 should succeed");
    ASSERT(fabsf(out[0] - 3.0f) < 1e-6f, "retrieved vector[0] should match");
    ASSERT(fabsf(out[3] - 9.2f) < 1e-6f, "retrieved vector[3] should match");

    gv_embedded_close(db);
    return 0;
}

/* ── Test: search ──────────────────────────────────────────────────────── */
static int test_search(void) {
    GV_EmbeddedConfig cfg;
    gv_embedded_config_init(&cfg);
    cfg.dimension = DIM;

    GV_EmbeddedDB *db = gv_embedded_open(&cfg);
    ASSERT(db != NULL, "db open");

    float vectors[5][DIM] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f},
    };
    for (int i = 0; i < 5; i++) {
        gv_embedded_add(db, vectors[i]);
    }

    float query[DIM] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_EmbeddedResult results[3];
    memset(results, 0, sizeof(results));

    int found = gv_embedded_search(db, query, 3, 0 /* euclidean */, results);
    ASSERT(found > 0, "search should return at least 1 result");
    /* Closest should be index 0 (exact match) */
    ASSERT(results[0].index == 0, "nearest neighbor should be index 0");
    ASSERT(results[0].distance < 0.01f, "distance to exact match should be near 0");

    gv_embedded_close(db);
    return 0;
}

/* ── Test: delete and compact ──────────────────────────────────────────── */
static int test_delete_compact(void) {
    GV_EmbeddedConfig cfg;
    gv_embedded_config_init(&cfg);
    cfg.dimension = DIM;

    GV_EmbeddedDB *db = gv_embedded_open(&cfg);
    ASSERT(db != NULL, "db open");

    float v[DIM] = {1.0f, 2.0f, 3.0f, 4.0f};
    gv_embedded_add(db, v);
    gv_embedded_add(db, v);
    gv_embedded_add(db, v);
    ASSERT(gv_embedded_count(db) == 3, "count should be 3 before delete");

    int rc = gv_embedded_delete(db, 1);
    ASSERT(rc == 0, "delete index 1 should succeed");
    ASSERT(gv_embedded_count(db) == 2, "count should be 2 after delete");

    rc = gv_embedded_compact(db);
    ASSERT(rc == 0, "compact should succeed");

    gv_embedded_close(db);
    return 0;
}

/* ── Test: save and load ───────────────────────────────────────────────── */
static int test_save_load(void) {
    GV_EmbeddedConfig cfg;
    gv_embedded_config_init(&cfg);
    cfg.dimension = DIM;

    GV_EmbeddedDB *db = gv_embedded_open(&cfg);
    ASSERT(db != NULL, "db open");

    float v[DIM] = {1.0f, 2.0f, 3.0f, 4.0f};
    gv_embedded_add(db, v);
    gv_embedded_add(db, v);

    int rc = gv_embedded_save(db, TMP_SAVE_PATH);
    ASSERT(rc == 0, "save should succeed");
    gv_embedded_close(db);

    /* Load */
    GV_EmbeddedDB *loaded = gv_embedded_load(TMP_SAVE_PATH);
    ASSERT(loaded != NULL, "load should succeed");
    ASSERT(gv_embedded_count(loaded) == 2, "loaded db should have 2 vectors");

    float out[DIM] = {0};
    rc = gv_embedded_get(loaded, 0, out);
    ASSERT(rc == 0, "get from loaded db should succeed");
    ASSERT(fabsf(out[0] - 1.0f) < 1e-6f, "loaded vector data should match");

    gv_embedded_close(loaded);
    unlink(TMP_SAVE_PATH);
    return 0;
}

/* ── Test: memory usage ────────────────────────────────────────────────── */
static int test_memory_usage(void) {
    GV_EmbeddedConfig cfg;
    gv_embedded_config_init(&cfg);
    cfg.dimension = DIM;

    GV_EmbeddedDB *db = gv_embedded_open(&cfg);
    ASSERT(db != NULL, "db open");

    size_t mem_before = gv_embedded_memory_usage(db);

    float v[DIM] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 100; i++) {
        gv_embedded_add(db, v);
    }

    size_t mem_after = gv_embedded_memory_usage(db);
    ASSERT(mem_after >= mem_before, "memory usage should not decrease after adding vectors");

    gv_embedded_close(db);
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config init...",          test_config_init},
        {"Testing open/close...",           test_open_close},
        {"Testing add/count...",            test_add_count},
        {"Testing add with id and get...",  test_add_with_id_and_get},
        {"Testing search...",               test_search},
        {"Testing delete/compact...",       test_delete_compact},
        {"Testing save/load...",            test_save_load},
        {"Testing memory usage...",         test_memory_usage},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
