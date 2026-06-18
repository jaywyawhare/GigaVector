/**
 * @file test_inference.c
 * @brief Unit tests for inference engine (inference.h).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/database.h"
#include "multimodal/inference.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

#define TEST_DB "tmp_test_inference.bin"
#define TEST_DB_GOOGLE "tmp_test_inference_google.bin"

static void remove_test_db(const char *path) {
    char wal_path[256];
    remove(path);
    snprintf(wal_path, sizeof(wal_path), "%s.wal", path);
    remove(wal_path);
}

static const char *get_google_api_key(void) {
    const char *key = getenv("GOOGLE_API_KEY");
    if (key && strlen(key) > 0) {
        return key;
    }
    key = getenv("GEMINI_API_KEY");
    return (key && strlen(key) > 0) ? key : NULL;
}

static int test_config_init_defaults(void) {
    GV_InferenceConfig config;
    memset(&config, 0xFF, sizeof(config));
    inference_config_init(&config);

    ASSERT(config.embed_provider != NULL, "default embed_provider should not be NULL");
    ASSERT(strcmp(config.embed_provider, "openai") == 0, "default provider should be openai");
    ASSERT(config.model != NULL, "default model should not be NULL");
    ASSERT(strcmp(config.model, "text-embedding-3-small") == 0, "default model check");
    ASSERT(config.dimension == 1536, "default dimension should be 1536");
    ASSERT(config.distance_type == 1, "default distance_type should be 1 (cosine)");
    ASSERT(config.cache_size == 10000, "default cache_size should be 10000");

    return 0;
}

static int test_config_modify(void) {
    GV_InferenceConfig config;
    inference_config_init(&config);

    config.embed_provider = "google";
    config.api_key = "test-google-key";
    config.model = "text-embedding-004";
    config.dimension = 768;
    config.distance_type = 0;
    config.cache_size = 5000;

    ASSERT(strcmp(config.embed_provider, "google") == 0, "modified provider");
    ASSERT(strcmp(config.api_key, "test-google-key") == 0, "modified api_key");
    ASSERT(config.dimension == 768, "modified dimension");
    ASSERT(config.distance_type == 0, "modified distance_type");
    ASSERT(config.cache_size == 5000, "modified cache_size");

    return 0;
}

static int test_create_engine(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.api_key = "test-key";
    config.dimension = 4;

    GV_InferenceEngine *eng = inference_create(db, &config);
    ASSERT(eng != NULL, "inference engine creation");

    inference_destroy(eng);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_create_null_db(void) {
    GV_InferenceConfig config;
    inference_config_init(&config);
    config.api_key = "test-key";

    GV_InferenceEngine *eng = inference_create(NULL, &config);
    ASSERT(eng == NULL, "engine creation with NULL db should fail");

    return 0;
}

static int test_create_null_config(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceEngine *eng = inference_create(db, NULL);
    ASSERT(eng == NULL, "engine creation with NULL config should fail");

    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_destroy_null(void) {
    inference_destroy(NULL);
    return 0;
}

static int test_add_no_provider(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.api_key = "fake-key-no-real-api";
    config.dimension = 4;

    GV_InferenceEngine *eng = inference_create(db, &config);
    ASSERT(eng != NULL, "inference engine creation");

    int ret = inference_add(eng, "Hello world, this is a test document.", NULL);
    (void)ret;

    inference_destroy(eng);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_search_empty(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.api_key = "fake-key-no-real-api";
    config.dimension = 4;

    GV_InferenceEngine *eng = inference_create(db, &config);
    ASSERT(eng != NULL, "inference engine creation");

    GV_InferenceResult results[5];
    memset(results, 0, sizeof(results));
    int count = inference_search(eng, "find similar documents", 5, results);

    if (count > 0) {
        inference_free_results(results, (size_t)count);
    }

    inference_destroy(eng);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}


static int test_free_results_null(void) {
    inference_free_results(NULL, 0);
    inference_free_results(NULL, 5);
    return 0;
}

static int test_free_results_empty(void) {
    GV_InferenceResult results[3];
    memset(results, 0, sizeof(results));
    inference_free_results(results, 0);
    return 0;
}

static int test_result_structure(void) {
    GV_InferenceResult result;
    memset(&result, 0, sizeof(result));

    ASSERT(result.index == 0, "default index should be 0");
    ASSERT(result.distance == 0.0f, "default distance should be 0.0");
    ASSERT(result.text == NULL, "default text should be NULL");
    ASSERT(result.metadata_json == NULL, "default metadata_json should be NULL");

    return 0;
}

static int test_create_destroy_cycle(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.api_key = "test-key";
    config.dimension = 4;

    for (int i = 0; i < 10; i++) {
        GV_InferenceEngine *eng = inference_create(db, &config);
        ASSERT(eng != NULL, "engine creation in loop");
        inference_destroy(eng);
    }

    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_add_with_metadata(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.api_key = "fake-key-no-real-api";
    config.dimension = 4;

    GV_InferenceEngine *eng = inference_create(db, &config);
    ASSERT(eng != NULL, "inference engine creation");

    const char *metadata = "{\"category\": \"science\", \"author\": \"John Doe\"}";
    int ret = inference_add(eng, "A research paper about quantum physics", metadata);
    (void)ret;

    inference_destroy(eng);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}


static int test_add_batch_no_provider(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.api_key = "fake-key-no-real-api";
    config.dimension = 4;

    GV_InferenceEngine *eng = inference_create(db, &config);
    ASSERT(eng != NULL, "inference engine creation");

    const char *texts[] = {
        "Document about cats",
        "Document about dogs",
        "Document about birds"
    };
    const char *metas[] = {
        "{\"animal\": \"cat\"}",
        "{\"animal\": \"dog\"}",
        NULL
    };

    int ret = inference_add_batch(eng, texts, metas, 3);
    (void)ret;

    inference_destroy(eng);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_search_filtered_no_provider(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.api_key = "fake-key-no-real-api";
    config.dimension = 4;

    GV_InferenceEngine *eng = inference_create(db, &config);
    ASSERT(eng != NULL, "inference engine creation");

    GV_InferenceResult results[5];
    memset(results, 0, sizeof(results));
    int count = inference_search_filtered(eng, "find cats", 5,
                                              "category == \"animals\"", results);
    if (count > 0) {
        inference_free_results(results, (size_t)count);
    }

    inference_destroy(eng);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_create_engine_google_defaults(void) {
    remove_test_db(TEST_DB_GOOGLE);
    GV_Database *db = db_open(TEST_DB_GOOGLE, 768, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.embed_provider = "google";
    config.api_key = "test-key";
    config.dimension = 0;

    GV_InferenceEngine *eng = inference_create(db, &config);
    ASSERT(eng != NULL, "google inference engine creation with defaults");

    inference_destroy(eng);
    db_close(db);
    remove_test_db(TEST_DB_GOOGLE);
    return 0;
}

static int test_google_live_add_search(void) {
    const char *api_key = get_google_api_key();
    if (!api_key) {
        return 0;
    }

    remove_test_db(TEST_DB_GOOGLE);
    GV_Database *db = db_open(TEST_DB_GOOGLE, 768, GV_INDEX_TYPE_FLAT);
    if (db == NULL) {
        return 0;
    }

    GV_InferenceConfig config;
    inference_config_init(&config);
    config.embed_provider = "google";
    config.api_key = (char *)api_key;
    config.model = "text-embedding-004";
    config.dimension = 768;

    GV_InferenceEngine *eng = inference_create(db, &config);
    if (eng == NULL) {
        db_close(db);
        remove_test_db(TEST_DB_GOOGLE);
        return 0;
    }

    if (inference_add(eng, "cats and dogs", "{\"topic\":\"pets\"}") == 0) {
        GV_InferenceResult results[5];
        memset(results, 0, sizeof(results));
        int count = inference_search(eng, "pets", 1, results);
        if (count > 0) {
            inference_free_results(results, (size_t)count);
        }
    }

    inference_destroy(eng);
    db_close(db);
    remove_test_db(TEST_DB_GOOGLE);
    return 0;
}

int main(void) {
    int failed = 0;
    int passed = 0;

    gv_test_remove_db(TEST_DB);

    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_config_init_defaults",         test_config_init_defaults},
        {"test_config_modify",                test_config_modify},
        {"test_create_engine",                test_create_engine},
        {"test_create_null_db",               test_create_null_db},
        {"test_create_null_config",           test_create_null_config},
        {"test_destroy_null",                 test_destroy_null},
        {"test_add_no_provider",              test_add_no_provider},
        {"test_search_empty",                 test_search_empty},
        {"test_free_results_null",            test_free_results_null},
        {"test_free_results_empty",           test_free_results_empty},
        {"test_result_structure",             test_result_structure},
        {"test_create_destroy_cycle",         test_create_destroy_cycle},
        {"test_add_with_metadata",            test_add_with_metadata},
        {"test_add_batch_no_provider",        test_add_batch_no_provider},
        {"test_search_filtered_no_provider",  test_search_filtered_no_provider},
        {"test_create_engine_google_defaults", test_create_engine_google_defaults},
        {"test_google_live_add_search",       test_google_live_add_search},
    };

    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < num_tests; i++) {
        int result = tests[i].fn();
        if (result == 0) {
            printf("  OK   %s\n", tests[i].name);
            passed++;
        } else {
            printf("  FAILED %s\n", tests[i].name);
            failed++;
        }
    }

    printf("\n%d/%d tests passed\n", passed, num_tests);
    gv_test_remove_db(TEST_DB);
    return failed > 0 ? 1 : 0;
}
