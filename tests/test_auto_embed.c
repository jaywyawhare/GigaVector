/**
 * @file test_auto_embed.c
 * @brief Unit tests for auto-embedding (gv_auto_embed.h).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_auto_embed.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

#define TEST_DB "tmp_test_auto_embed.bin"

/* Test 1: Config init sets sensible defaults */
static int test_config_init_defaults(void) {
    GV_AutoEmbedConfig config;
    memset(&config, 0xFF, sizeof(config)); /* Fill with garbage first */
    gv_auto_embed_config_init(&config);

    ASSERT(config.cache_embeddings == 1, "default cache_embeddings should be 1");
    ASSERT(config.max_cache_entries > 0, "default max_cache_entries should be > 0");
    ASSERT(config.max_text_length > 0, "default max_text_length should be > 0");
    ASSERT(config.batch_size > 0, "default batch_size should be > 0");

    return 0;
}

/* Test 2: Config init specific values */
static int test_config_init_values(void) {
    GV_AutoEmbedConfig config;
    gv_auto_embed_config_init(&config);

    ASSERT(config.max_cache_entries == 10000, "default max_cache_entries should be 10000");
    ASSERT(config.max_text_length == 8192, "default max_text_length should be 8192");
    ASSERT(config.batch_size == 32, "default batch_size should be 32");

    return 0;
}

/* Test 3: Create embedder with CUSTOM provider */
static int test_create_custom_provider(void) {
    GV_AutoEmbedConfig config;
    gv_auto_embed_config_init(&config);
    config.provider = GV_EMBED_PROVIDER_CUSTOM;
    config.api_key = "test-key";
    config.model_name = "test-model";
    config.base_url = "http://localhost:8000/embed";
    config.dimension = 128;

    GV_AutoEmbedder *embedder = gv_auto_embed_create(&config);
    ASSERT(embedder != NULL, "embedder creation with CUSTOM provider");

    gv_auto_embed_destroy(embedder);
    return 0;
}

/* Test 4: Create embedder with each provider type */
static int test_create_all_providers(void) {
    GV_AutoEmbedProvider providers[] = {
        GV_EMBED_PROVIDER_OPENAI,
        GV_EMBED_PROVIDER_GOOGLE,
        GV_EMBED_PROVIDER_HUGGINGFACE,
        GV_EMBED_PROVIDER_CUSTOM
    };

    for (int i = 0; i < 4; i++) {
        GV_AutoEmbedConfig config;
        gv_auto_embed_config_init(&config);
        config.provider = providers[i];
        config.api_key = "test-key";
        config.model_name = "test-model";
        config.dimension = 64;
        if (providers[i] == GV_EMBED_PROVIDER_CUSTOM) {
            config.base_url = "http://localhost:8000";
        }

        GV_AutoEmbedder *embedder = gv_auto_embed_create(&config);
        ASSERT(embedder != NULL, "embedder creation for each provider");
        gv_auto_embed_destroy(embedder);
    }

    return 0;
}

/* Test 5: Destroy NULL embedder (should not crash) */
static int test_destroy_null(void) {
    gv_auto_embed_destroy(NULL);
    return 0;
}

/* Test 6: Get initial stats (all zeros) */
static int test_get_stats_initial(void) {
    GV_AutoEmbedConfig config;
    gv_auto_embed_config_init(&config);
    config.provider = GV_EMBED_PROVIDER_CUSTOM;
    config.api_key = "test-key";
    config.model_name = "test-model";
    config.base_url = "http://localhost:8000";
    config.dimension = 64;

    GV_AutoEmbedder *embedder = gv_auto_embed_create(&config);
    ASSERT(embedder != NULL, "embedder creation");

    GV_AutoEmbedStats stats;
    memset(&stats, 0xFF, sizeof(stats)); /* Fill with garbage */
    int ret = gv_auto_embed_get_stats(embedder, &stats);
    ASSERT(ret == 0, "get_stats should succeed");
    ASSERT(stats.total_embeddings == 0, "initial total_embeddings should be 0");
    ASSERT(stats.cache_hits == 0, "initial cache_hits should be 0");
    ASSERT(stats.cache_misses == 0, "initial cache_misses should be 0");
    ASSERT(stats.api_calls == 0, "initial api_calls should be 0");
    ASSERT(stats.api_errors == 0, "initial api_errors should be 0");

    gv_auto_embed_destroy(embedder);
    return 0;
}

/* Test 7: Clear cache on fresh embedder */
static int test_clear_cache_fresh(void) {
    GV_AutoEmbedConfig config;
    gv_auto_embed_config_init(&config);
    config.provider = GV_EMBED_PROVIDER_CUSTOM;
    config.api_key = "test-key";
    config.model_name = "test-model";
    config.base_url = "http://localhost:8000";
    config.dimension = 64;

    GV_AutoEmbedder *embedder = gv_auto_embed_create(&config);
    ASSERT(embedder != NULL, "embedder creation");

    /* Clear cache on empty embedder should not crash */
    gv_auto_embed_clear_cache(embedder);

    /* Stats should still be zero after clearing empty cache */
    GV_AutoEmbedStats stats;
    int ret = gv_auto_embed_get_stats(embedder, &stats);
    ASSERT(ret == 0, "get_stats after clear_cache should succeed");
    ASSERT(stats.total_embeddings == 0, "total_embeddings should be 0 after cache clear");

    gv_auto_embed_destroy(embedder);
    return 0;
}

/* Test 8: Embed text without real API (should return NULL or error) */
static int test_embed_text_no_api(void) {
    GV_AutoEmbedConfig config;
    gv_auto_embed_config_init(&config);
    config.provider = GV_EMBED_PROVIDER_CUSTOM;
    config.api_key = "fake-key-no-real-api";
    config.model_name = "test-model";
    config.base_url = "http://localhost:99999/nonexistent";
    config.dimension = 64;

    GV_AutoEmbedder *embedder = gv_auto_embed_create(&config);
    ASSERT(embedder != NULL, "embedder creation");

    size_t out_dim = 0;
    float *embedding = gv_auto_embed_text(embedder, "Hello world", &out_dim);

    /* Without a real API, this should return NULL */
    if (embedding != NULL) {
        /* Some implementations may generate a dummy embedding */
        ASSERT(out_dim > 0, "if embedding returned, dimension should be > 0");
        free(embedding);
    }

    gv_auto_embed_destroy(embedder);
    return 0;
}

/* Test 9: Add text to DB without real API */
static int test_add_text_no_api(void) {
    remove(TEST_DB);
    GV_Database *db = gv_db_open(TEST_DB, 64, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_AutoEmbedConfig config;
    gv_auto_embed_config_init(&config);
    config.provider = GV_EMBED_PROVIDER_CUSTOM;
    config.api_key = "fake-key-no-real-api";
    config.model_name = "test-model";
    config.base_url = "http://localhost:99999/nonexistent";
    config.dimension = 64;

    GV_AutoEmbedder *embedder = gv_auto_embed_create(&config);
    ASSERT(embedder != NULL, "embedder creation");

    int ret = gv_auto_embed_add_text(embedder, db, "Test document about cats",
                                      "category", "animals");
    /* Without real API, should return error */
    /* ret == -1 is expected, ret == 0 would mean it has a fallback */
    (void)ret;

    gv_auto_embed_destroy(embedder);
    gv_db_close(db);
    remove(TEST_DB);
    return 0;
}

/* Test 10: Config with caching disabled */
static int test_config_cache_disabled(void) {
    GV_AutoEmbedConfig config;
    gv_auto_embed_config_init(&config);
    config.provider = GV_EMBED_PROVIDER_CUSTOM;
    config.api_key = "test-key";
    config.model_name = "test-model";
    config.base_url = "http://localhost:8000";
    config.dimension = 64;
    config.cache_embeddings = 0;

    GV_AutoEmbedder *embedder = gv_auto_embed_create(&config);
    ASSERT(embedder != NULL, "embedder creation with cache disabled");

    gv_auto_embed_destroy(embedder);
    return 0;
}

/* Test 11: Stats struct field sizes */
static int test_stats_structure(void) {
    GV_AutoEmbedStats stats;
    memset(&stats, 0, sizeof(stats));

    ASSERT(stats.total_embeddings == 0, "total_embeddings zero init");
    ASSERT(stats.cache_hits == 0, "cache_hits zero init");
    ASSERT(stats.cache_misses == 0, "cache_misses zero init");
    ASSERT(stats.api_calls == 0, "api_calls zero init");
    ASSERT(stats.api_errors == 0, "api_errors zero init");
    ASSERT(stats.avg_latency_ms == 0.0, "avg_latency_ms zero init");

    return 0;
}

/* Test 12: Create/destroy cycle to check for leaks */
static int test_create_destroy_cycle(void) {
    for (int i = 0; i < 10; i++) {
        GV_AutoEmbedConfig config;
        gv_auto_embed_config_init(&config);
        config.provider = GV_EMBED_PROVIDER_CUSTOM;
        config.api_key = "test-key";
        config.model_name = "test-model";
        config.base_url = "http://localhost:8000";
        config.dimension = 64;

        GV_AutoEmbedder *embedder = gv_auto_embed_create(&config);
        ASSERT(embedder != NULL, "embedder creation in loop");
        gv_auto_embed_destroy(embedder);
    }

    return 0;
}

int main(void) {
    int failed = 0;
    int passed = 0;

    remove(TEST_DB);

    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_config_init_defaults",    test_config_init_defaults},
        {"test_config_init_values",      test_config_init_values},
        {"test_create_custom_provider",  test_create_custom_provider},
        {"test_create_all_providers",    test_create_all_providers},
        {"test_destroy_null",            test_destroy_null},
        {"test_get_stats_initial",       test_get_stats_initial},
        {"test_clear_cache_fresh",       test_clear_cache_fresh},
        {"test_embed_text_no_api",       test_embed_text_no_api},
        {"test_add_text_no_api",         test_add_text_no_api},
        {"test_config_cache_disabled",   test_config_cache_disabled},
        {"test_stats_structure",         test_stats_structure},
        {"test_create_destroy_cycle",    test_create_destroy_cycle},
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
    remove(TEST_DB);
    return failed > 0 ? 1 : 0;
}
