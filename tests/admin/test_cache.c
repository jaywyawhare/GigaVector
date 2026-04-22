#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "admin/cache.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_cache_config_init(void) {
    GV_CacheConfig config;
    memset(&config, 0xFF, sizeof(config));
    cache_config_init(&config);

    ASSERT(config.max_entries > 0, "default max_entries is positive");
    ASSERT(config.max_memory_bytes > 0, "default max_memory_bytes is positive");
    ASSERT(config.policy == GV_CACHE_LRU, "default policy is LRU");
    return 0;
}

static int test_cache_create_destroy(void) {
    GV_Cache *cache = cache_create(NULL);
    ASSERT(cache != NULL, "cache creation with NULL config");
    cache_destroy(cache);

    GV_CacheConfig config;
    cache_config_init(&config);
    config.max_entries = 64;
    config.ttl_seconds = 30;
    cache = cache_create(&config);
    ASSERT(cache != NULL, "cache creation with explicit config");
    cache_destroy(cache);

    cache_destroy(NULL);
    return 0;
}

static int test_cache_store_and_lookup(void) {
    GV_CacheConfig config;
    cache_config_init(&config);
    config.ttl_seconds = 0; /* No expiry for testing */
    GV_Cache *cache = cache_create(&config);
    ASSERT(cache != NULL, "cache creation");

    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t indices[3] = {10, 20, 30};
    float distances[3] = {0.1f, 0.5f, 1.0f};

    ASSERT(cache_store(cache, query, 4, 3, 0, indices, distances, 3) == 0,
           "store cache entry");

    GV_CachedResult result;
    memset(&result, 0, sizeof(result));
    int hit = cache_lookup(cache, query, 4, 3, 0, &result);
    ASSERT(hit == 1, "cache hit on stored query");
    ASSERT(result.count == 3, "cached result has correct count");
    ASSERT(result.indices != NULL, "cached result has indices");
    ASSERT(result.distances != NULL, "cached result has distances");
    ASSERT(result.indices[0] == 10, "first cached index correct");
    ASSERT(result.distances[0] - 0.1f < 1e-5f, "first cached distance correct");

    cache_free_result(&result);
    cache_destroy(cache);
    return 0;
}

static int test_cache_miss(void) {
    GV_Cache *cache = cache_create(NULL);
    ASSERT(cache != NULL, "cache creation");

    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_CachedResult result;
    memset(&result, 0, sizeof(result));

    int hit = cache_lookup(cache, query, 4, 3, 0, &result);
    ASSERT(hit == 0, "cache miss on empty cache");

    cache_destroy(cache);
    return 0;
}

static int test_cache_invalidate_all(void) {
    GV_CacheConfig config;
    cache_config_init(&config);
    config.ttl_seconds = 0;
    GV_Cache *cache = cache_create(&config);
    ASSERT(cache != NULL, "cache creation");

    float query1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float query2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    size_t idx[1] = {0};
    float dist[1] = {0.0f};

    cache_store(cache, query1, 4, 1, 0, idx, dist, 1);
    cache_store(cache, query2, 4, 1, 0, idx, dist, 1);

    GV_CachedResult result;
    memset(&result, 0, sizeof(result));
    ASSERT(cache_lookup(cache, query1, 4, 1, 0, &result) == 1, "hit before invalidate");
    cache_free_result(&result);

    cache_invalidate_all(cache);

    memset(&result, 0, sizeof(result));
    ASSERT(cache_lookup(cache, query1, 4, 1, 0, &result) == 0, "miss after invalidate");
    memset(&result, 0, sizeof(result));
    ASSERT(cache_lookup(cache, query2, 4, 1, 0, &result) == 0, "miss after invalidate (query2)");

    cache_destroy(cache);
    return 0;
}

static int test_cache_mutation_invalidation(void) {
    GV_CacheConfig config;
    cache_config_init(&config);
    config.ttl_seconds = 0;
    config.invalidate_after_mutations = 3;
    GV_Cache *cache = cache_create(&config);
    ASSERT(cache != NULL, "cache creation");

    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t idx[1] = {5};
    float dist[1] = {0.2f};
    cache_store(cache, query, 4, 1, 0, idx, dist, 1);

    cache_notify_mutation(cache);
    cache_notify_mutation(cache);

    GV_CachedResult result;
    memset(&result, 0, sizeof(result));
    int hit = cache_lookup(cache, query, 4, 1, 0, &result);
    /* Might still be a hit (threshold not yet reached) */
    if (hit == 1) cache_free_result(&result);

    cache_notify_mutation(cache);

    memset(&result, 0, sizeof(result));
    hit = cache_lookup(cache, query, 4, 1, 0, &result);
    ASSERT(hit == 0, "cache invalidated after mutation threshold");

    cache_destroy(cache);
    return 0;
}

static int test_cache_stats(void) {
    GV_Cache *cache = cache_create(NULL);
    ASSERT(cache != NULL, "cache creation");

    GV_CacheStats stats;
    ASSERT(cache_get_stats(cache, &stats) == 0, "get stats on empty cache");
    ASSERT(stats.hits == 0, "initial hits is 0");
    ASSERT(stats.misses == 0, "initial misses is 0");

    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    GV_CachedResult result;
    memset(&result, 0, sizeof(result));
    cache_lookup(cache, query, 4, 1, 0, &result);

    ASSERT(cache_get_stats(cache, &stats) == 0, "get stats after miss");
    ASSERT(stats.misses == 1, "misses incremented to 1");

    size_t idx[1] = {0};
    float dist[1] = {0.0f};
    cache_store(cache, query, 4, 1, 0, idx, dist, 1);
    memset(&result, 0, sizeof(result));
    cache_lookup(cache, query, 4, 1, 0, &result);
    cache_free_result(&result);

    ASSERT(cache_get_stats(cache, &stats) == 0, "get stats after hit");
    ASSERT(stats.hits == 1, "hits incremented to 1");

    cache_reset_stats(cache);
    ASSERT(cache_get_stats(cache, &stats) == 0, "get stats after reset");
    ASSERT(stats.hits == 0, "hits reset to 0");
    ASSERT(stats.misses == 0, "misses reset to 0");

    cache_destroy(cache);
    return 0;
}

static int test_cache_different_params_no_hit(void) {
    GV_CacheConfig config;
    cache_config_init(&config);
    config.ttl_seconds = 0;
    GV_Cache *cache = cache_create(&config);
    ASSERT(cache != NULL, "cache creation");

    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t idx[2] = {0, 1};
    float dist[2] = {0.1f, 0.5f};

    cache_store(cache, query, 4, 2, 0, idx, dist, 2);

    GV_CachedResult result;
    memset(&result, 0, sizeof(result));
    int hit = cache_lookup(cache, query, 4, 5, 0, &result);
    ASSERT(hit == 0, "different k causes cache miss");

    memset(&result, 0, sizeof(result));
    hit = cache_lookup(cache, query, 4, 2, 1, &result);
    ASSERT(hit == 0, "different distance_type causes cache miss");

    cache_destroy(cache);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing cache config init...", test_cache_config_init},
        {"Testing cache create/destroy...", test_cache_create_destroy},
        {"Testing cache store and lookup...", test_cache_store_and_lookup},
        {"Testing cache miss...", test_cache_miss},
        {"Testing cache invalidate all...", test_cache_invalidate_all},
        {"Testing cache mutation invalidation...", test_cache_mutation_invalidation},
        {"Testing cache stats...", test_cache_stats},
        {"Testing cache different params no hit...", test_cache_different_params_no_hit},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
