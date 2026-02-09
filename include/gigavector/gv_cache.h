#ifndef GIGAVECTOR_GV_CACHE_H
#define GIGAVECTOR_GV_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_cache.h
 * @brief Client-side query result caching for GigaVector.
 *
 * Provides an LRU cache for search results. Cache keys are derived
 * from query vector content + search parameters. Entries automatically
 * expire based on configurable TTL or database mutation count.
 */

/**
 * @brief Cache eviction policy.
 */
typedef enum {
    GV_CACHE_LRU = 0,          /**< Least Recently Used (default). */
    GV_CACHE_LFU = 1           /**< Least Frequently Used. */
} GV_CachePolicy;

/**
 * @brief Cache configuration.
 */
typedef struct {
    size_t max_entries;         /**< Maximum cached results (default: 1024). */
    size_t max_memory_bytes;    /**< Maximum memory usage (default: 64MB). */
    uint32_t ttl_seconds;       /**< Entry TTL in seconds (0 = no expiry, default: 60). */
    uint64_t invalidate_after_mutations; /**< Invalidate after N mutations (0 = disabled). */
    GV_CachePolicy policy;     /**< Eviction policy (default: LRU). */
} GV_CacheConfig;

/**
 * @brief Cache statistics.
 */
typedef struct {
    uint64_t hits;              /**< Cache hits. */
    uint64_t misses;            /**< Cache misses. */
    uint64_t evictions;         /**< Entries evicted. */
    uint64_t invalidations;     /**< Entries invalidated. */
    size_t current_entries;     /**< Current number of entries. */
    size_t current_memory;      /**< Current memory usage in bytes. */
    double hit_rate;            /**< Hit rate (hits / (hits + misses)). */
} GV_CacheStats;

/**
 * @brief Cached search result entry.
 */
typedef struct {
    size_t *indices;            /**< Result vector indices. */
    float *distances;           /**< Result distances. */
    size_t count;               /**< Number of results. */
} GV_CachedResult;

/**
 * @brief Opaque cache handle.
 */
typedef struct GV_Cache GV_Cache;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Initialize cache configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void gv_cache_config_init(GV_CacheConfig *config);

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/**
 * @brief Create a cache instance.
 *
 * @param config Cache configuration; NULL for defaults.
 * @return Cache instance, or NULL on error.
 */
GV_Cache *gv_cache_create(const GV_CacheConfig *config);

/**
 * @brief Destroy a cache instance and free all resources.
 *
 * @param cache Cache instance (safe to call with NULL).
 */
void gv_cache_destroy(GV_Cache *cache);

/* ============================================================================
 * Cache Operations
 * ============================================================================ */

/**
 * @brief Look up a cached search result.
 *
 * @param cache Cache instance.
 * @param query_data Query vector data.
 * @param dimension Vector dimension.
 * @param k Number of neighbors requested.
 * @param distance_type Distance metric used.
 * @param result Output cached result (caller must free with gv_cache_free_result).
 * @return 1 if cache hit, 0 if miss, -1 on error.
 */
int gv_cache_lookup(GV_Cache *cache, const float *query_data, size_t dimension,
                    size_t k, int distance_type, GV_CachedResult *result);

/**
 * @brief Store a search result in the cache.
 *
 * @param cache Cache instance.
 * @param query_data Query vector data.
 * @param dimension Vector dimension.
 * @param k Number of neighbors.
 * @param distance_type Distance metric.
 * @param indices Result indices array.
 * @param distances Result distances array.
 * @param count Number of results.
 * @return 0 on success, -1 on error.
 */
int gv_cache_store(GV_Cache *cache, const float *query_data, size_t dimension,
                   size_t k, int distance_type,
                   const size_t *indices, const float *distances, size_t count);

/**
 * @brief Notify the cache of a database mutation (insert/update/delete).
 *
 * Increments the internal mutation counter. If invalidate_after_mutations
 * is configured, the cache will be flushed when the threshold is reached.
 *
 * @param cache Cache instance.
 */
void gv_cache_notify_mutation(GV_Cache *cache);

/**
 * @brief Invalidate all cache entries.
 *
 * @param cache Cache instance.
 */
void gv_cache_invalidate_all(GV_Cache *cache);

/**
 * @brief Free a cached result.
 *
 * @param result Result to free.
 */
void gv_cache_free_result(GV_CachedResult *result);

/* ============================================================================
 * Statistics
 * ============================================================================ */

/**
 * @brief Get cache statistics.
 *
 * @param cache Cache instance.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int gv_cache_get_stats(const GV_Cache *cache, GV_CacheStats *stats);

/**
 * @brief Reset cache statistics counters.
 *
 * @param cache Cache instance.
 */
void gv_cache_reset_stats(GV_Cache *cache);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CACHE_H */
