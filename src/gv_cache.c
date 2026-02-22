/**
 * @file gv_cache.c
 * @brief Client-side query result caching implementation.
 */

#include "gigavector/gv_cache.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

/*  Internal Structures  */

typedef struct CacheEntry {
    uint64_t key_hash;          /* Hash of query vector + params */
    float *query_data;          /* Cached query vector (for collision check) */
    size_t dimension;
    size_t k;
    int distance_type;

    size_t *indices;            /* Cached result indices */
    float *distances;           /* Cached result distances */
    size_t count;               /* Number of results */

    uint64_t created_at;        /* Creation timestamp (seconds) */
    uint64_t last_access;       /* Last access timestamp */
    uint64_t access_count;      /* Access frequency (for LFU) */

    size_t memory_size;         /* Total memory used by this entry */

    struct CacheEntry *lru_prev;  /* Doubly-linked list for LRU ordering */
    struct CacheEntry *lru_next;
    struct CacheEntry *hash_next; /* Hash bucket chain */
} CacheEntry;

#define CACHE_BUCKETS 1024

struct GV_Cache {
    GV_CacheConfig config;

    /* Hash table */
    CacheEntry *buckets[CACHE_BUCKETS];

    /* LRU doubly-linked list (head = most recently used) */
    CacheEntry *lru_head;
    CacheEntry *lru_tail;

    /* Stats */
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t invalidations;
    size_t current_entries;
    size_t current_memory;

    /* Mutation tracking */
    uint64_t mutation_count;

    pthread_mutex_t lock;
};

/*  Hash Functions  */

static uint64_t fnv1a_64(const void *data, size_t len) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t compute_cache_key(const float *query_data, size_t dimension,
                                   size_t k, int distance_type) {
    uint64_t hash = fnv1a_64(query_data, dimension * sizeof(float));
    hash ^= fnv1a_64(&k, sizeof(k));
    hash ^= fnv1a_64(&distance_type, sizeof(distance_type));
    return hash;
}

static size_t bucket_idx(uint64_t hash) {
    return hash % CACHE_BUCKETS;
}

/*  Internal Helpers  */

static uint64_t current_time_seconds(void) {
    return (uint64_t)time(NULL);
}

static int entries_match(const CacheEntry *entry, const float *query_data,
                         size_t dimension, size_t k, int distance_type) {
    if (entry->dimension != dimension) return 0;
    if (entry->k != k) return 0;
    if (entry->distance_type != distance_type) return 0;
    return memcmp(entry->query_data, query_data, dimension * sizeof(float)) == 0;
}

static size_t entry_memory_size(size_t dimension, size_t count) {
    return sizeof(CacheEntry) +
           dimension * sizeof(float) +    /* query_data */
           count * sizeof(size_t) +       /* indices */
           count * sizeof(float);         /* distances */
}

/* Remove entry from LRU list */
static void lru_remove(GV_Cache *cache, CacheEntry *entry) {
    if (entry->lru_prev) entry->lru_prev->lru_next = entry->lru_next;
    else cache->lru_head = entry->lru_next;

    if (entry->lru_next) entry->lru_next->lru_prev = entry->lru_prev;
    else cache->lru_tail = entry->lru_prev;

    entry->lru_prev = NULL;
    entry->lru_next = NULL;
}

/* Move entry to front (most recently used) of LRU list */
static void lru_touch(GV_Cache *cache, CacheEntry *entry) {
    lru_remove(cache, entry);

    entry->lru_next = cache->lru_head;
    entry->lru_prev = NULL;
    if (cache->lru_head) cache->lru_head->lru_prev = entry;
    cache->lru_head = entry;
    if (!cache->lru_tail) cache->lru_tail = entry;
}

/* Remove entry from hash bucket chain */
static void hash_remove(GV_Cache *cache, CacheEntry *entry) {
    size_t bi = bucket_idx(entry->key_hash);
    CacheEntry *prev = NULL;
    CacheEntry *cur = cache->buckets[bi];

    while (cur) {
        if (cur == entry) {
            if (prev) prev->hash_next = cur->hash_next;
            else cache->buckets[bi] = cur->hash_next;
            entry->hash_next = NULL;
            return;
        }
        prev = cur;
        cur = cur->hash_next;
    }
}

static void free_entry(CacheEntry *entry) {
    if (!entry) return;
    free(entry->query_data);
    free(entry->indices);
    free(entry->distances);
    free(entry);
}

/* Evict the least valuable entry based on policy */
static void evict_one(GV_Cache *cache) {
    CacheEntry *victim = NULL;

    if (cache->config.policy == GV_CACHE_LRU) {
        /* Evict from tail (least recently used) */
        victim = cache->lru_tail;
    } else {
        /* LFU: find entry with lowest access count */
        uint64_t min_access = UINT64_MAX;
        CacheEntry *cur = cache->lru_tail;
        while (cur) {
            if (cur->access_count < min_access) {
                min_access = cur->access_count;
                victim = cur;
            }
            cur = cur->lru_prev;
        }
    }

    if (!victim) return;

    lru_remove(cache, victim);
    hash_remove(cache, victim);

    cache->current_entries--;
    cache->current_memory -= victim->memory_size;
    cache->evictions++;

    free_entry(victim);
}

static int is_expired(const GV_Cache *cache, const CacheEntry *entry) {
    if (cache->config.ttl_seconds == 0) return 0;
    return (current_time_seconds() - entry->created_at) > cache->config.ttl_seconds;
}

/*  Configuration  */

static const GV_CacheConfig DEFAULT_CONFIG = {
    .max_entries = 1024,
    .max_memory_bytes = 64 * 1024 * 1024,  /* 64MB */
    .ttl_seconds = 60,
    .invalidate_after_mutations = 0,
    .policy = GV_CACHE_LRU
};

void gv_cache_config_init(GV_CacheConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/*  Lifecycle  */

GV_Cache *gv_cache_create(const GV_CacheConfig *config) {
    GV_Cache *cache = calloc(1, sizeof(GV_Cache));
    if (!cache) return NULL;

    cache->config = config ? *config : DEFAULT_CONFIG;

    if (pthread_mutex_init(&cache->lock, NULL) != 0) {
        free(cache);
        return NULL;
    }

    return cache;
}

void gv_cache_destroy(GV_Cache *cache) {
    if (!cache) return;

    /* Free all entries via hash table traversal */
    for (size_t i = 0; i < CACHE_BUCKETS; i++) {
        CacheEntry *cur = cache->buckets[i];
        while (cur) {
            CacheEntry *next = cur->hash_next;
            free_entry(cur);
            cur = next;
        }
    }

    pthread_mutex_destroy(&cache->lock);
    free(cache);
}

/*  Cache Operations  */

int gv_cache_lookup(GV_Cache *cache, const float *query_data, size_t dimension,
                    size_t k, int distance_type, GV_CachedResult *result) {
    if (!cache || !query_data || !result || dimension == 0 || k == 0) return -1;

    pthread_mutex_lock(&cache->lock);

    uint64_t hash = compute_cache_key(query_data, dimension, k, distance_type);
    size_t bi = bucket_idx(hash);

    CacheEntry *prev = NULL;
    CacheEntry *cur = cache->buckets[bi];

    while (cur) {
        if (cur->key_hash == hash && entries_match(cur, query_data, dimension, k, distance_type)) {
            /* Check TTL */
            if (is_expired(cache, cur)) {
                /* Remove expired entry */
                if (prev) prev->hash_next = cur->hash_next;
                else cache->buckets[bi] = cur->hash_next;

                lru_remove(cache, cur);
                cache->current_entries--;
                cache->current_memory -= cur->memory_size;
                cache->invalidations++;
                free_entry(cur);

                cache->misses++;
                pthread_mutex_unlock(&cache->lock);
                return 0;
            }

            /* Cache hit - copy results */
            result->count = cur->count;
            result->indices = malloc(cur->count * sizeof(size_t));
            result->distances = malloc(cur->count * sizeof(float));
            if (!result->indices || !result->distances) {
                free(result->indices);
                free(result->distances);
                result->indices = NULL;
                result->distances = NULL;
                result->count = 0;
                pthread_mutex_unlock(&cache->lock);
                return -1;
            }
            memcpy(result->indices, cur->indices, cur->count * sizeof(size_t));
            memcpy(result->distances, cur->distances, cur->count * sizeof(float));

            cur->last_access = current_time_seconds();
            cur->access_count++;
            lru_touch(cache, cur);

            cache->hits++;
            pthread_mutex_unlock(&cache->lock);
            return 1;
        }
        prev = cur;
        cur = cur->hash_next;
    }

    cache->misses++;
    pthread_mutex_unlock(&cache->lock);
    return 0;
}

int gv_cache_store(GV_Cache *cache, const float *query_data, size_t dimension,
                   size_t k, int distance_type,
                   const size_t *indices, const float *distances, size_t count) {
    if (!cache || !query_data || !indices || !distances || dimension == 0) return -1;

    size_t mem_needed = entry_memory_size(dimension, count);

    pthread_mutex_lock(&cache->lock);

    /* Evict entries until we have space */
    while (cache->current_entries >= cache->config.max_entries && cache->current_entries > 0) {
        evict_one(cache);
    }
    while (cache->current_memory + mem_needed > cache->config.max_memory_bytes &&
           cache->current_entries > 0) {
        evict_one(cache);
    }

    /* Allocate entry */
    CacheEntry *entry = calloc(1, sizeof(CacheEntry));
    if (!entry) {
        pthread_mutex_unlock(&cache->lock);
        return -1;
    }

    entry->key_hash = compute_cache_key(query_data, dimension, k, distance_type);
    entry->dimension = dimension;
    entry->k = k;
    entry->distance_type = distance_type;

    /* Copy query data */
    entry->query_data = malloc(dimension * sizeof(float));
    if (!entry->query_data) {
        free(entry);
        pthread_mutex_unlock(&cache->lock);
        return -1;
    }
    memcpy(entry->query_data, query_data, dimension * sizeof(float));

    /* Copy results */
    entry->count = count;
    entry->indices = malloc(count * sizeof(size_t));
    entry->distances = malloc(count * sizeof(float));
    if (!entry->indices || !entry->distances) {
        free(entry->query_data);
        free(entry->indices);
        free(entry->distances);
        free(entry);
        pthread_mutex_unlock(&cache->lock);
        return -1;
    }
    memcpy(entry->indices, indices, count * sizeof(size_t));
    memcpy(entry->distances, distances, count * sizeof(float));

    entry->created_at = current_time_seconds();
    entry->last_access = entry->created_at;
    entry->access_count = 1;
    entry->memory_size = mem_needed;

    /* Check for duplicate key - replace existing */
    size_t bi = bucket_idx(entry->key_hash);
    CacheEntry *cur = cache->buckets[bi];
    CacheEntry *prev = NULL;
    while (cur) {
        if (cur->key_hash == entry->key_hash &&
            entries_match(cur, query_data, dimension, k, distance_type)) {
            /* Replace existing */
            if (prev) prev->hash_next = cur->hash_next;
            else cache->buckets[bi] = cur->hash_next;

            lru_remove(cache, cur);
            cache->current_entries--;
            cache->current_memory -= cur->memory_size;
            free_entry(cur);
            break;
        }
        prev = cur;
        cur = cur->hash_next;
    }

    /* Insert into hash bucket (head) */
    entry->hash_next = cache->buckets[bi];
    cache->buckets[bi] = entry;

    /* Add to LRU front */
    entry->lru_prev = NULL;
    entry->lru_next = cache->lru_head;
    if (cache->lru_head) cache->lru_head->lru_prev = entry;
    cache->lru_head = entry;
    if (!cache->lru_tail) cache->lru_tail = entry;

    cache->current_entries++;
    cache->current_memory += mem_needed;

    pthread_mutex_unlock(&cache->lock);
    return 0;
}

void gv_cache_notify_mutation(GV_Cache *cache) {
    if (!cache) return;

    pthread_mutex_lock(&cache->lock);
    cache->mutation_count++;

    if (cache->config.invalidate_after_mutations > 0 &&
        cache->mutation_count >= cache->config.invalidate_after_mutations) {
        /* Flush entire cache */
        for (size_t i = 0; i < CACHE_BUCKETS; i++) {
            CacheEntry *cur = cache->buckets[i];
            while (cur) {
                CacheEntry *next = cur->hash_next;
                cache->invalidations++;
                free_entry(cur);
                cur = next;
            }
            cache->buckets[i] = NULL;
        }
        cache->lru_head = NULL;
        cache->lru_tail = NULL;
        cache->current_entries = 0;
        cache->current_memory = 0;
        cache->mutation_count = 0;
    }

    pthread_mutex_unlock(&cache->lock);
}

void gv_cache_invalidate_all(GV_Cache *cache) {
    if (!cache) return;

    pthread_mutex_lock(&cache->lock);

    for (size_t i = 0; i < CACHE_BUCKETS; i++) {
        CacheEntry *cur = cache->buckets[i];
        while (cur) {
            CacheEntry *next = cur->hash_next;
            cache->invalidations++;
            free_entry(cur);
            cur = next;
        }
        cache->buckets[i] = NULL;
    }

    cache->lru_head = NULL;
    cache->lru_tail = NULL;
    cache->current_entries = 0;
    cache->current_memory = 0;
    cache->mutation_count = 0;

    pthread_mutex_unlock(&cache->lock);
}

void gv_cache_free_result(GV_CachedResult *result) {
    if (!result) return;
    free(result->indices);
    free(result->distances);
    result->indices = NULL;
    result->distances = NULL;
    result->count = 0;
}

/*  Statistics  */

int gv_cache_get_stats(const GV_Cache *cache, GV_CacheStats *stats) {
    if (!cache || !stats) return -1;

    /* Cast away const for mutex lock */
    pthread_mutex_lock(&((GV_Cache *)cache)->lock);

    stats->hits = cache->hits;
    stats->misses = cache->misses;
    stats->evictions = cache->evictions;
    stats->invalidations = cache->invalidations;
    stats->current_entries = cache->current_entries;
    stats->current_memory = cache->current_memory;

    uint64_t total = cache->hits + cache->misses;
    stats->hit_rate = total > 0 ? (double)cache->hits / (double)total : 0.0;

    pthread_mutex_unlock(&((GV_Cache *)cache)->lock);
    return 0;
}

void gv_cache_reset_stats(GV_Cache *cache) {
    if (!cache) return;

    pthread_mutex_lock(&cache->lock);
    cache->hits = 0;
    cache->misses = 0;
    cache->evictions = 0;
    cache->invalidations = 0;
    pthread_mutex_unlock(&cache->lock);
}
