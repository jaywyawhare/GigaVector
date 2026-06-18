/**
 * @file disk_page_cache.c
 * @brief Shared LRU byte cache for on-disk index pages (DiskANN, posting segments).
 */

#include "storage/disk_page_cache.h"
#include "core/memory.h"

#include <stdlib.h>
#include <string.h>

#define GV_DISK_PAGE_CACHE_BUCKETS 64u

typedef struct DiskPageCacheNode {
    char *key;
    uint8_t *data;
    size_t len;
    struct DiskPageCacheNode *lru_prev;
    struct DiskPageCacheNode *lru_next;
    struct DiskPageCacheNode *hash_next;
    uint32_t hash;
} DiskPageCacheNode;

struct GV_DiskPageCache {
    DiskPageCacheNode *buckets[GV_DISK_PAGE_CACHE_BUCKETS];
    DiskPageCacheNode *lru_head;
    DiskPageCacheNode *lru_tail;
    size_t count;
    size_t used_bytes;
    size_t max_bytes;
    size_t hits;
    size_t misses;
};

static uint32_t disk_page_cache_hash(const char *key)
{
    uint32_t h = 2166136261u;
    for (const unsigned char *p = (const unsigned char *)key; *p; ++p) {
        h ^= *p;
        h *= 16777619u;
    }
    return h;
}

static void disk_page_cache_unlink(GV_DiskPageCache *cache, DiskPageCacheNode *node)
{
    if (node->lru_prev) node->lru_prev->lru_next = node->lru_next;
    else cache->lru_head = node->lru_next;
    if (node->lru_next) node->lru_next->lru_prev = node->lru_prev;
    else cache->lru_tail = node->lru_prev;
    node->lru_prev = node->lru_next = NULL;
}

static void disk_page_cache_touch(GV_DiskPageCache *cache, DiskPageCacheNode *node)
{
    disk_page_cache_unlink(cache, node);
    node->lru_prev = NULL;
    node->lru_next = cache->lru_head;
    if (cache->lru_head) cache->lru_head->lru_prev = node;
    cache->lru_head = node;
    if (!cache->lru_tail) cache->lru_tail = node;
}

static void disk_page_cache_node_free(DiskPageCacheNode *node)
{
    if (!node) return;
    gv_free(node->key);
    gv_free(node->data);
    gv_free(node);
}

static void disk_page_cache_evict(GV_DiskPageCache *cache)
{
    while (cache->lru_tail && cache->used_bytes > cache->max_bytes) {
        DiskPageCacheNode *victim = cache->lru_tail;
        disk_page_cache_unlink(cache, victim);
        uint32_t bucket = victim->hash % GV_DISK_PAGE_CACHE_BUCKETS;
        DiskPageCacheNode **pp = &cache->buckets[bucket];
        while (*pp && *pp != victim) pp = &(*pp)->hash_next;
        if (*pp == victim) *pp = victim->hash_next;
        cache->used_bytes -= victim->len;
        cache->count--;
        disk_page_cache_node_free(victim);
    }
}

GV_DiskPageCache *gv_disk_page_cache_create(size_t max_bytes)
{
    GV_DiskPageCache *cache = (GV_DiskPageCache *)gv_calloc(1, sizeof(GV_DiskPageCache));
    if (!cache) return NULL;
    cache->max_bytes = max_bytes;
    return cache;
}

void gv_disk_page_cache_destroy(GV_DiskPageCache *cache)
{
    if (!cache) return;
    for (size_t i = 0; i < GV_DISK_PAGE_CACHE_BUCKETS; ++i) {
        DiskPageCacheNode *node = cache->buckets[i];
        while (node) {
            DiskPageCacheNode *next = node->hash_next;
            disk_page_cache_node_free(node);
            node = next;
        }
    }
    gv_free(cache);
}

void gv_disk_page_cache_set_max_bytes(GV_DiskPageCache *cache, size_t max_bytes)
{
    if (!cache) return;
    cache->max_bytes = max_bytes;
    disk_page_cache_evict(cache);
}

void gv_disk_page_cache_get_stats(const GV_DiskPageCache *cache, GV_DiskPageCacheStats *out)
{
    if (!cache || !out) return;
    memset(out, 0, sizeof(*out));
    out->cache_hits = cache->hits;
    out->cache_misses = cache->misses;
    out->cached_entries = cache->count;
    out->used_bytes = cache->used_bytes;
    out->max_bytes = cache->max_bytes;
}

const uint8_t *gv_disk_page_cache_lookup(GV_DiskPageCache *cache, const char *key, size_t *len_out)
{
    if (!cache || !key) return NULL;
    uint32_t hash = disk_page_cache_hash(key);
    for (DiskPageCacheNode *node = cache->buckets[hash % GV_DISK_PAGE_CACHE_BUCKETS];
         node; node = node->hash_next) {
        if (node->hash == hash && strcmp(node->key, key) == 0) {
            cache->hits++;
            disk_page_cache_touch(cache, node);
            if (len_out) *len_out = node->len;
            return node->data;
        }
    }
    cache->misses++;
    return NULL;
}

int gv_disk_page_cache_insert(GV_DiskPageCache *cache, const char *key,
                              const uint8_t *data, size_t len)
{
    if (!cache || !key || !data || len == 0 || cache->max_bytes == 0) return -1;

    uint32_t hash = disk_page_cache_hash(key);
    uint32_t bucket = hash % GV_DISK_PAGE_CACHE_BUCKETS;
    for (DiskPageCacheNode *node = cache->buckets[bucket]; node; node = node->hash_next) {
        if (node->hash == hash && strcmp(node->key, key) == 0) {
            if (node->len != len) {
                uint8_t *tmp = (uint8_t *)gv_realloc(node->data, len);
                if (!tmp) return -1;
                cache->used_bytes -= node->len;
                cache->used_bytes += len;
                node->data = tmp;
                node->len = len;
            }
            memcpy(node->data, data, len);
            disk_page_cache_touch(cache, node);
            disk_page_cache_evict(cache);
            return 0;
        }
    }

    DiskPageCacheNode *node = (DiskPageCacheNode *)gv_calloc(1, sizeof(*node));
    if (!node) return -1;
    node->key = (char *)gv_alloc(strlen(key) + 1);
    node->data = (uint8_t *)gv_alloc(len);
    if (!node->key || !node->data) {
        disk_page_cache_node_free(node);
        return -1;
    }
    strcpy(node->key, key);
    memcpy(node->data, data, len);
    node->len = len;
    node->hash = hash;
    node->hash_next = cache->buckets[bucket];
    cache->buckets[bucket] = node;
    disk_page_cache_touch(cache, node);
    cache->count++;
    cache->used_bytes += len;
    disk_page_cache_evict(cache);
    return 0;
}

void gv_disk_page_cache_remove(GV_DiskPageCache *cache, const char *key)
{
    if (!cache || !key) return;
    uint32_t hash = disk_page_cache_hash(key);
    uint32_t bucket = hash % GV_DISK_PAGE_CACHE_BUCKETS;
    DiskPageCacheNode **pp = &cache->buckets[bucket];
    while (*pp) {
        DiskPageCacheNode *node = *pp;
        if (node->hash == hash && strcmp(node->key, key) == 0) {
            *pp = node->hash_next;
            disk_page_cache_unlink(cache, node);
            cache->used_bytes -= node->len;
            cache->count--;
            disk_page_cache_node_free(node);
            return;
        }
        pp = &node->hash_next;
    }
}
