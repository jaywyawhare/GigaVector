/**
 * @file disk_page_cache.c
 * @brief Shared LRU byte cache for on-disk index pages (DiskANN, posting segments).
 */

#include "storage/disk_page_cache.h"
#include "core/memory.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define GV_DISK_PAGE_CACHE_BUCKETS 64u

/*
 * Per-thread scratch buffer used by gv_disk_page_cache_lookup() to return a
 * private copy of a cached page. Because it is a copy, the returned pointer is
 * immune to a concurrent eviction/free of the underlying cache entry, and it
 * lets us keep the existing (borrowed-pointer) lookup signature that the
 * non-editable callers depend on. The buffer is thread-local and is freed by a
 * TLS destructor when the thread exits, so it does not leak. It is reused
 * (grown as needed) across lookups on the same thread, which is why the
 * returned pointer is only valid until the next lookup on that thread.
 */
typedef struct DiskPageLookupScratch {
    uint8_t *data;
    size_t cap;
} DiskPageLookupScratch;

static pthread_key_t g_scratch_key;
static pthread_once_t g_scratch_once = PTHREAD_ONCE_INIT;

static void disk_page_scratch_destroy(void *p)
{
    DiskPageLookupScratch *s = (DiskPageLookupScratch *)p;
    if (!s) return;
    gv_free(s->data);
    gv_free(s);
}

static void disk_page_scratch_key_create(void)
{
    (void)pthread_key_create(&g_scratch_key, disk_page_scratch_destroy);
}

/* Return this thread's scratch buffer sized to at least @p len, or NULL on OOM. */
static uint8_t *disk_page_scratch_get(size_t len)
{
    pthread_once(&g_scratch_once, disk_page_scratch_key_create);
    DiskPageLookupScratch *s = (DiskPageLookupScratch *)pthread_getspecific(g_scratch_key);
    if (!s) {
        s = (DiskPageLookupScratch *)gv_calloc(1, sizeof(*s));
        if (!s) return NULL;
        if (pthread_setspecific(g_scratch_key, s) != 0) {
            gv_free(s);
            return NULL;
        }
    }
    if (s->cap < len) {
        uint8_t *tmp = (uint8_t *)gv_realloc(s->data, len);
        if (!tmp) return NULL;
        s->data = tmp;
        s->cap = len;
    }
    return s->data;
}

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
    pthread_mutex_t lock; /**< Serializes all mutations of the fields above. */
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

/* Link a node that is NOT currently in the LRU list at the head (most-recent). */
static void disk_page_cache_push_front(GV_DiskPageCache *cache, DiskPageCacheNode *node)
{
    node->lru_prev = NULL;
    node->lru_next = cache->lru_head;
    if (cache->lru_head) cache->lru_head->lru_prev = node;
    cache->lru_head = node;
    if (!cache->lru_tail) cache->lru_tail = node;
}

/* Move an existing (already-linked) node to the head. Must not be called on a
 * node that is not in the list — unlink() would corrupt head/tail otherwise. */
static void disk_page_cache_touch(GV_DiskPageCache *cache, DiskPageCacheNode *node)
{
    disk_page_cache_unlink(cache, node);
    disk_page_cache_push_front(cache, node);
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
    if (pthread_mutex_init(&cache->lock, NULL) != 0) {
        gv_free(cache);
        return NULL;
    }
    cache->max_bytes = max_bytes;
    return cache;
}

void gv_disk_page_cache_destroy(GV_DiskPageCache *cache)
{
    if (!cache) return;
    pthread_mutex_destroy(&cache->lock);
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
    pthread_mutex_lock(&cache->lock);
    cache->max_bytes = max_bytes;
    disk_page_cache_evict(cache); /* lock-free helper; called with lock held */
    pthread_mutex_unlock(&cache->lock);
}

void gv_disk_page_cache_get_stats(const GV_DiskPageCache *cache, GV_DiskPageCacheStats *out)
{
    if (!cache || !out) return;
    /* The lock is a mutable implementation detail; take it even on the const
     * handle so stats are read as a consistent snapshot w.r.t. writers. */
    pthread_mutex_t *lock = (pthread_mutex_t *)&cache->lock;
    pthread_mutex_lock(lock);
    memset(out, 0, sizeof(*out));
    out->cache_hits = cache->hits;
    out->cache_misses = cache->misses;
    out->cached_entries = cache->count;
    out->used_bytes = cache->used_bytes;
    out->max_bytes = cache->max_bytes;
    pthread_mutex_unlock(lock);
}

const uint8_t *gv_disk_page_cache_lookup(GV_DiskPageCache *cache, const char *key, size_t *len_out)
{
    if (!cache || !key) return NULL;
    uint32_t hash = disk_page_cache_hash(key);
    pthread_mutex_lock(&cache->lock);
    for (DiskPageCacheNode *node = cache->buckets[hash % GV_DISK_PAGE_CACHE_BUCKETS];
         node; node = node->hash_next) {
        if (node->hash == hash && strcmp(node->key, key) == 0) {
            /* Copy the entry's bytes into this thread's private scratch buffer
             * WHILE HOLDING THE LOCK. The returned pointer aliases the scratch
             * copy, not node->data, so it stays valid even if another thread
             * evicts and frees this node immediately after we unlock. */
            uint8_t *scratch = disk_page_scratch_get(node->len);
            if (!scratch) {
                /* Out of memory for the copy: treat as a miss rather than
                 * returning a borrowed (unsafe) pointer. */
                cache->misses++;
                pthread_mutex_unlock(&cache->lock);
                return NULL;
            }
            memcpy(scratch, node->data, node->len);
            size_t len = node->len;
            cache->hits++;
            disk_page_cache_touch(cache, node);
            pthread_mutex_unlock(&cache->lock);
            if (len_out) *len_out = len;
            return scratch;
        }
    }
    cache->misses++;
    pthread_mutex_unlock(&cache->lock);
    return NULL;
}

int gv_disk_page_cache_insert(GV_DiskPageCache *cache, const char *key,
                              const uint8_t *data, size_t len)
{
    if (!cache || !key || !data || len == 0 || cache->max_bytes == 0) return -1;

    uint32_t hash = disk_page_cache_hash(key);
    uint32_t bucket = hash % GV_DISK_PAGE_CACHE_BUCKETS;
    pthread_mutex_lock(&cache->lock);
    for (DiskPageCacheNode *node = cache->buckets[bucket]; node; node = node->hash_next) {
        if (node->hash == hash && strcmp(node->key, key) == 0) {
            if (node->len != len) {
                uint8_t *tmp = (uint8_t *)gv_realloc(node->data, len);
                if (!tmp) {
                    pthread_mutex_unlock(&cache->lock);
                    return -1;
                }
                cache->used_bytes -= node->len;
                cache->used_bytes += len;
                node->data = tmp;
                node->len = len;
            }
            memcpy(node->data, data, len);
            disk_page_cache_touch(cache, node);
            disk_page_cache_evict(cache);
            pthread_mutex_unlock(&cache->lock);
            return 0;
        }
    }

    DiskPageCacheNode *node = (DiskPageCacheNode *)gv_calloc(1, sizeof(*node));
    if (!node) {
        pthread_mutex_unlock(&cache->lock);
        return -1;
    }
    node->key = (char *)gv_alloc(strlen(key) + 1);
    node->data = (uint8_t *)gv_alloc(len);
    if (!node->key || !node->data) {
        disk_page_cache_node_free(node);
        pthread_mutex_unlock(&cache->lock);
        return -1;
    }
    memcpy(node->key, key, strlen(key) + 1);
    memcpy(node->data, data, len);
    node->len = len;
    node->hash = hash;
    node->hash_next = cache->buckets[bucket];
    cache->buckets[bucket] = node;
    disk_page_cache_push_front(cache, node);  /* new node: link without unlink */
    cache->count++;
    cache->used_bytes += len;
    disk_page_cache_evict(cache);
    pthread_mutex_unlock(&cache->lock);
    return 0;
}

void gv_disk_page_cache_remove(GV_DiskPageCache *cache, const char *key)
{
    if (!cache || !key) return;
    uint32_t hash = disk_page_cache_hash(key);
    uint32_t bucket = hash % GV_DISK_PAGE_CACHE_BUCKETS;
    pthread_mutex_lock(&cache->lock);
    DiskPageCacheNode **pp = &cache->buckets[bucket];
    while (*pp) {
        DiskPageCacheNode *node = *pp;
        if (node->hash == hash && strcmp(node->key, key) == 0) {
            *pp = node->hash_next;
            disk_page_cache_unlink(cache, node);
            cache->used_bytes -= node->len;
            cache->count--;
            disk_page_cache_node_free(node);
            pthread_mutex_unlock(&cache->lock);
            return;
        }
        pp = &node->hash_next;
    }
    pthread_mutex_unlock(&cache->lock);
}
