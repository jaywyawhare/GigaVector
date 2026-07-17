#ifndef GIGAVECTOR_GV_DISK_PAGE_CACHE_H
#define GIGAVECTOR_GV_DISK_PAGE_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_DiskPageCache GV_DiskPageCache;

typedef struct {
    size_t cache_hits;
    size_t cache_misses;
    size_t cached_entries;
    size_t used_bytes;
    size_t max_bytes;
} GV_DiskPageCacheStats;

GV_DiskPageCache *gv_disk_page_cache_create(size_t max_bytes);
void gv_disk_page_cache_destroy(GV_DiskPageCache *cache);

void gv_disk_page_cache_set_max_bytes(GV_DiskPageCache *cache, size_t max_bytes);
void gv_disk_page_cache_get_stats(const GV_DiskPageCache *cache, GV_DiskPageCacheStats *out);

/**
 * @brief Lookup cached bytes by key.
 *
 * Thread-safety / lifetime contract:
 *   The cache is fully internally locked, so lookup/insert/remove/etc. may be
 *   called concurrently from multiple threads (e.g. concurrent searches holding
 *   only the DB read lock).
 *
 *   On a hit, the cached bytes are COPIED, under the cache lock, into a
 *   per-thread scratch buffer that is OWNED by the cache library and is
 *   thread-local to the CALLING thread. The returned pointer therefore does NOT
 *   alias the internal cache entry and is immune to a concurrent eviction/free
 *   of that entry — this is what makes the returned pointer safe under
 *   concurrent eviction without changing this function's signature.
 *
 *   The returned pointer is valid on the calling thread until that same thread's
 *   NEXT call to gv_disk_page_cache_lookup() (which reuses the scratch buffer),
 *   or until the thread exits (the buffer is freed by a TLS destructor). The
 *   caller must NOT free it and must NOT retain it across another lookup on the
 *   same thread. This matches all in-tree callers, which consume the bytes (copy
 *   the pointer value, or parse the buffer) before issuing another lookup.
 *
 *   Returns NULL on miss (or on a rare scratch-buffer allocation failure).
 */
const uint8_t *gv_disk_page_cache_lookup(GV_DiskPageCache *cache, const char *key, size_t *len_out);

/**
 * @brief Insert or replace an entry (copies @p data).
 */
int gv_disk_page_cache_insert(GV_DiskPageCache *cache, const char *key,
                              const uint8_t *data, size_t len);

void gv_disk_page_cache_remove(GV_DiskPageCache *cache, const char *key);

#ifdef __cplusplus
}
#endif

#endif
