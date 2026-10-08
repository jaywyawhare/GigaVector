#ifndef GV_MEMORY_H
#define GV_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_MemoryEntry {
    void  *ptr;
    size_t size;
} GV_MemoryEntry;

typedef struct GV_Memory {
    GV_MemoryEntry *owned;
    size_t owned_count;
    size_t owned_cap;
    size_t pool_bytes;
    pthread_mutex_t lock; /**< Serializes owned[]/counts/pool_bytes mutations. */
} GV_Memory;

void gv_memory_init(GV_Memory *mem);
void gv_memory_fini(GV_Memory *mem);

/*
 * Test-only allocation-failure injection. Compiled in but disabled by default
 * (fail_after < 0), costing one predictable compare per allocation. Tests arm it
 * to force an OOM path without a real OOM.
 *   gv_alloc_set_fail_after(n): n < 0 disables; n >= 0 makes the n-th subsequent
 *     allocation (0 = the next one) return NULL, firing exactly once then disarming.
 *   gv_alloc_reset_fail(): disable injection and clear the internal counter.
 */
void gv_alloc_set_fail_after(long n);
void gv_alloc_reset_fail(void);

/*
 * Allocation accounting (the profiler's memory hook). Every gv_alloc/calloc/
 * realloc/free is counted at this single choke point. Live/peak bytes use the
 * allocator's real block size (malloc_usable_size) so they match RSS, not just
 * the requested size. Counting is compiled in only under -DGV_PROFILE_ALLOC;
 * without it these read back zero and the alloc path pays nothing. Thread-safe
 * (lock-free atomics), so a parallel HNSW build is attributed correctly.
 */
typedef struct GV_AllocStats {
    size_t   live_bytes;   /**< currently-allocated bytes (usable size) */
    size_t   peak_bytes;   /**< high-water mark of live_bytes */
    size_t   total_bytes;  /**< cumulative bytes ever allocated */
    uint64_t alloc_count;  /**< gv_alloc + gv_calloc calls that returned non-NULL */
    uint64_t free_count;   /**< gv_free calls on non-NULL */
    uint64_t realloc_count;/**< gv_realloc calls that returned non-NULL */
} GV_AllocStats;

void gv_alloc_stats(GV_AllocStats *out);
int  gv_alloc_stats_enabled(void); /**< 1 if built with -DGV_PROFILE_ALLOC */

/* Process heap - caller-owned and long-lived allocations without a DB context. */
void *gv_alloc(size_t size);
void *gv_calloc(size_t nmemb, size_t size);
void *gv_realloc(void *ptr, size_t size);
void  gv_free(void *ptr);
char *gv_strdup(const char *s);

struct GV_Database;

/* Route through gv_db_* when @p db is non-NULL, else process heap. */
void *gv_pool_alloc(struct GV_Database *db, size_t size);
void *gv_pool_calloc(struct GV_Database *db, size_t nmemb, size_t size);
void *gv_pool_realloc(struct GV_Database *db, void *ptr, size_t size);
void  gv_pool_free(struct GV_Database *db, void *ptr);

void *gv_db_alloc(struct GV_Database *db, size_t size);
void *gv_db_calloc(struct GV_Database *db, size_t nmemb, size_t size);
void *gv_db_realloc(struct GV_Database *db, void *ptr, size_t size);
void  gv_db_free(struct GV_Database *db, void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* GV_MEMORY_H */
