#include "core/memory.h"

#include <stdlib.h>
#include <string.h>

#include "storage/database.h"

/* ---- Allocation accounting (see include/core/memory.h) ------------------- */
#ifdef GV_PROFILE_ALLOC
#include <malloc.h> /* malloc_usable_size (glibc) */

static size_t   s_live_bytes;
static size_t   s_peak_bytes;
static size_t   s_total_bytes;
static uint64_t s_alloc_count;
static uint64_t s_free_count;
static uint64_t s_realloc_count;

/* usable block size, or 0 for NULL — the real cost the allocator reserved. */
static inline size_t gv_block_size(void *p) {
    return p ? malloc_usable_size(p) : 0;
}

static inline void gv_acct_add(size_t bytes) {
    __atomic_add_fetch(&s_total_bytes, bytes, __ATOMIC_RELAXED);
    size_t live = __atomic_add_fetch(&s_live_bytes, bytes, __ATOMIC_RELAXED);
    size_t peak = __atomic_load_n(&s_peak_bytes, __ATOMIC_RELAXED);
    while (live > peak &&
           !__atomic_compare_exchange_n(&s_peak_bytes, &peak, live, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
        /* peak reloaded into `peak`; retry until we win or someone set higher */
    }
}
static inline void gv_acct_sub(size_t bytes) {
    __atomic_sub_fetch(&s_live_bytes, bytes, __ATOMIC_RELAXED);
}

void gv_alloc_stats(GV_AllocStats *out) {
    if (!out) return;
    out->live_bytes    = __atomic_load_n(&s_live_bytes, __ATOMIC_RELAXED);
    out->peak_bytes    = __atomic_load_n(&s_peak_bytes, __ATOMIC_RELAXED);
    out->total_bytes   = __atomic_load_n(&s_total_bytes, __ATOMIC_RELAXED);
    out->alloc_count   = __atomic_load_n(&s_alloc_count, __ATOMIC_RELAXED);
    out->free_count    = __atomic_load_n(&s_free_count, __ATOMIC_RELAXED);
    out->realloc_count = __atomic_load_n(&s_realloc_count, __ATOMIC_RELAXED);
}
int gv_alloc_stats_enabled(void) { return 1; }
#else
void gv_alloc_stats(GV_AllocStats *out) {
    if (out) memset(out, 0, sizeof(*out));
}
int gv_alloc_stats_enabled(void) { return 0; }
#endif /* GV_PROFILE_ALLOC */

/* Test-only allocation-failure injection (see include/core/memory.h contract).
 * Disabled by default (fail_after < 0), so normal builds pay one predictable
 * compare per allocation. fail_after = index of the allocation to fail (0 = the
 * next one), < 0 when disarmed. Not thread-synchronised - tests drive it single-
 * threaded. */
long gv_alloc_fail_after = -1;
long gv_alloc_fail_count = 0;

void gv_alloc_set_fail_after(long n) {
    gv_alloc_fail_after = n;
    gv_alloc_fail_count = 0;
}

void gv_alloc_reset_fail(void) {
    gv_alloc_fail_after = -1;
    gv_alloc_fail_count = 0;
}

/* Returns 1 if this allocation should be failed (and disarms injection). */
static int gv_alloc_should_fail(void) {
    if (gv_alloc_fail_after < 0) {
        return 0; /* disabled: single predictable branch, zero behaviour change */
    }
    long idx = gv_alloc_fail_count++;
    if (idx == gv_alloc_fail_after) {
        gv_alloc_fail_after = -1; /* fire exactly once, then disarm */
        return 1;
    }
    return 0;
}

void *gv_alloc(size_t size) {
    if (size == 0) {
        return NULL;
    }
    if (gv_alloc_should_fail()) {
        return NULL;
    }
    void *p = malloc(size);
#ifdef GV_PROFILE_ALLOC
    if (p) {
        __atomic_add_fetch(&s_alloc_count, 1, __ATOMIC_RELAXED);
        gv_acct_add(gv_block_size(p));
    }
#endif
    return p;
}

void *gv_calloc(size_t nmemb, size_t size) {
    if (nmemb == 0 || size == 0) {
        return NULL;
    }
    if (nmemb > SIZE_MAX / size) {
        return NULL;
    }
    if (gv_alloc_should_fail()) {
        return NULL;
    }
    void *p = calloc(nmemb, size);
#ifdef GV_PROFILE_ALLOC
    if (p) {
        __atomic_add_fetch(&s_alloc_count, 1, __ATOMIC_RELAXED);
        gv_acct_add(gv_block_size(p));
    }
#endif
    return p;
}

void *gv_realloc(void *ptr, size_t size) {
    if (size == 0) {
        gv_free(ptr);
        return NULL;
    }
    if (gv_alloc_should_fail()) {
        return NULL;
    }
#ifdef GV_PROFILE_ALLOC
    size_t old_sz = gv_block_size(ptr);
    void *p = realloc(ptr, size);
    if (p) {
        __atomic_add_fetch(&s_realloc_count, 1, __ATOMIC_RELAXED);
        size_t new_sz = gv_block_size(p);
        if (new_sz >= old_sz) gv_acct_add(new_sz - old_sz);
        else                  gv_acct_sub(old_sz - new_sz);
    }
    return p;
#else
    return realloc(ptr, size);
#endif
}

void gv_free(void *ptr) {
#ifdef GV_PROFILE_ALLOC
    if (ptr) {
        __atomic_add_fetch(&s_free_count, 1, __ATOMIC_RELAXED);
        gv_acct_sub(gv_block_size(ptr));
    }
#endif
    free(ptr);
}

char *gv_strdup(const char *s) {
    if (s == NULL) {
        return NULL;
    }
    size_t len = strlen(s) + 1;
    char *copy = (char *)gv_alloc(len);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, s, len);
    return copy;
}

void *gv_pool_alloc(GV_Database *db, size_t size) {
    if (db != NULL) {
        return gv_db_alloc(db, size);
    }
    return gv_alloc(size);
}

void *gv_pool_calloc(GV_Database *db, size_t nmemb, size_t size) {
    if (db != NULL) {
        return gv_db_calloc(db, nmemb, size);
    }
    return gv_calloc(nmemb, size);
}

void *gv_pool_realloc(GV_Database *db, void *ptr, size_t size) {
    if (db != NULL) {
        return gv_db_realloc(db, ptr, size);
    }
    return gv_realloc(ptr, size);
}

void gv_pool_free(GV_Database *db, void *ptr) {
    if (ptr == NULL) {
        return;
    }
    if (db != NULL) {
        gv_db_free(db, ptr);
        return;
    }
    gv_free(ptr);
}

void gv_memory_init(GV_Memory *mem) {
    if (mem == NULL) {
        return;
    }
    memset(mem, 0, sizeof(*mem));
    pthread_mutex_init(&mem->lock, NULL);
}

void gv_memory_fini(GV_Memory *mem) {
    if (mem == NULL) {
        return;
    }
    for (size_t i = 0; i < mem->owned_count; ++i) {
        gv_free(mem->owned[i].ptr);
    }
    gv_free(mem->owned);
    /* Destroy the lock AFTER freeing tracked allocations, then zero the pool. */
    pthread_mutex_destroy(&mem->lock);
    memset(mem, 0, sizeof(*mem));
}

/* Lock-free core: caller MUST hold mem->lock. */
static int gv_memory_track_locked(GV_Memory *mem, void *ptr, size_t size) {
    if (mem == NULL || ptr == NULL) {
        return -1;
    }
    if (mem->owned_count == mem->owned_cap) {
        size_t new_cap = mem->owned_cap == 0 ? 8 : mem->owned_cap * 2;
        GV_MemoryEntry *next =
            (GV_MemoryEntry *)gv_realloc(mem->owned, new_cap * sizeof(GV_MemoryEntry));
        if (next == NULL) {
            return -1;
        }
        mem->owned = next;
        mem->owned_cap = new_cap;
    }
    mem->owned[mem->owned_count].ptr = ptr;
    mem->owned[mem->owned_count].size = size;
    mem->owned_count++;
    mem->pool_bytes += size;
    return 0;
}

/* Lock-free core: caller MUST hold mem->lock. */
static size_t gv_memory_untrack_locked(GV_Memory *mem, void *ptr) {
    if (mem == NULL || ptr == NULL) {
        return 0;
    }
    for (size_t i = 0; i < mem->owned_count; ++i) {
        if (mem->owned[i].ptr != ptr) {
            continue;
        }
        size_t size = mem->owned[i].size;
        mem->owned[i] = mem->owned[mem->owned_count - 1];
        mem->owned_count--;
        if (mem->pool_bytes >= size) {
            mem->pool_bytes -= size;
        } else {
            mem->pool_bytes = 0;
        }
        return size;
    }
    return 0;
}

void *gv_db_alloc(GV_Database *db, size_t size) {
    if (db == NULL || size == 0) {
        return NULL;
    }
    if (db_check_resource_limits(db, 0, size) != 0) {
        return NULL;
    }
    void *ptr = gv_alloc(size);
    if (ptr == NULL) {
        return NULL;
    }
    pthread_mutex_lock(&db->memory_pool.lock);
    int rc = gv_memory_track_locked(&db->memory_pool, ptr, size);
    pthread_mutex_unlock(&db->memory_pool.lock);
    if (rc != 0) {
        gv_free(ptr);
        return NULL;
    }
    return ptr;
}

void *gv_db_calloc(GV_Database *db, size_t nmemb, size_t size) {
    if (db == NULL || nmemb == 0 || size == 0) {
        return NULL;
    }
    if (nmemb > SIZE_MAX / size) {
        return NULL;
    }
    size_t total = nmemb * size;
    void *ptr = gv_db_alloc(db, total);
    if (ptr == NULL) {
        return NULL;
    }
    memset(ptr, 0, total);
    return ptr;
}

void gv_db_free(GV_Database *db, void *ptr) {
    if (db == NULL || ptr == NULL) {
        return;
    }
    pthread_mutex_lock(&db->memory_pool.lock);
    gv_memory_untrack_locked(&db->memory_pool, ptr);
    gv_free(ptr);
    pthread_mutex_unlock(&db->memory_pool.lock);
}

void *gv_db_realloc(GV_Database *db, void *ptr, size_t size) {
    if (db == NULL || size == 0) {
        return NULL;
    }
    /* Hold the pool lock across untrack -> realloc -> retrack so owned[]/counts/
     * pool_bytes stay consistent. db_check_resource_limits acquires only
     * resource_mutex and always releases it before returning, so nesting it
     * here is safe (no path takes resource_mutex then memory_pool.lock). */
    pthread_mutex_lock(&db->memory_pool.lock);
    size_t old_size = 0;
    if (ptr != NULL) {
        old_size = gv_memory_untrack_locked(&db->memory_pool, ptr);
    }
    size_t delta = size > old_size ? size - old_size : 0;
    if (delta > 0 && db_check_resource_limits(db, 0, delta) != 0) {
        if (ptr != NULL) {
            (void)gv_memory_track_locked(&db->memory_pool, ptr, old_size);
        }
        pthread_mutex_unlock(&db->memory_pool.lock);
        return NULL;
    }
    void *next = gv_realloc(ptr, size);
    if (next == NULL) {
        if (ptr != NULL) {
            (void)gv_memory_track_locked(&db->memory_pool, ptr, old_size);
        }
        pthread_mutex_unlock(&db->memory_pool.lock);
        return NULL;
    }
    if (gv_memory_track_locked(&db->memory_pool, next, size) != 0) {
        gv_free(next);
        pthread_mutex_unlock(&db->memory_pool.lock);
        return NULL;
    }
    pthread_mutex_unlock(&db->memory_pool.lock);
    return next;
}
