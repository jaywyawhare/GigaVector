#include "core/memory.h"

#include <stdlib.h>
#include <string.h>

#include "storage/database.h"

void *gv_alloc(size_t size) {
    if (size == 0) {
        return NULL;
    }
    return malloc(size);
}

void *gv_calloc(size_t nmemb, size_t size) {
    if (nmemb == 0 || size == 0) {
        return NULL;
    }
    if (nmemb > SIZE_MAX / size) {
        return NULL;
    }
    return calloc(nmemb, size);
}

void *gv_realloc(void *ptr, size_t size) {
    if (size == 0) {
        gv_free(ptr);
        return NULL;
    }
    return realloc(ptr, size);
}

void gv_free(void *ptr) {
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
