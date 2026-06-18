#include "core/memory.h"

#include <stdlib.h>
#include <string.h>

#include "storage/database.h"

void gv_memory_init(GV_Memory *mem) {
    if (mem == NULL) {
        return;
    }
    memset(mem, 0, sizeof(*mem));
}

void gv_memory_fini(GV_Memory *mem) {
    if (mem == NULL) {
        return;
    }
    for (size_t i = 0; i < mem->owned_count; ++i) {
        free(mem->owned[i].ptr);
    }
    free(mem->owned);
    memset(mem, 0, sizeof(*mem));
}

static int gv_memory_track(GV_Memory *mem, void *ptr, size_t size) {
    if (mem == NULL || ptr == NULL) {
        return -1;
    }
    if (mem->owned_count == mem->owned_cap) {
        size_t new_cap = mem->owned_cap == 0 ? 8 : mem->owned_cap * 2;
        GV_MemoryEntry *next =
            (GV_MemoryEntry *)realloc(mem->owned, new_cap * sizeof(GV_MemoryEntry));
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

static size_t gv_memory_untrack(GV_Memory *mem, void *ptr) {
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
    void *ptr = malloc(size);
    if (ptr == NULL) {
        return NULL;
    }
    if (gv_memory_track(&db->memory_pool, ptr, size) != 0) {
        free(ptr);
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
    gv_memory_untrack(&db->memory_pool, ptr);
    free(ptr);
}
