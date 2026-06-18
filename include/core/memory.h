#ifndef GV_MEMORY_H
#define GV_MEMORY_H

#include <stddef.h>

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
} GV_Memory;

void gv_memory_init(GV_Memory *mem);
void gv_memory_fini(GV_Memory *mem);

struct GV_Database;

void *gv_db_alloc(struct GV_Database *db, size_t size);
void *gv_db_calloc(struct GV_Database *db, size_t nmemb, size_t size);
void  gv_db_free(struct GV_Database *db, void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* GV_MEMORY_H */
