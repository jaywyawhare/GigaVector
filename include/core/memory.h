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

/* Process heap — caller-owned and long-lived allocations without a DB context. */
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
