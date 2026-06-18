#ifndef GV_ARENA_H
#define GV_ARENA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GV_ARENA_OWN_HEAP  1u
#define GV_ARENA_STATIC    2u

typedef struct GV_Arena {
    uint8_t *base;
    size_t   capacity;
    size_t   offset;
    unsigned flags;
} GV_Arena;

int  gv_arena_init(GV_Arena *arena, size_t initial_capacity);
int  gv_arena_init_static(GV_Arena *arena, void *backing, size_t capacity);
void gv_arena_reset(GV_Arena *arena);
void gv_arena_fini(GV_Arena *arena);

void *gv_arena_alloc(GV_Arena *arena, size_t size, size_t alignment);
void *gv_arena_calloc(GV_Arena *arena, size_t nmemb, size_t size);
char *gv_arena_strdup(GV_Arena *arena, const char *s);

size_t gv_arena_used(const GV_Arena *arena);
size_t gv_arena_capacity(const GV_Arena *arena);

#ifdef __cplusplus
}
#endif

#endif /* GV_ARENA_H */
