#include "core/arena.h"

#include <stdlib.h>
#include <string.h>

#ifndef GV_ARENA_DEFAULT_ALIGN
#define GV_ARENA_DEFAULT_ALIGN 16u
#endif

static size_t gv_align_up(size_t value, size_t alignment) {
    if (alignment == 0) {
        alignment = 1;
    }
    size_t mask = alignment - 1;
    if ((alignment & mask) != 0) {
        return 0;
    }
    if (value > SIZE_MAX - mask) {
        return 0;
    }
    return (value + mask) & ~mask;
}

static int gv_arena_grow(GV_Arena *arena, size_t required) {
    if ((arena->flags & GV_ARENA_STATIC) != 0) {
        return -1;
    }

    size_t new_cap = arena->capacity > 0 ? arena->capacity : 4096;
    while (new_cap < required) {
        if (new_cap > SIZE_MAX / 2) {
            return -1;
        }
        new_cap *= 2;
    }

    uint8_t *grown = (uint8_t *)realloc(arena->base, new_cap);
    if (grown == NULL) {
        return -1;
    }

    arena->base = grown;
    arena->capacity = new_cap;
    arena->flags |= GV_ARENA_OWN_HEAP;
    return 0;
}

int gv_arena_init(GV_Arena *arena, size_t initial_capacity) {
    if (arena == NULL) {
        return -1;
    }

    arena->base = NULL;
    arena->capacity = 0;
    arena->offset = 0;
    arena->flags = 0;

    if (initial_capacity == 0) {
        initial_capacity = 4096;
    }

    arena->base = (uint8_t *)malloc(initial_capacity);
    if (arena->base == NULL) {
        return -1;
    }

    arena->capacity = initial_capacity;
    arena->flags = GV_ARENA_OWN_HEAP;
    return 0;
}

int gv_arena_init_static(GV_Arena *arena, void *backing, size_t capacity) {
    if (arena == NULL || backing == NULL || capacity == 0) {
        return -1;
    }

    arena->base = (uint8_t *)backing;
    arena->capacity = capacity;
    arena->offset = 0;
    arena->flags = GV_ARENA_STATIC;
    return 0;
}

void gv_arena_reset(GV_Arena *arena) {
    if (arena == NULL) {
        return;
    }
    arena->offset = 0;
}

void gv_arena_fini(GV_Arena *arena) {
    if (arena == NULL) {
        return;
    }

    if ((arena->flags & GV_ARENA_OWN_HEAP) != 0 && arena->base != NULL) {
        free(arena->base);
    }

    arena->base = NULL;
    arena->capacity = 0;
    arena->offset = 0;
    arena->flags = 0;
}

void *gv_arena_alloc(GV_Arena *arena, size_t size, size_t alignment) {
    if (arena == NULL || size == 0 || arena->base == NULL) {
        return NULL;
    }

    if (alignment == 0) {
        alignment = GV_ARENA_DEFAULT_ALIGN;
    }
    if ((alignment & (alignment - 1)) != 0) {
        return NULL;
    }

    size_t start = gv_align_up(arena->offset, alignment);
    if (start == 0 && arena->offset != 0) {
        return NULL;
    }
    if (start > SIZE_MAX - size) {
        return NULL;
    }

    size_t end = start + size;
    if (end > arena->capacity) {
        if (gv_arena_grow(arena, end) != 0) {
            return NULL;
        }
    }

    void *ptr = arena->base + start;
    arena->offset = end;
    return ptr;
}

void *gv_arena_calloc(GV_Arena *arena, size_t nmemb, size_t size) {
    if (nmemb != 0 && size > SIZE_MAX / nmemb) {
        return NULL;
    }

    size_t total = nmemb * size;
    void *ptr = gv_arena_alloc(arena, total, GV_ARENA_DEFAULT_ALIGN);
    if (ptr != NULL && total > 0) {
        memset(ptr, 0, total);
    }
    return ptr;
}

char *gv_arena_strdup(GV_Arena *arena, const char *s) {
    if (s == NULL) {
        return NULL;
    }

    size_t len = strlen(s) + 1;
    char *copy = (char *)gv_arena_alloc(arena, len, 1);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, s, len);
    return copy;
}

size_t gv_arena_used(const GV_Arena *arena) {
    return arena != NULL ? arena->offset : 0;
}

size_t gv_arena_capacity(const GV_Arena *arena) {
    return arena != NULL ? arena->capacity : 0;
}
