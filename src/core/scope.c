#include "core/scope.h"
#include "core/memory.h"

#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <pthread.h>

static pthread_key_t gv_tls_arena_key;
static pthread_once_t gv_tls_arena_once = PTHREAD_ONCE_INIT;

static void gv_tls_arena_key_destroy(void *value) {
    GV_Arena *arena = (GV_Arena *)value;
    if (arena == NULL) {
        return;
    }
    gv_arena_fini(arena);
    gv_free(arena);
}

static void gv_tls_arena_key_create(void) {
    (void)pthread_key_create(&gv_tls_arena_key, gv_tls_arena_key_destroy);
}

GV_Arena *gv_tls_arena(void) {
    pthread_once(&gv_tls_arena_once, gv_tls_arena_key_create);
    GV_Arena *arena = (GV_Arena *)pthread_getspecific(gv_tls_arena_key);
    if (arena != NULL) {
        return arena;
    }

    arena = (GV_Arena *)gv_alloc(sizeof(GV_Arena));
    if (arena == NULL) {
        return NULL;
    }
    if (gv_arena_init(arena, GV_TLS_ARENA_BYTES) != 0) {
        gv_free(arena);
        return NULL;
    }
    pthread_setspecific(gv_tls_arena_key, arena);
    return arena;
}

void gv_tls_arena_reset(void) {
    pthread_once(&gv_tls_arena_once, gv_tls_arena_key_create);
    GV_Arena *arena = (GV_Arena *)pthread_getspecific(gv_tls_arena_key);
    if (arena != NULL) {
        gv_arena_reset(arena);
    }
}

void *gv_tls_alloc(size_t size, size_t alignment) {
    GV_Arena *arena = gv_tls_arena();
    if (arena == NULL) {
        return NULL;
    }
    return gv_arena_alloc(arena, size, alignment);
}

void *gv_tls_calloc(size_t nmemb, size_t size) {
    GV_Arena *arena = gv_tls_arena();
    if (arena == NULL) {
        return NULL;
    }
    return gv_arena_calloc(arena, nmemb, size);
}

void *gv_tls_alloc_or_heap(size_t size, size_t alignment, int *on_heap) {
    void *ptr = gv_tls_alloc(size, alignment);
    if (ptr != NULL) {
        if (on_heap != NULL) {
            *on_heap = 0;
        }
        return ptr;
    }
    ptr = gv_alloc(size);
    if (on_heap != NULL) {
        *on_heap = ptr != NULL ? 1 : 0;
    }
    return ptr;
}

void *gv_tls_calloc_or_heap(size_t nmemb, size_t size, int *on_heap) {
    void *ptr = gv_tls_calloc(nmemb, size);
    if (ptr != NULL) {
        if (on_heap != NULL) {
            *on_heap = 0;
        }
        return ptr;
    }
    ptr = gv_calloc(nmemb, size);
    if (on_heap != NULL) {
        *on_heap = ptr != NULL ? 1 : 0;
    }
    return ptr;
}

void gv_tls_free_or_heap(void *ptr, int on_heap) {
    if (on_heap && ptr != NULL) {
        gv_free(ptr);
    }
}
#else
#include <windows.h>

static DWORD gv_tls_arena_fls = FLS_OUT_OF_INDEXES;
static INIT_ONCE gv_tls_arena_once = INIT_ONCE_STATIC_INIT;

static VOID WINAPI gv_tls_arena_fls_destroy(PVOID value) {
    GV_Arena *arena = (GV_Arena *)value;
    if (arena == NULL) {
        return;
    }
    gv_arena_fini(arena);
    gv_free(arena);
}

static BOOL CALLBACK gv_tls_arena_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once;
    (void)param;
    (void)ctx;
    gv_tls_arena_fls = FlsAlloc(gv_tls_arena_fls_destroy);
    return gv_tls_arena_fls != FLS_OUT_OF_INDEXES;
}

GV_Arena *gv_tls_arena(void) {
    InitOnceExecuteOnce(&gv_tls_arena_once, gv_tls_arena_init_once, NULL, NULL);
    if (gv_tls_arena_fls == FLS_OUT_OF_INDEXES) {
        return NULL;
    }
    GV_Arena *arena = (GV_Arena *)FlsGetValue(gv_tls_arena_fls);
    if (arena != NULL) {
        return arena;
    }
    arena = (GV_Arena *)gv_alloc(sizeof(GV_Arena));
    if (arena == NULL) {
        return NULL;
    }
    if (gv_arena_init(arena, GV_TLS_ARENA_BYTES) != 0) {
        gv_free(arena);
        return NULL;
    }
    FlsSetValue(gv_tls_arena_fls, arena);
    return arena;
}

void gv_tls_arena_reset(void) {
    InitOnceExecuteOnce(&gv_tls_arena_once, gv_tls_arena_init_once, NULL, NULL);
    if (gv_tls_arena_fls == FLS_OUT_OF_INDEXES) {
        return;
    }
    GV_Arena *arena = (GV_Arena *)FlsGetValue(gv_tls_arena_fls);
    if (arena != NULL) {
        gv_arena_reset(arena);
    }
}

void *gv_tls_alloc(size_t size, size_t alignment) {
    GV_Arena *arena = gv_tls_arena();
    if (arena == NULL) {
        return NULL;
    }
    return gv_arena_alloc(arena, size, alignment);
}

void *gv_tls_calloc(size_t nmemb, size_t size) {
    GV_Arena *arena = gv_tls_arena();
    if (arena == NULL) {
        return NULL;
    }
    return gv_arena_calloc(arena, nmemb, size);
}

void *gv_tls_alloc_or_heap(size_t size, size_t alignment, int *on_heap) {
    void *ptr = gv_tls_alloc(size, alignment);
    if (ptr != NULL) {
        if (on_heap != NULL) {
            *on_heap = 0;
        }
        return ptr;
    }
    ptr = gv_alloc(size);
    if (on_heap != NULL) {
        *on_heap = ptr != NULL ? 1 : 0;
    }
    return ptr;
}

void *gv_tls_calloc_or_heap(size_t nmemb, size_t size, int *on_heap) {
    void *ptr = gv_tls_calloc(nmemb, size);
    if (ptr != NULL) {
        if (on_heap != NULL) {
            *on_heap = 0;
        }
        return ptr;
    }
    ptr = gv_calloc(nmemb, size);
    if (on_heap != NULL) {
        *on_heap = ptr != NULL ? 1 : 0;
    }
    return ptr;
}

void gv_tls_free_or_heap(void *ptr, int on_heap) {
    if (on_heap && ptr != NULL) {
        gv_free(ptr);
    }
}

#endif

void gv_bytes_release(GV_Bytes *bytes) {
    if (bytes == NULL) {
        return;
    }

    if (bytes->kind == GV_OWN_HEAP && bytes->data != NULL) {
        gv_free(bytes->data);
    }

    bytes->data = NULL;
    bytes->len = 0;
    bytes->kind = GV_OWN_BORROW;
    bytes->arena = NULL;
}

GV_Bytes gv_bytes_from_arena(GV_Arena *arena, void *data, size_t len) {
    GV_Bytes bytes;
    bytes.data = data;
    bytes.len = len;
    bytes.kind = GV_OWN_ARENA;
    bytes.arena = arena;
    return bytes;
}

GV_Bytes gv_bytes_copy_heap(const void *data, size_t len) {
    GV_Bytes bytes;
    bytes.data = NULL;
    bytes.len = 0;
    bytes.kind = GV_OWN_BORROW;
    bytes.arena = NULL;

    if (len == 0) {
        bytes.kind = GV_OWN_HEAP;
        return bytes;
    }
    if (data == NULL) {
        return bytes;
    }

    void *copy = gv_alloc(len);
    if (copy == NULL) {
        return bytes;
    }
    memcpy(copy, data, len);

    bytes.data = copy;
    bytes.len = len;
    bytes.kind = GV_OWN_HEAP;
    return bytes;
}
