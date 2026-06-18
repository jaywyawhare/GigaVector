#include "core/scope.h"

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
    free(arena);
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

    arena = (GV_Arena *)malloc(sizeof(GV_Arena));
    if (arena == NULL) {
        return NULL;
    }
    if (gv_arena_init(arena, GV_TLS_ARENA_BYTES) != 0) {
        free(arena);
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
#else

GV_Arena *gv_tls_arena(void) {
    return NULL;
}

void gv_tls_arena_reset(void) {
}

#endif

void gv_bytes_release(GV_Bytes *bytes) {
    if (bytes == NULL) {
        return;
    }

    if (bytes->kind == GV_OWN_HEAP && bytes->data != NULL) {
        free(bytes->data);
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

    void *copy = malloc(len);
    if (copy == NULL) {
        return bytes;
    }
    memcpy(copy, data, len);

    bytes.data = copy;
    bytes.len = len;
    bytes.kind = GV_OWN_HEAP;
    return bytes;
}
