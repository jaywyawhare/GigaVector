#include "core/scope.h"

#include <stdlib.h>
#include <string.h>

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
