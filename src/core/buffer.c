#include "core/buffer.h"

#include <string.h>

void gv_buf_init(GV_Buffer *buf, GV_Arena *arena) {
    if (buf == NULL) {
        return;
    }
    buf->data = NULL;
    buf->len = 0;
    buf->cap = 0;
    buf->arena = arena;
}

int gv_buf_reserve(GV_Buffer *buf, size_t need) {
    if (buf == NULL || buf->arena == NULL) {
        return -1;
    }
    if (need <= buf->cap) {
        return 0;
    }

    uint8_t *grown = (uint8_t *)gv_arena_alloc(buf->arena, need, 1);
    if (grown == NULL) {
        return -1;
    }

    if (buf->data != NULL && buf->len > 0) {
        memcpy(grown, buf->data, buf->len);
    }

    buf->data = grown;
    buf->cap = need;
    return 0;
}

int gv_buf_push(GV_Buffer *buf, const void *src, size_t n) {
    if (buf == NULL || buf->arena == NULL) {
        return -1;
    }
    if (n == 0) {
        return 0;
    }
    if (src == NULL) {
        return -1;
    }
    if (buf->len > SIZE_MAX - n) {
        return -1;
    }

    size_t need = buf->len + n;
    if (need > buf->cap) {
        size_t new_cap = buf->cap > 0 ? buf->cap : 64;
        while (new_cap < need) {
            if (new_cap > SIZE_MAX / 2) {
                return -1;
            }
            new_cap *= 2;
        }
        if (gv_buf_reserve(buf, new_cap) != 0) {
            return -1;
        }
    }

    memcpy(buf->data + buf->len, src, n);
    buf->len += n;
    return 0;
}

void gv_buf_clear(GV_Buffer *buf) {
    if (buf == NULL) {
        return;
    }
    buf->len = 0;
}
