#ifndef GV_BUFFER_H
#define GV_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "core/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Buffer {
    uint8_t  *data;
    size_t    len;
    size_t    cap;
    GV_Arena *arena;
} GV_Buffer;

void gv_buf_init(GV_Buffer *buf, GV_Arena *arena);
int  gv_buf_reserve(GV_Buffer *buf, size_t need);
int  gv_buf_push(GV_Buffer *buf, const void *src, size_t n);
void gv_buf_clear(GV_Buffer *buf);

#ifdef __cplusplus
}
#endif

#endif /* GV_BUFFER_H */
