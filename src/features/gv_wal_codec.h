/**
 * @file gv_wal_codec.h
 * @brief Internal write-ahead-log record codec shared by the graph layers.
 *
 * Record framing: [u8 op][u32 payload_len][payload], all little-endian.
 * Strings are u32 length + raw bytes. The encoder grows its buffer on demand,
 * so records carrying large payloads (e.g. entity embeddings) are fine.
 *
 * Not part of the public API.
 */
#ifndef GIGAVECTOR_FEATURES_GV_WAL_CODEC_H
#define GIGAVECTOR_FEATURES_GV_WAL_CODEC_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include "core/memory.h"

typedef struct {
    uint8_t *buf;
    size_t   len;
    size_t   cap;
    int      overflow;
} GV_WalBuf;

static inline void gv_wal_init(GV_WalBuf *b)
{
    b->buf = NULL;
    b->len = 0;
    b->cap = 0;
    b->overflow = 0;
}

static inline void gv_wal_free(GV_WalBuf *b)
{
    gv_free(b->buf);
    b->buf = NULL;
    b->len = 0;
    b->cap = 0;
}

/* Returns 0 and flags the record unloggable if allocation fails. */
static inline int gv_wal_reserve(GV_WalBuf *b, size_t n)
{
    if (b->overflow) return 0;
    if (b->len + n <= b->cap) return 1;
    size_t nc = b->cap ? b->cap : 256;
    while (nc < b->len + n) nc *= 2;
    uint8_t *nb = (uint8_t *)gv_realloc(b->buf, nc);
    if (!nb) {
        b->overflow = 1;
        return 0;
    }
    b->buf = nb;
    b->cap = nc;
    return 1;
}

static inline void gv_wal_put_u8(GV_WalBuf *b, uint8_t v)
{
    if (!gv_wal_reserve(b, 1)) return;
    b->buf[b->len++] = v;
}

static inline void gv_wal_put_u32(GV_WalBuf *b, uint32_t v)
{
    if (!gv_wal_reserve(b, 4)) return;
    for (int i = 0; i < 4; i++) b->buf[b->len++] = (uint8_t)(v >> (8 * i));
}

static inline void gv_wal_put_u64(GV_WalBuf *b, uint64_t v)
{
    if (!gv_wal_reserve(b, 8)) return;
    for (int i = 0; i < 8; i++) b->buf[b->len++] = (uint8_t)(v >> (8 * i));
}

static inline void gv_wal_put_f32(GV_WalBuf *b, float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    gv_wal_put_u32(b, bits);
}

static inline void gv_wal_put_str(GV_WalBuf *b, const char *s)
{
    size_t n = s ? strlen(s) : 0;
    if (!gv_wal_reserve(b, 4)) return;
    if (n > UINT32_MAX || !gv_wal_reserve(b, n)) return;
    gv_wal_put_u32(b, (uint32_t)n);
    if (n) {
        memcpy(b->buf + b->len, s, n);
        b->len += n;
    }
}

static inline void gv_wal_put_bytes(GV_WalBuf *b, const void *p, size_t n)
{
    if (!gv_wal_reserve(b, n)) return;
    if (n) {
        memcpy(b->buf + b->len, p, n);
        b->len += n;
    }
}

static inline const uint8_t *gv_wal_get_u8(const uint8_t *p, const uint8_t *end,
                                           uint8_t *v)
{
    if (end - p < 1) return NULL;
    *v = *p;
    return p + 1;
}

static inline const uint8_t *gv_wal_get_u32(const uint8_t *p, const uint8_t *end,
                                            uint32_t *v)
{
    if (end - p < 4) return NULL;
    *v = 0;
    for (int i = 0; i < 4; i++) *v |= (uint32_t)p[i] << (8 * i);
    return p + 4;
}

static inline const uint8_t *gv_wal_get_u64(const uint8_t *p, const uint8_t *end,
                                            uint64_t *v)
{
    if (end - p < 8) return NULL;
    *v = 0;
    for (int i = 0; i < 8; i++) *v |= (uint64_t)p[i] << (8 * i);
    return p + 8;
}

static inline const uint8_t *gv_wal_get_f32(const uint8_t *p, const uint8_t *end,
                                            float *v)
{
    uint32_t bits;
    p = gv_wal_get_u32(p, end, &bits);
    if (p) memcpy(v, &bits, sizeof(*v));
    return p;
}

static inline const uint8_t *gv_wal_get_str(const uint8_t *p, const uint8_t *end,
                                            char **out)
{
    uint32_t n;
    if ((p = gv_wal_get_u32(p, end, &n)) == NULL || (size_t)(end - p) < n)
        return NULL;
    *out = (char *)gv_alloc((size_t)n + 1);
    if (!*out) return NULL;
    if (n) memcpy(*out, p, n);
    (*out)[n] = '\0';
    return p + n;
}

static inline const uint8_t *gv_wal_get_bytes(const uint8_t *p, const uint8_t *end,
                                              const void **out, uint32_t *n)
{
    if ((p = gv_wal_get_u32(p, end, n)) == NULL || (size_t)(end - p) < *n)
        return NULL;
    *out = p;
    return p + *n;
}

/* Append one framed record ([op][len][payload] where b->buf starts with the op
 * byte) and fsync it. Returns 0 on success. */
static inline int gv_wal_append_frame(FILE *f, const GV_WalBuf *b)
{
    if (!f || b->overflow || b->len == 0) return -1;
    uint8_t hdr[5];
    hdr[0] = b->buf[0];
    uint32_t plen = (uint32_t)(b->len - 1);
    for (int i = 0; i < 4; i++) hdr[1 + i] = (uint8_t)(plen >> (8 * i));
    if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) return -1;
    if (b->len > 1 && fwrite(b->buf + 1, 1, b->len - 1, f) != b->len - 1)
        return -1;
#ifndef _WIN32
    if (fflush(f) != 0 || fsync(fileno(f)) != 0) return -1;
#else
    if (fflush(f) != 0 || _commit(_fileno(f)) != 0) return -1;
#endif
    return 0;
}

#endif /* GIGAVECTOR_FEATURES_GV_WAL_CODEC_H */
