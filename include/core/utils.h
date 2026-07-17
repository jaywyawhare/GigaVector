#ifndef GV_UTILS_H
#define GV_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "core/types.h"
#include "core/memory.h"

/**
 * @brief Duplicate a C string using heap allocation.
 *
 * Alias for `gv_strdup()` kept for readability at call sites.
 */
static inline char *gv_dup_cstr(const char *s) {
    return gv_strdup(s);
}

/* CRC-32 (polynomial 0xEDB88320) */
static inline uint32_t gv_crc32_init(void) { return 0xFFFFFFFFu; }

static inline uint32_t gv_crc32_update(uint32_t crc, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}

static inline uint32_t gv_crc32_finish(uint32_t crc) { return crc ^ 0xFFFFFFFFu; }

/* DJB2 string hash */
static inline uint32_t hash_str(const char *s) {
    uint32_t h = 5381;
    int c;
    while ((c = *s++))
        h = ((h << 5) + h) + (uint32_t)c;
    return h;
}

static inline size_t hash_u64(uint64_t id, size_t bucket_count) {
    size_t hash = 5381;
    const unsigned char *p = (const unsigned char *)&id;
    for (size_t i = 0; i < sizeof(id); i++) {
        hash = ((hash << 5) + hash) + p[i];
    }
    return hash % bucket_count;
}

/*
 * Portable serialization primitives. All multi-byte scalars are written in
 * little-endian byte order and all floats/doubles via an IEEE-754 bit-cast, so
 * files are interchangeable across CPU endianness and 32/64-bit builds. On a
 * little-endian host (x86-64, ARM-LE — the common targets) the emitted bytes
 * are identical to the previous native fwrite(&v) layout, so existing on-disk
 * files remain readable. size_t is normalized to a fixed 8-byte value on disk.
 */
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#  define GV_LITTLE_ENDIAN (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#else
#  define GV_LITTLE_ENDIAN 1  /* assume LE; the byte-wise scalar helpers are correct either way */
#endif

static inline int write_u16(FILE *f, uint16_t v) {
    uint8_t b[2] = { (uint8_t)v, (uint8_t)(v >> 8) };
    return fwrite(b, 1, 2, f) == 2 ? 0 : -1;
}

static inline int read_u16(FILE *f, uint16_t *v) {
    uint8_t b[2];
    if (!v || fread(b, 1, 2, f) != 2) return -1;
    *v = (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
    return 0;
}

static inline int write_u32(FILE *f, uint32_t v) {
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}

static inline int read_u32(FILE *f, uint32_t *v) {
    uint8_t b[4];
    if (!v || fread(b, 1, 4, f) != 4) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

static inline int write_u8(FILE *f, uint8_t v) {
    return fwrite(&v, sizeof(uint8_t), 1, f) == 1 ? 0 : -1;
}

static inline int read_u8(FILE *f, uint8_t *v) {
    return (v && fread(v, sizeof(uint8_t), 1, f) == 1) ? 0 : -1;
}

static inline int write_str(FILE *f, const char *s, uint32_t len) {
    if (write_u32(f, len) != 0) return -1;
    if (len == 0) return 0;
    return fwrite(s, 1, len, f) == len ? 0 : -1;
}

static inline int read_str(FILE *f, char **s, uint32_t len) {
    *s = NULL;
    if (len == 0) {
        *s = (char *)gv_alloc(1);
        if (!*s) return -1;
        (*s)[0] = '\0';
        return 0;
    }
    char *buf = (char *)gv_alloc(len + 1);
    if (!buf) return -1;
    if (fread(buf, 1, len, f) != len) { gv_free(buf); return -1; }
    buf[len] = '\0';
    *s = buf;
    return 0;
}

static inline int write_u64(FILE *f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t)(v >> (8 * i));
    return fwrite(b, 1, 8, f) == 8 ? 0 : -1;
}

static inline int read_u64(FILE *f, uint64_t *v) {
    uint8_t b[8];
    if (!v || fread(b, 1, 8, f) != 8) return -1;
    uint64_t r = 0;
    for (int i = 0; i < 8; ++i) r |= (uint64_t)b[i] << (8 * i);
    *v = r;
    return 0;
}

static inline int write_f32(FILE *f, float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return write_u32(f, u);
}

static inline int read_f32(FILE *f, float *v) {
    uint32_t u;
    if (read_u32(f, &u) != 0) return -1;
    if (v) memcpy(v, &u, sizeof(u));
    return 0;
}

static inline int write_f64(FILE *f, double v) {
    uint64_t u;
    memcpy(&u, &v, sizeof(u));
    return write_u64(f, u);
}

static inline int read_f64(FILE *f, double *v) {
    uint64_t u;
    if (read_u64(f, &u) != 0) return -1;
    if (v) memcpy(v, &u, sizeof(u));
    return 0;
}

static inline int write_floats(FILE *f, const float *data, size_t count) {
#if GV_LITTLE_ENDIAN
    return fwrite(data, sizeof(float), count, f) == count ? 0 : -1;
#else
    for (size_t i = 0; i < count; ++i)
        if (write_f32(f, data[i]) != 0) return -1;
    return 0;
#endif
}

static inline int read_floats(FILE *f, float *data, size_t count) {
    if (!data) return -1;
#if GV_LITTLE_ENDIAN
    return fread(data, sizeof(float), count, f) == count ? 0 : -1;
#else
    for (size_t i = 0; i < count; ++i)
        if (read_f32(f, &data[i]) != 0) return -1;
    return 0;
#endif
}

static inline int write_bytes(FILE *f, const void *data, size_t count) {
    return fwrite(data, 1, count, f) == count ? 0 : -1;
}

static inline int read_bytes(FILE *f, void *data, size_t count) {
    return (data && fread(data, 1, count, f) == count) ? 0 : -1;
}

/* size_t is normalized to a fixed 8-byte on-disk value for 32/64-bit portability. */
static inline int write_size(FILE *f, size_t v) {
    return write_u64(f, (uint64_t)v);
}

static inline int read_size(FILE *f, size_t *v) {
    uint64_t u;
    if (read_u64(f, &u) != 0) return -1;
    if (u > (uint64_t)SIZE_MAX) return -1;  /* value exceeds this platform's size_t */
    if (v) *v = (size_t)u;
    return 0;
}

/* Write a string with automatic strlen. */
static inline int write_string(FILE *f, const char *s) {
    uint32_t len = s ? (uint32_t)strlen(s) : 0;
    if (write_u32(f, len) != 0) return -1;
    if (len > 0 && fwrite(s, 1, len, f) != len) return -1;
    return 0;
}

/* Read a length-prefixed string, allocating the buffer. Caller must free. */
static inline char *read_string(FILE *f) {
    uint32_t len;
    if (read_u32(f, &len) != 0) return NULL;
    char *s = (char *)gv_alloc((size_t)len + 1);
    if (!s) return NULL;
    if (len > 0 && fread(s, 1, len, f) != len) { gv_free(s); return NULL; }
    s[len] = '\0';
    return s;
}

/* Write a GV_Metadata linked list (count + key/value pairs). */
static inline int write_metadata(FILE *f, const GV_Metadata *meta) {
    uint32_t count = 0;
    for (const GV_Metadata *c = meta; c; c = c->next) count++;
    if (write_u32(f, count) != 0) return -1;
    for (const GV_Metadata *c = meta; c; c = c->next) {
        if (write_string(f, c->key) != 0) return -1;
        if (write_string(f, c->value) != 0) return -1;
    }
    return 0;
}

static inline float cosine_similarity(const float *a, const float *b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

static inline int metadata_match(const GV_Metadata *meta,
                                     const char *key, const char *value) {
    if (!key || !value) return 1;
    for (const GV_Metadata *cur = meta; cur; cur = cur->next) {
        if (cur->key && cur->value &&
            strcmp(cur->key, key) == 0 && strcmp(cur->value, value) == 0)
            return 1;
    }
    return 0;
}

static inline const char *metadata_get_direct(GV_Metadata *metadata, const char *key) {
    if (metadata == NULL || key == NULL) return NULL;
    GV_Metadata *current = metadata;
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) return current->value;
        current = current->next;
    }
    return NULL;
}

static inline GV_Metadata *metadata_copy(GV_Metadata *src) {
    if (src == NULL) return NULL;
    GV_Metadata *head = NULL, *tail = NULL;
    GV_Metadata *current = src;
    while (current != NULL) {
        GV_Metadata *new_meta = (GV_Metadata *)gv_alloc(sizeof(GV_Metadata));
        if (new_meta == NULL) {
            while (head) { GV_Metadata *n = head->next; gv_free(head->key); gv_free(head->value); gv_free(head); head = n; }
            return NULL;
        }
        new_meta->key = gv_dup_cstr(current->key);
        new_meta->value = gv_dup_cstr(current->value);
        if (!new_meta->key || !new_meta->value) {
            gv_free(new_meta->key); gv_free(new_meta->value); gv_free(new_meta);
            while (head) { GV_Metadata *n = head->next; gv_free(head->key); gv_free(head->value); gv_free(head); head = n; }
            return NULL;
        }
        new_meta->next = NULL;
        if (!head) { head = tail = new_meta; } else { tail->next = new_meta; tail = new_meta; }
        current = current->next;
    }
    return head;
}

#endif /* GV_UTILS_H */
