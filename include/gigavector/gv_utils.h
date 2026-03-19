#ifndef GV_UTILS_H
#define GV_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gigavector/gv_types.h"

static inline char *gv_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *copy = (char *)malloc(len);
    if (!copy) return NULL;
    memcpy(copy, s, len);
    return copy;
}

/* DJB2 string hash */
static inline uint32_t gv_hash_str(const char *s) {
    uint32_t h = 5381;
    int c;
    while ((c = *s++))
        h = ((h << 5) + h) + (uint32_t)c;
    return h;
}

/* DJB2 hash on raw bytes of a uint64_t */
static inline size_t gv_hash_u64(uint64_t id, size_t bucket_count) {
    size_t hash = 5381;
    const unsigned char *p = (const unsigned char *)&id;
    for (size_t i = 0; i < sizeof(id); i++) {
        hash = ((hash << 5) + hash) + p[i];
    }
    return hash % bucket_count;
}

/* Binary I/O helpers for index serialization */
static inline int gv_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static inline int gv_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static inline int gv_write_u8(FILE *f, uint8_t v) {
    return fwrite(&v, sizeof(uint8_t), 1, f) == 1 ? 0 : -1;
}

static inline int gv_read_u8(FILE *f, uint8_t *v) {
    return (v && fread(v, sizeof(uint8_t), 1, f) == 1) ? 0 : -1;
}

static inline int gv_write_str(FILE *f, const char *s, uint32_t len) {
    if (gv_write_u32(f, len) != 0) return -1;
    if (len == 0) return 0;
    return fwrite(s, 1, len, f) == len ? 0 : -1;
}

static inline int gv_read_str(FILE *f, char **s, uint32_t len) {
    *s = NULL;
    if (len == 0) {
        *s = (char *)malloc(1);
        if (!*s) return -1;
        (*s)[0] = '\0';
        return 0;
    }
    char *buf = (char *)malloc(len + 1);
    if (!buf) return -1;
    if (fread(buf, 1, len, f) != len) { free(buf); return -1; }
    buf[len] = '\0';
    *s = buf;
    return 0;
}

/* Metadata linked-list key-value match (NULL filter = always matches) */
static inline int gv_metadata_match(const GV_Metadata *meta,
                                     const char *key, const char *value) {
    if (!key || !value) return 1;
    for (const GV_Metadata *cur = meta; cur; cur = cur->next) {
        if (cur->key && cur->value &&
            strcmp(cur->key, key) == 0 && strcmp(cur->value, value) == 0)
            return 1;
    }
    return 0;
}

#endif /* GV_UTILS_H */
