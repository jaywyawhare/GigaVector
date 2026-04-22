#ifndef GV_UTILS_H
#define GV_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "core/types.h"

/**
 * @brief Duplicate a C string using heap allocation.
 *
 * This is a project-local replacement for `strdup(3)` to avoid portability and
 * feature-test-macro issues across platforms.
 *
 * @param s Input string; may be NULL.
 * @return Newly allocated copy of @p s, or NULL if @p s is NULL or allocation fails.
 */
static inline char *gv_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *copy = (char *)malloc(len);
    if (!copy) return NULL;
    memcpy(copy, s, len);
    return copy;
}

/**
 * @brief Duplicate a C string using heap allocation.
 *
 * Alias for `gv_strdup()` kept for readability at call sites.
 *
 * @param s Input string; may be NULL.
 * @return Newly allocated copy of @p s, or NULL if @p s is NULL or allocation fails.
 */
static inline char *gv_dup_cstr(const char *s) {
    return gv_strdup(s);
}

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

static inline int write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static inline int read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
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

static inline int write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, f) == 1 ? 0 : -1;
}

static inline int read_u64(FILE *f, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

static inline int write_f32(FILE *f, float v) {
    return fwrite(&v, sizeof(float), 1, f) == 1 ? 0 : -1;
}

static inline int read_f32(FILE *f, float *v) {
    return (v && fread(v, sizeof(float), 1, f) == 1) ? 0 : -1;
}

static inline int write_floats(FILE *f, const float *data, size_t count) {
    return fwrite(data, sizeof(float), count, f) == count ? 0 : -1;
}

static inline int read_floats(FILE *f, float *data, size_t count) {
    return (data && fread(data, sizeof(float), count, f) == count) ? 0 : -1;
}

static inline int write_bytes(FILE *f, const void *data, size_t count) {
    return fwrite(data, 1, count, f) == count ? 0 : -1;
}

static inline int read_bytes(FILE *f, void *data, size_t count) {
    return (data && fread(data, 1, count, f) == count) ? 0 : -1;
}

static inline int write_size(FILE *f, size_t v) {
    return fwrite(&v, sizeof(size_t), 1, f) == 1 ? 0 : -1;
}

static inline int read_size(FILE *f, size_t *v) {
    return (v && fread(v, sizeof(size_t), 1, f) == 1) ? 0 : -1;
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
    char *s = (char *)malloc((size_t)len + 1);
    if (!s) return NULL;
    if (len > 0 && fread(s, 1, len, f) != len) { free(s); return NULL; }
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
        GV_Metadata *new_meta = (GV_Metadata *)malloc(sizeof(GV_Metadata));
        if (new_meta == NULL) {
            while (head) { GV_Metadata *n = head->next; free(head->key); free(head->value); free(head); head = n; }
            return NULL;
        }
        new_meta->key = (char *)malloc(strlen(current->key) + 1);
        new_meta->value = (char *)malloc(strlen(current->value) + 1);
        if (!new_meta->key || !new_meta->value) {
            free(new_meta->key); free(new_meta->value); free(new_meta);
            while (head) { GV_Metadata *n = head->next; free(head->key); free(head->value); free(head); head = n; }
            return NULL;
        }
        strcpy(new_meta->key, current->key);
        strcpy(new_meta->value, current->value);
        new_meta->next = NULL;
        if (!head) { head = tail = new_meta; } else { tail->next = new_meta; tail = new_meta; }
        current = current->next;
    }
    return head;
}

#endif /* GV_UTILS_H */
