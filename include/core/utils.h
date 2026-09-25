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

/* Escape src into a JSON string body (no surrounding quotes) in dst. Named
 * escapes for " \ \n \r \t; \uXXXX for other control chars; passthrough else.
 * Bounds-safe (always NUL-terminates, never overruns dst_size). Returns bytes written. */
static inline size_t gv_json_escape(char *dst, size_t dst_size, const char *src) {
    size_t w = 0;
    if (!dst || dst_size == 0) return 0;
    for (; src && *src; src++) {
        unsigned char c = (unsigned char)*src;
        char esc[8]; const char *seq; size_t n;
        switch (c) {
            case '"':  seq = "\\\""; n = 2; break;
            case '\\': seq = "\\\\"; n = 2; break;
            case '\n': seq = "\\n";  n = 2; break;
            case '\r': seq = "\\r";  n = 2; break;
            case '\t': seq = "\\t";  n = 2; break;
            case '\b': seq = "\\b";  n = 2; break;
            case '\f': seq = "\\f";  n = 2; break;
            default:
                if (c < 0x20) { n = (size_t)snprintf(esc, sizeof(esc), "\\u%04x", c); seq = esc; }
                else { esc[0] = (char)c; esc[1] = '\0'; seq = esc; n = 1; }
        }
        if (w + n >= dst_size) break;
        memcpy(dst + w, seq, n); w += n;
    }
    dst[w] = '\0';
    return w;
}

/* Extract the string value of "key" from a FLAT JSON object into out, unescaping
 * \" \\ \/ \n \t \r. For flat claim objects (JWT/OIDC/SAML) — not nested JSON.
 * Fails closed: -1 if the key, opening, or closing quote is missing (e.g. a value
 * truncated by out_size). Returns 0 on success. */
static inline int gv_json_extract_string(const char *json, const char *key,
                                         char *out, size_t out_size) {
    if (!json || !key || !out || out_size == 0) return -1;
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (!pos) return -1;
    pos += strlen(pattern);
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r') pos++;
    if (*pos != ':') return -1;
    pos++;
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r') pos++;
    if (*pos != '"') return -1;
    pos++;
    size_t i = 0;
    while (*pos && *pos != '"' && i < out_size - 1) {
        if (*pos == '\\' && *(pos + 1)) {
            pos++;
            switch (*pos) {
                case '"':  out[i++] = '"';  break;
                case '\\': out[i++] = '\\'; break;
                case '/':  out[i++] = '/';  break;
                case 'n':  out[i++] = '\n'; break;
                case 't':  out[i++] = '\t'; break;
                case 'r':  out[i++] = '\r'; break;
                default:   out[i++] = *pos; break;
            }
        } else {
            out[i++] = *pos;
        }
        pos++;
    }
    out[i] = '\0';
    return (*pos == '"') ? 0 : -1;
}

/* FNV-1a 32-bit hash. String keys pass strlen(s) as len. */
static inline uint32_t gv_fnv1a(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

/* splitmix64 finalizer: decorrelates small-integer seeds and hashes uint64 keys. */
static inline uint64_t gv_mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Little-endian fixed-width packing into byte buffers (on-disk/wire formats). */
static inline void gv_put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;         p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline void gv_put_u64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static inline uint32_t gv_get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint64_t gv_get_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

/* Big-endian (network byte order) u32 packing, for the wire protocols. */
static inline void gv_put_u32_be(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
}
static inline uint32_t gv_get_u32_be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
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

/* Portable serialization primitives: multi-byte scalars in little-endian,
 * floats/doubles via IEEE-754 bit-cast, size_t normalized to 8 bytes — files are
 * interchangeable across endianness and 32/64-bit builds. On LE hosts the bytes
 * match the old native fwrite(&v) layout, so existing on-disk files stay readable. */
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

/*
 * Declares the pthread_key_t/pthread_once_t plumbing for a per-thread key
 * whose sole purpose is running `destructor_fn` when a thread that touched it
 * exits. pthread_once's init callback takes no arguments, so each key still
 * needs its own file-scope create-trampoline; this macro is what keeps that
 * trampoline's shape (and any future fix to it) in exactly one place instead
 * of hand-copied per module. Requires <pthread.h> to already be included by
 * the caller. Expands to `key_var`/`once_var`/`create_fn` at file scope —
 * follow with GV_TLS_KEY_ENSURE(once_var, create_fn) before first use.
 */
#define GV_TLS_KEY_DEFINE(key_var, once_var, create_fn, destructor_fn)   \
    static pthread_key_t  key_var;                                       \
    static pthread_once_t once_var = PTHREAD_ONCE_INIT;                  \
    static void create_fn(void) {                                       \
        (void)pthread_key_create(&(key_var), (destructor_fn));           \
    }
#define GV_TLS_KEY_ENSURE(once_var, create_fn) pthread_once(&(once_var), (create_fn))

#endif /* GV_UTILS_H */
