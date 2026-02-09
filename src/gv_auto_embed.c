/**
 * @file gv_auto_embed.c
 * @brief Server-side auto-embedding: accept text, call embedding service,
 *        store resulting vector.
 *
 * Supports OpenAI, Google Generative AI, HuggingFace TEI, and custom
 * OpenAI-compatible endpoints.  Includes an LRU cache (hash text ->
 * cached embedding vector) and is fully thread-safe via pthread_mutex_t.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

#include "gigavector/gv_auto_embed.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_json.h"

/* ========================================================================== */
/*  Constants                                                                  */
/* ========================================================================== */

#define AE_DEFAULT_DIMENSION         1536
#define AE_DEFAULT_MAX_CACHE         10000
#define AE_DEFAULT_MAX_TEXT_LENGTH   8192
#define AE_DEFAULT_BATCH_SIZE        32
#define AE_DEFAULT_TIMEOUT_SEC       30
#define AE_CACHE_BUCKET_COUNT        4096
#define AE_MAX_RESPONSE_SIZE         (10 * 1024 * 1024)  /* 10 MB */
#define AE_URL_BUFSIZE               2048
#define AE_AUTH_BUFSIZE              1024

/* ========================================================================== */
/*  LRU cache internals                                                        */
/* ========================================================================== */

typedef struct AE_CacheEntry {
    char                   *text;       /* Key: original text                */
    float                  *embedding;  /* Value: embedding vector           */
    size_t                  dimension;  /* Length of embedding               */
    struct AE_CacheEntry   *hash_next;  /* Hash-chain pointer                */
    struct AE_CacheEntry   *lru_prev;   /* Doubly-linked LRU list            */
    struct AE_CacheEntry   *lru_next;
} AE_CacheEntry;

typedef struct {
    AE_CacheEntry **buckets;
    size_t          bucket_count;
    size_t          max_entries;
    size_t          count;
    AE_CacheEntry  *lru_head;   /* Most recently used  */
    AE_CacheEntry  *lru_tail;   /* Least recently used */
    pthread_mutex_t mutex;
} AE_Cache;

/* ========================================================================== */
/*  Auto-embedder structure                                                    */
/* ========================================================================== */

struct GV_AutoEmbedder {
    /* Configuration (owned copies of strings) */
    GV_AutoEmbedProvider provider;
    char                *api_key;
    char                *model_name;
    char                *base_url;
    size_t               dimension;
    size_t               max_text_length;
    int                  batch_size;

    /* Cache (NULL when disabled) */
    AE_Cache            *cache;

    /* libcurl handle (per-embedder, reused across calls) */
#ifdef HAVE_CURL
    CURL                *curl;
#endif

    /* Statistics */
    uint64_t             total_embeddings;
    uint64_t             cache_hits;
    uint64_t             cache_misses;
    uint64_t             api_calls;
    uint64_t             api_errors;
    double               total_latency_ms;

    /* Thread safety for stats and curl handle */
    pthread_mutex_t      mutex;
};

/* ========================================================================== */
/*  CURL response buffer                                                       */
/* ========================================================================== */

#ifdef HAVE_CURL
typedef struct {
    char   *data;
    size_t  size;
    size_t  capacity;
} AE_ResponseBuf;

static size_t ae_curl_write_cb(void *contents, size_t size, size_t nmemb,
                               void *userp) {
    size_t realsize = size * nmemb;
    AE_ResponseBuf *buf = (AE_ResponseBuf *)userp;

    if (buf->size + realsize > AE_MAX_RESPONSE_SIZE) {
        return 0;
    }

    if (buf->size + realsize >= buf->capacity) {
        size_t new_cap = buf->capacity * 2;
        if (new_cap < buf->size + realsize + 1) {
            new_cap = buf->size + realsize + 1;
        }
        if (new_cap > AE_MAX_RESPONSE_SIZE) {
            new_cap = AE_MAX_RESPONSE_SIZE;
        }
        char *tmp = (char *)realloc(buf->data, new_cap);
        if (!tmp) {
            return 0;
        }
        buf->data     = tmp;
        buf->capacity = new_cap;
    }

    memcpy(buf->data + buf->size, contents, realsize);
    buf->size += realsize;
    buf->data[buf->size] = '\0';
    return realsize;
}

static void ae_response_buf_init(AE_ResponseBuf *buf) {
    buf->capacity = 4096;
    buf->size     = 0;
    buf->data     = (char *)malloc(buf->capacity);
    if (buf->data) {
        buf->data[0] = '\0';
    }
}

static void ae_response_buf_free(AE_ResponseBuf *buf) {
    free(buf->data);
    buf->data     = NULL;
    buf->size     = 0;
    buf->capacity = 0;
}
#endif /* HAVE_CURL */

/* ========================================================================== */
/*  Hash helper (djb2)                                                         */
/* ========================================================================== */

static size_t ae_hash_string(const char *str, size_t bucket_count) {
    size_t h = 5381;
    int c;
    while ((c = (unsigned char)*str++)) {
        h = ((h << 5) + h) + (size_t)c;
    }
    return h % bucket_count;
}

/* ========================================================================== */
/*  LRU cache implementation                                                   */
/* ========================================================================== */

static AE_Cache *ae_cache_create(size_t max_entries) {
    AE_Cache *c = (AE_Cache *)calloc(1, sizeof(AE_Cache));
    if (!c) return NULL;

    c->bucket_count = AE_CACHE_BUCKET_COUNT;
    c->max_entries  = max_entries;
    c->buckets = (AE_CacheEntry **)calloc(c->bucket_count, sizeof(AE_CacheEntry *));
    if (!c->buckets) {
        free(c);
        return NULL;
    }
    if (pthread_mutex_init(&c->mutex, NULL) != 0) {
        free(c->buckets);
        free(c);
        return NULL;
    }
    return c;
}

static void ae_lru_remove(AE_Cache *c, AE_CacheEntry *e) {
    if (e->lru_prev) {
        e->lru_prev->lru_next = e->lru_next;
    } else {
        c->lru_head = e->lru_next;
    }
    if (e->lru_next) {
        e->lru_next->lru_prev = e->lru_prev;
    } else {
        c->lru_tail = e->lru_prev;
    }
    e->lru_prev = NULL;
    e->lru_next = NULL;
}

static void ae_lru_push_front(AE_Cache *c, AE_CacheEntry *e) {
    e->lru_next = c->lru_head;
    e->lru_prev = NULL;
    if (c->lru_head) {
        c->lru_head->lru_prev = e;
    }
    c->lru_head = e;
    if (!c->lru_tail) {
        c->lru_tail = e;
    }
}

static void ae_cache_evict_lru(AE_Cache *c) {
    AE_CacheEntry *victim = c->lru_tail;
    if (!victim) return;

    /* Remove from hash chain */
    size_t h = ae_hash_string(victim->text, c->bucket_count);
    AE_CacheEntry **pp = &c->buckets[h];
    while (*pp && *pp != victim) {
        pp = &(*pp)->hash_next;
    }
    if (*pp) {
        *pp = victim->hash_next;
    }

    ae_lru_remove(c, victim);
    free(victim->text);
    free(victim->embedding);
    free(victim);
    c->count--;
}

/**
 * Look up text in cache.  On hit copies the embedding into *out_vec
 * (caller frees).  Returns 1 on hit, 0 on miss, -1 on error.
 */
static int ae_cache_get(AE_Cache *c, const char *text,
                        float **out_vec, size_t *out_dim) {
    if (!c || !text) return -1;

    pthread_mutex_lock(&c->mutex);

    size_t h = ae_hash_string(text, c->bucket_count);
    AE_CacheEntry *e = c->buckets[h];
    while (e) {
        if (strcmp(e->text, text) == 0) {
            /* Hit -- promote in LRU and copy out */
            ae_lru_remove(c, e);
            ae_lru_push_front(c, e);

            *out_dim = e->dimension;
            *out_vec = (float *)malloc(e->dimension * sizeof(float));
            if (!*out_vec) {
                pthread_mutex_unlock(&c->mutex);
                return -1;
            }
            memcpy(*out_vec, e->embedding, e->dimension * sizeof(float));
            pthread_mutex_unlock(&c->mutex);
            return 1;
        }
        e = e->hash_next;
    }

    pthread_mutex_unlock(&c->mutex);
    return 0;
}

/**
 * Store (or update) an embedding in the cache.
 */
static int ae_cache_put(AE_Cache *c, const char *text,
                        const float *embedding, size_t dim) {
    if (!c || !text || !embedding || dim == 0) return -1;

    pthread_mutex_lock(&c->mutex);

    size_t h = ae_hash_string(text, c->bucket_count);

    /* Check for existing entry and update */
    AE_CacheEntry *e = c->buckets[h];
    while (e) {
        if (strcmp(e->text, text) == 0) {
            if (e->dimension != dim) {
                float *tmp = (float *)realloc(e->embedding, dim * sizeof(float));
                if (!tmp) {
                    pthread_mutex_unlock(&c->mutex);
                    return -1;
                }
                e->embedding = tmp;
                e->dimension = dim;
            }
            memcpy(e->embedding, embedding, dim * sizeof(float));
            ae_lru_remove(c, e);
            ae_lru_push_front(c, e);
            pthread_mutex_unlock(&c->mutex);
            return 0;
        }
        e = e->hash_next;
    }

    /* Evict if at capacity */
    if (c->max_entries > 0 && c->count >= c->max_entries) {
        ae_cache_evict_lru(c);
    }

    /* Insert new entry */
    e = (AE_CacheEntry *)calloc(1, sizeof(AE_CacheEntry));
    if (!e) {
        pthread_mutex_unlock(&c->mutex);
        return -1;
    }
    e->text = strdup(text);
    if (!e->text) {
        free(e);
        pthread_mutex_unlock(&c->mutex);
        return -1;
    }
    e->embedding = (float *)malloc(dim * sizeof(float));
    if (!e->embedding) {
        free(e->text);
        free(e);
        pthread_mutex_unlock(&c->mutex);
        return -1;
    }
    memcpy(e->embedding, embedding, dim * sizeof(float));
    e->dimension = dim;

    /* Prepend to hash chain */
    e->hash_next  = c->buckets[h];
    c->buckets[h] = e;

    ae_lru_push_front(c, e);
    c->count++;

    pthread_mutex_unlock(&c->mutex);
    return 0;
}

static void ae_cache_clear(AE_Cache *c) {
    if (!c) return;

    pthread_mutex_lock(&c->mutex);
    for (size_t i = 0; i < c->bucket_count; i++) {
        AE_CacheEntry *e = c->buckets[i];
        while (e) {
            AE_CacheEntry *next = e->hash_next;
            free(e->text);
            free(e->embedding);
            free(e);
            e = next;
        }
        c->buckets[i] = NULL;
    }
    c->count    = 0;
    c->lru_head = NULL;
    c->lru_tail = NULL;
    pthread_mutex_unlock(&c->mutex);
}

static void ae_cache_destroy(AE_Cache *c) {
    if (!c) return;
    ae_cache_clear(c);
    free(c->buckets);
    pthread_mutex_destroy(&c->mutex);
    free(c);
}

/* ========================================================================== */
/*  JSON request builders                                                      */
/* ========================================================================== */

/**
 * Escape a string for inclusion in a JSON string literal.
 * Writes into dst (up to dst_cap-1 bytes) and NUL-terminates.
 * Returns the number of characters written (excluding NUL).
 */
static size_t ae_json_escape(char *dst, size_t dst_cap, const char *src) {
    size_t w = 0;
    while (*src && w + 6 < dst_cap) {   /* 6 = worst case \uXXXX */
        unsigned char ch = (unsigned char)*src;
        if (ch == '"') {
            dst[w++] = '\\'; dst[w++] = '"';
        } else if (ch == '\\') {
            dst[w++] = '\\'; dst[w++] = '\\';
        } else if (ch == '\n') {
            dst[w++] = '\\'; dst[w++] = 'n';
        } else if (ch == '\r') {
            dst[w++] = '\\'; dst[w++] = 'r';
        } else if (ch == '\t') {
            dst[w++] = '\\'; dst[w++] = 't';
        } else if (ch < 0x20) {
            w += (size_t)snprintf(dst + w, dst_cap - w, "\\u%04x", ch);
        } else {
            dst[w++] = (char)ch;
        }
        src++;
    }
    if (w < dst_cap) dst[w] = '\0';
    return w;
}

/**
 * Build a single-text OpenAI-compatible JSON request body.
 * Caller must free the returned string.
 */
static char *ae_build_openai_request(const char *text, const char *model,
                                     size_t dimension) {
    size_t text_len = strlen(text);
    size_t alloc    = text_len * 2 + 512;
    char *json = (char *)malloc(alloc);
    if (!json) return NULL;

    /* Escape text */
    char *escaped = (char *)malloc(text_len * 2 + 1);
    if (!escaped) { free(json); return NULL; }
    ae_json_escape(escaped, text_len * 2 + 1, text);

    if (dimension > 0) {
        snprintf(json, alloc,
                 "{\"input\":\"%s\",\"model\":\"%s\",\"dimensions\":%zu}",
                 escaped, model, dimension);
    } else {
        snprintf(json, alloc,
                 "{\"input\":\"%s\",\"model\":\"%s\"}",
                 escaped, model);
    }
    free(escaped);
    return json;
}

/**
 * Build a single-text Google embedContent JSON request body.
 * Caller must free the returned string.
 */
static char *ae_build_google_request(const char *text, const char *model,
                                     size_t dimension) {
    size_t text_len = strlen(text);
    size_t alloc    = text_len * 2 + 512;
    char *json = (char *)malloc(alloc);
    if (!json) return NULL;

    char *escaped = (char *)malloc(text_len * 2 + 1);
    if (!escaped) { free(json); return NULL; }
    ae_json_escape(escaped, text_len * 2 + 1, text);

    const char *prefix = (strncmp(model, "models/", 7) != 0) ? "models/" : "";

    if (dimension > 0) {
        snprintf(json, alloc,
                 "{\"model\":\"%s%s\","
                 "\"content\":{\"parts\":[{\"text\":\"%s\"}]},"
                 "\"outputDimensionality\":%zu}",
                 prefix, model, escaped, dimension);
    } else {
        snprintf(json, alloc,
                 "{\"model\":\"%s%s\","
                 "\"content\":{\"parts\":[{\"text\":\"%s\"}]}}",
                 prefix, model, escaped);
    }
    free(escaped);
    return json;
}

/**
 * Build a batch OpenAI-compatible JSON request body.
 * Caller must free the returned string.
 */
static char *ae_build_openai_batch_request(const char *const *texts,
                                           size_t count,
                                           const char *model,
                                           size_t dimension) {
    /* Estimate upper bound for allocation */
    size_t alloc = 512;
    for (size_t i = 0; i < count; i++) {
        alloc += strlen(texts[i]) * 2 + 16;
    }
    char *json = (char *)malloc(alloc);
    if (!json) return NULL;

    size_t pos = 0;
    pos += (size_t)snprintf(json + pos, alloc - pos, "{\"input\":[");

    for (size_t i = 0; i < count; i++) {
        if (i > 0) json[pos++] = ',';
        json[pos++] = '"';
        pos += ae_json_escape(json + pos, alloc - pos, texts[i]);
        json[pos++] = '"';
    }

    if (dimension > 0) {
        snprintf(json + pos, alloc - pos,
                 "],\"model\":\"%s\",\"dimensions\":%zu}", model, dimension);
    } else {
        snprintf(json + pos, alloc - pos, "],\"model\":\"%s\"}", model);
    }
    return json;
}

/**
 * Build a batch Google batchEmbedContents JSON request body.
 * Caller must free the returned string.
 */
static char *ae_build_google_batch_request(const char *const *texts,
                                           size_t count,
                                           const char *model,
                                           size_t dimension) {
    size_t alloc = 512;
    for (size_t i = 0; i < count; i++) {
        alloc += strlen(texts[i]) * 2 + 256;
    }
    char *json = (char *)malloc(alloc);
    if (!json) return NULL;

    const char *prefix = (strncmp(model, "models/", 7) != 0) ? "models/" : "";

    size_t pos = 0;
    pos += (size_t)snprintf(json + pos, alloc - pos, "{\"requests\":[");

    for (size_t i = 0; i < count; i++) {
        if (i > 0) json[pos++] = ',';

        pos += (size_t)snprintf(json + pos, alloc - pos,
                                "{\"model\":\"%s%s\","
                                "\"content\":{\"parts\":[{\"text\":\"",
                                prefix, model);
        pos += ae_json_escape(json + pos, alloc - pos, texts[i]);

        if (dimension > 0) {
            pos += (size_t)snprintf(json + pos, alloc - pos,
                                    "\"}]},\"outputDimensionality\":%zu}",
                                    dimension);
        } else {
            pos += (size_t)snprintf(json + pos, alloc - pos, "\"}]}}");
        }
    }
    snprintf(json + pos, alloc - pos, "]}");
    return json;
}

/**
 * Build a single-text HuggingFace TEI JSON request body.
 * Caller must free the returned string.
 */
static char *ae_build_hf_request(const char *text) {
    size_t text_len = strlen(text);
    size_t alloc    = text_len * 2 + 128;
    char *json = (char *)malloc(alloc);
    if (!json) return NULL;

    char *escaped = (char *)malloc(text_len * 2 + 1);
    if (!escaped) { free(json); return NULL; }
    ae_json_escape(escaped, text_len * 2 + 1, text);

    snprintf(json, alloc, "{\"input\":\"%s\"}", escaped);
    free(escaped);
    return json;
}

/**
 * Build a batch HuggingFace TEI JSON request body.
 * Caller must free the returned string.
 */
static char *ae_build_hf_batch_request(const char *const *texts, size_t count) {
    size_t alloc = 128;
    for (size_t i = 0; i < count; i++) {
        alloc += strlen(texts[i]) * 2 + 16;
    }
    char *json = (char *)malloc(alloc);
    if (!json) return NULL;

    size_t pos = 0;
    pos += (size_t)snprintf(json + pos, alloc - pos, "{\"input\":[");
    for (size_t i = 0; i < count; i++) {
        if (i > 0) json[pos++] = ',';
        json[pos++] = '"';
        pos += ae_json_escape(json + pos, alloc - pos, texts[i]);
        json[pos++] = '"';
    }
    snprintf(json + pos, alloc - pos, "]}");
    return json;
}

/* ========================================================================== */
/*  JSON response parsers (using project gv_json utilities)                    */
/* ========================================================================== */

/**
 * Parse an OpenAI /v1/embeddings response to extract the first embedding
 * vector.  Returns 0 on success, -1 on error.  Caller must free *out_vec.
 */
static int ae_parse_openai_single(const char *body, float **out_vec,
                                  size_t *out_dim) {
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(body, &err);
    if (!root) return -1;

    /* data.0.embedding */
    GV_JsonValue *embedding_arr = gv_json_get_path(root, "data.0.embedding");
    if (!embedding_arr || !gv_json_is_array(embedding_arr)) {
        gv_json_free(root);
        return -1;
    }

    size_t dim = gv_json_array_length(embedding_arr);
    if (dim == 0) {
        gv_json_free(root);
        return -1;
    }

    float *vec = (float *)malloc(dim * sizeof(float));
    if (!vec) {
        gv_json_free(root);
        return -1;
    }

    for (size_t i = 0; i < dim; i++) {
        GV_JsonValue *elem = gv_json_array_get(embedding_arr, i);
        double val = 0.0;
        if (elem && gv_json_get_number(elem, &val) == GV_JSON_OK) {
            vec[i] = (float)val;
        } else {
            vec[i] = 0.0f;
        }
    }

    *out_vec = vec;
    *out_dim = dim;
    gv_json_free(root);
    return 0;
}

/**
 * Parse an OpenAI /v1/embeddings batch response to extract all embedding
 * vectors.  Writes into pre-allocated arrays out_vecs[count] and
 * out_dims[count].  Returns number of successfully parsed embeddings.
 */
static int ae_parse_openai_batch(const char *body, size_t count,
                                 float **out_vecs, size_t *out_dims) {
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(body, &err);
    if (!root) return -1;

    GV_JsonValue *data_arr = gv_json_object_get(root, "data");
    if (!data_arr || !gv_json_is_array(data_arr)) {
        gv_json_free(root);
        return -1;
    }

    size_t arr_len = gv_json_array_length(data_arr);
    if (arr_len < count) count = arr_len;

    int success = 0;
    for (size_t i = 0; i < count; i++) {
        GV_JsonValue *item = gv_json_array_get(data_arr, i);
        if (!item) { out_vecs[i] = NULL; out_dims[i] = 0; continue; }

        GV_JsonValue *emb = gv_json_object_get(item, "embedding");
        if (!emb || !gv_json_is_array(emb)) {
            out_vecs[i] = NULL; out_dims[i] = 0; continue;
        }

        size_t dim = gv_json_array_length(emb);
        float *vec = (float *)malloc(dim * sizeof(float));
        if (!vec) { out_vecs[i] = NULL; out_dims[i] = 0; continue; }

        for (size_t j = 0; j < dim; j++) {
            GV_JsonValue *e = gv_json_array_get(emb, j);
            double v = 0.0;
            if (e) gv_json_get_number(e, &v);
            vec[j] = (float)v;
        }
        out_vecs[i] = vec;
        out_dims[i] = dim;
        success++;
    }

    gv_json_free(root);
    return success;
}

/**
 * Parse a Google embedContent response.  Format:
 *   { "embedding": { "values": [ ... ] } }
 */
static int ae_parse_google_single(const char *body, float **out_vec,
                                  size_t *out_dim) {
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(body, &err);
    if (!root) return -1;

    GV_JsonValue *values = gv_json_get_path(root, "embedding.values");
    if (!values || !gv_json_is_array(values)) {
        gv_json_free(root);
        return -1;
    }

    size_t dim = gv_json_array_length(values);
    if (dim == 0) { gv_json_free(root); return -1; }

    float *vec = (float *)malloc(dim * sizeof(float));
    if (!vec) { gv_json_free(root); return -1; }

    for (size_t i = 0; i < dim; i++) {
        GV_JsonValue *e = gv_json_array_get(values, i);
        double v = 0.0;
        if (e) gv_json_get_number(e, &v);
        vec[i] = (float)v;
    }

    *out_vec = vec;
    *out_dim = dim;
    gv_json_free(root);
    return 0;
}

/**
 * Parse a Google batchEmbedContents response.  Format:
 *   { "embeddings": [ { "values": [ ... ] }, ... ] }
 */
static int ae_parse_google_batch(const char *body, size_t count,
                                 float **out_vecs, size_t *out_dims) {
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(body, &err);
    if (!root) return -1;

    GV_JsonValue *embeddings = gv_json_object_get(root, "embeddings");
    if (!embeddings || !gv_json_is_array(embeddings)) {
        gv_json_free(root);
        return -1;
    }

    size_t arr_len = gv_json_array_length(embeddings);
    if (arr_len < count) count = arr_len;

    int success = 0;
    for (size_t i = 0; i < count; i++) {
        GV_JsonValue *item = gv_json_array_get(embeddings, i);
        GV_JsonValue *vals = item ? gv_json_object_get(item, "values") : NULL;
        if (!vals || !gv_json_is_array(vals)) {
            out_vecs[i] = NULL; out_dims[i] = 0; continue;
        }

        size_t dim = gv_json_array_length(vals);
        float *vec = (float *)malloc(dim * sizeof(float));
        if (!vec) { out_vecs[i] = NULL; out_dims[i] = 0; continue; }

        for (size_t j = 0; j < dim; j++) {
            GV_JsonValue *e = gv_json_array_get(vals, j);
            double v = 0.0;
            if (e) gv_json_get_number(e, &v);
            vec[j] = (float)v;
        }
        out_vecs[i] = vec;
        out_dims[i] = dim;
        success++;
    }

    gv_json_free(root);
    return success;
}

/* ========================================================================== */
/*  HTTP call helpers (libcurl)                                                */
/* ========================================================================== */

#ifdef HAVE_CURL

/**
 * Resolve the endpoint URL for a single-text embedding call.
 */
static void ae_resolve_url(const GV_AutoEmbedder *em, char *url, size_t url_cap) {
    switch (em->provider) {
    case GV_EMBED_PROVIDER_OPENAI:
        snprintf(url, url_cap, "https://api.openai.com/v1/embeddings");
        break;
    case GV_EMBED_PROVIDER_GOOGLE: {
        const char *m = em->model_name ? em->model_name : "text-embedding-004";
        if (strncmp(m, "models/", 7) == 0) {
            snprintf(url, url_cap,
                     "https://generativelanguage.googleapis.com/v1beta/%s:embedContent", m);
        } else {
            snprintf(url, url_cap,
                     "https://generativelanguage.googleapis.com/v1beta/models/%s:embedContent", m);
        }
        break;
    }
    case GV_EMBED_PROVIDER_HUGGINGFACE: {
        const char *base = em->base_url ? em->base_url
                                        : "http://localhost:3000/v1/embeddings";
        if (strstr(base, "/embeddings")) {
            snprintf(url, url_cap, "%s", base);
        } else {
            snprintf(url, url_cap, "%s/embeddings", base);
        }
        break;
    }
    case GV_EMBED_PROVIDER_CUSTOM:
        snprintf(url, url_cap, "%s",
                 em->base_url ? em->base_url : "http://localhost:8080/v1/embeddings");
        break;
    }
}

/**
 * Resolve the endpoint URL for a batch embedding call (Google uses a
 * different endpoint).
 */
static void ae_resolve_batch_url(const GV_AutoEmbedder *em, char *url,
                                 size_t url_cap) {
    if (em->provider == GV_EMBED_PROVIDER_GOOGLE) {
        const char *m = em->model_name ? em->model_name : "text-embedding-004";
        if (strncmp(m, "models/", 7) == 0) {
            snprintf(url, url_cap,
                     "https://generativelanguage.googleapis.com/v1beta/%s:batchEmbedContents", m);
        } else {
            snprintf(url, url_cap,
                     "https://generativelanguage.googleapis.com/v1beta/models/%s:batchEmbedContents", m);
        }
    } else {
        ae_resolve_url(em, url, url_cap);
    }
}

/**
 * Build auth + content-type headers.  Returns a curl_slist that must be
 * freed with curl_slist_free_all().
 */
static struct curl_slist *ae_build_headers(const GV_AutoEmbedder *em) {
    struct curl_slist *hdrs = NULL;
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");

    if (em->api_key && em->api_key[0]) {
        char auth[AE_AUTH_BUFSIZE];
        if (em->provider == GV_EMBED_PROVIDER_GOOGLE) {
            snprintf(auth, sizeof(auth), "x-goog-api-key: %s", em->api_key);
        } else {
            snprintf(auth, sizeof(auth), "Authorization: Bearer %s", em->api_key);
        }
        hdrs = curl_slist_append(hdrs, auth);
    }
    return hdrs;
}

/**
 * Perform a POST request and return the response body.
 * On success, buf->data is NUL-terminated.  Caller must free buf.
 * Returns 0 on success, -1 on network/curl error.
 */
static int ae_http_post(GV_AutoEmbedder *em, const char *url,
                        const char *req_body, AE_ResponseBuf *buf) {
    ae_response_buf_init(buf);
    if (!buf->data) return -1;

    struct curl_slist *hdrs = ae_build_headers(em);

    curl_easy_reset(em->curl);
    curl_easy_setopt(em->curl, CURLOPT_URL, url);
    curl_easy_setopt(em->curl, CURLOPT_POSTFIELDS, req_body);
    curl_easy_setopt(em->curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(em->curl, CURLOPT_WRITEFUNCTION, ae_curl_write_cb);
    curl_easy_setopt(em->curl, CURLOPT_WRITEDATA, buf);
    curl_easy_setopt(em->curl, CURLOPT_TIMEOUT, (long)AE_DEFAULT_TIMEOUT_SEC);
    curl_easy_setopt(em->curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(em->curl);
    curl_slist_free_all(hdrs);

    if (res != CURLE_OK) {
        ae_response_buf_free(buf);
        return -1;
    }

    /* Check HTTP status */
    long http_code = 0;
    curl_easy_getinfo(em->curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code < 200 || http_code >= 300) {
        ae_response_buf_free(buf);
        return -1;
    }

    return 0;
}

#endif /* HAVE_CURL */

/* ========================================================================== */
/*  Timing helper                                                              */
/* ========================================================================== */

static double ae_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ========================================================================== */
/*  Core: single-text embedding via HTTP                                       */
/* ========================================================================== */

/**
 * Internal: perform a single-text embedding API call.
 * Returns 0 on success (-1 on error).  Caller must free *out_vec.
 */
static int ae_embed_single_http(GV_AutoEmbedder *em, const char *text,
                                float **out_vec, size_t *out_dim) {
#ifdef HAVE_CURL
    if (!em->curl) return -1;

    const char *model = em->model_name ? em->model_name : "text-embedding-3-small";
    char *req_body = NULL;

    switch (em->provider) {
    case GV_EMBED_PROVIDER_OPENAI:
    case GV_EMBED_PROVIDER_CUSTOM:
        req_body = ae_build_openai_request(text, model, em->dimension);
        break;
    case GV_EMBED_PROVIDER_GOOGLE:
        req_body = ae_build_google_request(text, model, em->dimension);
        break;
    case GV_EMBED_PROVIDER_HUGGINGFACE:
        req_body = ae_build_hf_request(text);
        break;
    }

    if (!req_body) return -1;

    char url[AE_URL_BUFSIZE];
    ae_resolve_url(em, url, sizeof(url));

    AE_ResponseBuf resp;
    int rc = ae_http_post(em, url, req_body, &resp);
    free(req_body);
    if (rc != 0) return -1;

    /* Parse provider-specific response */
    int parse_rc = -1;
    switch (em->provider) {
    case GV_EMBED_PROVIDER_OPENAI:
    case GV_EMBED_PROVIDER_CUSTOM:
    case GV_EMBED_PROVIDER_HUGGINGFACE:
        parse_rc = ae_parse_openai_single(resp.data, out_vec, out_dim);
        break;
    case GV_EMBED_PROVIDER_GOOGLE:
        parse_rc = ae_parse_google_single(resp.data, out_vec, out_dim);
        break;
    }

    ae_response_buf_free(&resp);
    return parse_rc;
#else
    (void)em; (void)text; (void)out_vec; (void)out_dim;
    return -1;
#endif
}

/**
 * Internal: perform a batch embedding API call.
 * Writes into pre-allocated out_vecs[count] / out_dims[count].
 * Returns number of successful embeddings, or -1 on total failure.
 */
static int ae_embed_batch_http(GV_AutoEmbedder *em, const char *const *texts,
                               size_t count, float **out_vecs,
                               size_t *out_dims) {
#ifdef HAVE_CURL
    if (!em->curl) return -1;

    const char *model = em->model_name ? em->model_name : "text-embedding-3-small";
    char *req_body = NULL;

    switch (em->provider) {
    case GV_EMBED_PROVIDER_OPENAI:
    case GV_EMBED_PROVIDER_CUSTOM:
        req_body = ae_build_openai_batch_request(texts, count, model, em->dimension);
        break;
    case GV_EMBED_PROVIDER_GOOGLE:
        req_body = ae_build_google_batch_request(texts, count, model, em->dimension);
        break;
    case GV_EMBED_PROVIDER_HUGGINGFACE:
        req_body = ae_build_hf_batch_request(texts, count);
        break;
    }

    if (!req_body) return -1;

    char url[AE_URL_BUFSIZE];
    ae_resolve_batch_url(em, url, sizeof(url));

    AE_ResponseBuf resp;
    int rc = ae_http_post(em, url, req_body, &resp);
    free(req_body);
    if (rc != 0) return -1;

    int parsed = -1;
    switch (em->provider) {
    case GV_EMBED_PROVIDER_OPENAI:
    case GV_EMBED_PROVIDER_CUSTOM:
    case GV_EMBED_PROVIDER_HUGGINGFACE:
        parsed = ae_parse_openai_batch(resp.data, count, out_vecs, out_dims);
        break;
    case GV_EMBED_PROVIDER_GOOGLE:
        parsed = ae_parse_google_batch(resp.data, count, out_vecs, out_dims);
        break;
    }

    ae_response_buf_free(&resp);
    return parsed;
#else
    (void)em; (void)texts; (void)count; (void)out_vecs; (void)out_dims;
    return -1;
#endif
}

/* ========================================================================== */
/*  Public API                                                                 */
/* ========================================================================== */

void gv_auto_embed_config_init(GV_AutoEmbedConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->provider          = GV_EMBED_PROVIDER_OPENAI;
    config->dimension         = AE_DEFAULT_DIMENSION;
    config->cache_embeddings  = 1;
    config->max_cache_entries = AE_DEFAULT_MAX_CACHE;
    config->max_text_length   = AE_DEFAULT_MAX_TEXT_LENGTH;
    config->batch_size        = AE_DEFAULT_BATCH_SIZE;
}

GV_AutoEmbedder *gv_auto_embed_create(const GV_AutoEmbedConfig *config) {
    if (!config) return NULL;

    GV_AutoEmbedder *em = (GV_AutoEmbedder *)calloc(1, sizeof(GV_AutoEmbedder));
    if (!em) return NULL;

    em->provider        = config->provider;
    em->api_key         = config->api_key    ? strdup(config->api_key)    : NULL;
    em->model_name      = config->model_name ? strdup(config->model_name) : NULL;
    em->base_url        = config->base_url   ? strdup(config->base_url)   : NULL;
    em->dimension       = config->dimension > 0 ? config->dimension : AE_DEFAULT_DIMENSION;
    em->max_text_length = config->max_text_length > 0 ? config->max_text_length
                                                      : AE_DEFAULT_MAX_TEXT_LENGTH;
    em->batch_size      = config->batch_size > 0 ? config->batch_size
                                                 : AE_DEFAULT_BATCH_SIZE;

    if (config->cache_embeddings) {
        size_t max_ent = config->max_cache_entries > 0 ? config->max_cache_entries
                                                       : AE_DEFAULT_MAX_CACHE;
        em->cache = ae_cache_create(max_ent);
    }

    if (pthread_mutex_init(&em->mutex, NULL) != 0) {
        ae_cache_destroy(em->cache);
        free(em->api_key);
        free(em->model_name);
        free(em->base_url);
        free(em);
        return NULL;
    }

#ifdef HAVE_CURL
    em->curl = curl_easy_init();
    if (!em->curl) {
        pthread_mutex_destroy(&em->mutex);
        ae_cache_destroy(em->cache);
        free(em->api_key);
        free(em->model_name);
        free(em->base_url);
        free(em);
        return NULL;
    }
#endif

    return em;
}

void gv_auto_embed_destroy(GV_AutoEmbedder *embedder) {
    if (!embedder) return;

#ifdef HAVE_CURL
    if (embedder->curl) {
        curl_easy_cleanup(embedder->curl);
    }
#endif

    ae_cache_destroy(embedder->cache);
    pthread_mutex_destroy(&embedder->mutex);
    free(embedder->api_key);
    free(embedder->model_name);
    free(embedder->base_url);
    free(embedder);
}

/* ------------------------------------------------------------------------ */
/*  gv_auto_embed_text -- raw embedding                                      */
/* ------------------------------------------------------------------------ */

float *gv_auto_embed_text(GV_AutoEmbedder *embedder, const char *text,
                          size_t *out_dimension) {
    if (!embedder || !text || !out_dimension) return NULL;

    /* Truncate if exceeding configured limit */
    size_t text_len = strlen(text);
    char *truncated  = NULL;
    const char *input = text;
    if (text_len > embedder->max_text_length) {
        truncated = (char *)malloc(embedder->max_text_length + 1);
        if (!truncated) return NULL;
        memcpy(truncated, text, embedder->max_text_length);
        truncated[embedder->max_text_length] = '\0';
        input = truncated;
    }

    /* Cache lookup */
    float *vec  = NULL;
    size_t dim  = 0;
    if (embedder->cache) {
        int hit = ae_cache_get(embedder->cache, input, &vec, &dim);
        if (hit == 1) {
            pthread_mutex_lock(&embedder->mutex);
            embedder->cache_hits++;
            embedder->total_embeddings++;
            pthread_mutex_unlock(&embedder->mutex);
            *out_dimension = dim;
            free(truncated);
            return vec;
        }
    }

    /* API call */
    double t0 = ae_time_ms();
    int rc = ae_embed_single_http(embedder, input, &vec, &dim);
    double elapsed = ae_time_ms() - t0;

    pthread_mutex_lock(&embedder->mutex);
    embedder->api_calls++;
    if (rc != 0) {
        embedder->api_errors++;
        if (embedder->cache) embedder->cache_misses++;
        pthread_mutex_unlock(&embedder->mutex);
        free(truncated);
        return NULL;
    }
    embedder->total_embeddings++;
    if (embedder->cache) embedder->cache_misses++;
    /* Running average latency */
    double n = (double)embedder->api_calls;
    embedder->total_latency_ms += elapsed;
    (void)n; /* total_latency_ms used in get_stats */
    pthread_mutex_unlock(&embedder->mutex);

    /* Store in cache */
    if (embedder->cache) {
        ae_cache_put(embedder->cache, input, vec, dim);
    }

    *out_dimension = dim;
    free(truncated);
    return vec;
}

/* ------------------------------------------------------------------------ */
/*  gv_auto_embed_add_text                                                   */
/* ------------------------------------------------------------------------ */

int gv_auto_embed_add_text(GV_AutoEmbedder *embedder, GV_Database *db,
                           const char *text,
                           const char *metadata_key,
                           const char *metadata_value) {
    if (!embedder || !db || !text) return -1;

    size_t dim = 0;
    float *vec = gv_auto_embed_text(embedder, text, &dim);
    if (!vec) return -1;

    int rc;
    if (metadata_key && metadata_value) {
        rc = gv_db_add_vector_with_metadata(db, vec, dim, metadata_key, metadata_value);
    } else {
        rc = gv_db_add_vector(db, vec, dim);
    }

    free(vec);
    return rc;
}

/* ------------------------------------------------------------------------ */
/*  gv_auto_embed_search_text                                                */
/* ------------------------------------------------------------------------ */

int gv_auto_embed_search_text(GV_AutoEmbedder *embedder, const GV_Database *db,
                              const char *text, size_t k, int distance_type,
                              size_t *out_indices, float *out_distances,
                              size_t *out_count) {
    if (!embedder || !db || !text || k == 0 || !out_count) return -1;

    size_t dim = 0;
    float *vec = gv_auto_embed_text(embedder, text, &dim);
    if (!vec) return -1;

    /* Allocate result buffer */
    GV_SearchResult *results = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    if (!results) { free(vec); return -1; }

    int found = gv_db_search(db, vec, k, results, (GV_DistanceType)distance_type);
    free(vec);

    if (found < 0) {
        free(results);
        return -1;
    }

    size_t n = (size_t)found;
    if (n > k) n = k;

    for (size_t i = 0; i < n; i++) {
        if (out_indices)   out_indices[i]   = (size_t)results[i].vector->dimension; /* index is stored differently */
        if (out_distances) out_distances[i] = results[i].distance;
    }

    /*
     * GV_SearchResult stores the vector pointer rather than an explicit index.
     * We need to map back to SoA indices.  The vector pointer comes from
     * SoA storage, so we walk the database to find the matching index.
     * For efficiency, use the result ordering as-is and extract the index
     * from the vector data pointer offset in the SoA storage.
     */
    for (size_t i = 0; i < n; i++) {
        /* Search through database vectors to find matching pointer */
        size_t db_count = gv_database_count(db);
        size_t idx = 0;
        for (size_t j = 0; j < db_count; j++) {
            const float *candidate = gv_database_get_vector(db, j);
            if (candidate == results[i].vector->data) {
                idx = j;
                break;
            }
        }
        if (out_indices) out_indices[i] = idx;
        if (out_distances) out_distances[i] = results[i].distance;
    }

    *out_count = n;
    free(results);
    return 0;
}

/* ------------------------------------------------------------------------ */
/*  gv_auto_embed_add_texts (batch)                                          */
/* ------------------------------------------------------------------------ */

int gv_auto_embed_add_texts(GV_AutoEmbedder *embedder, GV_Database *db,
                            const char *const *texts, size_t count,
                            const char *const *metadata_keys,
                            const char *const *metadata_values) {
    if (!embedder || !db || !texts || count == 0) return -1;

    int total_added = 0;
    size_t batch_sz = (size_t)embedder->batch_size;
    if (batch_sz == 0) batch_sz = AE_DEFAULT_BATCH_SIZE;

    /* Process in batches */
    for (size_t offset = 0; offset < count; offset += batch_sz) {
        size_t chunk = count - offset;
        if (chunk > batch_sz) chunk = batch_sz;

        /* Separate texts into cached and uncached */
        float **vecs = (float **)calloc(chunk, sizeof(float *));
        size_t *dims = (size_t *)calloc(chunk, sizeof(size_t));
        if (!vecs || !dims) {
            free(vecs); free(dims);
            return total_added > 0 ? total_added : -1;
        }

        /* Indices of texts that need API calls */
        size_t *need_api = (size_t *)malloc(chunk * sizeof(size_t));
        size_t  need_count = 0;
        if (!need_api) {
            free(vecs); free(dims);
            return total_added > 0 ? total_added : -1;
        }

        /* Try cache first for each text */
        for (size_t i = 0; i < chunk; i++) {
            const char *t = texts[offset + i];
            if (embedder->cache) {
                int hit = ae_cache_get(embedder->cache, t, &vecs[i], &dims[i]);
                if (hit == 1) {
                    pthread_mutex_lock(&embedder->mutex);
                    embedder->cache_hits++;
                    embedder->total_embeddings++;
                    pthread_mutex_unlock(&embedder->mutex);
                    continue;
                }
            }
            need_api[need_count++] = i;
        }

        /* Batch API call for uncached texts */
        if (need_count > 0) {
            /* Build sub-array of texts to embed */
            const char **sub_texts = (const char **)malloc(need_count * sizeof(char *));
            if (!sub_texts) {
                for (size_t i = 0; i < chunk; i++) free(vecs[i]);
                free(vecs); free(dims); free(need_api);
                return total_added > 0 ? total_added : -1;
            }
            for (size_t i = 0; i < need_count; i++) {
                sub_texts[i] = texts[offset + need_api[i]];
            }

            float **api_vecs = (float **)calloc(need_count, sizeof(float *));
            size_t *api_dims = (size_t *)calloc(need_count, sizeof(size_t));
            if (!api_vecs || !api_dims) {
                free(sub_texts); free(api_vecs); free(api_dims);
                for (size_t i = 0; i < chunk; i++) free(vecs[i]);
                free(vecs); free(dims); free(need_api);
                return total_added > 0 ? total_added : -1;
            }

            double t0 = ae_time_ms();
            int batch_ok = ae_embed_batch_http(embedder, sub_texts, need_count,
                                               api_vecs, api_dims);
            double elapsed = ae_time_ms() - t0;

            pthread_mutex_lock(&embedder->mutex);
            embedder->api_calls++;
            embedder->total_latency_ms += elapsed;
            if (batch_ok < 0) {
                embedder->api_errors++;
            }
            pthread_mutex_unlock(&embedder->mutex);

            if (batch_ok < 0) {
                /* Fallback: embed individually */
                for (size_t i = 0; i < need_count; i++) {
                    size_t idx = need_api[i];
                    vecs[idx] = gv_auto_embed_text(embedder, texts[offset + idx],
                                                   &dims[idx]);
                }
            } else {
                /* Copy batch results into main arrays */
                for (size_t i = 0; i < need_count; i++) {
                    size_t idx = need_api[i];
                    vecs[idx] = api_vecs[i];
                    dims[idx] = api_dims[i];

                    pthread_mutex_lock(&embedder->mutex);
                    embedder->total_embeddings++;
                    if (embedder->cache) embedder->cache_misses++;
                    pthread_mutex_unlock(&embedder->mutex);

                    /* Cache the result */
                    if (embedder->cache && api_vecs[i]) {
                        ae_cache_put(embedder->cache, texts[offset + idx],
                                     api_vecs[i], api_dims[i]);
                    }
                }
            }

            free(sub_texts);
            free(api_vecs);
            free(api_dims);
        }

        free(need_api);

        /* Insert vectors into the database */
        for (size_t i = 0; i < chunk; i++) {
            if (!vecs[i]) continue;

            int rc;
            if (metadata_keys && metadata_values &&
                metadata_keys[offset + i] && metadata_values[offset + i]) {
                rc = gv_db_add_vector_with_metadata(db, vecs[i], dims[i],
                                                    metadata_keys[offset + i],
                                                    metadata_values[offset + i]);
            } else {
                rc = gv_db_add_vector(db, vecs[i], dims[i]);
            }
            if (rc == 0) total_added++;
            free(vecs[i]);
        }

        free(vecs);
        free(dims);
    }

    return total_added;
}

/* ------------------------------------------------------------------------ */
/*  Statistics                                                               */
/* ------------------------------------------------------------------------ */

int gv_auto_embed_get_stats(const GV_AutoEmbedder *embedder,
                            GV_AutoEmbedStats *stats) {
    if (!embedder || !stats) return -1;

    /* Cast away const for mutex -- stats are conceptually read-only but
       the mutex is mutable.  This is safe because we only read. */
    GV_AutoEmbedder *em = (GV_AutoEmbedder *)(uintptr_t)embedder;
    pthread_mutex_lock(&em->mutex);

    stats->total_embeddings = em->total_embeddings;
    stats->cache_hits       = em->cache_hits;
    stats->cache_misses     = em->cache_misses;
    stats->api_calls        = em->api_calls;
    stats->api_errors       = em->api_errors;
    stats->avg_latency_ms   = em->api_calls > 0
                              ? em->total_latency_ms / (double)em->api_calls
                              : 0.0;

    pthread_mutex_unlock(&em->mutex);
    return 0;
}

void gv_auto_embed_clear_cache(GV_AutoEmbedder *embedder) {
    if (!embedder || !embedder->cache) return;
    ae_cache_clear(embedder->cache);
}
