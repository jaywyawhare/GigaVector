/**
 * @file embedding.c
 * @brief Embedding service implementation with support for multiple providers,
 * batch processing, and caching.
 */

#include <stdlib.h>
#include "core/memory.h"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

#include "multimodal/embedding.h"
#include "core/http_client.h"
#include "core/utils.h"

#define MAX_RESPONSE_SIZE (10 * 1024 * 1024)  // 10MB for batch responses
#define DEFAULT_BATCH_SIZE 100
#define DEFAULT_TIMEOUT 30
#define MAX_CACHE_KEY_LEN 1024

typedef struct CacheEntry {
    char *key;                    /* Text key */
    float *embedding;             /* Embedding vector */
    size_t embedding_dim;         /* Dimension */
    time_t timestamp;             /* Creation time */
    uint64_t access_count;        /* Access count for LRU */
    struct CacheEntry *next;      /* Hash table chain */
    struct CacheEntry *prev_lru;  /* LRU list */
    struct CacheEntry *next_lru;  /* LRU list */
} CacheEntry;

struct GV_EmbeddingCache {
    CacheEntry **buckets;         /* Hash table buckets */
    size_t bucket_count;          /* Number of buckets */
    size_t max_size;              /* Maximum entries (0 = unlimited) */
    size_t current_size;          /* Current number of entries */
    uint64_t hits;                /* Cache hits */
    uint64_t misses;              /* Cache misses */
    CacheEntry *lru_head;         /* LRU head (most recently used) */
    CacheEntry *lru_tail;         /* LRU tail (least recently used) */
    pthread_mutex_t mutex;        /* Thread safety */
};

struct GV_EmbeddingService {
    GV_EmbeddingConfig config;
    GV_EmbeddingCache *cache;
    char *last_error;             /* Last error message */
};

GV_EmbeddingCache *embedding_cache_create(size_t max_size) {
    GV_EmbeddingCache *cache = (GV_EmbeddingCache *)gv_calloc(1, sizeof(GV_EmbeddingCache));
    if (cache == NULL) {
        return NULL;
    }
    
    cache->bucket_count = 1024;
    cache->max_size = max_size;
    cache->current_size = 0;
    cache->hits = 0;
    cache->misses = 0;
    cache->lru_head = NULL;
    cache->lru_tail = NULL;
    
    cache->buckets = (CacheEntry **)gv_calloc(cache->bucket_count, sizeof(CacheEntry *));
    if (cache->buckets == NULL) {
        gv_free(cache);
        return NULL;
    }
    
    if (pthread_mutex_init(&cache->mutex, NULL) != 0) {
        gv_free(cache->buckets);
        gv_free(cache);
        return NULL;
    }
    
    return cache;
}

static void remove_from_lru(GV_EmbeddingCache *cache, CacheEntry *entry) {
    if (entry->prev_lru) {
        entry->prev_lru->next_lru = entry->next_lru;
    } else {
        cache->lru_head = entry->next_lru;
    }
    if (entry->next_lru) {
        entry->next_lru->prev_lru = entry->prev_lru;
    } else {
        cache->lru_tail = entry->prev_lru;
    }
    entry->prev_lru = NULL;
    entry->next_lru = NULL;
}

static void add_to_lru_head(GV_EmbeddingCache *cache, CacheEntry *entry) {
    entry->next_lru = cache->lru_head;
    entry->prev_lru = NULL;
    if (cache->lru_head) {
        cache->lru_head->prev_lru = entry;
    }
    cache->lru_head = entry;
    if (cache->lru_tail == NULL) {
        cache->lru_tail = entry;
    }
}

static void evict_lru(GV_EmbeddingCache *cache) {
    if (cache->lru_tail == NULL) {
        return;
    }
    
    CacheEntry *entry = cache->lru_tail;
    size_t hash = hash_str(entry->key) % cache->bucket_count;
    
    CacheEntry **prev = &cache->buckets[hash];
    while (*prev != entry) {
        prev = &(*prev)->next;
    }
    *prev = entry->next;
    remove_from_lru(cache, entry);
    gv_free(entry->key);
    gv_free(entry->embedding);
    gv_free(entry);
    cache->current_size--;
}

int embedding_cache_get(GV_EmbeddingCache *cache,
                          const char *text,
                          size_t *embedding_dim,
                          const float **embedding) {
    if (cache == NULL || text == NULL || embedding_dim == NULL || embedding == NULL) {
        return -1;
    }
    
    pthread_mutex_lock(&cache->mutex);
    
    size_t hash = hash_str(text) % cache->bucket_count;
    CacheEntry *entry = cache->buckets[hash];
    
    while (entry != NULL) {
        if (strcmp(entry->key, text) == 0) {
            remove_from_lru(cache, entry);
            add_to_lru_head(cache, entry);
            entry->access_count++;
            cache->hits++;
            
            *embedding_dim = entry->embedding_dim;
            *embedding = entry->embedding;
            
            pthread_mutex_unlock(&cache->mutex);
            return 1;
        }
        entry = entry->next;
    }
    
    cache->misses++;
    pthread_mutex_unlock(&cache->mutex);
    return 0;
}

int embedding_cache_put(GV_EmbeddingCache *cache,
                           const char *text,
                           size_t embedding_dim,
                           const float *embedding) {
    if (cache == NULL || text == NULL || embedding == NULL || embedding_dim == 0) {
        return -1;
    }
    
    pthread_mutex_lock(&cache->mutex);
    
    size_t hash = hash_str(text) % cache->bucket_count;
    CacheEntry *entry = cache->buckets[hash];
    
    while (entry != NULL) {
        if (strcmp(entry->key, text) == 0) {
            if (entry->embedding_dim != embedding_dim) {
                gv_free(entry->embedding);
                entry->embedding = (float *)gv_alloc(embedding_dim * sizeof(float));
                if (entry->embedding == NULL) {
                    pthread_mutex_unlock(&cache->mutex);
                    return -1;
                }
                entry->embedding_dim = embedding_dim;
            }
            memcpy(entry->embedding, embedding, embedding_dim * sizeof(float));
            remove_from_lru(cache, entry);
            add_to_lru_head(cache, entry);
            entry->access_count++;
            
            pthread_mutex_unlock(&cache->mutex);
            return 0;
        }
        entry = entry->next;
    }
    
    if (cache->max_size > 0 && cache->current_size >= cache->max_size) {
        evict_lru(cache);
    }

    entry = (CacheEntry *)gv_calloc(1, sizeof(CacheEntry));
    if (entry == NULL) {
        pthread_mutex_unlock(&cache->mutex);
        return -1;
    }
    
    entry->key = gv_dup_cstr(text);
    if (entry->key == NULL) {
        gv_free(entry);
        pthread_mutex_unlock(&cache->mutex);
        return -1;
    }
    
    entry->embedding = (float *)gv_alloc(embedding_dim * sizeof(float));
    if (entry->embedding == NULL) {
        gv_free(entry->key);
        gv_free(entry);
        pthread_mutex_unlock(&cache->mutex);
        return -1;
    }
    
    memcpy(entry->embedding, embedding, embedding_dim * sizeof(float));
    entry->embedding_dim = embedding_dim;
    entry->timestamp = time(NULL);
    entry->access_count = 1;
    
    entry->next = cache->buckets[hash];
    cache->buckets[hash] = entry;
    add_to_lru_head(cache, entry);
    
    cache->current_size++;
    pthread_mutex_unlock(&cache->mutex);
    return 0;
}

void embedding_cache_clear(GV_EmbeddingCache *cache) {
    if (cache == NULL) {
        return;
    }
    
    pthread_mutex_lock(&cache->mutex);
    
    for (size_t i = 0; i < cache->bucket_count; i++) {
        CacheEntry *entry = cache->buckets[i];
        while (entry != NULL) {
            CacheEntry *next = entry->next;
            gv_free(entry->key);
            gv_free(entry->embedding);
            gv_free(entry);
            entry = next;
        }
        cache->buckets[i] = NULL;
    }
    
    cache->current_size = 0;
    cache->lru_head = NULL;
    cache->lru_tail = NULL;
    cache->hits = 0;
    cache->misses = 0;
    
    pthread_mutex_unlock(&cache->mutex);
}

void embedding_cache_stats(GV_EmbeddingCache *cache,
                              size_t *size,
                              uint64_t *hits,
                              uint64_t *misses) {
    if (cache == NULL) {
        return;
    }
    
    pthread_mutex_lock(&cache->mutex);
    if (size) *size = cache->current_size;
    if (hits) *hits = cache->hits;
    if (misses) *misses = cache->misses;
    pthread_mutex_unlock(&cache->mutex);
}

void embedding_cache_destroy(GV_EmbeddingCache *cache) {
    if (cache == NULL) {
        return;
    }
    
    embedding_cache_clear(cache);
    gv_free(cache->buckets);
    pthread_mutex_destroy(&cache->mutex);
    gv_free(cache);
}

GV_EmbeddingConfig embedding_config_default(void) {
    GV_EmbeddingConfig config;
    memset(&config, 0, sizeof(config));
    config.provider = GV_EMBEDDING_PROVIDER_NONE;
    config.embedding_dimension = 0;
    config.batch_size = DEFAULT_BATCH_SIZE;
    config.enable_cache = 1;
    config.cache_size = 1000;
    config.timeout_seconds = DEFAULT_TIMEOUT;
    return config;
}

void embedding_config_free(GV_EmbeddingConfig *config) {
    if (config == NULL) {
        return;
    }
    if (config->api_key) {
        gv_free(config->api_key);
        config->api_key = NULL;
    }
    if (config->model) {
        gv_free(config->model);
        config->model = NULL;
    }
    if (config->base_url) {
        gv_free(config->base_url);
        config->base_url = NULL;
    }
    if (config->huggingface_model_path) {
        gv_free(config->huggingface_model_path);
        config->huggingface_model_path = NULL;
    }
}

/* Network-backed embedding generators compile only with an HTTP transport;
 * the dispatchers below gate on HAVE_CURL so a no-libcurl build degrades to
 * the same failure path these functions would return. */
#ifdef HAVE_CURL

static int generate_huggingface_embedding(GV_EmbeddingService *service,
                                         const char *text,
                                         size_t *embedding_dim,
                                         float **embedding);
static int generate_huggingface_embedding_batch(GV_EmbeddingService *service,
                                               const char **texts,
                                               size_t text_count,
                                               size_t **embedding_dims,
                                               float ***embeddings);
static int generate_google_embedding(GV_EmbeddingService *service,
                                    const char *text,
                                    size_t *embedding_dim,
                                    float **embedding);
static int generate_google_embedding_batch(GV_EmbeddingService *service,
                                          const char **texts,
                                          size_t text_count,
                                          size_t **embedding_dims,
                                          float ***embeddings);

/**
 * POST request_json to url with a JSON content-type and optional auth header,
 * returning the response body (gv_alloc'd, caller gv_free) in *out_body.
 * Returns 0 on transport success, -1 otherwise.
 */
static int embedding_http_post(GV_EmbeddingService *service, const char *url,
                               const char *request_json, const char *auth_header,
                               char **out_body) {
    const char *hdrs[2];
    size_t num_hdrs = 0;
    hdrs[num_hdrs++] = "Content-Type: application/json";
    if (auth_header) hdrs[num_hdrs++] = auth_header;

    GV_HttpRequest req = {
        .method = GV_HTTP_POST,
        .url = url,
        .headers = hdrs,
        .num_headers = num_hdrs,
        .body = request_json,
        .body_len = strlen(request_json),
        .timeout_ms = (long)(service->config.timeout_seconds > 0 ?
                             service->config.timeout_seconds : DEFAULT_TIMEOUT) * 1000L,
        .verify_tls = 1,
        .max_response_bytes = MAX_RESPONSE_SIZE,
    };
    GV_HttpResponse resp;
    if (gv_http_perform(&req, &resp) != GV_HTTP_OK) return -1;
    *out_body = resp.body;
    return 0;
}

static int parse_openai_embedding_response(const char *json, float **embedding, size_t *dim) {

    const char *embedding_start = strstr(json, "\"embedding\":");
    if (embedding_start == NULL) {
        return -1;
    }
    embedding_start += 12;
    while (*embedding_start == ' ' || *embedding_start == '\t' ||
           *embedding_start == '\n' || *embedding_start == '\r') embedding_start++;
    if (*embedding_start != '[') {
        return -1;
    }
    embedding_start++;
    
    size_t count = 0;
    const char *p = embedding_start;
    while (*p && *p != ']') {
        if (*p == ',') count++;
        p++;
    }
    /* Reject an empty array "[]": the count++ below assumes a trailing element
     * after the last comma, which for empty content would fabricate a 1-element
     * vector with an unwritten (uninitialized) float and report success. */
    int has_content = 0;
    for (const char *q = embedding_start; *q && *q != ']'; q++) {
        if (*q != ' ' && *q != '\t' && *q != '\n' && *q != '\r') { has_content = 1; break; }
    }
    if (!has_content) {
        return -1;
    }
    count++;

    if (count == 0) {
        return -1;
    }
    
    *dim = count;
    *embedding = (float *)gv_alloc(count * sizeof(float));
    if (*embedding == NULL) {
        return -1;
    }
    
    p = embedding_start;
    size_t idx = 0;
    char *endptr;
    while (*p && *p != ']' && idx < count) {
        if (*p == ',' || *p == ' ') {
            p++;
            continue;
        }
        (*embedding)[idx++] = strtof(p, &endptr);
        p = endptr;
        while (*p && (*p == ',' || *p == ' ')) p++;
    }
    
    return 0;
}


static int generate_openai_embedding(GV_EmbeddingService *service,
                                     const char *text,
                                     size_t *embedding_dim,
                                     float **embedding) {
    char escaped_text[3072];
    gv_json_escape(escaped_text, sizeof(escaped_text), text);

    char request_json[4096];
    const char *model = service->config.model ? service->config.model : "text-embedding-3-small";
    int dim = service->config.embedding_dimension > 0 ? (int)service->config.embedding_dimension : 0;

    if (dim > 0) {
        snprintf(request_json, sizeof(request_json),
                "{\"input\":\"%s\",\"model\":\"%s\",\"dimensions\":%d}",
                escaped_text, model, dim);
    } else {
        snprintf(request_json, sizeof(request_json),
                "{\"input\":\"%s\",\"model\":\"%s\"}",
                escaped_text, model);
    }

    const char *url = service->config.base_url ? service->config.base_url : "https://api.openai.com/v1/embeddings";

    char auth_buf[1024];
    snprintf(auth_buf, sizeof(auth_buf), "Authorization: Bearer %s", service->config.api_key);

    char *body = NULL;
    if (embedding_http_post(service, url, request_json, auth_buf, &body) != 0) {
        return -1;
    }

    int result = parse_openai_embedding_response(body, embedding, embedding_dim);
    gv_free(body);

    return result;
}

static int parse_google_embedding_response(const char *json, float **embedding, size_t *dim) {
    
    const char *values_start = strstr(json, "\"values\"");
    if (values_start == NULL) {
        return -1;
    }
    
    values_start = strchr(values_start, '[');
    if (values_start == NULL) {
        return -1;
    }
    values_start++;
    size_t count = 0;
    const char *p = values_start;
    while (*p && *p != ']') {
        if (*p == ',') count++;
        p++;
    }
    count++;

    if (count == 0) {
        return -1;
    }

    *dim = count;
    *embedding = (float *)gv_alloc(count * sizeof(float));
    if (*embedding == NULL) {
        return -1;
    }

    p = values_start;
    size_t idx = 0;
    char *endptr;
    while (*p && *p != ']' && idx < count) {
        if (*p == ',' || *p == ' ') {
            p++;
            continue;
        }
        (*embedding)[idx++] = strtof(p, &endptr);
        p = endptr;
        while (*p && (*p == ',' || *p == ' ')) p++;
    }
    
    return 0;
}

static int generate_google_embedding(GV_EmbeddingService *service,
                                    const char *text,
                                    size_t *embedding_dim,
                                    float **embedding) {
    const char *model = service->config.model ? service->config.model : "text-embedding-004";
    int dim = service->config.embedding_dimension > 0 ? (int)service->config.embedding_dimension : 768;
    char request_json[16384];
    size_t json_pos = 0;

    /* Google API expects model in format "models/{model}" */
    if (strncmp(model, "models/", 7) != 0) {
        json_pos += snprintf(request_json + json_pos, sizeof(request_json) - json_pos,
                            "{\"model\":\"models/%s\",\"content\":{\"parts\":[{\"text\":\"", model);
    } else {
        json_pos += snprintf(request_json + json_pos, sizeof(request_json) - json_pos,
                            "{\"model\":\"%s\",\"content\":{\"parts\":[{\"text\":\"", model);
    }
    
    const char *p = text;
    while (*p && json_pos < sizeof(request_json) - 10) {
        if (*p == '"') {
            json_pos += snprintf(request_json + json_pos, sizeof(request_json) - json_pos, "\\\"");
        } else if (*p == '\\') {
            json_pos += snprintf(request_json + json_pos, sizeof(request_json) - json_pos, "\\\\");
        } else if (*p == '\n') {
            json_pos += snprintf(request_json + json_pos, sizeof(request_json) - json_pos, " ");
        } else if (*p == '\r') {
            /* Skip carriage return */
        } else if (*p == '\t') {
            json_pos += snprintf(request_json + json_pos, sizeof(request_json) - json_pos, " ");
        } else {
            request_json[json_pos++] = *p;
        }
        p++;
    }
    
    json_pos += snprintf(request_json + json_pos, sizeof(request_json) - json_pos,
                        "\"}]},\"outputDimensionality\":%d}", dim);
    
    char url[2048];
    if (service->config.base_url) {
        snprintf(url, sizeof(url), "%s", service->config.base_url);
    } else {
        if (strncmp(model, "models/", 7) == 0) {
            snprintf(url, sizeof(url), 
                    "https://generativelanguage.googleapis.com/v1beta/%s:embedContent",
                    model);
        } else {
            snprintf(url, sizeof(url), 
                    "https://generativelanguage.googleapis.com/v1beta/models/%s:embedContent",
                    model);
        }
    }
    
    char auth_buf[1024];
    snprintf(auth_buf, sizeof(auth_buf), "x-goog-api-key: %s", service->config.api_key);

    char *body = NULL;
    if (embedding_http_post(service, url, request_json, auth_buf, &body) != 0) {
        return -1;
    }

    int result = parse_google_embedding_response(body, embedding, embedding_dim);
    gv_free(body);

    return result;
}

static int generate_google_embedding_batch(GV_EmbeddingService *service,
                                          const char **texts,
                                          size_t text_count,
                                          size_t **embedding_dims,
                                          float ***embeddings) {
    /* Google batch API uses batchEmbedContents endpoint */
    const char *model = service->config.model ? service->config.model : "text-embedding-004";
    int dim = service->config.embedding_dimension > 0 ? (int)service->config.embedding_dimension : 768;
    
    /* Build request JSON with array of requests */
    size_t json_size = 4096 + text_count * 512;
    char *request_json = (char *)gv_alloc(json_size);
    if (request_json == NULL) {
        return -1;
    }
    
    memcpy(request_json, "{\"requests\":[", 14);
    size_t pos = 13;
    
    /* Google API expects model in format "models/{model}" */
    const char *model_prefix = (strncmp(model, "models/", 7) != 0) ? "models/" : "";
    
    for (size_t i = 0; i < text_count; i++) {
        if (i > 0) {
            request_json[pos++] = ',';
        }
        
        pos += snprintf(request_json + pos, json_size - pos,
                       "{\"model\":\"%s%s\",\"content\":{\"parts\":[{\"text\":\"", model_prefix, model);
        const char *p = texts[i];
        while (*p && pos < json_size - 50) {
            if (*p == '"') {
                pos += snprintf(request_json + pos, json_size - pos, "\\\"");
            } else if (*p == '\\') {
                pos += snprintf(request_json + pos, json_size - pos, "\\\\");
            } else if (*p == '\n') {
                pos += snprintf(request_json + pos, json_size - pos, " ");
            } else if (*p == '\r') {
                /* Skip carriage return */
            } else if (*p == '\t') {
                pos += snprintf(request_json + pos, json_size - pos, " ");
            } else {
                request_json[pos++] = *p;
            }
            p++;
        }
        
        pos += snprintf(request_json + pos, json_size - pos,
                       "\"}]},\"outputDimensionality\":%d}", dim);
        
        if (pos >= json_size - 10) {
            gv_free(request_json);
            return -1;
        }
    }
    
    snprintf(request_json + pos, json_size - pos, "]}");
    
    char url[2048];
    if (service->config.base_url) {
        snprintf(url, sizeof(url), "%s", service->config.base_url);
    } else {
        if (strncmp(model, "models/", 7) == 0) {
            snprintf(url, sizeof(url), 
                    "https://generativelanguage.googleapis.com/v1beta/%s:batchEmbedContents",
                    model);
        } else {
            snprintf(url, sizeof(url), 
                    "https://generativelanguage.googleapis.com/v1beta/models/%s:batchEmbedContents",
                    model);
        }
    }
    
    char auth_buf[1024];
    snprintf(auth_buf, sizeof(auth_buf), "x-goog-api-key: %s", service->config.api_key);

    char *body = NULL;
    int post_rc = embedding_http_post(service, url, request_json, auth_buf, &body);
    gv_free(request_json);
    if (post_rc != 0) {
        return -1;
    }

    /* Parse batch response - Google format: {"embeddings": [{"values": [...]}, {"values": [...]}, ...]} */
    const char *embeddings_start = strstr(body, "\"embeddings\"");
    if (embeddings_start == NULL) {
        gv_free(body);
        return -1;
    }

    embeddings_start = strchr(embeddings_start, '[');
    if (embeddings_start == NULL) {
        gv_free(body);
        return -1;
    }
    embeddings_start++;

    size_t success_count = 0;
    const char *p = embeddings_start;

    for (size_t i = 0; i < text_count; i++) {
        const char *values_start = strstr(p, "\"values\"");
        if (values_start == NULL) {
            break;
        }

        values_start = strchr(values_start, '[');
        if (values_start == NULL) {
            break;
        }
        values_start++;

        const char *values_end = strchr(values_start, ']');
        if (values_end == NULL) {
            break;
        }

        size_t count = 0;
        const char *q = values_start;
        while (q < values_end) {
            if (*q == ',') count++;
            q++;
        }
        count++;

        if (count > 0) {
            (*embedding_dims)[i] = count;
            (*embeddings)[i] = (float *)gv_alloc(count * sizeof(float));
            if ((*embeddings)[i] != NULL) {
                q = values_start;
                size_t idx = 0;
                char *endptr;
                while (q < values_end && idx < count) {
                    if (*q == ',' || *q == ' ' || *q == '\n' || *q == '\r' || *q == '\t') {
                        q++;
                        continue;
                    }
                    (*embeddings)[i][idx++] = strtof(q, &endptr);
                    if (endptr == q) {
                        break;
                    }
                    q = endptr;
                    while (q < values_end && (*q == ',' || *q == ' ' || *q == '\n' || *q == '\r' || *q == '\t')) {
                        q++;
                    }
                }
                
                if (idx == count) {
                    success_count++;
                } else {
                    gv_free((*embeddings)[i]);
                    (*embeddings)[i] = NULL;
                    (*embedding_dims)[i] = 0;
                }
            }
        }
        
        p = values_end + 1;
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
            p++;
        }
        if (*p == '}') {
            p++;
            while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) {
                p++;
            }
        } else {
            p = strchr(p, '}');
            if (p) {
                p++;
                while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) {
                    p++;
                }
            } else {
                break;  /* No more objects */
            }
        }
        
        /* Check if we've reached the end of the embeddings array */
        if (*p == '\0' || *p == ']') {
            break;
        }
    }

    gv_free(body);
    return (int)success_count;
}

static int generate_huggingface_embedding(GV_EmbeddingService *service,
                                         const char *text,
                                         size_t *embedding_dim,
                                         float **embedding) {
    char request_json[4096];
    snprintf(request_json, sizeof(request_json),
            "{\"input\":\"%s\"}", text);
    
    const char *url = service->config.base_url ?
                     service->config.base_url :
                     "http://localhost:3000/v1/embeddings";
    char full_url[2048];
    if (strstr(url, "/embeddings") == NULL) {
        snprintf(full_url, sizeof(full_url), "%s/embeddings", url);
        url = full_url;
    }
    
    char auth_buf[1024];
    const char *auth_header = NULL;
    if (service->config.api_key) {
        snprintf(auth_buf, sizeof(auth_buf), "Authorization: Bearer %s", service->config.api_key);
        auth_header = auth_buf;
    }

    char *body = NULL;
    if (embedding_http_post(service, url, request_json, auth_header, &body) != 0) {
        return -1;
    }

    int result = parse_openai_embedding_response(body, embedding, embedding_dim);
    gv_free(body);

    return result;
}

/* Generate batch embeddings using HuggingFace TEI API */
static int generate_huggingface_embedding_batch(GV_EmbeddingService *service,
                                               const char **texts,
                                               size_t text_count,
                                               size_t **embedding_dims,
                                               float ***embeddings) {
    size_t json_size = 4096 + text_count * 256;
    char *request_json = (char *)gv_alloc(json_size);
    if (request_json == NULL) {
        return -1;
    }

    memcpy(request_json, "{\"input\":[", 11);
    size_t pos = 10;

    for (size_t i = 0; i < text_count; i++) {
        if (i > 0) {
            request_json[pos++] = ',';
        }
        request_json[pos++] = '"';
        const char *p = texts[i];
        while (*p && pos < json_size - 10) {
            if (*p == '"' || *p == '\\') {
                request_json[pos++] = '\\';
            }
            request_json[pos++] = *p++;
        }
        request_json[pos++] = '"';
    }

    snprintf(request_json + pos, json_size - pos, "]}");

    const char *url = service->config.base_url ?
                     service->config.base_url :
                     "http://localhost:3000/v1/embeddings";
    char full_url[2048];
    if (strstr(url, "/embeddings") == NULL) {
        snprintf(full_url, sizeof(full_url), "%s/embeddings", url);
        url = full_url;
    }

    char auth_buf[1024];
    const char *auth_header = NULL;
    if (service->config.api_key) {
        snprintf(auth_buf, sizeof(auth_buf), "Authorization: Bearer %s", service->config.api_key);
        auth_header = auth_buf;
    }

    char *body = NULL;
    int post_rc = embedding_http_post(service, url, request_json, auth_header, &body);
    gv_free(request_json);
    if (post_rc != 0) {
        return -1;
    }

    const char *data_start = strstr(body, "\"data\":[");
    if (data_start == NULL) {
        gv_free(body);
        return -1;
    }

    data_start += 8;

    size_t success_count = 0;
    const char *p = data_start;

    for (size_t i = 0; i < text_count && *p && *p != ']'; i++) {
        const char *emb_start = strstr(p, "\"embedding\":[");
        if (emb_start == NULL) {
            break;
        }
        
        emb_start += 13;

        size_t count = 0;
        const char *q = emb_start;
        while (*q && *q != ']') {
            if (*q == ',') count++;
            q++;
        }
        count++;

        if (count > 0) {
            (*embedding_dims)[i] = count;
            (*embeddings)[i] = (float *)gv_alloc(count * sizeof(float));
            if ((*embeddings)[i] != NULL) {
                q = emb_start;
                size_t idx = 0;
                char *endptr;
                while (*q && *q != ']' && idx < count) {
                    if (*q == ',' || *q == ' ') {
                        q++;
                        continue;
                    }
                    (*embeddings)[i][idx++] = strtof(q, &endptr);
                    q = endptr;
                    while (*q && (*q == ',' || *q == ' ')) q++;
                }
                success_count++;
            }
        }

        p = strchr(emb_start, '}');
        if (p) p++;
    }

    gv_free(body);
    return (int)success_count;
}

static int generate_openai_embedding_batch(GV_EmbeddingService *service,
                                           const char **texts,
                                           size_t text_count,
                                           size_t **embedding_dims,
                                           float ***embeddings) {
    size_t json_size = 4096 + text_count * 256;
    char *request_json = (char *)gv_alloc(json_size);
    if (request_json == NULL) {
        return -1;
    }

    const char *model = service->config.model ? service->config.model : "text-embedding-3-small";
    int dim = service->config.embedding_dimension > 0 ? (int)service->config.embedding_dimension : 0;

    memcpy(request_json, "{\"input\":[", 11);
    size_t pos = 10;

    for (size_t i = 0; i < text_count; i++) {
        if (i > 0) {
            request_json[pos++] = ',';
        }
        request_json[pos++] = '"';
        const char *p = texts[i];
        while (*p && pos < json_size - 10) {
            if (*p == '"' || *p == '\\') {
                request_json[pos++] = '\\';
            }
            request_json[pos++] = *p++;
        }
        request_json[pos++] = '"';
    }
    
    if (dim > 0) {
        snprintf(request_json + pos, json_size - pos, "],\"model\":\"%s\",\"dimensions\":%d}", model, dim);
    } else {
        snprintf(request_json + pos, json_size - pos, "],\"model\":\"%s\"}", model);
    }
    
    const char *url = service->config.base_url ? service->config.base_url : "https://api.openai.com/v1/embeddings";

    char auth_buf[1024];
    snprintf(auth_buf, sizeof(auth_buf), "Authorization: Bearer %s", service->config.api_key);

    char *body = NULL;
    int post_rc = embedding_http_post(service, url, request_json, auth_buf, &body);
    gv_free(request_json);
    if (post_rc != 0) {
        return -1;
    }

    const char *data_start = strstr(body, "\"data\":[");
    if (data_start == NULL) {
        gv_free(body);
        return -1;
    }

    data_start += 8;

    size_t success_count = 0;
    const char *p = data_start;

    for (size_t i = 0; i < text_count && *p && *p != ']'; i++) {
        const char *emb_start = strstr(p, "\"embedding\":[");
        if (emb_start == NULL) {
            break;
        }

        emb_start += 13;

        size_t count = 0;
        const char *q = emb_start;
        while (*q && *q != ']') {
            if (*q == ',') count++;
            q++;
        }
        count++;

        if (count > 0) {
            (*embedding_dims)[i] = count;
            (*embeddings)[i] = (float *)gv_alloc(count * sizeof(float));
            if ((*embeddings)[i] != NULL) {
                q = emb_start;
                size_t idx = 0;
                char *endptr;
                while (*q && *q != ']' && idx < count) {
                    if (*q == ',' || *q == ' ') {
                        q++;
                        continue;
                    }
                    (*embeddings)[i][idx++] = strtof(q, &endptr);
                    q = endptr;
                    while (*q && (*q == ',' || *q == ' ')) q++;
                }
                success_count++;
            }
        }

        p = strchr(emb_start, '}');
        if (p) p++;
    }

    gv_free(body);
    return (int)success_count;
}

#endif /* HAVE_CURL */

GV_EmbeddingService *embedding_service_create(const GV_EmbeddingConfig *config) {
    if (config == NULL || config->provider == GV_EMBEDDING_PROVIDER_NONE) {
        return NULL;
    }
    
    GV_EmbeddingService *service = (GV_EmbeddingService *)gv_calloc(1, sizeof(GV_EmbeddingService));
    if (service == NULL) {
        return NULL;
    }
    
    service->config = *config;
    service->last_error = NULL;
    
    if (config->api_key) {
        service->config.api_key = gv_dup_cstr(config->api_key);
    }
    if (config->model) {
        service->config.model = gv_dup_cstr(config->model);
    }
    if (config->base_url) {
        service->config.base_url = gv_dup_cstr(config->base_url);
    }
    if (config->huggingface_model_path) {
        service->config.huggingface_model_path = gv_dup_cstr(config->huggingface_model_path);
    }
    
    if (config->enable_cache) {
        service->cache = embedding_cache_create(config->cache_size);
    }
    
    /* Providers that reach the network require an HTTP transport. */
    if ((config->provider == GV_EMBEDDING_PROVIDER_OPENAI ||
         config->provider == GV_EMBEDDING_PROVIDER_CUSTOM ||
         config->provider == GV_EMBEDDING_PROVIDER_GOOGLE) &&
        !gv_http_available()) {
        embedding_cache_destroy(service->cache);
        embedding_config_free(&service->config);
        gv_free(service);
        return NULL;
    }

    return service;
}

void embedding_service_destroy(GV_EmbeddingService *service) {
    if (service == NULL) {
        return;
    }
    
    embedding_cache_destroy(service->cache);

    embedding_config_free(&service->config);
    if (service->last_error) {
        gv_free(service->last_error);
    }
    gv_free(service);
}

int embedding_generate(GV_EmbeddingService *service,
                          const char *text,
                          size_t *embedding_dim,
                          float **embedding) {
    if (service == NULL || text == NULL || embedding_dim == NULL || embedding == NULL) {
        return -1;
    }
    
    if (service->cache) {
        const float *cached_embedding = NULL;
        size_t cached_dim = 0;
        if (embedding_cache_get(service->cache, text, &cached_dim, &cached_embedding) == 1) {
            *embedding_dim = cached_dim;
            *embedding = (float *)gv_alloc(cached_dim * sizeof(float));
            if (*embedding == NULL) {
                return -1;
            }
            memcpy(*embedding, cached_embedding, cached_dim * sizeof(float));
            return 0;
        }
    }

    int result = -1;
    
#ifdef HAVE_CURL
    if (service->config.provider == GV_EMBEDDING_PROVIDER_OPENAI ||
        service->config.provider == GV_EMBEDDING_PROVIDER_CUSTOM) {
        result = generate_openai_embedding(service, text, embedding_dim, embedding);
    } else if (service->config.provider == GV_EMBEDDING_PROVIDER_GOOGLE) {
        result = generate_google_embedding(service, text, embedding_dim, embedding);
    } else if (service->config.provider == GV_EMBEDDING_PROVIDER_HUGGINGFACE) {
        result = generate_huggingface_embedding(service, text, embedding_dim, embedding);
    }
#endif
    
    if (result == 0 && service->cache && *embedding != NULL) {
        embedding_cache_put(service->cache, text, *embedding_dim, *embedding);
    }

    return result;
}

int embedding_generate_batch(GV_EmbeddingService *service,
                                const char **texts,
                                size_t text_count,
                                size_t **embedding_dims,
                                float ***embeddings) {
    if (service == NULL || texts == NULL || text_count == 0 ||
        embedding_dims == NULL || embeddings == NULL) {
        return -1;
    }
    
    *embedding_dims = (size_t *)gv_alloc(text_count * sizeof(size_t));
    *embeddings = (float **)gv_alloc(text_count * sizeof(float *));
    if (*embedding_dims == NULL || *embeddings == NULL) {
        if (*embedding_dims) gv_free(*embedding_dims);
        if (*embeddings) gv_free(*embeddings);
        return -1;
    }
    
    size_t success_count = 0;
    
#ifdef HAVE_CURL
    if (service->config.provider == GV_EMBEDDING_PROVIDER_OPENAI ||
        service->config.provider == GV_EMBEDDING_PROVIDER_CUSTOM) {
        int batch_result = generate_openai_embedding_batch(service, texts, text_count,
                                                          embedding_dims, embeddings);
        /* The batch call returns the count parsed (>=0) on success, -1 on error
         * - never 0-means-success. The old `== 0` check therefore treated every
         * successful batch as failure (re-running per item, leaking the batch's
         * already-allocated vectors) and treated a 0-count as full success with
         * uninitialized outputs. Mirror the Google branch's >= 0 convention. */
        if (batch_result >= 0) {
            success_count = (size_t)batch_result;
        } else {
            for (size_t i = 0; i < text_count; i++) {
                if (embedding_generate(service, texts[i], &(*embedding_dims)[i],
                                         &(*embeddings)[i]) == 0) {
                    success_count++;
                } else {
                    (*embedding_dims)[i] = 0;
                    (*embeddings)[i] = NULL;
                }
            }
        }
    } else if (service->config.provider == GV_EMBEDDING_PROVIDER_GOOGLE) {
        int batch_result = generate_google_embedding_batch(service, texts, text_count,
                                                          embedding_dims, embeddings);
        if (batch_result >= 0) {
            success_count = batch_result;
        } else {
            for (size_t i = 0; i < text_count; i++) {
                if (embedding_generate(service, texts[i], &(*embedding_dims)[i],
                                         &(*embeddings)[i]) == 0) {
                    success_count++;
                } else {
                    (*embedding_dims)[i] = 0;
                    (*embeddings)[i] = NULL;
                }
            }
        }
    }
#endif
    
    if (service->config.provider == GV_EMBEDDING_PROVIDER_HUGGINGFACE) {
#ifdef HAVE_CURL
        int batch_result = generate_huggingface_embedding_batch(service, texts, text_count,
                                                                embedding_dims, embeddings);
        if (batch_result >= 0) {
            success_count = batch_result;
        } else {
            for (size_t i = 0; i < text_count; i++) {
                if (embedding_generate(service, texts[i], &(*embedding_dims)[i],
                                         &(*embeddings)[i]) == 0) {
                    success_count++;
                } else {
                    (*embedding_dims)[i] = 0;
                    (*embeddings)[i] = NULL;
                }
            }
        }
#else
        for (size_t i = 0; i < text_count; i++) {
            (*embedding_dims)[i] = 0;
            (*embeddings)[i] = NULL;
        }
        return -1;
#endif
    }

    return (int)success_count;
}

