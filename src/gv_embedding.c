/**
 * @file gv_embedding.c
 * @brief Embedding service implementation with support for multiple providers,
 * batch processing, and caching.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

#include "gigavector/gv_embedding.h"

#define MAX_RESPONSE_SIZE (10 * 1024 * 1024)  // 10MB for batch responses
#define DEFAULT_BATCH_SIZE 100
#define DEFAULT_TIMEOUT 30
#define MAX_CACHE_KEY_LEN 1024

/* Embedding cache entry */
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

/* Embedding cache structure */
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

/* Embedding service structure */
struct GV_EmbeddingService {
    GV_EmbeddingConfig config;
    GV_EmbeddingCache *cache;
#ifdef HAVE_CURL
    void *curl_handle;            /* CURL handle for HTTP requests */
#endif
    char *last_error;             /* Last error message */
};

/* Hash function for cache keys */
static size_t hash_string(const char *str, size_t bucket_count) {
    size_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % bucket_count;
}

/* Create embedding cache */
GV_EmbeddingCache *gv_embedding_cache_create(size_t max_size) {
    GV_EmbeddingCache *cache = (GV_EmbeddingCache *)calloc(1, sizeof(GV_EmbeddingCache));
    if (cache == NULL) {
        return NULL;
    }
    
    cache->bucket_count = 1024;  /* Fixed bucket count */
    cache->max_size = max_size;
    cache->current_size = 0;
    cache->hits = 0;
    cache->misses = 0;
    cache->lru_head = NULL;
    cache->lru_tail = NULL;
    
    cache->buckets = (CacheEntry **)calloc(cache->bucket_count, sizeof(CacheEntry *));
    if (cache->buckets == NULL) {
        free(cache);
        return NULL;
    }
    
    if (pthread_mutex_init(&cache->mutex, NULL) != 0) {
        free(cache->buckets);
        free(cache);
        return NULL;
    }
    
    return cache;
}

/* Remove entry from LRU list */
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

/* Add entry to LRU head */
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

/* Evict least recently used entry */
static void evict_lru(GV_EmbeddingCache *cache) {
    if (cache->lru_tail == NULL) {
        return;
    }
    
    CacheEntry *entry = cache->lru_tail;
    size_t hash = hash_string(entry->key, cache->bucket_count);
    
    /* Remove from hash table */
    CacheEntry **prev = &cache->buckets[hash];
    while (*prev != entry) {
        prev = &(*prev)->next;
    }
    *prev = entry->next;
    
    /* Remove from LRU */
    remove_from_lru(cache, entry);
    
    /* Free entry */
    free(entry->key);
    free(entry->embedding);
    free(entry);
    cache->current_size--;
}

/* Get embedding from cache */
int gv_embedding_cache_get(GV_EmbeddingCache *cache,
                          const char *text,
                          size_t *embedding_dim,
                          const float **embedding) {
    if (cache == NULL || text == NULL || embedding_dim == NULL || embedding == NULL) {
        return -1;
    }
    
    pthread_mutex_lock(&cache->mutex);
    
    size_t hash = hash_string(text, cache->bucket_count);
    CacheEntry *entry = cache->buckets[hash];
    
    while (entry != NULL) {
        if (strcmp(entry->key, text) == 0) {
            /* Found - move to head of LRU */
            remove_from_lru(cache, entry);
            add_to_lru_head(cache, entry);
            entry->access_count++;
            cache->hits++;
            
            *embedding_dim = entry->embedding_dim;
            *embedding = entry->embedding;
            
            pthread_mutex_unlock(&cache->mutex);
            return 1;  /* Found */
        }
        entry = entry->next;
    }
    
    cache->misses++;
    pthread_mutex_unlock(&cache->mutex);
    return 0;  /* Not found */
}

/* Store embedding in cache */
int gv_embedding_cache_put(GV_EmbeddingCache *cache,
                           const char *text,
                           size_t embedding_dim,
                           const float *embedding) {
    if (cache == NULL || text == NULL || embedding == NULL || embedding_dim == 0) {
        return -1;
    }
    
    pthread_mutex_lock(&cache->mutex);
    
    /* Check if entry exists */
    size_t hash = hash_string(text, cache->bucket_count);
    CacheEntry *entry = cache->buckets[hash];
    
    while (entry != NULL) {
        if (strcmp(entry->key, text) == 0) {
            /* Update existing entry */
            if (entry->embedding_dim != embedding_dim) {
                free(entry->embedding);
                entry->embedding = (float *)malloc(embedding_dim * sizeof(float));
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
    
    /* Evict if at capacity */
    if (cache->max_size > 0 && cache->current_size >= cache->max_size) {
        evict_lru(cache);
    }
    
    /* Create new entry */
    entry = (CacheEntry *)calloc(1, sizeof(CacheEntry));
    if (entry == NULL) {
        pthread_mutex_unlock(&cache->mutex);
        return -1;
    }
    
    entry->key = strdup(text);
    if (entry->key == NULL) {
        free(entry);
        pthread_mutex_unlock(&cache->mutex);
        return -1;
    }
    
    entry->embedding = (float *)malloc(embedding_dim * sizeof(float));
    if (entry->embedding == NULL) {
        free(entry->key);
        free(entry);
        pthread_mutex_unlock(&cache->mutex);
        return -1;
    }
    
    memcpy(entry->embedding, embedding, embedding_dim * sizeof(float));
    entry->embedding_dim = embedding_dim;
    entry->timestamp = time(NULL);
    entry->access_count = 1;
    
    /* Add to hash table */
    entry->next = cache->buckets[hash];
    cache->buckets[hash] = entry;
    
    /* Add to LRU head */
    add_to_lru_head(cache, entry);
    
    cache->current_size++;
    pthread_mutex_unlock(&cache->mutex);
    return 0;
}

/* Clear embedding cache */
void gv_embedding_cache_clear(GV_EmbeddingCache *cache) {
    if (cache == NULL) {
        return;
    }
    
    pthread_mutex_lock(&cache->mutex);
    
    for (size_t i = 0; i < cache->bucket_count; i++) {
        CacheEntry *entry = cache->buckets[i];
        while (entry != NULL) {
            CacheEntry *next = entry->next;
            free(entry->key);
            free(entry->embedding);
            free(entry);
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

/* Get cache statistics */
void gv_embedding_cache_stats(GV_EmbeddingCache *cache,
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

/* Destroy embedding cache */
void gv_embedding_cache_destroy(GV_EmbeddingCache *cache) {
    if (cache == NULL) {
        return;
    }
    
    gv_embedding_cache_clear(cache);
    free(cache->buckets);
    pthread_mutex_destroy(&cache->mutex);
    free(cache);
}

/* Default embedding configuration */
GV_EmbeddingConfig gv_embedding_config_default(void) {
    GV_EmbeddingConfig config;
    memset(&config, 0, sizeof(config));
    config.provider = GV_EMBEDDING_PROVIDER_NONE;
    config.embedding_dimension = 0;
    config.batch_size = DEFAULT_BATCH_SIZE;
    config.enable_cache = 1;
    config.cache_size = 1000;  /* Default: 1000 entries */
    config.timeout_seconds = DEFAULT_TIMEOUT;
    return config;
}

/* Free embedding configuration */
void gv_embedding_config_free(GV_EmbeddingConfig *config) {
    if (config == NULL) {
        return;
    }
    if (config->api_key) {
        free(config->api_key);
        config->api_key = NULL;
    }
    if (config->model) {
        free(config->model);
        config->model = NULL;
    }
    if (config->base_url) {
        free(config->base_url);
        config->base_url = NULL;
    }
    if (config->huggingface_model_path) {
        free(config->huggingface_model_path);
        config->huggingface_model_path = NULL;
    }
}

#ifdef HAVE_CURL

/* Response buffer for CURL */
struct ResponseBuffer {
    char *data;
    size_t size;
    size_t capacity;
};

static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct ResponseBuffer *buf = (struct ResponseBuffer *)userp;
    
    if (buf->size + realsize > MAX_RESPONSE_SIZE) {
        return 0;
    }
    
    if (buf->size + realsize >= buf->capacity) {
        size_t new_capacity = buf->capacity * 2;
        if (new_capacity < buf->size + realsize + 1) {
            new_capacity = buf->size + realsize + 1;
        }
        if (new_capacity > MAX_RESPONSE_SIZE) {
            new_capacity = MAX_RESPONSE_SIZE;
        }
        char *new_data = (char *)realloc(buf->data, new_capacity);
        if (new_data == NULL) {
            return 0;
        }
        buf->data = new_data;
        buf->capacity = new_capacity;
    }
    
    memcpy(buf->data + buf->size, contents, realsize);
    buf->size += realsize;
    buf->data[buf->size] = '\0';
    
    return realsize;
}

/* Forward declarations */
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

/* Parse OpenAI embedding response */
static int parse_openai_embedding_response(const char *json, float **embedding, size_t *dim) {
    
    const char *embedding_start = strstr(json, "\"embedding\":[");
    if (embedding_start == NULL) {
        return -1;
    }
    
    embedding_start += 13;
    
    size_t count = 0;
    const char *p = embedding_start;
    while (*p && *p != ']') {
        if (*p == ',') count++;
        p++;
    }
    count++;
    
    if (count == 0) {
        return -1;
    }
    
    *dim = count;
    *embedding = (float *)malloc(count * sizeof(float));
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

/* Generate embedding using OpenAI API */
static int generate_openai_embedding(GV_EmbeddingService *service,
                                     const char *text,
                                     size_t *embedding_dim,
                                     float **embedding) {
    CURL *curl = (CURL *)service->curl_handle;
    if (curl == NULL) {
        return -1;
    }
    
    /* Build request JSON */
    char request_json[4096];
    const char *model = service->config.model ? service->config.model : "text-embedding-3-small";
    int dim = service->config.embedding_dimension > 0 ? (int)service->config.embedding_dimension : 0;
    
    if (dim > 0) {
        snprintf(request_json, sizeof(request_json),
                "{\"input\":\"%s\",\"model\":\"%s\",\"dimensions\":%d}",
                text, model, dim);
    } else {
        snprintf(request_json, sizeof(request_json),
                "{\"input\":\"%s\",\"model\":\"%s\"}",
                text, model);
    }
    
    const char *url = service->config.base_url ? service->config.base_url : "https://api.openai.com/v1/embeddings";
    
    struct ResponseBuffer buf;
    buf.data = (char *)malloc(4096);
    buf.size = 0;
    buf.capacity = 4096;
    if (buf.data == NULL) {
        return -1;
    }
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    char auth_buf[1024];
    snprintf(auth_buf, sizeof(auth_buf), "Authorization: Bearer %s", service->config.api_key);
    headers = curl_slist_append(headers, auth_buf);
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, service->config.timeout_seconds > 0 ?
                     service->config.timeout_seconds : DEFAULT_TIMEOUT);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        free(buf.data);
        return -1;
    }
    
    /* Parse response */
    int result = parse_openai_embedding_response(buf.data, embedding, embedding_dim);
    free(buf.data);
    
    return result;
}

/* Parse Google embedding response */
static int parse_google_embedding_response(const char *json, float **embedding, size_t *dim) {
    /* Google API response format: {"embedding": {"values": [0.1, 0.2, ...]}} */
    
    const char *values_start = strstr(json, "\"values\"");
    if (values_start == NULL) {
        return -1;
    }
    
    /* Find the opening bracket after "values" */
    values_start = strchr(values_start, '[');
    if (values_start == NULL) {
        return -1;
    }
    values_start++;  /* Skip '[' */
    
    /* Count values */
    size_t count = 0;
    const char *p = values_start;
    while (*p && *p != ']') {
        if (*p == ',') count++;
        p++;
    }
    count++;  /* Last value */
    
    if (count == 0) {
        return -1;
    }
    
    *dim = count;
    *embedding = (float *)malloc(count * sizeof(float));
    if (*embedding == NULL) {
        return -1;
    }
    
    /* Parse values */
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

/* Generate embedding using Google Generative AI API */
static int generate_google_embedding(GV_EmbeddingService *service,
                                    const char *text,
                                    size_t *embedding_dim,
                                    float **embedding) {
#ifdef HAVE_CURL
    CURL *curl = (CURL *)service->curl_handle;
    if (curl == NULL) {
        return -1;
    }
    
    /* Clean and escape text for JSON */
    const char *model = service->config.model ? service->config.model : "text-embedding-004";
    int dim = service->config.embedding_dimension > 0 ? (int)service->config.embedding_dimension : 768;
    
    /* Build request JSON with proper escaping */
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
    
    /* Escape text */
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
    
    /* Build URL - Google API uses models/{model}:embedContent format */
    char url[2048];
    if (service->config.base_url) {
        snprintf(url, sizeof(url), "%s", service->config.base_url);
    } else {
        /* If model already starts with "models/", use as-is, otherwise prepend "models/" */
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
    
    struct ResponseBuffer buf;
    buf.data = (char *)malloc(4096);
    buf.size = 0;
    buf.capacity = 4096;
    if (buf.data == NULL) {
        return -1;
    }
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    /* Google uses x-goog-api-key header */
    char auth_buf[1024];
    snprintf(auth_buf, sizeof(auth_buf), "x-goog-api-key: %s", service->config.api_key);
    headers = curl_slist_append(headers, auth_buf);
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, service->config.timeout_seconds > 0 ?
                     service->config.timeout_seconds : DEFAULT_TIMEOUT);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        free(buf.data);
        return -1;
    }
    
    /* Parse response */
    int result = parse_google_embedding_response(buf.data, embedding, embedding_dim);
    free(buf.data);
    
    return result;
#else
    (void)service;
    (void)text;
    (void)embedding_dim;
    (void)embedding;
    return -1;
#endif
}

/* Generate batch embeddings using Google Generative AI API */
static int generate_google_embedding_batch(GV_EmbeddingService *service,
                                          const char **texts,
                                          size_t text_count,
                                          size_t **embedding_dims,
                                          float ***embeddings) {
#ifdef HAVE_CURL
    CURL *curl = (CURL *)service->curl_handle;
    if (curl == NULL) {
        return -1;
    }
    
    /* Google batch API uses batchEmbedContents endpoint */
    const char *model = service->config.model ? service->config.model : "text-embedding-004";
    int dim = service->config.embedding_dimension > 0 ? (int)service->config.embedding_dimension : 768;
    
    /* Build request JSON with array of requests */
    size_t json_size = 4096 + text_count * 512;
    char *request_json = (char *)malloc(json_size);
    if (request_json == NULL) {
        return -1;
    }
    
    strcpy(request_json, "{\"requests\":[");
    size_t pos = strlen(request_json);
    
    /* Google API expects model in format "models/{model}" */
    const char *model_prefix = (strncmp(model, "models/", 7) != 0) ? "models/" : "";
    
    for (size_t i = 0; i < text_count; i++) {
        if (i > 0) {
            request_json[pos++] = ',';
        }
        
        /* Start request object */
        pos += snprintf(request_json + pos, json_size - pos,
                       "{\"model\":\"%s%s\",\"content\":{\"parts\":[{\"text\":\"", model_prefix, model);
        
        /* Escape and append text */
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
        
        /* Close request object */
        pos += snprintf(request_json + pos, json_size - pos,
                       "\"}]},\"outputDimensionality\":%d}", dim);
        
        if (pos >= json_size - 10) {
            free(request_json);
            return -1;
        }
    }
    
    snprintf(request_json + pos, json_size - pos, "]}");
    
    /* Build URL - Google API uses models/{model}:batchEmbedContents format */
    char url[2048];
    if (service->config.base_url) {
        snprintf(url, sizeof(url), "%s", service->config.base_url);
    } else {
        /* If model already starts with "models/", use as-is, otherwise prepend "models/" */
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
    
    struct ResponseBuffer buf;
    buf.data = (char *)malloc(262144);  /* 256KB for batch responses */
    buf.size = 0;
    buf.capacity = 262144;
    if (buf.data == NULL) {
        free(request_json);
        return -1;
    }
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    char auth_buf[1024];
    snprintf(auth_buf, sizeof(auth_buf), "x-goog-api-key: %s", service->config.api_key);
    headers = curl_slist_append(headers, auth_buf);
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, service->config.timeout_seconds > 0 ?
                     service->config.timeout_seconds : DEFAULT_TIMEOUT);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    free(request_json);
    
    if (res != CURLE_OK) {
        free(buf.data);
        return -1;
    }
    
    /* Parse batch response - Google format: {"embeddings": [{"values": [...]}, {"values": [...]}, ...]} */
    const char *embeddings_start = strstr(buf.data, "\"embeddings\"");
    if (embeddings_start == NULL) {
        free(buf.data);
        return -1;
    }
    
    /* Find the opening bracket after "embeddings" */
    embeddings_start = strchr(embeddings_start, '[');
    if (embeddings_start == NULL) {
        free(buf.data);
        return -1;
    }
    embeddings_start++;  /* Skip '[' */
    
    /* Parse each embedding in the array */
    size_t success_count = 0;
    const char *p = embeddings_start;
    
    for (size_t i = 0; i < text_count; i++) {
        /* Find values array for this embedding */
        const char *values_start = strstr(p, "\"values\"");
        if (values_start == NULL) {
            /* No more embeddings found */
            break;
        }
        
        /* Find the opening bracket after "values" */
        values_start = strchr(values_start, '[');
        if (values_start == NULL) {
            break;
        }
        values_start++;  /* Skip '[' */
        
        /* Find the closing bracket to count values accurately */
        const char *values_end = strchr(values_start, ']');
        if (values_end == NULL) {
            break;
        }
        
        /* Count values */
        size_t count = 0;
        const char *q = values_start;
        while (q < values_end) {
            if (*q == ',') count++;
            q++;
        }
        count++;  /* Last value */
        
        if (count > 0) {
            (*embedding_dims)[i] = count;
            (*embeddings)[i] = (float *)malloc(count * sizeof(float));
            if ((*embeddings)[i] != NULL) {
                /* Parse values */
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
                        /* Failed to parse number */
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
                    /* Failed to parse all values */
                    free((*embeddings)[i]);
                    (*embeddings)[i] = NULL;
                    (*embedding_dims)[i] = 0;
                }
            }
        }
        
        /* Move to next embedding object - start from after the closing bracket of values */
        p = values_end + 1;  /* Skip ']' */
        /* Skip whitespace and find the closing brace of this embedding object */
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
            p++;
        }
        if (*p == '}') {
            p++;  /* Skip '}' */
            /* Skip whitespace and comma if present */
            while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) {
                p++;
            }
        } else {
            /* Try to find the closing brace */
            p = strchr(p, '}');
            if (p) {
                p++;  /* Skip '}' */
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
    
    free(buf.data);
    return (int)success_count;
#else
    (void)service;
    (void)texts;
    (void)text_count;
    (void)embedding_dims;
    (void)embeddings;
    return -1;
#endif
}

/* Generate embedding using HuggingFace Text Embeddings Inference (TEI) API */
static int generate_huggingface_embedding(GV_EmbeddingService *service,
                                         const char *text,
                                         size_t *embedding_dim,
                                         float **embedding) {
#ifdef HAVE_CURL
    CURL *curl = (CURL *)service->curl_handle;
    if (curl == NULL) {
        return -1;
    }
    
    /* Use TEI API format (OpenAI-compatible) */
    char request_json[4096];
    snprintf(request_json, sizeof(request_json),
            "{\"input\":\"%s\"}", text);
    
    /* Default to localhost TEI if base_url not set */
    const char *url = service->config.base_url ? 
                     service->config.base_url : 
                     "http://localhost:3000/v1/embeddings";
    
    /* Build full URL with /embeddings if needed */
    char full_url[2048];
    if (strstr(url, "/embeddings") == NULL) {
        snprintf(full_url, sizeof(full_url), "%s/embeddings", url);
        url = full_url;
    }
    
    struct ResponseBuffer buf;
    buf.data = (char *)malloc(4096);
    buf.size = 0;
    buf.capacity = 4096;
    if (buf.data == NULL) {
        return -1;
    }
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    /* TEI may not require auth, but support it if provided */
    if (service->config.api_key) {
        char auth_buf[1024];
        snprintf(auth_buf, sizeof(auth_buf), "Authorization: Bearer %s", service->config.api_key);
        headers = curl_slist_append(headers, auth_buf);
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, service->config.timeout_seconds > 0 ?
                     service->config.timeout_seconds : DEFAULT_TIMEOUT);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        free(buf.data);
        return -1;
    }
    
    /* Parse response (same format as OpenAI) */
    int result = parse_openai_embedding_response(buf.data, embedding, embedding_dim);
    free(buf.data);
    
    return result;
#else
    (void)service;
    (void)text;
    (void)embedding_dim;
    (void)embedding;
    return -1;
#endif
}

/* Generate batch embeddings using HuggingFace TEI API */
static int generate_huggingface_embedding_batch(GV_EmbeddingService *service,
                                               const char **texts,
                                               size_t text_count,
                                               size_t **embedding_dims,
                                               float ***embeddings) {
#ifdef HAVE_CURL
    CURL *curl = (CURL *)service->curl_handle;
    if (curl == NULL) {
        return -1;
    }
    
    /* Build request JSON with array of inputs */
    size_t json_size = 4096 + text_count * 256;
    char *request_json = (char *)malloc(json_size);
    if (request_json == NULL) {
        return -1;
    }
    
    strcpy(request_json, "{\"input\":[");
    size_t pos = strlen(request_json);
    
    for (size_t i = 0; i < text_count; i++) {
        if (i > 0) {
            request_json[pos++] = ',';
        }
        request_json[pos++] = '"';
        /* Escape text */
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
    
    /* Default to localhost TEI if base_url not set */
    const char *url = service->config.base_url ? 
                     service->config.base_url : 
                     "http://localhost:3000/v1/embeddings";
    
    char full_url[2048];
    if (strstr(url, "/embeddings") == NULL) {
        snprintf(full_url, sizeof(full_url), "%s/embeddings", url);
        url = full_url;
    }
    
    struct ResponseBuffer buf;
    buf.data = (char *)malloc(8192);
    buf.size = 0;
    buf.capacity = 8192;
    if (buf.data == NULL) {
        free(request_json);
        return -1;
    }
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    if (service->config.api_key) {
        char auth_buf[1024];
        snprintf(auth_buf, sizeof(auth_buf), "Authorization: Bearer %s", service->config.api_key);
        headers = curl_slist_append(headers, auth_buf);
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, service->config.timeout_seconds > 0 ?
                     service->config.timeout_seconds : DEFAULT_TIMEOUT);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    free(request_json);
    
    if (res != CURLE_OK) {
        free(buf.data);
        return -1;
    }
    
    /* Parse batch response - simplified parsing */
    /* Format: {"data": [{"embedding": [...]}, {"embedding": [...]}, ...]} */
    const char *data_start = strstr(buf.data, "\"data\":[");
    if (data_start == NULL) {
        free(buf.data);
        return -1;
    }
    
    data_start += 8;  /* Skip "data":[ */
    
    /* Parse each embedding in the array */
    size_t success_count = 0;
    const char *p = data_start;
    
    for (size_t i = 0; i < text_count && *p && *p != ']'; i++) {
        /* Find embedding array */
        const char *emb_start = strstr(p, "\"embedding\":[");
        if (emb_start == NULL) {
            break;
        }
        
        emb_start += 13;  /* Skip "embedding":[ */
        
        /* Count values */
        size_t count = 0;
        const char *q = emb_start;
        while (*q && *q != ']') {
            if (*q == ',') count++;
            q++;
        }
        count++;  /* Last value */
        
        if (count > 0) {
            (*embedding_dims)[i] = count;
            (*embeddings)[i] = (float *)malloc(count * sizeof(float));
            if ((*embeddings)[i] != NULL) {
                /* Parse values */
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
        
        /* Move to next object */
        p = strchr(emb_start, '}');
        if (p) p++;
    }
    
    free(buf.data);
    return (int)success_count;
#else
    (void)service;
    (void)texts;
    (void)text_count;
    (void)embedding_dims;
    (void)embeddings;
    return -1;
#endif
}

/* Generate batch embeddings using OpenAI API */
static int generate_openai_embedding_batch(GV_EmbeddingService *service,
                                           const char **texts,
                                           size_t text_count,
                                           size_t **embedding_dims,
                                           float ***embeddings) {
    CURL *curl = (CURL *)service->curl_handle;
    if (curl == NULL) {
        return -1;
    }
    
    /* Build request JSON with array of inputs */
    size_t json_size = 4096 + text_count * 256;  /* Estimate */
    char *request_json = (char *)malloc(json_size);
    if (request_json == NULL) {
        return -1;
    }
    
    const char *model = service->config.model ? service->config.model : "text-embedding-3-small";
    int dim = service->config.embedding_dimension > 0 ? (int)service->config.embedding_dimension : 0;
    
    strcpy(request_json, "{\"input\":[");
    size_t pos = strlen(request_json);
    
    for (size_t i = 0; i < text_count; i++) {
        if (i > 0) {
            request_json[pos++] = ',';
        }
        request_json[pos++] = '"';
        /* Escape text */
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
    
    struct ResponseBuffer buf;
    buf.data = (char *)malloc(8192);
    buf.size = 0;
    buf.capacity = 8192;
    if (buf.data == NULL) {
        free(request_json);
        return -1;
    }
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    char auth_buf[1024];
    snprintf(auth_buf, sizeof(auth_buf), "Authorization: Bearer %s", service->config.api_key);
    headers = curl_slist_append(headers, auth_buf);
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, service->config.timeout_seconds > 0 ?
                     service->config.timeout_seconds : DEFAULT_TIMEOUT);
    
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    free(request_json);
    
    if (res != CURLE_OK) {
        free(buf.data);
        return -1;
    }
    
    /* Parse batch response - OpenAI format: {"data": [{"embedding": [...]}, {"embedding": [...]}, ...]} */
    const char *data_start = strstr(buf.data, "\"data\":[");
    if (data_start == NULL) {
        free(buf.data);
        return -1;
    }
    
    data_start += 8;  /* Skip "data":[ */
    
    /* Parse each embedding in the array */
    size_t success_count = 0;
    const char *p = data_start;
    
    for (size_t i = 0; i < text_count && *p && *p != ']'; i++) {
        /* Find embedding array */
        const char *emb_start = strstr(p, "\"embedding\":[");
        if (emb_start == NULL) {
            break;
        }
        
        emb_start += 13;  /* Skip "embedding":[ */
        
        /* Count values */
        size_t count = 0;
        const char *q = emb_start;
        while (*q && *q != ']') {
            if (*q == ',') count++;
            q++;
        }
        count++;  /* Last value */
        
        if (count > 0) {
            (*embedding_dims)[i] = count;
            (*embeddings)[i] = (float *)malloc(count * sizeof(float));
            if ((*embeddings)[i] != NULL) {
                /* Parse values */
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
        
        /* Move to next object */
        p = strchr(emb_start, '}');
        if (p) p++;
    }
    
    free(buf.data);
    return (int)success_count;
}

#endif /* HAVE_CURL */

/* Create embedding service */
GV_EmbeddingService *gv_embedding_service_create(const GV_EmbeddingConfig *config) {
    if (config == NULL || config->provider == GV_EMBEDDING_PROVIDER_NONE) {
        return NULL;
    }
    
    GV_EmbeddingService *service = (GV_EmbeddingService *)calloc(1, sizeof(GV_EmbeddingService));
    if (service == NULL) {
        return NULL;
    }
    
    service->config = *config;
    service->last_error = NULL;
    
    /* Copy strings */
    if (config->api_key) {
        service->config.api_key = strdup(config->api_key);
    }
    if (config->model) {
        service->config.model = strdup(config->model);
    }
    if (config->base_url) {
        service->config.base_url = strdup(config->base_url);
    }
    if (config->huggingface_model_path) {
        service->config.huggingface_model_path = strdup(config->huggingface_model_path);
    }
    
    /* Create cache if enabled */
    if (config->enable_cache) {
        service->cache = gv_embedding_cache_create(config->cache_size);
    }
    
#ifdef HAVE_CURL
    if (config->provider == GV_EMBEDDING_PROVIDER_OPENAI || 
        config->provider == GV_EMBEDDING_PROVIDER_CUSTOM ||
        config->provider == GV_EMBEDDING_PROVIDER_GOOGLE) {
        service->curl_handle = curl_easy_init();
        if (service->curl_handle == NULL) {
            gv_embedding_cache_destroy(service->cache);
            gv_embedding_config_free(&service->config);
            free(service);
            return NULL;
        }
    }
#else
    if (config->provider == GV_EMBEDDING_PROVIDER_OPENAI || 
        config->provider == GV_EMBEDDING_PROVIDER_CUSTOM ||
        config->provider == GV_EMBEDDING_PROVIDER_GOOGLE) {
        /* No CURL support */
        gv_embedding_cache_destroy(service->cache);
        gv_embedding_config_free(&service->config);
        free(service);
        return NULL;
    }
#endif
    
    return service;
}

/* Destroy embedding service */
void gv_embedding_service_destroy(GV_EmbeddingService *service) {
    if (service == NULL) {
        return;
    }
    
    gv_embedding_cache_destroy(service->cache);
    
#ifdef HAVE_CURL
    if (service->curl_handle) {
        curl_easy_cleanup((CURL *)service->curl_handle);
    }
#endif
    
    gv_embedding_config_free(&service->config);
    if (service->last_error) {
        free(service->last_error);
    }
    free(service);
}

/* Generate embedding */
int gv_embedding_generate(GV_EmbeddingService *service,
                          const char *text,
                          size_t *embedding_dim,
                          float **embedding) {
    if (service == NULL || text == NULL || embedding_dim == NULL || embedding == NULL) {
        return -1;
    }
    
    /* Check cache first */
    if (service->cache) {
        const float *cached_embedding = NULL;
        size_t cached_dim = 0;
        if (gv_embedding_cache_get(service->cache, text, &cached_dim, &cached_embedding) == 1) {
            /* Found in cache - copy it */
            *embedding_dim = cached_dim;
            *embedding = (float *)malloc(cached_dim * sizeof(float));
            if (*embedding == NULL) {
                return -1;
            }
            memcpy(*embedding, cached_embedding, cached_dim * sizeof(float));
            return 0;
        }
    }
    
    /* Generate embedding */
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
        /* Store in cache */
        gv_embedding_cache_put(service->cache, text, *embedding_dim, *embedding);
    }
    
    return result;
}

/* Generate batch embeddings */
int gv_embedding_generate_batch(GV_EmbeddingService *service,
                                const char **texts,
                                size_t text_count,
                                size_t **embedding_dims,
                                float ***embeddings) {
    if (service == NULL || texts == NULL || text_count == 0 ||
        embedding_dims == NULL || embeddings == NULL) {
        return -1;
    }
    
    *embedding_dims = (size_t *)malloc(text_count * sizeof(size_t));
    *embeddings = (float **)malloc(text_count * sizeof(float *));
    if (*embedding_dims == NULL || *embeddings == NULL) {
        if (*embedding_dims) free(*embedding_dims);
        if (*embeddings) free(*embeddings);
        return -1;
    }
    
    size_t success_count = 0;
    
#ifdef HAVE_CURL
    if (service->config.provider == GV_EMBEDDING_PROVIDER_OPENAI ||
        service->config.provider == GV_EMBEDDING_PROVIDER_CUSTOM) {
        /* Try batch API first */
        int batch_result = generate_openai_embedding_batch(service, texts, text_count,
                                                          embedding_dims, embeddings);
        if (batch_result == 0) {
            success_count = text_count;
        } else {
            /* Fall back to individual requests */
            for (size_t i = 0; i < text_count; i++) {
                if (gv_embedding_generate(service, texts[i], &(*embedding_dims)[i],
                                         &(*embeddings)[i]) == 0) {
                    success_count++;
                } else {
                    (*embedding_dims)[i] = 0;
                    (*embeddings)[i] = NULL;
                }
            }
        }
    } else if (service->config.provider == GV_EMBEDDING_PROVIDER_GOOGLE) {
        /* Try batch API first */
        int batch_result = generate_google_embedding_batch(service, texts, text_count,
                                                          embedding_dims, embeddings);
        if (batch_result >= 0) {
            success_count = batch_result;
        } else {
            /* Fall back to individual requests */
            for (size_t i = 0; i < text_count; i++) {
                if (gv_embedding_generate(service, texts[i], &(*embedding_dims)[i],
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
        /* Try batch API first, fall back to individual */
        int batch_result = generate_huggingface_embedding_batch(service, texts, text_count,
                                                                embedding_dims, embeddings);
        if (batch_result >= 0) {
            success_count = batch_result;
        } else {
            /* Fall back to individual requests */
            for (size_t i = 0; i < text_count; i++) {
                if (gv_embedding_generate(service, texts[i], &(*embedding_dims)[i],
                                         &(*embeddings)[i]) == 0) {
                    success_count++;
                } else {
                    (*embedding_dims)[i] = 0;
                    (*embeddings)[i] = NULL;
                }
            }
        }
#else
        /* No CURL support */
        for (size_t i = 0; i < text_count; i++) {
            (*embedding_dims)[i] = 0;
            (*embeddings)[i] = NULL;
        }
        return -1;
#endif
    }
    
    return (int)success_count;
}

