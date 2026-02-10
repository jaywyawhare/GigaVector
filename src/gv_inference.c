/**
 * @file gv_inference.c
 * @brief Integrated Inference API -- text in, search results out.
 *
 * Wraps the auto-embedding API (gv_auto_embed.h) and the database API
 * (gv_database.h) to provide a single-call interface: supply plain text,
 * get back search results with original text and metadata.
 *
 * Original text is stored in vector metadata under the reserved key
 * "_text".  User-supplied JSON metadata is stored under "_meta".
 * Thread-safe via pthread_mutex_t.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

#include "gigavector/gv_inference.h"
#include "gigavector/gv_auto_embed.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_json.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_metadata.h"
#include "gigavector/gv_soa_storage.h"

/* ========================================================================== */
/*  Constants                                                                  */
/* ========================================================================== */

#define INF_DEFAULT_DIMENSION     1536
#define INF_DEFAULT_CACHE_SIZE    10000
#define INF_DEFAULT_DISTANCE_TYPE 1       /* GV_DISTANCE_COSINE */

/** Reserved metadata key for the original text. */
#define INF_META_KEY_TEXT          "_text"
/** Reserved metadata key for user-supplied JSON metadata. */
#define INF_META_KEY_META          "_meta"

/* ========================================================================== */
/*  Engine structure                                                           */
/* ========================================================================== */

struct GV_InferenceEngine {
    GV_Database       *db;            /**< Backing vector database (not owned). */
    GV_AutoEmbedder   *embedder;      /**< Auto-embedding handle (owned). */
    int                distance_type;  /**< Configured distance metric. */
    size_t             dimension;      /**< Embedding dimension. */
    pthread_mutex_t    mutex;          /**< Serializes all public operations. */
};

/* ========================================================================== */
/*  Provider string to enum mapping                                            */
/* ========================================================================== */

/**
 * Map a provider name string to the corresponding GV_AutoEmbedProvider
 * enum value.  Returns GV_EMBED_PROVIDER_OPENAI for unrecognized strings.
 */
static GV_AutoEmbedProvider inf_resolve_provider(const char *name) {
    if (!name) return GV_EMBED_PROVIDER_OPENAI;

    if (strcmp(name, "google") == 0)      return GV_EMBED_PROVIDER_GOOGLE;
    if (strcmp(name, "huggingface") == 0)  return GV_EMBED_PROVIDER_HUGGINGFACE;

    /* Default / "openai" */
    return GV_EMBED_PROVIDER_OPENAI;
}

/* ========================================================================== */
/*  Metadata helpers                                                           */
/* ========================================================================== */

/**
 * Build a metadata key/value array that stores the original text and
 * optional user JSON.  Fills @p keys and @p values (caller-owned pointers
 * into the provided strings -- no copies are made here).
 *
 * @return Number of metadata entries (1 or 2).
 */
static size_t inf_build_metadata(const char *text, const char *metadata_json,
                                 const char *keys[2], const char *values[2]) {
    size_t n = 0;
    keys[n]   = INF_META_KEY_TEXT;
    values[n] = text;
    n++;

    if (metadata_json && metadata_json[0]) {
        keys[n]   = INF_META_KEY_META;
        values[n] = metadata_json;
        n++;
    }
    return n;
}

/**
 * Extract the "_text" value from a metadata linked list.
 * Returns a heap-allocated copy, or NULL if not found.
 */
static char *inf_extract_text(const GV_Metadata *meta) {
    while (meta) {
        if (meta->key && strcmp(meta->key, INF_META_KEY_TEXT) == 0) {
            return meta->value ? strdup(meta->value) : NULL;
        }
        meta = meta->next;
    }
    return NULL;
}

/**
 * Extract the "_meta" value from a metadata linked list.
 * Returns a heap-allocated copy, or NULL if not found.
 */
static char *inf_extract_meta_json(const GV_Metadata *meta) {
    while (meta) {
        if (meta->key && strcmp(meta->key, INF_META_KEY_META) == 0) {
            return meta->value ? strdup(meta->value) : NULL;
        }
        meta = meta->next;
    }
    return NULL;
}

/**
 * Given a database and a GV_SearchResult (which holds a vector pointer),
 * resolve the 0-based insertion index by comparing the data pointer
 * against the SoA storage.
 */
static size_t inf_resolve_index(const GV_Database *db,
                                const GV_SearchResult *sr) {
    if (!db || !db->soa_storage || !sr || !sr->vector) return (size_t)-1;

    const float *target = sr->vector->data;
    size_t count = gv_database_count(db);
    for (size_t i = 0; i < count; i++) {
        const float *candidate = gv_database_get_vector(db, i);
        if (candidate == target) return i;
    }
    return (size_t)-1;
}

/**
 * Populate a GV_InferenceResult from a search result by resolving the
 * index and extracting metadata.  Returns 0 on success, -1 on error.
 */
static int inf_populate_result(const GV_Database *db,
                               const GV_SearchResult *sr,
                               GV_InferenceResult *out) {
    if (!db || !sr || !out) return -1;

    out->distance = sr->distance;
    out->index    = inf_resolve_index(db, sr);

    /* Retrieve metadata from SoA storage */
    out->text          = NULL;
    out->metadata_json = NULL;

    if (out->index != (size_t)-1 && db->soa_storage) {
        GV_Metadata *meta = gv_soa_storage_get_metadata(db->soa_storage,
                                                         out->index);
        if (meta) {
            out->text          = inf_extract_text(meta);
            out->metadata_json = inf_extract_meta_json(meta);
        }
    }
    return 0;
}

/* ========================================================================== */
/*  Lifecycle                                                                  */
/* ========================================================================== */

void gv_inference_config_init(GV_InferenceConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->embed_provider = "openai";
    config->model          = "text-embedding-3-small";
    config->dimension      = INF_DEFAULT_DIMENSION;
    config->distance_type  = INF_DEFAULT_DISTANCE_TYPE;
    config->cache_size     = INF_DEFAULT_CACHE_SIZE;
}

GV_InferenceEngine *gv_inference_create(void *db,
                                        const GV_InferenceConfig *config) {
    if (!db || !config) return NULL;

    GV_InferenceEngine *eng = (GV_InferenceEngine *)calloc(1, sizeof(*eng));
    if (!eng) return NULL;

    eng->db            = (GV_Database *)db;
    eng->distance_type = config->distance_type;
    eng->dimension     = config->dimension > 0 ? config->dimension
                                                : INF_DEFAULT_DIMENSION;

    /* Initialize the mutex */
    if (pthread_mutex_init(&eng->mutex, NULL) != 0) {
        free(eng);
        return NULL;
    }

    /* Configure auto-embedder */
    GV_AutoEmbedConfig ae_cfg;
    gv_auto_embed_config_init(&ae_cfg);

    ae_cfg.provider          = inf_resolve_provider(config->embed_provider);
    ae_cfg.api_key           = config->api_key;
    ae_cfg.model_name        = config->model ? config->model
                                             : "text-embedding-3-small";
    ae_cfg.dimension         = eng->dimension;
    ae_cfg.cache_embeddings  = 1;
    ae_cfg.max_cache_entries = config->cache_size > 0 ? config->cache_size
                                                      : INF_DEFAULT_CACHE_SIZE;

    eng->embedder = gv_auto_embed_create(&ae_cfg);
    if (!eng->embedder) {
        pthread_mutex_destroy(&eng->mutex);
        free(eng);
        return NULL;
    }

    return eng;
}

void gv_inference_destroy(GV_InferenceEngine *eng) {
    if (!eng) return;

    gv_auto_embed_destroy(eng->embedder);
    pthread_mutex_destroy(&eng->mutex);
    free(eng);
}

/* ========================================================================== */
/*  Insert -- single                                                           */
/* ========================================================================== */

int gv_inference_add(GV_InferenceEngine *eng, const char *text,
                     const char *metadata_json) {
    if (!eng || !text) return -1;

    pthread_mutex_lock(&eng->mutex);

    /* Embed the text */
    size_t dim = 0;
    float *vec = gv_auto_embed_text(eng->embedder, text, &dim);
    if (!vec) {
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Build metadata key/value pairs */
    const char *keys[2];
    const char *values[2];
    size_t meta_count = inf_build_metadata(text, metadata_json, keys, values);

    /* Record the index before insertion (it will be the current count) */
    size_t new_index = gv_database_count(eng->db);

    int rc = gv_db_add_vector_with_rich_metadata(
        eng->db, vec, dim,
        (const char *const *)keys, (const char *const *)values, meta_count);

    free(vec);
    pthread_mutex_unlock(&eng->mutex);

    return rc == 0 ? (int)new_index : -1;
}

/* ========================================================================== */
/*  Insert -- batch                                                            */
/* ========================================================================== */

int gv_inference_add_batch(GV_InferenceEngine *eng, const char **texts,
                           const char **metadata_jsons, size_t count) {
    if (!eng || !texts || count == 0) return -1;

    pthread_mutex_lock(&eng->mutex);

    /*
     * Embed each text individually via the auto-embedder (which already
     * handles caching internally).  Then insert each vector with its
     * rich metadata.  A future optimization could use
     * gv_auto_embed_add_texts for a true batch HTTP call, but that API
     * only supports a single metadata key-value pair per vector, whereas
     * we need two ("_text" and "_meta").
     */
    int failures = 0;

    for (size_t i = 0; i < count; i++) {
        if (!texts[i]) { failures++; continue; }

        size_t dim = 0;
        float *vec = gv_auto_embed_text(eng->embedder, texts[i], &dim);
        if (!vec) { failures++; continue; }

        const char *keys[2];
        const char *values[2];
        const char *meta_json = (metadata_jsons && metadata_jsons[i])
                                    ? metadata_jsons[i] : NULL;
        size_t meta_count = inf_build_metadata(texts[i], meta_json,
                                               keys, values);

        int rc = gv_db_add_vector_with_rich_metadata(
            eng->db, vec, dim,
            (const char *const *)keys, (const char *const *)values,
            meta_count);

        free(vec);
        if (rc != 0) failures++;
    }

    pthread_mutex_unlock(&eng->mutex);
    return failures == 0 ? 0 : -1;
}

/* ========================================================================== */
/*  Search -- unfiltered                                                       */
/* ========================================================================== */

int gv_inference_search(GV_InferenceEngine *eng, const char *query_text,
                        size_t k, GV_InferenceResult *results) {
    if (!eng || !query_text || k == 0 || !results) return -1;

    pthread_mutex_lock(&eng->mutex);

    /* Embed the query */
    size_t dim = 0;
    float *vec = gv_auto_embed_text(eng->embedder, query_text, &dim);
    if (!vec) {
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Allocate temporary search results */
    GV_SearchResult *sr = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    if (!sr) {
        free(vec);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    int found = gv_db_search(eng->db, vec, k, sr,
                             (GV_DistanceType)eng->distance_type);
    free(vec);

    if (found < 0) {
        free(sr);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Convert to inference results */
    int count = found > (int)k ? (int)k : found;
    for (int i = 0; i < count; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        inf_populate_result(eng->db, &sr[i], &results[i]);
    }

    free(sr);
    pthread_mutex_unlock(&eng->mutex);
    return count;
}

/* ========================================================================== */
/*  Search -- filtered                                                         */
/* ========================================================================== */

int gv_inference_search_filtered(GV_InferenceEngine *eng,
                                 const char *query_text, size_t k,
                                 const char *filter_expr,
                                 GV_InferenceResult *results) {
    if (!eng || !query_text || k == 0 || !filter_expr || !results) return -1;

    pthread_mutex_lock(&eng->mutex);

    /* Embed the query */
    size_t dim = 0;
    float *vec = gv_auto_embed_text(eng->embedder, query_text, &dim);
    if (!vec) {
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Allocate temporary search results */
    GV_SearchResult *sr = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    if (!sr) {
        free(vec);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    int found = gv_db_search_with_filter_expr(
        eng->db, vec, k, sr,
        (GV_DistanceType)eng->distance_type, filter_expr);
    free(vec);

    if (found < 0) {
        free(sr);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Convert to inference results */
    int count = found > (int)k ? (int)k : found;
    for (int i = 0; i < count; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        inf_populate_result(eng->db, &sr[i], &results[i]);
    }

    free(sr);
    pthread_mutex_unlock(&eng->mutex);
    return count;
}

/* ========================================================================== */
/*  Upsert                                                                     */
/* ========================================================================== */

int gv_inference_upsert(GV_InferenceEngine *eng, size_t index,
                        const char *text, const char *metadata_json) {
    if (!eng || !text) return -1;

    pthread_mutex_lock(&eng->mutex);

    /* Embed the new text */
    size_t dim = 0;
    float *vec = gv_auto_embed_text(eng->embedder, text, &dim);
    if (!vec) {
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Build metadata key/value pairs */
    const char *keys[2];
    const char *values[2];
    size_t meta_count = inf_build_metadata(text, metadata_json, keys, values);

    int rc = gv_db_upsert_with_metadata(
        eng->db, index, vec, dim,
        (const char *const *)keys, (const char *const *)values, meta_count);

    free(vec);
    pthread_mutex_unlock(&eng->mutex);
    return rc;
}

/* ========================================================================== */
/*  Result cleanup                                                             */
/* ========================================================================== */

void gv_inference_free_results(GV_InferenceResult *results, size_t count) {
    if (!results) return;

    for (size_t i = 0; i < count; i++) {
        free(results[i].text);
        results[i].text = NULL;
        free(results[i].metadata_json);
        results[i].metadata_json = NULL;
    }
}
