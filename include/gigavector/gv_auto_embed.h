#ifndef GIGAVECTOR_GV_AUTO_EMBED_H
#define GIGAVECTOR_GV_AUTO_EMBED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Database GV_Database;

typedef enum {
    GV_EMBED_PROVIDER_OPENAI = 0,
    GV_EMBED_PROVIDER_GOOGLE = 1,
    GV_EMBED_PROVIDER_HUGGINGFACE = 2,
    GV_EMBED_PROVIDER_CUSTOM = 3
} GV_AutoEmbedProvider;

typedef struct {
    GV_AutoEmbedProvider provider;
    const char *api_key;
    const char *model_name;          /* e.g., "text-embedding-3-small" */
    const char *base_url;            /* Custom endpoint URL (for CUSTOM provider) */
    size_t dimension;                /* Expected output dimension */
    int cache_embeddings;            /* Cache computed embeddings (default: 1) */
    size_t max_cache_entries;        /* Max cached embeddings (default: 10000) */
    size_t max_text_length;          /* Max input text length in chars (default: 8192) */
    int batch_size;                  /* Batch size for bulk operations (default: 32) */
} GV_AutoEmbedConfig;

typedef struct GV_AutoEmbedder GV_AutoEmbedder;

typedef struct {
    uint64_t total_embeddings;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t api_calls;
    uint64_t api_errors;
    double avg_latency_ms;
} GV_AutoEmbedStats;

/* Lifecycle */
void gv_auto_embed_config_init(GV_AutoEmbedConfig *config);
GV_AutoEmbedder *gv_auto_embed_create(const GV_AutoEmbedConfig *config);
void gv_auto_embed_destroy(GV_AutoEmbedder *embedder);

/* Embed text and add to database */
int gv_auto_embed_add_text(GV_AutoEmbedder *embedder, GV_Database *db,
                            const char *text, const char *metadata_key, const char *metadata_value);

/* Embed text and search database */
int gv_auto_embed_search_text(GV_AutoEmbedder *embedder, const GV_Database *db,
                               const char *text, size_t k, int distance_type,
                               size_t *out_indices, float *out_distances, size_t *out_count);

/* Batch embed texts */
int gv_auto_embed_add_texts(GV_AutoEmbedder *embedder, GV_Database *db,
                             const char *const *texts, size_t count,
                             const char *const *metadata_keys, const char *const *metadata_values);

/* Raw embedding (returns float array, caller must free) */
float *gv_auto_embed_text(GV_AutoEmbedder *embedder, const char *text, size_t *out_dimension);

/* Stats */
int gv_auto_embed_get_stats(const GV_AutoEmbedder *embedder, GV_AutoEmbedStats *stats);
void gv_auto_embed_clear_cache(GV_AutoEmbedder *embedder);

#ifdef __cplusplus
}
#endif
#endif
