/**
 * @file gv_embedding.h
 * @brief Embedding service API for generating vector embeddings from text.
 * 
 * Supports multiple providers (OpenAI, HuggingFace, Google, local models) with
 * batch processing and caching capabilities.
 */

#ifndef GV_EMBEDDING_H
#define GV_EMBEDDING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Embedding provider types.
 */
typedef enum {
    GV_EMBEDDING_PROVIDER_OPENAI = 0,        /**< OpenAI embeddings API */
    GV_EMBEDDING_PROVIDER_HUGGINGFACE = 1,   /**< HuggingFace sentence-transformers (local) */
    GV_EMBEDDING_PROVIDER_CUSTOM = 2,        /**< Custom/OpenAI-compatible API */
    GV_EMBEDDING_PROVIDER_GOOGLE = 4,        /**< Google Generative AI embeddings API */
    GV_EMBEDDING_PROVIDER_NONE = 3           /**< No provider (disabled) */
} GV_EmbeddingProvider;

/**
 * @brief Embedding service configuration.
 */
typedef struct {
    GV_EmbeddingProvider provider;           /**< Provider type */
    char *api_key;                           /**< API key (for OpenAI/Custom) */
    char *model;                             /**< Model name/identifier */
    char *base_url;                          /**< Base URL (for Custom provider) */
    size_t embedding_dimension;              /**< Expected embedding dimension (0 = auto-detect) */
    size_t batch_size;                       /**< Batch size for batch operations (default: 100) */
    int enable_cache;                        /**< Enable embedding cache (1) or disable (0) */
    size_t cache_size;                       /**< Maximum cache entries (0 = unlimited) */
    int timeout_seconds;                     /**< Request timeout in seconds */
    char *huggingface_model_path;           /**< Path to local HuggingFace model (for HuggingFace provider, optional - uses TEI API by default) */
} GV_EmbeddingConfig;

/**
 * @brief Embedding service structure (opaque).
 */
typedef struct GV_EmbeddingService GV_EmbeddingService;

/**
 * @brief Embedding cache structure (opaque).
 */
typedef struct GV_EmbeddingCache GV_EmbeddingCache;

/**
 * @brief Create a new embedding service.
 * 
 * @param config Configuration structure; NULL uses defaults.
 * @return Allocated embedding service or NULL on failure.
 */
GV_EmbeddingService *gv_embedding_service_create(const GV_EmbeddingConfig *config);

/**
 * @brief Destroy an embedding service.
 * 
 * @param service Service to destroy.
 */
void gv_embedding_service_destroy(GV_EmbeddingService *service);

/**
 * @brief Generate embedding for a single text.
 * 
 * @param service Embedding service instance.
 * @param text Text to embed.
 * @param embedding_dim Output: dimension of embedding vector.
 * @param embedding Output: allocated embedding vector (caller must free with free()).
 * @return 0 on success, negative on error.
 */
int gv_embedding_generate(GV_EmbeddingService *service,
                          const char *text,
                          size_t *embedding_dim,
                          float **embedding);

/**
 * @brief Generate embeddings for multiple texts (batch operation).
 * 
 * @param service Embedding service instance.
 * @param texts Array of text strings.
 * @param text_count Number of texts.
 * @param embedding_dims Output: array of embedding dimensions (one per text).
 * @param embeddings Output: array of embedding vectors (caller must free each with free()).
 * @return Number of successful embeddings, negative on error.
 */
int gv_embedding_generate_batch(GV_EmbeddingService *service,
                                const char **texts,
                                size_t text_count,
                                size_t **embedding_dims,
                                float ***embeddings);

/**
 * @brief Get default embedding configuration.
 * 
 * @return Default configuration structure.
 */
GV_EmbeddingConfig gv_embedding_config_default(void);

/**
 * @brief Free embedding configuration (frees allocated strings).
 * 
 * @param config Configuration to free.
 */
void gv_embedding_config_free(GV_EmbeddingConfig *config);

/**
 * @brief Create embedding cache.
 * 
 * @param max_size Maximum number of cached entries (0 = unlimited).
 * @return Allocated cache or NULL on failure.
 */
GV_EmbeddingCache *gv_embedding_cache_create(size_t max_size);

/**
 * @brief Destroy embedding cache.
 * 
 * @param cache Cache to destroy.
 */
void gv_embedding_cache_destroy(GV_EmbeddingCache *cache);

/**
 * @brief Get embedding from cache.
 * 
 * @param cache Cache instance.
 * @param text Text to look up.
 * @param embedding_dim Output: dimension of embedding.
 * @param embedding Output: embedding vector (do not free, owned by cache).
 * @return 1 if found, 0 if not found, negative on error.
 */
int gv_embedding_cache_get(GV_EmbeddingCache *cache,
                          const char *text,
                          size_t *embedding_dim,
                          const float **embedding);

/**
 * @brief Store embedding in cache.
 * 
 * @param cache Cache instance.
 * @param text Text key.
 * @param embedding_dim Dimension of embedding.
 * @param embedding Embedding vector (will be copied).
 * @return 0 on success, negative on error.
 */
int gv_embedding_cache_put(GV_EmbeddingCache *cache,
                           const char *text,
                           size_t embedding_dim,
                           const float *embedding);

/**
 * @brief Clear embedding cache.
 * 
 * @param cache Cache instance.
 */
void gv_embedding_cache_clear(GV_EmbeddingCache *cache);

/**
 * @brief Get cache statistics.
 * 
 * @param cache Cache instance.
 * @param size Output: current number of entries.
 * @param hits Output: number of cache hits.
 * @param misses Output: number of cache misses.
 */
void gv_embedding_cache_stats(GV_EmbeddingCache *cache,
                              size_t *size,
                              uint64_t *hits,
                              uint64_t *misses);

#ifdef __cplusplus
}
#endif

#endif /* GV_EMBEDDING_H */


