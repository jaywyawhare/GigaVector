/**
 * @file gv_inference.h
 * @brief Integrated Inference API -- text in, search results out.
 *
 * Combines embedding, indexing, and search in a single high-level call,
 * similar to Pinecone's Integrated Inference.  Users supply plain text;
 * the engine embeds it via the configured provider, stores the vector
 * (with metadata and the original text), and searches the underlying
 * GV_Database transparently.
 *
 * Thread-safe: all public functions are serialized via an internal mutex.
 */

#ifndef GIGAVECTOR_GV_INFERENCE_H
#define GIGAVECTOR_GV_INFERENCE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/*  Configuration                                                              */
/* ========================================================================== */

/**
 * @brief Configuration for the inference engine.
 *
 * Pass to gv_inference_create() after initializing with
 * gv_inference_config_init().  All string pointers must remain valid
 * until gv_inference_create() returns (they are copied internally).
 */
typedef struct {
    const char *embed_provider;  /**< Embedding provider: "openai", "google", or "huggingface". */
    const char *api_key;         /**< API key for the embedding service. */
    const char *model;           /**< Model identifier (default: "text-embedding-3-small"). */
    size_t      dimension;       /**< Embedding dimension (default: 1536). */
    int         distance_type;   /**< Distance metric, see GV_DistanceType (default: cosine = 1). */
    size_t      cache_size;      /**< Maximum number of cached embeddings (default: 10000). */
} GV_InferenceConfig;

/* ========================================================================== */
/*  Opaque engine handle                                                       */
/* ========================================================================== */

/**
 * @brief Opaque handle to an inference engine instance.
 */
typedef struct GV_InferenceEngine GV_InferenceEngine;

/* ========================================================================== */
/*  Result type                                                                */
/* ========================================================================== */

/**
 * @brief A single search result returned by the inference engine.
 *
 * The @c text and @c metadata_json fields are heap-allocated strings
 * owned by the result.  Free an array of results with
 * gv_inference_free_results().
 */
typedef struct {
    size_t  index;          /**< Vector index in the underlying database. */
    float   distance;       /**< Distance / similarity score. */
    char   *text;           /**< Original text (if stored), or NULL. */
    char   *metadata_json;  /**< User-supplied metadata as JSON string, or NULL. */
} GV_InferenceResult;

/* ========================================================================== */
/*  Lifecycle                                                                  */
/* ========================================================================== */

/**
 * @brief Initialize an inference configuration with sensible defaults.
 *
 * Defaults:
 *   - embed_provider: "openai"
 *   - model:          "text-embedding-3-small"
 *   - dimension:      1536
 *   - distance_type:  1 (cosine)
 *   - cache_size:     10000
 *
 * @param config Configuration to initialize; must be non-NULL.
 */
void gv_inference_config_init(GV_InferenceConfig *config);

/**
 * @brief Create an inference engine backed by the given database.
 *
 * The database must already be open (via gv_db_open or equivalent) and
 * its dimension must match the configured embedding dimension.  The
 * engine does NOT take ownership of the database -- the caller is
 * responsible for closing it after destroying the engine.
 *
 * @param db     Open GV_Database handle (cast to void* for ABI flexibility).
 * @param config Inference configuration; must be non-NULL.
 * @return New engine handle, or NULL on error.
 */
GV_InferenceEngine *gv_inference_create(void *db, const GV_InferenceConfig *config);

/**
 * @brief Destroy an inference engine and release all associated resources.
 *
 * Does NOT close the underlying database.  Safe to call with NULL.
 *
 * @param eng Engine handle to destroy.
 */
void gv_inference_destroy(GV_InferenceEngine *eng);

/* ========================================================================== */
/*  Insert                                                                     */
/* ========================================================================== */

/**
 * @brief Embed a single text and add it to the database.
 *
 * The original text is stored alongside the vector as a special "_text"
 * metadata key so that it can be retrieved with search results.  Any
 * additional metadata is passed as a JSON string and stored under a
 * "_meta" key.
 *
 * @param eng           Engine handle; must be non-NULL.
 * @param text          Input text to embed and store; must be non-NULL.
 * @param metadata_json Optional JSON string of user metadata (may be NULL).
 * @return Index of the newly inserted vector (>= 0), or -1 on error.
 */
int gv_inference_add(GV_InferenceEngine *eng, const char *text,
                     const char *metadata_json);

/**
 * @brief Embed and insert multiple texts in a single batch.
 *
 * Uses batch embedding for efficiency.  Metadata strings in
 * @p metadata_jsons may be NULL for individual entries.
 *
 * @param eng            Engine handle; must be non-NULL.
 * @param texts          Array of text strings to embed; must be non-NULL.
 * @param metadata_jsons Array of JSON metadata strings (may be NULL, or contain NULL entries).
 * @param count          Number of texts.
 * @return 0 on success, -1 on error.
 */
int gv_inference_add_batch(GV_InferenceEngine *eng, const char **texts,
                           const char **metadata_jsons, size_t count);

/* ========================================================================== */
/*  Search                                                                     */
/* ========================================================================== */

/**
 * @brief Embed a query text and search the database for the k nearest results.
 *
 * @param eng        Engine handle; must be non-NULL.
 * @param query_text Query text to embed; must be non-NULL.
 * @param k          Maximum number of results to return.
 * @param results    Output array of at least @p k elements; filled on success.
 * @return Number of results found (0 to k), or -1 on error.
 */
int gv_inference_search(GV_InferenceEngine *eng, const char *query_text,
                        size_t k, GV_InferenceResult *results);

/**
 * @brief Embed a query text and search with a metadata filter expression.
 *
 * The filter expression follows the same syntax as gv_filter_parse().
 *
 * @param eng         Engine handle; must be non-NULL.
 * @param query_text  Query text to embed; must be non-NULL.
 * @param k           Maximum number of results to return.
 * @param filter_expr Metadata filter expression string; must be non-NULL.
 * @param results     Output array of at least @p k elements; filled on success.
 * @return Number of results found (0 to k), or -1 on error.
 */
int gv_inference_search_filtered(GV_InferenceEngine *eng,
                                 const char *query_text, size_t k,
                                 const char *filter_expr,
                                 GV_InferenceResult *results);

/* ========================================================================== */
/*  Upsert                                                                     */
/* ========================================================================== */

/**
 * @brief Upsert: embed new text and replace the vector at @p index.
 *
 * If @p index equals the current vector count, a new vector is appended
 * (equivalent to gv_inference_add).  Otherwise the existing vector and
 * its metadata are replaced.
 *
 * @param eng           Engine handle; must be non-NULL.
 * @param index         Target vector index.
 * @param text          New text to embed and store; must be non-NULL.
 * @param metadata_json Optional JSON metadata string (may be NULL).
 * @return 0 on success, -1 on error.
 */
int gv_inference_upsert(GV_InferenceEngine *eng, size_t index,
                        const char *text, const char *metadata_json);

/* ========================================================================== */
/*  Result cleanup                                                             */
/* ========================================================================== */

/**
 * @brief Free heap-allocated fields inside an array of inference results.
 *
 * Frees the @c text and @c metadata_json strings in each element.
 * Does NOT free the array itself (it may be stack-allocated).
 *
 * @param results Array of results to free; may be NULL.
 * @param count   Number of elements in the array.
 */
void gv_inference_free_results(GV_InferenceResult *results, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_INFERENCE_H */
