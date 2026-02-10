#ifndef GIGAVECTOR_GV_MUVERA_H
#define GIGAVECTOR_GV_MUVERA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Configuration for the MUVERA encoder.
 *
 * MUVERA converts variable-length multi-vector embeddings (e.g., ColBERT
 * token embeddings) into fixed-size single vectors, dramatically reducing
 * memory and compute costs for late-interaction models.
 */
typedef struct {
    size_t   token_dimension;   /**< Per-token embedding dimension (e.g., 128 for ColBERT). */
    size_t   num_projections;   /**< Number of random projections / hash functions (default: 64). */
    size_t   output_dimension;  /**< Output vector size (default: num_projections * token_dimension / 4). */
    uint64_t seed;              /**< Seed for reproducible random projections. */
    int      normalize;         /**< If non-zero, L2-normalize the output vector (default: 1). */
} GV_MuveraConfig;

/**
 * @brief Opaque MUVERA encoder handle.
 *
 * All const methods are thread-safe; no shared mutable state exists after
 * creation.
 */
typedef struct GV_MuveraEncoder GV_MuveraEncoder;

/**
 * @brief Initialize a MUVERA configuration with sensible defaults.
 *
 * Sets token_dimension = 128, num_projections = 64,
 * output_dimension = 0 (auto-computed as num_projections * token_dimension / 4),
 * seed = 42, normalize = 1.
 *
 * @param config Configuration struct to initialize; must be non-NULL.
 */
void gv_muvera_config_init(GV_MuveraConfig *config);

/**
 * @brief Create a new MUVERA encoder from the given configuration.
 *
 * Generates the internal random projection matrix and hash functions
 * deterministically from the seed.  The encoder is immutable after creation.
 *
 * @param config Encoder configuration; NULL uses defaults.
 * @return Allocated encoder, or NULL on error.
 */
GV_MuveraEncoder *gv_muvera_create(const GV_MuveraConfig *config);

/**
 * @brief Destroy a MUVERA encoder and free all resources.
 *
 * Safe to call with NULL; no action is taken.
 *
 * @param enc Encoder instance.
 */
void gv_muvera_destroy(GV_MuveraEncoder *enc);

/**
 * @brief Encode a variable-length set of token embeddings into a fixed-size vector.
 *
 * For each projection, tokens are hashed into buckets, bucket means are
 * computed, and the results are concatenated to form the output vector.
 * Optionally L2-normalizes the result.
 *
 * @param enc        Encoder instance; must be non-NULL.
 * @param tokens     Flat array of num_tokens * token_dimension floats.
 * @param num_tokens Number of token embeddings.
 * @param output     Pre-allocated output buffer of at least output_dimension floats.
 * @return 0 on success, -1 on error.
 */
int gv_muvera_encode(const GV_MuveraEncoder *enc,
                     const float *tokens, size_t num_tokens,
                     float *output);

/**
 * @brief Return the output dimensionality of the encoder.
 *
 * @param enc Encoder instance; must be non-NULL.
 * @return Output dimension, or 0 if enc is NULL.
 */
size_t gv_muvera_output_dimension(const GV_MuveraEncoder *enc);

/**
 * @brief Encode a batch of token sets in one call.
 *
 * Each element in token_sets is a flat array of token_counts[i] * token_dimension
 * floats.  The outputs buffer must hold batch_size * output_dimension floats.
 *
 * @param enc          Encoder instance; must be non-NULL.
 * @param token_sets   Array of batch_size token arrays.
 * @param token_counts Array of batch_size token counts (one per set).
 * @param batch_size   Number of token sets to encode.
 * @param outputs      Pre-allocated output buffer (batch_size * output_dimension floats).
 * @return 0 on success, -1 on error.
 */
int gv_muvera_encode_batch(const GV_MuveraEncoder *enc,
                           const float **token_sets,
                           const size_t *token_counts,
                           size_t batch_size,
                           float *outputs);

/**
 * @brief Serialize the encoder to a file path.
 *
 * Writes the configuration and all random projection data so the encoder
 * can be reconstructed without regenerating from the seed.
 *
 * @param enc  Encoder instance; must be non-NULL.
 * @param path Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_muvera_save(const GV_MuveraEncoder *enc, const char *path);

/**
 * @brief Load an encoder from a file path.
 *
 * @param path Input file path.
 * @return Loaded encoder, or NULL on error.
 */
GV_MuveraEncoder *gv_muvera_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_MUVERA_H */
