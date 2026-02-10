#ifndef GIGAVECTOR_GV_QUANTIZATION_H
#define GIGAVECTOR_GV_QUANTIZATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Quantization type controlling the number of bits per dimension.
 */
typedef enum {
    GV_QUANT_BINARY  = 0, /**< 1-bit: sign-based binary quantization.          */
    GV_QUANT_TERNARY = 1, /**< 1.5-bit: ternary {-1, 0, +1}, stored as 2 bits.*/
    GV_QUANT_2BIT    = 2, /**< 2-bit: 4 uniform levels per dimension.          */
    GV_QUANT_4BIT    = 3, /**< 4-bit: 16 uniform levels per dimension.         */
    GV_QUANT_8BIT    = 4  /**< 8-bit: 256 uniform levels per dimension.        */
} GV_QuantType;

/**
 * @brief Quantization mode controlling how queries and stored vectors interact.
 */
typedef enum {
    GV_QUANT_SYMMETRIC  = 0, /**< Both query and stored vectors are quantized.  */
    GV_QUANT_ASYMMETRIC = 1  /**< Query stays float32; only stored is quantized.*/
} GV_QuantMode;

/**
 * @brief Configuration for advanced quantization.
 */
typedef struct {
    GV_QuantType type;        /**< Quantization bit-width.                      */
    GV_QuantMode mode;        /**< Symmetric or asymmetric distance mode.       */
    int          use_rabitq;  /**< Non-zero to enable RaBitQ (binary mode only).*/
    uint64_t     rabitq_seed; /**< Seed for the RaBitQ random rotation matrix.  */
} GV_QuantConfig;

/**
 * @brief Opaque codebook trained from data, used for encode/decode/distance.
 */
typedef struct GV_QuantCodebook GV_QuantCodebook;

/**
 * @brief Initialise a quantization config to safe defaults.
 *
 * Sets type to GV_QUANT_8BIT, mode to GV_QUANT_SYMMETRIC,
 * use_rabitq to 0, and rabitq_seed to 0.
 *
 * @param config Configuration to initialise; must be non-NULL.
 */
void gv_quant_config_init(GV_QuantConfig *config);

/**
 * @brief Train a quantization codebook from a set of vectors.
 *
 * Computes per-dimension statistics (min/max for asymmetric, mean/std for
 * symmetric) required for encoding.  For RaBitQ mode a seeded random
 * orthogonal rotation matrix is also generated and stored in the codebook.
 *
 * @param vectors   Row-major training data (count * dimension floats).
 * @param count     Number of training vectors.
 * @param dimension Vector dimensionality.
 * @param config    Quantization configuration.
 * @return Newly allocated codebook, or NULL on error.
 */
GV_QuantCodebook *gv_quant_train(const float *vectors, size_t count,
                                 size_t dimension,
                                 const GV_QuantConfig *config);

/**
 * @brief Encode a single float vector into quantized codes.
 *
 * @param cb        Trained codebook.
 * @param vector    Input vector (dimension floats).
 * @param dimension Vector dimensionality (must match codebook).
 * @param codes     Output buffer; must be at least gv_quant_code_size() bytes.
 * @return 0 on success, -1 on error.
 */
int gv_quant_encode(const GV_QuantCodebook *cb, const float *vector,
                    size_t dimension, uint8_t *codes);

/**
 * @brief Decode quantized codes back to an approximate float vector.
 *
 * @param cb        Trained codebook.
 * @param codes     Encoded data (gv_quant_code_size() bytes).
 * @param dimension Vector dimensionality (must match codebook).
 * @param output    Output buffer (dimension floats).
 * @return 0 on success, -1 on error.
 */
int gv_quant_decode(const GV_QuantCodebook *cb, const uint8_t *codes,
                    size_t dimension, float *output);

/**
 * @brief Compute asymmetric distance: raw float query vs quantized codes.
 *
 * In asymmetric mode a per-dimension lookup table is built from the codebook
 * so that the query is never quantized, preserving accuracy.
 *
 * @param cb        Trained codebook.
 * @param query     Raw query vector (dimension floats).
 * @param dimension Vector dimensionality (must match codebook).
 * @param codes     Encoded database vector.
 * @return Squared Euclidean distance, or -1.0f on error.
 */
float gv_quant_distance(const GV_QuantCodebook *cb, const float *query,
                        size_t dimension, const uint8_t *codes);

/**
 * @brief Compute symmetric distance: both operands are quantized codes.
 *
 * Works directly on packed codes, using Hamming distance for binary/RaBitQ
 * and reconstructed-value comparison for higher bit-widths.
 *
 * @param cb      Trained codebook.
 * @param codes_a First encoded vector.
 * @param codes_b Second encoded vector.
 * @param dimension Vector dimensionality (must match codebook).
 * @return Squared Euclidean distance (or Hamming for binary), or -1.0f on error.
 */
float gv_quant_distance_qq(const GV_QuantCodebook *cb,
                           const uint8_t *codes_a, const uint8_t *codes_b,
                           size_t dimension);

/**
 * @brief Return the number of bytes needed to store one encoded vector.
 *
 * @param cb        Trained codebook.
 * @param dimension Vector dimensionality.
 * @return Bytes per encoded vector, or 0 on error.
 */
size_t gv_quant_code_size(const GV_QuantCodebook *cb, size_t dimension);

/**
 * @brief Save a codebook to a file path.
 *
 * @param cb   Codebook to save.
 * @param path Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_quant_codebook_save(const GV_QuantCodebook *cb, const char *path);

/**
 * @brief Load a codebook from a file path.
 *
 * @param path Input file path.
 * @return Loaded codebook, or NULL on error.
 */
GV_QuantCodebook *gv_quant_codebook_load(const char *path);

/**
 * @brief Destroy a codebook and free all associated memory.
 *
 * Safe to call with NULL.
 *
 * @param cb Codebook to destroy.
 */
void gv_quant_codebook_destroy(GV_QuantCodebook *cb);

/**
 * @brief Compute the memory compression ratio vs float32.
 *
 * Returns sizeof(float)*dimension / code_size.  For example 8-bit
 * quantization of 128-d vectors yields a ratio of 4.0.
 *
 * @param cb        Trained codebook.
 * @param dimension Vector dimensionality.
 * @return Compression ratio (>= 1.0), or 0.0f on error.
 */
float gv_quant_memory_ratio(const GV_QuantCodebook *cb, size_t dimension);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_QUANTIZATION_H */
