#ifndef GIGAVECTOR_GV_SCALAR_QUANT_H
#define GIGAVECTOR_GV_SCALAR_QUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scalar quantization configuration.
 */
typedef struct {
    uint8_t bits;        /**< Number of bits per dimension (4, 8, or 16) */
    int per_dimension;   /**< 1 to use per-dimension min/max, 0 for global */
} GV_ScalarQuantConfig;

/**
 * @brief Scalar quantized vector representation.
 *
 * Stores quantized vectors with min/max values for dequantization.
 */
typedef struct {
    uint8_t *quantized;      /**< Quantized data (bits per dimension, packed) */
    float *min_vals;         /**< Minimum values (per dimension or single global) */
    float *max_vals;         /**< Maximum values (per dimension or single global) */
    size_t dimension;        /**< Vector dimensionality */
    uint8_t bits;            /**< Bits per dimension */
    int per_dimension;       /**< Whether min/max are per-dimension */
    size_t bytes_per_vector; /**< Bytes needed for quantized data */
} GV_ScalarQuantVector;

/**
 * @brief Quantize a float vector using scalar quantization.
 *
 * @param data Input float vector of dimension @p dimension.
 * @param dimension Vector dimensionality.
 * @param config Quantization configuration.
 * @return Allocated quantized vector, or NULL on error.
 */
GV_ScalarQuantVector *gv_scalar_quantize(const float *data, size_t dimension, const GV_ScalarQuantConfig *config);

/**
 * @brief Quantize vectors and compute min/max from training data.
 *
 * This function analyzes training data to determine optimal min/max values,
 * then quantizes all vectors using these values.
 *
 * @param data Training vectors (count * dimension floats).
 * @param count Number of training vectors.
 * @param dimension Vector dimensionality.
 * @param config Quantization configuration.
 * @return Allocated quantized vector structure with computed min/max, or NULL on error.
 */
GV_ScalarQuantVector *gv_scalar_quantize_train(const float *data, size_t count, size_t dimension,
                                                 const GV_ScalarQuantConfig *config);

/**
 * @brief Dequantize a scalar quantized vector back to float.
 *
 * @param sqv Quantized vector; must be non-NULL.
 * @param output Output buffer of at least dimension floats.
 * @return 0 on success, -1 on error.
 */
int gv_scalar_dequantize(const GV_ScalarQuantVector *sqv, float *output);

/**
 * @brief Calculate distance between a float vector and a quantized vector.
 *
 * @param query Float query vector.
 * @param sqv Quantized vector.
 * @param distance_type Distance metric to use.
 * @return Distance value, or negative on error.
 */
float gv_scalar_quant_distance(const float *query, const GV_ScalarQuantVector *sqv, int distance_type);

/**
 * @brief Free a scalar quantized vector.
 *
 * @param sqv Quantized vector to free; safe to call with NULL.
 */
void gv_scalar_quant_vector_destroy(GV_ScalarQuantVector *sqv);

/**
 * @brief Get the number of bytes needed for quantized data.
 *
 * @param dimension Vector dimensionality.
 * @param bits Bits per dimension.
 * @return Number of bytes needed (rounded up).
 */
size_t gv_scalar_quant_bytes_needed(size_t dimension, uint8_t bits);

#ifdef __cplusplus
}
#endif

#endif

