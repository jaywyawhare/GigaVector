#ifndef GIGAVECTOR_GV_BINARY_QUANT_H
#define GIGAVECTOR_GV_BINARY_QUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Binary quantized vector representation.
 *
 * Each dimension is represented by 1 bit (0 for negative, 1 for positive/zero).
 * This provides up to 32x memory compression compared to float32 vectors.
 */
typedef struct {
    uint8_t *bits;          /**< Bit array, packed 8 bits per byte */
    size_t dimension;        /**< Original vector dimension */
    size_t bytes_per_vector; /**< Number of bytes needed for this dimension */
} GV_BinaryVector;

/**
 * @brief Quantize a float vector to binary representation.
 *
 * Each dimension is converted to 1 bit: 0 if value < 0, 1 if value >= 0.
 *
 * @param data Input float vector of dimension @p dimension.
 * @param dimension Vector dimensionality.
 * @return Allocated binary vector, or NULL on error.
 */
GV_BinaryVector *gv_binary_quantize(const float *data, size_t dimension);

/**
 * @brief Free a binary quantized vector.
 *
 * @param bv Binary vector to free; safe to call with NULL.
 */
void gv_binary_vector_destroy(GV_BinaryVector *bv);

/**
 * @brief Calculate Hamming distance between two binary vectors.
 *
 * Hamming distance is the number of differing bits. Lower distance indicates
 * higher similarity in the quantized space.
 *
 * @param a First binary vector; must be non-NULL with matching dimension.
 * @param b Second binary vector; must be non-NULL with matching dimension.
 * @return Hamming distance (0 to dimension), or SIZE_MAX on error.
 */
size_t gv_binary_hamming_distance(const GV_BinaryVector *a, const GV_BinaryVector *b);

/**
 * @brief Calculate Hamming distance with SIMD optimization.
 *
 * Uses popcount instructions when available for faster computation.
 *
 * @param a First binary vector; must be non-NULL with matching dimension.
 * @param b Second binary vector; must be non-NULL with matching dimension.
 * @return Hamming distance (0 to dimension), or SIZE_MAX on error.
 */
size_t gv_binary_hamming_distance_fast(const GV_BinaryVector *a, const GV_BinaryVector *b);

/**
 * @brief Get the number of bytes needed to store a binary vector of given dimension.
 *
 * @param dimension Vector dimensionality.
 * @return Number of bytes needed (rounded up).
 */
size_t gv_binary_bytes_needed(size_t dimension);

/**
 * @brief Create a binary vector from pre-allocated bit array.
 *
 * @param bits Pre-allocated bit array (ownership not transferred).
 * @param dimension Vector dimensionality.
 * @return Allocated binary vector wrapper, or NULL on error.
 */
GV_BinaryVector *gv_binary_vector_wrap(uint8_t *bits, size_t dimension);

#ifdef __cplusplus
}
#endif

#endif

