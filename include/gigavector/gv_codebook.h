#ifndef GIGAVECTOR_GV_CODEBOOK_H
#define GIGAVECTOR_GV_CODEBOOK_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Shareable product quantization codebook.
 *
 * Contains the trained centroids that can be saved, loaded, copied, and
 * shared across multiple indices.  The struct is fully defined (not opaque)
 * so that callers may inspect its fields directly.
 */
typedef struct {
    size_t   dimension;   /**< Original vector dimension.                        */
    size_t   m;           /**< Number of subspaces.                              */
    size_t   ksub;        /**< Centroids per subspace (= 1 << nbits).           */
    uint8_t  nbits;       /**< Bits per sub-quantizer code.                      */
    size_t   dsub;        /**< Sub-vector dimension (= dimension / m).           */
    float   *centroids;   /**< Centroid data: m * ksub * dsub floats.            */
    int      trained;     /**< Non-zero after successful training.               */
} GV_Codebook;

/**
 * @brief Allocate a new, untrained codebook.
 *
 * @param dimension  Full vector dimension (must be divisible by @p m).
 * @param m          Number of subspaces.
 * @param nbits      Bits per code (max 8).
 * @return Newly allocated codebook, or NULL on error.
 */
GV_Codebook *gv_codebook_create(size_t dimension, size_t m, uint8_t nbits);

/**
 * @brief Free a codebook and all associated memory.
 */
void gv_codebook_destroy(GV_Codebook *cb);

/**
 * @brief Train the codebook via K-means on the supplied data.
 *
 * @param cb          Codebook to train.
 * @param data        Row-major training vectors (count * dimension floats).
 * @param count       Number of training vectors.
 * @param train_iters Number of K-means iterations.
 * @return 0 on success, -1 on error.
 */
int gv_codebook_train(GV_Codebook *cb, const float *data, size_t count,
                      size_t train_iters);

/**
 * @brief Encode a single vector into @p m sub-quantizer codes.
 *
 * @param cb     Trained codebook.
 * @param vector Input vector (dimension floats).
 * @param codes  Output array (m bytes).
 * @return 0 on success, -1 on error.
 */
int gv_codebook_encode(const GV_Codebook *cb, const float *vector,
                       uint8_t *codes);

/**
 * @brief Decode sub-quantizer codes back to an approximate vector.
 *
 * @param cb     Trained codebook.
 * @param codes  Input codes (m bytes).
 * @param output Output vector (dimension floats).
 * @return 0 on success, -1 on error.
 */
int gv_codebook_decode(const GV_Codebook *cb, const uint8_t *codes,
                       float *output);

/**
 * @brief Compute asymmetric distance between a raw query and PQ codes.
 *
 * Builds a per-subspace distance lookup table from @p query, then sums the
 * looked-up squared distances for each code.  Returns the Euclidean distance
 * (i.e. the square root of the sum).
 *
 * @param cb    Trained codebook.
 * @param query Raw query vector (dimension floats).
 * @param codes Encoded database vector (m bytes).
 * @return Approximate Euclidean distance, or -1.0f on error.
 */
float gv_codebook_distance_adc(const GV_Codebook *cb, const float *query,
                               const uint8_t *codes);

/**
 * @brief Save the codebook to a file path.
 *
 * @return 0 on success, -1 on error.
 */
int gv_codebook_save(const GV_Codebook *cb, const char *filepath);

/**
 * @brief Load a codebook from a file path.
 *
 * @return Loaded codebook, or NULL on error.
 */
GV_Codebook *gv_codebook_load(const char *filepath);

/**
 * @brief Save the codebook to an already-open FILE stream.
 *
 * @return 0 on success, -1 on error.
 */
int gv_codebook_save_fp(const GV_Codebook *cb, FILE *out);

/**
 * @brief Load a codebook from an already-open FILE stream.
 *
 * @return Loaded codebook, or NULL on error.
 */
GV_Codebook *gv_codebook_load_fp(FILE *in);

/**
 * @brief Create a deep copy of a codebook.
 *
 * @return New codebook with its own centroid storage, or NULL on error.
 */
GV_Codebook *gv_codebook_copy(const GV_Codebook *cb);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CODEBOOK_H */
