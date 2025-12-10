#ifndef GIGAVECTOR_GV_IVFPQ_H
#define GIGAVECTOR_GV_IVFPQ_H

#include <stddef.h>
#include <stdint.h>

#include "gv_types.h"
#include "gv_distance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief IVF-PQ configuration.
 */
typedef struct {
    size_t nlist;      /**< Number of coarse centroids (inverted lists). */
    size_t m;          /**< Number of subquantizers (must divide dimension). */
    uint8_t nbits;     /**< Bits per subquantizer code (typically 8). */
    size_t nprobe;     /**< Lists to probe during search (<= nlist). */
    size_t train_iters;/**< K-means iterations for coarse/PQ codebooks. */
    size_t default_rerank; /**< Default rerank pool size (0 to disable). */
    int use_cosine;     /**< Treat vectors as cosine (normalize query; LUT uses dot). */
} GV_IVFPQConfig;

/**
 * @brief Create an IVF-PQ index.
 *
 * @param dimension Vector dimensionality.
 * @param config Optional config; if NULL, defaults are used.
 * @return Allocated index or NULL on error.
 */
void *gv_ivfpq_create(size_t dimension, const GV_IVFPQConfig *config);

/**
 * @brief Train coarse centroids and PQ codebooks.
 *
 * @param index IVF-PQ index (from gv_ivfpq_create).
 * @param data Contiguous training vectors (count * dimension floats).
 * @param count Number of training vectors.
 * @return 0 on success, -1 on error.
 */
int gv_ivfpq_train(void *index, const float *data, size_t count);

/**
 * @brief Insert a vector into IVF-PQ (requires training first).
 *
 * @param index IVF-PQ index.
 * @param vector Ownership of the vector transfers to the index.
 * @return 0 on success, -1 on error.
 */
int gv_ivfpq_insert(void *index, GV_Vector *vector);

/**
 * @brief Search IVF-PQ for k nearest neighbors.
 *
 * @param index IVF-PQ index.
 * @param query Query vector.
 * @param k Number of neighbors.
 * @param results Output array of at least k elements.
 * @param distance_type Distance metric (currently supports GV_DISTANCE_EUCLIDEAN).
 * @param nprobe_override If >0, use this nprobe instead of configured; capped at nlist.
 * @param rerank_top If >0, rerank this many top PQ results with exact L2 on raw vectors.
 * @return Number of neighbors found (0..k) or -1 on error.
 */
int gv_ivfpq_search(void *index, const GV_Vector *query, size_t k,
                    GV_SearchResult *results, GV_DistanceType distance_type,
                    size_t nprobe_override, size_t rerank_top);

/**
 * @brief Destroy IVF-PQ index and free resources.
 *
 * @param index IVF-PQ index; safe to call with NULL.
 */
void gv_ivfpq_destroy(void *index);

/**
 * @brief Save IVF-PQ index to file.
 *
 * @param index IVF-PQ index.
 * @param out FILE* opened for writing.
 * @param version File format version.
 * @return 0 on success, -1 on error.
 */
int gv_ivfpq_save(const void *index, FILE *out, uint32_t version);

/**
 * @brief Load IVF-PQ index from file.
 *
 * @param index_ptr Output pointer to allocated index.
 * @param in FILE* opened for reading.
 * @param dimension Expected dimension.
 * @param version File version.
 * @return 0 on success, -1 on error.
 */
int gv_ivfpq_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version);

/**
 * @brief Count vectors in IVF-PQ.
 */
size_t gv_ivfpq_count(const void *index);

/**
 * @brief Check whether the IVF-PQ index has completed training.
 *
 * @param index IVF-PQ index handle.
 * @return 1 if trained, 0 otherwise.
 */
int gv_ivfpq_is_trained(const void *index);

#ifdef __cplusplus
}
#endif

#endif

