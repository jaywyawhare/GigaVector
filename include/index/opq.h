#ifndef GIGAVECTOR_GV_OPQ_H
#define GIGAVECTOR_GV_OPQ_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file opq.h
 * @brief Optimized Product Quantization rotation (parametric OPQ).
 *
 * Learns an orthonormal rotation R from training data that (1) aligns axes with
 * the data's principal components and (2) permutes them so each of @p m
 * sub-quantizers receives a balanced share of the variance. Applying R before PQ
 * lowers quantization error — and thus raises recall — at the same bit budget.
 * Because R is orthonormal, Euclidean distances are preserved, so PQ's ADC
 * distance tables stay valid on the rotated space.
 */

typedef struct GV_OPQ GV_OPQ;

/**
 * @brief Train a rotation from @p count vectors of @p dim dimensions (row-major),
 *        balancing variance across @p m sub-quantizers.
 * @return New rotation, or NULL on error.
 */
GV_OPQ *opq_train(size_t dim, size_t m, const float *data, size_t count);

/** @brief Apply the rotation: out[dim] = R * in[dim]. */
void opq_rotate(const GV_OPQ *opq, const float *in, float *out);

/** @brief Dimension the rotation operates on. */
size_t opq_dimension(const GV_OPQ *opq);

/** @brief The dim×dim row-major rotation matrix (for serialization). */
const float *opq_matrix(const GV_OPQ *opq);

/** @brief Rebuild a rotation from a saved dim×dim row-major matrix. */
GV_OPQ *opq_from_matrix(size_t dim, const float *matrix);

/** @brief Free a rotation (safe with NULL). */
void opq_free(GV_OPQ *opq);

#ifdef __cplusplus
}
#endif

#endif
