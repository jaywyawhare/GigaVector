#ifndef GIGAVECTOR_GV_RECALL_H
#define GIGAVECTOR_GV_RECALL_H

#include <stddef.h>

#include "search/distance.h"
#include "storage/database.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double recall;   /**< Mean recall@k over all queries, in [0, 1]. */
    size_t queries;  /**< Number of queries evaluated. */
    size_t k;        /**< Neighbours requested per query (clamped to vector count). */
} GV_RecallReport;

/**
 * @brief Measure recall@k of the database's ANN search against exact ground truth.
 *
 * For each query, computes the exact top-k by brute force over the database's own
 * stored vectors (skipping deletes), runs @ref db_search for the ANN top-k, and
 * reports the mean overlap fraction. This is the standard quality metric for tuning
 * index parameters (ef / nprobe) against measured recall.
 *
 * @param db      Database to evaluate; must be non-NULL and non-empty.
 * @param queries Row-major query matrix of @p nq × @p dim floats.
 * @param nq      Number of queries (> 0).
 * @param dim     Query dimension (must equal the database dimension).
 * @param k       Neighbours per query (> 0; clamped to the stored vector count).
 * @param metric  Distance metric to use for both ground truth and ANN.
 * @param out     Output report; must be non-NULL.
 * @return 0 on success, -1 on invalid arguments or an empty database.
 */
int db_evaluate_recall(const GV_Database *db, const float *queries, size_t nq,
                       size_t dim, size_t k, GV_DistanceType metric,
                       GV_RecallReport *out);

#ifdef __cplusplus
}
#endif

#endif
