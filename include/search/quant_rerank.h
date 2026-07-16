/**
 * @file quant_rerank.h
 * @brief Quantization-aware two-stage reranking pipeline.
 *
 * Implements a two-stage retrieval pipeline:
 *   Stage 1 — coarse ANN retrieval of oversample_factor*k candidates.
 *   Stage 2 — asymmetric quantized distance refinement; return top-k.
 *
 * The caller is responsible for maintaining the codes array in sync with
 * insertions by calling quant_encode() after each db_add_vector().
 * code_stride is obtained once via quant_code_size().
 */

#ifndef GIGAVECTOR_GV_QUANT_RERANK_H
#define GIGAVECTOR_GV_QUANT_RERANK_H

#include <stddef.h>
#include <stdint.h>
#include "specialized/quantization.h"
#include "core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration */
typedef struct GV_Database GV_Database;

/**
 * @brief Configuration for the quantization-aware reranking pipeline.
 *
 * The caller owns the codebook and the flat codes array and must keep
 * codes in sync with the database (call quant_encode after each insert).
 */
typedef struct {
    GV_QuantCodebook *codebook;   /**< Trained quantization codebook. */
    const uint8_t    *codes;      /**< Flat array: one code_stride-byte block per vector. */
    size_t            code_stride; /**< Bytes per encoded vector (from quant_code_size()). */
    size_t            codes_count; /**< Number of encoded vectors in codes[]; candidates with a SoA id >= this are skipped to prevent OOB reads. */
    size_t            oversample_factor; /**< Stage-1 retrieves oversample_factor*k candidates. */
    int               distance_type;     /**< GV_DistanceType for stage-1 ANN search. */
} GV_QuantRerankConfig;

/**
 * @brief Single result from the quantization-aware reranking pipeline.
 */
typedef struct {
    size_t index;         /**< Vector index in the database. */
    float  approx_score;  /**< Stage-1 ANN distance (lower = more similar). */
    float  rerank_score;  /**< Stage-2 asymmetric quant_distance() value. */
} GV_QuantRerankResult;

/**
 * @brief Initialise a GV_QuantRerankConfig to safe defaults.
 *
 * Sets oversample_factor = 4 and distance_type = GV_DISTANCE_L2.
 * codebook, codes, and code_stride are zeroed and must be supplied by the caller.
 *
 * @param cfg Configuration to initialise; must be non-NULL.
 */
void quant_rerank_config_init(GV_QuantRerankConfig *cfg);

/**
 * @brief Two-stage quantization-aware search.
 *
 * Stage 1: retrieve oversample_factor*k ANN candidates via db_search().
 * Stage 2: for each candidate, compute quant_distance() (asymmetric:
 *          raw float query vs stored quantized code) and re-sort by
 *          rerank_score ascending. The top-k are written to @p out.
 *
 * @param db    Database handle; must be non-NULL.
 * @param query Query vector (dimension floats); must be non-NULL.
 * @param k     Number of final results to return.
 * @param cfg   Reranking configuration; must be non-NULL with a valid codebook and codes.
 * @param out   Output array of at least @p k GV_QuantRerankResult elements.
 * @return Number of results written (0 to k), or -1 on error.
 */
int quant_rerank_search(GV_Database *db, const float *query, size_t k,
                        const GV_QuantRerankConfig *cfg,
                        GV_QuantRerankResult *out);

/**
 * @brief Simplified reranking search using exact float32 distances.
 *
 * Retrieves @p oversample * @p k ANN candidates, then reranks them using
 * exact L2 distance computed against the raw float32 vectors stored in the
 * database.  Returns results in @p out as standard GV_SearchResult with
 * updated distances.
 *
 * This function does not require a pre-trained codebook and can be used
 * as a drop-in replacement for db_search() when higher recall is needed.
 *
 * @param db         Database handle; must be non-NULL.
 * @param query      Query vector; must be non-NULL.
 * @param k          Number of final results.
 * @param oversample Stage-1 retrieves oversample*k candidates (minimum 1).
 * @param out        Output array of at least @p k GV_SearchResult elements.
 * @return Number of results written (0 to k), or -1 on error.
 */
int gv_db_search_with_rerank(GV_Database *db, const float *query, size_t k,
                              size_t oversample, GV_SearchResult *out);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_QUANT_RERANK_H */
