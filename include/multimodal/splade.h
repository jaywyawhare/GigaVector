#ifndef GIGAVECTOR_GV_SPLADE_H
#define GIGAVECTOR_GV_SPLADE_H

#include <stddef.h>

#include "multimodal/learned_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file splade.h
 * @brief SPLADE learned-sparse text encoder.
 *
 * Turns raw text into the `(token_id, weight)` sparse entries consumed by the
 * learned-sparse index (@ref learned_sparse.h), bridging the gap between text
 * and the term-weight representation that SPLADE models produce.
 *
 * Two back-ends, selected automatically:
 *
 *   - **Neural (SPLADE)** — when built with `GV_HAVE_ONNX` and given a SPLADE
 *     MLM model path, the text is run through the model and the classic SPLADE
 *     pooling `w_j = max_i log(1 + ReLU(logit_ij))` over the vocabulary yields a
 *     sparse term-weight vector.
 *
 *   - **Deterministic fallback** — otherwise, text is tokenised and each unique
 *     term is mapped to a stable vocabulary slot with a saturating
 *     `log(1 + tf)` weight. This keeps the whole text -> encode -> index ->
 *     search pipeline working (and testable) without a model; shared terms
 *     between a query and a document still produce a dot-product overlap.
 */

typedef struct {
    size_t vocab_size;          /**< Vocabulary slots for the fallback hash (default 30522). */
    size_t top_k;               /**< Max non-zero terms to keep per text (default 128). */
    const char *onnx_model_path;/**< SPLADE MLM .onnx model; NULL => fallback encoder. */
} GV_SpladeConfig;

typedef struct GV_SpladeEncoder GV_SpladeEncoder;

/** @brief Fill @p config with defaults (vocab_size 30522, top_k 128, no model). */
void splade_config_init(GV_SpladeConfig *config);

/**
 * @brief Create an encoder. Loads the ONNX SPLADE model when a path is given and
 *        the build has `GV_HAVE_ONNX`; otherwise uses the deterministic fallback.
 * @return Encoder instance, or NULL on allocation failure.
 */
GV_SpladeEncoder *splade_create(const GV_SpladeConfig *config);

/** @brief Destroy an encoder (NULL-safe). */
void splade_destroy(GV_SpladeEncoder *enc);

/** @brief 1 if the encoder uses a neural SPLADE model, 0 if the fallback. */
int splade_is_neural(const GV_SpladeEncoder *enc);

/**
 * @brief Encode text into learned-sparse entries, sorted by descending weight.
 *
 * @param enc        Encoder instance.
 * @param text       Input text (null-terminated).
 * @param out        Receives a heap array of entries (free with splade_free_entries).
 * @param out_count  Receives the number of entries (0..top_k).
 * @return 0 on success, -1 on error.
 */
int splade_encode(GV_SpladeEncoder *enc, const char *text,
                  GV_LSSparseEntry **out, size_t *out_count);

/** @brief Free an entries array returned by splade_encode (NULL-safe). */
void splade_free_entries(GV_LSSparseEntry *entries);

/**
 * @brief Encode @p text and insert it into a learned-sparse index.
 * @return The index-assigned document ID (>= 0), or -1 on error.
 */
int splade_index_add(GV_SpladeEncoder *enc, GV_LearnedSparseIndex *idx,
                     const char *text);

/**
 * @brief Encode a query and search a learned-sparse index.
 * @param results Pre-allocated array of @p k results.
 * @return Number of results (0..k), or -1 on error.
 */
int splade_index_search(GV_SpladeEncoder *enc, const GV_LearnedSparseIndex *idx,
                        const char *query_text, size_t k,
                        GV_LearnedSparseResult *results);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SPLADE_H */
