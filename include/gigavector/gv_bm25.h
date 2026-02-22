#ifndef GIGAVECTOR_GV_BM25_H
#define GIGAVECTOR_GV_BM25_H

#include <stddef.h>
#include <stdint.h>

#include "gv_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_bm25.h
 * @brief BM25 full-text search index.
 *
 * Implements the Okapi BM25 ranking function for text retrieval.
 */

/**
 * @brief BM25 configuration.
 */
typedef struct {
    double k1;                      /**< Term frequency saturation (default: 1.2). */
    double b;                       /**< Length normalization (default: 0.75). */
    GV_TokenizerConfig tokenizer;   /**< Tokenizer configuration. */
} GV_BM25Config;

/**
 * @brief BM25 search result.
 */
typedef struct {
    size_t doc_id;                  /**< Document ID (vector index). */
    double score;                   /**< BM25 score. */
} GV_BM25Result;

/**
 * @brief BM25 index statistics.
 */
typedef struct {
    size_t total_documents;         /**< Total number of documents. */
    size_t total_terms;             /**< Total unique terms in index. */
    size_t total_postings;          /**< Total term-document pairs. */
    double avg_document_length;     /**< Average document length. */
    size_t memory_bytes;            /**< Estimated memory usage. */
} GV_BM25Stats;

/**
 * @brief Opaque BM25 index handle.
 */
typedef struct GV_BM25Index GV_BM25Index;

/* Configuration */

/**
 * @brief Initialize BM25 configuration with defaults.
 *
 * Default values:
 * - k1: 1.2 (term frequency saturation)
 * - b: 0.75 (length normalization)
 * - tokenizer: simple tokenizer with lowercase
 *
 * @param config Configuration to initialize.
 */
void gv_bm25_config_init(GV_BM25Config *config);

/* Index Lifecycle */

/**
 * @brief Create a BM25 index.
 *
 * @param config BM25 configuration (NULL for defaults).
 * @return BM25 index instance, or NULL on error.
 */
GV_BM25Index *gv_bm25_create(const GV_BM25Config *config);

/**
 * @brief Destroy a BM25 index.
 *
 * @param index BM25 index instance (safe to call with NULL).
 */
void gv_bm25_destroy(GV_BM25Index *index);

/* Indexing Operations */

/**
 * @brief Add a document to the index.
 *
 * @param index BM25 index.
 * @param doc_id Document ID (typically the vector index).
 * @param text Document text.
 * @return 0 on success, -1 on error.
 */
int gv_bm25_add_document(GV_BM25Index *index, size_t doc_id, const char *text);

/**
 * @brief Add a document with pre-tokenized terms.
 *
 * @param index BM25 index.
 * @param doc_id Document ID.
 * @param terms Array of terms.
 * @param term_count Number of terms.
 * @return 0 on success, -1 on error.
 */
int gv_bm25_add_document_terms(GV_BM25Index *index, size_t doc_id,
                                const char **terms, size_t term_count);

/**
 * @brief Remove a document from the index.
 *
 * @param index BM25 index.
 * @param doc_id Document ID to remove.
 * @return 0 on success, -1 on error (not found).
 */
int gv_bm25_remove_document(GV_BM25Index *index, size_t doc_id);

/**
 * @brief Update a document in the index.
 *
 * Equivalent to remove + add.
 *
 * @param index BM25 index.
 * @param doc_id Document ID.
 * @param text New document text.
 * @return 0 on success, -1 on error.
 */
int gv_bm25_update_document(GV_BM25Index *index, size_t doc_id, const char *text);

/* Search Operations */

/**
 * @brief Search the index with a text query.
 *
 * @param index BM25 index.
 * @param query Query text.
 * @param k Maximum results to return.
 * @param results Output results array (must be pre-allocated with k elements).
 * @return Number of results found, or -1 on error.
 */
int gv_bm25_search(GV_BM25Index *index, const char *query, size_t k,
                   GV_BM25Result *results);

/**
 * @brief Search with pre-tokenized query terms.
 *
 * @param index BM25 index.
 * @param terms Query terms.
 * @param term_count Number of terms.
 * @param k Maximum results.
 * @param results Output results array.
 * @return Number of results found, or -1 on error.
 */
int gv_bm25_search_terms(GV_BM25Index *index, const char **terms, size_t term_count,
                          size_t k, GV_BM25Result *results);

/**
 * @brief Get BM25 score for a specific document and query.
 *
 * @param index BM25 index.
 * @param doc_id Document ID.
 * @param query Query text.
 * @param score Output score.
 * @return 0 on success, -1 on error.
 */
int gv_bm25_score_document(GV_BM25Index *index, size_t doc_id, const char *query,
                            double *score);

/* Index Information */

/**
 * @brief Get index statistics.
 *
 * @param index BM25 index.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int gv_bm25_get_stats(const GV_BM25Index *index, GV_BM25Stats *stats);

/**
 * @brief Get document frequency for a term.
 *
 * @param index BM25 index.
 * @param term Term to look up.
 * @return Document frequency, or 0 if term not found.
 */
size_t gv_bm25_get_doc_freq(const GV_BM25Index *index, const char *term);

/**
 * @brief Check if a document exists in the index.
 *
 * @param index BM25 index.
 * @param doc_id Document ID.
 * @return 1 if exists, 0 if not.
 */
int gv_bm25_has_document(const GV_BM25Index *index, size_t doc_id);

/* Persistence */

/**
 * @brief Save index to file.
 *
 * @param index BM25 index.
 * @param filepath Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_bm25_save(const GV_BM25Index *index, const char *filepath);

/**
 * @brief Load index from file.
 *
 * @param filepath Input file path.
 * @return BM25 index, or NULL on error.
 */
GV_BM25Index *gv_bm25_load(const char *filepath);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_BM25_H */
