#ifndef GIGAVECTOR_GV_LEARNED_SPARSE_H
#define GIGAVECTOR_GV_LEARNED_SPARSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_learned_sparse.h
 * @brief Learned sparse vector index (SPLADE, BGE-M3 sparse, etc.).
 *
 * Specialized inverted index for neural learned sparse representations.
 * Unlike traditional BM25 term frequencies, learned sparse vectors have
 * float weights per vocabulary token produced by neural models.  Scoring
 * is a dot product over the shared non-zero token dimensions.
 *
 * Supports optional WAND (Weighted AND) pruning for efficient top-k
 * retrieval.
 */

/* Configuration */

/**
 * @brief Configuration for a learned sparse index.
 */
typedef struct {
    size_t vocab_size;          /**< Vocabulary size (e.g., 30522 for BERT). */
    size_t max_nonzeros;        /**< Max non-zero entries per vector (default: 256). */
    int    use_wand;            /**< Enable WAND optimization (default: 1). */
    size_t wand_block_size;     /**< Block size for WAND upper-bound tracking (default: 128). */
} GV_LearnedSparseConfig;

/* Core Types */

/**
 * @brief Opaque learned sparse index handle.
 */
typedef struct GV_LearnedSparseIndex GV_LearnedSparseIndex;

/**
 * @brief A single non-zero entry in a learned sparse vector.
 */
typedef struct {
    uint32_t token_id;          /**< Vocabulary token ID. */
    float    weight;            /**< Learned weight for this token. */
} GV_SparseEntry;

/**
 * @brief Search result from a learned sparse query.
 */
typedef struct {
    size_t doc_index;           /**< Document index (insertion order). */
    float  score;               /**< Dot-product score. */
} GV_LearnedSparseResult;

/**
 * @brief Index statistics.
 */
typedef struct {
    size_t doc_count;           /**< Number of active (non-deleted) documents. */
    size_t total_postings;      /**< Total entries across all posting lists. */
    double avg_doc_length;      /**< Average non-zero entries per document. */
    size_t vocab_used;          /**< Number of distinct tokens with postings. */
} GV_LearnedSparseStats;

/* Configuration */

/**
 * @brief Initialize configuration with defaults.
 *
 * Default values:
 * - vocab_size: 30522 (BERT WordPiece vocabulary)
 * - max_nonzeros: 256
 * - use_wand: 1
 * - wand_block_size: 128
 *
 * @param config Configuration to initialize.
 */
void gv_ls_config_init(GV_LearnedSparseConfig *config);

/* Index Lifecycle */

/**
 * @brief Create a learned sparse index.
 *
 * @param config Index configuration (NULL for defaults).
 * @return Index instance, or NULL on error.
 */
GV_LearnedSparseIndex *gv_ls_create(const GV_LearnedSparseConfig *config);

/**
 * @brief Destroy a learned sparse index and free all resources.
 *
 * @param idx Index instance (safe to call with NULL).
 */
void gv_ls_destroy(GV_LearnedSparseIndex *idx);

/* Indexing Operations */

/**
 * @brief Insert a learned sparse vector into the index.
 *
 * The entries array is copied; the caller retains ownership.
 *
 * @param idx   Index instance; must be non-NULL.
 * @param entries Array of non-zero sparse entries.
 * @param count   Number of entries (must be <= max_nonzeros).
 * @return Assigned document ID (>= 0) on success, -1 on error.
 */
int gv_ls_insert(GV_LearnedSparseIndex *idx, const GV_SparseEntry *entries,
                 size_t count);

/**
 * @brief Delete a document from the index by its document ID.
 *
 * Performs a logical (soft) delete.
 *
 * @param idx    Index instance; must be non-NULL.
 * @param doc_id Document ID to delete.
 * @return 0 on success, -1 on error (not found or already deleted).
 */
int gv_ls_delete(GV_LearnedSparseIndex *idx, size_t doc_id);

/* Search Operations */

/**
 * @brief Search for top-k documents by dot-product score.
 *
 * Uses WAND optimization if enabled in the configuration, otherwise
 * falls back to simple score accumulation.
 *
 * @param idx         Index instance; must be non-NULL.
 * @param query       Query sparse entries.
 * @param query_count Number of query entries.
 * @param k           Maximum results to return.
 * @param results     Output results array (must be pre-allocated with k elements).
 * @return Number of results found (0..k), or -1 on error.
 */
int gv_ls_search(const GV_LearnedSparseIndex *idx, const GV_SparseEntry *query,
                 size_t query_count, size_t k, GV_LearnedSparseResult *results);

/**
 * @brief Search with a minimum score threshold.
 *
 * Only results with score >= min_score are returned.
 *
 * @param idx         Index instance; must be non-NULL.
 * @param query       Query sparse entries.
 * @param query_count Number of query entries.
 * @param min_score   Minimum score threshold.
 * @param k           Maximum results to return.
 * @param results     Output results array (must be pre-allocated with k elements).
 * @return Number of results found, or -1 on error.
 */
int gv_ls_search_with_threshold(const GV_LearnedSparseIndex *idx,
                                const GV_SparseEntry *query, size_t query_count,
                                float min_score, size_t k,
                                GV_LearnedSparseResult *results);

/* Index Information */

/**
 * @brief Get index statistics.
 *
 * @param idx   Index instance; must be non-NULL.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int gv_ls_get_stats(const GV_LearnedSparseIndex *idx, GV_LearnedSparseStats *stats);

/**
 * @brief Get the number of active (non-deleted) documents.
 *
 * @param idx Index instance; must be non-NULL.
 * @return Document count, or 0 if idx is NULL.
 */
size_t gv_ls_count(const GV_LearnedSparseIndex *idx);

/* Persistence */

/**
 * @brief Save index to file.
 *
 * @param idx      Index instance; must be non-NULL.
 * @param path     Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_ls_save(const GV_LearnedSparseIndex *idx, const char *path);

/**
 * @brief Load index from file.
 *
 * @param path Input file path.
 * @return Index instance, or NULL on error.
 */
GV_LearnedSparseIndex *gv_ls_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_LEARNED_SPARSE_H */
