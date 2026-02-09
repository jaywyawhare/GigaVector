#ifndef GIGAVECTOR_GV_MULTIVEC_H
#define GIGAVECTOR_GV_MULTIVEC_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "gv_types.h"
#include "gv_distance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Aggregation strategy for combining per-chunk scores into a
 *        document-level score during multi-vector search.
 */
typedef enum {
    GV_DOC_AGG_MAX_SIM = 0,  /**< Use the maximum chunk similarity. */
    GV_DOC_AGG_AVG_SIM = 1,  /**< Use the average chunk similarity. */
    GV_DOC_AGG_SUM_SIM = 2   /**< Use the sum of chunk similarities. */
} GV_DocAggregation;

/**
 * @brief Configuration for a multi-vector index.
 */
typedef struct {
    size_t max_chunks_per_doc;    /**< Maximum number of chunks allowed per document. */
    GV_DocAggregation aggregation; /**< Aggregation strategy for document scoring. */
} GV_MultiVecConfig;

/**
 * @brief A single document-level search result.
 */
typedef struct {
    uint64_t doc_id;           /**< Document identifier. */
    float score;               /**< Aggregated document score (lower = closer). */
    size_t num_chunks;         /**< Number of chunks in this document. */
    size_t best_chunk_index;   /**< Index of the best-matching chunk within the document. */
} GV_DocSearchResult;

/**
 * @brief Create a new multi-vector index.
 *
 * @param dimension Vector dimensionality for all chunks.
 * @param config Configuration parameters; NULL for defaults (256 max chunks, max-sim).
 * @return Allocated multi-vector index, or NULL on error.
 */
void *gv_multivec_create(size_t dimension, const GV_MultiVecConfig *config);

/**
 * @brief Destroy a multi-vector index and free all resources.
 *
 * Safe to call with NULL; no action is taken.
 *
 * @param index Multi-vector index instance.
 */
void gv_multivec_destroy(void *index);

/**
 * @brief Add a document with one or more chunk vectors.
 *
 * The chunk data is copied; the caller retains ownership of the input buffer.
 * The chunks array must contain num_chunks vectors of the given dimension,
 * laid out contiguously (chunk i starts at chunks[i * dimension]).
 *
 * @param index Multi-vector index instance; must be non-NULL.
 * @param doc_id Unique document identifier.
 * @param chunks Contiguous array of chunk vectors (num_chunks * dimension floats).
 * @param num_chunks Number of chunks in the document.
 * @param dimension Dimensionality of each chunk vector; must match the index dimension.
 * @return 0 on success, -1 on error.
 */
int gv_multivec_add_document(void *index, uint64_t doc_id,
                             const float *chunks, size_t num_chunks,
                             size_t dimension);

/**
 * @brief Delete a document by its identifier.
 *
 * Marks the document as deleted and frees its chunk data.
 *
 * @param index Multi-vector index instance; must be non-NULL.
 * @param doc_id Document identifier to delete.
 * @return 0 on success, -1 if not found or on error.
 */
int gv_multivec_delete_document(void *index, uint64_t doc_id);

/**
 * @brief Search for the top-k most similar documents to a query vector.
 *
 * For each document, the query is compared against every chunk and the
 * per-chunk distances are aggregated according to the index configuration.
 *
 * @param index Multi-vector index instance; must be non-NULL.
 * @param query Query vector (dimension floats).
 * @param k Maximum number of results to return.
 * @param results Output array of at least k elements.
 * @param distance_type Distance metric to use.
 * @return Number of results found (0 to k), or -1 on error.
 */
int gv_multivec_search(void *index, const float *query, size_t k,
                       GV_DocSearchResult *results,
                       GV_DistanceType distance_type);

/**
 * @brief Return the number of non-deleted documents in the index.
 *
 * @param index Multi-vector index instance; must be non-NULL.
 * @return Number of documents, or 0 if index is NULL.
 */
size_t gv_multivec_count_documents(const void *index);

/**
 * @brief Return the total number of chunks across all non-deleted documents.
 *
 * @param index Multi-vector index instance; must be non-NULL.
 * @return Total chunk count, or 0 if index is NULL.
 */
size_t gv_multivec_count_chunks(const void *index);

/**
 * @brief Serialize the multi-vector index to a file stream.
 *
 * @param index Multi-vector index instance; must be non-NULL.
 * @param out File stream opened for writing.
 * @return 0 on success, -1 on error.
 */
int gv_multivec_save(const void *index, FILE *out);

/**
 * @brief Deserialize a multi-vector index from a file stream.
 *
 * @param index_ptr Pointer to index pointer (will be allocated).
 * @param in File stream opened for reading.
 * @param dimension Expected vector dimensionality.
 * @return 0 on success, -1 on error.
 */
int gv_multivec_load(void **index_ptr, FILE *in, size_t dimension);

#ifdef __cplusplus
}
#endif

#endif
