#ifndef GIGAVECTOR_GV_HNSW_OPT_H
#define GIGAVECTOR_GV_HNSW_OPT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_hnsw_opt.h
 * @brief Optimized HNSW index with inline quantized storage and incremental rebuild.
 *
 * Two key optimizations over the standard HNSW implementation:
 *   1. Inline storage: each graph node embeds a scalar-quantized copy of its
 *      vector (4-bit or 8-bit), eliminating separate vector lookups during
 *      candidate selection.
 *   2. Incremental rebuild: reuses the existing graph structure, iterating
 *      over nodes in batches, searching for better neighbor candidates, and
 *      pruning with the standard HNSW heuristic.  Can run in a background
 *      thread.
 */

/* ============================================================================
 * Configuration Structures
 * ============================================================================ */

/**
 * @brief Configuration for inline quantized vector storage and prefetch.
 */
typedef struct {
    int quant_bits;             /**< Quantization bits per dimension: 4 or 8 (default: 8) */
    int enable_prefetch;        /**< Enable software prefetch during traversal (default: 0) */
    size_t prefetch_distance;   /**< Prefetch N hops ahead in neighbor lists (default: 2) */
} GV_HNSWInlineConfig;

/**
 * @brief Opaque optimized HNSW index with inline quantized storage.
 */
typedef struct GV_HNSWInlineIndex GV_HNSWInlineIndex;

/**
 * @brief Configuration for incremental graph rebuild.
 */
typedef struct {
    float connectivity_ratio;   /**< Fraction of existing edges to retain (default: 0.8) */
    size_t batch_size;          /**< Nodes processed per batch (default: 1000) */
    int background;             /**< Run rebuild in a background thread (default: 0) */
} GV_HNSWRebuildConfig;

/**
 * @brief Statistics returned by an incremental rebuild.
 */
typedef struct {
    size_t nodes_processed;     /**< Total nodes visited during rebuild */
    size_t edges_added;         /**< New edges created */
    size_t edges_removed;       /**< Existing edges pruned */
    double elapsed_ms;          /**< Wall-clock time in milliseconds */
    int completed;              /**< 1 when rebuild has finished, 0 while running */
} GV_HNSWRebuildStats;

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/**
 * @brief Create a new optimized HNSW index with inline quantized storage.
 *
 * @param dimension  Vector dimensionality.
 * @param max_elements  Maximum number of elements the index can hold.
 * @param M  Number of bi-directional links per node (upper layers).
 *           Layer 0 uses 2*M connections.
 * @param ef_construction  Candidate list size during construction.
 * @param config  Inline storage / prefetch configuration; NULL for defaults.
 * @return Allocated index, or NULL on error.
 */
GV_HNSWInlineIndex *gv_hnsw_inline_create(size_t dimension, size_t max_elements,
                                           size_t M, size_t ef_construction,
                                           const GV_HNSWInlineConfig *config);

/**
 * @brief Destroy an optimized HNSW index and free all resources.
 *
 * @param idx  Index to destroy; safe to call with NULL.
 */
void gv_hnsw_inline_destroy(GV_HNSWInlineIndex *idx);

/* ============================================================================
 * Insert / Search
 * ============================================================================ */

/**
 * @brief Insert a vector into the index.
 *
 * The vector is scalar-quantized and stored inline in the graph node.
 * A copy of the full-precision vector is also kept for final reranking.
 *
 * @param idx  Index instance; must be non-NULL.
 * @param vector  Float vector of length dimension.
 * @param label  User-supplied label for the vector.
 * @return 0 on success, -1 on error.
 */
int gv_hnsw_inline_insert(GV_HNSWInlineIndex *idx, const float *vector,
                           size_t label);

/**
 * @brief Search for k approximate nearest neighbors.
 *
 * Uses quantized inline vectors for fast candidate selection, then reranks
 * the top candidates with full-precision distance.
 *
 * @param idx  Index instance; must be non-NULL.
 * @param query  Float query vector of length dimension.
 * @param k  Number of nearest neighbors to return.
 * @param ef_search  Candidate list size during search (higher = more accurate).
 * @param labels  Output array of at least k labels (caller-allocated).
 * @param distances  Output array of at least k distances (caller-allocated).
 * @return Number of results found (0 to k), or -1 on error.
 */
int gv_hnsw_inline_search(const GV_HNSWInlineIndex *idx, const float *query,
                           size_t k, size_t ef_search,
                           size_t *labels, float *distances);

/* ============================================================================
 * Incremental Rebuild
 * ============================================================================ */

/**
 * @brief Start an incremental graph rebuild.
 *
 * Iterates over nodes in batches, searches for better neighbor candidates
 * using the existing graph, and prunes using the standard HNSW neighbor
 * selection heuristic.  When config->background is set, the rebuild runs
 * in a dedicated thread and this function returns immediately.
 *
 * @param idx  Index instance; must be non-NULL.
 * @param config  Rebuild configuration; NULL for defaults.
 * @return 0 on success (or background thread started), -1 on error.
 */
int gv_hnsw_inline_rebuild(GV_HNSWInlineIndex *idx,
                            const GV_HNSWRebuildConfig *config);

/**
 * @brief Query the status of an in-progress or completed rebuild.
 *
 * @param idx  Index instance; must be non-NULL.
 * @param stats  Output statistics; must be non-NULL.
 * @return 0 on success, -1 if no rebuild has been started.
 */
int gv_hnsw_inline_rebuild_status(const GV_HNSWInlineIndex *idx,
                                   GV_HNSWRebuildStats *stats);

/* ============================================================================
 * Utility
 * ============================================================================ */

/**
 * @brief Get the number of vectors currently stored in the index.
 *
 * @param idx  Index instance; must be non-NULL.
 * @return Number of vectors, or 0 if idx is NULL.
 */
size_t gv_hnsw_inline_count(const GV_HNSWInlineIndex *idx);

/**
 * @brief Save the index to a file.
 *
 * @param idx  Index instance; must be non-NULL.
 * @param path  Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_hnsw_inline_save(const GV_HNSWInlineIndex *idx, const char *path);

/**
 * @brief Load an index from a file.
 *
 * @param path  Input file path.
 * @return Allocated index, or NULL on error.
 */
GV_HNSWInlineIndex *gv_hnsw_inline_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_HNSW_OPT_H */
