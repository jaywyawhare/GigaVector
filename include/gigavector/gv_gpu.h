#ifndef GIGAVECTOR_GV_GPU_H
#define GIGAVECTOR_GV_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_gpu.h
 * @brief GPU acceleration for GigaVector.
 *
 * Provides CUDA-accelerated distance computations and k-NN search.
 * Falls back to CPU implementations when CUDA is not available.
 */

/* Forward declarations */
struct GV_Database;
typedef struct GV_Database GV_Database;

/**
 * @brief GPU device information.
 */
typedef struct {
    int device_id;                  /**< CUDA device ID. */
    char name[256];                 /**< Device name. */
    size_t total_memory;            /**< Total memory in bytes. */
    size_t free_memory;             /**< Free memory in bytes. */
    int compute_capability_major;   /**< Compute capability major. */
    int compute_capability_minor;   /**< Compute capability minor. */
    int multiprocessor_count;       /**< Number of SMs. */
    int max_threads_per_block;      /**< Max threads per block. */
    int warp_size;                  /**< Warp size (typically 32). */
} GV_GPUDeviceInfo;

/**
 * @brief GPU memory pool configuration.
 */
typedef struct {
    size_t initial_size;            /**< Initial pool size (default: 256MB). */
    size_t max_size;                /**< Maximum pool size (default: 2GB). */
    int allow_growth;               /**< Allow pool to grow (default: 1). */
} GV_GPUMemoryConfig;

/**
 * @brief GPU context configuration.
 */
typedef struct {
    int device_id;                  /**< Device to use (-1 for auto). */
    size_t max_vectors_per_batch;   /**< Max vectors per batch (default: 65536). */
    size_t max_query_batch_size;    /**< Max queries per batch (default: 1024). */
    int enable_tensor_cores;        /**< Use tensor cores if available (default: 1). */
    int enable_async_transfers;     /**< Use async memory transfers (default: 1). */
    int stream_count;               /**< Number of CUDA streams (default: 4). */
    GV_GPUMemoryConfig memory;      /**< Memory pool configuration. */
} GV_GPUConfig;

/**
 * @brief Distance metric for GPU operations.
 */
typedef enum {
    GV_GPU_EUCLIDEAN = 0,           /**< Euclidean (L2) distance. */
    GV_GPU_COSINE = 1,              /**< Cosine similarity. */
    GV_GPU_DOT_PRODUCT = 2,         /**< Dot product (inner product). */
    GV_GPU_MANHATTAN = 3            /**< Manhattan (L1) distance. */
} GV_GPUDistanceMetric;

/**
 * @brief GPU search parameters.
 */
typedef struct {
    GV_GPUDistanceMetric metric;    /**< Distance metric. */
    size_t k;                       /**< Number of nearest neighbors. */
    float radius;                   /**< Radius for range search (0 = disabled). */
    int use_precomputed_norms;      /**< Use precomputed L2 norms (default: 1). */
} GV_GPUSearchParams;

/**
 * @brief GPU operation statistics.
 */
typedef struct {
    uint64_t total_searches;        /**< Total search operations. */
    uint64_t total_vectors_processed; /**< Total vectors processed. */
    uint64_t total_distance_computations; /**< Total distance computations. */
    double total_gpu_time_ms;       /**< Total GPU execution time. */
    double total_transfer_time_ms;  /**< Total memory transfer time. */
    double avg_search_time_ms;      /**< Average search time. */
    size_t peak_memory_usage;       /**< Peak GPU memory usage. */
    size_t current_memory_usage;    /**< Current GPU memory usage. */
} GV_GPUStats;

/**
 * @brief Opaque GPU context handle.
 */
typedef struct GV_GPUContext GV_GPUContext;

/**
 * @brief Opaque GPU index handle (vectors stored on GPU).
 */
typedef struct GV_GPUIndex GV_GPUIndex;

/* ============================================================================
 * Device Query
 * ============================================================================ */

/**
 * @brief Check if CUDA is available.
 *
 * @return 1 if CUDA is available, 0 otherwise.
 */
int gv_gpu_available(void);

/**
 * @brief Get the number of CUDA devices.
 *
 * @return Number of devices, or 0 if CUDA is not available.
 */
int gv_gpu_device_count(void);

/**
 * @brief Get device information.
 *
 * @param device_id Device ID.
 * @param info Output device information.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_get_device_info(int device_id, GV_GPUDeviceInfo *info);

/* ============================================================================
 * Context Management
 * ============================================================================ */

/**
 * @brief Initialize default GPU configuration.
 *
 * @param config Configuration to initialize.
 */
void gv_gpu_config_init(GV_GPUConfig *config);

/**
 * @brief Create a GPU context.
 *
 * @param config GPU configuration (NULL for defaults).
 * @return GPU context, or NULL on error.
 */
GV_GPUContext *gv_gpu_create(const GV_GPUConfig *config);

/**
 * @brief Destroy a GPU context.
 *
 * @param ctx GPU context (safe to call with NULL).
 */
void gv_gpu_destroy(GV_GPUContext *ctx);

/**
 * @brief Synchronize all GPU operations.
 *
 * @param ctx GPU context.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_synchronize(GV_GPUContext *ctx);

/* ============================================================================
 * GPU Index Management
 * ============================================================================ */

/**
 * @brief Create a GPU index from vectors.
 *
 * @param ctx GPU context.
 * @param vectors Vector data (row-major).
 * @param count Number of vectors.
 * @param dimension Vector dimension.
 * @return GPU index, or NULL on error.
 */
GV_GPUIndex *gv_gpu_index_create(GV_GPUContext *ctx, const float *vectors,
                                  size_t count, size_t dimension);

/**
 * @brief Create a GPU index from a database.
 *
 * @param ctx GPU context.
 * @param db Database to upload.
 * @return GPU index, or NULL on error.
 */
GV_GPUIndex *gv_gpu_index_from_db(GV_GPUContext *ctx, GV_Database *db);

/**
 * @brief Add vectors to a GPU index.
 *
 * @param index GPU index.
 * @param vectors Vector data.
 * @param count Number of vectors.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_index_add(GV_GPUIndex *index, const float *vectors, size_t count);

/**
 * @brief Remove vectors from a GPU index.
 *
 * @param index GPU index.
 * @param indices Indices to remove.
 * @param count Number of indices.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_index_remove(GV_GPUIndex *index, const size_t *indices, size_t count);

/**
 * @brief Update vectors in a GPU index.
 *
 * @param index GPU index.
 * @param indices Indices to update.
 * @param vectors New vector data.
 * @param count Number of vectors.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_index_update(GV_GPUIndex *index, const size_t *indices,
                         const float *vectors, size_t count);

/**
 * @brief Get GPU index statistics.
 *
 * @param index GPU index.
 * @param count Output vector count.
 * @param dimension Output dimension.
 * @param memory_usage Output memory usage in bytes.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_index_info(GV_GPUIndex *index, size_t *count, size_t *dimension,
                       size_t *memory_usage);

/**
 * @brief Destroy a GPU index.
 *
 * @param index GPU index (safe to call with NULL).
 */
void gv_gpu_index_destroy(GV_GPUIndex *index);

/* ============================================================================
 * Distance Computation
 * ============================================================================ */

/**
 * @brief Compute distances between query and database vectors.
 *
 * @param ctx GPU context.
 * @param queries Query vectors (row-major, num_queries x dimension).
 * @param num_queries Number of queries.
 * @param database Database vectors (row-major, num_vectors x dimension).
 * @param num_vectors Number of database vectors.
 * @param dimension Vector dimension.
 * @param metric Distance metric.
 * @param distances Output distance matrix (num_queries x num_vectors).
 * @return 0 on success, -1 on error.
 */
int gv_gpu_compute_distances(GV_GPUContext *ctx, const float *queries,
                              size_t num_queries, const float *database,
                              size_t num_vectors, size_t dimension,
                              GV_GPUDistanceMetric metric, float *distances);

/**
 * @brief Compute distances using GPU index.
 *
 * @param index GPU index.
 * @param queries Query vectors.
 * @param num_queries Number of queries.
 * @param metric Distance metric.
 * @param distances Output distance matrix.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_index_compute_distances(GV_GPUIndex *index, const float *queries,
                                    size_t num_queries,
                                    GV_GPUDistanceMetric metric,
                                    float *distances);

/* ============================================================================
 * k-NN Search
 * ============================================================================ */

/**
 * @brief Perform k-NN search on GPU.
 *
 * @param ctx GPU context.
 * @param queries Query vectors.
 * @param num_queries Number of queries.
 * @param database Database vectors.
 * @param num_vectors Number of database vectors.
 * @param dimension Vector dimension.
 * @param params Search parameters.
 * @param indices Output indices (num_queries x k).
 * @param distances Output distances (num_queries x k).
 * @return 0 on success, -1 on error.
 */
int gv_gpu_knn_search(GV_GPUContext *ctx, const float *queries,
                       size_t num_queries, const float *database,
                       size_t num_vectors, size_t dimension,
                       const GV_GPUSearchParams *params,
                       size_t *indices, float *distances);

/**
 * @brief Perform k-NN search using GPU index.
 *
 * @param index GPU index.
 * @param queries Query vectors.
 * @param num_queries Number of queries.
 * @param params Search parameters.
 * @param indices Output indices.
 * @param distances Output distances.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_index_knn_search(GV_GPUIndex *index, const float *queries,
                             size_t num_queries, const GV_GPUSearchParams *params,
                             size_t *indices, float *distances);

/**
 * @brief Perform single query k-NN search.
 *
 * @param index GPU index.
 * @param query Query vector.
 * @param params Search parameters.
 * @param indices Output indices.
 * @param distances Output distances.
 * @return Number of results found, or -1 on error.
 */
int gv_gpu_index_search(GV_GPUIndex *index, const float *query,
                         const GV_GPUSearchParams *params,
                         size_t *indices, float *distances);

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

/**
 * @brief Batch add vectors with automatic GPU upload.
 *
 * @param ctx GPU context.
 * @param db Database to add to.
 * @param vectors Vector data.
 * @param count Number of vectors.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_batch_add(GV_GPUContext *ctx, GV_Database *db,
                      const float *vectors, size_t count);

/**
 * @brief Batch search with automatic GPU acceleration.
 *
 * @param ctx GPU context.
 * @param db Database to search.
 * @param queries Query vectors.
 * @param num_queries Number of queries.
 * @param k Number of neighbors.
 * @param indices Output indices.
 * @param distances Output distances.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_batch_search(GV_GPUContext *ctx, GV_Database *db,
                         const float *queries, size_t num_queries, size_t k,
                         size_t *indices, float *distances);

/* ============================================================================
 * IVF-PQ GPU Support
 * ============================================================================ */

/**
 * @brief Train IVF-PQ codebook on GPU.
 *
 * @param ctx GPU context.
 * @param vectors Training vectors.
 * @param num_vectors Number of training vectors.
 * @param dimension Vector dimension.
 * @param num_centroids Number of coarse centroids.
 * @param num_subquantizers Number of PQ subquantizers.
 * @param bits_per_subquantizer Bits per subquantizer (typically 8).
 * @param centroids Output coarse centroids.
 * @param codebooks Output PQ codebooks.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_train_ivfpq(GV_GPUContext *ctx, const float *vectors,
                        size_t num_vectors, size_t dimension,
                        size_t num_centroids, size_t num_subquantizers,
                        size_t bits_per_subquantizer,
                        float *centroids, float *codebooks);

/* ============================================================================
 * Statistics
 * ============================================================================ */

/**
 * @brief Get GPU statistics.
 *
 * @param ctx GPU context.
 * @param stats Output statistics.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_get_stats(GV_GPUContext *ctx, GV_GPUStats *stats);

/**
 * @brief Reset GPU statistics.
 *
 * @param ctx GPU context.
 * @return 0 on success, -1 on error.
 */
int gv_gpu_reset_stats(GV_GPUContext *ctx);

/**
 * @brief Get last GPU error message.
 *
 * @param ctx GPU context.
 * @return Error message string.
 */
const char *gv_gpu_get_error(GV_GPUContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_GPU_H */
