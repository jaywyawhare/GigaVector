/**
 * @file gv_gpu.c
 * @brief GPU acceleration implementation.
 *
 * This file provides the CPU fallback implementation.
 * When compiled with CUDA support, gv_gpu_cuda.cu provides GPU implementations.
 */

#include "gigavector/gv_gpu.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Check for CUDA availability at compile time */
#ifdef HAVE_CUDA
extern int gv_cuda_available(void);
extern int gv_cuda_device_count(void);
extern int gv_cuda_get_device_info(int device_id, GV_GPUDeviceInfo *info);
extern GV_GPUContext *gv_cuda_create(const GV_GPUConfig *config);
extern void gv_cuda_destroy(GV_GPUContext *ctx);
extern int gv_cuda_synchronize(GV_GPUContext *ctx);
extern int gv_cuda_compute_distances(GV_GPUContext *ctx, const float *queries,
                                      size_t num_queries, const float *database,
                                      size_t num_vectors, size_t dimension,
                                      GV_GPUDistanceMetric metric, float *distances);
extern int gv_cuda_knn_search(GV_GPUContext *ctx, const float *queries,
                               size_t num_queries, const float *database,
                               size_t num_vectors, size_t dimension,
                               const GV_GPUSearchParams *params,
                               size_t *indices, float *distances);
#endif

/*  Internal Structures  */

struct GV_GPUContext {
    GV_GPUConfig config;
    int device_id;
    int cuda_available;
    GV_GPUStats stats;
    char last_error[256];

#ifdef HAVE_CUDA
    void *cuda_context;  /* CUcontext */
    void *cuda_streams;  /* cudaStream_t array */
#endif
};

struct GV_GPUIndex {
    GV_GPUContext *ctx;
    float *vectors;         /* Host copy (CPU fallback) */
    float *d_vectors;       /* Device copy (CUDA) */
    float *norms;           /* Precomputed L2 norms */
    float *d_norms;         /* Device norms */
    size_t count;
    size_t capacity;
    size_t dimension;
    size_t memory_usage;
};

/*  Configuration  */

static const GV_GPUConfig DEFAULT_CONFIG = {
    .device_id = -1,
    .max_vectors_per_batch = 65536,
    .max_query_batch_size = 1024,
    .enable_tensor_cores = 1,
    .enable_async_transfers = 1,
    .stream_count = 4,
    .memory = {
        .initial_size = 256 * 1024 * 1024,  /* 256 MB */
        .max_size = 2UL * 1024 * 1024 * 1024, /* 2 GB */
        .allow_growth = 1
    }
};

void gv_gpu_config_init(GV_GPUConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/*  Device Query  */

int gv_gpu_available(void) {
#ifdef HAVE_CUDA
    return gv_cuda_available();
#else
    return 0;  /* CPU fallback always available, but no GPU */
#endif
}

int gv_gpu_device_count(void) {
#ifdef HAVE_CUDA
    return gv_cuda_device_count();
#else
    return 0;
#endif
}

int gv_gpu_get_device_info(int device_id, GV_GPUDeviceInfo *info) {
    if (!info) return -1;

#ifdef HAVE_CUDA
    return gv_cuda_get_device_info(device_id, info);
#else
    (void)device_id;
    /* Return CPU "device" info for fallback */
    memset(info, 0, sizeof(*info));
    info->device_id = -1;
    strncpy(info->name, "CPU Fallback (No CUDA)", sizeof(info->name) - 1);
    return 0;
#endif
}

/*  Context Management  */

GV_GPUContext *gv_gpu_create(const GV_GPUConfig *config) {
    GV_GPUContext *ctx = calloc(1, sizeof(GV_GPUContext));
    if (!ctx) return NULL;

    ctx->config = config ? *config : DEFAULT_CONFIG;

#ifdef HAVE_CUDA
    ctx->cuda_available = gv_cuda_available();
    if (ctx->cuda_available) {
        /* Initialize CUDA context */
        GV_GPUContext *cuda_ctx = gv_cuda_create(&ctx->config);
        if (cuda_ctx) {
            ctx->cuda_context = cuda_ctx->cuda_context;
            ctx->cuda_streams = cuda_ctx->cuda_streams;
            ctx->device_id = cuda_ctx->device_id;
            free(cuda_ctx);
        } else {
            ctx->cuda_available = 0;
        }
    }
#else
    ctx->cuda_available = 0;
#endif

    ctx->device_id = ctx->cuda_available ? ctx->config.device_id : -1;

    return ctx;
}

void gv_gpu_destroy(GV_GPUContext *ctx) {
    if (!ctx) return;

#ifdef HAVE_CUDA
    if (ctx->cuda_available) {
        gv_cuda_destroy(ctx);
    }
#endif

    free(ctx);
}

int gv_gpu_synchronize(GV_GPUContext *ctx) {
    if (!ctx) return -1;

#ifdef HAVE_CUDA
    if (ctx->cuda_available) {
        return gv_cuda_synchronize(ctx);
    }
#endif

    return 0;  /* CPU is always synchronized */
}

/*  GPU Index Management  */

GV_GPUIndex *gv_gpu_index_create(GV_GPUContext *ctx, const float *vectors,
                                  size_t count, size_t dimension) {
    if (!ctx || !vectors || count == 0 || dimension == 0) return NULL;

    GV_GPUIndex *index = calloc(1, sizeof(GV_GPUIndex));
    if (!index) return NULL;

    index->ctx = ctx;
    index->count = count;
    index->capacity = count;
    index->dimension = dimension;

    /* Allocate host memory */
    size_t data_size = count * dimension * sizeof(float);
    index->vectors = malloc(data_size);
    if (!index->vectors) {
        free(index);
        return NULL;
    }
    memcpy(index->vectors, vectors, data_size);

    /* Precompute L2 norms for optimized cosine/euclidean */
    index->norms = malloc(count * sizeof(float));
    if (index->norms) {
        for (size_t i = 0; i < count; i++) {
            float norm = 0;
            const float *v = vectors + i * dimension;
            for (size_t j = 0; j < dimension; j++) {
                norm += v[j] * v[j];
            }
            index->norms[i] = sqrtf(norm);
        }
    }

    index->memory_usage = data_size + count * sizeof(float);

#ifdef HAVE_CUDA
    if (ctx->cuda_available) {
        /* TODO: Allocate device memory and copy */
        /* cudaMalloc(&index->d_vectors, data_size); */
        /* cudaMemcpy(index->d_vectors, vectors, data_size, cudaMemcpyHostToDevice); */
    }
#endif

    return index;
}

GV_GPUIndex *gv_gpu_index_from_db(GV_GPUContext *ctx, GV_Database *db) {
    if (!ctx || !db) return NULL;

    size_t count = gv_database_count(db);
    size_t dimension = gv_database_dimension(db);

    if (count == 0 || dimension == 0) return NULL;

    /* Extract all vectors from database */
    float *vectors = malloc(count * dimension * sizeof(float));
    if (!vectors) return NULL;

    for (size_t i = 0; i < count; i++) {
        const float *v = gv_database_get_vector(db, i);
        if (v) {
            memcpy(vectors + i * dimension, v, dimension * sizeof(float));
        }
    }

    GV_GPUIndex *index = gv_gpu_index_create(ctx, vectors, count, dimension);
    free(vectors);

    return index;
}

int gv_gpu_index_add(GV_GPUIndex *index, const float *vectors, size_t count) {
    if (!index || !vectors || count == 0) return -1;

    size_t new_count = index->count + count;

    /* Resize if needed */
    if (new_count > index->capacity) {
        size_t new_capacity = index->capacity * 2;
        if (new_capacity < new_count) new_capacity = new_count;

        float *new_vectors = realloc(index->vectors,
                                      new_capacity * index->dimension * sizeof(float));
        if (!new_vectors) return -1;
        index->vectors = new_vectors;

        float *new_norms = realloc(index->norms, new_capacity * sizeof(float));
        if (!new_norms) return -1;
        index->norms = new_norms;

        index->capacity = new_capacity;
    }

    /* Copy new vectors */
    memcpy(index->vectors + index->count * index->dimension,
           vectors, count * index->dimension * sizeof(float));

    /* Compute norms for new vectors */
    for (size_t i = 0; i < count; i++) {
        float norm = 0;
        const float *v = vectors + i * index->dimension;
        for (size_t j = 0; j < index->dimension; j++) {
            norm += v[j] * v[j];
        }
        index->norms[index->count + i] = sqrtf(norm);
    }

    index->count = new_count;
    index->memory_usage = index->count * (index->dimension * sizeof(float) + sizeof(float));

#ifdef HAVE_CUDA
    if (index->ctx->cuda_available) {
        /* TODO: Update device memory */
    }
#endif

    return 0;
}

int gv_gpu_index_remove(GV_GPUIndex *index, const size_t *indices, size_t count) {
    if (!index || !indices || count == 0) return -1;

    /* Mark indices for removal (simple implementation - compact later) */
    /* For now, just set vectors to zero */
    for (size_t i = 0; i < count; i++) {
        if (indices[i] < index->count) {
            memset(index->vectors + indices[i] * index->dimension, 0,
                   index->dimension * sizeof(float));
            index->norms[indices[i]] = 0;
        }
    }

    return 0;
}

int gv_gpu_index_update(GV_GPUIndex *index, const size_t *indices,
                         const float *vectors, size_t count) {
    if (!index || !indices || !vectors || count == 0) return -1;

    for (size_t i = 0; i < count; i++) {
        if (indices[i] < index->count) {
            memcpy(index->vectors + indices[i] * index->dimension,
                   vectors + i * index->dimension,
                   index->dimension * sizeof(float));

            /* Update norm */
            float norm = 0;
            const float *v = vectors + i * index->dimension;
            for (size_t j = 0; j < index->dimension; j++) {
                norm += v[j] * v[j];
            }
            index->norms[indices[i]] = sqrtf(norm);
        }
    }

#ifdef HAVE_CUDA
    if (index->ctx->cuda_available) {
        /* TODO: Update device memory */
    }
#endif

    return 0;
}

int gv_gpu_index_info(GV_GPUIndex *index, size_t *count, size_t *dimension,
                       size_t *memory_usage) {
    if (!index) return -1;

    if (count) *count = index->count;
    if (dimension) *dimension = index->dimension;
    if (memory_usage) *memory_usage = index->memory_usage;

    return 0;
}

void gv_gpu_index_destroy(GV_GPUIndex *index) {
    if (!index) return;

#ifdef HAVE_CUDA
    if (index->ctx && index->ctx->cuda_available) {
        /* TODO: Free device memory */
        /* cudaFree(index->d_vectors); */
        /* cudaFree(index->d_norms); */
    }
#endif

    free(index->vectors);
    free(index->norms);
    free(index);
}

/*  CPU Distance Computation (Fallback)  */

static float cpu_euclidean_distance(const float *a, const float *b, size_t dim) {
    float sum = 0;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

static float cpu_cosine_distance(const float *a, const float *b, size_t dim,
                                  float norm_a, float norm_b) {
    float dot = 0;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    if (norm_a > 0 && norm_b > 0) {
        return 1.0f - (dot / (norm_a * norm_b));
    }
    return 1.0f;
}

static float cpu_dot_product(const float *a, const float *b, size_t dim) {
    float dot = 0;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return -dot;  /* Negative for min-heap compatibility */
}

static float cpu_manhattan_distance(const float *a, const float *b, size_t dim) {
    float sum = 0;
    for (size_t i = 0; i < dim; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return sum;
}

/*  Distance Computation  */

int gv_gpu_compute_distances(GV_GPUContext *ctx, const float *queries,
                              size_t num_queries, const float *database,
                              size_t num_vectors, size_t dimension,
                              GV_GPUDistanceMetric metric, float *distances) {
    if (!ctx || !queries || !database || !distances) return -1;
    if (num_queries == 0 || num_vectors == 0 || dimension == 0) return -1;

#ifdef HAVE_CUDA
    if (ctx->cuda_available) {
        return gv_cuda_compute_distances(ctx, queries, num_queries, database,
                                          num_vectors, dimension, metric, distances);
    }
#endif

    /* CPU fallback */
    for (size_t q = 0; q < num_queries; q++) {
        const float *query = queries + q * dimension;

        /* Precompute query norm for cosine */
        float query_norm = 0;
        if (metric == GV_GPU_COSINE) {
            for (size_t i = 0; i < dimension; i++) {
                query_norm += query[i] * query[i];
            }
            query_norm = sqrtf(query_norm);
        }

        for (size_t v = 0; v < num_vectors; v++) {
            const float *vec = database + v * dimension;
            float dist;

            switch (metric) {
                case GV_GPU_EUCLIDEAN:
                    dist = cpu_euclidean_distance(query, vec, dimension);
                    break;
                case GV_GPU_COSINE: {
                    float vec_norm = 0;
                    for (size_t i = 0; i < dimension; i++) {
                        vec_norm += vec[i] * vec[i];
                    }
                    vec_norm = sqrtf(vec_norm);
                    dist = cpu_cosine_distance(query, vec, dimension, query_norm, vec_norm);
                    break;
                }
                case GV_GPU_DOT_PRODUCT:
                    dist = cpu_dot_product(query, vec, dimension);
                    break;
                case GV_GPU_MANHATTAN:
                    dist = cpu_manhattan_distance(query, vec, dimension);
                    break;
                default:
                    dist = cpu_euclidean_distance(query, vec, dimension);
            }

            distances[q * num_vectors + v] = dist;
        }
    }

    ctx->stats.total_distance_computations += num_queries * num_vectors;

    return 0;
}

int gv_gpu_index_compute_distances(GV_GPUIndex *index, const float *queries,
                                    size_t num_queries,
                                    GV_GPUDistanceMetric metric,
                                    float *distances) {
    if (!index || !queries || !distances) return -1;

    return gv_gpu_compute_distances(index->ctx, queries, num_queries,
                                     index->vectors, index->count,
                                     index->dimension, metric, distances);
}

/*  k-NN Search  */

/* Simple insertion sort for maintaining top-k */
static void insert_result(size_t *indices, float *distances, size_t k,
                          size_t idx, float dist, size_t *count) {
    if (*count < k) {
        /* Find insertion point */
        size_t pos = *count;
        while (pos > 0 && distances[pos - 1] > dist) {
            distances[pos] = distances[pos - 1];
            indices[pos] = indices[pos - 1];
            pos--;
        }
        distances[pos] = dist;
        indices[pos] = idx;
        (*count)++;
    } else if (dist < distances[k - 1]) {
        /* Replace worst result */
        size_t pos = k - 1;
        while (pos > 0 && distances[pos - 1] > dist) {
            distances[pos] = distances[pos - 1];
            indices[pos] = indices[pos - 1];
            pos--;
        }
        distances[pos] = dist;
        indices[pos] = idx;
    }
}

int gv_gpu_knn_search(GV_GPUContext *ctx, const float *queries,
                       size_t num_queries, const float *database,
                       size_t num_vectors, size_t dimension,
                       const GV_GPUSearchParams *params,
                       size_t *indices, float *distances) {
    if (!ctx || !queries || !database || !params || !indices || !distances) return -1;
    if (num_queries == 0 || num_vectors == 0 || dimension == 0) return -1;
    if (params->k == 0 || params->k > num_vectors) return -1;

#ifdef HAVE_CUDA
    if (ctx->cuda_available) {
        return gv_cuda_knn_search(ctx, queries, num_queries, database,
                                   num_vectors, dimension, params,
                                   indices, distances);
    }
#endif

    /* CPU fallback - brute force k-NN */
    size_t k = params->k;

    for (size_t q = 0; q < num_queries; q++) {
        const float *query = queries + q * dimension;
        size_t *q_indices = indices + q * k;
        float *q_distances = distances + q * k;

        /* Initialize with invalid values */
        for (size_t i = 0; i < k; i++) {
            q_indices[i] = (size_t)-1;
            q_distances[i] = FLT_MAX;
        }

        /* Precompute query norm */
        float query_norm = 0;
        if (params->metric == GV_GPU_COSINE) {
            for (size_t i = 0; i < dimension; i++) {
                query_norm += query[i] * query[i];
            }
            query_norm = sqrtf(query_norm);
        }

        size_t count = 0;

        for (size_t v = 0; v < num_vectors; v++) {
            const float *vec = database + v * dimension;
            float dist;

            switch (params->metric) {
                case GV_GPU_EUCLIDEAN:
                    dist = cpu_euclidean_distance(query, vec, dimension);
                    break;
                case GV_GPU_COSINE: {
                    float vec_norm = 0;
                    for (size_t i = 0; i < dimension; i++) {
                        vec_norm += vec[i] * vec[i];
                    }
                    vec_norm = sqrtf(vec_norm);
                    dist = cpu_cosine_distance(query, vec, dimension, query_norm, vec_norm);
                    break;
                }
                case GV_GPU_DOT_PRODUCT:
                    dist = cpu_dot_product(query, vec, dimension);
                    break;
                case GV_GPU_MANHATTAN:
                    dist = cpu_manhattan_distance(query, vec, dimension);
                    break;
                default:
                    dist = cpu_euclidean_distance(query, vec, dimension);
            }

            /* Range filter */
            if (params->radius > 0 && dist > params->radius) {
                continue;
            }

            insert_result(q_indices, q_distances, k, v, dist, &count);
        }
    }

    ctx->stats.total_searches += num_queries;
    ctx->stats.total_vectors_processed += num_queries * num_vectors;

    return 0;
}

int gv_gpu_index_knn_search(GV_GPUIndex *index, const float *queries,
                             size_t num_queries, const GV_GPUSearchParams *params,
                             size_t *indices, float *distances) {
    if (!index || !queries || !params || !indices || !distances) return -1;

    return gv_gpu_knn_search(index->ctx, queries, num_queries,
                              index->vectors, index->count,
                              index->dimension, params,
                              indices, distances);
}

int gv_gpu_index_search(GV_GPUIndex *index, const float *query,
                         const GV_GPUSearchParams *params,
                         size_t *indices, float *distances) {
    if (!index || !query || !params || !indices || !distances) return -1;

    int result = gv_gpu_index_knn_search(index, query, 1, params, indices, distances);
    if (result != 0) return -1;

    /* Count valid results */
    int count = 0;
    for (size_t i = 0; i < params->k; i++) {
        if (indices[i] != (size_t)-1) count++;
    }

    return count;
}

/*  Batch Operations  */

int gv_gpu_batch_add(GV_GPUContext *ctx, GV_Database *db,
                      const float *vectors, size_t count) {
    if (!ctx || !db || !vectors || count == 0) return -1;

    /* Add vectors to database one by one */
    /* In a real implementation, this would batch the operations */
    size_t dim = gv_database_dimension(db);
    for (size_t i = 0; i < count; i++) {
        if (gv_db_add_vector(db, vectors + i * dim, dim) < 0) {
            return -1;
        }
    }

    return 0;
}

int gv_gpu_batch_search(GV_GPUContext *ctx, GV_Database *db,
                         const float *queries, size_t num_queries, size_t k,
                         size_t *indices, float *distances) {
    if (!ctx || !db || !queries || !indices || !distances) return -1;
    if (num_queries == 0 || k == 0) return -1;

    size_t count = gv_database_count(db);
    size_t dimension = gv_database_dimension(db);

    if (count == 0) return -1;

    /* Extract all vectors for GPU search */
    float *vectors = malloc(count * dimension * sizeof(float));
    if (!vectors) return -1;

    for (size_t i = 0; i < count; i++) {
        const float *v = gv_database_get_vector(db, i);
        if (v) {
            memcpy(vectors + i * dimension, v, dimension * sizeof(float));
        }
    }

    GV_GPUSearchParams params = {
        .metric = GV_GPU_EUCLIDEAN,
        .k = k,
        .radius = 0,
        .use_precomputed_norms = 1
    };

    int result = gv_gpu_knn_search(ctx, queries, num_queries, vectors,
                                    count, dimension, &params, indices, distances);

    free(vectors);
    return result;
}

/*  IVF-PQ GPU Support  */

int gv_gpu_train_ivfpq(GV_GPUContext *ctx, const float *vectors,
                        size_t num_vectors, size_t dimension,
                        size_t num_centroids, size_t num_subquantizers,
                        size_t bits_per_subquantizer,
                        float *centroids, float *codebooks) {
    if (!ctx || !vectors || !centroids || !codebooks) return -1;
    if (num_vectors == 0 || dimension == 0) return -1;
    if (num_centroids == 0 || num_subquantizers == 0) return -1;

#ifdef HAVE_CUDA
    if (ctx->cuda_available) {
        /* TODO: CUDA k-means for centroid training */
    }
#endif

    /* CPU fallback - simple k-means */
    (void)bits_per_subquantizer;

    /* Initialize centroids randomly */
    size_t stride = num_vectors / num_centroids;
    if (stride == 0) stride = 1;

    for (size_t i = 0; i < num_centroids && i * stride < num_vectors; i++) {
        memcpy(centroids + i * dimension,
               vectors + (i * stride) * dimension,
               dimension * sizeof(float));
    }

    /* Simple k-means iterations */
    size_t *assignments = calloc(num_vectors, sizeof(size_t));
    size_t *counts = calloc(num_centroids, sizeof(size_t));
    float *new_centroids = calloc(num_centroids * dimension, sizeof(float));

    if (!assignments || !counts || !new_centroids) {
        free(assignments);
        free(counts);
        free(new_centroids);
        return -1;
    }

    for (int iter = 0; iter < 10; iter++) {  /* Fixed iterations */
        /* Assign vectors to nearest centroid */
        for (size_t v = 0; v < num_vectors; v++) {
            float min_dist = FLT_MAX;
            size_t best = 0;

            for (size_t c = 0; c < num_centroids; c++) {
                float dist = cpu_euclidean_distance(
                    vectors + v * dimension,
                    centroids + c * dimension,
                    dimension
                );
                if (dist < min_dist) {
                    min_dist = dist;
                    best = c;
                }
            }
            assignments[v] = best;
        }

        /* Update centroids */
        memset(new_centroids, 0, num_centroids * dimension * sizeof(float));
        memset(counts, 0, num_centroids * sizeof(size_t));

        for (size_t v = 0; v < num_vectors; v++) {
            size_t c = assignments[v];
            counts[c]++;
            for (size_t d = 0; d < dimension; d++) {
                new_centroids[c * dimension + d] += vectors[v * dimension + d];
            }
        }

        for (size_t c = 0; c < num_centroids; c++) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < dimension; d++) {
                    centroids[c * dimension + d] = new_centroids[c * dimension + d] / counts[c];
                }
            }
        }
    }

    /* Initialize PQ codebooks (simplified) */
    size_t subdim = dimension / num_subquantizers;
    size_t codes_per_sub = 1 << bits_per_subquantizer;

    for (size_t s = 0; s < num_subquantizers; s++) {
        for (size_t c = 0; c < codes_per_sub && c < num_vectors; c++) {
            memcpy(codebooks + (s * codes_per_sub + c) * subdim,
                   vectors + c * dimension + s * subdim,
                   subdim * sizeof(float));
        }
    }

    free(assignments);
    free(counts);
    free(new_centroids);

    return 0;
}

/*  Statistics  */

int gv_gpu_get_stats(GV_GPUContext *ctx, GV_GPUStats *stats) {
    if (!ctx || !stats) return -1;

    *stats = ctx->stats;

    if (stats->total_searches > 0) {
        stats->avg_search_time_ms = stats->total_gpu_time_ms / stats->total_searches;
    }

    return 0;
}

int gv_gpu_reset_stats(GV_GPUContext *ctx) {
    if (!ctx) return -1;

    memset(&ctx->stats, 0, sizeof(ctx->stats));
    return 0;
}

const char *gv_gpu_get_error(GV_GPUContext *ctx) {
    if (!ctx) return "Invalid context";
    return ctx->last_error[0] ? ctx->last_error : "No error";
}
