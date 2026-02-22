/**
 * @file gv_gpu_kernels.cu
 * @brief CUDA kernels for GPU-accelerated vector operations.
 *
 * This file is only compiled when CUDA is available.
 * Requires CUDA Toolkit 11.0+.
 */

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

extern "C" {
#include "gigavector/gv_gpu.h"
}

/* CUDA Error Handling */

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUDA_CHECK_LAST() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* Constants */

#define BLOCK_SIZE 256
#define TILE_SIZE 32
#define WARP_SIZE 32

/* Distance Computation Kernels */

/**
 * @brief Euclidean distance kernel (L2).
 *
 * Each thread computes one query-vector distance.
 */
__global__ void euclidean_distance_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    size_t num_queries,
    size_t num_vectors,
    size_t dimension
) {
    size_t q = blockIdx.y;
    size_t v = blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= num_queries || v >= num_vectors) return;

    const float* query = queries + q * dimension;
    const float* vec = database + v * dimension;

    float sum = 0.0f;
    for (size_t d = 0; d < dimension; d++) {
        float diff = query[d] - vec[d];
        sum += diff * diff;
    }

    distances[q * num_vectors + v] = sqrtf(sum);
}

/**
 * @brief Cosine distance kernel.
 */
__global__ void cosine_distance_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    const float* __restrict__ query_norms,
    const float* __restrict__ db_norms,
    float* __restrict__ distances,
    size_t num_queries,
    size_t num_vectors,
    size_t dimension
) {
    size_t q = blockIdx.y;
    size_t v = blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= num_queries || v >= num_vectors) return;

    const float* query = queries + q * dimension;
    const float* vec = database + v * dimension;

    float dot = 0.0f;
    for (size_t d = 0; d < dimension; d++) {
        dot += query[d] * vec[d];
    }

    float norm_q = query_norms[q];
    float norm_v = db_norms[v];

    float sim = (norm_q > 0 && norm_v > 0) ? (dot / (norm_q * norm_v)) : 0.0f;
    distances[q * num_vectors + v] = 1.0f - sim;
}

/**
 * @brief Dot product kernel (inner product).
 */
__global__ void dot_product_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    size_t num_queries,
    size_t num_vectors,
    size_t dimension
) {
    size_t q = blockIdx.y;
    size_t v = blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= num_queries || v >= num_vectors) return;

    const float* query = queries + q * dimension;
    const float* vec = database + v * dimension;

    float dot = 0.0f;
    for (size_t d = 0; d < dimension; d++) {
        dot += query[d] * vec[d];
    }

    distances[q * num_vectors + v] = -dot;  /* Negative for min-heap */
}

/**
 * @brief Manhattan distance kernel (L1).
 */
__global__ void manhattan_distance_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    size_t num_queries,
    size_t num_vectors,
    size_t dimension
) {
    size_t q = blockIdx.y;
    size_t v = blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= num_queries || v >= num_vectors) return;

    const float* query = queries + q * dimension;
    const float* vec = database + v * dimension;

    float sum = 0.0f;
    for (size_t d = 0; d < dimension; d++) {
        sum += fabsf(query[d] - vec[d]);
    }

    distances[q * num_vectors + v] = sum;
}

/* L2 Norm Computation */

__global__ void compute_norms_kernel(
    const float* __restrict__ vectors,
    float* __restrict__ norms,
    size_t num_vectors,
    size_t dimension
) {
    size_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vectors) return;

    const float* vec = vectors + v * dimension;
    float sum = 0.0f;
    for (size_t d = 0; d < dimension; d++) {
        sum += vec[d] * vec[d];
    }
    norms[v] = sqrtf(sum);
}

/* k-NN Search Kernel (Bitonic Sort) */

/**
 * @brief Bitonic sort step for top-k selection.
 */
__device__ void bitonic_sort_step(float* dist, size_t* idx, int j, int k, int n) {
    int tid = threadIdx.x;
    int ixj = tid ^ j;

    if (ixj > tid && tid < n && ixj < n) {
        if ((tid & k) == 0) {
            if (dist[tid] > dist[ixj]) {
                float tmp_d = dist[tid];
                dist[tid] = dist[ixj];
                dist[ixj] = tmp_d;

                size_t tmp_i = idx[tid];
                idx[tid] = idx[ixj];
                idx[ixj] = tmp_i;
            }
        } else {
            if (dist[tid] < dist[ixj]) {
                float tmp_d = dist[tid];
                dist[tid] = dist[ixj];
                dist[ixj] = tmp_d;

                size_t tmp_i = idx[tid];
                idx[tid] = idx[ixj];
                idx[ixj] = tmp_i;
            }
        }
    }
}

/**
 * @brief Top-k selection kernel using partial bitonic sort.
 */
__global__ void topk_kernel(
    const float* __restrict__ distances,
    size_t* __restrict__ indices,
    float* __restrict__ out_distances,
    size_t num_queries,
    size_t num_vectors,
    size_t k
) {
    extern __shared__ char shared_mem[];
    float* s_dist = (float*)shared_mem;
    size_t* s_idx = (size_t*)(s_dist + blockDim.x);

    size_t q = blockIdx.x;
    if (q >= num_queries) return;

    const float* q_distances = distances + q * num_vectors;

    /* Load first batch into shared memory */
    int tid = threadIdx.x;
    if (tid < num_vectors) {
        s_dist[tid] = q_distances[tid];
        s_idx[tid] = tid;
    } else {
        s_dist[tid] = FLT_MAX;
        s_idx[tid] = (size_t)-1;
    }
    __syncthreads();

    /* Bitonic sort */
    int n = min((int)num_vectors, (int)blockDim.x);
    for (int k_step = 2; k_step <= n; k_step *= 2) {
        for (int j = k_step / 2; j > 0; j /= 2) {
            bitonic_sort_step(s_dist, s_idx, j, k_step, n);
            __syncthreads();
        }
    }

    /* Write top-k results */
    if (tid < k && tid < num_vectors) {
        indices[q * k + tid] = s_idx[tid];
        out_distances[q * k + tid] = s_dist[tid];
    }
}

/* C Interface Functions */

extern "C" {

int gv_cuda_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

int gv_cuda_device_count(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

int gv_cuda_get_device_info(int device_id, GV_GPUDeviceInfo *info) {
    if (!info) return -1;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    info->device_id = device_id;
    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    info->total_memory = prop.totalGlobalMem;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    info->free_memory = free_mem;

    info->compute_capability_major = prop.major;
    info->compute_capability_minor = prop.minor;
    info->multiprocessor_count = prop.multiProcessorCount;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->warp_size = prop.warpSize;

    return 0;
}

typedef struct {
    GV_GPUConfig config;
    int device_id;
    cudaStream_t *streams;
    int stream_count;
} CUDAContext;

GV_GPUContext *gv_cuda_create(const GV_GPUConfig *config) {
    CUDAContext *ctx = (CUDAContext*)calloc(1, sizeof(CUDAContext));
    if (!ctx) return NULL;

    ctx->config = config ? *config : (GV_GPUConfig){
        .device_id = 0,
        .stream_count = 4
    };

    /* Select device */
    int device = ctx->config.device_id;
    if (device < 0) {
        /* Auto-select: use device with most memory */
        int device_count = gv_cuda_device_count();
        size_t max_mem = 0;
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if (prop.totalGlobalMem > max_mem) {
                max_mem = prop.totalGlobalMem;
                device = i;
            }
        }
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        free(ctx);
        return NULL;
    }
    ctx->device_id = device;

    /* Create streams */
    ctx->stream_count = ctx->config.stream_count;
    ctx->streams = (cudaStream_t*)malloc(ctx->stream_count * sizeof(cudaStream_t));
    for (int i = 0; i < ctx->stream_count; i++) {
        cudaStreamCreate(&ctx->streams[i]);
    }

    return (GV_GPUContext*)ctx;
}

void gv_cuda_destroy(GV_GPUContext *gpu_ctx) {
    if (!gpu_ctx) return;
    CUDAContext *ctx = (CUDAContext*)gpu_ctx;

    for (int i = 0; i < ctx->stream_count; i++) {
        cudaStreamDestroy(ctx->streams[i]);
    }
    free(ctx->streams);
    free(ctx);
}

int gv_cuda_synchronize(GV_GPUContext *gpu_ctx) {
    (void)gpu_ctx;
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

int gv_cuda_compute_distances(
    GV_GPUContext *ctx,
    const float *queries,
    size_t num_queries,
    const float *database,
    size_t num_vectors,
    size_t dimension,
    GV_GPUDistanceMetric metric,
    float *distances
) {
    (void)ctx;

    /* Allocate device memory */
    float *d_queries, *d_database, *d_distances;
    size_t query_size = num_queries * dimension * sizeof(float);
    size_t db_size = num_vectors * dimension * sizeof(float);
    size_t dist_size = num_queries * num_vectors * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_queries, query_size));
    CUDA_CHECK(cudaMalloc(&d_database, db_size));
    CUDA_CHECK(cudaMalloc(&d_distances, dist_size));

    /* Copy to device */
    CUDA_CHECK(cudaMemcpy(d_queries, queries, query_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_database, database, db_size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE, num_queries);

    switch (metric) {
        case GV_GPU_EUCLIDEAN:
            euclidean_distance_kernel<<<grid, block>>>(
                d_queries, d_database, d_distances,
                num_queries, num_vectors, dimension
            );
            break;

        case GV_GPU_COSINE: {
            /* Compute norms first */
            float *d_query_norms, *d_db_norms;
            CUDA_CHECK(cudaMalloc(&d_query_norms, num_queries * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_db_norms, num_vectors * sizeof(float)));

            int norm_blocks = (num_queries + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compute_norms_kernel<<<norm_blocks, BLOCK_SIZE>>>(
                d_queries, d_query_norms, num_queries, dimension
            );

            norm_blocks = (num_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compute_norms_kernel<<<norm_blocks, BLOCK_SIZE>>>(
                d_database, d_db_norms, num_vectors, dimension
            );

            cosine_distance_kernel<<<grid, block>>>(
                d_queries, d_database, d_query_norms, d_db_norms, d_distances,
                num_queries, num_vectors, dimension
            );

            cudaFree(d_query_norms);
            cudaFree(d_db_norms);
            break;
        }

        case GV_GPU_DOT_PRODUCT:
            dot_product_kernel<<<grid, block>>>(
                d_queries, d_database, d_distances,
                num_queries, num_vectors, dimension
            );
            break;

        case GV_GPU_MANHATTAN:
            manhattan_distance_kernel<<<grid, block>>>(
                d_queries, d_database, d_distances,
                num_queries, num_vectors, dimension
            );
            break;
    }

    CUDA_CHECK_LAST();

    /* Copy results back */
    CUDA_CHECK(cudaMemcpy(distances, d_distances, dist_size, cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_queries);
    cudaFree(d_database);
    cudaFree(d_distances);

    return 0;
}

int gv_cuda_knn_search(
    GV_GPUContext *ctx,
    const float *queries,
    size_t num_queries,
    const float *database,
    size_t num_vectors,
    size_t dimension,
    const GV_GPUSearchParams *params,
    size_t *indices,
    float *distances
) {
    if (!params || params->k == 0) return -1;

    size_t k = params->k;

    /* Compute all distances first */
    float *all_distances = (float*)malloc(num_queries * num_vectors * sizeof(float));
    if (!all_distances) return -1;

    int result = gv_cuda_compute_distances(ctx, queries, num_queries, database,
                                            num_vectors, dimension, params->metric,
                                            all_distances);
    if (result != 0) {
        free(all_distances);
        return result;
    }

    /* Allocate device memory for top-k */
    float *d_distances, *d_out_distances;
    size_t *d_indices;

    CUDA_CHECK(cudaMalloc(&d_distances, num_queries * num_vectors * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_distances, num_queries * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_queries * k * sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(d_distances, all_distances,
                          num_queries * num_vectors * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Launch top-k kernel */
    int block_size = min((int)num_vectors, 1024);
    size_t shared_size = block_size * (sizeof(float) + sizeof(size_t));

    topk_kernel<<<num_queries, block_size, shared_size>>>(
        d_distances, d_indices, d_out_distances,
        num_queries, num_vectors, k
    );
    CUDA_CHECK_LAST();

    /* Copy results back */
    CUDA_CHECK(cudaMemcpy(indices, d_indices,
                          num_queries * k * sizeof(size_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(distances, d_out_distances,
                          num_queries * k * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_distances);
    cudaFree(d_out_distances);
    cudaFree(d_indices);
    free(all_distances);

    return 0;
}

} /* extern "C" */

#endif /* HAVE_CUDA */
