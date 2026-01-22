#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "gigavector/gigavector.h"

#define ITERATIONS 500000
#define DIMENSION 128

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static float euclidean_scalar(const float *a, const float *b, size_t dim) {
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
    }
    return sqrtf(sum_sq_diff);
}

static float cosine_scalar(const float *a, const float *b, size_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

int main(void) {
    printf("SIMD vs Scalar Performance Comparison\n");
    printf("=====================================\n\n");
    printf("Dimension: %d, Iterations: %d\n\n", DIMENSION, ITERATIONS);

    float *a_data = (float *)malloc(DIMENSION * sizeof(float));
    float *b_data = (float *)malloc(DIMENSION * sizeof(float));
    
    for (size_t i = 0; i < DIMENSION; ++i) {
        a_data[i] = (float)(i % 100) / 10.0f;
        b_data[i] = (float)((i + 1) % 100) / 10.0f;
    }

    printf("Euclidean Distance:\n");
    printf("-------------------\n");
    
    double start = get_time_ms();
    float total = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        total += euclidean_scalar(a_data, b_data, DIMENSION);
    }
    double scalar_time = get_time_ms() - start;
    printf("Scalar:    %.2f ms (%.2f ops/sec)\n",
           scalar_time, (ITERATIONS / scalar_time) * 1000.0);

    GV_Vector a, b;
    a.dimension = DIMENSION;
    a.data = a_data;
    a.metadata = NULL;
    b.dimension = DIMENSION;
    b.data = b_data;
    b.metadata = NULL;

    start = get_time_ms();
    total = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        total += gv_distance_euclidean(&a, &b);
    }
    double simd_time = get_time_ms() - start;
    printf("SIMD:      %.2f ms (%.2f ops/sec)\n", 
           simd_time, (ITERATIONS / simd_time) * 1000.0);
    printf("Speedup:   %.2fx\n\n", scalar_time / simd_time);

    printf("Cosine Similarity:\n");
    printf("------------------\n");
    
    start = get_time_ms();
    total = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        total += cosine_scalar(a_data, b_data, DIMENSION);
    }
    scalar_time = get_time_ms() - start;
    printf("Scalar:    %.2f ms (%.2f ops/sec)\n", 
           scalar_time, (ITERATIONS / scalar_time) * 1000.0);

    start = get_time_ms();
    total = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        total += gv_distance_cosine(&a, &b);
    }
    simd_time = get_time_ms() - start;
    printf("SIMD:      %.2f ms (%.2f ops/sec)\n", 
           simd_time, (ITERATIONS / simd_time) * 1000.0);
    printf("Speedup:   %.2fx\n\n", scalar_time / simd_time);

    free(a_data);
    free(b_data);
    return 0;
}

