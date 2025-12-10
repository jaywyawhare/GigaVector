#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "gigavector/gigavector.h"

#define ITERATIONS 1000000
#define DIMENSION 128

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void benchmark_euclidean(void) {
    printf("=== Euclidean Distance Benchmark ===\n");
    printf("Dimension: %d, Iterations: %d\n\n", DIMENSION, ITERATIONS);

    float *a_data = (float *)malloc(DIMENSION * sizeof(float));
    float *b_data = (float *)malloc(DIMENSION * sizeof(float));
    
    for (size_t i = 0; i < DIMENSION; ++i) {
        a_data[i] = (float)(i % 100) / 10.0f;
        b_data[i] = (float)((i + 1) % 100) / 10.0f;
    }

    GV_Vector a, b;
    a.dimension = DIMENSION;
    a.data = a_data;
    a.metadata = NULL;
    b.dimension = DIMENSION;
    b.data = b_data;
    b.metadata = NULL;

    double start = get_time_ms();
    float total = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        float dist = gv_distance_euclidean(&a, &b);
        total += dist;
    }
    double end = get_time_ms();
    double elapsed = end - start;
    
    printf("Total distance sum: %.2f\n", total);
    printf("Time: %.2f ms\n", elapsed);
    printf("Throughput: %.2f ops/ms (%.2f ops/sec)\n", 
           ITERATIONS / elapsed, (ITERATIONS / elapsed) * 1000.0);
    printf("Average time per operation: %.4f microseconds\n\n", 
           (elapsed / ITERATIONS) * 1000.0);

    free(a_data);
    free(b_data);
}

static void benchmark_cosine(void) {
    printf("=== Cosine Similarity Benchmark ===\n");
    printf("Dimension: %d, Iterations: %d\n\n", DIMENSION, ITERATIONS);

    float *a_data = (float *)malloc(DIMENSION * sizeof(float));
    float *b_data = (float *)malloc(DIMENSION * sizeof(float));
    
    for (size_t i = 0; i < DIMENSION; ++i) {
        a_data[i] = (float)(i % 100) / 10.0f;
        b_data[i] = (float)((i + 1) % 100) / 10.0f;
    }

    GV_Vector a, b;
    a.dimension = DIMENSION;
    a.data = a_data;
    a.metadata = NULL;
    b.dimension = DIMENSION;
    b.data = b_data;
    b.metadata = NULL;

    double start = get_time_ms();
    float total = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        float sim = gv_distance_cosine(&a, &b);
        total += sim;
    }
    double end = get_time_ms();
    double elapsed = end - start;
    
    printf("Total similarity sum: %.2f\n", total);
    printf("Time: %.2f ms\n", elapsed);
    printf("Throughput: %.2f ops/ms (%.2f ops/sec)\n", 
           ITERATIONS / elapsed, (ITERATIONS / elapsed) * 1000.0);
    printf("Average time per operation: %.4f microseconds\n\n", 
           (elapsed / ITERATIONS) * 1000.0);

    free(a_data);
    free(b_data);
}

static void benchmark_different_dimensions(void) {
    printf("=== Performance by Dimension ===\n");
    int dims[] = {16, 32, 64, 128, 256, 512};
    int num_dims = sizeof(dims) / sizeof(dims[0]);
    
    printf("%-10s %-15s %-20s %-15s\n", "Dimension", "Time (ms)", "Ops/sec", "us/op");
    printf("------------------------------------------------------------\n");
    
    for (int d = 0; d < num_dims; ++d) {
        int dim = dims[d];
        float *a_data = (float *)malloc(dim * sizeof(float));
        float *b_data = (float *)malloc(dim * sizeof(float));
        
        for (int i = 0; i < dim; ++i) {
            a_data[i] = (float)(i % 100) / 10.0f;
            b_data[i] = (float)((i + 1) % 100) / 10.0f;
        }

        GV_Vector a, b;
        a.dimension = dim;
        a.data = a_data;
        a.metadata = NULL;
        b.dimension = dim;
        b.data = b_data;
        b.metadata = NULL;

        int iterations = ITERATIONS;
        if (dim >= 256) {
            iterations = ITERATIONS / 4;
        }
        if (dim >= 512) {
            iterations = ITERATIONS / 10;
        }

        double start = get_time_ms();
        for (int i = 0; i < iterations; ++i) {
            gv_distance_euclidean(&a, &b);
        }
        double end = get_time_ms();
        double elapsed = end - start;
        
        printf("%-10d %-15.2f %-20.0f %-15.4f\n", 
               dim, elapsed, (iterations / elapsed) * 1000.0, 
               (elapsed / iterations) * 1000.0);

        free(a_data);
        free(b_data);
    }
    printf("\n");
}

int main(void) {
    printf("GigaVector SIMD Performance Benchmark\n");
    printf("=====================================\n\n");
    
    unsigned int features = gv_cpu_detect_features();
    printf("CPU Features:\n");
    if (features & GV_CPU_FEATURE_SSE) printf("  SSE\n");
    if (features & GV_CPU_FEATURE_SSE4_2) printf("  SSE4.2\n");
    if (features & GV_CPU_FEATURE_AVX) printf("  AVX\n");
    if (features & GV_CPU_FEATURE_AVX2) printf("  AVX2\n");
    if (features & GV_CPU_FEATURE_FMA) printf("  FMA\n");
    printf("\n");

    benchmark_euclidean();
    benchmark_cosine();
    benchmark_different_dimensions();

    return 0;
}

