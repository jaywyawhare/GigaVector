#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_scalar_quant.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 16

static void fill_vector(float *data, size_t dim, float base) {
    for (size_t i = 0; i < dim; i++) {
        data[i] = sinf(base + (float)i);
    }
}

/* 1. test_scalar_quant_8bit                                           */
static int test_scalar_quant_8bit(void) {
    float data[DIM];
    fill_vector(data, DIM, 0.0f);

    GV_ScalarQuantConfig config;
    config.bits = 8;
    config.per_dimension = 0;

    GV_ScalarQuantVector *sqv = gv_scalar_quantize(data, DIM, &config);
    ASSERT(sqv != NULL, "gv_scalar_quantize returned NULL for 8-bit");
    ASSERT(sqv->dimension == DIM, "dimension mismatch");
    ASSERT(sqv->bits == 8, "bits mismatch");

    gv_scalar_quant_vector_destroy(sqv);
    return 0;
}

/* 2. test_scalar_quant_4bit                                           */
static int test_scalar_quant_4bit(void) {
    float data[DIM];
    fill_vector(data, DIM, 1.0f);

    GV_ScalarQuantConfig config;
    config.bits = 4;
    config.per_dimension = 0;

    GV_ScalarQuantVector *sqv = gv_scalar_quantize(data, DIM, &config);
    ASSERT(sqv != NULL, "gv_scalar_quantize returned NULL for 4-bit");
    ASSERT(sqv->bits == 4, "bits should be 4");

    gv_scalar_quant_vector_destroy(sqv);
    return 0;
}

/* 3. test_scalar_dequantize_roundtrip                                 */
static int test_scalar_dequantize_roundtrip(void) {
    float data[DIM];
    fill_vector(data, DIM, 2.0f);

    GV_ScalarQuantConfig config;
    config.bits = 8;
    config.per_dimension = 1;

    GV_ScalarQuantVector *sqv = gv_scalar_quantize(data, DIM, &config);
    ASSERT(sqv != NULL, "quantize failed");

    float output[DIM];
    int rc = gv_scalar_dequantize(sqv, output);
    ASSERT(rc == 0, "dequantize failed");

    /* 8-bit quantization should be reasonably accurate (within ~1% of range) */
    for (size_t i = 0; i < DIM; i++) {
        float diff = fabsf(data[i] - output[i]);
        ASSERT(diff < 0.1f, "dequantized value too far from original");
    }

    gv_scalar_quant_vector_destroy(sqv);
    return 0;
}

/* 4. test_scalar_quant_bytes_needed                                   */
static int test_scalar_quant_bytes_needed(void) {
    /* 8-bit: 16 dims -> 16 bytes */
    size_t bytes_8 = gv_scalar_quant_bytes_needed(DIM, 8);
    ASSERT(bytes_8 == DIM, "8-bit 16-dim should need 16 bytes");

    /* 4-bit: 16 dims -> 8 bytes (2 dims per byte) */
    size_t bytes_4 = gv_scalar_quant_bytes_needed(DIM, 4);
    ASSERT(bytes_4 == DIM / 2, "4-bit 16-dim should need 8 bytes");

    /* 16-bit: 16 dims -> 32 bytes */
    size_t bytes_16 = gv_scalar_quant_bytes_needed(DIM, 16);
    ASSERT(bytes_16 == DIM * 2, "16-bit 16-dim should need 32 bytes");

    return 0;
}

/* 5. test_scalar_quant_per_dimension                                  */
static int test_scalar_quant_per_dimension(void) {
    float data[DIM];
    fill_vector(data, DIM, 3.0f);

    GV_ScalarQuantConfig config;
    config.bits = 8;
    config.per_dimension = 1;

    GV_ScalarQuantVector *sqv = gv_scalar_quantize(data, DIM, &config);
    ASSERT(sqv != NULL, "quantize with per_dimension failed");
    ASSERT(sqv->per_dimension == 1, "per_dimension flag not set");

    gv_scalar_quant_vector_destroy(sqv);
    return 0;
}

/* 6. test_scalar_quant_train                                          */
static int test_scalar_quant_train(void) {
    size_t count = 64;
    float *train_data = (float *)malloc(count * DIM * sizeof(float));
    ASSERT(train_data != NULL, "malloc failed");

    for (size_t i = 0; i < count; i++) {
        fill_vector(&train_data[i * DIM], DIM, (float)i * 0.5f);
    }

    GV_ScalarQuantConfig config;
    config.bits = 8;
    config.per_dimension = 1;

    GV_ScalarQuantVector *sqv = gv_scalar_quantize_train(train_data, count, DIM, &config);
    ASSERT(sqv != NULL, "gv_scalar_quantize_train returned NULL");
    ASSERT(sqv->dimension == DIM, "dimension mismatch after train");

    gv_scalar_quant_vector_destroy(sqv);
    free(train_data);
    return 0;
}

/* 7. test_scalar_quant_distance                                       */
static int test_scalar_quant_distance(void) {
    float data[DIM], query[DIM];
    fill_vector(data, DIM, 0.0f);
    fill_vector(query, DIM, 0.0f); /* same vector */

    GV_ScalarQuantConfig config;
    config.bits = 8;
    config.per_dimension = 0;

    GV_ScalarQuantVector *sqv = gv_scalar_quantize(data, DIM, &config);
    ASSERT(sqv != NULL, "quantize failed");

    /* Distance of identical vectors should be near zero */
    float dist = gv_scalar_quant_distance(query, sqv, 0);
    ASSERT(dist >= 0.0f, "distance should be non-negative");
    ASSERT(dist < 1.0f, "distance of same vector should be near zero");

    gv_scalar_quant_vector_destroy(sqv);
    return 0;
}

/* 8. test_scalar_quant_destroy_null                                   */
static int test_scalar_quant_destroy_null(void) {
    /* Should be safe to call with NULL */
    gv_scalar_quant_vector_destroy(NULL);
    return 0;
}

/* main */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing scalar quant 8-bit...",            test_scalar_quant_8bit},
        {"Testing scalar quant 4-bit...",            test_scalar_quant_4bit},
        {"Testing scalar dequantize roundtrip...",   test_scalar_dequantize_roundtrip},
        {"Testing scalar quant bytes needed...",     test_scalar_quant_bytes_needed},
        {"Testing scalar quant per dimension...",    test_scalar_quant_per_dimension},
        {"Testing scalar quant train...",            test_scalar_quant_train},
        {"Testing scalar quant distance...",         test_scalar_quant_distance},
        {"Testing scalar quant destroy null...",     test_scalar_quant_destroy_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
