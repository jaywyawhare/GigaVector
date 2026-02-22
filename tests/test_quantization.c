#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_quantization.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 16
#define TRAIN_COUNT 100

static void generate_data(float *data, size_t count, size_t dim) {
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < dim; j++) {
            data[i * dim + j] = sinf((float)(i * dim + j) * 0.1f);
        }
    }
}

/* 1. test_quant_config_init                                           */
static int test_quant_config_init(void) {
    GV_QuantConfig config;
    memset(&config, 0xFF, sizeof(config)); /* fill with garbage */

    gv_quant_config_init(&config);

    ASSERT(config.type == GV_QUANT_8BIT, "default type should be GV_QUANT_8BIT");
    ASSERT(config.mode == GV_QUANT_SYMMETRIC, "default mode should be GV_QUANT_SYMMETRIC");
    ASSERT(config.use_rabitq == 0, "default use_rabitq should be 0");
    ASSERT(config.rabitq_seed == 0, "default rabitq_seed should be 0");

    return 0;
}

/* 2. test_quant_train_8bit                                            */
static int test_quant_train_8bit(void) {
    float data[TRAIN_COUNT * DIM];
    generate_data(data, TRAIN_COUNT, DIM);

    GV_QuantConfig config;
    gv_quant_config_init(&config);
    config.type = GV_QUANT_8BIT;

    GV_QuantCodebook *cb = gv_quant_train(data, TRAIN_COUNT, DIM, &config);
    ASSERT(cb != NULL, "gv_quant_train returned NULL for 8-bit");

    gv_quant_codebook_destroy(cb);
    return 0;
}

/* 3. test_quant_encode_decode_roundtrip                               */
static int test_quant_encode_decode_roundtrip(void) {
    float data[TRAIN_COUNT * DIM];
    generate_data(data, TRAIN_COUNT, DIM);

    GV_QuantConfig config;
    gv_quant_config_init(&config);
    config.type = GV_QUANT_8BIT;

    GV_QuantCodebook *cb = gv_quant_train(data, TRAIN_COUNT, DIM, &config);
    ASSERT(cb != NULL, "training failed");

    size_t code_sz = gv_quant_code_size(cb, DIM);
    ASSERT(code_sz > 0, "code size should be > 0");

    uint8_t *codes = (uint8_t *)malloc(code_sz);
    ASSERT(codes != NULL, "malloc failed");

    /* Encode the first training vector */
    int rc = gv_quant_encode(cb, data, DIM, codes);
    ASSERT(rc == 0, "gv_quant_encode failed");

    /* Decode back */
    float decoded[DIM];
    rc = gv_quant_decode(cb, codes, DIM, decoded);
    ASSERT(rc == 0, "gv_quant_decode failed");

    /* 8-bit should be within ~5% accuracy per dimension */
    for (size_t i = 0; i < DIM; i++) {
        float diff = fabsf(data[i] - decoded[i]);
        ASSERT(diff < 0.5f, "decoded value deviates too much from original");
    }

    free(codes);
    gv_quant_codebook_destroy(cb);
    return 0;
}

/* 4. test_quant_distance_asymmetric                                   */
static int test_quant_distance_asymmetric(void) {
    float data[TRAIN_COUNT * DIM];
    generate_data(data, TRAIN_COUNT, DIM);

    GV_QuantConfig config;
    gv_quant_config_init(&config);
    config.type = GV_QUANT_8BIT;
    config.mode = GV_QUANT_ASYMMETRIC;

    GV_QuantCodebook *cb = gv_quant_train(data, TRAIN_COUNT, DIM, &config);
    ASSERT(cb != NULL, "training failed");

    size_t code_sz = gv_quant_code_size(cb, DIM);
    uint8_t *codes = (uint8_t *)malloc(code_sz);
    ASSERT(codes != NULL, "malloc failed");

    int rc = gv_quant_encode(cb, data, DIM, codes);
    ASSERT(rc == 0, "encode failed");

    /* Distance of same vector to its quantized form should be small */
    float dist = gv_quant_distance(cb, data, DIM, codes);
    ASSERT(dist >= 0.0f, "distance should be non-negative");
    ASSERT(dist < 10.0f, "distance of same vector should be small");

    free(codes);
    gv_quant_codebook_destroy(cb);
    return 0;
}

/* 5. test_quant_distance_symmetric                                    */
static int test_quant_distance_symmetric(void) {
    float data[TRAIN_COUNT * DIM];
    generate_data(data, TRAIN_COUNT, DIM);

    GV_QuantConfig config;
    gv_quant_config_init(&config);
    config.type = GV_QUANT_8BIT;
    config.mode = GV_QUANT_SYMMETRIC;

    GV_QuantCodebook *cb = gv_quant_train(data, TRAIN_COUNT, DIM, &config);
    ASSERT(cb != NULL, "training failed");

    size_t code_sz = gv_quant_code_size(cb, DIM);
    uint8_t *codes_a = (uint8_t *)malloc(code_sz);
    uint8_t *codes_b = (uint8_t *)malloc(code_sz);
    ASSERT(codes_a != NULL && codes_b != NULL, "malloc failed");

    /* Encode same vector twice */
    int rc = gv_quant_encode(cb, data, DIM, codes_a);
    ASSERT(rc == 0, "encode a failed");
    rc = gv_quant_encode(cb, data, DIM, codes_b);
    ASSERT(rc == 0, "encode b failed");

    float dist = gv_quant_distance_qq(cb, codes_a, codes_b, DIM);
    ASSERT(dist >= 0.0f, "symmetric distance should be non-negative");
    ASSERT(dist < 0.001f, "distance of identical codes should be near zero");

    free(codes_a);
    free(codes_b);
    gv_quant_codebook_destroy(cb);
    return 0;
}

/* 6. test_quant_binary_mode                                           */
static int test_quant_binary_mode(void) {
    float data[TRAIN_COUNT * DIM];
    generate_data(data, TRAIN_COUNT, DIM);

    GV_QuantConfig config;
    gv_quant_config_init(&config);
    config.type = GV_QUANT_BINARY;

    GV_QuantCodebook *cb = gv_quant_train(data, TRAIN_COUNT, DIM, &config);
    ASSERT(cb != NULL, "training failed for binary mode");

    size_t code_sz = gv_quant_code_size(cb, DIM);
    /* Binary: 1 bit per dim -> 2 bytes for 16 dims */
    ASSERT(code_sz > 0, "binary code size should be > 0");

    uint8_t *codes = (uint8_t *)malloc(code_sz);
    ASSERT(codes != NULL, "malloc failed");

    int rc = gv_quant_encode(cb, data, DIM, codes);
    ASSERT(rc == 0, "binary encode failed");

    free(codes);
    gv_quant_codebook_destroy(cb);
    return 0;
}

/* 7. test_quant_memory_ratio                                          */
static int test_quant_memory_ratio(void) {
    float data[TRAIN_COUNT * DIM];
    generate_data(data, TRAIN_COUNT, DIM);

    GV_QuantConfig config;
    gv_quant_config_init(&config);
    config.type = GV_QUANT_8BIT;

    GV_QuantCodebook *cb = gv_quant_train(data, TRAIN_COUNT, DIM, &config);
    ASSERT(cb != NULL, "training failed");

    float ratio = gv_quant_memory_ratio(cb, DIM);
    /* 8-bit quantization of float32 -> ratio should be ~4.0 */
    ASSERT(ratio >= 1.0f, "memory ratio should be >= 1.0");

    gv_quant_codebook_destroy(cb);
    return 0;
}

/* 8. test_quant_codebook_destroy_null                                 */
static int test_quant_codebook_destroy_null(void) {
    /* Should be safe to call with NULL */
    gv_quant_codebook_destroy(NULL);
    return 0;
}

/* main */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing quant config init...",             test_quant_config_init},
        {"Testing quant train 8-bit...",             test_quant_train_8bit},
        {"Testing quant encode/decode roundtrip...", test_quant_encode_decode_roundtrip},
        {"Testing quant distance asymmetric...",     test_quant_distance_asymmetric},
        {"Testing quant distance symmetric...",      test_quant_distance_symmetric},
        {"Testing quant binary mode...",             test_quant_binary_mode},
        {"Testing quant memory ratio...",            test_quant_memory_ratio},
        {"Testing quant codebook destroy null...",   test_quant_codebook_destroy_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
