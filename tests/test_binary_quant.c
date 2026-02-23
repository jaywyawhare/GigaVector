#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_binary_quant.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 16

/* 1. test_binary_quantize_basic                                       */
static int test_binary_quantize_basic(void) {
    float data[DIM];
    for (size_t i = 0; i < DIM; i++) {
        data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    GV_BinaryVector *bv = gv_binary_quantize(data, DIM);
    ASSERT(bv != NULL, "gv_binary_quantize returned NULL");
    ASSERT(bv->dimension == DIM, "dimension mismatch");
    ASSERT(bv->bytes_per_vector == gv_binary_bytes_needed(DIM),
           "bytes_per_vector mismatch");

    gv_binary_vector_destroy(bv);
    return 0;
}

/* 2. test_binary_bytes_needed                                         */
static int test_binary_bytes_needed(void) {
    /* 8 dimensions -> 1 byte */
    ASSERT(gv_binary_bytes_needed(8) == 1, "8 dims should need 1 byte");
    /* 16 dimensions -> 2 bytes */
    ASSERT(gv_binary_bytes_needed(16) == 2, "16 dims should need 2 bytes");
    /* 1 dimension -> 1 byte (rounded up) */
    ASSERT(gv_binary_bytes_needed(1) == 1, "1 dim should need 1 byte");
    /* 9 dimensions -> 2 bytes (rounded up) */
    ASSERT(gv_binary_bytes_needed(9) == 2, "9 dims should need 2 bytes");
    /* 0 dimensions -> 0 bytes */
    ASSERT(gv_binary_bytes_needed(0) == 0, "0 dims should need 0 bytes");

    return 0;
}

/* 3. test_binary_hamming_identical                                    */
static int test_binary_hamming_identical(void) {
    float data[DIM];
    for (size_t i = 0; i < DIM; i++) {
        data[i] = sinf((float)i);
    }

    GV_BinaryVector *a = gv_binary_quantize(data, DIM);
    GV_BinaryVector *b = gv_binary_quantize(data, DIM);
    ASSERT(a != NULL && b != NULL, "quantization failed");

    size_t dist = gv_binary_hamming_distance(a, b);
    ASSERT(dist == 0, "identical vectors should have hamming distance 0");

    gv_binary_vector_destroy(a);
    gv_binary_vector_destroy(b);
    return 0;
}

/* 4. test_binary_hamming_opposite                                     */
static int test_binary_hamming_opposite(void) {
    float pos[DIM], neg[DIM];
    for (size_t i = 0; i < DIM; i++) {
        pos[i] =  1.0f;
        neg[i] = -1.0f;
    }

    GV_BinaryVector *a = gv_binary_quantize(pos, DIM);
    GV_BinaryVector *b = gv_binary_quantize(neg, DIM);
    ASSERT(a != NULL && b != NULL, "quantization failed");

    size_t dist = gv_binary_hamming_distance(a, b);
    ASSERT(dist == DIM, "opposite vectors should have max hamming distance");

    gv_binary_vector_destroy(a);
    gv_binary_vector_destroy(b);
    return 0;
}

/* 5. test_binary_hamming_fast_matches_normal                          */
static int test_binary_hamming_fast_matches_normal(void) {
    float data_a[DIM], data_b[DIM];
    for (size_t i = 0; i < DIM; i++) {
        data_a[i] = sinf((float)i);
        data_b[i] = cosf((float)i);
    }

    GV_BinaryVector *a = gv_binary_quantize(data_a, DIM);
    GV_BinaryVector *b = gv_binary_quantize(data_b, DIM);
    ASSERT(a != NULL && b != NULL, "quantization failed");

    size_t dist_normal = gv_binary_hamming_distance(a, b);
    size_t dist_fast   = gv_binary_hamming_distance_fast(a, b);
    ASSERT(dist_normal == dist_fast,
           "fast hamming distance should match normal hamming distance");

    gv_binary_vector_destroy(a);
    gv_binary_vector_destroy(b);
    return 0;
}

/* 6. test_binary_vector_wrap                                          */
static int test_binary_vector_wrap(void) {
    size_t nbytes = gv_binary_bytes_needed(DIM);
    uint8_t *bits = (uint8_t *)calloc(nbytes, 1);
    ASSERT(bits != NULL, "calloc failed");

    /* Set some bits manually */
    bits[0] = 0xAA; /* 10101010 */

    GV_BinaryVector *bv = gv_binary_vector_wrap(bits, DIM);
    ASSERT(bv != NULL, "gv_binary_vector_wrap returned NULL");
    ASSERT(bv->dimension == DIM, "dimension mismatch after wrap");
    ASSERT(bv->bytes_per_vector == nbytes, "bytes_per_vector mismatch after wrap");

    gv_binary_vector_destroy(bv);
    /* bits is owned by bv after wrap — do NOT free(bits) */
    return 0;
}

/* 7. test_binary_destroy_null                                         */
static int test_binary_destroy_null(void) {
    /* Should be safe to call with NULL */
    gv_binary_vector_destroy(NULL);
    return 0;
}

/* 8. test_binary_quantize_sign_threshold                              */
static int test_binary_quantize_sign_threshold(void) {
    /* Test that the sign threshold is exactly 0:
       values >= 0 map to 1, values < 0 map to 0 */
    float data[8] = { 0.0f, -0.0001f, 0.0001f, -1.0f, 1.0f, 0.5f, -0.5f, 0.0f };

    GV_BinaryVector *bv = gv_binary_quantize(data, 8);
    ASSERT(bv != NULL, "quantization failed");

    /* Expected bits: 1,0,1,0,1,1,0,1 = 0xB5 (LSB first) or 0xAD (MSB first)
       The exact encoding depends on bit order, but hamming distance
       between this vector and an all-positive vector should tell us
       how many negatives there are. */

    float all_pos[8] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    GV_BinaryVector *bv_pos = gv_binary_quantize(all_pos, 8);
    ASSERT(bv_pos != NULL, "quantization of all-positive failed");

    size_t dist = gv_binary_hamming_distance(bv, bv_pos);
    /* data has 3 negative values: -0.0001, -1.0, -0.5 */
    ASSERT(dist == 3, "expected 3 differing bits for 3 negative values");

    gv_binary_vector_destroy(bv);
    gv_binary_vector_destroy(bv_pos);
    return 0;
}

/* main */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing binary quantize basic...",             test_binary_quantize_basic},
        {"Testing binary bytes needed...",               test_binary_bytes_needed},
        {"Testing binary hamming identical...",           test_binary_hamming_identical},
        {"Testing binary hamming opposite...",            test_binary_hamming_opposite},
        {"Testing binary hamming fast matches normal...", test_binary_hamming_fast_matches_normal},
        {"Testing binary vector wrap...",                 test_binary_vector_wrap},
        {"Testing binary destroy null...",                test_binary_destroy_null},
        {"Testing binary quantize sign threshold...",     test_binary_quantize_sign_threshold},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
