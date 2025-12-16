#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gigavector/gv_binary_quant.h"

#ifdef __SSE4_2__
#include <nmmintrin.h>
#include <emmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __POPCNT__
#include <popcntintrin.h>
#endif

size_t gv_binary_bytes_needed(size_t dimension) {
    return (dimension + 7) / 8;
}

GV_BinaryVector *gv_binary_quantize(const float *data, size_t dimension) {
    if (data == NULL || dimension == 0) {
        return NULL;
    }

    GV_BinaryVector *bv = (GV_BinaryVector *)malloc(sizeof(GV_BinaryVector));
    if (bv == NULL) {
        return NULL;
    }

    bv->dimension = dimension;
    bv->bytes_per_vector = gv_binary_bytes_needed(dimension);
    bv->bits = (uint8_t *)calloc(bv->bytes_per_vector, sizeof(uint8_t));
    if (bv->bits == NULL) {
        free(bv);
        return NULL;
    }

    for (size_t i = 0; i < dimension; ++i) {
        if (data[i] >= 0.0f) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            bv->bits[byte_idx] |= (1U << (7 - bit_idx));
        }
    }

    return bv;
}

GV_BinaryVector *gv_binary_vector_wrap(uint8_t *bits, size_t dimension) {
    if (bits == NULL || dimension == 0) {
        return NULL;
    }

    GV_BinaryVector *bv = (GV_BinaryVector *)malloc(sizeof(GV_BinaryVector));
    if (bv == NULL) {
        return NULL;
    }

    bv->dimension = dimension;
    bv->bytes_per_vector = gv_binary_bytes_needed(dimension);
    bv->bits = bits;
    return bv;
}

void gv_binary_vector_destroy(GV_BinaryVector *bv) {
    if (bv == NULL) {
        return;
    }
    if (bv->bits != NULL) {
        free(bv->bits);
    }
    free(bv);
}

#ifdef __SSE4_2__
static size_t gv_popcount_sse(uint64_t x) {
    return (size_t)_mm_popcnt_u64(x);
}
#endif

static size_t gv_popcount(uint64_t x) {
#ifdef __SSE4_2__
    return gv_popcount_sse(x);
#else
    size_t count = 0;
    while (x) {
        count += (x & 1);
        x >>= 1;
    }
    return count;
#endif
}

size_t gv_binary_hamming_distance(const GV_BinaryVector *a, const GV_BinaryVector *b) {
    if (a == NULL || b == NULL || a->bits == NULL || b->bits == NULL) {
        return SIZE_MAX;
    }
    if (a->dimension != b->dimension) {
        return SIZE_MAX;
    }

    return gv_binary_hamming_distance_fast(a, b);
}

size_t gv_binary_hamming_distance_fast(const GV_BinaryVector *a, const GV_BinaryVector *b) {
    if (a == NULL || b == NULL || a->bits == NULL || b->bits == NULL) {
        return SIZE_MAX;
    }
    if (a->dimension != b->dimension) {
        return SIZE_MAX;
    }

    size_t distance = 0;
    size_t bytes = a->bytes_per_vector;
    size_t full_uint64s = bytes / 8;
    size_t remaining_bytes = bytes % 8;

#ifdef __AVX2__
    if (full_uint64s >= 4) {
        size_t avx_count = (full_uint64s / 4) * 4;
        for (size_t i = 0; i < avx_count; i += 4) {
            __m256i va = _mm256_loadu_si256((__m256i *)(a->bits + i * 8));
            __m256i vb = _mm256_loadu_si256((__m256i *)(b->bits + i * 8));
            __m256i xored = _mm256_xor_si256(va, vb);
            
            __m128i low = _mm256_extracti128_si256(xored, 0);
            __m128i high = _mm256_extracti128_si256(xored, 1);
            
            uint64_t low64 = _mm_extract_epi64(low, 0);
            uint64_t low64_1 = _mm_extract_epi64(low, 1);
            uint64_t high64 = _mm_extract_epi64(high, 0);
            uint64_t high64_1 = _mm_extract_epi64(high, 1);
            
            distance += gv_popcount(low64);
            distance += gv_popcount(low64_1);
            distance += gv_popcount(high64);
            distance += gv_popcount(high64_1);
        }
        
        size_t processed = avx_count * 8;
        for (size_t i = processed; i < bytes - remaining_bytes; i += 8) {
            uint64_t x = *(uint64_t *)(a->bits + i) ^ *(uint64_t *)(b->bits + i);
            distance += gv_popcount(x);
        }
    } else {
#endif
        for (size_t i = 0; i < full_uint64s * 8; i += 8) {
            uint64_t x = *(uint64_t *)(a->bits + i) ^ *(uint64_t *)(b->bits + i);
            distance += gv_popcount(x);
        }
#ifdef __AVX2__
    }
#endif

    for (size_t i = full_uint64s * 8; i < bytes; ++i) {
        uint8_t x = a->bits[i] ^ b->bits[i];
        distance += gv_popcount((uint64_t)x);
    }

    size_t bits_used = a->dimension;
    size_t bits_in_last_byte = bits_used % 8;
    if (bits_in_last_byte > 0 && bytes > 0) {
        uint8_t mask = (uint8_t)(0xFF << (8 - bits_in_last_byte));
        uint8_t last_byte_a = a->bits[bytes - 1] & mask;
        uint8_t last_byte_b = b->bits[bytes - 1] & mask;
        uint8_t last_xor = last_byte_a ^ last_byte_b;
        
        size_t last_distance = gv_popcount((uint64_t)last_xor);
        size_t full_last_distance = gv_popcount((uint64_t)(a->bits[bytes - 1] ^ b->bits[bytes - 1]));
        distance = distance - full_last_distance + last_distance;
    }

    return distance;
}

