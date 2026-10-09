#include <math.h>
#include <string.h>

#include "search/distance.h"
#include "core/config.h"

#ifdef __SSE4_2__
#include <nmmintrin.h>
#include <emmintrin.h>
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/*
 * Hot-path SIMD dispatch read. cpu_detect_features() is correct but does an
 * atomic acquire load through a cross-TU call on every invocation; the distance
 * kernels below ran it 2-3x per call, which cost ~6% of all instructions in an
 * e2e search profile. The CPU feature set is immutable after first detection,
 * so cache it in a file-local the first time and read it directly thereafter.
 * Accessed with relaxed atomics so parallel searches don't data-race the cache
 * (ThreadSanitizer-clean); on x86 a relaxed load/store is a plain mov with no
 * barrier, so the fast path stays as cheap as a bare read. The first-use
 * recompute is idempotent, matching the contract cpu_detect_features documents.
 * Callers fetch it ONCE into a local and bit-test, instead of calling
 * cpu_has_feature() per candidate kernel.
 */
static unsigned s_dist_feats = (unsigned)-1;
static inline unsigned dist_features(void) {
    unsigned f = __atomic_load_n(&s_dist_feats, __ATOMIC_RELAXED);
    if (f == (unsigned)-1) {
        f = cpu_detect_features();
        __atomic_store_n(&s_dist_feats, f, __ATOMIC_RELAXED);
    }
    return f;
}

#ifdef __AVX2__
/* Horizontal sum of an 8-lane vector. */
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

#ifdef __AVX512F__
static float vector_dot_avx512(const float *a, const float *b, size_t dimension) {
    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= dimension; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
    }

    float tmp[16];
    _mm512_storeu_ps(tmp, sum_vec);
    float sum = 0.0f;
    for (int t = 0; t < 16; ++t) {
        sum += tmp[t];
    }

    for (; i < dimension; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

static float vector_norm_avx512(const float *v, size_t dimension) {
    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= dimension; i += 16) {
        __m512 vv = _mm512_loadu_ps(&v[i]);
        sum_vec = _mm512_fmadd_ps(vv, vv, sum_vec);
    }

    float tmp[16];
    _mm512_storeu_ps(tmp, sum_vec);
    float sum_sq = 0.0f;
    for (int t = 0; t < 16; ++t) {
        sum_sq += tmp[t];
    }

    for (; i < dimension; ++i) {
        float val = v[i];
        sum_sq += val * val;
    }

    return sqrtf(sum_sq);
}
#endif

#ifdef __AVX2__
static float vector_dot_avx2(const float *a, const float *b, size_t dimension) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= dimension; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
    }
    
    float sum = 0.0f;
    __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum = _mm_cvtss_f32(sum_128);
    
    for (; i < dimension; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

static float vector_norm_avx2(const float *v, size_t dimension) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= dimension; i += 8) {
        __m256 vv = _mm256_loadu_ps(&v[i]);
        sum_vec = _mm256_fmadd_ps(vv, vv, sum_vec);
    }
    
    float sum_sq = 0.0f;
    __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_sq = _mm_cvtss_f32(sum_128);
    
    for (; i < dimension; ++i) {
        sum_sq += v[i] * v[i];
    }
    
    return sqrtf(sum_sq);
}
#endif

#ifdef __SSE4_2__
static float vector_dot_sse(const float *a, const float *b, size_t dimension) {
    __m128 sum_vec = _mm_setzero_ps();
    size_t i = 0;
    
    for (; i + 4 <= dimension; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(va, vb));
    }
    
    float sum = 0.0f;
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum = _mm_cvtss_f32(sum_vec);
    
    for (; i < dimension; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

static float vector_norm_sse(const float *v, size_t dimension) {
    __m128 sum_vec = _mm_setzero_ps();
    size_t i = 0;
    
    for (; i + 4 <= dimension; i += 4) {
        __m128 vv = _mm_loadu_ps(&v[i]);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(vv, vv));
    }
    
    float sum_sq = 0.0f;
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_sq = _mm_cvtss_f32(sum_vec);
    
    for (; i < dimension; ++i) {
        sum_sq += v[i] * v[i];
    }
    
    return sqrtf(sum_sq);
}
#endif

static float vector_dot_scalar(const float *a, const float *b, size_t dimension) {
    float sum = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static float vector_norm_scalar(const float *v, size_t dimension) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        sum_sq += v[i] * v[i];
    }
    return sqrtf(sum_sq);
}

static float vector_dot(const GV_Vector *a, const GV_Vector *b) {
    unsigned feats = dist_features();
    (void)feats;
#ifdef __AVX512F__
    if ((feats & GV_CPU_FEATURE_AVX512F) && a->dimension >= 32 && (a->dimension % 16 == 0)) {
        return vector_dot_avx512(a->data, b->data, a->dimension);
    }
#endif
#ifdef __AVX2__
    if ((feats & GV_CPU_FEATURE_AVX2) && (feats & GV_CPU_FEATURE_FMA) && a->dimension >= 16 && (a->dimension % 8 == 0)) {
        return vector_dot_avx2(a->data, b->data, a->dimension);
    }
#endif
#ifdef __SSE4_2__
    if ((feats & GV_CPU_FEATURE_SSE4_2) && a->dimension >= 8 && (a->dimension % 4 == 0)) {
        return vector_dot_sse(a->data, b->data, a->dimension);
    }
#endif
    return vector_dot_scalar(a->data, b->data, a->dimension);
}

static float vector_norm(const GV_Vector *v) {
    unsigned feats = dist_features();
    (void)feats;
#ifdef __AVX512F__
    if ((feats & GV_CPU_FEATURE_AVX512F) && v->dimension >= 32 && (v->dimension % 16 == 0)) {
        return vector_norm_avx512(v->data, v->dimension);
    }
#endif
#ifdef __AVX2__
    if ((feats & GV_CPU_FEATURE_AVX2) && (feats & GV_CPU_FEATURE_FMA) && v->dimension >= 16 && (v->dimension % 8 == 0)) {
        return vector_norm_avx2(v->data, v->dimension);
    }
#endif
#ifdef __SSE4_2__
    if ((feats & GV_CPU_FEATURE_SSE4_2) && v->dimension >= 8 && (v->dimension % 4 == 0)) {
        return vector_norm_sse(v->data, v->dimension);
    }
#endif
    return vector_norm_scalar(v->data, v->dimension);
}

#ifdef __AVX512F__
static float distance_euclidean_avx512(const float *a, const float *b, size_t dimension) {
    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= dimension; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }

    float tmp[16];
    _mm512_storeu_ps(tmp, sum_vec);
    float sum_sq_diff = 0.0f;
    for (int t = 0; t < 16; ++t) {
        sum_sq_diff += tmp[t];
    }

    for (; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
    }

    return sqrtf(sum_sq_diff);
}
#endif

#ifdef __AVX2__
static float distance_euclidean_avx2(const float *a, const float *b, size_t dimension) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= dimension; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }
    
    float sum_sq_diff = 0.0f;
    __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_sq_diff = _mm_cvtss_f32(sum_128);
    
    for (; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
    }
    
    return sqrtf(sum_sq_diff);
}
#endif

#ifdef __SSE4_2__
static float distance_euclidean_sse(const float *a, const float *b, size_t dimension) {
    __m128 sum_vec = _mm_setzero_ps();
    size_t i = 0;
    
    for (; i + 4 <= dimension; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 diff = _mm_sub_ps(va, vb);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(diff, diff));
    }
    
    float sum_sq_diff = 0.0f;
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_sq_diff = _mm_cvtss_f32(sum_vec);
    
    for (; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
    }
    
    return sqrtf(sum_sq_diff);
}
#endif

static float distance_euclidean_scalar(const float *a, const float *b, size_t dimension) {
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
    }
    return sqrtf(sum_sq_diff);
}

float distance_euclidean(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -1.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -1.0f;
    }

    unsigned feats = dist_features();
    (void)feats;
#ifdef __AVX512F__
    if ((feats & GV_CPU_FEATURE_AVX512F) && a->dimension >= 16 && (a->dimension % 16 == 0)) {
        return distance_euclidean_avx512(a->data, b->data, a->dimension);
    }
#endif
#ifdef __AVX2__
    if ((feats & GV_CPU_FEATURE_AVX2) && (feats & GV_CPU_FEATURE_FMA) && a->dimension >= 8 && (a->dimension % 8 == 0)) {
        return distance_euclidean_avx2(a->data, b->data, a->dimension);
    }
#endif
#ifdef __SSE4_2__
    if ((feats & GV_CPU_FEATURE_SSE4_2) && a->dimension >= 4 && (a->dimension % 4 == 0)) {
        return distance_euclidean_sse(a->data, b->data, a->dimension);
    }
#endif
    return distance_euclidean_scalar(a->data, b->data, a->dimension);
}

float distance_cosine(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -2.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -2.0f;
    }

    float dot_product = vector_dot(a, b);
    float norm_a = vector_norm(a);
    float norm_b = vector_norm(b);

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 1.0f;
    }

    return 1.0f - (dot_product / (norm_a * norm_b));
}

#ifdef __AVX2__
static float distance_manhattan_avx2(const float *a, const float *b, size_t dimension) {
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    size_t i = 0;
    
    for (; i + 8 <= dimension; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 abs_diff = _mm256_andnot_ps(sign_mask, diff);
        sum_vec = _mm256_add_ps(sum_vec, abs_diff);
    }
    
    float sum = 0.0f;
    __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum = _mm_cvtss_f32(sum_128);
    
    for (; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum += (diff < 0.0f) ? -diff : diff;
    }
    
    return sum;
}
#endif

#ifdef __SSE4_2__
static float distance_manhattan_sse(const float *a, const float *b, size_t dimension) {
    __m128 sum_vec = _mm_setzero_ps();
    __m128 sign_mask = _mm_set1_ps(-0.0f);
    size_t i = 0;
    
    for (; i + 4 <= dimension; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 abs_diff = _mm_andnot_ps(sign_mask, diff);
        sum_vec = _mm_add_ps(sum_vec, abs_diff);
    }
    
    float sum = 0.0f;
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum = _mm_cvtss_f32(sum_vec);
    
    for (; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum += (diff < 0.0f) ? -diff : diff;
    }
    
    return sum;
}
#endif

static float distance_manhattan_scalar(const float *a, const float *b, size_t dimension) {
    float sum = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum += (diff < 0.0f) ? -diff : diff;
    }
    return sum;
}

float distance_manhattan(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -1.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -1.0f;
    }

    unsigned feats = dist_features();
    (void)feats;
#ifdef __AVX2__
    if (feats & GV_CPU_FEATURE_AVX2) {
        return distance_manhattan_avx2(a->data, b->data, a->dimension);
    }
#endif
#ifdef __SSE4_2__
    if (feats & GV_CPU_FEATURE_SSE4_2) {
        return distance_manhattan_sse(a->data, b->data, a->dimension);
    }
#endif
    return distance_manhattan_scalar(a->data, b->data, a->dimension);
}

float distance_dot_product(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -1.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -1.0f;
    }

    float dot = vector_dot(a, b);
    return -dot;
}

#ifdef __AVX2__
/* Binarize (>0) eight lanes at a time, XOR the sign masks, popcount the
 * differing lanes. Replaces a per-element data-dependent branch that mispredicts
 * ~50% of the time on real (sign-balanced) data - the dominant cost. */
static float distance_hamming_avx2(const float *a, const float *b, size_t dimension) {
    const __m256 zero = _mm256_setzero_ps();
    int diff = 0;
    size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        int ma = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(&a[i]), zero, _CMP_GT_OQ));
        int mb = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(&b[i]), zero, _CMP_GT_OQ));
        diff += __builtin_popcount((unsigned)(ma ^ mb));
    }
    for (; i < dimension; i++)
        diff += ((a[i] > 0.0f) ? 1 : 0) ^ ((b[i] > 0.0f) ? 1 : 0);
    return (float)diff;
}
#endif

float distance_hamming(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -1.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -1.0f;
    }

#ifdef __AVX2__
    if (dist_features() & GV_CPU_FEATURE_AVX2) {
        return distance_hamming_avx2(a->data, b->data, a->dimension);
    }
#endif
    /* Branchless scalar: XOR of the two sign bits, no mispredicting branch. */
    int diff = 0;
    for (size_t i = 0; i < a->dimension; i++) {
        diff += ((a->data[i] > 0.0f) ? 1 : 0) ^ ((b->data[i] > 0.0f) ? 1 : 0);
    }
    return (float)diff;
}

#ifdef __AVX2__
/* Fused single pass accumulating dot, |a|^2 and |b|^2 together (three FMAs per
 * 8 lanes) instead of a scalar triple-accumulate loop. */
static void jaccard_sums_avx2(const float *a, const float *b, size_t dimension,
                              float *out_dot, float *out_na, float *out_nb) {
    __m256 vdot = _mm256_setzero_ps(), vna = _mm256_setzero_ps(), vnb = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        __m256 x = _mm256_loadu_ps(&a[i]);
        __m256 y = _mm256_loadu_ps(&b[i]);
        vdot = _mm256_fmadd_ps(x, y, vdot);
        vna  = _mm256_fmadd_ps(x, x, vna);
        vnb  = _mm256_fmadd_ps(y, y, vnb);
    }
    float dot = hsum256_ps(vdot), na = hsum256_ps(vna), nb = hsum256_ps(vnb);
    for (; i < dimension; i++) {
        float x = a[i], y = b[i];
        dot += x * y; na += x * x; nb += y * y;
    }
    *out_dot = dot; *out_na = na; *out_nb = nb;
}
#endif

float distance_jaccard(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -1.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -1.0f;
    }

    /* Continuous Tanimoto: dot / (|a|^2 + |b|^2 - dot). For binary inputs this is
     * exactly intersection/union. Both-zero vectors are identical (distance 0). */
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
#ifdef __AVX2__
    if ((dist_features() & GV_CPU_FEATURE_AVX2) && (dist_features() & GV_CPU_FEATURE_FMA)) {
        jaccard_sums_avx2(a->data, b->data, a->dimension, &dot, &na, &nb);
    } else
#endif
    for (size_t i = 0; i < a->dimension; i++) {
        float x = a->data[i], y = b->data[i];
        dot += x * y;
        na  += x * x;
        nb  += y * y;
    }
    float denom = na + nb - dot;
    if (denom <= 0.0f) return 0.0f;         /* a == b == 0 */
    float t = dot / denom;
    if (t > 1.0f) t = 1.0f; else if (t < 0.0f) t = 0.0f;
    return 1.0f - t;
}

float distance(const GV_Vector *a, const GV_Vector *b, GV_DistanceType type) {
    if (a == NULL || b == NULL) {
        return -1.0f;
    }

    switch (type) {
        case GV_DISTANCE_EUCLIDEAN:
            return distance_euclidean(a, b);
        case GV_DISTANCE_COSINE:
            return distance_cosine(a, b);
        case GV_DISTANCE_DOT_PRODUCT:
            return distance_dot_product(a, b);
        case GV_DISTANCE_MANHATTAN:
            return distance_manhattan(a, b);
        case GV_DISTANCE_HAMMING:
            return distance_hamming(a, b);
        case GV_DISTANCE_JACCARD:
            return distance_jaccard(a, b);
        default:
            return -1.0f;
    }
}

float distance_scalar(const GV_Vector *a, const GV_Vector *b, GV_DistanceType type) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -1.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -1.0f;
    }

    switch (type) {
        case GV_DISTANCE_EUCLIDEAN:
            return distance_euclidean_scalar(a->data, b->data, a->dimension);
        case GV_DISTANCE_MANHATTAN:
            return distance_manhattan_scalar(a->data, b->data, a->dimension);
        case GV_DISTANCE_DOT_PRODUCT:
            return -vector_dot_scalar(a->data, b->data, a->dimension);
        case GV_DISTANCE_COSINE: {
            float dot = vector_dot_scalar(a->data, b->data, a->dimension);
            float norm_a = vector_norm_scalar(a->data, a->dimension);
            float norm_b = vector_norm_scalar(b->data, b->dimension);
            if (norm_a == 0.0f || norm_b == 0.0f) {
                return 1.0f;
            }
            return 1.0f - (dot / (norm_a * norm_b));
        }
        case GV_DISTANCE_HAMMING:
            /* Hamming has no SIMD variant; distance() is already scalar. */
            return distance_hamming(a, b);
        case GV_DISTANCE_JACCARD:
            return distance_jaccard(a, b);
        default:
            return -1.0f;
    }
}

/* Fall back to four scalar ::distance calls for a metric without a batched
 * kernel, or when AVX2 is unavailable - keeps gv_distance_batch4 correct for
 * every metric. */
static void dist_batch4_fallback(const float *q,
                                 const float *v0, const float *v1,
                                 const float *v2, const float *v3,
                                 size_t dim, GV_DistanceType type,
                                 float *d0, float *d1, float *d2, float *d3) {
    GV_Vector qv = {0}, cv = {0};
    qv.dimension = dim; qv.data = (float *)q;
    cv.dimension = dim;
    cv.data = (float *)v0; *d0 = distance(&qv, &cv, type);
    cv.data = (float *)v1; *d1 = distance(&qv, &cv, type);
    cv.data = (float *)v2; *d2 = distance(&qv, &cv, type);
    cv.data = (float *)v3; *d3 = distance(&qv, &cv, type);
}

void gv_distance_batch4(const float *q,
                        const float *v0, const float *v1,
                        const float *v2, const float *v3,
                        size_t dim, GV_DistanceType type,
                        float *d0, float *d1, float *d2, float *d3) {
#ifdef __AVX2__
    if (!(dist_features() & GV_CPU_FEATURE_AVX2) || dim < 8) {
        dist_batch4_fallback(q, v0, v1, v2, v3, dim, type, d0, d1, d2, d3);
        return;
    }
    switch (type) {
    case GV_DISTANCE_EUCLIDEAN:
    case GV_DISTANCE_DOT_PRODUCT:
    case GV_DISTANCE_COSINE:
    case GV_DISTANCE_MANHATTAN:
        break;
    default:
        dist_batch4_fallback(q, v0, v1, v2, v3, dim, type, d0, d1, d2, d3);
        return;
    }

    /* Four independent accumulators hide FMA latency across candidates. For
     * COSINE we also need each candidate's norm plus the shared query norm. */
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 nq = _mm256_setzero_ps();
    __m256 n0 = _mm256_setzero_ps(), n1 = _mm256_setzero_ps();
    __m256 n2 = _mm256_setzero_ps(), n3 = _mm256_setzero_ps();
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vq = _mm256_loadu_ps(q + i);
        __m256 w0 = _mm256_loadu_ps(v0 + i), w1 = _mm256_loadu_ps(v1 + i);
        __m256 w2 = _mm256_loadu_ps(v2 + i), w3 = _mm256_loadu_ps(v3 + i);
        if (type == GV_DISTANCE_EUCLIDEAN) {
            __m256 e0 = _mm256_sub_ps(vq, w0), e1 = _mm256_sub_ps(vq, w1);
            __m256 e2 = _mm256_sub_ps(vq, w2), e3 = _mm256_sub_ps(vq, w3);
            a0 = _mm256_fmadd_ps(e0, e0, a0); a1 = _mm256_fmadd_ps(e1, e1, a1);
            a2 = _mm256_fmadd_ps(e2, e2, a2); a3 = _mm256_fmadd_ps(e3, e3, a3);
        } else if (type == GV_DISTANCE_MANHATTAN) {
            a0 = _mm256_add_ps(a0, _mm256_and_ps(absmask, _mm256_sub_ps(vq, w0)));
            a1 = _mm256_add_ps(a1, _mm256_and_ps(absmask, _mm256_sub_ps(vq, w1)));
            a2 = _mm256_add_ps(a2, _mm256_and_ps(absmask, _mm256_sub_ps(vq, w2)));
            a3 = _mm256_add_ps(a3, _mm256_and_ps(absmask, _mm256_sub_ps(vq, w3)));
        } else { /* DOT_PRODUCT or COSINE: accumulate dot products */
            a0 = _mm256_fmadd_ps(vq, w0, a0); a1 = _mm256_fmadd_ps(vq, w1, a1);
            a2 = _mm256_fmadd_ps(vq, w2, a2); a3 = _mm256_fmadd_ps(vq, w3, a3);
            if (type == GV_DISTANCE_COSINE) {
                nq = _mm256_fmadd_ps(vq, vq, nq);
                n0 = _mm256_fmadd_ps(w0, w0, n0); n1 = _mm256_fmadd_ps(w1, w1, n1);
                n2 = _mm256_fmadd_ps(w2, w2, n2); n3 = _mm256_fmadd_ps(w3, w3, n3);
            }
        }
    }
    float s0 = hsum256_ps(a0), s1 = hsum256_ps(a1), s2 = hsum256_ps(a2), s3 = hsum256_ps(a3);
    float q2 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    if (type == GV_DISTANCE_COSINE) {
        q2 = hsum256_ps(nq);
        c0 = hsum256_ps(n0); c1 = hsum256_ps(n1); c2 = hsum256_ps(n2); c3 = hsum256_ps(n3);
    }
    for (; i < dim; ++i) { /* scalar tail (dim % 8) */
        float x = q[i], y0 = v0[i], y1 = v1[i], y2 = v2[i], y3 = v3[i];
        if (type == GV_DISTANCE_EUCLIDEAN) {
            s0 += (x-y0)*(x-y0); s1 += (x-y1)*(x-y1); s2 += (x-y2)*(x-y2); s3 += (x-y3)*(x-y3);
        } else if (type == GV_DISTANCE_MANHATTAN) {
            s0 += fabsf(x-y0); s1 += fabsf(x-y1); s2 += fabsf(x-y2); s3 += fabsf(x-y3);
        } else {
            s0 += x*y0; s1 += x*y1; s2 += x*y2; s3 += x*y3;
            if (type == GV_DISTANCE_COSINE) {
                q2 += x*x; c0 += y0*y0; c1 += y1*y1; c2 += y2*y2; c3 += y3*y3;
            }
        }
    }

    switch (type) {
    case GV_DISTANCE_EUCLIDEAN:
        *d0 = sqrtf(s0); *d1 = sqrtf(s1); *d2 = sqrtf(s2); *d3 = sqrtf(s3);
        return;
    case GV_DISTANCE_MANHATTAN:
        *d0 = s0; *d1 = s1; *d2 = s2; *d3 = s3;
        return;
    case GV_DISTANCE_DOT_PRODUCT:
        *d0 = -s0; *d1 = -s1; *d2 = -s2; *d3 = -s3;
        return;
    case GV_DISTANCE_COSINE: {
        float nqn = sqrtf(q2);
        float e0 = nqn * sqrtf(c0), e1 = nqn * sqrtf(c1);
        float e2 = nqn * sqrtf(c2), e3 = nqn * sqrtf(c3);
        *d0 = (e0 == 0.0f) ? 1.0f : 1.0f - s0/e0;
        *d1 = (e1 == 0.0f) ? 1.0f : 1.0f - s1/e1;
        *d2 = (e2 == 0.0f) ? 1.0f : 1.0f - s2/e2;
        *d3 = (e3 == 0.0f) ? 1.0f : 1.0f - s3/e3;
        return;
    }
    default:
        break;
    }
#endif
    dist_batch4_fallback(q, v0, v1, v2, v3, dim, type, d0, d1, d2, d3);
}

