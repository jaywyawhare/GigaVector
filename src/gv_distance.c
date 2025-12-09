#include <math.h>
#include <string.h>

#include "gigavector/gv_distance.h"
#include "gigavector/gv_config.h"

#ifdef __SSE4_2__
#include <nmmintrin.h>
#include <emmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX2__
static float gv_vector_dot_avx2(const float *a, const float *b, size_t dimension) {
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

static float gv_vector_norm_avx2(const float *v, size_t dimension) {
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
static float gv_vector_dot_sse(const float *a, const float *b, size_t dimension) {
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

static float gv_vector_norm_sse(const float *v, size_t dimension) {
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

static float gv_vector_dot_scalar(const float *a, const float *b, size_t dimension) {
    float sum = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static float gv_vector_norm_scalar(const float *v, size_t dimension) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        sum_sq += v[i] * v[i];
    }
    return sqrtf(sum_sq);
}

static float gv_vector_dot(const GV_Vector *a, const GV_Vector *b) {
#ifdef __AVX2__
    if (gv_cpu_has_feature(GV_CPU_FEATURE_AVX2) && gv_cpu_has_feature(GV_CPU_FEATURE_FMA)) {
        return gv_vector_dot_avx2(a->data, b->data, a->dimension);
    }
#endif
#ifdef __SSE4_2__
    if (gv_cpu_has_feature(GV_CPU_FEATURE_SSE4_2)) {
        return gv_vector_dot_sse(a->data, b->data, a->dimension);
    }
#endif
    return gv_vector_dot_scalar(a->data, b->data, a->dimension);
}

static float gv_vector_norm(const GV_Vector *v) {
#ifdef __AVX2__
    if (gv_cpu_has_feature(GV_CPU_FEATURE_AVX2) && gv_cpu_has_feature(GV_CPU_FEATURE_FMA)) {
        return gv_vector_norm_avx2(v->data, v->dimension);
    }
#endif
#ifdef __SSE4_2__
    if (gv_cpu_has_feature(GV_CPU_FEATURE_SSE4_2)) {
        return gv_vector_norm_sse(v->data, v->dimension);
    }
#endif
    return gv_vector_norm_scalar(v->data, v->dimension);
}

#ifdef __AVX2__
static float gv_distance_euclidean_avx2(const float *a, const float *b, size_t dimension) {
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
static float gv_distance_euclidean_sse(const float *a, const float *b, size_t dimension) {
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

static float gv_distance_euclidean_scalar(const float *a, const float *b, size_t dimension) {
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
    }
    return sqrtf(sum_sq_diff);
}

float gv_distance_euclidean(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -1.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -1.0f;
    }

#ifdef __AVX2__
    if (gv_cpu_has_feature(GV_CPU_FEATURE_AVX2) && gv_cpu_has_feature(GV_CPU_FEATURE_FMA)) {
        return gv_distance_euclidean_avx2(a->data, b->data, a->dimension);
    }
#endif
#ifdef __SSE4_2__
    if (gv_cpu_has_feature(GV_CPU_FEATURE_SSE4_2)) {
        return gv_distance_euclidean_sse(a->data, b->data, a->dimension);
    }
#endif
    return gv_distance_euclidean_scalar(a->data, b->data, a->dimension);
}

float gv_distance_cosine(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -2.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -2.0f;
    }

    float dot_product = gv_vector_dot(a, b);
    float norm_a = gv_vector_norm(a);
    float norm_b = gv_vector_norm(b);

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }

    return dot_product / (norm_a * norm_b);
}

float gv_distance(const GV_Vector *a, const GV_Vector *b, GV_DistanceType type) {
    if (a == NULL || b == NULL) {
        return -1.0f;
    }

    switch (type) {
        case GV_DISTANCE_EUCLIDEAN:
            return gv_distance_euclidean(a, b);
        case GV_DISTANCE_COSINE:
            return gv_distance_cosine(a, b);
        default:
            return -1.0f;
    }
}

