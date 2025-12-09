#include <math.h>
#include <string.h>

#include "gigavector/gv_distance.h"

static float gv_vector_dot(const GV_Vector *a, const GV_Vector *b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a->dimension; ++i) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

static float gv_vector_norm(const GV_Vector *v) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < v->dimension; ++i) {
        sum_sq += v->data[i] * v->data[i];
    }
    return sqrtf(sum_sq);
}

float gv_distance_euclidean(const GV_Vector *a, const GV_Vector *b) {
    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        return -1.0f;
    }
    if (a->dimension != b->dimension || a->dimension == 0) {
        return -1.0f;
    }

    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < a->dimension; ++i) {
        float diff = a->data[i] - b->data[i];
        sum_sq_diff += diff * diff;
    }
    return sqrtf(sum_sq_diff);
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

