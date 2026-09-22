#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "index/ivf_base.h"
#include "core/memory.h"
#include "core/utils.h"
#include "schema/vector.h"
#include "schema/metadata.h"

void ivf_fill_result(GV_SearchResult *out, const GV_Vector *vec, size_t id, float dist) {
    GV_Vector *copy = vector_create_from_data(vec->dimension, vec->data);
    if (copy) vector_copy_metadata(copy, vec);
    out->vector = copy;
    out->distance = dist;
    out->is_sparse = 0;
    out->sparse_vector = NULL;
    out->id = id;
}

int ivf_write_entry(FILE *out, size_t id, int deleted, const GV_Vector *vec, size_t dim) {
    if (write_u32(out, (uint32_t)id) != 0) return -1;
    if (write_u32(out, (uint32_t)deleted) != 0) return -1;
    if (fwrite(vec->data, sizeof(float), dim, out) != dim) return -1;
    return write_metadata(out, vec->metadata);
}

float *ivf_prepare_training_buffer(const float *data, size_t count, size_t dim,
                                   int use_cosine) {
    float *buf = (float *)gv_alloc(count * dim * sizeof(float));
    if (!buf) return NULL;
    memcpy(buf, data, count * dim * sizeof(float));
    if (use_cosine) {
        for (size_t i = 0; i < count; ++i) {
            float *v = buf + i * dim;
            float norm = 0.0f;
            for (size_t j = 0; j < dim; ++j) norm += v[j] * v[j];
            if (norm > 0.0f) {
                norm = 1.0f / sqrtf(norm);
                for (size_t j = 0; j < dim; ++j) v[j] *= norm;
            }
        }
    }
    return buf;
}

size_t ivf_nearest_centroid(const float *vec, const float *centroids,
                            size_t nlist, size_t dim) {
    float best = INFINITY;
    size_t best_i = 0;
    for (size_t i = 0; i < nlist; i++) {
        const float *c = centroids + i * dim;
        float dist = 0.0f;
        for (size_t d = 0; d < dim; d++) { float diff = vec[d] - c[d]; dist += diff * diff; }
        if (dist < best) { best = dist; best_i = i; }
    }
    return best_i;
}

void ivf_assign_to_list(const float *data, size_t count, size_t dim,
                        const float *centroids, size_t nlist, int *assign) {
    for (size_t i = 0; i < count; i++) {
        const float *vec = data + i * dim;
        float best_dist = INFINITY;
        int best_idx = -1;
        for (size_t c = 0; c < nlist; c++) {
            const float *centroid = centroids + c * dim;
            float dist = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float diff = vec[d] - centroid[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = (int)c;
            }
        }
        assign[i] = best_idx;
    }
}

int ivf_train_centroids(const float *data, size_t count, size_t dim,
                        size_t nlist, size_t iters, float *out_centroids) {
    if (count < nlist || !data || !out_centroids) return -1;

    memcpy(out_centroids, data, nlist * dim * sizeof(float));

    int *assign = (int *)malloc(count * sizeof(int));
    float *new_centroids = (float *)calloc(nlist * dim, sizeof(float));
    size_t *counts = (size_t *)calloc(nlist, sizeof(size_t));

    if (!assign || !new_centroids || !counts) {
        free(assign);
        free(new_centroids);
        free(counts);
        return -1;
    }

    for (size_t iter = 0; iter < iters; iter++) {
        ivf_assign_to_list(data, count, dim, out_centroids, nlist, assign);

        memset(new_centroids, 0, nlist * dim * sizeof(float));
        memset(counts, 0, nlist * sizeof(size_t));

        for (size_t i = 0; i < count; i++) {
            int c = assign[i];
            if (c < 0) continue;
            const float *vec = data + i * dim;
            for (size_t d = 0; d < dim; d++) {
                new_centroids[c * dim + d] += vec[d];
            }
            counts[c]++;
        }

        /* Write back only clusters that received at least one point, directly
         * into out_centroids. Empty clusters (counts[c]==0) are left with their
         * previous centroid instead of being reset to all-zeros. */
        for (size_t c = 0; c < nlist; c++) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < dim; d++) {
                    out_centroids[c * dim + d] = new_centroids[c * dim + d] / (float)counts[c];
                }
            }
        }
    }

    free(assign);
    free(new_centroids);
    free(counts);
    return 0;
}
