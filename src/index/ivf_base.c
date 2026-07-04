#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "index/ivf_base.h"

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

        for (size_t c = 0; c < nlist; c++) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < dim; d++) {
                    new_centroids[c * dim + d] /= (float)counts[c];
                }
            }
        }

        memcpy(out_centroids, new_centroids, nlist * dim * sizeof(float));
    }

    free(assign);
    free(new_centroids);
    free(counts);
    return 0;
}
