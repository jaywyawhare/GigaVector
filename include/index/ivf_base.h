#ifndef GIGAVECTOR_INDEX_IVF_BASE_H
#define GIGAVECTOR_INDEX_IVF_BASE_H

#include <stddef.h>

/* Assign each of the `count` vectors in `data` to its nearest centroid.
   Results written to `assign` (length count). */
void ivf_assign_to_list(const float *data, size_t count, size_t dim,
                        const float *centroids, size_t nlist, int *assign);

/* Lloyd's k-means on `data` (count x dim) for `iters` iterations.
   `nlist` initial centroids are seeded from the first `nlist` rows of `data`.
   Returns 0 on success, -1 on OOM or invalid args. */
int ivf_train_centroids(const float *data, size_t count, size_t dim,
                        size_t nlist, size_t iters, float *out_centroids);

#endif
