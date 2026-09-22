#ifndef GIGAVECTOR_INDEX_IVF_BASE_H
#define GIGAVECTOR_INDEX_IVF_BASE_H

#include <stddef.h>
#include <stdio.h>
#include "core/types.h"

/* Assign each of the `count` vectors in `data` to its nearest centroid.
   Results written to `assign` (length count). */
void ivf_assign_to_list(const float *data, size_t count, size_t dim,
                        const float *centroids, size_t nlist, int *assign);

/* Index of the centroid nearest `vec` (squared-L2). Used on the insert path to
   pick a vector's inverted list. */
size_t ivf_nearest_centroid(const float *vec, const float *centroids,
                            size_t nlist, size_t dim);

/* Lloyd's k-means on `data` (count x dim) for `iters` iterations.
   `nlist` initial centroids are seeded from the first `nlist` rows of `data`.
   Returns 0 on success, -1 on OOM or invalid args. */
int ivf_train_centroids(const float *data, size_t count, size_t dim,
                        size_t nlist, size_t iters, float *out_centroids);

/* Copy `data` (count x dim) into a fresh gv_alloc'd buffer, L2-normalising each
   row when `use_cosine` is set (so cosine reduces to inner product). The caller
   owns the result and frees it with gv_free. Returns NULL on OOM. */
float *ivf_prepare_training_buffer(const float *data, size_t count, size_t dim,
                                   int use_cosine);

/* Serialize one inverted-list entry to `out`: id, deleted flag, `dim` floats of
   vector data, then write_metadata(). For variants that store raw floats;
   quantizing variants write their payload then call write_metadata() directly. */
int ivf_write_entry(FILE *out, size_t id, int deleted, const GV_Vector *vec, size_t dim);

/* Populate a GV_SearchResult from an inverted-list entry's stored vector: deep-
   copies `vec` (data + metadata) into out->vector, and sets distance/id/flags.
   Shared by the IVF variants' result assembly. */
void ivf_fill_result(GV_SearchResult *out, const GV_Vector *vec, size_t id, float dist);

#endif
