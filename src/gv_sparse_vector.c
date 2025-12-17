#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_sparse_vector.h"

GV_SparseVector *gv_sparse_vector_create(size_t dimension,
                                         const uint32_t *indices,
                                         const float *values,
                                         size_t nnz) {
    if (dimension == 0) {
        return NULL;
    }
    if ((indices == NULL || values == NULL) && nnz > 0) {
        return NULL;
    }
    GV_SparseVector *sv = (GV_SparseVector *)calloc(1, sizeof(GV_SparseVector));
    if (!sv) {
        return NULL;
    }
    sv->dimension = dimension;
    sv->nnz = nnz;
    if (nnz > 0) {
        sv->entries = (GV_SparseEntry *)malloc(nnz * sizeof(GV_SparseEntry));
        if (!sv->entries) {
            free(sv);
            return NULL;
        }
        for (size_t i = 0; i < nnz; ++i) {
            sv->entries[i].index = indices[i];
            sv->entries[i].value = values[i];
        }
    }
    sv->metadata = NULL;
    return sv;
}

void gv_sparse_vector_destroy(GV_SparseVector *sv) {
    if (!sv) return;
    free(sv->entries);
    gv_metadata_destroy(sv->metadata);
    free(sv);
}



