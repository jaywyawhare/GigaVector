#include <stdlib.h>
#include "core/memory.h"
#include <string.h>

#include "storage/sparse_vector.h"
#include "schema/metadata.h"

GV_SparseVector *sparse_vector_create(size_t dimension,
                                         const uint32_t *indices,
                                         const float *values,
                                         size_t nnz) {
    if (dimension == 0) {
        return NULL;
    }
    if ((indices == NULL || values == NULL) && nnz > 0) {
        return NULL;
    }
    GV_SparseVector *sv = (GV_SparseVector *)gv_calloc(1, sizeof(GV_SparseVector));
    if (!sv) {
        return NULL;
    }
    sv->dimension = dimension;
    sv->nnz = nnz;
    if (nnz > 0) {
        sv->entries = (GV_SparseEntry *)gv_alloc(nnz * sizeof(GV_SparseEntry));
        if (!sv->entries) {
            gv_free(sv);
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

void sparse_vector_destroy(GV_SparseVector *sv) {
    if (!sv) return;
    if (sv->entries != NULL) {
        gv_free(sv->entries);
    }
    if (sv->metadata != NULL) {
        vector_clear_metadata((GV_Vector *)sv);
    }
    gv_free(sv);
}

