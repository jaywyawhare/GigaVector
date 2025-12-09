#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_vector.h"

static int gv_vector_validate_index(const GV_Vector *vector, size_t index) {
    if (vector == NULL || vector->data == NULL) {
        return -1;
    }
    return (index < vector->dimension) ? 0 : -1;
}

GV_Vector *gv_vector_create(size_t dimension) {
    if (dimension == 0) {
        return NULL;
    }

    GV_Vector *vector = (GV_Vector *)malloc(sizeof(GV_Vector));
    if (vector == NULL) {
        return NULL;
    }

    vector->dimension = dimension;
    vector->data = (float *)calloc(dimension, sizeof(float));
    if (vector->data == NULL) {
        free(vector);
        return NULL;
    }

    return vector;
}

GV_Vector *gv_vector_create_from_data(size_t dimension, const float *data) {
    if (dimension == 0 || data == NULL) {
        return NULL;
    }

    GV_Vector *vector = gv_vector_create(dimension);
    if (vector == NULL) {
        return NULL;
    }

    memcpy(vector->data, data, dimension * sizeof(float));
    return vector;
}

void gv_vector_destroy(GV_Vector *vector) {
    if (vector == NULL) {
        return;
    }

    free(vector->data);
    vector->data = NULL;
    free(vector);
}

int gv_vector_set(GV_Vector *vector, size_t index, float value) {
    if (gv_vector_validate_index(vector, index) != 0) {
        return -1;
    }

    vector->data[index] = value;
    return 0;
}

int gv_vector_get(const GV_Vector *vector, size_t index, float *out_value) {
    if (out_value == NULL || gv_vector_validate_index(vector, index) != 0) {
        return -1;
    }

    *out_value = vector->data[index];
    return 0;
}

int gv_vector_clear(GV_Vector *vector) {
    if (vector == NULL || vector->data == NULL) {
        return -1;
    }

    memset(vector->data, 0, vector->dimension * sizeof(float));
    return 0;
}

