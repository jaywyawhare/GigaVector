#include <stdio.h>

#include "gigavector/gv_vector.h"

static void print_vector(const GV_Vector *vector) {
    printf("Vector (dim=%zu): [", vector->dimension);
    for (size_t i = 0; i < vector->dimension; ++i) {
        float value = 0.0f;
        gv_vector_get(vector, i, &value);
        printf("%s%.3f", (i == 0) ? "" : ", ", value);
    }
    printf("]\n");
}

int main(void) {
    float sample_data[] = {1.0f, 2.0f, 3.5f};
    GV_Vector *vector = gv_vector_create_from_data(3, sample_data);
    if (vector == NULL) {
        fprintf(stderr, "Failed to create vector\n");
        return 1;
    }

    gv_vector_set(vector, 2, 4.25f);

    print_vector(vector);

    gv_vector_destroy(vector);
    return 0;
}