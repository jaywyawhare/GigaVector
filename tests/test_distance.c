#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gigavector/gigavector.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b, msg) \
    do {                           \
        if (fabsf((a) - (b)) > 1e-5f) { \
            fprintf(stderr, "FAIL: %s (expected %.6f, got %.6f)\n", msg, (b), (a)); \
            return -1;             \
        }                          \
    } while (0)

static int test_euclidean_distance(void) {
    float v1_data[3] = {1.0f, 2.0f, 3.0f};
    float v2_data[3] = {4.0f, 5.0f, 6.0f};
    
    GV_Vector *v1 = gv_vector_create_from_data(3, v1_data);
    GV_Vector *v2 = gv_vector_create_from_data(3, v2_data);
    ASSERT(v1 != NULL, "v1 creation");
    ASSERT(v2 != NULL, "v2 creation");
    
    float dist = gv_distance_euclidean(v1, v2);
    float expected = sqrtf((4.0f-1.0f)*(4.0f-1.0f) + (5.0f-2.0f)*(5.0f-2.0f) + (6.0f-3.0f)*(6.0f-3.0f));
    ASSERT_FLOAT_EQ(dist, expected, "euclidean distance");
    
    float dist_zero = gv_distance_euclidean(v1, v1);
    ASSERT_FLOAT_EQ(dist_zero, 0.0f, "euclidean distance to self");
    
    gv_vector_destroy(v1);
    gv_vector_destroy(v2);
    return 0;
}

static int test_cosine_distance(void) {
    float v1_data[3] = {1.0f, 0.0f, 0.0f};
    float v2_data[3] = {0.0f, 1.0f, 0.0f};
    
    GV_Vector *v1 = gv_vector_create_from_data(3, v1_data);
    GV_Vector *v2 = gv_vector_create_from_data(3, v2_data);
    ASSERT(v1 != NULL, "v1 creation");
    ASSERT(v2 != NULL, "v2 creation");
    
    float sim = gv_distance_cosine(v1, v2);
    ASSERT_FLOAT_EQ(sim, 0.0f, "cosine similarity of orthogonal vectors");
    
    float sim_self = gv_distance_cosine(v1, v1);
    ASSERT_FLOAT_EQ(sim_self, 1.0f, "cosine similarity to self");
    
    float v3_data[3] = {2.0f, 0.0f, 0.0f};
    GV_Vector *v3 = gv_vector_create_from_data(3, v3_data);
    float sim_parallel = gv_distance_cosine(v1, v3);
    ASSERT_FLOAT_EQ(sim_parallel, 1.0f, "cosine similarity of parallel vectors");
    
    gv_vector_destroy(v1);
    gv_vector_destroy(v2);
    gv_vector_destroy(v3);
    return 0;
}

static int test_dot_product_distance(void) {
    float v1_data[3] = {1.0f, 2.0f, 3.0f};
    float v2_data[3] = {4.0f, 5.0f, 6.0f};
    
    GV_Vector *v1 = gv_vector_create_from_data(3, v1_data);
    GV_Vector *v2 = gv_vector_create_from_data(3, v2_data);
    ASSERT(v1 != NULL, "v1 creation");
    ASSERT(v2 != NULL, "v2 creation");
    
    float dist = gv_distance_dot_product(v1, v2);
    float expected_dot = 1.0f*4.0f + 2.0f*5.0f + 3.0f*6.0f;
    ASSERT_FLOAT_EQ(dist, -expected_dot, "dot product distance (negated)");
    
    gv_vector_destroy(v1);
    gv_vector_destroy(v2);
    return 0;
}

static int test_manhattan_distance(void) {
    float v1_data[3] = {1.0f, 2.0f, 3.0f};
    float v2_data[3] = {4.0f, 5.0f, 6.0f};
    
    GV_Vector *v1 = gv_vector_create_from_data(3, v1_data);
    GV_Vector *v2 = gv_vector_create_from_data(3, v2_data);
    ASSERT(v1 != NULL, "v1 creation");
    ASSERT(v2 != NULL, "v2 creation");
    
    float dist = gv_distance_manhattan(v1, v2);
    float expected = fabsf(4.0f-1.0f) + fabsf(5.0f-2.0f) + fabsf(6.0f-3.0f);
    ASSERT_FLOAT_EQ(dist, expected, "manhattan distance");
    
    float dist_zero = gv_distance_manhattan(v1, v1);
    ASSERT_FLOAT_EQ(dist_zero, 0.0f, "manhattan distance to self");
    
    gv_vector_destroy(v1);
    gv_vector_destroy(v2);
    return 0;
}

static int test_distance_null_vectors(void) {
    float v_data[3] = {1.0f, 2.0f, 3.0f};
    GV_Vector *v = gv_vector_create_from_data(3, v_data);
    ASSERT(v != NULL, "vector creation");
    
    float dist = gv_distance_euclidean(NULL, v);
    ASSERT(dist < 0.0f, "euclidean with NULL first vector");
    
    dist = gv_distance_euclidean(v, NULL);
    ASSERT(dist < 0.0f, "euclidean with NULL second vector");
    
    dist = gv_distance_cosine(NULL, v);
    ASSERT(dist < -1.0f, "cosine with NULL first vector");
    
    dist = gv_distance_dot_product(NULL, v);
    ASSERT(dist < 0.0f, "dot product with NULL first vector");
    
    dist = gv_distance_manhattan(NULL, v);
    ASSERT(dist < 0.0f, "manhattan with NULL first vector");
    
    gv_vector_destroy(v);
    return 0;
}

static int test_distance_mismatched_dimensions(void) {
    float v1_data[2] = {1.0f, 2.0f};
    float v2_data[3] = {1.0f, 2.0f, 3.0f};
    
    GV_Vector *v1 = gv_vector_create_from_data(2, v1_data);
    GV_Vector *v2 = gv_vector_create_from_data(3, v2_data);
    ASSERT(v1 != NULL, "v1 creation");
    ASSERT(v2 != NULL, "v2 creation");
    
    float dist = gv_distance_euclidean(v1, v2);
    ASSERT(dist < 0.0f, "euclidean with mismatched dimensions");
    
    gv_vector_destroy(v1);
    gv_vector_destroy(v2);
    return 0;
}

int main(void) {
    int rc = 0;
    printf("Running distance metric tests...\n");
    rc |= test_euclidean_distance();
    rc |= test_cosine_distance();
    rc |= test_dot_product_distance();
    rc |= test_manhattan_distance();
    rc |= test_distance_null_vectors();
    rc |= test_distance_mismatched_dimensions();
    if (rc == 0) {
        printf("All distance tests passed\n");
    }
    return rc;
}

