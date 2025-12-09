#include <stdint.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "gigavector/gv_kdtree.h"
#include "gigavector/gv_vector.h"

static GV_KDNode *gv_kdtree_create_node(GV_Vector *point, size_t axis) {
    GV_KDNode *node = (GV_KDNode *)malloc(sizeof(GV_KDNode));
    if (node == NULL) {
        return NULL;
    }

    node->point = point;
    node->axis = axis;
    node->left = NULL;
    node->right = NULL;
    return node;
}

int gv_kdtree_insert(GV_KDNode **root, GV_Vector *point, size_t depth) {
    if (root == NULL || point == NULL || point->dimension == 0 || point->data == NULL) {
        return -1;
    }

    if (*root == NULL) {
        size_t axis = depth % point->dimension;
        GV_KDNode *node = gv_kdtree_create_node(point, axis);
        if (node == NULL) {
            return -1;
        }
        *root = node;
        return 0;
    }

    GV_KDNode *current = *root;
    if (current->point == NULL || current->point->dimension != point->dimension) {
        return -1;
    }

    size_t axis = current->axis;
    float point_value = point->data[axis];
    float current_value = current->point->data[axis];

    if (point_value < current_value) {
        return gv_kdtree_insert(&(current->left), point, depth + 1);
    }

    return gv_kdtree_insert(&(current->right), point, depth + 1);
}

static int gv_write_uint8(FILE *out, uint8_t value) {
    return (fwrite(&value, sizeof(uint8_t), 1, out) == 1) ? 0 : -1;
}

static int gv_write_uint32(FILE *out, uint32_t value) {
    return (fwrite(&value, sizeof(uint32_t), 1, out) == 1) ? 0 : -1;
}

static int gv_write_floats(FILE *out, const float *data, size_t count) {
    return (fwrite(data, sizeof(float), count, out) == count) ? 0 : -1;
}

int gv_kdtree_save_recursive(const GV_KDNode *node, FILE *out) {
    if (out == NULL) {
        return -1;
    }

    if (node == NULL) {
        return gv_write_uint8(out, 0);
    }

    if (gv_write_uint8(out, 1) != 0) {
        return -1;
    }

    if (node->point == NULL || node->point->data == NULL) {
        return -1;
    }

    if (node->point->dimension > UINT32_MAX || node->axis > UINT32_MAX) {
        return -1;
    }

    if (gv_write_uint32(out, (uint32_t)node->axis) != 0) {
        return -1;
    }

    if (gv_write_floats(out, node->point->data, node->point->dimension) != 0) {
        return -1;
    }

    if (gv_kdtree_save_recursive(node->left, out) != 0) {
        return -1;
    }

    return gv_kdtree_save_recursive(node->right, out);
}

static int gv_read_uint8(FILE *in, uint8_t *value) {
    return (value != NULL && fread(value, sizeof(uint8_t), 1, in) == 1) ? 0 : -1;
}

static int gv_read_uint32(FILE *in, uint32_t *value) {
    return (value != NULL && fread(value, sizeof(uint32_t), 1, in) == 1) ? 0 : -1;
}

static int gv_read_floats(FILE *in, float *data, size_t count) {
    return (data != NULL && fread(data, sizeof(float), count, in) == count) ? 0 : -1;
}

int gv_kdtree_load_recursive(GV_KDNode **root, FILE *in, size_t dimension) {
    if (root == NULL || in == NULL || dimension == 0) {
        return -1;
    }

    uint8_t present = 0;
    if (gv_read_uint8(in, &present) != 0) {
        return -1;
    }

    if (present == 0) {
        *root = NULL;
        return 0;
    }

    uint32_t axis_u32 = 0;
    if (gv_read_uint32(in, &axis_u32) != 0) {
        return -1;
    }

    GV_Vector *vec = gv_vector_create(dimension);
    if (vec == NULL) {
        return -1;
    }

    if (gv_read_floats(in, vec->data, dimension) != 0) {
        gv_vector_destroy(vec);
        return -1;
    }

    GV_KDNode *node = gv_kdtree_create_node(vec, (size_t)axis_u32);
    if (node == NULL) {
        gv_vector_destroy(vec);
        return -1;
    }

    if (gv_kdtree_load_recursive(&(node->left), in, dimension) != 0 ||
        gv_kdtree_load_recursive(&(node->right), in, dimension) != 0) {
        gv_vector_destroy(node->point);
        free(node);
        return -1;
    }

    *root = node;
    return 0;
}

