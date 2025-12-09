#include <stdlib.h>

#include "gigavector/gv_kdtree.h"

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

