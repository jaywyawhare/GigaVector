#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_vector.h"

static char *gv_db_strdup(const char *src) {
    if (src == NULL) {
        return NULL;
    }
    size_t len = strlen(src) + 1;
    char *copy = (char *)malloc(len);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, src, len);
    return copy;
}

static void gv_kdtree_destroy_recursive(GV_KDNode *node) {
    if (node == NULL) {
        return;
    }
    gv_kdtree_destroy_recursive(node->left);
    gv_kdtree_destroy_recursive(node->right);
    gv_vector_destroy(node->point);
    free(node);
}

GV_Database *gv_db_open(const char *filepath, size_t dimension) {
    if (dimension == 0) {
        return NULL;
    }

    GV_Database *db = (GV_Database *)malloc(sizeof(GV_Database));
    if (db == NULL) {
        return NULL;
    }

    db->dimension = dimension;
    db->root = NULL;
    db->filepath = gv_db_strdup(filepath);
    db->count = 0;

    if (filepath != NULL && db->filepath == NULL) {
        free(db);
        return NULL;
    }

    return db;
}

void gv_db_close(GV_Database *db) {
    if (db == NULL) {
        return;
    }

    gv_kdtree_destroy_recursive(db->root);
    free(db->filepath);
    free(db);
}

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension) {
    if (db == NULL || data == NULL || dimension == 0 || dimension != db->dimension) {
        return -1;
    }

    GV_Vector *vector = gv_vector_create_from_data(dimension, data);
    if (vector == NULL) {
        return -1;
    }

    if (gv_kdtree_insert(&(db->root), vector, 0) != 0) {
        gv_vector_destroy(vector);
        return -1;
    }

    db->count += 1;
    return 0;
}

