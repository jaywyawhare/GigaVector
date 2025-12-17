#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#include "gigavector/gv_kdtree.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_metadata.h"
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

static int gv_write_string(FILE *out, const char *str, uint32_t len) {
    if (str == NULL && len > 0) {
        return -1;
    }
    if (gv_write_uint32(out, len) != 0) {
        return -1;
    }
    if (len == 0) {
        return 0;
    }
    return (fwrite(str, 1, len, out) == len) ? 0 : -1;
}

static int gv_write_metadata(FILE *out, const GV_Metadata *meta_head) {
    uint32_t count = 0;
    const GV_Metadata *cursor = meta_head;
    while (cursor != NULL) {
        count++;
        cursor = cursor->next;
    }

    if (gv_write_uint32(out, count) != 0) {
        return -1;
    }

    cursor = meta_head;
    while (cursor != NULL) {
        size_t key_len = strlen(cursor->key);
        size_t val_len = strlen(cursor->value);
        if (key_len > UINT32_MAX || val_len > UINT32_MAX) {
            return -1;
        }
        if (gv_write_string(out, cursor->key, (uint32_t)key_len) != 0) {
            return -1;
        }
        if (gv_write_string(out, cursor->value, (uint32_t)val_len) != 0) {
            return -1;
        }
        cursor = cursor->next;
    }
    return 0;
}

int gv_kdtree_save_recursive(const GV_KDNode *node, FILE *out, uint32_t version) {
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

    if (version >= 2) {
        if (gv_write_metadata(out, node->point->metadata) != 0) {
            return -1;
        }
    }

    if (gv_kdtree_save_recursive(node->left, out, version) != 0) {
        return -1;
    }

    return gv_kdtree_save_recursive(node->right, out, version);
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

static int gv_read_string(FILE *in, char **out_str, uint32_t len) {
    if (out_str == NULL) {
        return -1;
    }
    *out_str = NULL;
    if (len == 0) {
        *out_str = (char *)malloc(1);
        if (*out_str == NULL) {
            return -1;
        }
        (*out_str)[0] = '\0';
        return 0;
    }

    char *buf = (char *)malloc(len + 1);
    if (buf == NULL) {
        return -1;
    }
    if (fread(buf, 1, len, in) != len) {
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    *out_str = buf;
    return 0;
}

static int gv_read_metadata(FILE *in, GV_Vector *vec) {
    if (vec == NULL) {
        return -1;
    }

    uint32_t count = 0;
    if (gv_read_uint32(in, &count) != 0) {
        return -1;
    }

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t key_len = 0;
        uint32_t val_len = 0;
        char *key = NULL;
        char *value = NULL;

        if (gv_read_uint32(in, &key_len) != 0) {
            return -1;
        }
        if (gv_read_string(in, &key, key_len) != 0) {
            free(key);
            return -1;
        }

        if (gv_read_uint32(in, &val_len) != 0) {
            free(key);
            return -1;
        }
        if (gv_read_string(in, &value, val_len) != 0) {
            free(key);
            free(value);
            return -1;
        }

        if (gv_vector_set_metadata(vec, key, value) != 0) {
            free(key);
            free(value);
            return -1;
        }

        free(key);
        free(value);
    }

    return 0;
}

int gv_kdtree_load_recursive(GV_KDNode **root, FILE *in, size_t dimension, uint32_t version) {
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

    if (present != 1) {
        return -1;
    }

    uint32_t axis_u32 = 0;
    if (gv_read_uint32(in, &axis_u32) != 0) {
        return -1;
    }

    if (axis_u32 >= dimension) {
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

    if (version >= 2) {
        if (gv_read_metadata(in, vec) != 0) {
            gv_vector_destroy(vec);
            free(node);
            return -1;
        }
    }

    if (gv_kdtree_load_recursive(&(node->left), in, dimension, version) != 0) {
        gv_kdtree_destroy_recursive(node);
        return -1;
    }

    if (gv_kdtree_load_recursive(&(node->right), in, dimension, version) != 0) {
        gv_kdtree_destroy_recursive(node);
        return -1;
    }

    *root = node;
    return 0;
}

void gv_kdtree_destroy_recursive(GV_KDNode *node) {
    if (node == NULL) {
        return;
    }
    gv_kdtree_destroy_recursive(node->left);
    gv_kdtree_destroy_recursive(node->right);
    gv_vector_destroy(node->point);
    free(node);
}

typedef struct {
    GV_SearchResult *results;
    size_t count;
    size_t capacity;
    float worst_distance;
    GV_DistanceType distance_type;
    const char *filter_key;
    const char *filter_value;
} GV_KNNContext;

static void gv_knn_insert_result(GV_KNNContext *ctx, const GV_Vector *vector, float distance) {
    if (ctx == NULL || vector == NULL) {
        return;
    }

    if (ctx->count < ctx->capacity) {
        ctx->results[ctx->count].vector = vector;
        ctx->results[ctx->count].distance = distance;
        ctx->count++;

        for (size_t i = ctx->count - 1; i > 0; --i) {
            if (ctx->results[i].distance < ctx->results[i - 1].distance) {
                GV_SearchResult temp = ctx->results[i];
                ctx->results[i] = ctx->results[i - 1];
                ctx->results[i - 1] = temp;
            } else {
                break;
            }
        }

        if (ctx->count == ctx->capacity) {
            ctx->worst_distance = ctx->results[ctx->count - 1].distance;
        }
    } else if (distance < ctx->worst_distance) {
        ctx->results[ctx->count - 1].vector = vector;
        ctx->results[ctx->count - 1].distance = distance;

        for (size_t i = ctx->count - 1; i > 0; --i) {
            if (ctx->results[i].distance < ctx->results[i - 1].distance) {
                GV_SearchResult temp = ctx->results[i];
                ctx->results[i] = ctx->results[i - 1];
                ctx->results[i - 1] = temp;
            } else {
                break;
            }
        }

        ctx->worst_distance = ctx->results[ctx->count - 1].distance;
    }
}

static int gv_knn_check_metadata_filter(const GV_Vector *vector, const char *key, const char *value) {
    if (key == NULL || value == NULL) {
        return 1;
    }
    if (vector == NULL) {
        return 0;
    }
    const char *meta_value = gv_vector_get_metadata(vector, key);
    if (meta_value == NULL) {
        return 0;
    }
    return (strcmp(meta_value, value) == 0) ? 1 : 0;
}

static void gv_knn_search_recursive(const GV_KDNode *node, const GV_Vector *query,
                                     GV_KNNContext *ctx) {
    if (node == NULL || query == NULL || ctx == NULL || node->point == NULL) {
        return;
    }

    if (!gv_knn_check_metadata_filter(node->point, ctx->filter_key, ctx->filter_value)) {
        return;
    }

    float dist = gv_distance(node->point, query, ctx->distance_type);
    if (dist < 0.0f && ctx->distance_type != GV_DISTANCE_DOT_PRODUCT) {
        return;
    }

    if (ctx->distance_type == GV_DISTANCE_COSINE) {
        dist = 1.0f - dist;
    }

    gv_knn_insert_result(ctx, node->point, dist);

    size_t axis = node->axis;
    float query_value = query->data[axis];
    float node_value = node->point->data[axis];
    float diff = query_value - node_value;
    float axis_distance = diff * diff;

    const GV_KDNode *near = (query_value < node_value) ? node->left : node->right;
    const GV_KDNode *far = (query_value < node_value) ? node->right : node->left;

    if (near != NULL) {
        gv_knn_search_recursive(near, query, ctx);
    }

    if (far != NULL && (ctx->count < ctx->capacity || axis_distance < ctx->worst_distance)) {
        gv_knn_search_recursive(far, query, ctx);
    }
}

int gv_kdtree_knn_search(const GV_KDNode *root, const GV_Vector *query, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type) {
    if (root == NULL || query == NULL || results == NULL || k == 0) {
        return -1;
    }

    if (query->dimension == 0 || query->data == NULL) {
        return -1;
    }

    GV_KNNContext ctx;
    ctx.results = results;
    ctx.count = 0;
    ctx.capacity = k;
    ctx.worst_distance = FLT_MAX;
    ctx.distance_type = distance_type;
    ctx.filter_key = NULL;
    ctx.filter_value = NULL;

    memset(results, 0, k * sizeof(GV_SearchResult));

    gv_knn_search_recursive(root, query, &ctx);

    return (int)ctx.count;
}

int gv_kdtree_knn_search_filtered(const GV_KDNode *root, const GV_Vector *query, size_t k,
                                   GV_SearchResult *results, GV_DistanceType distance_type,
                                   const char *filter_key, const char *filter_value) {
    if (root == NULL || query == NULL || results == NULL || k == 0) {
        return -1;
    }

    if (query->dimension == 0 || query->data == NULL) {
        return -1;
    }

    GV_KNNContext ctx;
    ctx.results = results;
    ctx.count = 0;
    ctx.capacity = k;
    ctx.worst_distance = FLT_MAX;
    ctx.distance_type = distance_type;
    ctx.filter_key = filter_key;
    ctx.filter_value = filter_value;

    memset(results, 0, k * sizeof(GV_SearchResult));

    gv_knn_search_recursive(root, query, &ctx);

    return (int)ctx.count;
}


typedef struct {
    GV_SearchResult *results;
    size_t count;
    size_t capacity;
    float radius;
    GV_DistanceType distance_type;
    const char *filter_key;
    const char *filter_value;
} GV_RangeContext;

static void gv_range_insert_result(GV_RangeContext *ctx, const GV_Vector *vector, float distance) {
    if (ctx == NULL || vector == NULL || distance > ctx->radius) {
        return;
    }

    if (ctx->count >= ctx->capacity) {
        return;
    }

    ctx->results[ctx->count].vector = vector;
    ctx->results[ctx->count].distance = distance;
    ctx->count++;
}

static void gv_range_search_recursive(const GV_KDNode *node, const GV_Vector *query,
                                      GV_RangeContext *ctx) {
    if (node == NULL || query == NULL || ctx == NULL || node->point == NULL) {
        return;
    }

    if (!gv_knn_check_metadata_filter(node->point, ctx->filter_key, ctx->filter_value)) {
        return;
    }

    float dist = gv_distance(node->point, query, ctx->distance_type);
    if (dist < 0.0f && ctx->distance_type != GV_DISTANCE_DOT_PRODUCT) {
        return;
    }

    if (dist <= ctx->radius) {
        gv_range_insert_result(ctx, node->point, dist);
    }

    size_t axis = node->axis;
    float query_value = query->data[axis];
    float node_value = node->point->data[axis];
    float diff = query_value - node_value;
    float diff_sq = diff * diff;

    if (ctx->distance_type == GV_DISTANCE_EUCLIDEAN) {
        if (diff_sq <= ctx->radius * ctx->radius) {
            if (query_value < node_value) {
                gv_range_search_recursive(node->left, query, ctx);
                gv_range_search_recursive(node->right, query, ctx);
            } else {
                gv_range_search_recursive(node->right, query, ctx);
                gv_range_search_recursive(node->left, query, ctx);
            }
        } else {
            if (query_value < node_value) {
                gv_range_search_recursive(node->left, query, ctx);
            } else {
                gv_range_search_recursive(node->right, query, ctx);
            }
        }
    } else {
        if (query_value < node_value) {
            gv_range_search_recursive(node->left, query, ctx);
            if (diff_sq <= ctx->radius * ctx->radius || ctx->distance_type != GV_DISTANCE_EUCLIDEAN) {
                gv_range_search_recursive(node->right, query, ctx);
            }
        } else {
            gv_range_search_recursive(node->right, query, ctx);
            if (diff_sq <= ctx->radius * ctx->radius || ctx->distance_type != GV_DISTANCE_EUCLIDEAN) {
                gv_range_search_recursive(node->left, query, ctx);
            }
        }
    }
}

int gv_kdtree_range_search(const GV_KDNode *root, const GV_Vector *query, float radius,
                            GV_SearchResult *results, size_t max_results, GV_DistanceType distance_type) {
    if (root == NULL || query == NULL || results == NULL || max_results == 0 || radius < 0.0f) {
        return -1;
    }

    if (query->dimension == 0 || query->data == NULL) {
        return -1;
    }

    GV_RangeContext ctx;
    ctx.results = results;
    ctx.count = 0;
    ctx.capacity = max_results;
    ctx.radius = radius;
    ctx.distance_type = distance_type;
    ctx.filter_key = NULL;
    ctx.filter_value = NULL;

    memset(results, 0, max_results * sizeof(GV_SearchResult));

    gv_range_search_recursive(root, query, &ctx);

    return (int)ctx.count;
}

int gv_kdtree_range_search_filtered(const GV_KDNode *root, const GV_Vector *query, float radius,
                                     GV_SearchResult *results, size_t max_results,
                                     GV_DistanceType distance_type,
                                     const char *filter_key, const char *filter_value) {
    if (root == NULL || query == NULL || results == NULL || max_results == 0 || radius < 0.0f) {
        return -1;
    }

    if (query->dimension == 0 || query->data == NULL) {
        return -1;
    }

    GV_RangeContext ctx;
    ctx.results = results;
    ctx.count = 0;
    ctx.capacity = max_results;
    ctx.radius = radius;
    ctx.distance_type = distance_type;
    ctx.filter_key = filter_key;
    ctx.filter_value = filter_value;

    memset(results, 0, max_results * sizeof(GV_SearchResult));

    gv_range_search_recursive(root, query, &ctx);

    return (int)ctx.count;
}
