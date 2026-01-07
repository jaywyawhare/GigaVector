#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#include "gigavector/gv_kdtree.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_metadata.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_soa_storage.h"
#include <math.h>

static GV_KDNode *gv_kdtree_create_node(size_t vector_index, size_t axis) {
    GV_KDNode *node = (GV_KDNode *)malloc(sizeof(GV_KDNode));
    if (node == NULL) {
        return NULL;
    }

    node->vector_index = vector_index;
    node->axis = axis;
    node->left = NULL;
    node->right = NULL;
    return node;
}

int gv_kdtree_insert(GV_KDNode **root, GV_SoAStorage *storage, size_t vector_index, size_t depth) {
    if (root == NULL || storage == NULL || vector_index >= storage->count) {
        return -1;
    }

    if (*root == NULL) {
        size_t axis = depth % storage->dimension;
        GV_KDNode *node = gv_kdtree_create_node(vector_index, axis);
        if (node == NULL) {
            return -1;
        }
        *root = node;
        return 0;
    }

    GV_KDNode *current = *root;
    const float *current_data = gv_soa_storage_get_data(storage, current->vector_index);
    const float *point_data = gv_soa_storage_get_data(storage, vector_index);
    if (current_data == NULL || point_data == NULL) {
        return -1;
    }

    size_t axis = current->axis;
    float point_value = point_data[axis];
    float current_value = current_data[axis];

    if (point_value < current_value) {
        return gv_kdtree_insert(&(current->left), storage, vector_index, depth + 1);
    }

    return gv_kdtree_insert(&(current->right), storage, vector_index, depth + 1);
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

int gv_kdtree_save_recursive(const GV_KDNode *node, const GV_SoAStorage *storage, FILE *out, uint32_t version) {
    if (out == NULL || storage == NULL) {
        return -1;
    }

    if (node == NULL) {
        return gv_write_uint8(out, 0);
    }

    if (gv_write_uint8(out, 1) != 0) {
        return -1;
    }

    if (node->vector_index >= storage->count) {
        return -1;
    }

    const float *data = gv_soa_storage_get_data(storage, node->vector_index);
    if (data == NULL) {
        return -1;
    }

    if (storage->dimension > UINT32_MAX || node->axis > UINT32_MAX) {
        return -1;
    }

    if (gv_write_uint32(out, (uint32_t)node->axis) != 0) {
        return -1;
    }

    if (gv_write_floats(out, data, storage->dimension) != 0) {
        return -1;
    }

    if (version >= 2) {
        GV_Metadata *metadata = gv_soa_storage_get_metadata(storage, node->vector_index);
        if (gv_write_metadata(out, metadata) != 0) {
            return -1;
        }
    }

    if (gv_kdtree_save_recursive(node->left, storage, out, version) != 0) {
        return -1;
    }

    return gv_kdtree_save_recursive(node->right, storage, out, version);
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

int gv_kdtree_load_recursive(GV_KDNode **root, GV_SoAStorage *storage, FILE *in, size_t dimension, uint32_t version) {
    if (root == NULL || in == NULL || dimension == 0 || storage == NULL) {
        return -1;
    }

    if (storage->dimension != dimension) {
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

    float *temp_data = (float *)malloc(dimension * sizeof(float));
    if (temp_data == NULL) {
        return -1;
    }

    if (gv_read_floats(in, temp_data, dimension) != 0) {
        free(temp_data);
        return -1;
    }

    GV_Metadata *metadata = NULL;
    if (version >= 2) {
        GV_Vector temp_vec;
        temp_vec.dimension = dimension;
        temp_vec.data = NULL;
        temp_vec.metadata = NULL;
        if (gv_read_metadata(in, &temp_vec) != 0) {
            free(temp_data);
            return -1;
        }
        metadata = temp_vec.metadata;
    }

    size_t vector_index = gv_soa_storage_add(storage, temp_data, metadata);
    free(temp_data);
    if (vector_index == (size_t)-1) {
        if (metadata != NULL) {
            GV_Vector temp_vec;
            temp_vec.dimension = dimension;
            temp_vec.data = NULL;
            temp_vec.metadata = metadata;
            gv_vector_clear_metadata(&temp_vec);
        }
        return -1;
    }

    GV_KDNode *node = gv_kdtree_create_node(vector_index, (size_t)axis_u32);
    if (node == NULL) {
        return -1;
    }

    if (gv_kdtree_load_recursive(&(node->left), storage, in, dimension, version) != 0) {
        gv_kdtree_destroy_recursive(node);
        return -1;
    }

    if (gv_kdtree_load_recursive(&(node->right), storage, in, dimension, version) != 0) {
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

typedef struct {
    GV_SoAStorage *storage;
    GV_Vector *temp_views;
    size_t temp_views_capacity;
} GV_KNNStorageContext;

static GV_Vector *gv_knn_get_vector_view(GV_KNNStorageContext *storage_ctx, size_t index) {
    if (storage_ctx == NULL || storage_ctx->storage == NULL) {
        return NULL;
    }
    if (index >= storage_ctx->temp_views_capacity) {
        size_t new_capacity = (storage_ctx->temp_views_capacity == 0) ? 16 : storage_ctx->temp_views_capacity * 2;
        while (new_capacity <= index) {
            new_capacity *= 2;
        }
        GV_Vector *new_views = (GV_Vector *)realloc(storage_ctx->temp_views, new_capacity * sizeof(GV_Vector));
        if (new_views == NULL) {
            return NULL;
        }
        storage_ctx->temp_views = new_views;
        storage_ctx->temp_views_capacity = new_capacity;
    }
    if (gv_soa_storage_get_vector_view(storage_ctx->storage, index, &storage_ctx->temp_views[index]) != 0) {
        return NULL;
    }
    return &storage_ctx->temp_views[index];
}

static void gv_knn_insert_result(GV_KNNContext *ctx, GV_KNNStorageContext *storage_ctx, size_t vector_index, float distance) {
    if (ctx == NULL || storage_ctx == NULL) {
        return;
    }

    GV_Vector *vector = gv_knn_get_vector_view(storage_ctx, vector_index);
    if (vector == NULL) {
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

static int gv_knn_check_metadata_filter(const GV_SoAStorage *storage, size_t vector_index, const char *key, const char *value) {
    if (key == NULL || value == NULL) {
        return 1;
    }
    if (storage == NULL) {
        return 0;
    }
    GV_Metadata *metadata = gv_soa_storage_get_metadata(storage, vector_index);
    if (metadata == NULL) {
        return 0;
    }
    GV_Vector temp_vec;
    temp_vec.dimension = storage->dimension;
    temp_vec.data = NULL;
    temp_vec.metadata = metadata;
    const char *meta_value = gv_vector_get_metadata(&temp_vec, key);
    if (meta_value == NULL) {
        return 0;
    }
    return (strcmp(meta_value, value) == 0) ? 1 : 0;
}

static void gv_knn_search_recursive(const GV_KDNode *node, const GV_SoAStorage *storage, const GV_Vector *query,
                                     GV_KNNContext *ctx, GV_KNNStorageContext *storage_ctx) {
    if (node == NULL || query == NULL || ctx == NULL || storage == NULL || storage_ctx == NULL) {
        return;
    }

    if (gv_soa_storage_is_deleted(storage, node->vector_index) == 1) {
        return;
    }

    if (!gv_knn_check_metadata_filter(storage, node->vector_index, ctx->filter_key, ctx->filter_value)) {
        return;
    }

    const float *node_data = gv_soa_storage_get_data(storage, node->vector_index);
    if (node_data == NULL) {
        return;
    }

    GV_Vector node_vec;
    node_vec.dimension = storage->dimension;
    node_vec.data = (float *)node_data;
    node_vec.metadata = gv_soa_storage_get_metadata(storage, node->vector_index);

    float dist = gv_distance(&node_vec, query, ctx->distance_type);
    if (dist < 0.0f && ctx->distance_type != GV_DISTANCE_DOT_PRODUCT) {
        return;
    }

    if (ctx->distance_type == GV_DISTANCE_COSINE) {
        dist = 1.0f - dist;
    }

    gv_knn_insert_result(ctx, storage_ctx, node->vector_index, dist);

    size_t axis = node->axis;
    float query_value = query->data[axis];
    float node_value = node_data[axis];
    float diff = query_value - node_value;
    float axis_distance = diff * diff;

    const GV_KDNode *near = (query_value < node_value) ? node->left : node->right;
    const GV_KDNode *far = (query_value < node_value) ? node->right : node->left;

    if (near != NULL) {
        gv_knn_search_recursive(near, storage, query, ctx, storage_ctx);
    }

    if (far != NULL && (ctx->count < ctx->capacity || axis_distance < ctx->worst_distance)) {
        gv_knn_search_recursive(far, storage, query, ctx, storage_ctx);
    }
}

int gv_kdtree_knn_search(const GV_KDNode *root, const GV_SoAStorage *storage, const GV_Vector *query, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type) {
    if (root == NULL || storage == NULL || query == NULL || results == NULL || k == 0) {
        return -1;
    }

    if (query->dimension == 0 || query->data == NULL || query->dimension != storage->dimension) {
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

    GV_KNNStorageContext storage_ctx;
    storage_ctx.storage = (GV_SoAStorage *)storage;
    storage_ctx.temp_views = NULL;
    storage_ctx.temp_views_capacity = 0;

    memset(results, 0, k * sizeof(GV_SearchResult));

    gv_knn_search_recursive(root, storage, query, &ctx, &storage_ctx);

    /* Copy vector data from views to ensure results remain valid after temp_views is freed */
    for (size_t i = 0; i < ctx.count; ++i) {
        if (ctx.results[i].vector != NULL && !ctx.results[i].is_sparse) {
            const GV_Vector *view = ctx.results[i].vector;
            GV_Vector *copy = gv_vector_create_from_data(view->dimension, view->data);
            if (copy != NULL && view->metadata != NULL) {
                /* Copy metadata chain */
                GV_Metadata *src = view->metadata;
                GV_Metadata *dst_head = NULL;
                GV_Metadata *dst_tail = NULL;
                while (src != NULL) {
                    GV_Metadata *new_meta = (GV_Metadata *)malloc(sizeof(GV_Metadata));
                    if (new_meta == NULL) {
                        /* Free what we've copied so far */
                        while (dst_head != NULL) {
                            GV_Metadata *next = dst_head->next;
                            free(dst_head->key);
                            free(dst_head->value);
                            free(dst_head);
                            dst_head = next;
                        }
                        gv_vector_destroy(copy);
                        copy = NULL;
                        break;
                    }
                    new_meta->key = src->key ? strdup(src->key) : NULL;
                    new_meta->value = src->value ? strdup(src->value) : NULL;
                    new_meta->next = NULL;
                    if (dst_head == NULL) {
                        dst_head = dst_tail = new_meta;
                    } else {
                        dst_tail->next = new_meta;
                        dst_tail = new_meta;
                    }
                    src = src->next;
                }
                copy->metadata = dst_head;
            }
            ctx.results[i].vector = copy;
        }
    }

    free(storage_ctx.temp_views);
    return (int)ctx.count;
}

int gv_kdtree_knn_search_filtered(const GV_KDNode *root, const GV_SoAStorage *storage, const GV_Vector *query, size_t k,
                                   GV_SearchResult *results, GV_DistanceType distance_type,
                                   const char *filter_key, const char *filter_value) {
    if (root == NULL || storage == NULL || query == NULL || results == NULL || k == 0) {
        return -1;
    }

    if (query->dimension == 0 || query->data == NULL || query->dimension != storage->dimension) {
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

    GV_KNNStorageContext storage_ctx;
    storage_ctx.storage = (GV_SoAStorage *)storage;
    storage_ctx.temp_views = NULL;
    storage_ctx.temp_views_capacity = 0;

    memset(results, 0, k * sizeof(GV_SearchResult));

    gv_knn_search_recursive(root, storage, query, &ctx, &storage_ctx);

    /* Copy vector data from views to ensure results remain valid after temp_views is freed */
    for (size_t i = 0; i < ctx.count; ++i) {
        if (ctx.results[i].vector != NULL && !ctx.results[i].is_sparse) {
            const GV_Vector *view = ctx.results[i].vector;
            GV_Vector *copy = gv_vector_create_from_data(view->dimension, view->data);
            if (copy != NULL && view->metadata != NULL) {
                /* Copy metadata chain */
                GV_Metadata *src = view->metadata;
                GV_Metadata *dst_head = NULL;
                GV_Metadata *dst_tail = NULL;
                while (src != NULL) {
                    GV_Metadata *new_meta = (GV_Metadata *)malloc(sizeof(GV_Metadata));
                    if (new_meta == NULL) {
                        while (dst_head != NULL) {
                            GV_Metadata *next = dst_head->next;
                            free(dst_head->key);
                            free(dst_head->value);
                            free(dst_head);
                            dst_head = next;
                        }
                        gv_vector_destroy(copy);
                        copy = NULL;
                        break;
                    }
                    new_meta->key = src->key ? strdup(src->key) : NULL;
                    new_meta->value = src->value ? strdup(src->value) : NULL;
                    new_meta->next = NULL;
                    if (dst_head == NULL) {
                        dst_head = dst_tail = new_meta;
                    } else {
                        dst_tail->next = new_meta;
                        dst_tail = new_meta;
                    }
                    src = src->next;
                }
                copy->metadata = dst_head;
            }
            ctx.results[i].vector = copy;
        }
    }

    free(storage_ctx.temp_views);
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

static void gv_range_insert_result(GV_RangeContext *ctx, GV_KNNStorageContext *storage_ctx, size_t vector_index, float distance) {
    if (ctx == NULL || storage_ctx == NULL || distance > ctx->radius) {
        return;
    }

    if (ctx->count >= ctx->capacity) {
        return;
    }

    GV_Vector *vector = gv_knn_get_vector_view(storage_ctx, vector_index);
    if (vector == NULL) {
        return;
    }

    ctx->results[ctx->count].vector = vector;
    ctx->results[ctx->count].distance = distance;
    ctx->count++;
}

static void gv_range_search_recursive(const GV_KDNode *node, const GV_SoAStorage *storage, const GV_Vector *query,
                                      GV_RangeContext *ctx, GV_KNNStorageContext *storage_ctx) {
    if (node == NULL || query == NULL || ctx == NULL || storage == NULL || storage_ctx == NULL) {
        return;
    }

    if (gv_soa_storage_is_deleted(storage, node->vector_index) == 1) {
        return;
    }

    if (!gv_knn_check_metadata_filter(storage, node->vector_index, ctx->filter_key, ctx->filter_value)) {
        return;
    }

    const float *node_data = gv_soa_storage_get_data(storage, node->vector_index);
    if (node_data == NULL) {
        return;
    }

    GV_Vector node_vec;
    node_vec.dimension = storage->dimension;
    node_vec.data = (float *)node_data;
    node_vec.metadata = gv_soa_storage_get_metadata(storage, node->vector_index);

    float dist = gv_distance(&node_vec, query, ctx->distance_type);
    if (dist < 0.0f && ctx->distance_type != GV_DISTANCE_DOT_PRODUCT) {
        return;
    }

    if (dist <= ctx->radius) {
        gv_range_insert_result(ctx, storage_ctx, node->vector_index, dist);
    }

    size_t axis = node->axis;
    float query_value = query->data[axis];
    float node_value = node_data[axis];
    float diff = query_value - node_value;
    float diff_sq = diff * diff;

    if (ctx->distance_type == GV_DISTANCE_EUCLIDEAN) {
        if (diff_sq <= ctx->radius * ctx->radius) {
            if (query_value < node_value) {
                gv_range_search_recursive(node->left, storage, query, ctx, storage_ctx);
                gv_range_search_recursive(node->right, storage, query, ctx, storage_ctx);
            } else {
                gv_range_search_recursive(node->right, storage, query, ctx, storage_ctx);
                gv_range_search_recursive(node->left, storage, query, ctx, storage_ctx);
            }
        } else {
            if (query_value < node_value) {
                gv_range_search_recursive(node->left, storage, query, ctx, storage_ctx);
            } else {
                gv_range_search_recursive(node->right, storage, query, ctx, storage_ctx);
            }
        }
    } else {
        if (query_value < node_value) {
            gv_range_search_recursive(node->left, storage, query, ctx, storage_ctx);
            if (diff_sq <= ctx->radius * ctx->radius || ctx->distance_type != GV_DISTANCE_EUCLIDEAN) {
                gv_range_search_recursive(node->right, storage, query, ctx, storage_ctx);
            }
        } else {
            gv_range_search_recursive(node->right, storage, query, ctx, storage_ctx);
            if (diff_sq <= ctx->radius * ctx->radius || ctx->distance_type != GV_DISTANCE_EUCLIDEAN) {
                gv_range_search_recursive(node->left, storage, query, ctx, storage_ctx);
            }
        }
    }
}

int gv_kdtree_range_search(const GV_KDNode *root, const GV_SoAStorage *storage, const GV_Vector *query, float radius,
                            GV_SearchResult *results, size_t max_results, GV_DistanceType distance_type) {
    if (root == NULL || storage == NULL || query == NULL || results == NULL || max_results == 0 || radius < 0.0f) {
        return -1;
    }

    if (query->dimension == 0 || query->data == NULL || query->dimension != storage->dimension) {
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

    GV_KNNStorageContext storage_ctx;
    storage_ctx.storage = (GV_SoAStorage *)storage;
    storage_ctx.temp_views = NULL;
    storage_ctx.temp_views_capacity = 0;

    memset(results, 0, max_results * sizeof(GV_SearchResult));

    gv_range_search_recursive(root, storage, query, &ctx, &storage_ctx);

    free(storage_ctx.temp_views);
    return (int)ctx.count;
}

int gv_kdtree_range_search_filtered(const GV_KDNode *root, const GV_SoAStorage *storage, const GV_Vector *query, float radius,
                                     GV_SearchResult *results, size_t max_results,
                                     GV_DistanceType distance_type,
                                     const char *filter_key, const char *filter_value) {
    if (root == NULL || storage == NULL || query == NULL || results == NULL || max_results == 0 || radius < 0.0f) {
        return -1;
    }

    if (query->dimension == 0 || query->data == NULL || query->dimension != storage->dimension) {
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

    GV_KNNStorageContext storage_ctx;
    storage_ctx.storage = (GV_SoAStorage *)storage;
    storage_ctx.temp_views = NULL;
    storage_ctx.temp_views_capacity = 0;

    memset(results, 0, max_results * sizeof(GV_SearchResult));

    gv_range_search_recursive(root, storage, query, &ctx, &storage_ctx);

    free(storage_ctx.temp_views);
    return (int)ctx.count;
}

static GV_KDNode *gv_kdtree_find_min(GV_KDNode *node, size_t axis, size_t target_axis, const GV_SoAStorage *storage) {
    if (node == NULL) {
        return NULL;
    }

    if (node->axis == target_axis) {
        if (node->left == NULL) {
            return node;
        }
        return gv_kdtree_find_min(node->left, (axis + 1) % storage->dimension, target_axis, storage);
    }

    GV_KDNode *best = node;
    const float *best_data = gv_soa_storage_get_data(storage, node->vector_index);
    if (best_data == NULL) {
        return NULL;
    }

    GV_KDNode *left_min = gv_kdtree_find_min(node->left, (axis + 1) % storage->dimension, target_axis, storage);
    if (left_min != NULL) {
        const float *left_data = gv_soa_storage_get_data(storage, left_min->vector_index);
        if (left_data != NULL && left_data[target_axis] < best_data[target_axis]) {
            best = left_min;
        }
    }

    GV_KDNode *right_min = gv_kdtree_find_min(node->right, (axis + 1) % storage->dimension, target_axis, storage);
    if (right_min != NULL) {
        const float *right_data = gv_soa_storage_get_data(storage, right_min->vector_index);
        if (right_data != NULL) {
            const float *best_data_check = gv_soa_storage_get_data(storage, best->vector_index);
            if (best_data_check != NULL && right_data[target_axis] < best_data_check[target_axis]) {
                best = right_min;
            }
        }
    }

    return best;
}

static GV_KDNode *gv_kdtree_delete_recursive(GV_KDNode *node, size_t vector_index, size_t depth, 
                                               const GV_SoAStorage *storage) {
    if (node == NULL) {
        return NULL;
    }

    size_t axis = depth % storage->dimension;

    if (node->vector_index == vector_index) {
        if (node->right != NULL) {
            GV_KDNode *min_node = gv_kdtree_find_min(node->right, (axis + 1) % storage->dimension, axis, storage);
            if (min_node != NULL) {
                node->vector_index = min_node->vector_index;
                node->right = gv_kdtree_delete_recursive(node->right, min_node->vector_index, depth + 1, storage);
            } else {
                GV_KDNode *temp = node->left;
                free(node);
                return temp;
            }
        } else if (node->left != NULL) {
            GV_KDNode *min_node = gv_kdtree_find_min(node->left, (axis + 1) % storage->dimension, axis, storage);
            if (min_node != NULL) {
                node->vector_index = min_node->vector_index;
                node->right = node->left;
                node->left = NULL;
                node->right = gv_kdtree_delete_recursive(node->right, min_node->vector_index, depth + 1, storage);
            } else {
                GV_KDNode *temp = node->left;
                free(node);
                return temp;
            }
        } else {
            free(node);
            return NULL;
        }
    } else {
        const float *node_data = gv_soa_storage_get_data(storage, node->vector_index);
        const float *target_data = gv_soa_storage_get_data(storage, vector_index);
        if (node_data == NULL || target_data == NULL) {
            return node;
        }

        if (target_data[axis] < node_data[axis]) {
            node->left = gv_kdtree_delete_recursive(node->left, vector_index, depth + 1, storage);
        } else {
            node->right = gv_kdtree_delete_recursive(node->right, vector_index, depth + 1, storage);
        }
    }

    return node;
}

int gv_kdtree_delete(GV_KDNode **root, GV_SoAStorage *storage, size_t vector_index) {
    if (root == NULL || storage == NULL || vector_index >= storage->count) {
        return -1;
    }

    if (*root == NULL) {
        return -1;
    }

    if (gv_soa_storage_is_deleted(storage, vector_index) == 1) {
        return -1;
    }

    *root = gv_kdtree_delete_recursive(*root, vector_index, 0, storage);
    if (gv_soa_storage_mark_deleted(storage, vector_index) != 0) {
        return -1;
    }

    return 0;
}

int gv_kdtree_update(GV_KDNode **root, GV_SoAStorage *storage, size_t vector_index, const float *new_data) {
    if (root == NULL || storage == NULL || new_data == NULL || vector_index >= storage->count) {
        return -1;
    }

    if (gv_soa_storage_is_deleted(storage, vector_index) == 1) {
        return -1;
    }

    /* Delete the old node from the tree */
    int delete_status = gv_kdtree_delete(root, storage, vector_index);
    if (delete_status != 0) {
        return -1;
    }

    /* Update the data in SoA storage */
    if (gv_soa_storage_update_data(storage, vector_index, new_data) != 0) {
        return -1;
    }

    /* Reinsert the updated vector into the tree */
    int insert_status = gv_kdtree_insert(root, storage, vector_index, 0);
    if (insert_status != 0) {
        return -1;
    }

    return 0;
}
