#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_exact_search.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_soa_storage.h"
#include "gigavector/gv_vector.h"

int gv_exact_knn_search_vectors(GV_Vector *const *vectors, size_t count,
                                const GV_Vector *query, size_t k,
                                GV_SearchResult *results, GV_DistanceType distance_type) {
    if (vectors == NULL && count > 0) {
        return -1;
    }
    if (query == NULL || results == NULL || k == 0) {
        return -1;
    }
    if (query->dimension == 0 || query->data == NULL) {
        return -1;
    }
    if (count == 0) {
        return 0;
    }

    if (k > count) {
        k = count;
    }

    for (size_t i = 0; i < k; ++i) {
        results[i].vector = NULL;
        results[i].sparse_vector = NULL;
        results[i].is_sparse = 0;
        results[i].distance = FLT_MAX;
    }

    size_t filled = 0;
    for (size_t i = 0; i < count; ++i) {
        GV_Vector *v = vectors[i];
        if (v == NULL || v->data == NULL || v->dimension != query->dimension) {
            continue;
        }
        float dist = gv_distance(v, query, distance_type);
        if (dist < 0.0f && distance_type != GV_DISTANCE_DOT_PRODUCT) {
            continue;
        }

        if (filled < k) {
            GV_Vector *copy = gv_vector_create_from_data(v->dimension, v->data);
            if (copy != NULL && v->metadata != NULL) {
                // Deep copy metadata
                GV_Metadata *src = v->metadata;
                GV_Metadata **dst = &copy->metadata;
                while (src != NULL) {
                    GV_Metadata *new_meta = (GV_Metadata *)malloc(sizeof(GV_Metadata));
                    if (new_meta == NULL) break;
                    new_meta->key = strdup(src->key);
                    if (new_meta->key == NULL) {
                        free(new_meta);
                        break;
                    }
                    new_meta->value = strdup(src->value);
                    if (new_meta->value == NULL) {
                        free(new_meta->key);
                        free(new_meta);
                        break;
                    }
                    new_meta->next = NULL;
                    *dst = new_meta;
                    dst = &new_meta->next;
                    src = src->next;
                }
            }
            results[filled].vector = copy ? copy : v;
            results[filled].sparse_vector = NULL;
            results[filled].is_sparse = 0;
            results[filled].distance = dist;
            ++filled;
            for (size_t j = filled; j > 0 && j > 1; --j) {
                if (results[j - 1].distance < results[j - 2].distance) {
                    GV_SearchResult tmp = results[j - 1];
                    results[j - 1] = results[j - 2];
                    results[j - 2] = tmp;
                } else {
                    break;
                }
            }
        } else if (dist < results[k - 1].distance) {
            GV_Vector *copy = gv_vector_create_from_data(v->dimension, v->data);
            if (copy != NULL && v->metadata != NULL) {
                // Deep copy metadata
                GV_Metadata *src = v->metadata;
                GV_Metadata **dst = &copy->metadata;
                while (src != NULL) {
                    GV_Metadata *new_meta = (GV_Metadata *)malloc(sizeof(GV_Metadata));
                    if (new_meta == NULL) break;
                    new_meta->key = strdup(src->key);
                    if (new_meta->key == NULL) {
                        free(new_meta);
                        break;
                    }
                    new_meta->value = strdup(src->value);
                    if (new_meta->value == NULL) {
                        free(new_meta->key);
                        free(new_meta);
                        break;
                    }
                    new_meta->next = NULL;
                    *dst = new_meta;
                    dst = &new_meta->next;
                    src = src->next;
                }
            }
            results[k - 1].vector = copy ? copy : v;
            results[k - 1].sparse_vector = NULL;
            results[k - 1].is_sparse = 0;
            results[k - 1].distance = dist;
            for (size_t j = k; j > 0 && j > 1; --j) {
                if (results[j - 1].distance < results[j - 2].distance) {
                    GV_SearchResult tmp = results[j - 1];
                    results[j - 1] = results[j - 2];
                    results[j - 2] = tmp;
                } else {
                    break;
                }
            }
        }
    }
    return (int)filled;
}

static void gv_exact_collect_kdtree(const GV_KDNode *node, const GV_SoAStorage *storage,
                                    GV_Vector *out_views, size_t max_count, size_t *count) {
    if (node == NULL || storage == NULL || out_views == NULL || count == NULL) {
        return;
    }
    if (*count >= max_count) {
        return;
    }
    if (node->vector_index < storage->count) {
        if (gv_soa_storage_get_vector_view(storage, node->vector_index, &out_views[*count]) == 0) {
            (*count)++;
            if (*count >= max_count) {
                return;
            }
        }
    }
    gv_exact_collect_kdtree(node->left, storage, out_views, max_count, count);
    gv_exact_collect_kdtree(node->right, storage, out_views, max_count, count);
}

int gv_exact_knn_search_kdtree(const GV_KDNode *root, const GV_SoAStorage *storage, size_t total_count,
                               const GV_Vector *query, size_t k,
                               GV_SearchResult *results, GV_DistanceType distance_type) {
    if (query == NULL || results == NULL || k == 0 || storage == NULL) {
        return -1;
    }
    if (query->dimension == 0 || query->data == NULL || query->dimension != storage->dimension) {
        return -1;
    }
    if (total_count == 0) {
        return 0;
    }

    GV_Vector *vec_views = NULL;
    GV_Vector **vec_ptrs = (GV_Vector **)malloc(total_count * sizeof(GV_Vector *));
    if (!vec_ptrs) {
        return -1;
    }
    size_t collected = 0;
    if (root == NULL) {
        // For in-memory databases, collect all vectors from SOA storage
        for (size_t i = 0; i < total_count; i++) {
            GV_Vector *vec = gv_vector_create(storage->dimension);
            if (vec == NULL) {
                for (size_t j = 0; j < i; j++) {
                    gv_vector_destroy(vec_ptrs[j]);
                }
                free(vec_ptrs);
                return -1;
            }
            vec->data = (float *)gv_soa_storage_get_data(storage, i);
            vec->metadata = gv_soa_storage_get_metadata(storage, i);
            vec_ptrs[i] = vec;
        }
        collected = total_count;
    } else {
        vec_views = (GV_Vector *)malloc(total_count * sizeof(GV_Vector));
        if (!vec_views) {
            free(vec_ptrs);
            return -1;
        }
        gv_exact_collect_kdtree(root, storage, vec_views, total_count, &collected);
        for (size_t i = 0; i < collected; i++) {
            vec_ptrs[i] = &vec_views[i];
        }
    }

    int r = gv_exact_knn_search_vectors(vec_ptrs, collected, query, k, results, distance_type);
    if (root != NULL) {
        free(vec_views);
    }
    // For root == NULL, don't free vec_ptrs[i] as they are used by results
    free(vec_ptrs);
    return r;
}



