#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_exact_search.h"
#include "gigavector/gv_distance.h"

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
            results[filled].vector = v;
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
            results[k - 1].vector = v;
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

static void gv_exact_collect_kdtree(const GV_KDNode *node, GV_Vector **out,
                                    size_t max_count, size_t *count) {
    if (node == NULL || out == NULL || count == NULL) {
        return;
    }
    if (*count >= max_count) {
        return;
    }
    if (node->point != NULL) {
        out[*count] = node->point;
        (*count)++;
        if (*count >= max_count) {
            return;
        }
    }
    gv_exact_collect_kdtree(node->left, out, max_count, count);
    gv_exact_collect_kdtree(node->right, out, max_count, count);
}

int gv_exact_knn_search_kdtree(const GV_KDNode *root, size_t total_count,
                               const GV_Vector *query, size_t k,
                               GV_SearchResult *results, GV_DistanceType distance_type) {
    if (query == NULL || results == NULL || k == 0) {
        return -1;
    }
    if (query->dimension == 0 || query->data == NULL) {
        return -1;
    }
    if (root == NULL || total_count == 0) {
        return 0;
    }

    GV_Vector **vecs = (GV_Vector **)malloc(total_count * sizeof(GV_Vector *));
    if (!vecs) {
        return -1;
    }
    size_t collected = 0;
    gv_exact_collect_kdtree(root, vecs, total_count, &collected);

    int r = gv_exact_knn_search_vectors(vecs, collected, query, k, results, distance_type);
    free(vecs);
    return r;
}


