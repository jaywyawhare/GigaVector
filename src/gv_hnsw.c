#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <limits.h>

#include "gigavector/gv_hnsw.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_metadata.h"
#include "gigavector/gv_vector.h"
#include "gigavector/gv_binary_quant.h"

typedef struct GV_HNSWNode {
    GV_Vector *vector;
    GV_BinaryVector *binary_vector;  /**< Binary quantized version for fast search */
    struct GV_HNSWNode ***neighbors;
    size_t *neighbor_counts;
    size_t level;
} GV_HNSWNode;

typedef struct {
    GV_HNSWNode *node;
    float distance;
} GV_HNSWCandidate;

typedef struct {
    size_t dimension;
    size_t M;
    size_t efConstruction;
    size_t efSearch;
    size_t maxLevel;
    int use_binary_quant;
    size_t quant_rerank;
    GV_HNSWNode *entryPoint;
    size_t count;
    GV_HNSWNode **nodes;
    size_t nodes_capacity;
} GV_HNSWIndex;

static double random_level(void) {
    return -log((double)rand() / (RAND_MAX + 1.0)) * 1.4426950408889634;
}

static size_t calculate_level(size_t maxLevel) {
    size_t level = 0;
    double r = random_level();
    while (r < 1.0 && level < maxLevel) {
        level++;
        r = random_level();
    }
    return level;
}

static int compare_candidates(const void *a, const void *b) {
    const GV_HNSWCandidate *ca = (const GV_HNSWCandidate *)a;
    const GV_HNSWCandidate *cb = (const GV_HNSWCandidate *)b;
    if (ca->distance < cb->distance) return -1;
    if (ca->distance > cb->distance) return 1;
    return 0;
}

static void gv_hnsw_node_destroy(GV_HNSWNode *node) {
    if (node == NULL) return;
    if (node->neighbors) {
        for (size_t l = 0; l <= node->level; ++l) {
            free(node->neighbors[l]);
        }
        free(node->neighbors);
    }
    free(node->neighbor_counts);
    gv_vector_destroy(node->vector);
    if (node->binary_vector != NULL) {
        gv_binary_vector_destroy(node->binary_vector);
    }
    free(node);
}

void *gv_hnsw_create(size_t dimension, const GV_HNSWConfig *config) {
    if (dimension == 0) {
        return NULL;
    }

    GV_HNSWIndex *index = (GV_HNSWIndex *)malloc(sizeof(GV_HNSWIndex));
    if (index == NULL) {
        return NULL;
    }

    index->dimension = dimension;
    index->M = (config && config->M > 0) ? config->M : 16;
    index->efConstruction = (config && config->efConstruction > 0) ? config->efConstruction : 200;
    index->efSearch = (config && config->efSearch > 0) ? config->efSearch : 50;
    index->maxLevel = (config && config->maxLevel > 0) ? config->maxLevel : 16;
    index->use_binary_quant = (config && config->use_binary_quant) ? 1 : 0;
    index->quant_rerank = (config && config->quant_rerank > 0) ? config->quant_rerank : 0;
    index->entryPoint = NULL;
    index->count = 0;
    index->nodes_capacity = 1024;
    index->nodes = (GV_HNSWNode **)malloc(index->nodes_capacity * sizeof(GV_HNSWNode *));
    if (index->nodes == NULL) {
        free(index);
        return NULL;
    }

    return index;
}

int gv_hnsw_insert(void *index_ptr, GV_Vector *vector) {
    if (index_ptr == NULL || vector == NULL) {
        return -1;
    }

    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (vector->dimension != index->dimension) {
        return -1;
    }

    size_t level = calculate_level(index->maxLevel);
    GV_HNSWNode *new_node = (GV_HNSWNode *)malloc(sizeof(GV_HNSWNode));
    if (new_node == NULL) {
        return -1;
    }

    new_node->vector = vector;
    new_node->binary_vector = NULL;
    new_node->level = level;
    new_node->neighbors = (GV_HNSWNode ***)malloc((level + 1) * sizeof(GV_HNSWNode **));
    new_node->neighbor_counts = (size_t *)calloc(level + 1, sizeof(size_t));
    if (new_node->neighbors == NULL || new_node->neighbor_counts == NULL) {
        free(new_node->neighbors);
        free(new_node->neighbor_counts);
        free(new_node);
        return -1;
    }

    if (index->use_binary_quant) {
        new_node->binary_vector = gv_binary_quantize(vector->data, vector->dimension);
        if (new_node->binary_vector == NULL) {
            for (size_t l = 0; l <= level; ++l) {
                free(new_node->neighbors[l]);
            }
            free(new_node->neighbors);
            free(new_node->neighbor_counts);
            free(new_node);
            return -1;
        }
    }

    for (size_t l = 0; l <= level; ++l) {
        new_node->neighbors[l] = (GV_HNSWNode **)malloc(index->M * sizeof(GV_HNSWNode *));
        if (new_node->neighbors[l] == NULL) {
            for (size_t i = 0; i < l; ++i) {
                free(new_node->neighbors[i]);
            }
            free(new_node->neighbors);
            free(new_node->neighbor_counts);
            free(new_node);
            return -1;
        }
        new_node->neighbor_counts[l] = 0;
    }

    if (index->entryPoint == NULL) {
        index->entryPoint = new_node;
    } else if (level > index->entryPoint->level) {
        index->entryPoint = new_node;
    }

    if (index->count >= index->nodes_capacity) {
        size_t new_capacity = index->nodes_capacity * 2;
        GV_HNSWNode **new_nodes = (GV_HNSWNode **)realloc(index->nodes, 
                                                           new_capacity * sizeof(GV_HNSWNode *));
        if (new_nodes == NULL) {
            gv_hnsw_node_destroy(new_node);
            return -1;
        }
        index->nodes = new_nodes;
        index->nodes_capacity = new_capacity;
    }

    index->nodes[index->count++] = new_node;

    if (index->count == 1) {
        return 0;
    }

    GV_HNSWNode *current = index->entryPoint;
    if (current == NULL || current->vector == NULL) {
        return 0;
    }
    
    size_t currentLevel = current->level;
    if (currentLevel > index->maxLevel) {
        currentLevel = index->maxLevel;
    }

    for (int lc = (int)currentLevel; lc > (int)level; --lc) {
        if (current == NULL || (size_t)lc > current->level) continue;
        
        float minDist = FLT_MAX;
        GV_HNSWNode *closest = current;
        
        if (current->neighbor_counts != NULL && current->neighbors != NULL &&
            (size_t)lc <= current->level && current->neighbor_counts[lc] > 0 && 
            current->neighbors[lc] != NULL) {
            for (size_t i = 0; i < current->neighbor_counts[lc]; ++i) {
                if (current->neighbors[lc][i] != NULL && 
                    current->neighbors[lc][i]->vector != NULL) {
                    float dist = gv_distance_euclidean(current->neighbors[lc][i]->vector, vector);
                    if (dist < minDist) {
                        minDist = dist;
                        closest = current->neighbors[lc][i];
                    }
                }
            }
        }
        if (closest != NULL && closest != current && closest->vector != NULL) {
            current = closest;
            currentLevel = current->level;
            if (currentLevel > index->maxLevel) {
                currentLevel = index->maxLevel;
            }
        }
    }
    
    if (current == NULL || current->vector == NULL) {
        return 0;
    }

    size_t searchLevel = (currentLevel < level ? currentLevel : level);
    if (searchLevel > current->level) {
        searchLevel = current->level;
    }
    if (searchLevel > new_node->level) {
        searchLevel = new_node->level;
    }
    
    for (int lc = (int)searchLevel; lc >= 0; --lc) {
        if (current == NULL || new_node == NULL || 
            (size_t)lc > current->level || (size_t)lc > new_node->level) {
            continue;
        }
        
        GV_HNSWCandidate *candidates = (GV_HNSWCandidate *)malloc(index->efConstruction * sizeof(GV_HNSWCandidate));
        if (candidates == NULL) continue;

        size_t candidate_count = 0;
        if (current->vector != NULL) {
            candidates[candidate_count].node = current;
            candidates[candidate_count++].distance = gv_distance_euclidean(current->vector, vector);
        }

        if (current->neighbor_counts != NULL && current->neighbors != NULL &&
            (size_t)lc <= current->level && current->neighbor_counts[lc] > 0 && 
            current->neighbors[lc] != NULL) {
            for (size_t i = 0; i < current->neighbor_counts[lc] && candidate_count < index->efConstruction; ++i) {
                if (current->neighbors[lc][i] != NULL && 
                    current->neighbors[lc][i]->vector != NULL) {
                    candidates[candidate_count].node = current->neighbors[lc][i];
                    candidates[candidate_count++].distance = gv_distance_euclidean(current->neighbors[lc][i]->vector, vector);
                }
            }
        }

        if (candidate_count == 0) {
            free(candidates);
            continue;
        }

        qsort(candidates, candidate_count, sizeof(GV_HNSWCandidate), compare_candidates);

        size_t selected_count = (candidate_count < index->M) ? candidate_count : index->M;
        for (size_t i = 0; i < selected_count; ++i) {
            if (candidates[i].node == NULL || candidates[i].node == new_node) continue;
            if (candidates[i].node->neighbors == NULL || candidates[i].node->neighbor_counts == NULL) continue;
            
            if ((size_t)lc <= new_node->level && new_node->neighbor_counts != NULL &&
                new_node->neighbors != NULL && new_node->neighbor_counts[lc] < index->M &&
                new_node->neighbors[lc] != NULL) {
                new_node->neighbors[lc][new_node->neighbor_counts[lc]++] = candidates[i].node;
            }
            if ((size_t)lc <= candidates[i].node->level && 
                candidates[i].node->neighbor_counts[lc] < index->M &&
                candidates[i].node->neighbors[lc] != NULL) {
                candidates[i].node->neighbors[lc][candidates[i].node->neighbor_counts[lc]++] = new_node;
            }
        }

        free(candidates);
    }

    return 0;
}

int gv_hnsw_search(void *index_ptr, const GV_Vector *query, size_t k,
                   GV_SearchResult *results, GV_DistanceType distance_type,
                   const char *filter_key, const char *filter_value) {
    if (index_ptr == NULL || query == NULL || results == NULL || k == 0) {
        return -1;
    }

    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (query->dimension != index->dimension || index->entryPoint == NULL) {
        return 0;
    }

    memset(results, 0, k * sizeof(GV_SearchResult));

    GV_BinaryVector *query_binary = NULL;
    if (index->use_binary_quant) {
        query_binary = gv_binary_quantize(query->data, query->dimension);
        if (query_binary == NULL) {
            return -1;
        }
    }

    GV_HNSWNode *current = index->entryPoint;
    size_t currentLevel = index->entryPoint->level;

    for (int lc = (int)currentLevel; lc > 0; --lc) {
        if (current == NULL || (size_t)lc > current->level) continue;
        
        float minDist = FLT_MAX;
        GV_HNSWNode *closest = current;
        
        if (current->neighbor_counts != NULL && current->neighbors != NULL &&
            (size_t)lc <= current->level && current->neighbor_counts[lc] > 0 && 
            current->neighbors[lc] != NULL) {
            for (size_t i = 0; i < current->neighbor_counts[lc]; ++i) {
                if (current->neighbors[lc][i] != NULL && 
                    current->neighbors[lc][i]->vector != NULL) {
                    float dist;
                    if (index->use_binary_quant && query_binary != NULL && 
                        current->neighbors[lc][i]->binary_vector != NULL) {
                        size_t hamming = gv_binary_hamming_distance_fast(
                            query_binary, current->neighbors[lc][i]->binary_vector);
                        dist = (float)hamming;
                    } else {
                        dist = gv_distance(current->neighbors[lc][i]->vector, query, distance_type);
                    }
                    if (dist < minDist) {
                        minDist = dist;
                        closest = current->neighbors[lc][i];
                    }
                }
            }
        }
        if (closest != NULL && closest != current && closest->vector != NULL) {
            current = closest;
            currentLevel = current->level;
        }
    }
    
    if (current == NULL || current->vector == NULL) {
        return 0;
    }

    GV_HNSWCandidate *candidates = (GV_HNSWCandidate *)malloc(index->efSearch * sizeof(GV_HNSWCandidate));
    if (candidates == NULL) {
        return -1;
    }

    size_t candidate_count = 0;
    int *visited = (int *)calloc(index->count, sizeof(int));
    if (visited == NULL) {
        free(candidates);
        return -1;
    }

    float current_dist;
    if (index->use_binary_quant && query_binary != NULL && current->binary_vector != NULL) {
        size_t hamming = gv_binary_hamming_distance_fast(query_binary, current->binary_vector);
        current_dist = (float)hamming;
    } else {
        current_dist = gv_distance(current->vector, query, distance_type);
    }
    candidates[candidate_count].node = current;
    candidates[candidate_count++].distance = current_dist;
    
    size_t current_idx = 0;
    for (size_t i = 0; i < index->count; ++i) {
        if (index->nodes[i] == current) {
            current_idx = i;
            break;
        }
    }
    visited[current_idx] = 1;

    size_t visited_count = 1;
    size_t candidate_idx = 0;

    while (candidate_idx < candidate_count) {
        GV_HNSWNode *candidate_node = candidates[candidate_idx].node;
        candidate_idx++;

        if (candidate_node == NULL || candidate_node->neighbor_counts == NULL || 
            candidate_node->neighbors == NULL || candidate_node->neighbor_counts[0] == 0 || 
            candidate_node->neighbors[0] == NULL) {
            continue;
        }
        
        if (candidate_count >= index->efSearch) {
            break;
        }

        for (size_t i = 0; i < candidate_node->neighbor_counts[0]; ++i) {
            GV_HNSWNode *neighbor = candidate_node->neighbors[0][i];
            if (neighbor == NULL || neighbor->vector == NULL) continue;

            size_t node_idx = SIZE_MAX;
            for (size_t j = 0; j < index->count; ++j) {
                if (index->nodes[j] == neighbor) {
                    node_idx = j;
                    break;
                }
            }
            if (node_idx == SIZE_MAX || node_idx >= index->count || visited[node_idx]) continue;

            visited[node_idx] = 1;
            visited_count++;

            float dist;
            if (index->use_binary_quant && query_binary != NULL && neighbor->binary_vector != NULL) {
                size_t hamming = gv_binary_hamming_distance_fast(query_binary, neighbor->binary_vector);
                dist = (float)hamming;
            } else {
                dist = gv_distance(neighbor->vector, query, distance_type);
            }

            if (candidate_count < index->efSearch) {
                candidates[candidate_count].node = neighbor;
                candidates[candidate_count++].distance = dist;
                qsort(candidates, candidate_count, sizeof(GV_HNSWCandidate), compare_candidates);
            } else {
                float worst_dist = candidates[candidate_count - 1].distance;
                if (dist < worst_dist) {
                    candidates[candidate_count - 1].node = neighbor;
                    candidates[candidate_count - 1].distance = dist;
                    qsort(candidates, candidate_count, sizeof(GV_HNSWCandidate), compare_candidates);
                }
            }
        }
    }

    free(visited);

    if (candidate_count == 0) {
        if (query_binary != NULL) {
            gv_binary_vector_destroy(query_binary);
        }
        free(candidates);
        return 0;
    }

    qsort(candidates, candidate_count, sizeof(GV_HNSWCandidate), compare_candidates);

    if (index->use_binary_quant && index->quant_rerank > 0 && query_binary != NULL) {
        size_t rerank_count = (candidate_count < index->quant_rerank) ? candidate_count : index->quant_rerank;
        for (size_t i = 0; i < rerank_count; ++i) {
            if (candidates[i].node != NULL && candidates[i].node->vector != NULL) {
                candidates[i].distance = gv_distance(candidates[i].node->vector, query, distance_type);
            }
        }
        qsort(candidates, candidate_count, sizeof(GV_HNSWCandidate), compare_candidates);
    }

    size_t result_count = 0;
    for (size_t i = 0; i < candidate_count && result_count < k; ++i) {
        if (candidates[i].node == NULL || candidates[i].node->vector == NULL) continue;
        
        if (filter_key == NULL || gv_vector_get_metadata(candidates[i].node->vector, filter_key) != NULL) {
            if (filter_key == NULL || strcmp(gv_vector_get_metadata(candidates[i].node->vector, filter_key), filter_value) == 0) {
                results[result_count].vector = candidates[i].node->vector;
                results[result_count].distance = candidates[i].distance;
                if (distance_type == GV_DISTANCE_COSINE) {
                    results[result_count].distance = 1.0f - results[result_count].distance;
                }
                result_count++;
            }
        }
    }

    if (query_binary != NULL) {
        gv_binary_vector_destroy(query_binary);
    }
    free(candidates);
    return (int)result_count;
}

void gv_hnsw_destroy(void *index_ptr) {
    if (index_ptr == NULL) {
        return;
    }

    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    for (size_t i = 0; i < index->count; ++i) {
        gv_hnsw_node_destroy(index->nodes[i]);
    }
    free(index->nodes);
    free(index);
}

size_t gv_hnsw_count(const void *index_ptr) {
    if (index_ptr == NULL) {
        return 0;
    }
    return ((GV_HNSWIndex *)index_ptr)->count;
}

static int gv_write_uint32(FILE *out, uint32_t value) {
    return (fwrite(&value, sizeof(uint32_t), 1, out) == 1) ? 0 : -1;
}

static int gv_write_uint64(FILE *out, uint64_t value) {
    return (fwrite(&value, sizeof(uint64_t), 1, out) == 1) ? 0 : -1;
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

static int gv_read_uint32(FILE *in, uint32_t *value) {
    return (value != NULL && fread(value, sizeof(uint32_t), 1, in) == 1) ? 0 : -1;
}

static int gv_read_uint64(FILE *in, uint64_t *value) {
    return (value != NULL && fread(value, sizeof(uint64_t), 1, in) == 1) ? 0 : -1;
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

int gv_hnsw_save(const void *index_ptr, FILE *out, uint32_t version) {
    if (index_ptr == NULL || out == NULL) {
        return -1;
    }

    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (gv_write_uint32(out, (uint32_t)index->M) != 0) {
        return -1;
    }
    if (gv_write_uint32(out, (uint32_t)index->efConstruction) != 0) {
        return -1;
    }
    if (gv_write_uint32(out, (uint32_t)index->efSearch) != 0) {
        return -1;
    }
    if (gv_write_uint32(out, (uint32_t)index->maxLevel) != 0) {
        return -1;
    }
    if (gv_write_uint64(out, (uint64_t)index->count) != 0) {
        return -1;
    }

    uint64_t entry_point_idx = UINT64_MAX;
    if (index->entryPoint != NULL) {
        for (size_t i = 0; i < index->count; ++i) {
            if (index->nodes[i] == index->entryPoint) {
                entry_point_idx = (uint64_t)i;
                break;
            }
        }
    }
    if (gv_write_uint64(out, entry_point_idx) != 0) {
        return -1;
    }

    /* First pass: write node metadata and vector data so the reader can
       reconstruct nodes before wiring up graph links. */
    for (size_t i = 0; i < index->count; ++i) {
        GV_HNSWNode *node = index->nodes[i];
        if (node == NULL || node->vector == NULL) {
            return -1;
        }

        if (gv_write_uint32(out, (uint32_t)node->level) != 0) {
            return -1;
        }

        if (gv_write_floats(out, node->vector->data, node->vector->dimension) != 0) {
            return -1;
        }

        if (version >= 2) {
            if (gv_write_metadata(out, node->vector->metadata) != 0) {
                return -1;
            }
        }
    }

    /* Second pass: write neighbor graph connectivity. */
    for (size_t i = 0; i < index->count; ++i) {
        GV_HNSWNode *node = index->nodes[i];
        if (node == NULL) {
            return -1;
        }
        for (size_t l = 0; l <= node->level; ++l) {
            if (gv_write_uint32(out, (uint32_t)node->neighbor_counts[l]) != 0) {
                return -1;
            }
            for (size_t j = 0; j < node->neighbor_counts[l]; ++j) {
                size_t neighbor_idx = SIZE_MAX;
                for (size_t k = 0; k < index->count; ++k) {
                    if (index->nodes[k] == node->neighbors[l][j]) {
                        neighbor_idx = k;
                        break;
                    }
                }
                if (neighbor_idx == SIZE_MAX) {
                    return -1;
                }
                if (gv_write_uint64(out, (uint64_t)neighbor_idx) != 0) {
                    return -1;
                }
            }
        }
    }

    return 0;
}

int gv_hnsw_load(void **index_ptr, FILE *in, size_t dimension, uint32_t version) {
    if (index_ptr == NULL || in == NULL || dimension == 0) {
        return -1;
    }

    uint32_t M = 0;
    uint32_t efConstruction = 0;
    uint32_t efSearch = 0;
    uint32_t maxLevel = 0;
    uint64_t count = 0;
    uint64_t entry_point_idx = 0;
    if (gv_read_uint32(in, &M) != 0) {
        return -1;
    }
    if (gv_read_uint32(in, &efConstruction) != 0) {
        return -1;
    }
    if (gv_read_uint32(in, &efSearch) != 0) {
        return -1;
    }
    if (gv_read_uint32(in, &maxLevel) != 0) {
        return -1;
    }
    if (gv_read_uint64(in, &count) != 0) {
        return -1;
    }
    if (gv_read_uint64(in, &entry_point_idx) != 0) {
        return -1;
    }

    GV_HNSWConfig config = {.M = M, .efConstruction = efConstruction, .efSearch = efSearch, .maxLevel = maxLevel};
    void *index = gv_hnsw_create(dimension, &config);
    if (index == NULL) {
        return -1;
    }

    GV_HNSWIndex *hnsw_index = (GV_HNSWIndex *)index;
    if (count == 0) {
        *index_ptr = index;
        return 0;
    }
    hnsw_index->count = (size_t)count;

    if (count > hnsw_index->nodes_capacity) {
        size_t new_capacity = (size_t)count;
        GV_HNSWNode **new_nodes = (GV_HNSWNode **)realloc(hnsw_index->nodes, new_capacity * sizeof(GV_HNSWNode *));
        if (new_nodes == NULL) {
            gv_hnsw_destroy(index);
            return -1;
        }
        hnsw_index->nodes = new_nodes;
        hnsw_index->nodes_capacity = new_capacity;
    }

    for (size_t i = 0; i < (size_t)count; ++i) {
        uint32_t level = 0;
        if (gv_read_uint32(in, &level) != 0) {
            gv_hnsw_destroy(index);
            return -1;
        }

        GV_Vector *vector = gv_vector_create(dimension);
        if (vector == NULL) {
            gv_hnsw_destroy(index);
            return -1;
        }

        if (gv_read_floats(in, vector->data, dimension) != 0) {
            gv_vector_destroy(vector);
            gv_hnsw_destroy(index);
            return -1;
        }

        if (version >= 2) {
            if (gv_read_metadata(in, vector) != 0) {
                gv_vector_destroy(vector);
                gv_hnsw_destroy(index);
                return -1;
            }
        }

        GV_HNSWNode *node = (GV_HNSWNode *)malloc(sizeof(GV_HNSWNode));
        if (node == NULL) {
            gv_vector_destroy(vector);
            gv_hnsw_destroy(index);
            return -1;
        }

        node->vector = vector;
        node->level = level;
        node->neighbors = (GV_HNSWNode ***)malloc((level + 1) * sizeof(GV_HNSWNode **));
        node->neighbor_counts = (size_t *)calloc(level + 1, sizeof(size_t));
        if (node->neighbors == NULL || node->neighbor_counts == NULL) {
            free(node->neighbors);
            free(node->neighbor_counts);
            gv_vector_destroy(vector);
            free(node);
            gv_hnsw_destroy(index);
            return -1;
        }

        for (size_t l = 0; l <= level; ++l) {
            size_t max_neighbors = (size_t)M;
            node->neighbors[l] = (GV_HNSWNode **)malloc(max_neighbors * sizeof(GV_HNSWNode *));
            if (node->neighbors[l] == NULL) {
                for (size_t j = 0; j < l; ++j) {
                    free(node->neighbors[j]);
                }
                free(node->neighbors);
                free(node->neighbor_counts);
                gv_vector_destroy(vector);
                free(node);
                gv_hnsw_destroy(index);
                return -1;
            }
            node->neighbor_counts[l] = 0;
        }

        hnsw_index->nodes[i] = node;
    }

    for (size_t i = 0; i < (size_t)count; ++i) {
        GV_HNSWNode *node = hnsw_index->nodes[i];
        for (size_t l = 0; l <= node->level; ++l) {
            uint32_t neighbor_count = 0;
            if (gv_read_uint32(in, &neighbor_count) != 0) {
                gv_hnsw_destroy(index);
                return -1;
            }
            size_t max_neighbors = (size_t)M;
            for (uint32_t j = 0; j < neighbor_count; ++j) {
                uint64_t neighbor_idx = 0;
                if (gv_read_uint64(in, &neighbor_idx) != 0) {
                    gv_hnsw_destroy(index);
                    return -1;
                }
                if (neighbor_idx >= count) {
                    gv_hnsw_destroy(index);
                    return -1;
                }
                if (node->neighbor_counts[l] < max_neighbors) {
                    node->neighbors[l][node->neighbor_counts[l]++] = hnsw_index->nodes[neighbor_idx];
                }
            }
        }
    }

    if (entry_point_idx != UINT64_MAX && entry_point_idx < count) {
        hnsw_index->entryPoint = hnsw_index->nodes[entry_point_idx];
    } else if (count > 0) {
        hnsw_index->entryPoint = hnsw_index->nodes[0];
    } else {
        hnsw_index->entryPoint = NULL;
    }

    *index_ptr = index;
    return 0;
}
