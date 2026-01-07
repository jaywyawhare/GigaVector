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
#include "gigavector/gv_soa_storage.h"

typedef struct GV_HNSWNode {
    size_t vector_index;             /**< Index into GV_SoAStorage */
    GV_BinaryVector *binary_vector;  /**< Binary quantized version for fast search */
    struct GV_HNSWNode ***neighbors;
    size_t *neighbor_counts;
    size_t level;
    size_t index;                    /**< Stable index position in GV_HNSWIndex::nodes */
    int deleted;                     /**< Deletion flag: 1 if deleted, 0 if active */
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
    int use_acorn;
    size_t acorn_hops;
    GV_HNSWNode *entryPoint;
    size_t count;
    GV_HNSWNode **nodes;
    size_t nodes_capacity;
    GV_SoAStorage *soa_storage;      /**< Structure-of-Arrays storage for vectors */
    int soa_storage_owned;           /**< 1 if we own the storage (should destroy), 0 if shared */
} GV_HNSWIndex;

static const char *gv_metadata_get_direct(GV_Metadata *metadata, const char *key) {
    if (metadata == NULL || key == NULL) {
        return NULL;
    }
    GV_Metadata *current = metadata;
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }
    return NULL;
}

static GV_Metadata *gv_metadata_copy(GV_Metadata *src) {
    if (src == NULL) {
        return NULL;
    }
    GV_Metadata *head = NULL;
    GV_Metadata *tail = NULL;
    GV_Metadata *current = src;
    
    while (current != NULL) {
        GV_Metadata *new_meta = (GV_Metadata *)malloc(sizeof(GV_Metadata));
        if (new_meta == NULL) {
            /* Free what we've allocated so far */
            while (head != NULL) {
                GV_Metadata *next = head->next;
                free(head->key);
                free(head->value);
                free(head);
                head = next;
            }
            return NULL;
        }
        new_meta->key = (char *)malloc(strlen(current->key) + 1);
        new_meta->value = (char *)malloc(strlen(current->value) + 1);
        if (new_meta->key == NULL || new_meta->value == NULL) {
            free(new_meta->key);
            free(new_meta->value);
            free(new_meta);
            /* Free what we've allocated so far */
            while (head != NULL) {
                GV_Metadata *next = head->next;
                free(head->key);
                free(head->value);
                free(head);
                head = next;
            }
            return NULL;
        }
        strcpy(new_meta->key, current->key);
        strcpy(new_meta->value, current->value);
        new_meta->next = NULL;
        
        if (head == NULL) {
            head = tail = new_meta;
        } else {
            tail->next = new_meta;
            tail = new_meta;
        }
        current = current->next;
    }
    return head;
}

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
    /* Vector is stored in SoA, don't destroy it here */
    if (node->binary_vector != NULL) {
        gv_binary_vector_destroy(node->binary_vector);
    }
    free(node);
}

void *gv_hnsw_create(size_t dimension, const GV_HNSWConfig *config, GV_SoAStorage *soa_storage) {
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
    index->use_acorn = (config && config->use_acorn) ? 1 : 0;
    index->acorn_hops = (config && config->acorn_hops > 0 && config->acorn_hops <= 2) ? config->acorn_hops : 1;
    index->entryPoint = NULL;
    index->count = 0;
    index->nodes_capacity = 1024;
    index->nodes = (GV_HNSWNode **)malloc(index->nodes_capacity * sizeof(GV_HNSWNode *));
    if (index->nodes == NULL) {
        free(index);
        return NULL;
    }

    if (soa_storage != NULL) {
        index->soa_storage = soa_storage;
        index->soa_storage_owned = 0;  /* Shared storage, don't destroy */
    } else {
        index->soa_storage = gv_soa_storage_create(dimension, 1024);
        if (index->soa_storage == NULL) {
            free(index->nodes);
            free(index);
            return NULL;
        }
        index->soa_storage_owned = 1;  /* We own it, should destroy */
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

    /* Add vector to SoA storage */
    /* Transfer metadata ownership to SoA storage */
    GV_Metadata *metadata = vector->metadata;
    vector->metadata = NULL;  /* Clear from original vector to transfer ownership */
    size_t vector_index = gv_soa_storage_add(index->soa_storage, vector->data, metadata);
    if (vector_index == (size_t)-1) {
        /* Restore metadata if add failed */
        vector->metadata = metadata;
        return -1;
    }

    size_t level = calculate_level(index->maxLevel);
    GV_HNSWNode *new_node = (GV_HNSWNode *)malloc(sizeof(GV_HNSWNode));
    if (new_node == NULL) {
        return -1;
    }

    new_node->vector_index = vector_index;
    new_node->binary_vector = NULL;
    new_node->level = level;
    new_node->deleted = 0;
    new_node->neighbors = (GV_HNSWNode ***)malloc((level + 1) * sizeof(GV_HNSWNode **));
    new_node->neighbor_counts = (size_t *)calloc(level + 1, sizeof(size_t));
    if (new_node->neighbors == NULL || new_node->neighbor_counts == NULL) {
        free(new_node->neighbors);
        free(new_node->neighbor_counts);
        free(new_node);
        return -1;
    }

    if (index->use_binary_quant) {
        const float *vector_data = gv_soa_storage_get_data(index->soa_storage, vector_index);
        if (vector_data == NULL) {
            for (size_t l = 0; l <= level; ++l) {
                free(new_node->neighbors[l]);
            }
            free(new_node->neighbors);
            free(new_node->neighbor_counts);
            free(new_node);
            return -1;
        }
        new_node->binary_vector = gv_binary_quantize(vector_data, vector->dimension);
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

    new_node->index = index->count;
    index->nodes[index->count++] = new_node;

    if (index->count == 1) {
        return 0;
    }

    const float *new_vector_data = gv_soa_storage_get_data(index->soa_storage, vector_index);
    if (new_vector_data == NULL) {
        return -1;
    }

    GV_HNSWNode *current = index->entryPoint;
    if (current == NULL) {
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
                    current->neighbors[lc][i]->deleted == 0) {
                    const float *neighbor_data = gv_soa_storage_get_data(index->soa_storage, current->neighbors[lc][i]->vector_index);
                    if (neighbor_data != NULL) {
                        float dist = 0.0f;
                        for (size_t d = 0; d < index->dimension; ++d) {
                            float diff = new_vector_data[d] - neighbor_data[d];
                            dist += diff * diff;
                        }
                    if (dist < minDist) {
                        minDist = dist;
                        closest = current->neighbors[lc][i];
                    }
                }
            }
        }
        }
        if (closest != NULL && closest != current && closest->deleted == 0) {
            current = closest;
            currentLevel = current->level;
            if (currentLevel > index->maxLevel) {
                currentLevel = index->maxLevel;
            }
        }
    }
    
    if (current == NULL) {
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
        const float *current_data = gv_soa_storage_get_data(index->soa_storage, current->vector_index);
        if (current_data != NULL) {
            candidates[candidate_count].node = current;
            float dist = 0.0f;
            for (size_t d = 0; d < index->dimension; ++d) {
                float diff = new_vector_data[d] - current_data[d];
                dist += diff * diff;
            }
            candidates[candidate_count++].distance = dist;
        }

        if (current->neighbor_counts != NULL && current->neighbors != NULL &&
            (size_t)lc <= current->level && current->neighbor_counts[lc] > 0 && 
            current->neighbors[lc] != NULL) {
            for (size_t i = 0; i < current->neighbor_counts[lc] && candidate_count < index->efConstruction; ++i) {
                if (current->neighbors[lc][i] != NULL && 
                    current->neighbors[lc][i]->deleted == 0) {
                    const float *neighbor_data = gv_soa_storage_get_data(index->soa_storage, current->neighbors[lc][i]->vector_index);
                    if (neighbor_data != NULL) {
                    candidates[candidate_count].node = current->neighbors[lc][i];
                        float dist = 0.0f;
                        for (size_t d = 0; d < index->dimension; ++d) {
                            float diff = new_vector_data[d] - neighbor_data[d];
                            dist += diff * diff;
                        }
                        candidates[candidate_count++].distance = dist;
                    }
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
            if (candidates[i].node == NULL || candidates[i].node == new_node || candidates[i].node->deleted != 0) continue;
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
                    current->neighbors[lc][i]->deleted == 0) {
                    float dist;
                    if (index->use_binary_quant && query_binary != NULL && 
                        current->neighbors[lc][i]->binary_vector != NULL) {
                        size_t hamming = gv_binary_hamming_distance_fast(
                            query_binary, current->neighbors[lc][i]->binary_vector);
                        dist = (float)hamming;
                    } else {
                        const float *neighbor_data = gv_soa_storage_get_data(index->soa_storage, current->neighbors[lc][i]->vector_index);
                        if (neighbor_data != NULL) {
                            GV_Vector temp_neighbor = {.data = (float *)neighbor_data, .dimension = query->dimension, .metadata = NULL};
                            dist = gv_distance(&temp_neighbor, query, distance_type);
                        } else {
                            continue;
                        }
                    }
                    if (dist < minDist) {
                        minDist = dist;
                        closest = current->neighbors[lc][i];
                    }
                }
            }
        }
        if (closest != NULL && closest != current && closest->deleted == 0) {
            current = closest;
            currentLevel = current->level;
        }
    }
    
    if (current == NULL) {
        return 0;
    }

    size_t ef = index->efSearch;
    if (filter_key != NULL && index->use_acorn) {
        size_t factor = index->acorn_hops + 1;
        if (factor > 3) {
            factor = 3;
        }
        if (ef > 0 && ef <= SIZE_MAX / factor) {
            ef *= factor;
        }
    }

    GV_HNSWCandidate *candidates = (GV_HNSWCandidate *)malloc(ef * sizeof(GV_HNSWCandidate));
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
        const float *current_data = gv_soa_storage_get_data(index->soa_storage, current->vector_index);
        if (current_data == NULL) {
            free(candidates);
            free(visited);
            if (query_binary != NULL) {
                gv_binary_vector_destroy(query_binary);
            }
            return -1;
        }
        GV_Vector temp_current = {.data = (float *)current_data, .dimension = query->dimension, .metadata = NULL};
        current_dist = gv_distance(&temp_current, query, distance_type);
    }
    candidates[candidate_count].node = current;
    candidates[candidate_count++].distance = current_dist;
    
    visited[current->index] = 1;

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
        
        if (candidate_count >= ef) {
            break;
        }

        for (size_t i = 0; i < candidate_node->neighbor_counts[0]; ++i) {
            GV_HNSWNode *neighbor = candidate_node->neighbors[0][i];
            if (neighbor == NULL || neighbor->deleted != 0) continue;

            size_t node_idx = neighbor->index;
            if (node_idx >= index->count || visited[node_idx]) continue;

            visited[node_idx] = 1;
            visited_count++;

            float dist;
            if (index->use_binary_quant && query_binary != NULL && neighbor->binary_vector != NULL) {
                size_t hamming = gv_binary_hamming_distance_fast(query_binary, neighbor->binary_vector);
                dist = (float)hamming;
            } else {
                const float *neighbor_data = gv_soa_storage_get_data(index->soa_storage, neighbor->vector_index);
                if (neighbor_data == NULL) continue;
                GV_Vector temp_neighbor = {.data = (float *)neighbor_data, .dimension = query->dimension, .metadata = NULL};
                dist = gv_distance(&temp_neighbor, query, distance_type);
            }

            if (candidate_count < ef) {
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
            if (candidates[i].node != NULL && candidates[i].node->deleted == 0) {
                const float *node_data = gv_soa_storage_get_data(index->soa_storage, candidates[i].node->vector_index);
                if (node_data != NULL) {
                    GV_Vector temp_vec = {.data = (float *)node_data, .dimension = query->dimension, .metadata = NULL};
                    candidates[i].distance = gv_distance(&temp_vec, query, distance_type);
                }
            }
        }
        qsort(candidates, candidate_count, sizeof(GV_HNSWCandidate), compare_candidates);
    }

    size_t result_count = 0;
    for (size_t i = 0; i < candidate_count && result_count < k; ++i) {
        if (candidates[i].node == NULL || candidates[i].node->deleted != 0) continue;
        
        const float *node_data = gv_soa_storage_get_data(index->soa_storage, candidates[i].node->vector_index);
        if (node_data == NULL) continue;
        
        GV_Metadata *node_metadata = gv_soa_storage_get_metadata(index->soa_storage, candidates[i].node->vector_index);
        if (filter_key == NULL || gv_metadata_get_direct(node_metadata, filter_key) != NULL) {
            if (filter_key == NULL || strcmp(gv_metadata_get_direct(node_metadata, filter_key), filter_value) == 0) {
                /* Create a temporary vector view for the result */
                GV_Vector *result_vec = (GV_Vector *)malloc(sizeof(GV_Vector));
                if (result_vec == NULL) continue;
                result_vec->data = (float *)malloc(query->dimension * sizeof(float));
                if (result_vec->data == NULL) {
                    free(result_vec);
                    continue;
                }
                memcpy(result_vec->data, node_data, query->dimension * sizeof(float));
                result_vec->dimension = query->dimension;
                result_vec->metadata = gv_metadata_copy(node_metadata);
                results[result_count].vector = result_vec;
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
    /* Only destroy SoA storage if we own it (created it ourselves) */
    if (index->soa_storage != NULL && index->soa_storage_owned != 0) {
        gv_soa_storage_destroy(index->soa_storage);
    }
    free(index);
}

size_t gv_hnsw_count(const void *index_ptr) {
    if (index_ptr == NULL) {
        return 0;
    }
    return ((GV_HNSWIndex *)index_ptr)->count;
}

int gv_hnsw_delete(void *index_ptr, size_t node_index) {
    if (index_ptr == NULL) {
        return -1;
    }

    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (node_index >= index->count) {
        return -1;
    }

    GV_HNSWNode *node = index->nodes[node_index];
    if (node == NULL || node->deleted != 0) {
        return -1;
    }

    node->deleted = 1;

    for (size_t i = 0; i < index->count; ++i) {
        GV_HNSWNode *other = index->nodes[i];
        if (other == NULL || other->deleted != 0 || other == node) {
            continue;
        }

        for (size_t l = 0; l <= other->level; ++l) {
            if (other->neighbors == NULL || other->neighbor_counts == NULL) {
                continue;
            }
            if ((size_t)l > other->level || other->neighbors[l] == NULL) {
                continue;
            }

            size_t write_pos = 0;
            for (size_t j = 0; j < other->neighbor_counts[l]; ++j) {
                if (other->neighbors[l][j] != node) {
                    other->neighbors[l][write_pos++] = other->neighbors[l][j];
                }
            }
            other->neighbor_counts[l] = write_pos;
        }
    }

    if (index->entryPoint == node) {
        for (size_t i = 0; i < index->count; ++i) {
            if (index->nodes[i] != NULL && index->nodes[i]->deleted == 0) {
                index->entryPoint = index->nodes[i];
                break;
            }
        }
    }

    return 0;
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
        if (node == NULL) {
            return -1;
        }

        if (gv_write_uint32(out, (uint32_t)node->level) != 0) {
            return -1;
        }

        const float *vector_data = gv_soa_storage_get_data(index->soa_storage, node->vector_index);
        if (vector_data == NULL) {
            return -1;
        }
        if (gv_write_floats(out, vector_data, index->dimension) != 0) {
            return -1;
        }

        if (version >= 2) {
            GV_Metadata *metadata = gv_soa_storage_get_metadata(index->soa_storage, node->vector_index);
            if (gv_write_metadata(out, metadata) != 0) {
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
    void *index = gv_hnsw_create(dimension, &config, NULL);
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

        float *vector_data = (float *)malloc(dimension * sizeof(float));
        if (vector_data == NULL) {
            gv_hnsw_destroy(index);
            return -1;
        }

        if (gv_read_floats(in, vector_data, dimension) != 0) {
            free(vector_data);
            gv_hnsw_destroy(index);
            return -1;
        }

        GV_Metadata *metadata = NULL;
        if (version >= 2) {
            GV_Vector temp_vec = {.data = vector_data, .dimension = dimension, .metadata = NULL};
            if (gv_read_metadata(in, &temp_vec) != 0) {
                free(vector_data);
                gv_hnsw_destroy(index);
                return -1;
            }
            metadata = temp_vec.metadata;
        }

        size_t vector_index = gv_soa_storage_add(hnsw_index->soa_storage, vector_data, metadata);
        free(vector_data);
        if (vector_index == (size_t)-1) {
            gv_hnsw_destroy(index);
            return -1;
        }

        GV_HNSWNode *node = (GV_HNSWNode *)malloc(sizeof(GV_HNSWNode));
        if (node == NULL) {
            gv_hnsw_destroy(index);
            return -1;
        }

        node->vector_index = vector_index;
        node->level = level;
        node->deleted = 0;
        node->neighbors = (GV_HNSWNode ***)malloc((level + 1) * sizeof(GV_HNSWNode **));
        node->neighbor_counts = (size_t *)calloc(level + 1, sizeof(size_t));
        if (node->neighbors == NULL || node->neighbor_counts == NULL) {
            free(node->neighbors);
            free(node->neighbor_counts);
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

int gv_hnsw_range_search(void *index_ptr, const GV_Vector *query, float radius,
                         GV_SearchResult *results, size_t max_results,
                         GV_DistanceType distance_type,
                         const char *filter_key, const char *filter_value) {
    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (index == NULL || query == NULL || results == NULL || max_results == 0 || radius < 0.0f ||
        query->dimension != index->dimension || query->data == NULL) {
        return -1;
    }

    if (index->count == 0 || index->entryPoint == NULL) {
        return 0;
    }

    /* Reuse the standard HNSW k-NN search and post-filter by radius. This keeps
     * the implementation simple and consistent with the main search path. */
    GV_SearchResult *tmp = (GV_SearchResult *)malloc(max_results * sizeof(GV_SearchResult));
    if (tmp == NULL) {
        return -1;
    }

    int n = gv_hnsw_search(index_ptr, query, max_results, tmp, distance_type,
                           filter_key, filter_value);
    if (n <= 0) {
        free(tmp);
        return n;
    }

    size_t out = 0;
    for (int i = 0; i < n && out < max_results; ++i) {
        if (tmp[i].distance <= radius) {
            results[out++] = tmp[i];
        }
    }

    free(tmp);
    return (int)out;
}

int gv_hnsw_update(void *index_ptr, size_t node_index, const float *new_data, size_t dimension) {
    if (index_ptr == NULL || new_data == NULL) {
        return -1;
    }

    GV_HNSWIndex *index = (GV_HNSWIndex *)index_ptr;
    if (node_index >= index->count || dimension != index->dimension) {
        return -1;
    }

    GV_HNSWNode *node = index->nodes[node_index];
    if (node == NULL || node->deleted != 0) {
        return -1;
    }

    /* Update vector data in SoA storage */
    if (gv_soa_storage_update_data(index->soa_storage, node->vector_index, new_data) != 0) {
        return -1;
    }

    /* Rebuild binary quantization if enabled */
    if (index->use_binary_quant) {
        if (node->binary_vector != NULL) {
            gv_binary_vector_destroy(node->binary_vector);
        }
        node->binary_vector = gv_binary_quantize(new_data, dimension);
        if (node->binary_vector == NULL) {
            return -1;
        }
    }

    return 0;
}
