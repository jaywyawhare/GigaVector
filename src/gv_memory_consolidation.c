#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "gigavector/gv_memory_consolidation.h"
#include "gigavector/gv_memory_layer.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_metadata.h"
#include "gigavector/gv_soa_storage.h"

static float calculate_similarity_from_distance(float distance, GV_DistanceType dist_type) {
    switch (dist_type) {
        case GV_DISTANCE_COSINE:
            return 1.0f - distance;
        case GV_DISTANCE_EUCLIDEAN:
            return 1.0f / (1.0f + distance);
        case GV_DISTANCE_DOT_PRODUCT:
            return distance;
        default:
            return 1.0f - (distance / 2.0f);
    }
}

int gv_memory_find_similar(GV_MemoryLayer *layer, double threshold,
                           GV_MemoryPair *pairs, size_t max_pairs,
                           size_t *actual_count) {
    if (layer == NULL || pairs == NULL || actual_count == NULL) {
        return -1;
    }
    
    *actual_count = 0;
    
    size_t db_count = layer->db->count;
    if (db_count < 2 || layer->db->soa_storage == NULL) {
        return 0;
    }
    
    GV_SearchResult *search_results = (GV_SearchResult *)malloc(100 * sizeof(GV_SearchResult));
    if (search_results == NULL) {
        return -1;
    }
    
    size_t pair_idx = 0;
    
    for (size_t i = 0; i < db_count && pair_idx < max_pairs; i++) {
        if (gv_soa_storage_is_deleted(layer->db->soa_storage, i) != 0) {
            continue;
        }
        
        const float *vec_data = gv_soa_storage_get_data(layer->db->soa_storage, i);
        if (vec_data == NULL) {
            continue;
        }
        
        GV_Metadata *meta = gv_soa_storage_get_metadata(layer->db->soa_storage, i);
        const char *mem_id_1 = NULL;
        if (meta != NULL) {
            GV_Vector tmp_vec;
            tmp_vec.dimension = 0;
            tmp_vec.data = NULL;
            tmp_vec.metadata = meta;
            mem_id_1 = gv_vector_get_metadata(&tmp_vec, "memory_id");
        }
        if (mem_id_1 == NULL) {
            continue;
        }
        
        int count = gv_db_search(layer->db, vec_data, 10, search_results, GV_DISTANCE_COSINE);
        if (count < 0) {
            continue;
        }
        
        for (int j = 0; j < count && pair_idx < max_pairs; j++) {
            if (search_results[j].vector == NULL) {
                continue;
            }
            
            const char *mem_id_2 = gv_vector_get_metadata(search_results[j].vector, "memory_id");
            if (mem_id_2 == NULL || strcmp(mem_id_1, mem_id_2) == 0) {
                continue;
            }
            
            float similarity = calculate_similarity_from_distance(search_results[j].distance,
                                                                   GV_DISTANCE_COSINE);
            
            if (similarity >= (float)threshold) {
                int duplicate = 0;
                for (size_t k = 0; k < pair_idx; k++) {
                    if ((strcmp(pairs[k].memory_id_1, mem_id_1) == 0 &&
                         strcmp(pairs[k].memory_id_2, mem_id_2) == 0) ||
                        (strcmp(pairs[k].memory_id_1, mem_id_2) == 0 &&
                         strcmp(pairs[k].memory_id_2, mem_id_1) == 0)) {
                        duplicate = 1;
                        break;
                    }
                }
                
                if (!duplicate) {
                    pairs[pair_idx].memory_id_1 = strdup(mem_id_1);
                    pairs[pair_idx].memory_id_2 = strdup(mem_id_2);
                    pairs[pair_idx].similarity = similarity;
                    pair_idx++;
                }
            }
        }
    }
    
    free(search_results);
    *actual_count = pair_idx;
    return 0;
}

char *gv_memory_consolidate_pair(GV_MemoryLayer *layer,
                                  const char *memory_id_1,
                                  const char *memory_id_2,
                                  GV_ConsolidationStrategy strategy) {
    if (layer == NULL || memory_id_1 == NULL || memory_id_2 == NULL) {
        return NULL;
    }
    
    switch (strategy) {
        case GV_CONSOLIDATION_MERGE:
            return gv_memory_merge(layer, memory_id_1, memory_id_2);
        case GV_CONSOLIDATION_UPDATE:
            return gv_memory_update_from_new(layer, memory_id_1, memory_id_2) == 0 ?
                strdup(memory_id_1) : NULL;
        case GV_CONSOLIDATION_LINK:
            return gv_memory_link(layer, memory_id_1, memory_id_2) == 0 ?
                strdup(memory_id_1) : NULL;
        case GV_CONSOLIDATION_ARCHIVE:
            return gv_memory_archive(layer, memory_id_2) == 0 ?
                strdup(memory_id_1) : NULL;
        default:
            return NULL;
    }
}

char *gv_memory_merge(GV_MemoryLayer *layer, const char *memory_id_1,
                      const char *memory_id_2) {
    if (layer == NULL || memory_id_1 == NULL || memory_id_2 == NULL) {
        return NULL;
    }
    
    GV_MemoryResult mem1, mem2;
    if (gv_memory_get(layer, memory_id_1, &mem1) != 0 ||
        gv_memory_get(layer, memory_id_2, &mem2) != 0) {
        return NULL;
    }
    
    size_t merged_len = 0;
    if (mem1.content) merged_len += strlen(mem1.content);
    if (mem2.content) merged_len += strlen(mem2.content);
    merged_len += 10;
    
    char *merged_content = (char *)malloc(merged_len);
    if (merged_content == NULL) {
        gv_memory_result_free(&mem1);
        gv_memory_result_free(&mem2);
        return NULL;
    }
    
    merged_content[0] = '\0';
    if (mem1.content) {
        strcat(merged_content, mem1.content);
    }
    if (mem2.content) {
        if (mem1.content) {
            strcat(merged_content, ". ");
        }
        strcat(merged_content, mem2.content);
    }
    
    float *merged_embedding = (float *)malloc(layer->db->dimension * sizeof(float));
    if (merged_embedding == NULL) {
        free(merged_content);
        gv_memory_result_free(&mem1);
        gv_memory_result_free(&mem2);
        return NULL;
    }
    
    for (size_t i = 0; i < layer->db->dimension; i++) {
        merged_embedding[i] = 0.5f;
    }
    
    GV_MemoryMetadata merged_meta;
    memset(&merged_meta, 0, sizeof(merged_meta));
    merged_meta.memory_type = mem1.metadata ? mem1.metadata->memory_type : GV_MEMORY_TYPE_FACT;
    merged_meta.importance_score = mem1.metadata && mem2.metadata ?
        (mem1.metadata->importance_score + mem2.metadata->importance_score) / 2.0 : 0.5;
    merged_meta.timestamp = time(NULL);
    merged_meta.consolidated = 1;
    
    char *new_id = gv_memory_add(layer, merged_content, merged_embedding, &merged_meta);
    
    free(merged_embedding);
    free(merged_content);
    gv_memory_result_free(&mem1);
    gv_memory_result_free(&mem2);
    gv_memory_metadata_free(&merged_meta);
    
    if (new_id != NULL) {
        gv_memory_delete(layer, memory_id_1);
        gv_memory_delete(layer, memory_id_2);
    }
    
    return new_id;
}

int gv_memory_update_from_new(GV_MemoryLayer *layer,
                               const char *existing_memory_id,
                               const char *new_memory_id) {
    if (layer == NULL || existing_memory_id == NULL || new_memory_id == NULL) {
        return -1;
    }
    
    GV_MemoryResult existing, new_mem;
    if (gv_memory_get(layer, existing_memory_id, &existing) != 0 ||
        gv_memory_get(layer, new_memory_id, &new_mem) != 0) {
        return -1;
    }
    
    if (new_mem.metadata) {
        new_mem.metadata->timestamp = time(NULL);
        gv_memory_update(layer, existing_memory_id, NULL, new_mem.metadata);
    }
    
    gv_memory_delete(layer, new_memory_id);
    
    gv_memory_result_free(&existing);
    gv_memory_result_free(&new_mem);
    
    return 0;
}

int gv_memory_link(GV_MemoryLayer *layer, const char *memory_id_1,
                    const char *memory_id_2) {
    if (layer == NULL || memory_id_1 == NULL || memory_id_2 == NULL) {
        return -1;
    }
    
    GV_MemoryResult mem1, mem2;
    if (gv_memory_get(layer, memory_id_1, &mem1) != 0 ||
        gv_memory_get(layer, memory_id_2, &mem2) != 0) {
        return -1;
    }
    
    if (mem1.metadata == NULL) {
        gv_memory_result_free(&mem1);
        gv_memory_result_free(&mem2);
        return -1;
    }
    
    size_t new_count = mem1.metadata->related_count + 1;
    char **new_related = (char **)realloc(mem1.metadata->related_memory_ids,
                                            new_count * sizeof(char *));
    if (new_related == NULL) {
        gv_memory_result_free(&mem1);
        gv_memory_result_free(&mem2);
        return -1;
    }
    
    new_related[mem1.metadata->related_count] = strdup(memory_id_2);
    mem1.metadata->related_memory_ids = new_related;
    mem1.metadata->related_count = new_count;
    
    int result = gv_memory_update(layer, memory_id_1, NULL, mem1.metadata);
    
    gv_memory_result_free(&mem1);
    gv_memory_result_free(&mem2);
    
    return result;
}

int gv_memory_archive(GV_MemoryLayer *layer, const char *memory_id) {
    if (layer == NULL || memory_id == NULL) {
        return -1;
    }
    
    GV_MemoryResult mem;
    if (gv_memory_get(layer, memory_id, &mem) != 0) {
        return -1;
    }
    
    if (mem.metadata == NULL) {
        gv_memory_result_free(&mem);
        return -1;
    }
    
    gv_memory_result_free(&mem);
    return 0;
}

void gv_memory_pair_free(GV_MemoryPair *pair) {
    if (pair == NULL) {
        return;
    }
    
    free(pair->memory_id_1);
    free(pair->memory_id_2);
    memset(pair, 0, sizeof(GV_MemoryPair));
}

void gv_memory_pairs_free(GV_MemoryPair *pairs, size_t count) {
    if (pairs == NULL) {
        return;
    }
    
    for (size_t i = 0; i < count; i++) {
        gv_memory_pair_free(&pairs[i]);
    }
}

