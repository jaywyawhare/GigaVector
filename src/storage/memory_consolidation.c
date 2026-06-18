#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "core/utils.h"

#include "storage/memory_consolidation.h"
#include "storage/memory_layer.h"
#include "storage/database.h"
#include "schema/metadata.h"
#include "storage/soa_storage.h"

static void l2_normalize(float *vector, size_t dimension) {
    if (vector == NULL || dimension == 0) {
        return;
    }
    double sum = 0.0;
    for (size_t i = 0; i < dimension; i++) {
        sum += (double)vector[i] * (double)vector[i];
    }
    if (sum <= 0.0) {
        return;
    }
    float norm = (float)sqrt(sum);
    for (size_t i = 0; i < dimension; i++) {
        vector[i] /= norm;
    }
}

static char *merge_provenance_source(const char *source_a, const char *source_b) {
    const char *a = source_a ? source_a : "";
    const char *b = source_b ? source_b : "";
    int a_swap = strncmp(a, "swap:", 5) == 0;
    int b_swap = strncmp(b, "swap:", 5) == 0;

    if (a_swap && b_swap) {
        size_t len = strlen(a) + strlen(b) + 2;
        char *combined = (char *)gv_alloc(len);
        if (combined == NULL) {
            return NULL;
        }
        snprintf(combined, len, "%s;%s", a, b);
        return combined;
    }
    if (a_swap) {
        return gv_dup_cstr(a);
    }
    if (b_swap) {
        return gv_dup_cstr(b);
    }
    if (source_a && source_a[0] != '\0') {
        return gv_dup_cstr(source_a);
    }
    if (source_b && source_b[0] != '\0') {
        return gv_dup_cstr(source_b);
    }
    return gv_dup_cstr("merged");
}

static void copy_parent_links(GV_MemoryLayer *layer,
                              const char *new_id,
                              const GV_MemoryMetadata *meta,
                              const char *skip_id) {
    if (layer == NULL || new_id == NULL || meta == NULL || meta->links == NULL) {
        return;
    }

    for (size_t i = 0; i < meta->link_count; i++) {
        const char *target = meta->links[i].target_memory_id;
        if (target == NULL || strcmp(target, skip_id) == 0) {
            continue;
        }
        memory_link_create(layer, new_id, target, meta->links[i].link_type,
                           meta->links[i].strength, meta->links[i].reason);
    }
}

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

int memory_find_similar(GV_MemoryLayer *layer, double threshold,
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
    
    GV_SearchResult *search_results = (GV_SearchResult *)gv_alloc(100 * sizeof(GV_SearchResult));
    if (search_results == NULL) {
        return -1;
    }
    
    size_t pair_idx = 0;
    
    for (size_t i = 0; i < db_count && pair_idx < max_pairs; i++) {
        if (soa_storage_is_deleted(layer->db->soa_storage, i) != 0) {
            continue;
        }
        
        const float *vec_data = soa_storage_get_data(layer->db->soa_storage, i);
        if (vec_data == NULL) {
            continue;
        }
        
        GV_Metadata *meta = soa_storage_get_metadata(layer->db->soa_storage, i);
        const char *mem_id_1 = NULL;
        if (meta != NULL) {
            GV_Vector tmp_vec;
            tmp_vec.dimension = 0;
            tmp_vec.data = NULL;
            tmp_vec.metadata = meta;
            mem_id_1 = vector_get_metadata(&tmp_vec, "memory_id");
        }
        if (mem_id_1 == NULL) {
            continue;
        }
        
        int count = db_search(layer->db, vec_data, 10, search_results, GV_DISTANCE_COSINE);
        if (count < 0) {
            continue;
        }
        
        for (int j = 0; j < count && pair_idx < max_pairs; j++) {
            if (search_results[j].vector == NULL) {
                continue;
            }
            
            const char *mem_id_2 = vector_get_metadata(search_results[j].vector, "memory_id");
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
                    pairs[pair_idx].memory_id_1 = gv_dup_cstr(mem_id_1);
                    pairs[pair_idx].memory_id_2 = gv_dup_cstr(mem_id_2);
                    pairs[pair_idx].similarity = similarity;
                    pair_idx++;
                }
            }
        }
    }
    
    gv_free(search_results);
    *actual_count = pair_idx;
    return 0;
}

char *memory_consolidate_pair(GV_MemoryLayer *layer,
                                  const char *memory_id_1,
                                  const char *memory_id_2,
                                  GV_ConsolidationStrategy strategy) {
    if (layer == NULL || memory_id_1 == NULL || memory_id_2 == NULL) {
        return NULL;
    }
    
    switch (strategy) {
        case GV_CONSOLIDATION_MERGE:
            return memory_merge(layer, memory_id_1, memory_id_2);
        case GV_CONSOLIDATION_UPDATE:
            return memory_update_from_new(layer, memory_id_1, memory_id_2) == 0 ?
                gv_dup_cstr(memory_id_1) : NULL;
        case GV_CONSOLIDATION_LINK:
            return memory_link(layer, memory_id_1, memory_id_2) == 0 ?
                gv_dup_cstr(memory_id_1) : NULL;
        case GV_CONSOLIDATION_ARCHIVE:
            return memory_archive(layer, memory_id_2) == 0 ?
                gv_dup_cstr(memory_id_1) : NULL;
        default:
            return NULL;
    }
}

char *memory_merge(GV_MemoryLayer *layer, const char *memory_id_1,
                      const char *memory_id_2) {
    if (layer == NULL || memory_id_1 == NULL || memory_id_2 == NULL) {
        return NULL;
    }
    
    GV_MemoryResult mem1, mem2;
    if (memory_get(layer, memory_id_1, &mem1) != 0 ||
        memory_get(layer, memory_id_2, &mem2) != 0) {
        return NULL;
    }
    
    size_t merged_len = 0;
    if (mem1.content) merged_len += strlen(mem1.content);
    if (mem2.content) merged_len += strlen(mem2.content);
    merged_len += 10;
    
    char *merged_content = (char *)gv_alloc(merged_len);
    if (merged_content == NULL) {
        memory_result_free(&mem1);
        memory_result_free(&mem2);
        return NULL;
    }
    
    size_t pos = 0;
    if (mem1.content) {
        size_t l = strlen(mem1.content);
        memcpy(merged_content + pos, mem1.content, l);
        pos += l;
    }
    if (mem2.content) {
        if (mem1.content) {
            memcpy(merged_content + pos, ". ", 2);
            pos += 2;
        }
        size_t l = strlen(mem2.content);
        memcpy(merged_content + pos, mem2.content, l);
        pos += l;
    }
    merged_content[pos] = '\0';

    size_t dimension = layer->db->dimension;
    float *merged_embedding = (float *)gv_alloc(dimension * sizeof(float));
    if (merged_embedding == NULL) {
        gv_free(merged_content);
        memory_result_free(&mem1);
        memory_result_free(&mem2);
        return NULL;
    }

    float *emb1 = (float *)gv_alloc(dimension * sizeof(float));
    float *emb2 = (float *)gv_alloc(dimension * sizeof(float));
    if (emb1 == NULL || emb2 == NULL) {
        gv_free(emb1);
        gv_free(emb2);
        gv_free(merged_embedding);
        gv_free(merged_content);
        memory_result_free(&mem1);
        memory_result_free(&mem2);
        return NULL;
    }

    if (memory_get_embedding(layer, memory_id_1, emb1, dimension) != 0 ||
        memory_get_embedding(layer, memory_id_2, emb2, dimension) != 0) {
        gv_free(emb1);
        gv_free(emb2);
        gv_free(merged_embedding);
        gv_free(merged_content);
        memory_result_free(&mem1);
        memory_result_free(&mem2);
        return NULL;
    }

    for (size_t i = 0; i < dimension; i++) {
        merged_embedding[i] = 0.5f * (emb1[i] + emb2[i]);
    }
    l2_normalize(merged_embedding, dimension);
    gv_free(emb1);
    gv_free(emb2);

    GV_MemoryMetadata merged_meta;
    memset(&merged_meta, 0, sizeof(merged_meta));
    merged_meta.memory_type = mem1.metadata ? mem1.metadata->memory_type : GV_MEMORY_TYPE_FACT;
    merged_meta.importance_score = mem1.metadata && mem2.metadata ?
        (mem1.metadata->importance_score + mem2.metadata->importance_score) / 2.0 : 0.5;
    merged_meta.timestamp = time(NULL);
    merged_meta.consolidated = 1;
    merged_meta.source = merge_provenance_source(
        mem1.metadata ? mem1.metadata->source : NULL,
        mem2.metadata ? mem2.metadata->source : NULL);

    char *new_id = memory_add(layer, merged_content, merged_embedding, &merged_meta, NULL);

    gv_free(merged_meta.source);
    gv_free(merged_embedding);
    gv_free(merged_content);

    if (new_id != NULL) {
        copy_parent_links(layer, new_id, mem1.metadata, memory_id_2);
        copy_parent_links(layer, new_id, mem2.metadata, memory_id_1);
        memory_delete(layer, memory_id_1);
        memory_delete(layer, memory_id_2);
    }

    memory_result_free(&mem1);
    memory_result_free(&mem2);
    merged_meta.source = NULL;
    memory_metadata_free(&merged_meta);

    return new_id;
}

int memory_update_from_new(GV_MemoryLayer *layer,
                               const char *existing_memory_id,
                               const char *new_memory_id) {
    if (layer == NULL || existing_memory_id == NULL || new_memory_id == NULL) {
        return -1;
    }
    
    GV_MemoryResult existing, new_mem;
    if (memory_get(layer, existing_memory_id, &existing) != 0 ||
        memory_get(layer, new_memory_id, &new_mem) != 0) {
        return -1;
    }
    
    if (new_mem.metadata) {
        new_mem.metadata->timestamp = time(NULL);
        memory_update(layer, existing_memory_id, NULL, new_mem.metadata);
    }
    
    memory_delete(layer, new_memory_id);
    
    memory_result_free(&existing);
    memory_result_free(&new_mem);
    
    return 0;
}

int memory_link(GV_MemoryLayer *layer, const char *memory_id_1,
                    const char *memory_id_2) {
    if (layer == NULL || memory_id_1 == NULL || memory_id_2 == NULL) {
        return -1;
    }

    return memory_link_create(layer, memory_id_1, memory_id_2, GV_LINK_SIMILAR,
                              0.9f, "consolidated related");
}

int memory_archive(GV_MemoryLayer *layer, const char *memory_id) {
    if (layer == NULL || memory_id == NULL) {
        return -1;
    }
    
    GV_MemoryResult mem;
    if (memory_get(layer, memory_id, &mem) != 0) {
        return -1;
    }
    
    if (mem.metadata == NULL) {
        memory_result_free(&mem);
        return -1;
    }
    
    memory_result_free(&mem);
    return 0;
}

void memory_pair_free(GV_MemoryPair *pair) {
    if (pair == NULL) {
        return;
    }
    
    gv_free(pair->memory_id_1);
    gv_free(pair->memory_id_2);
    memset(pair, 0, sizeof(GV_MemoryPair));
}

void memory_pairs_free(GV_MemoryPair *pairs, size_t count) {
    if (pairs == NULL) {
        return;
    }
    
    for (size_t i = 0; i < count; i++) {
        memory_pair_free(&pairs[i]);
    }
}

