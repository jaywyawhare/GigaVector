#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <pthread.h>
#include <stdint.h>
#include <math.h>

#include "gigavector/gv_memory_layer.h"
#include "gigavector/gv_memory_extraction.h"
#include "gigavector/gv_memory_consolidation.h"
#include "gigavector/gv_importance.h"
#include "gigavector/gv_metadata.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_soa_storage.h"
#include "gigavector/gv_llm.h"
#include "gigavector/gv_context_graph.h"

#define MEMORY_ID_PREFIX "mem_"
#define MEMORY_ID_BUFFER_SIZE 64

static char *generate_memory_id(uint64_t counter) {
    char *id = (char *)malloc(MEMORY_ID_BUFFER_SIZE);
    if (id == NULL) {
        return NULL;
    }
    snprintf(id, MEMORY_ID_BUFFER_SIZE, "%s%lu", MEMORY_ID_PREFIX, (unsigned long)counter);
    return id;
}

static int serialize_related_ids(char **ids, size_t count, char **out) {
    if (count == 0) {
        *out = NULL;
        return 0;
    }
    
    size_t total_len = 0;
    for (size_t i = 0; i < count; i++) {
        total_len += strlen(ids[i]) + 1;
    }
    
    char *result = (char *)malloc(total_len);
    if (result == NULL) {
        return -1;
    }
    
    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        size_t len = strlen(ids[i]);
        memcpy(result + pos, ids[i], len);
        pos += len;
        if (i < count - 1) {
            result[pos++] = ',';
        }
    }
    result[pos] = '\0';
    *out = result;
    return 0;
}

static int deserialize_related_ids(const char *serialized, char ***out, size_t *count) {
    if (serialized == NULL || strlen(serialized) == 0) {
        *out = NULL;
        *count = 0;
        return 0;
    }
    
    size_t len = strlen(serialized);
    size_t id_count = 1;
    for (size_t i = 0; i < len; i++) {
        if (serialized[i] == ',') {
            id_count++;
        }
    }
    
    char **ids = (char **)malloc(id_count * sizeof(char *));
    if (ids == NULL) {
        return -1;
    }
    
    const char *start = serialized;
    size_t idx = 0;
    for (size_t i = 0; i <= len; i++) {
        if (serialized[i] == ',' || serialized[i] == '\0') {
            size_t id_len = &serialized[i] - start;
            ids[idx] = (char *)malloc(id_len + 1);
            if (ids[idx] == NULL) {
                for (size_t j = 0; j < idx; j++) {
                    free(ids[j]);
                }
                free(ids);
                return -1;
            }
            memcpy(ids[idx], start, id_len);
            ids[idx][id_len] = '\0';
            idx++;
            start = &serialized[i + 1];
        }
    }
    
    *out = ids;
    *count = id_count;
    return 0;
}

/**
 * @brief Serialize memory links to JSON string.
 * Format: [{"target":"id","type":0,"strength":0.8,"created":123456,"reason":"text"},...]
 */
static int serialize_links(const GV_MemoryLink *links, size_t count, char **out) {
    if (count == 0 || links == NULL) {
        *out = NULL;
        return 0;
    }

    /* Estimate buffer size: ~200 bytes per link should be sufficient */
    size_t buffer_size = count * 200 + 3;
    char *result = (char *)malloc(buffer_size);
    if (result == NULL) {
        return -1;
    }

    size_t pos = 0;
    result[pos++] = '[';

    for (size_t i = 0; i < count; i++) {
        if (i > 0) {
            result[pos++] = ',';
        }

        int written = snprintf(result + pos, buffer_size - pos,
            "{\"target\":\"%s\",\"type\":%d,\"strength\":%.4f,\"created\":%ld",
            links[i].target_memory_id ? links[i].target_memory_id : "",
            (int)links[i].link_type,
            links[i].strength,
            (long)links[i].created_at);

        if (written < 0 || (size_t)written >= buffer_size - pos) {
            free(result);
            return -1;
        }
        pos += written;

        /* Add reason if present */
        if (links[i].reason != NULL) {
            written = snprintf(result + pos, buffer_size - pos,
                ",\"reason\":\"%s\"", links[i].reason);
            if (written < 0 || (size_t)written >= buffer_size - pos) {
                free(result);
                return -1;
            }
            pos += written;
        }

        result[pos++] = '}';
    }

    result[pos++] = ']';
    result[pos] = '\0';

    *out = result;
    return 0;
}

/**
 * @brief Deserialize memory links from JSON string.
 */
static int deserialize_links(const char *serialized, GV_MemoryLink **out, size_t *count) {
    if (serialized == NULL || strlen(serialized) == 0 || serialized[0] != '[') {
        *out = NULL;
        *count = 0;
        return 0;
    }

    /* Count links by counting '{' characters */
    size_t link_count = 0;
    for (const char *p = serialized; *p; p++) {
        if (*p == '{') link_count++;
    }

    if (link_count == 0) {
        *out = NULL;
        *count = 0;
        return 0;
    }

    GV_MemoryLink *links = (GV_MemoryLink *)calloc(link_count, sizeof(GV_MemoryLink));
    if (links == NULL) {
        return -1;
    }

    /* Simple JSON parsing - find each object and extract fields */
    const char *p = serialized + 1;  /* Skip '[' */
    size_t idx = 0;

    while (*p && idx < link_count) {
        /* Find start of object */
        while (*p && *p != '{') p++;
        if (*p != '{') break;
        p++;

        /* Parse fields within object */
        char target[MEMORY_ID_BUFFER_SIZE] = {0};
        int type = 0;
        float strength = 0.5f;
        long created = 0;
        char reason[256] = {0};

        while (*p && *p != '}') {
            /* Skip whitespace and commas */
            while (*p && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t')) p++;
            if (*p == '}') break;

            /* Parse key */
            if (*p == '"') {
                p++;
                const char *key_start = p;
                while (*p && *p != '"') p++;
                size_t key_len = p - key_start;
                p++;  /* Skip closing quote */

                /* Skip colon */
                while (*p && *p == ':') p++;
                while (*p && *p == ' ') p++;

                /* Parse value */
                if (strncmp(key_start, "target", key_len) == 0 && key_len == 6) {
                    if (*p == '"') {
                        p++;
                        const char *val_start = p;
                        while (*p && *p != '"') p++;
                        size_t val_len = p - val_start;
                        if (val_len < sizeof(target)) {
                            memcpy(target, val_start, val_len);
                            target[val_len] = '\0';
                        }
                        p++;
                    }
                } else if (strncmp(key_start, "type", key_len) == 0 && key_len == 4) {
                    type = (int)strtol(p, (char **)&p, 10);
                } else if (strncmp(key_start, "strength", key_len) == 0 && key_len == 8) {
                    strength = strtof(p, (char **)&p);
                } else if (strncmp(key_start, "created", key_len) == 0 && key_len == 7) {
                    created = strtol(p, (char **)&p, 10);
                } else if (strncmp(key_start, "reason", key_len) == 0 && key_len == 6) {
                    if (*p == '"') {
                        p++;
                        const char *val_start = p;
                        while (*p && *p != '"') p++;
                        size_t val_len = p - val_start;
                        if (val_len < sizeof(reason)) {
                            memcpy(reason, val_start, val_len);
                            reason[val_len] = '\0';
                        }
                        p++;
                    }
                } else {
                    /* Skip unknown field value */
                    if (*p == '"') {
                        p++;
                        while (*p && *p != '"') p++;
                        if (*p == '"') p++;
                    } else {
                        while (*p && *p != ',' && *p != '}') p++;
                    }
                }
            }
        }

        /* Store parsed link */
        links[idx].target_memory_id = strlen(target) > 0 ? strdup(target) : NULL;
        links[idx].link_type = (GV_MemoryLinkType)type;
        links[idx].strength = strength;
        links[idx].created_at = (time_t)created;
        links[idx].reason = strlen(reason) > 0 ? strdup(reason) : NULL;
        idx++;

        if (*p == '}') p++;
    }

    *out = links;
    *count = idx;
    return 0;
}

static GV_Metadata *create_memory_metadata(const GV_MemoryMetadata *meta) {
    if (meta == NULL) {
        return NULL;
    }
    
    GV_Metadata *head = NULL;
    GV_Metadata *tail = NULL;
    
    if (meta->memory_id) {
        GV_Metadata *m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
        if (m == NULL) goto error;
        m->key = strdup("memory_id");
        m->value = strdup(meta->memory_id);
        m->next = NULL;
        head = tail = m;
    }
    
    char type_str[32];
    snprintf(type_str, sizeof(type_str), "%d", (int)meta->memory_type);
    GV_Metadata *m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
    if (m == NULL) goto error;
    m->key = strdup("memory_type");
    m->value = strdup(type_str);
    m->next = NULL;
    if (head == NULL) {
        head = tail = m;
    } else {
        tail->next = m;
        tail = m;
    }
    
    if (meta->source) {
        m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
        if (m == NULL) goto error;
        m->key = strdup("source");
        m->value = strdup(meta->source);
        m->next = NULL;
        tail->next = m;
        tail = m;
    }
    
    char timestamp_str[64];
    snprintf(timestamp_str, sizeof(timestamp_str), "%ld", (long)meta->timestamp);
    m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
    if (m == NULL) goto error;
    m->key = strdup("timestamp");
    m->value = strdup(timestamp_str);
    m->next = NULL;
    tail->next = m;
    tail = m;
    
    char importance_str[32];
    snprintf(importance_str, sizeof(importance_str), "%.6f", meta->importance_score);
    m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
    if (m == NULL) goto error;
    m->key = strdup("importance_score");
    m->value = strdup(importance_str);
    m->next = NULL;
    tail->next = m;
    tail = m;
    
    if (meta->extraction_metadata) {
        m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
        if (m == NULL) goto error;
        m->key = strdup("extraction_metadata");
        m->value = strdup(meta->extraction_metadata);
        m->next = NULL;
        tail->next = m;
        tail = m;
    }
    
    char consolidated_str[8];
    snprintf(consolidated_str, sizeof(consolidated_str), "%d", meta->consolidated);
    m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
    if (m == NULL) goto error;
    m->key = strdup("consolidated");
    m->value = strdup(consolidated_str);
    m->next = NULL;
    tail->next = m;
    tail = m;
    
    if (meta->related_count > 0) {
        char *related_str = NULL;
        if (serialize_related_ids(meta->related_memory_ids, meta->related_count, &related_str) == 0 && related_str) {
            m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
            if (m == NULL) {
                free(related_str);
                goto error;
            }
            m->key = strdup("related_memories");
            m->value = related_str;
            m->next = NULL;
            tail->next = m;
            tail = m;
        }
    }

    /* Serialize memory links */
    if (meta->link_count > 0 && meta->links != NULL) {
        char *links_str = NULL;
        if (serialize_links(meta->links, meta->link_count, &links_str) == 0 && links_str) {
            m = (GV_Metadata *)malloc(sizeof(GV_Metadata));
            if (m == NULL) {
                free(links_str);
                goto error;
            }
            m->key = strdup("memory_links");
            m->value = links_str;
            m->next = NULL;
            tail->next = m;
            tail = m;
        }
    }

    return head;
    
error:
    gv_metadata_free(head);
    return NULL;
}

static int parse_memory_metadata(const GV_Metadata *meta_list, GV_MemoryMetadata *out) {
    if (meta_list == NULL || out == NULL) {
        return -1;
    }
    
    memset(out, 0, sizeof(GV_MemoryMetadata));
    
    const GV_Metadata *current = meta_list;
    while (current != NULL) {
        if (strcmp(current->key, "memory_id") == 0) {
            out->memory_id = strdup(current->value);
        } else if (strcmp(current->key, "memory_type") == 0) {
            out->memory_type = (GV_MemoryType)atoi(current->value);
        } else if (strcmp(current->key, "source") == 0) {
            out->source = strdup(current->value);
        } else if (strcmp(current->key, "timestamp") == 0) {
            out->timestamp = (time_t)atol(current->value);
        } else if (strcmp(current->key, "importance_score") == 0) {
            out->importance_score = atof(current->value);
        } else if (strcmp(current->key, "extraction_metadata") == 0) {
            out->extraction_metadata = strdup(current->value);
        } else if (strcmp(current->key, "consolidated") == 0) {
            out->consolidated = atoi(current->value);
        } else if (strcmp(current->key, "related_memories") == 0) {
            deserialize_related_ids(current->value, &out->related_memory_ids, &out->related_count);
        } else if (strcmp(current->key, "memory_links") == 0) {
            deserialize_links(current->value, &out->links, &out->link_count);
        }
        current = current->next;
    }

    return 0;
}

GV_MemoryLayerConfig gv_memory_layer_config_default(void) {
    GV_MemoryLayerConfig config;
    config.extraction_threshold = 0.5;
    config.consolidation_threshold = 0.85;
    config.default_strategy = GV_CONSOLIDATION_MERGE;
    config.enable_temporal_weighting = 1;
    config.enable_relationship_retrieval = 1;
    config.max_related_memories = 5;
    config.llm_config = NULL;
    config.use_llm_extraction = 1;  // LLM extraction enabled by default
    config.use_llm_consolidation = 0;  // Consolidation via similarity by default
    config.context_graph_config = NULL;
    config.enable_context_graph = 0;  // Context graph disabled by default
    return config;
}

GV_MemoryLayer *gv_memory_layer_create(GV_Database *db, const GV_MemoryLayerConfig *config) {
    if (db == NULL) {
        return NULL;
    }
    
    GV_MemoryLayer *layer = (GV_MemoryLayer *)malloc(sizeof(GV_MemoryLayer));
    if (layer == NULL) {
        return NULL;
    }
    
    layer->db = db;
    if (config != NULL) {
        layer->config = *config;
    } else {
        layer->config = gv_memory_layer_config_default();
    }
    layer->next_memory_id = 1;
    layer->llm = NULL;
    layer->context_graph = NULL;
    
    // Create LLM instance if config provided
    if (layer->config.llm_config != NULL) {
        layer->llm = gv_llm_create((GV_LLMConfig *)layer->config.llm_config);
    }
    
    // Create context graph if enabled
    if (layer->config.enable_context_graph && layer->config.context_graph_config != NULL) {
        GV_ContextGraphConfig *graph_config = (GV_ContextGraphConfig *)layer->config.context_graph_config;
        graph_config->llm = layer->llm;  // Share LLM instance
        layer->context_graph = gv_context_graph_create(graph_config);
    }
    
    if (pthread_mutex_init(&layer->mutex, NULL) != 0) {
        if (layer->llm) {
            gv_llm_destroy((GV_LLM *)layer->llm);
        }
        if (layer->context_graph) {
            gv_context_graph_destroy((GV_ContextGraph *)layer->context_graph);
        }
        free(layer);
        return NULL;
    }
    
    return layer;
}

void gv_memory_layer_destroy(GV_MemoryLayer *layer) {
    if (layer == NULL) {
        return;
    }
    
    if (layer->llm) {
        gv_llm_destroy((GV_LLM *)layer->llm);
    }
    
    if (layer->context_graph) {
        gv_context_graph_destroy((GV_ContextGraph *)layer->context_graph);
    }
    
    pthread_mutex_destroy(&layer->mutex);
    free(layer);
}

char *gv_memory_add(GV_MemoryLayer *layer, const char *content,
                     const float *embedding, GV_MemoryMetadata *metadata) {
    if (layer == NULL || content == NULL || embedding == NULL) {
        return NULL;
    }
    
    pthread_mutex_lock(&layer->mutex);
    
    char *memory_id = generate_memory_id(layer->next_memory_id++);
    if (memory_id == NULL) {
        pthread_mutex_unlock(&layer->mutex);
        return NULL;
    }
    
    GV_MemoryMetadata meta;
    if (metadata != NULL) {
        meta = *metadata;
    } else {
        memset(&meta, 0, sizeof(meta));
        meta.memory_type = GV_MEMORY_TYPE_FACT;
        meta.timestamp = time(NULL);
        meta.importance_score = 0.5;
        meta.consolidated = 0;
    }
    
    if (meta.memory_id == NULL) {
        meta.memory_id = strdup(memory_id);
    }
    if (meta.timestamp == 0) {
        meta.timestamp = time(NULL);
    }
    
    GV_Metadata *meta_list = create_memory_metadata(&meta);
    if (meta_list == NULL) {
        free(memory_id);
        pthread_mutex_unlock(&layer->mutex);
        return NULL;
    }
    
    int result = gv_db_add_vector_with_rich_metadata(
        layer->db, embedding, layer->db->dimension,
        (const char *const[]){ "content", "memory_id" },
        (const char *const[]){ content, memory_id },
        2
    );
    
    if (result != 0) {
        gv_metadata_free(meta_list);
        free(memory_id);
        pthread_mutex_unlock(&layer->mutex);
        return NULL;
    }
    
    pthread_mutex_unlock(&layer->mutex);
    return memory_id;
}

int gv_memory_search(GV_MemoryLayer *layer, const float *query_embedding,
                      size_t k, GV_MemoryResult *results,
                      GV_DistanceType distance_type) {
    if (layer == NULL || query_embedding == NULL || results == NULL) {
        return -1;
    }

    /* Fetch more results than needed for reranking */
    size_t fetch_k = k * 3;
    if (fetch_k < 10) fetch_k = 10;
    if (fetch_k > 100) fetch_k = 100;

    GV_SearchResult *search_results = (GV_SearchResult *)malloc(fetch_k * sizeof(GV_SearchResult));
    if (search_results == NULL) {
        return -1;
    }

    int count = gv_db_search(layer->db, query_embedding, fetch_k, search_results, distance_type);
    if (count < 0) {
        free(search_results);
        return -1;
    }

    /* Build importance contexts for reranking */
    GV_ImportanceContext *contexts = (GV_ImportanceContext *)calloc(count, sizeof(GV_ImportanceContext));
    GV_ImportanceResult *importance_results = (GV_ImportanceResult *)calloc(count, sizeof(GV_ImportanceResult));
    GV_MemoryResult *temp_results = (GV_MemoryResult *)calloc(count, sizeof(GV_MemoryResult));

    if (contexts == NULL || importance_results == NULL || temp_results == NULL) {
        free(search_results);
        free(contexts);
        free(importance_results);
        free(temp_results);
        return -1;
    }

    time_t current_time = time(NULL);

    for (int i = 0; i < count; i++) {
        temp_results[i].distance = search_results[i].distance;
        float similarity = 1.0f - (search_results[i].distance / 2.0f);
        if (similarity < 0.0f) similarity = 0.0f;
        temp_results[i].relevance_score = similarity;

        const GV_Vector *vec = search_results[i].vector;
        if (vec != NULL && vec->metadata != NULL) {
            const char *content = gv_vector_get_metadata(vec, "content");
            const char *mem_id = gv_vector_get_metadata(vec, "memory_id");
            const char *timestamp_str = gv_vector_get_metadata(vec, "timestamp");

            if (content) {
                temp_results[i].content = strdup(content);
                contexts[i].content = temp_results[i].content;
                contexts[i].content_length = strlen(content);
            }
            if (mem_id) {
                temp_results[i].memory_id = strdup(mem_id);
            }

            /* Set up importance context */
            contexts[i].current_time = current_time;
            if (timestamp_str) {
                contexts[i].creation_time = (time_t)atol(timestamp_str);
            }
            contexts[i].semantic_similarity = similarity;

            GV_MemoryMetadata *meta = (GV_MemoryMetadata *)malloc(sizeof(GV_MemoryMetadata));
            if (meta != NULL) {
                parse_memory_metadata(vec->metadata, meta);
                temp_results[i].metadata = meta;
            }
        }

        temp_results[i].related = NULL;
        temp_results[i].related_count = 0;
    }

    /* Calculate importance and rerank */
    size_t *indices = (size_t *)malloc(count * sizeof(size_t));
    if (indices != NULL) {
        GV_ImportanceConfig config = gv_importance_config_default();

        /* Apply temporal weighting if enabled */
        if (!layer->config.enable_temporal_weighting) {
            config.weights.temporal_weight = 0.0;
        }

        /* Rerank: 60% similarity, 40% importance */
        gv_importance_rerank(&config, contexts, importance_results, indices, count, 0.6);

        /* Copy reranked results to output */
        size_t out_count = (size_t)count < k ? (size_t)count : k;
        for (size_t i = 0; i < out_count; i++) {
            size_t src_idx = indices[i];
            results[i] = temp_results[src_idx];

            /* Blend relevance score with importance */
            results[i].relevance_score = 0.6f * temp_results[src_idx].relevance_score +
                                         0.4f * (float)importance_results[src_idx].final_score;

            /* Clear the source so we don't double-free */
            memset(&temp_results[src_idx], 0, sizeof(GV_MemoryResult));
        }

        /* Free unused results */
        for (int i = 0; i < count; i++) {
            if (temp_results[i].content != NULL || temp_results[i].memory_id != NULL) {
                gv_memory_result_free(&temp_results[i]);
            }
        }

        count = (int)out_count;
        free(indices);
    } else {
        /* Fallback: just copy first k results without reranking */
        size_t out_count = (size_t)count < k ? (size_t)count : k;
        for (size_t i = 0; i < out_count; i++) {
            results[i] = temp_results[i];
        }
        for (size_t i = out_count; i < (size_t)count; i++) {
            gv_memory_result_free(&temp_results[i]);
        }
        count = (int)out_count;
    }

    free(search_results);
    free(contexts);
    free(importance_results);
    free(temp_results);

    return count;
}

int gv_memory_get(GV_MemoryLayer *layer, const char *memory_id,
                   GV_MemoryResult *result) {
    if (layer == NULL || memory_id == NULL || result == NULL) {
        return -1;
    }
    
    GV_SearchResult search_result;
    float dummy_query[layer->db->dimension];
    memset(dummy_query, 0, sizeof(dummy_query));
    
    int count = gv_db_search_filtered(layer->db, dummy_query, 1, &search_result,
                                       GV_DISTANCE_EUCLIDEAN, "memory_id", memory_id);
    if (count <= 0) {
        return -1;
    }
    
    const GV_Vector *vec = search_result.vector;
    if (vec == NULL) {
        return -2;
    }
    
    result->distance = search_result.distance;
    result->relevance_score = 1.0f;
    
    const char *content = gv_vector_get_metadata(vec, "content");
    if (content) {
        result->content = strdup(content);
    }
    result->memory_id = strdup(memory_id);
    
    GV_MemoryMetadata *meta = (GV_MemoryMetadata *)malloc(sizeof(GV_MemoryMetadata));
    if (meta != NULL) {
        parse_memory_metadata(vec->metadata, meta);
        result->metadata = meta;
    }
    
    result->related = NULL;
    result->related_count = 0;
    
    return 0;
}

int gv_memory_delete(GV_MemoryLayer *layer, const char *memory_id) {
    if (layer == NULL || memory_id == NULL) {
        return -1;
    }
    
    GV_SearchResult search_result;
    float dummy_query[layer->db->dimension];
    memset(dummy_query, 0, sizeof(dummy_query));
    
    int count = gv_db_search_filtered(layer->db, dummy_query, 1, &search_result,
                                       GV_DISTANCE_EUCLIDEAN, "memory_id", memory_id);
    if (count <= 0) {
        return -1;
    }
    
    return 0;
}

void gv_memory_result_free(GV_MemoryResult *result) {
    if (result == NULL) {
        return;
    }
    
    free(result->memory_id);
    free(result->content);
    gv_memory_metadata_free(result->metadata);
    
    if (result->related != NULL) {
        for (size_t i = 0; i < result->related_count; i++) {
            gv_memory_metadata_free(result->related[i]);
        }
        free(result->related);
    }
}

void gv_memory_metadata_free(GV_MemoryMetadata *metadata) {
    if (metadata == NULL) {
        return;
    }

    free(metadata->memory_id);
    free(metadata->source);
    free(metadata->extraction_metadata);

    if (metadata->related_memory_ids != NULL) {
        for (size_t i = 0; i < metadata->related_count; i++) {
            free(metadata->related_memory_ids[i]);
        }
        free(metadata->related_memory_ids);
    }

    /* Free typed links */
    if (metadata->links != NULL) {
        for (size_t i = 0; i < metadata->link_count; i++) {
            gv_memory_link_free(&metadata->links[i]);
        }
        free(metadata->links);
    }

    free(metadata);
}

char **gv_memory_extract_from_conversation(GV_MemoryLayer *layer,
                                             const char *conversation,
                                             const char *conversation_id,
                                             float **embeddings,
                                             size_t *memory_count) {
    if (layer == NULL || conversation == NULL || embeddings == NULL || memory_count == NULL) {
        return NULL;
    }
    
    GV_MemoryCandidate candidates[100];
    size_t actual_count = 0;
    int result = -1;
    
    // Try LLM extraction first if enabled and LLM is available
    if (layer->config.use_llm_extraction && layer->llm != NULL) {
        result = gv_memory_extract_candidates_from_conversation_llm(
            (GV_LLM *)layer->llm, conversation, conversation_id, 0, NULL,
            candidates, 100, &actual_count
        );
    }
    
    // Fallback to heuristic extraction if LLM failed or not available
    if (result != 0 || actual_count == 0) {
        result = gv_memory_extract_candidates_from_conversation(
            conversation, conversation_id, layer->config.extraction_threshold,
            candidates, 100, &actual_count
        );
    }
    
    if (result != 0 || actual_count == 0) {
        *memory_count = 0;
        return NULL;
    }
    
    char **memory_ids = (char **)malloc(actual_count * sizeof(char *));
    if (memory_ids == NULL) {
        *memory_count = 0;
        return NULL;
    }
    
    *embeddings = (float *)malloc(actual_count * layer->db->dimension * sizeof(float));
    if (*embeddings == NULL) {
        free(memory_ids);
        *memory_count = 0;
        return NULL;
    }
    
    /* Extract entities and relationships if context graph is enabled */
    if (layer->config.enable_context_graph && layer->context_graph != NULL) {
        GV_GraphEntity *entities = NULL;
        size_t entity_count = 0;
        GV_GraphRelationship *relationships = NULL;
        size_t relationship_count = 0;
        
        if (gv_context_graph_extract((GV_ContextGraph *)layer->context_graph, conversation,
                                     NULL, NULL, NULL,
                                     &entities, &entity_count,
                                     &relationships, &relationship_count) == 0) {
            if (entity_count > 0) {
                gv_context_graph_add_entities((GV_ContextGraph *)layer->context_graph,
                                             entities, entity_count);
            }
            if (relationship_count > 0) {
                gv_context_graph_add_relationships((GV_ContextGraph *)layer->context_graph,
                                                   relationships, relationship_count);
            }
            
            /* Free extracted entities and relationships */
            for (size_t j = 0; j < entity_count; j++) {
                gv_graph_entity_free(&entities[j]);
            }
            free(entities);
            for (size_t j = 0; j < relationship_count; j++) {
                gv_graph_relationship_free(&relationships[j]);
            }
            free(relationships);
        }
    }
    
    for (size_t i = 0; i < actual_count; i++) {
        GV_MemoryMetadata meta;
        memset(&meta, 0, sizeof(meta));
        meta.memory_type = candidates[i].memory_type;
        meta.importance_score = candidates[i].importance_score;
        meta.source = conversation_id ? strdup(conversation_id) : NULL;
        meta.timestamp = time(NULL);
        meta.consolidated = 0;
        
        char *mem_id = gv_memory_add(layer, candidates[i].content,
                                      &(*embeddings)[i * layer->db->dimension], &meta);
        memory_ids[i] = mem_id;
        
        gv_memory_metadata_free(&meta);
        gv_memory_candidate_free(&candidates[i]);
    }
    
    *memory_count = actual_count;
    return memory_ids;
}

char **gv_memory_extract_from_text(GV_MemoryLayer *layer,
                                    const char *text,
                                    const char *source,
                                    float **embeddings,
                                    size_t *memory_count) {
    if (layer == NULL || text == NULL || embeddings == NULL || memory_count == NULL) {
        return NULL;
    }
    
    GV_MemoryCandidate candidates[100];
    size_t actual_count = 0;
    
    int result = gv_memory_extract_candidates_from_text(
        text, source, layer->config.extraction_threshold,
        candidates, 100, &actual_count
    );
    
    if (result != 0 || actual_count == 0) {
        *memory_count = 0;
        return NULL;
    }
    
    char **memory_ids = (char **)malloc(actual_count * sizeof(char *));
    if (memory_ids == NULL) {
        *memory_count = 0;
        return NULL;
    }
    
    *embeddings = (float *)malloc(actual_count * layer->db->dimension * sizeof(float));
    if (*embeddings == NULL) {
        free(memory_ids);
        *memory_count = 0;
        return NULL;
    }
    
    for (size_t i = 0; i < actual_count; i++) {
        GV_MemoryMetadata meta;
        memset(&meta, 0, sizeof(meta));
        meta.memory_type = candidates[i].memory_type;
        meta.importance_score = candidates[i].importance_score;
        meta.source = source ? strdup(source) : NULL;
        meta.timestamp = time(NULL);
        meta.consolidated = 0;
        
        char *mem_id = gv_memory_add(layer, candidates[i].content,
                                      &(*embeddings)[i * layer->db->dimension], &meta);
        memory_ids[i] = mem_id;
        
        gv_memory_metadata_free(&meta);
        gv_memory_candidate_free(&candidates[i]);
    }
    
    *memory_count = actual_count;
    return memory_ids;
}

int gv_memory_consolidate(GV_MemoryLayer *layer, double threshold, int strategy) {
    if (layer == NULL) {
        return -1;
    }
    
    double actual_threshold = (threshold > 0.0) ? threshold : layer->config.consolidation_threshold;
    GV_ConsolidationStrategy actual_strategy = (strategy >= 0) ? 
        (GV_ConsolidationStrategy)strategy : layer->config.default_strategy;
    
    GV_MemoryPair pairs[1000];
    size_t pair_count = 0;
    
    int result = gv_memory_find_similar(layer, actual_threshold, pairs, 1000, &pair_count);
    if (result != 0) {
        return -1;
    }
    
    int consolidated = 0;
    for (size_t i = 0; i < pair_count; i++) {
        char *new_id = gv_memory_consolidate_pair(layer, pairs[i].memory_id_1,
                                                    pairs[i].memory_id_2, actual_strategy);
        if (new_id != NULL) {
            consolidated++;
            free(new_id);
        }
    }
    
    gv_memory_pairs_free(pairs, pair_count);
    return consolidated;
}

int gv_memory_search_filtered(GV_MemoryLayer *layer, const float *query_embedding,
                               size_t k, GV_MemoryResult *results,
                               GV_DistanceType distance_type,
                               int memory_type, const char *source,
                               time_t min_timestamp, time_t max_timestamp) {
    if (layer == NULL || query_embedding == NULL || results == NULL) {
        return -1;
    }
    
    char filter_expr[512] = {0};
    int has_filter = 0;
    
    if (memory_type >= 0) {
        snprintf(filter_expr, sizeof(filter_expr), "memory_type == \"%d\"", memory_type);
        has_filter = 1;
    }
    
    if (source != NULL) {
        if (has_filter) {
            strcat(filter_expr, " AND ");
        }
        char source_filter[256];
        snprintf(source_filter, sizeof(source_filter), "source == \"%s\"", source);
        strcat(filter_expr, source_filter);
        has_filter = 1;
    }
    
    GV_SearchResult *search_results = (GV_SearchResult *)malloc(k * sizeof(GV_SearchResult));
    if (search_results == NULL) {
        return -1;
    }
    
    int count;
    if (has_filter) {
        count = gv_db_search_with_filter_expr(layer->db, query_embedding, k, search_results,
                                               distance_type, filter_expr);
    } else {
        count = gv_db_search(layer->db, query_embedding, k, search_results, distance_type);
    }
    
    if (count < 0) {
        free(search_results);
        return -1;
    }
    
    size_t valid_count = 0;
    for (int i = 0; i < count && valid_count < k; i++) {
        const GV_Vector *vec = search_results[i].vector;
        if (vec == NULL || vec->metadata == NULL) {
            continue;
        }
        
        if (min_timestamp > 0 || max_timestamp > 0) {
            const char *ts_str = gv_vector_get_metadata(vec, "timestamp");
            if (ts_str != NULL) {
                time_t ts = (time_t)atol(ts_str);
                if (min_timestamp > 0 && ts < min_timestamp) continue;
                if (max_timestamp > 0 && ts > max_timestamp) continue;
            }
        }
        
        results[valid_count].distance = search_results[i].distance;
        results[valid_count].relevance_score = 1.0f - (search_results[i].distance / 2.0f);
        if (results[valid_count].relevance_score < 0.0f) {
            results[valid_count].relevance_score = 0.0f;
        }
        
        const char *content = gv_vector_get_metadata(vec, "content");
        const char *mem_id = gv_vector_get_metadata(vec, "memory_id");
        
        if (content) {
            results[valid_count].content = strdup(content);
        }
        if (mem_id) {
            results[valid_count].memory_id = strdup(mem_id);
        }
        
        GV_MemoryMetadata *meta = (GV_MemoryMetadata *)malloc(sizeof(GV_MemoryMetadata));
        if (meta != NULL) {
            parse_memory_metadata(vec->metadata, meta);
            results[valid_count].metadata = meta;
        }
        
        results[valid_count].related = NULL;
        results[valid_count].related_count = 0;
        
        valid_count++;
    }
    
    free(search_results);
    return (int)valid_count;
}

int gv_memory_get_related(GV_MemoryLayer *layer, const char *memory_id,
                          size_t k, GV_MemoryResult *results) {
    if (layer == NULL || memory_id == NULL || results == NULL) {
        return -1;
    }
    
    GV_MemoryResult mem_result;
    int ret = gv_memory_get(layer, memory_id, &mem_result);
    if (ret != 0) {
        return -1;
    }
    
    if (mem_result.metadata == NULL || mem_result.metadata->related_count == 0) {
        gv_memory_result_free(&mem_result);
        return 0;
    }
    
    size_t found = 0;
    for (size_t i = 0; i < mem_result.metadata->related_count && found < k; i++) {
        GV_MemoryResult related;
        if (gv_memory_get(layer, mem_result.metadata->related_memory_ids[i], &related) == 0) {
            results[found++] = related;
        }
    }
    
    gv_memory_result_free(&mem_result);
    return (int)found;
}

int gv_memory_update(GV_MemoryLayer *layer, const char *memory_id,
                      const float *new_embedding, GV_MemoryMetadata *new_metadata) {
    if (layer == NULL || memory_id == NULL) {
        return -1;
    }
    
    GV_SearchResult search_result;
    float dummy_query[layer->db->dimension];
    memset(dummy_query, 0, sizeof(dummy_query));
    
    int count = gv_db_search_filtered(layer->db, dummy_query, 1, &search_result,
                                       GV_DISTANCE_EUCLIDEAN, "memory_id", memory_id);
    if (count <= 0) {
        return -1;
    }
    
    if (new_embedding != NULL) {
        return -1;
    }
    
    if (new_metadata != NULL) {
        GV_Metadata *meta_list = create_memory_metadata(new_metadata);
        if (meta_list == NULL) {
            return -1;
        }
    }

    return 0;
}

/* ============================================================================
 * Search Options and Advanced Search
 * ============================================================================ */

GV_MemorySearchOptions gv_memory_search_options_default(void) {
    GV_MemorySearchOptions options;
    options.temporal_weight = 0.0f;      /* Pure semantic by default */
    options.importance_weight = 0.4f;    /* 40% importance, 60% similarity */
    options.include_linked = 0;          /* Don't include linked by default */
    options.link_boost = 0.1f;           /* 10% boost for linked memories */
    options.min_timestamp = 0;
    options.max_timestamp = 0;
    options.memory_type = -1;            /* All types */
    options.source = NULL;
    return options;
}

/**
 * @brief Apply Cortex-style temporal weighting to scores.
 *
 * Formula: combined = semantic * (1 - temporal_weight) + recency * temporal_weight
 * Recency uses exponential decay: e^(-days/7) with ~5-day half-life
 */
static float apply_temporal_blend(float semantic_score, time_t creation_time,
                                   time_t current_time, float temporal_weight) {
    if (temporal_weight <= 0.0f || creation_time == 0) {
        return semantic_score;
    }

    double age_seconds = difftime(current_time, creation_time);
    double days_ago = age_seconds / (24.0 * 3600.0);

    /* Exponential decay with ~5-day half-life (Cortex uses 7.0 divisor) */
    double recency_score = exp(-days_ago / 7.0);
    if (recency_score < 0.0) recency_score = 0.0;
    if (recency_score > 1.0) recency_score = 1.0;

    /* Blend semantic and recency */
    float combined = semantic_score * (1.0f - temporal_weight) +
                     (float)recency_score * temporal_weight;

    return combined;
}

int gv_memory_search_advanced(GV_MemoryLayer *layer, const float *query_embedding,
                               size_t k, GV_MemoryResult *results,
                               GV_DistanceType distance_type,
                               const GV_MemorySearchOptions *options) {
    if (layer == NULL || query_embedding == NULL || results == NULL) {
        return -1;
    }

    GV_MemorySearchOptions opts;
    if (options != NULL) {
        opts = *options;
    } else {
        opts = gv_memory_search_options_default();
    }

    /* Fetch more results for reranking */
    size_t fetch_k = k * 3;
    if (fetch_k < 10) fetch_k = 10;
    if (fetch_k > 100) fetch_k = 100;

    GV_SearchResult *search_results = (GV_SearchResult *)malloc(fetch_k * sizeof(GV_SearchResult));
    if (search_results == NULL) {
        return -1;
    }

    /* Build filter expression */
    char filter_expr[512] = {0};
    int has_filter = 0;

    if (opts.memory_type >= 0) {
        snprintf(filter_expr, sizeof(filter_expr), "memory_type == \"%d\"", opts.memory_type);
        has_filter = 1;
    }

    if (opts.source != NULL) {
        if (has_filter) strcat(filter_expr, " AND ");
        char source_filter[256];
        snprintf(source_filter, sizeof(source_filter), "source == \"%s\"", opts.source);
        strcat(filter_expr, source_filter);
        has_filter = 1;
    }

    int count;
    if (has_filter) {
        count = gv_db_search_with_filter_expr(layer->db, query_embedding, fetch_k,
                                               search_results, distance_type, filter_expr);
    } else {
        count = gv_db_search(layer->db, query_embedding, fetch_k, search_results, distance_type);
    }

    if (count < 0) {
        free(search_results);
        return -1;
    }

    /* Build importance contexts for reranking */
    GV_ImportanceContext *contexts = (GV_ImportanceContext *)calloc(count, sizeof(GV_ImportanceContext));
    GV_ImportanceResult *importance_results = (GV_ImportanceResult *)calloc(count, sizeof(GV_ImportanceResult));
    GV_MemoryResult *temp_results = (GV_MemoryResult *)calloc(count, sizeof(GV_MemoryResult));
    float *combined_scores = (float *)calloc(count, sizeof(float));

    if (contexts == NULL || importance_results == NULL || temp_results == NULL || combined_scores == NULL) {
        free(search_results);
        free(contexts);
        free(importance_results);
        free(temp_results);
        free(combined_scores);
        return -1;
    }

    time_t current_time = time(NULL);

    /* First pass: collect results and compute base scores */
    int valid_count = 0;
    for (int i = 0; i < count; i++) {
        const GV_Vector *vec = search_results[i].vector;
        if (vec == NULL || vec->metadata == NULL) {
            continue;
        }

        const char *timestamp_str = gv_vector_get_metadata(vec, "timestamp");
        time_t creation_time = timestamp_str ? (time_t)atol(timestamp_str) : 0;

        /* Apply timestamp filter */
        if (opts.min_timestamp > 0 && creation_time < opts.min_timestamp) continue;
        if (opts.max_timestamp > 0 && creation_time > opts.max_timestamp) continue;

        temp_results[valid_count].distance = search_results[i].distance;
        float semantic_score = 1.0f - (search_results[i].distance / 2.0f);
        if (semantic_score < 0.0f) semantic_score = 0.0f;

        /* Apply temporal blending (Cortex-style) */
        float blended_score = apply_temporal_blend(semantic_score, creation_time,
                                                    current_time, opts.temporal_weight);
        temp_results[valid_count].relevance_score = blended_score;

        const char *content = gv_vector_get_metadata(vec, "content");
        const char *mem_id = gv_vector_get_metadata(vec, "memory_id");

        if (content) {
            temp_results[valid_count].content = strdup(content);
            contexts[valid_count].content = temp_results[valid_count].content;
            contexts[valid_count].content_length = strlen(content);
        }
        if (mem_id) {
            temp_results[valid_count].memory_id = strdup(mem_id);
        }

        /* Set up importance context */
        contexts[valid_count].current_time = current_time;
        contexts[valid_count].creation_time = creation_time;
        contexts[valid_count].semantic_similarity = semantic_score;

        GV_MemoryMetadata *meta = (GV_MemoryMetadata *)malloc(sizeof(GV_MemoryMetadata));
        if (meta != NULL) {
            memset(meta, 0, sizeof(GV_MemoryMetadata));
            parse_memory_metadata(vec->metadata, meta);
            temp_results[valid_count].metadata = meta;

            /* Add structural context from links */
            contexts[valid_count].relationship_count = meta->link_count;
            contexts[valid_count].incoming_links = meta->link_count / 2;
            contexts[valid_count].outgoing_links = meta->link_count - contexts[valid_count].incoming_links;
        }

        temp_results[valid_count].related = NULL;
        temp_results[valid_count].related_count = 0;

        valid_count++;
    }

    free(search_results);

    if (valid_count == 0) {
        free(contexts);
        free(importance_results);
        free(temp_results);
        free(combined_scores);
        return 0;
    }

    /* Calculate importance scores */
    GV_ImportanceConfig importance_config = gv_importance_config_default();
    if (!layer->config.enable_temporal_weighting) {
        importance_config.weights.temporal_weight = 0.0;
    }

    gv_importance_calculate_batch(&importance_config, contexts, importance_results, valid_count);

    /* Compute final combined scores */
    float similarity_weight = 1.0f - opts.importance_weight;
    for (int i = 0; i < valid_count; i++) {
        combined_scores[i] = similarity_weight * temp_results[i].relevance_score +
                             opts.importance_weight * (float)importance_results[i].final_score;
    }

    /* Sort by combined score (simple selection sort for small k) */
    size_t *indices = (size_t *)malloc(valid_count * sizeof(size_t));
    if (indices == NULL) {
        /* Fallback: return unsorted */
        size_t out_count = (size_t)valid_count < k ? (size_t)valid_count : k;
        for (size_t i = 0; i < out_count; i++) {
            results[i] = temp_results[i];
            results[i].relevance_score = combined_scores[i];
        }
        for (size_t i = out_count; i < (size_t)valid_count; i++) {
            gv_memory_result_free(&temp_results[i]);
        }
        free(contexts);
        free(importance_results);
        free(temp_results);
        free(combined_scores);
        return (int)out_count;
    }

    for (int i = 0; i < valid_count; i++) {
        indices[i] = i;
    }

    /* Sort indices by combined_scores descending */
    for (int i = 0; i < valid_count - 1; i++) {
        for (int j = i + 1; j < valid_count; j++) {
            if (combined_scores[indices[j]] > combined_scores[indices[i]]) {
                size_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    /* Copy sorted results to output */
    size_t out_count = (size_t)valid_count < k ? (size_t)valid_count : k;
    for (size_t i = 0; i < out_count; i++) {
        size_t src_idx = indices[i];
        results[i] = temp_results[src_idx];
        results[i].relevance_score = combined_scores[src_idx];
        memset(&temp_results[src_idx], 0, sizeof(GV_MemoryResult));
    }

    /* Free unused results */
    for (int i = 0; i < valid_count; i++) {
        if (temp_results[i].content != NULL || temp_results[i].memory_id != NULL) {
            gv_memory_result_free(&temp_results[i]);
        }
    }

    free(indices);
    free(contexts);
    free(importance_results);
    free(temp_results);
    free(combined_scores);

    return (int)out_count;
}

/* ============================================================================
 * Memory Link Management
 * ============================================================================ */

/** Reciprocal strength multiplier (Cortex uses 0.9) */
#define RECIPROCAL_STRENGTH_FACTOR 0.9f

/** Minimum link strength */
#define MIN_LINK_STRENGTH 0.1f

GV_MemoryLinkType gv_memory_link_reciprocal(GV_MemoryLinkType link_type) {
    switch (link_type) {
        case GV_LINK_SUPPORTS:
            return GV_LINK_SUPPORTS;  /* Symmetric for simplicity */
        case GV_LINK_CONTRADICTS:
            return GV_LINK_CONTRADICTS;  /* Symmetric */
        case GV_LINK_EXTENDS:
            return GV_LINK_EXTENDS;  /* Symmetric */
        case GV_LINK_CAUSAL:
            return GV_LINK_CAUSAL;  /* Effect -> Cause */
        case GV_LINK_EXAMPLE:
            return GV_LINK_EXAMPLE;  /* Symmetric */
        case GV_LINK_PREREQUISITE:
            return GV_LINK_PREREQUISITE;  /* Depends -> Required */
        case GV_LINK_TEMPORAL:
            return GV_LINK_TEMPORAL;  /* Symmetric */
        case GV_LINK_SIMILAR:
        default:
            return GV_LINK_SIMILAR;
    }
}

void gv_memory_link_free(GV_MemoryLink *link) {
    if (link == NULL) {
        return;
    }
    free(link->target_memory_id);
    free(link->reason);
    memset(link, 0, sizeof(GV_MemoryLink));
}

/**
 * @brief Helper to add a link to a memory's metadata.
 */
static int add_link_to_memory(GV_MemoryLayer *layer, const char *memory_id,
                               const char *target_id, GV_MemoryLinkType link_type,
                               float strength, const char *reason) {
    /* Get current memory */
    GV_MemoryResult result;
    int ret = gv_memory_get(layer, memory_id, &result);
    if (ret != 0) {
        return -1;
    }

    if (result.metadata == NULL) {
        gv_memory_result_free(&result);
        return -1;
    }

    /* Check if link already exists */
    for (size_t i = 0; i < result.metadata->link_count; i++) {
        if (result.metadata->links[i].target_memory_id &&
            strcmp(result.metadata->links[i].target_memory_id, target_id) == 0) {
            /* Link already exists - update it */
            result.metadata->links[i].link_type = link_type;
            result.metadata->links[i].strength = strength;
            if (result.metadata->links[i].reason) {
                free(result.metadata->links[i].reason);
            }
            result.metadata->links[i].reason = reason ? strdup(reason) : NULL;

            ret = gv_memory_update(layer, memory_id, NULL, result.metadata);
            gv_memory_result_free(&result);
            return ret;
        }
    }

    /* Add new link */
    size_t new_count = result.metadata->link_count + 1;
    GV_MemoryLink *new_links = (GV_MemoryLink *)realloc(
        result.metadata->links, new_count * sizeof(GV_MemoryLink));
    if (new_links == NULL) {
        gv_memory_result_free(&result);
        return -1;
    }

    result.metadata->links = new_links;
    GV_MemoryLink *new_link = &result.metadata->links[result.metadata->link_count];
    new_link->target_memory_id = strdup(target_id);
    new_link->link_type = link_type;
    new_link->strength = strength;
    new_link->created_at = time(NULL);
    new_link->reason = reason ? strdup(reason) : NULL;
    result.metadata->link_count = new_count;

    ret = gv_memory_update(layer, memory_id, NULL, result.metadata);
    gv_memory_result_free(&result);
    return ret;
}

int gv_memory_link_create(GV_MemoryLayer *layer,
                           const char *source_id,
                           const char *target_id,
                           GV_MemoryLinkType link_type,
                           float strength,
                           const char *reason) {
    if (layer == NULL || source_id == NULL || target_id == NULL) {
        return -1;
    }

    /* Clamp strength */
    if (strength < MIN_LINK_STRENGTH) strength = MIN_LINK_STRENGTH;
    if (strength > 1.0f) strength = 1.0f;

    pthread_mutex_lock(&layer->mutex);

    /* Add forward link (source -> target) */
    int ret = add_link_to_memory(layer, source_id, target_id, link_type, strength, reason);
    if (ret != 0) {
        pthread_mutex_unlock(&layer->mutex);
        return -1;
    }

    /* Add reciprocal link (target -> source) with reduced strength */
    GV_MemoryLinkType reciprocal_type = gv_memory_link_reciprocal(link_type);
    float reciprocal_strength = strength * RECIPROCAL_STRENGTH_FACTOR;
    if (reciprocal_strength < MIN_LINK_STRENGTH) {
        reciprocal_strength = MIN_LINK_STRENGTH;
    }

    ret = add_link_to_memory(layer, target_id, source_id, reciprocal_type,
                             reciprocal_strength, reason);

    pthread_mutex_unlock(&layer->mutex);
    return ret;
}

/**
 * @brief Helper to remove a link from a memory's metadata.
 */
static int remove_link_from_memory(GV_MemoryLayer *layer, const char *memory_id,
                                    const char *target_id) {
    /* Get current memory */
    GV_MemoryResult result;
    int ret = gv_memory_get(layer, memory_id, &result);
    if (ret != 0) {
        return -1;
    }

    if (result.metadata == NULL || result.metadata->link_count == 0) {
        gv_memory_result_free(&result);
        return 0;  /* No links to remove */
    }

    /* Find and remove the link */
    int found = 0;
    for (size_t i = 0; i < result.metadata->link_count; i++) {
        if (result.metadata->links[i].target_memory_id &&
            strcmp(result.metadata->links[i].target_memory_id, target_id) == 0) {
            /* Free the link data */
            free(result.metadata->links[i].target_memory_id);
            free(result.metadata->links[i].reason);

            /* Shift remaining links */
            for (size_t j = i; j < result.metadata->link_count - 1; j++) {
                result.metadata->links[j] = result.metadata->links[j + 1];
            }
            result.metadata->link_count--;
            found = 1;
            break;
        }
    }

    if (found) {
        ret = gv_memory_update(layer, memory_id, NULL, result.metadata);
    }

    gv_memory_result_free(&result);
    return ret;
}

int gv_memory_link_remove(GV_MemoryLayer *layer,
                           const char *source_id,
                           const char *target_id) {
    if (layer == NULL || source_id == NULL || target_id == NULL) {
        return -1;
    }

    pthread_mutex_lock(&layer->mutex);

    /* Remove forward link (source -> target) */
    int ret = remove_link_from_memory(layer, source_id, target_id);
    if (ret != 0) {
        pthread_mutex_unlock(&layer->mutex);
        return -1;
    }

    /* Remove reciprocal link (target -> source) */
    ret = remove_link_from_memory(layer, target_id, source_id);

    pthread_mutex_unlock(&layer->mutex);
    return ret;
}

int gv_memory_link_get(GV_MemoryLayer *layer,
                        const char *memory_id,
                        GV_MemoryLink *links,
                        size_t max_links) {
    if (layer == NULL || memory_id == NULL || links == NULL) {
        return -1;
    }

    GV_MemoryResult result;
    int ret = gv_memory_get(layer, memory_id, &result);
    if (ret != 0) {
        return -1;
    }

    size_t link_count = 0;
    if (result.metadata != NULL && result.metadata->links != NULL) {
        link_count = result.metadata->link_count;
        if (link_count > max_links) {
            link_count = max_links;
        }

        for (size_t i = 0; i < link_count; i++) {
            links[i].target_memory_id = result.metadata->links[i].target_memory_id ?
                strdup(result.metadata->links[i].target_memory_id) : NULL;
            links[i].link_type = result.metadata->links[i].link_type;
            links[i].strength = result.metadata->links[i].strength;
            links[i].created_at = result.metadata->links[i].created_at;
            links[i].reason = result.metadata->links[i].reason ?
                strdup(result.metadata->links[i].reason) : NULL;
        }
    }

    gv_memory_result_free(&result);
    return (int)link_count;
}

int gv_memory_record_access(GV_MemoryLayer *layer,
                             const char *memory_id,
                             float relevance) {
    if (layer == NULL || memory_id == NULL) {
        return -1;
    }

    /* relevance is stored for access pattern analysis but currently unused
     * in metadata update - will be used when access history is fully implemented */
    (void)relevance;

    /* Get current memory */
    GV_MemoryResult result;
    int ret = gv_memory_get(layer, memory_id, &result);
    if (ret != 0) {
        return -1;
    }

    if (result.metadata != NULL) {
        result.metadata->access_count++;
        result.metadata->last_accessed = time(NULL);

        /* Update metadata in storage */
        ret = gv_memory_update(layer, memory_id, NULL, result.metadata);
    }

    gv_memory_result_free(&result);
    return ret;
}

