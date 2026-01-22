#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <pthread.h>
#include <stdint.h>
#include <math.h>

#include "gigavector/gv_context_graph.h"
#include "gigavector/gv_llm.h"
#include "gigavector/gv_embedding.h"
#include "gigavector/gv_types.h"

#define ENTITY_ID_PREFIX "ent_"
#define RELATIONSHIP_ID_PREFIX "rel_"
#define ENTITY_ID_BUFFER_SIZE 64
#define RELATIONSHIP_ID_BUFFER_SIZE 64

/* Hash table for entities */
typedef struct EntityNode {
    char *entity_id;
    GV_GraphEntity entity;
    struct EntityNode *next;
} EntityNode;

/* Hash table for relationships */
typedef struct RelationshipNode {
    char *relationship_id;
    GV_GraphRelationship relationship;
    struct RelationshipNode *next;
} RelationshipNode;

/* Graph structure */
struct GV_ContextGraph {
    GV_ContextGraphConfig config;
    EntityNode **entity_table;
    RelationshipNode **relationship_table;
    size_t entity_table_size;
    size_t relationship_table_size;
    uint64_t next_entity_id;
    uint64_t next_relationship_id;
    pthread_mutex_t mutex;
};

/* Hash function for strings */
static size_t hash_string(const char *str, size_t table_size) {
    size_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % table_size;
}

/* Generate entity ID */
static char *generate_entity_id(uint64_t counter) {
    char *id = (char *)malloc(ENTITY_ID_BUFFER_SIZE);
    if (id == NULL) {
        return NULL;
    }
    snprintf(id, ENTITY_ID_BUFFER_SIZE, "%s%lu", ENTITY_ID_PREFIX, (unsigned long)counter);
    return id;
}

/* Generate relationship ID */
static char *generate_relationship_id(uint64_t counter) {
    char *id = (char *)malloc(RELATIONSHIP_ID_BUFFER_SIZE);
    if (id == NULL) {
        return NULL;
    }
    snprintf(id, RELATIONSHIP_ID_BUFFER_SIZE, "%s%lu", RELATIONSHIP_ID_PREFIX, (unsigned long)counter);
    return id;
}

/* Cosine similarity */
static float cosine_similarity(const float *a, const float *b, size_t dim) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < dim; i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (sqrtf(norm_a) * sqrtf(norm_b));
}

GV_ContextGraphConfig gv_context_graph_config_default(void) {
    GV_ContextGraphConfig config;
    config.llm = NULL;
    config.embedding_service = NULL;
    config.similarity_threshold = 0.7;
    config.enable_entity_extraction = 1;
    config.enable_relationship_extraction = 1;
    config.max_traversal_depth = 3;
    config.max_results = 100;
    config.embedding_callback = NULL;
    config.embedding_user_data = NULL;
    config.embedding_dimension = 0;
    return config;
}

GV_ContextGraph *gv_context_graph_create(const GV_ContextGraphConfig *config) {
    GV_ContextGraph *graph = (GV_ContextGraph *)calloc(1, sizeof(GV_ContextGraph));
    if (graph == NULL) {
        return NULL;
    }
    
    if (config != NULL) {
        graph->config = *config;
    } else {
        graph->config = gv_context_graph_config_default();
    }
    
    graph->entity_table_size = 1024;
    graph->relationship_table_size = 1024;
    graph->entity_table = (EntityNode **)calloc(graph->entity_table_size, sizeof(EntityNode *));
    graph->relationship_table = (RelationshipNode **)calloc(graph->relationship_table_size, sizeof(RelationshipNode *));
    
    if (graph->entity_table == NULL || graph->relationship_table == NULL) {
        free(graph->entity_table);
        free(graph->relationship_table);
        free(graph);
        return NULL;
    }
    
    graph->next_entity_id = 1;
    graph->next_relationship_id = 1;
    
    if (pthread_mutex_init(&graph->mutex, NULL) != 0) {
        free(graph->entity_table);
        free(graph->relationship_table);
        free(graph);
        return NULL;
    }
    
    return graph;
}

/* Parse entities from JSON response */
static int parse_entities_json(const char *json_response, GV_GraphEntity **entities, size_t *entity_count) {
    if (json_response == NULL || entities == NULL || entity_count == NULL) {
        return -1;
    }
    
    *entities = NULL;
    *entity_count = 0;
    
    /* Find entities array - look for "entities" or direct array */
    const char *array_start = strstr(json_response, "\"entities\"");
    if (array_start == NULL) {
        /* Try direct array format */
        array_start = strchr(json_response, '[');
    } else {
        array_start = strchr(array_start, '[');
    }
    
    if (array_start == NULL) {
        return -1;
    }
    
    /* Count entities by counting "entity" fields */
    size_t count = 0;
    const char *p = array_start;
    while ((p = strstr(p, "\"entity\"")) != NULL) {
        count++;
        p += 7;
    }
    
    if (count == 0) {
        return 0;  // No entities found, but valid JSON
    }
    
    GV_GraphEntity *ents = (GV_GraphEntity *)calloc(count, sizeof(GV_GraphEntity));
    if (ents == NULL) {
        return -1;
    }
    
    /* Parse each entity */
    p = array_start;
    size_t idx = 0;
    while (idx < count && (p = strstr(p, "\"entity\"")) != NULL) {
        p += 8;  // Skip past "entity"
        p = strchr(p, '"');
        if (p == NULL) break;
        p++;  // Skip opening quote
        
        const char *name_start = p;
        const char *name_end = strchr(p, '"');
        if (name_end == NULL) break;
        
        size_t name_len = name_end - name_start;
        ents[idx].name = (char *)malloc(name_len + 1);
        if (ents[idx].name == NULL) {
            for (size_t i = 0; i < idx; i++) {
                free(ents[i].name);
            }
            free(ents);
            return -1;
        }
        memcpy(ents[idx].name, name_start, name_len);
        ents[idx].name[name_len] = '\0';
        
        /* Find entity_type */
        const char *type_start = strstr(p, "\"entity_type\"");
        if (type_start != NULL) {
            type_start = strchr(type_start, '"');
            if (type_start != NULL) {
                type_start++;
                const char *type_end = strchr(type_start, '"');
                if (type_end != NULL) {
                    size_t type_len = type_end - type_start;
                    char type_str[64];
                    if (type_len < sizeof(type_str)) {
                        memcpy(type_str, type_start, type_len);
                        type_str[type_len] = '\0';
                        
                        /* Map string to enum */
                        if (strcmp(type_str, "person") == 0) {
                            ents[idx].entity_type = GV_ENTITY_TYPE_PERSON;
                        } else if (strcmp(type_str, "organization") == 0) {
                            ents[idx].entity_type = GV_ENTITY_TYPE_ORGANIZATION;
                        } else if (strcmp(type_str, "location") == 0) {
                            ents[idx].entity_type = GV_ENTITY_TYPE_LOCATION;
                        } else if (strcmp(type_str, "event") == 0) {
                            ents[idx].entity_type = GV_ENTITY_TYPE_EVENT;
                        } else if (strcmp(type_str, "object") == 0) {
                            ents[idx].entity_type = GV_ENTITY_TYPE_OBJECT;
                        } else if (strcmp(type_str, "concept") == 0) {
                            ents[idx].entity_type = GV_ENTITY_TYPE_CONCEPT;
                        } else if (strcmp(type_str, "user") == 0) {
                            ents[idx].entity_type = GV_ENTITY_TYPE_USER;
                        } else {
                            ents[idx].entity_type = GV_ENTITY_TYPE_PERSON;  // Default
                        }
                    }
                }
            }
        }
        
        idx++;
        p = name_end;
    }
    
    *entities = ents;
    *entity_count = idx;
    return 0;
}

/* Parse relationships from JSON response */
static int parse_relationships_json(const char *json_response, GV_GraphRelationship **relationships, size_t *relationship_count) {
    if (json_response == NULL || relationships == NULL || relationship_count == NULL) {
        return -1;
    }
    
    *relationships = NULL;
    *relationship_count = 0;
    
    /* Find relationships array - look for "entities" or direct array */
    const char *array_start = strstr(json_response, "\"entities\"");
    if (array_start == NULL) {
        array_start = strstr(json_response, "\"relationships\"");
        if (array_start == NULL) {
            /* Try direct array format */
            array_start = strchr(json_response, '[');
        } else {
            array_start = strchr(array_start, '[');
        }
    } else {
        array_start = strchr(array_start, '[');
    }
    
    if (array_start == NULL) {
        return -1;
    }
    
    /* Count relationships by counting "source" fields */
    size_t count = 0;
    const char *p = array_start;
    while ((p = strstr(p, "\"source\"")) != NULL) {
        count++;
        p += 8;
    }
    
    if (count == 0) {
        return 0;  // No relationships found, but valid JSON
    }
    
    GV_GraphRelationship *rels = (GV_GraphRelationship *)calloc(count, sizeof(GV_GraphRelationship));
    if (rels == NULL) {
        return -1;
    }
    
    /* Parse each relationship */
    p = array_start;
    size_t idx = 0;
    while (idx < count && (p = strstr(p, "\"source\"")) != NULL) {
        p += 8;  // Skip past "source"
        p = strchr(p, '"');
        if (p == NULL) break;
        p++;  // Skip opening quote
        
        const char *source_start = p;
        const char *source_end = strchr(p, '"');
        if (source_end == NULL) break;
        
        size_t source_len = source_end - source_start;
        char *source = (char *)malloc(source_len + 1);
        if (source == NULL) {
            for (size_t i = 0; i < idx; i++) {
                free(rels[i].source_entity_id);
                free(rels[i].destination_entity_id);
                free(rels[i].relationship_type);
            }
            free(rels);
            return -1;
        }
        memcpy(source, source_start, source_len);
        source[source_len] = '\0';
        
        /* Find relationship */
        const char *rel_start = strstr(p, "\"relationship\"");
        if (rel_start == NULL) {
            rel_start = strstr(p, "\"relationship_type\"");
        }
        if (rel_start != NULL) {
            rel_start = strchr(rel_start, '"');
            if (rel_start != NULL) {
                rel_start++;
                const char *rel_end = strchr(rel_start, '"');
                if (rel_end != NULL) {
                    size_t rel_len = rel_end - rel_start;
                    rels[idx].relationship_type = (char *)malloc(rel_len + 1);
                    if (rels[idx].relationship_type != NULL) {
                        memcpy(rels[idx].relationship_type, rel_start, rel_len);
                        rels[idx].relationship_type[rel_len] = '\0';
                    }
                }
            }
        }
        
        /* Find destination */
        const char *dest_start = strstr(p, "\"destination\"");
        if (dest_start != NULL) {
            dest_start = strchr(dest_start, '"');
            if (dest_start != NULL) {
                dest_start++;
                const char *dest_end = strchr(dest_start, '"');
                if (dest_end != NULL) {
                    size_t dest_len = dest_end - dest_start;
                    char *dest = (char *)malloc(dest_len + 1);
                    if (dest != NULL) {
                        memcpy(dest, dest_start, dest_len);
                        dest[dest_len] = '\0';
                        
                        /* Generate entity IDs from names (simplified - in real implementation, 
                           would look up actual entity IDs) */
                        char source_id[128];
                        char dest_id[128];
                        snprintf(source_id, sizeof(source_id), "ent_%s", source);
                        snprintf(dest_id, sizeof(dest_id), "ent_%s", dest);
                        
                        rels[idx].source_entity_id = strdup(source_id);
                        rels[idx].destination_entity_id = strdup(dest_id);
                        
                        free(source);
                        free(dest);
                    } else {
                        free(source);
                        for (size_t i = 0; i < idx; i++) {
                            free(rels[i].source_entity_id);
                            free(rels[i].destination_entity_id);
                            free(rels[i].relationship_type);
                        }
                        free(rels);
                        return -1;
                    }
                }
            }
        }
        
        if (rels[idx].source_entity_id == NULL || rels[idx].destination_entity_id == NULL) {
            free(source);
            for (size_t i = 0; i < idx; i++) {
                free(rels[i].source_entity_id);
                free(rels[i].destination_entity_id);
                free(rels[i].relationship_type);
            }
            free(rels);
            return -1;
        }
        
        idx++;
        p = source_end;
    }
    
    *relationships = rels;
    *relationship_count = idx;
    return 0;
}

void gv_context_graph_destroy(GV_ContextGraph *graph) {
    if (graph == NULL) {
        return;
    }
    
    pthread_mutex_lock(&graph->mutex);
    
    /* Free entity table */
    for (size_t i = 0; i < graph->entity_table_size; i++) {
        EntityNode *node = graph->entity_table[i];
        while (node != NULL) {
            EntityNode *next = node->next;
            gv_graph_entity_free(&node->entity);
            free(node->entity_id);
            free(node);
            node = next;
        }
    }
    free(graph->entity_table);
    
    /* Free relationship table */
    for (size_t i = 0; i < graph->relationship_table_size; i++) {
        RelationshipNode *node = graph->relationship_table[i];
        while (node != NULL) {
            RelationshipNode *next = node->next;
            gv_graph_relationship_free(&node->relationship);
            free(node->relationship_id);
            free(node);
            node = next;
        }
    }
    free(graph->relationship_table);
    
    pthread_mutex_unlock(&graph->mutex);
    pthread_mutex_destroy(&graph->mutex);
    free(graph);
}

/* Extract entities using LLM */
static int extract_entities_llm(GV_LLM *llm, const char *text, const char *user_id,
                                 GV_GraphEntity **entities, size_t *entity_count) {
    if (llm == NULL || text == NULL || entities == NULL || entity_count == NULL) {
        return -1;
    }
    
    /* Build prompt for entity extraction */
    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
        "You are an expert at extracting entities from text. Extract all entities mentioned in the following text.\n\n"
        "If the text contains self-references like 'I', 'me', 'my', use '%s' as the entity name.\n\n"
        "For each entity, provide:\n"
        "- entity: the entity name\n"
        "- entity_type: one of (person, organization, location, event, object, concept, user)\n\n"
        "Text: %s\n\n"
        "Return a JSON array of entities in this format:\n"
        "[{\"entity\": \"name\", \"entity_type\": \"type\"}, ...]",
        user_id ? user_id : "user", text);
    
    GV_LLMMessage messages[2];
    messages[0].role = "system";
    messages[0].content = "You are an expert at extracting entities from text. Return only valid JSON.";
    messages[1].role = "user";
    messages[1].content = prompt;
    
    GV_LLMResponse response;
    int result = gv_llm_generate_response(llm, messages, 2, "{\"type\":\"json_object\"}", &response);
    
    if (result != 0 || response.content == NULL) {
        return -1;
    }
    
    /* Parse JSON response - simple string-based parser */
    int parse_result = parse_entities_json(response.content, entities, entity_count);
    
    free(response.content);
    return parse_result;
}

/* Extract relationships using LLM */
static int extract_relationships_llm(GV_LLM *llm, const char *text, const char *user_id,
                                     const GV_GraphEntity *entities, size_t entity_count,
                                     GV_GraphRelationship **relationships, size_t *relationship_count) {
    if (llm == NULL || text == NULL || relationships == NULL || relationship_count == NULL) {
        return -1;
    }
    
    /* Build entity list string */
    char entity_list[4096] = "";
    for (size_t i = 0; i < entity_count && i < 50; i++) {
        if (i > 0) {
            strncat(entity_list, ", ", sizeof(entity_list) - strlen(entity_list) - 1);
        }
        strncat(entity_list, entities[i].name, sizeof(entity_list) - strlen(entity_list) - 1);
    }
    
    /* Build prompt for relationship extraction */
    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
        "You are an expert at extracting relationships between entities from text.\n\n"
        "If the text contains self-references like 'I', 'me', 'my', use '%s' as the source entity.\n\n"
        "Entities mentioned: %s\n\n"
        "Text: %s\n\n"
        "Extract relationships between entities. Return a JSON array in this format:\n"
        "[{\"source\": \"entity1\", \"relationship\": \"relationship_type\", \"destination\": \"entity2\"}, ...]\n\n"
        "Relationship types should be concise and timeless (e.g., 'knows', 'works_with', 'located_in').",
        user_id ? user_id : "user", entity_list, text);
    
    GV_LLMMessage messages[2];
    messages[0].role = "system";
    messages[0].content = "You are an expert at extracting relationships from text. Return only valid JSON.";
    messages[1].role = "user";
    messages[1].content = prompt;
    
    GV_LLMResponse response;
    int result = gv_llm_generate_response(llm, messages, 2, "{\"type\":\"json_object\"}", &response);
    
    if (result != 0 || response.content == NULL) {
        return -1;
    }
    
    /* Parse JSON response - simple string-based parser */
    int parse_result = parse_relationships_json(response.content, relationships, relationship_count);
    
    free(response.content);
    return parse_result;
}

int gv_context_graph_extract(GV_ContextGraph *graph,
                             const char *text,
                             const char *user_id,
                             const char *agent_id,
                             const char *run_id,
                             GV_GraphEntity **entities,
                             size_t *entity_count,
                             GV_GraphRelationship **relationships,
                             size_t *relationship_count) {
    /* Note: agent_id and run_id available for future metadata/filtering use */
    (void)agent_id;
    (void)run_id;
    if (graph == NULL || text == NULL || entities == NULL || entity_count == NULL ||
        relationships == NULL || relationship_count == NULL) {
        return -1;
    }
    
    *entities = NULL;
    *entity_count = 0;
    *relationships = NULL;
    *relationship_count = 0;
    
    if (!graph->config.enable_entity_extraction && !graph->config.enable_relationship_extraction) {
        return 0;
    }
    
    GV_LLM *llm = (GV_LLM *)graph->config.llm;
    if (llm == NULL) {
        return -1;
    }
    
    /* Extract entities */
    if (graph->config.enable_entity_extraction) {
        if (extract_entities_llm(llm, text, user_id, entities, entity_count) != 0) {
            return -1;
        }
        
        /* Set agent_id and run_id on extracted entities */
        if (*entities != NULL && *entity_count > 0) {
            for (size_t i = 0; i < *entity_count; i++) {
                if (agent_id != NULL) {
                    (*entities)[i].agent_id = strdup(agent_id);
                }
                if (run_id != NULL) {
                    (*entities)[i].run_id = strdup(run_id);
                }
            }
        }
        
        /* Generate embeddings for extracted entities */
        if (*entities != NULL && *entity_count > 0) {
            GV_EmbeddingService *embedding_service = (GV_EmbeddingService *)graph->config.embedding_service;
            if (embedding_service != NULL) {
                /* Use embedding service - batch generate */
                const char **texts = (const char **)malloc(*entity_count * sizeof(const char *));
                if (texts != NULL) {
                    for (size_t i = 0; i < *entity_count; i++) {
                        texts[i] = ((*entities)[i].embedding == NULL && (*entities)[i].name != NULL) 
                                  ? (*entities)[i].name : NULL;
                    }
                    
                    size_t *embedding_dims = NULL;
                    float **embeddings = NULL;
                    int batch_result = gv_embedding_generate_batch(embedding_service, texts, *entity_count,
                                                                   &embedding_dims, &embeddings);
                    
                    if (batch_result > 0 && embeddings != NULL) {
                        for (size_t i = 0; i < *entity_count; i++) {
                            if (embeddings[i] != NULL && embedding_dims[i] > 0) {
                                (*entities)[i].embedding = embeddings[i];
                                (*entities)[i].embedding_dim = embedding_dims[i];
                                if (graph->config.embedding_dimension == 0) {
                                    graph->config.embedding_dimension = embedding_dims[i];
                                }
                            }
                        }
                    }
                    
                    free(texts);
                    if (embedding_dims) free(embedding_dims);
                    /* Note: embeddings are owned by entities now */
                }
            } else if (graph->config.embedding_callback != NULL) {
                /* Fall back to callback */
                for (size_t i = 0; i < *entity_count; i++) {
                    if ((*entities)[i].embedding == NULL && (*entities)[i].name != NULL) {
                        size_t dim = graph->config.embedding_dimension;
                        float *embedding = graph->config.embedding_callback((*entities)[i].name, &dim, graph->config.embedding_user_data);
                        if (embedding != NULL && dim > 0) {
                            (*entities)[i].embedding = embedding;
                            (*entities)[i].embedding_dim = dim;
                            if (graph->config.embedding_dimension == 0) {
                                graph->config.embedding_dimension = dim;
                            }
                        }
                    }
                }
            }
        }
    }
    
    /* Extract relationships */
    if (graph->config.enable_relationship_extraction && *entity_count > 0) {
        if (extract_relationships_llm(llm, text, user_id, *entities, *entity_count,
                                      relationships, relationship_count) != 0) {
            /* Free entities on error */
            for (size_t i = 0; i < *entity_count; i++) {
                gv_graph_entity_free(&(*entities)[i]);
            }
            free(*entities);
            *entities = NULL;
            *entity_count = 0;
            return -1;
        }
    }
    
    return 0;
}

int gv_context_graph_add_entities(GV_ContextGraph *graph,
                                  const GV_GraphEntity *entities,
                                  size_t entity_count) {
    if (graph == NULL || entities == NULL) {
        return -1;
    }
    
    pthread_mutex_lock(&graph->mutex);
    
    for (size_t i = 0; i < entity_count; i++) {
        const GV_GraphEntity *ent = &entities[i];
        
        /* Check if entity already exists */
        size_t hash = hash_string(ent->name, graph->entity_table_size);
        EntityNode *node = graph->entity_table[hash];
        EntityNode *found = NULL;
        
        while (node != NULL) {
            if (strcmp(node->entity.name, ent->name) == 0 &&
                (ent->user_id == NULL || node->entity.user_id == NULL ||
                 strcmp(node->entity.user_id, ent->user_id) == 0)) {
                found = node;
                break;
            }
            node = node->next;
        }
        
        if (found != NULL) {
            /* Update existing entity */
            found->entity.updated = time(NULL);
            found->entity.mentions++;
            if (ent->embedding != NULL && ent->embedding_dim > 0) {
                free(found->entity.embedding);
                found->entity.embedding = (float *)malloc(ent->embedding_dim * sizeof(float));
                if (found->entity.embedding != NULL) {
                    memcpy(found->entity.embedding, ent->embedding, ent->embedding_dim * sizeof(float));
                    found->entity.embedding_dim = ent->embedding_dim;
                }
            }
        } else {
            /* Create new entity */
            EntityNode *new_node = (EntityNode *)calloc(1, sizeof(EntityNode));
            if (new_node == NULL) {
                continue;
            }
            
            new_node->entity_id = generate_entity_id(graph->next_entity_id++);
            if (new_node->entity_id == NULL) {
                free(new_node);
                continue;
            }
            
            new_node->entity.entity_id = strdup(new_node->entity_id);
            new_node->entity.name = strdup(ent->name);
            new_node->entity.entity_type = ent->entity_type;
            new_node->entity.created = time(NULL);
            new_node->entity.updated = time(NULL);
            new_node->entity.mentions = 1;
            
            if (ent->user_id != NULL) {
                new_node->entity.user_id = strdup(ent->user_id);
            }
            if (ent->agent_id != NULL) {
                new_node->entity.agent_id = strdup(ent->agent_id);
            }
            if (ent->run_id != NULL) {
                new_node->entity.run_id = strdup(ent->run_id);
            }
            
            if (ent->embedding != NULL && ent->embedding_dim > 0) {
                new_node->entity.embedding = (float *)malloc(ent->embedding_dim * sizeof(float));
                if (new_node->entity.embedding != NULL) {
                    memcpy(new_node->entity.embedding, ent->embedding, ent->embedding_dim * sizeof(float));
                    new_node->entity.embedding_dim = ent->embedding_dim;
                }
            } else {
                /* Try embedding service first */
                GV_EmbeddingService *embedding_service = (GV_EmbeddingService *)graph->config.embedding_service;
                if (embedding_service != NULL) {
                    size_t dim = 0;
                    float *embedding = NULL;
                    if (gv_embedding_generate(embedding_service, new_node->entity.name, &dim, &embedding) == 0) {
                        new_node->entity.embedding = embedding;
                        new_node->entity.embedding_dim = dim;
                        if (graph->config.embedding_dimension == 0) {
                            graph->config.embedding_dimension = dim;
                        }
                    }
                } else if (graph->config.embedding_callback != NULL) {
                    /* Fall back to callback */
                    size_t dim = graph->config.embedding_dimension;
                    float *embedding = graph->config.embedding_callback(new_node->entity.name, &dim, graph->config.embedding_user_data);
                    if (embedding != NULL && dim > 0) {
                        new_node->entity.embedding = embedding;
                        new_node->entity.embedding_dim = dim;
                        if (graph->config.embedding_dimension == 0) {
                            graph->config.embedding_dimension = dim;
                        }
                    }
                }
            }
            
            /* Insert into hash table */
            new_node->next = graph->entity_table[hash];
            graph->entity_table[hash] = new_node;
        }
    }
    
    pthread_mutex_unlock(&graph->mutex);
    return 0;
}

int gv_context_graph_add_relationships(GV_ContextGraph *graph,
                                      const GV_GraphRelationship *relationships,
                                      size_t relationship_count) {
    if (graph == NULL || relationships == NULL) {
        return -1;
    }
    
    pthread_mutex_lock(&graph->mutex);
    
    for (size_t i = 0; i < relationship_count; i++) {
        const GV_GraphRelationship *rel = &relationships[i];
        
        /* Check if relationship already exists */
        size_t hash = hash_string(rel->source_entity_id, graph->relationship_table_size);
        RelationshipNode *node = graph->relationship_table[hash];
        RelationshipNode *found = NULL;
        
        while (node != NULL) {
            if (strcmp(node->relationship.source_entity_id, rel->source_entity_id) == 0 &&
                strcmp(node->relationship.destination_entity_id, rel->destination_entity_id) == 0 &&
                strcmp(node->relationship.relationship_type, rel->relationship_type) == 0) {
                found = node;
                break;
            }
            node = node->next;
        }
        
        if (found != NULL) {
            /* Update existing relationship */
            found->relationship.updated = time(NULL);
            found->relationship.mentions++;
        } else {
            /* Create new relationship */
            RelationshipNode *new_node = (RelationshipNode *)calloc(1, sizeof(RelationshipNode));
            if (new_node == NULL) {
                continue;
            }
            
            new_node->relationship_id = generate_relationship_id(graph->next_relationship_id++);
            if (new_node->relationship_id == NULL) {
                free(new_node);
                continue;
            }
            
            new_node->relationship.relationship_id = strdup(new_node->relationship_id);
            new_node->relationship.source_entity_id = strdup(rel->source_entity_id);
            new_node->relationship.destination_entity_id = strdup(rel->destination_entity_id);
            new_node->relationship.relationship_type = strdup(rel->relationship_type);
            new_node->relationship.created = time(NULL);
            new_node->relationship.updated = time(NULL);
            new_node->relationship.mentions = 1;
            
            /* Insert into hash table */
            new_node->next = graph->relationship_table[hash];
            graph->relationship_table[hash] = new_node;
        }
    }
    
    pthread_mutex_unlock(&graph->mutex);
    return 0;
}

int gv_context_graph_search(GV_ContextGraph *graph,
                             const float *query_embedding,
                             size_t embedding_dim,
                             const char *user_id,
                             const char *agent_id,
                             const char *run_id,
                             GV_GraphQueryResult *results,
                             size_t max_results) {
    if (graph == NULL || query_embedding == NULL || results == NULL) {
        return -1;
    }
    
    pthread_mutex_lock(&graph->mutex);
    
    size_t result_count = 0;
    
    /* Search all entities for similar embeddings */
    for (size_t i = 0; i < graph->entity_table_size && result_count < max_results; i++) {
        EntityNode *node = graph->entity_table[i];
        while (node != NULL && result_count < max_results) {
            /* Check filters */
            if (user_id != NULL && (node->entity.user_id == NULL || strcmp(node->entity.user_id, user_id) != 0)) {
                node = node->next;
                continue;
            }
            if (agent_id != NULL && (node->entity.agent_id == NULL || strcmp(node->entity.agent_id, agent_id) != 0)) {
                node = node->next;
                continue;
            }
            if (run_id != NULL && (node->entity.run_id == NULL || strcmp(node->entity.run_id, run_id) != 0)) {
                node = node->next;
                continue;
            }
            
            /* Calculate similarity */
            if (node->entity.embedding != NULL && node->entity.embedding_dim == embedding_dim) {
                float similarity = cosine_similarity(query_embedding, node->entity.embedding, embedding_dim);
                
                if (similarity >= graph->config.similarity_threshold) {
                    /* Find relationships for this entity */
                    size_t rel_hash = hash_string(node->entity_id, graph->relationship_table_size);
                    RelationshipNode *rel_node = graph->relationship_table[rel_hash];
                    
                    while (rel_node != NULL && result_count < max_results) {
                        if (strcmp(rel_node->relationship.source_entity_id, node->entity_id) == 0) {
                            /* Find destination entity */
                            EntityNode *dest_node = NULL;
                            for (size_t j = 0; j < graph->entity_table_size; j++) {
                                EntityNode *n = graph->entity_table[j];
                                while (n != NULL) {
                                    if (strcmp(n->entity_id, rel_node->relationship.destination_entity_id) == 0) {
                                        dest_node = n;
                                        break;
                                    }
                                    n = n->next;
                                }
                                if (dest_node != NULL) {
                                    break;
                                }
                            }
                            
                            if (dest_node != NULL) {
                                results[result_count].source_name = strdup(node->entity.name);
                                results[result_count].relationship_type = strdup(rel_node->relationship.relationship_type);
                                results[result_count].destination_name = strdup(dest_node->entity.name);
                                results[result_count].similarity = similarity;
                                result_count++;
                            }
                        }
                        rel_node = rel_node->next;
                    }
                }
            }
            
            node = node->next;
        }
    }
    
    pthread_mutex_unlock(&graph->mutex);
    return (int)result_count;
}

/* Helper: Find entity by ID */
static EntityNode *find_entity_by_id(GV_ContextGraph *graph, const char *entity_id) {
    if (graph == NULL || entity_id == NULL) {
        return NULL;
    }
    
    size_t hash = hash_string(entity_id, graph->entity_table_size);
    EntityNode *node = graph->entity_table[hash];
    while (node != NULL) {
        if (strcmp(node->entity_id, entity_id) == 0) {
            return node;
        }
        node = node->next;
    }
    return NULL;
}

/* Helper: Get relationships for an entity */
static void get_entity_relationships(GV_ContextGraph *graph, const char *entity_id,
                                     RelationshipNode **out_rels, size_t *out_count, size_t max_count) {
    if (graph == NULL || entity_id == NULL || out_rels == NULL || out_count == NULL) {
        return;
    }
    
    *out_count = 0;
    size_t hash = hash_string(entity_id, graph->relationship_table_size);
    RelationshipNode *node = graph->relationship_table[hash];
    
    while (node != NULL && *out_count < max_count) {
        if (strcmp(node->relationship.source_entity_id, entity_id) == 0) {
            out_rels[*out_count] = node;
            (*out_count)++;
        }
        node = node->next;
    }
    
    /* Also check reverse relationships */
    for (size_t i = 0; i < graph->relationship_table_size && *out_count < max_count; i++) {
        RelationshipNode *n = graph->relationship_table[i];
        while (n != NULL && *out_count < max_count) {
            if (strcmp(n->relationship.destination_entity_id, entity_id) == 0) {
                /* Check if already added */
                int found = 0;
                for (size_t j = 0; j < *out_count; j++) {
                    if (out_rels[j] == n) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    out_rels[*out_count] = n;
                    (*out_count)++;
                }
            }
            n = n->next;
        }
    }
}

int gv_context_graph_get_related(GV_ContextGraph *graph,
                                  const char *entity_id,
                                  size_t max_depth,
                                  GV_GraphQueryResult *results,
                                  size_t max_results) {
    if (graph == NULL || entity_id == NULL || results == NULL) {
        return -1;
    }
    
    pthread_mutex_lock(&graph->mutex);
    
    size_t result_count = 0;
    
    /* Find starting entity */
    EntityNode *start_entity = find_entity_by_id(graph, entity_id);
    if (start_entity == NULL) {
        pthread_mutex_unlock(&graph->mutex);
        return 0;
    }
    
    /* BFS traversal with depth limit */
    typedef struct {
        const char *entity_id;
        size_t depth;
    } QueueItem;
    
    QueueItem *queue = (QueueItem *)malloc(max_results * 2 * sizeof(QueueItem));
    if (queue == NULL) {
        pthread_mutex_unlock(&graph->mutex);
        return -1;
    }
    
    size_t queue_front = 0;
    size_t queue_back = 0;
    int *visited = (int *)calloc(graph->entity_table_size * 100, sizeof(int));
    if (visited == NULL) {
        free(queue);
        pthread_mutex_unlock(&graph->mutex);
        return -1;
    }
    
    /* Add starting entity to queue */
    queue[queue_back].entity_id = entity_id;
    queue[queue_back].depth = 0;
    queue_back++;
    
    size_t visited_count = 0;
    visited[visited_count++] = hash_string(entity_id, graph->entity_table_size);
    
    while (queue_front < queue_back && result_count < max_results) {
        QueueItem current = queue[queue_front++];
        
        if (current.depth >= max_depth) {
            continue;
        }
        
        /* Get relationships for current entity */
        RelationshipNode *rels[100];
        size_t rel_count = 0;
        get_entity_relationships(graph, current.entity_id, rels, &rel_count, 100);
        
        for (size_t i = 0; i < rel_count && result_count < max_results; i++) {
            RelationshipNode *rel = rels[i];
            const char *next_entity_id = NULL;
            EntityNode *next_entity = NULL;
            
            if (strcmp(rel->relationship.source_entity_id, current.entity_id) == 0) {
                next_entity_id = rel->relationship.destination_entity_id;
            } else {
                next_entity_id = rel->relationship.source_entity_id;
            }
            
            next_entity = find_entity_by_id(graph, next_entity_id);
            if (next_entity == NULL) {
                continue;
            }
            
            /* Check if already visited */
            int already_visited = 0;
            size_t next_hash = hash_string(next_entity_id, graph->entity_table_size);
            for (size_t j = 0; j < visited_count; j++) {
                if (visited[j] == (int)next_hash) {
                    already_visited = 1;
                    break;
                }
            }
            
            /* Add result */
            EntityNode *source_entity = find_entity_by_id(graph, rel->relationship.source_entity_id);
            EntityNode *dest_entity = find_entity_by_id(graph, rel->relationship.destination_entity_id);
            
            if (source_entity != NULL && dest_entity != NULL) {
                results[result_count].source_name = strdup(source_entity->entity.name);
                results[result_count].relationship_type = strdup(rel->relationship.relationship_type);
                results[result_count].destination_name = strdup(dest_entity->entity.name);
                results[result_count].similarity = 1.0f / (current.depth + 1.0f);  // Decay with depth
                result_count++;
            }
            
            /* Add to queue if not visited and within depth limit */
            if (!already_visited && current.depth + 1 < max_depth && queue_back < max_results * 2) {
                queue[queue_back].entity_id = next_entity_id;
                queue[queue_back].depth = current.depth + 1;
                queue_back++;
                visited[visited_count++] = (int)next_hash;
            }
        }
    }
    
    free(queue);
    free(visited);
    
    pthread_mutex_unlock(&graph->mutex);
    return (int)result_count;
}

int gv_context_graph_delete_entities(GV_ContextGraph *graph,
                                      const char **entity_ids,
                                      size_t entity_count) {
    if (graph == NULL || entity_ids == NULL) {
        return -1;
    }
    
    pthread_mutex_lock(&graph->mutex);
    
    for (size_t i = 0; i < entity_count; i++) {
        size_t hash = hash_string(entity_ids[i], graph->entity_table_size);
        EntityNode **node_ptr = &graph->entity_table[hash];
        EntityNode *node = *node_ptr;
        
        while (node != NULL) {
            if (strcmp(node->entity_id, entity_ids[i]) == 0) {
                *node_ptr = node->next;
                gv_graph_entity_free(&node->entity);
                free(node->entity_id);
                free(node);
                break;
            }
            node_ptr = &node->next;
            node = *node_ptr;
        }
    }
    
    pthread_mutex_unlock(&graph->mutex);
    return 0;
}

int gv_context_graph_delete_relationships(GV_ContextGraph *graph,
                                          const char **relationship_ids,
                                          size_t relationship_count) {
    if (graph == NULL || relationship_ids == NULL) {
        return -1;
    }
    
    pthread_mutex_lock(&graph->mutex);
    
    for (size_t i = 0; i < relationship_count; i++) {
        /* Search all buckets */
        for (size_t j = 0; j < graph->relationship_table_size; j++) {
            RelationshipNode **node_ptr = &graph->relationship_table[j];
            RelationshipNode *node = *node_ptr;
            
            while (node != NULL) {
                if (strcmp(node->relationship_id, relationship_ids[i]) == 0) {
                    *node_ptr = node->next;
                    gv_graph_relationship_free(&node->relationship);
                    free(node->relationship_id);
                    free(node);
                    goto next_relationship;
                }
                node_ptr = &node->next;
                node = *node_ptr;
            }
        }
        next_relationship:;
    }
    
    pthread_mutex_unlock(&graph->mutex);
    return 0;
}

void gv_graph_entity_free(GV_GraphEntity *entity) {
    if (entity == NULL) {
        return;
    }
    
    free(entity->entity_id);
    free(entity->name);
    free(entity->embedding);
    free(entity->user_id);
    free(entity->agent_id);
    free(entity->run_id);
    memset(entity, 0, sizeof(GV_GraphEntity));
}

void gv_graph_relationship_free(GV_GraphRelationship *relationship) {
    if (relationship == NULL) {
        return;
    }
    
    free(relationship->relationship_id);
    free(relationship->source_entity_id);
    free(relationship->destination_entity_id);
    free(relationship->relationship_type);
    memset(relationship, 0, sizeof(GV_GraphRelationship));
}

void gv_graph_query_result_free(GV_GraphQueryResult *result) {
    if (result == NULL) {
        return;
    }
    
    free(result->source_name);
    free(result->relationship_type);
    free(result->destination_name);
    memset(result, 0, sizeof(GV_GraphQueryResult));
}

