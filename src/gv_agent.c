/**
 * @file gv_agent.c
 * @brief Agentic interfaces: LLM-powered natural language to database operations.
 *
 * Implements three agent types:
 *  - Query Agent:          NL -> search vector + filter + k + distance type.
 *  - Transformation Agent: NL -> operation type (delete/update) + filter.
 *  - Personalization Agent: NL + user profile -> re-ranked search results.
 *
 * Thread-safe via pthread_mutex_t on the agent handle.  Uses gv_llm.h for
 * LLM calls and gv_json.h for structured response parsing.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <math.h>
#include <pthread.h>

#include "gigavector/gv_agent.h"
#include "gigavector/gv_llm.h"
#include "gigavector/gv_json.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_filter.h"
#include "gigavector/gv_embedding.h"
#include "gigavector/gv_soa_storage.h"

/*  Constants  */

#define AGENT_MAX_PROMPT_SIZE      (16 * 1024)
#define AGENT_MAX_USER_MSG_SIZE    (8 * 1024)
#define AGENT_MAX_FILTER_SIZE      1024
#define AGENT_MAX_RESPONSE_TEXT    4096
#define AGENT_DEFAULT_MODEL        "gpt-4o-mini"
#define AGENT_DEFAULT_TEMPERATURE  0.0f
#define AGENT_DEFAULT_MAX_RETRIES  2
#define AGENT_OVERSAMPLE_FACTOR    4

/*  Internal Agent Structure  */

struct GV_Agent {
    GV_AgentType type;
    GV_Database *db;                    /* Non-owning reference. */
    GV_LLM *llm;                       /* Owning handle to LLM instance. */
    GV_EmbeddingService *embed_svc;     /* Embedding service for text-to-vector; may be NULL. */
    char *schema_json;                  /* Optional schema hint (owned copy). */
    char *system_prompt;                /* Resolved system prompt (owned). */
    float temperature;
    int max_retries;
    pthread_mutex_t mutex;
};

/*  Prompt Templates  */

static const char *QUERY_AGENT_SYSTEM_PROMPT =
    "You are a vector database query assistant for GigaVector. "
    "Given a natural language query, you produce a JSON object with the following fields:\n"
    "{\n"
    "  \"search_text\": \"<text to embed as the search vector>\",\n"
    "  \"filter\": \"<optional filter expression or null>\",\n"
    "  \"k\": <number of results>,\n"
    "  \"distance\": \"<euclidean|cosine|dot_product|manhattan>\"\n"
    "}\n\n"
    "Filter expression syntax:\n"
    "  - Comparisons: field == \"value\", field != \"value\", field > 0.5, field >= 1, field < 10, field <= 100\n"
    "  - Logical: AND, OR, NOT\n"
    "  - Grouping: parentheses\n"
    "  - String ops: field CONTAINS \"substr\", field PREFIX \"prefix\"\n\n"
    "If no filter is needed, set filter to null.\n"
    "Always respond with valid JSON only, no markdown fences.";

static const char *TRANSFORM_AGENT_SYSTEM_PROMPT =
    "You are a vector database transformation assistant for GigaVector. "
    "Given a natural language instruction describing a data mutation, you produce a JSON object:\n"
    "{\n"
    "  \"operation\": \"<delete|update>\",\n"
    "  \"filter\": \"<filter expression selecting affected vectors>\",\n"
    "  \"update_metadata\": {\"key\": \"new_value\", ...}  // only for update operations\n"
    "}\n\n"
    "Filter expression syntax:\n"
    "  - Comparisons: field == \"value\", field != \"value\", field > 0.5\n"
    "  - Logical: AND, OR, NOT\n"
    "  - Grouping: parentheses\n"
    "  - String ops: field CONTAINS \"substr\", field PREFIX \"prefix\"\n\n"
    "Always respond with valid JSON only, no markdown fences.";

static const char *PERSONALIZE_AGENT_SYSTEM_PROMPT =
    "You are a personalization assistant for GigaVector. "
    "Given a user query and a JSON user profile, you produce a JSON object with "
    "per-attribute boost and demote factors to re-rank search results:\n"
    "{\n"
    "  \"search_text\": \"<text to embed as the search vector>\",\n"
    "  \"filter\": \"<optional filter expression or null>\",\n"
    "  \"distance\": \"<euclidean|cosine|dot_product|manhattan>\",\n"
    "  \"adjustments\": [\n"
    "    {\"field\": \"category\", \"value\": \"sports\", \"boost\": 1.5},\n"
    "    {\"field\": \"language\", \"value\": \"en\", \"boost\": 1.2},\n"
    "    {\"field\": \"region\", \"value\": \"spam\", \"boost\": 0.3}\n"
    "  ]\n"
    "}\n\n"
    "A boost > 1.0 promotes matching results; boost < 1.0 demotes them.\n"
    "Always respond with valid JSON only, no markdown fences.";

/*  Internal Helpers  */

/**
 * @brief Map provider string to GV_LLMProvider enum.
 */
static GV_LLMProvider provider_from_string(const char *provider) {
    if (provider == NULL) return GV_LLM_PROVIDER_OPENAI;
    if (strcmp(provider, "openai") == 0)    return GV_LLM_PROVIDER_OPENAI;
    if (strcmp(provider, "anthropic") == 0) return GV_LLM_PROVIDER_ANTHROPIC;
    if (strcmp(provider, "google") == 0)    return GV_LLM_PROVIDER_GOOGLE;
    return GV_LLM_PROVIDER_OPENAI;
}

/**
 * @brief Map distance string from LLM response to GV_DistanceType.
 */
static GV_DistanceType distance_from_string(const char *distance_str) {
    if (distance_str == NULL) return GV_DISTANCE_COSINE;
    if (strcmp(distance_str, "euclidean") == 0)    return GV_DISTANCE_EUCLIDEAN;
    if (strcmp(distance_str, "cosine") == 0)       return GV_DISTANCE_COSINE;
    if (strcmp(distance_str, "dot_product") == 0)  return GV_DISTANCE_DOT_PRODUCT;
    if (strcmp(distance_str, "manhattan") == 0)     return GV_DISTANCE_MANHATTAN;
    return GV_DISTANCE_COSINE;
}

/**
 * @brief Build the full system prompt with optional schema hint.
 */
static char *build_system_prompt(GV_AgentType type, const char *override,
                                 const char *schema_json, const GV_Database *db) {
    const char *base_prompt;

    if (override != NULL) {
        base_prompt = override;
    } else {
        switch (type) {
            case GV_AGENT_QUERY:       base_prompt = QUERY_AGENT_SYSTEM_PROMPT;       break;
            case GV_AGENT_TRANSFORM:   base_prompt = TRANSFORM_AGENT_SYSTEM_PROMPT;   break;
            case GV_AGENT_PERSONALIZE: base_prompt = PERSONALIZE_AGENT_SYSTEM_PROMPT; break;
            default:                   base_prompt = QUERY_AGENT_SYSTEM_PROMPT;       break;
        }
    }

    /* Calculate buffer size: base prompt + db info + schema */
    size_t needed = strlen(base_prompt) + 512;
    if (schema_json) {
        needed += strlen(schema_json) + 128;
    }

    char *prompt = (char *)malloc(needed);
    if (prompt == NULL) return NULL;

    size_t pos = 0;
    int written;

    /* Copy base prompt */
    written = snprintf(prompt + pos, needed - pos, "%s", base_prompt);
    if (written < 0 || (size_t)written >= needed - pos) {
        free(prompt);
        return NULL;
    }
    pos += (size_t)written;

    /* Append database context */
    if (db != NULL) {
        const char *index_name;
        switch (db->index_type) {
            case GV_INDEX_TYPE_KDTREE:  index_name = "kdtree";  break;
            case GV_INDEX_TYPE_HNSW:    index_name = "hnsw";    break;
            case GV_INDEX_TYPE_IVFPQ:   index_name = "ivfpq";   break;
            case GV_INDEX_TYPE_FLAT:    index_name = "flat";     break;
            case GV_INDEX_TYPE_IVFFLAT: index_name = "ivfflat";  break;
            case GV_INDEX_TYPE_PQ:      index_name = "pq";       break;
            case GV_INDEX_TYPE_LSH:     index_name = "lsh";      break;
            case GV_INDEX_TYPE_SPARSE:  index_name = "sparse";   break;
            default:                    index_name = "unknown";  break;
        }

        written = snprintf(prompt + pos, needed - pos,
                           "\n\nDatabase context: dimension=%zu, index_type=%s, vector_count=%zu.",
                           db->dimension, index_name, db->count);
        if (written > 0 && (size_t)written < needed - pos) {
            pos += (size_t)written;
        }
    }

    /* Append schema hint */
    if (schema_json != NULL) {
        written = snprintf(prompt + pos, needed - pos,
                           "\n\nAvailable metadata fields:\n%s", schema_json);
        if (written > 0 && (size_t)written < needed - pos) {
            pos += (size_t)written;
        }
    }

    return prompt;
}

/**
 * @brief Allocate and initialize an empty agent result.
 */
static GV_AgentResult *alloc_result(void) {
    GV_AgentResult *r = (GV_AgentResult *)calloc(1, sizeof(GV_AgentResult));
    return r;
}

/**
 * @brief Set an error on the result and return it.
 */
static GV_AgentResult *result_error(GV_AgentResult *r, const char *fmt, ...) {
    if (r == NULL) return NULL;
    r->success = 0;

    char buf[AGENT_MAX_RESPONSE_TEXT];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    r->error_message = strdup(buf);
    return r;
}

/**
 * @brief Call the LLM with retry logic.
 *
 * Sends a system + user message pair and returns the response content string
 * (caller must free).  Returns NULL on failure after all retries.
 */
static char *llm_call_with_retry(GV_Agent *agent, const char *system_prompt,
                                 const char *user_message) {
    GV_LLMMessage messages[2];
    messages[0].role = strdup("system");
    messages[0].content = strdup(system_prompt);
    messages[1].role = strdup("user");
    messages[1].content = strdup(user_message);

    if (!messages[0].role || !messages[0].content ||
        !messages[1].role || !messages[1].content) {
        free(messages[0].role);
        free(messages[0].content);
        free(messages[1].role);
        free(messages[1].content);
        return NULL;
    }

    int attempts = agent->max_retries + 1;
    char *content = NULL;

    for (int i = 0; i < attempts; i++) {
        GV_LLMResponse response;
        memset(&response, 0, sizeof(response));

        int rc = gv_llm_generate_response(agent->llm, messages, 2, "json_object", &response);
        if (rc == GV_LLM_SUCCESS && response.content != NULL) {
            content = strdup(response.content);
            gv_llm_response_free(&response);
            break;
        }
        gv_llm_response_free(&response);
    }

    free(messages[0].role);
    free(messages[0].content);
    free(messages[1].role);
    free(messages[1].content);

    return content;
}

/**
 * @brief Parse a JSON number value, returning a default on failure.
 */
static double json_get_number_or(const GV_JsonValue *obj, const char *key, double fallback) {
    GV_JsonValue *val = gv_json_object_get(obj, key);
    if (val == NULL || !gv_json_is_number(val)) return fallback;
    double out;
    if (gv_json_get_number(val, &out) != GV_JSON_OK) return fallback;
    return out;
}

/**
 * @brief Parse a JSON string value, returning NULL on failure.
 */
static const char *json_get_string_or_null(const GV_JsonValue *obj, const char *key) {
    GV_JsonValue *val = gv_json_object_get(obj, key);
    if (val == NULL || !gv_json_is_string(val)) return NULL;
    return gv_json_get_string(val);
}

/*  Lifecycle  */

GV_Agent *gv_agent_create(const void *db, const GV_AgentConfig *config) {
    if (db == NULL || config == NULL) return NULL;
    if (config->api_key == NULL) return NULL;

    GV_Agent *agent = (GV_Agent *)calloc(1, sizeof(GV_Agent));
    if (agent == NULL) return NULL;

    agent->type = (GV_AgentType)config->agent_type;
    agent->db = (GV_Database *)db;  /* Cast away const; mutations only in transform agent. */

    /* Configure LLM */
    GV_LLMConfig llm_cfg;
    memset(&llm_cfg, 0, sizeof(llm_cfg));
    llm_cfg.provider = provider_from_string(config->llm_provider);
    llm_cfg.api_key = strdup(config->api_key);
    if (llm_cfg.api_key == NULL) {
        free(agent);
        return NULL;
    }
    llm_cfg.model = strdup(config->model ? config->model : AGENT_DEFAULT_MODEL);
    if (llm_cfg.model == NULL) {
        free(llm_cfg.api_key);
        free(agent);
        return NULL;
    }
    llm_cfg.temperature = (config->temperature >= 0.0f) ? config->temperature : AGENT_DEFAULT_TEMPERATURE;
    llm_cfg.max_tokens = 2048;
    llm_cfg.timeout_seconds = 60;
    llm_cfg.base_url = NULL;
    llm_cfg.custom_prompt = NULL;

    agent->llm = gv_llm_create(&llm_cfg);
    free(llm_cfg.api_key);
    free(llm_cfg.model);
    if (agent->llm == NULL) {
        free(agent);
        return NULL;
    }

    agent->temperature = (config->temperature >= 0.0f) ? config->temperature : AGENT_DEFAULT_TEMPERATURE;
    agent->max_retries = (config->max_retries > 0) ? config->max_retries : AGENT_DEFAULT_MAX_RETRIES;
    agent->schema_json = NULL;
    agent->embed_svc = NULL;

    /* Build system prompt */
    agent->system_prompt = build_system_prompt(agent->type, config->system_prompt_override,
                                               NULL, agent->db);
    if (agent->system_prompt == NULL) {
        gv_llm_destroy(agent->llm);
        free(agent);
        return NULL;
    }

    if (pthread_mutex_init(&agent->mutex, NULL) != 0) {
        free(agent->system_prompt);
        gv_llm_destroy(agent->llm);
        free(agent);
        return NULL;
    }

    return agent;
}

void gv_agent_destroy(GV_Agent *agent) {
    if (agent == NULL) return;

    pthread_mutex_destroy(&agent->mutex);
    gv_llm_destroy(agent->llm);
    free(agent->system_prompt);
    free(agent->schema_json);
    /* embed_svc is not owned; caller manages its lifecycle. */
    free(agent);
}

/*  Schema Hints  */

void gv_agent_set_schema_hint(GV_Agent *agent, const char *schema_json) {
    if (agent == NULL || schema_json == NULL) return;

    pthread_mutex_lock(&agent->mutex);

    free(agent->schema_json);
    agent->schema_json = strdup(schema_json);

    /* Rebuild system prompt with the new schema */
    free(agent->system_prompt);
    agent->system_prompt = build_system_prompt(agent->type, NULL,
                                               agent->schema_json, agent->db);

    pthread_mutex_unlock(&agent->mutex);
}

/*  Query Agent  */

GV_AgentResult *gv_agent_query(GV_Agent *agent, const char *natural_language_query, size_t k) {
    GV_AgentResult *result = alloc_result();
    if (result == NULL) return NULL;

    if (agent == NULL || natural_language_query == NULL) {
        return result_error(result, "agent and natural_language_query must be non-NULL");
    }
    if (agent->type != GV_AGENT_QUERY) {
        return result_error(result, "agent type must be GV_AGENT_QUERY for query operations");
    }

    pthread_mutex_lock(&agent->mutex);

    /* Build user message */
    char user_msg[AGENT_MAX_USER_MSG_SIZE];
    snprintf(user_msg, sizeof(user_msg),
             "Query: \"%s\"\nReturn at most %zu results.", natural_language_query, k);

    /* Call LLM */
    char *llm_response = llm_call_with_retry(agent, agent->system_prompt, user_msg);
    if (llm_response == NULL) {
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "LLM call failed after retries");
    }

    /* Parse JSON response */
    GV_JsonError json_err;
    GV_JsonValue *root = gv_json_parse(llm_response, &json_err);
    if (root == NULL) {
        pthread_mutex_unlock(&agent->mutex);
        result_error(result, "failed to parse LLM JSON response: %s",
                     gv_json_error_string(json_err));
        free(llm_response);
        return result;
    }

    /* Extract fields */
    const char *search_text = json_get_string_or_null(root, "search_text");
    const char *filter_str  = json_get_string_or_null(root, "filter");
    double resp_k           = json_get_number_or(root, "k", (double)k);
    const char *dist_str    = json_get_string_or_null(root, "distance");

    size_t effective_k = (size_t)resp_k;
    if (effective_k == 0 || effective_k > k) effective_k = k;

    GV_DistanceType dist_type = distance_from_string(dist_str);

    /* Save response text */
    result->response_text = strdup(llm_response);

    /* Save generated filter */
    if (filter_str != NULL) {
        result->generated_filter = strdup(filter_str);
    }

    /* Embed the search text to obtain a query vector */
    float *query_vec = NULL;
    size_t embed_dim = 0;

    if (search_text != NULL && agent->embed_svc != NULL) {
        int embed_rc = gv_embedding_generate(agent->embed_svc, search_text, &embed_dim, &query_vec);
        if (embed_rc != 0 || query_vec == NULL) {
            gv_json_free(root);
            free(llm_response);
            pthread_mutex_unlock(&agent->mutex);
            return result_error(result, "embedding generation failed for search text");
        }
    }

    /* If no embedding service, try to parse a vector array from the JSON response */
    if (query_vec == NULL) {
        GV_JsonValue *vec_val = gv_json_object_get(root, "search_vector");
        if (vec_val != NULL && gv_json_is_array(vec_val)) {
            size_t vec_len = gv_json_array_length(vec_val);
            if (vec_len > 0 && vec_len == agent->db->dimension) {
                query_vec = (float *)malloc(vec_len * sizeof(float));
                if (query_vec != NULL) {
                    embed_dim = vec_len;
                    for (size_t i = 0; i < vec_len; i++) {
                        GV_JsonValue *elem = gv_json_array_get(vec_val, i);
                        double v = 0.0;
                        if (elem && gv_json_is_number(elem)) {
                            gv_json_get_number(elem, &v);
                        }
                        query_vec[i] = (float)v;
                    }
                }
            }
        }
    }

    gv_json_free(root);
    free(llm_response);

    /* Execute the search if we have a query vector */
    if (query_vec != NULL && embed_dim == agent->db->dimension) {
        GV_SearchResult *sr = (GV_SearchResult *)calloc(effective_k, sizeof(GV_SearchResult));
        if (sr == NULL) {
            free(query_vec);
            pthread_mutex_unlock(&agent->mutex);
            return result_error(result, "memory allocation failed for search results");
        }

        int found;
        if (result->generated_filter != NULL) {
            found = gv_db_search_with_filter_expr(agent->db, query_vec, effective_k,
                                                   sr, dist_type, result->generated_filter);
        } else {
            found = gv_db_search(agent->db, query_vec, effective_k, sr, dist_type);
        }

        free(query_vec);

        if (found < 0) {
            free(sr);
            pthread_mutex_unlock(&agent->mutex);
            return result_error(result, "database search failed");
        }

        /* Convert SearchResult to indices + distances */
        result->result_count = (size_t)found;
        result->result_indices = (size_t *)malloc((size_t)found * sizeof(size_t));
        result->result_distances = (float *)malloc((size_t)found * sizeof(float));

        if (result->result_indices == NULL || result->result_distances == NULL) {
            free(result->result_indices);
            result->result_indices = NULL;
            free(result->result_distances);
            result->result_distances = NULL;
            result->result_count = 0;
            free(sr);
            pthread_mutex_unlock(&agent->mutex);
            return result_error(result, "memory allocation failed");
        }

        for (int i = 0; i < found; i++) {
            /* Recover SoA index from vector pointer. */
            const float *base = gv_database_get_vector(agent->db, 0);
            size_t dim = gv_database_dimension(agent->db);
            if (base != NULL && dim > 0 && sr[i].vector != NULL && sr[i].vector->data != NULL) {
                ptrdiff_t diff = sr[i].vector->data - base;
                result->result_indices[i] = (diff >= 0) ? (size_t)diff / dim : 0;
            } else {
                result->result_indices[i] = 0;
            }
            result->result_distances[i] = sr[i].distance;
        }

        free(sr);
        result->success = 1;
    } else {
        free(query_vec);
        result_error(result, "could not obtain a valid query vector from LLM response");
    }

    pthread_mutex_unlock(&agent->mutex);
    return result;
}

/*  Transformation Agent  */

GV_AgentResult *gv_agent_transform(GV_Agent *agent, const char *natural_language_instruction) {
    GV_AgentResult *result = alloc_result();
    if (result == NULL) return NULL;

    if (agent == NULL || natural_language_instruction == NULL) {
        return result_error(result, "agent and natural_language_instruction must be non-NULL");
    }
    if (agent->type != GV_AGENT_TRANSFORM) {
        return result_error(result, "agent type must be GV_AGENT_TRANSFORM for transform operations");
    }

    pthread_mutex_lock(&agent->mutex);

    /* Build user message */
    char user_msg[AGENT_MAX_USER_MSG_SIZE];
    snprintf(user_msg, sizeof(user_msg), "Instruction: \"%s\"", natural_language_instruction);

    /* Call LLM */
    char *llm_response = llm_call_with_retry(agent, agent->system_prompt, user_msg);
    if (llm_response == NULL) {
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "LLM call failed after retries");
    }

    /* Parse JSON response */
    GV_JsonError json_err;
    GV_JsonValue *root = gv_json_parse(llm_response, &json_err);
    if (root == NULL) {
        pthread_mutex_unlock(&agent->mutex);
        result_error(result, "failed to parse LLM JSON response: %s",
                     gv_json_error_string(json_err));
        free(llm_response);
        return result;
    }

    const char *operation = json_get_string_or_null(root, "operation");
    const char *filter_str = json_get_string_or_null(root, "filter");

    result->response_text = strdup(llm_response);
    if (filter_str != NULL) {
        result->generated_filter = strdup(filter_str);
    }

    free(llm_response);

    if (operation == NULL || filter_str == NULL) {
        gv_json_free(root);
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "LLM response missing 'operation' or 'filter' field");
    }

    /* Parse the filter expression to validate it */
    GV_Filter *filter = gv_filter_parse(filter_str);
    if (filter == NULL) {
        gv_json_free(root);
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "failed to parse filter expression: %s", filter_str);
    }

    /* Execute the operation */
    if (strcmp(operation, "delete") == 0) {
        /* Walk all vectors and delete those matching the filter */
        size_t total = gv_database_count(agent->db);
        size_t deleted = 0;

        /* Collect matching indices first to avoid mutation during iteration */
        size_t *match_indices = (size_t *)malloc(total * sizeof(size_t));
        size_t match_count = 0;

        if (match_indices != NULL) {
            for (size_t i = 0; i < total; i++) {
                const float *vec_data = gv_database_get_vector(agent->db, i);
                if (vec_data == NULL) continue;

                /* Build a temporary vector for filter evaluation */
                GV_Vector tmp_vec;
                tmp_vec.dimension = agent->db->dimension;
                tmp_vec.data = (float *)vec_data;
                tmp_vec.metadata = NULL;

                /* Retrieve metadata from SoA storage if available */
                if (agent->db->soa_storage != NULL) {
                    tmp_vec.metadata = gv_soa_storage_get_metadata(agent->db->soa_storage, i);
                }

                if (gv_filter_eval(filter, &tmp_vec) == 1) {
                    match_indices[match_count++] = i;
                }
            }

            /* Delete in reverse order to avoid index shifting issues */
            for (size_t j = match_count; j > 0; j--) {
                if (gv_db_delete_vector_by_index(agent->db, match_indices[j - 1]) == 0) {
                    deleted++;
                }
            }

            free(match_indices);
        }

        result->result_count = deleted;
        result->success = 1;

    } else if (strcmp(operation, "update") == 0) {
        /* Extract update_metadata from the response */
        GV_JsonValue *update_obj = gv_json_object_get(root, "update_metadata");

        size_t total = gv_database_count(agent->db);
        size_t updated = 0;

        for (size_t i = 0; i < total; i++) {
            const float *vec_data = gv_database_get_vector(agent->db, i);
            if (vec_data == NULL) continue;

            GV_Vector tmp_vec;
            tmp_vec.dimension = agent->db->dimension;
            tmp_vec.data = (float *)vec_data;
            tmp_vec.metadata = NULL;

            if (agent->db->soa_storage != NULL) {
                tmp_vec.metadata = gv_soa_storage_get_metadata(agent->db->soa_storage, i);
            }

            if (gv_filter_eval(filter, &tmp_vec) != 1) continue;

            /* Apply metadata updates */
            if (update_obj != NULL && gv_json_is_object(update_obj)) {
                size_t n_keys = gv_json_object_length(update_obj);
                if (n_keys > 0) {
                    const char **keys = (const char **)malloc(n_keys * sizeof(const char *));
                    const char **vals = (const char **)malloc(n_keys * sizeof(const char *));
                    char **val_copies = (char **)calloc(n_keys, sizeof(char *));

                    if (keys && vals && val_copies) {
                        size_t ki = 0;
                        for (size_t e = 0; e < update_obj->data.object.count && ki < n_keys; e++) {
                            keys[ki] = update_obj->data.object.entries[e].key;
                            const char *sv = gv_json_get_string(update_obj->data.object.entries[e].value);
                            if (sv != NULL) {
                                val_copies[ki] = strdup(sv);
                                vals[ki] = val_copies[ki];
                            } else {
                                /* For non-string values, stringify */
                                char *s = gv_json_stringify(update_obj->data.object.entries[e].value, 0);
                                val_copies[ki] = s;
                                vals[ki] = s ? s : "";
                            }
                            ki++;
                        }

                        if (gv_db_update_vector_metadata(agent->db, i, keys, vals, ki) == 0) {
                            updated++;
                        }

                        for (size_t c = 0; c < ki; c++) {
                            free(val_copies[c]);
                        }
                    }

                    free(keys);
                    free(vals);
                    free(val_copies);
                }
            } else {
                /* No metadata updates specified; just count the match */
                updated++;
            }
        }

        result->result_count = updated;
        result->success = 1;

    } else {
        gv_filter_destroy(filter);
        gv_json_free(root);
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "unsupported operation: %s", operation);
    }

    gv_filter_destroy(filter);
    gv_json_free(root);
    pthread_mutex_unlock(&agent->mutex);
    return result;
}

/*  Personalization Agent  */

/**
 * @brief Compare helper for qsort: sort by adjusted distance ascending.
 */
typedef struct {
    size_t index;
    float  distance;
    float  adjusted_distance;
} PersonalizedEntry;

static int personalized_cmp(const void *a, const void *b) {
    const PersonalizedEntry *ea = (const PersonalizedEntry *)a;
    const PersonalizedEntry *eb = (const PersonalizedEntry *)b;
    if (ea->adjusted_distance < eb->adjusted_distance) return -1;
    if (ea->adjusted_distance > eb->adjusted_distance) return  1;
    return 0;
}

GV_AgentResult *gv_agent_personalize(GV_Agent *agent, const char *query,
                                     const char *user_profile_json, size_t k) {
    GV_AgentResult *result = alloc_result();
    if (result == NULL) return NULL;

    if (agent == NULL || query == NULL || user_profile_json == NULL) {
        return result_error(result, "agent, query, and user_profile_json must be non-NULL");
    }
    if (agent->type != GV_AGENT_PERSONALIZE) {
        return result_error(result, "agent type must be GV_AGENT_PERSONALIZE for personalize operations");
    }

    pthread_mutex_lock(&agent->mutex);

    /* Build user message with query and profile */
    size_t msg_size = strlen(query) + strlen(user_profile_json) + 256;
    char *user_msg = (char *)malloc(msg_size);
    if (user_msg == NULL) {
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "memory allocation failed");
    }
    snprintf(user_msg, msg_size,
             "Query: \"%s\"\nUser profile:\n%s\nReturn at most %zu results.",
             query, user_profile_json, k);

    /* Call LLM */
    char *llm_response = llm_call_with_retry(agent, agent->system_prompt, user_msg);
    free(user_msg);
    if (llm_response == NULL) {
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "LLM call failed after retries");
    }

    /* Parse JSON response */
    GV_JsonError json_err;
    GV_JsonValue *root = gv_json_parse(llm_response, &json_err);
    if (root == NULL) {
        pthread_mutex_unlock(&agent->mutex);
        result_error(result, "failed to parse LLM JSON response: %s",
                     gv_json_error_string(json_err));
        free(llm_response);
        return result;
    }

    const char *search_text = json_get_string_or_null(root, "search_text");
    const char *filter_str  = json_get_string_or_null(root, "filter");
    const char *dist_str    = json_get_string_or_null(root, "distance");

    GV_DistanceType dist_type = distance_from_string(dist_str);

    result->response_text = strdup(llm_response);
    if (filter_str != NULL) {
        result->generated_filter = strdup(filter_str);
    }

    free(llm_response);

    /* Embed search text */
    float *query_vec = NULL;
    size_t embed_dim = 0;

    if (search_text != NULL && agent->embed_svc != NULL) {
        int embed_rc = gv_embedding_generate(agent->embed_svc, search_text, &embed_dim, &query_vec);
        if (embed_rc != 0 || query_vec == NULL) {
            gv_json_free(root);
            pthread_mutex_unlock(&agent->mutex);
            return result_error(result, "embedding generation failed for search text");
        }
    }

    /* Fallback: try to parse a vector array */
    if (query_vec == NULL) {
        GV_JsonValue *vec_val = gv_json_object_get(root, "search_vector");
        if (vec_val != NULL && gv_json_is_array(vec_val)) {
            size_t vec_len = gv_json_array_length(vec_val);
            if (vec_len > 0 && vec_len == agent->db->dimension) {
                query_vec = (float *)malloc(vec_len * sizeof(float));
                if (query_vec != NULL) {
                    embed_dim = vec_len;
                    for (size_t i = 0; i < vec_len; i++) {
                        GV_JsonValue *elem = gv_json_array_get(vec_val, i);
                        double v = 0.0;
                        if (elem && gv_json_is_number(elem)) {
                            gv_json_get_number(elem, &v);
                        }
                        query_vec[i] = (float)v;
                    }
                }
            }
        }
    }

    if (query_vec == NULL || embed_dim != agent->db->dimension) {
        free(query_vec);
        gv_json_free(root);
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "could not obtain a valid query vector from LLM response");
    }

    /* Oversample: fetch more results to allow re-ranking */
    size_t oversample_k = k * AGENT_OVERSAMPLE_FACTOR;
    if (oversample_k < k) oversample_k = k;  /* Overflow guard. */

    GV_SearchResult *sr = (GV_SearchResult *)calloc(oversample_k, sizeof(GV_SearchResult));
    if (sr == NULL) {
        free(query_vec);
        gv_json_free(root);
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "memory allocation failed for search results");
    }

    int found;
    if (result->generated_filter != NULL) {
        found = gv_db_search_with_filter_expr(agent->db, query_vec, oversample_k,
                                               sr, dist_type, result->generated_filter);
    } else {
        found = gv_db_search(agent->db, query_vec, oversample_k, sr, dist_type);
    }

    free(query_vec);

    if (found < 0) {
        free(sr);
        gv_json_free(root);
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "database search failed");
    }

    /* Parse personalization adjustments */
    GV_JsonValue *adjustments = gv_json_object_get(root, "adjustments");

    /* Build personalized entries */
    PersonalizedEntry *entries = (PersonalizedEntry *)malloc((size_t)found * sizeof(PersonalizedEntry));
    if (entries == NULL) {
        free(sr);
        gv_json_free(root);
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "memory allocation failed");
    }

    const float *base = gv_database_get_vector(agent->db, 0);
    size_t dim = gv_database_dimension(agent->db);

    for (int i = 0; i < found; i++) {
        entries[i].distance = sr[i].distance;
        entries[i].adjusted_distance = sr[i].distance;

        /* Recover vector index */
        if (base != NULL && dim > 0 && sr[i].vector != NULL && sr[i].vector->data != NULL) {
            ptrdiff_t diff = sr[i].vector->data - base;
            entries[i].index = (diff >= 0) ? (size_t)diff / dim : 0;
        } else {
            entries[i].index = 0;
        }

        /* Apply boost/demote adjustments based on metadata matches */
        if (adjustments != NULL && gv_json_is_array(adjustments)) {
            GV_Metadata *meta = sr[i].vector ? sr[i].vector->metadata : NULL;
            size_t adj_count = gv_json_array_length(adjustments);

            for (size_t a = 0; a < adj_count; a++) {
                GV_JsonValue *adj = gv_json_array_get(adjustments, a);
                if (adj == NULL || !gv_json_is_object(adj)) continue;

                const char *field = json_get_string_or_null(adj, "field");
                const char *value = json_get_string_or_null(adj, "value");
                double boost = json_get_number_or(adj, "boost", 1.0);

                if (field == NULL || value == NULL) continue;

                /* Check if vector metadata matches the field/value */
                GV_Metadata *m = meta;
                while (m != NULL) {
                    if (m->key != NULL && m->value != NULL &&
                        strcmp(m->key, field) == 0 && strcmp(m->value, value) == 0) {
                        /* Adjust distance: lower distance = more relevant.
                         * Multiply by inverse of boost so boost > 1 reduces distance. */
                        if (boost > 0.0) {
                            entries[i].adjusted_distance *= (float)(1.0 / boost);
                        }
                        break;
                    }
                    m = m->next;
                }
            }
        }
    }

    free(sr);
    gv_json_free(root);

    /* Sort by adjusted distance */
    qsort(entries, (size_t)found, sizeof(PersonalizedEntry), personalized_cmp);

    /* Take top k */
    size_t final_count = ((size_t)found < k) ? (size_t)found : k;
    result->result_indices = (size_t *)malloc(final_count * sizeof(size_t));
    result->result_distances = (float *)malloc(final_count * sizeof(float));

    if (result->result_indices == NULL || result->result_distances == NULL) {
        free(result->result_indices);
        result->result_indices = NULL;
        free(result->result_distances);
        result->result_distances = NULL;
        free(entries);
        pthread_mutex_unlock(&agent->mutex);
        return result_error(result, "memory allocation failed");
    }

    for (size_t i = 0; i < final_count; i++) {
        result->result_indices[i] = entries[i].index;
        result->result_distances[i] = entries[i].adjusted_distance;
    }

    result->result_count = final_count;
    result->success = 1;

    free(entries);
    pthread_mutex_unlock(&agent->mutex);
    return result;
}

/*  Result Cleanup  */

void gv_agent_free_result(GV_AgentResult *result) {
    if (result == NULL) return;

    free(result->response_text);
    free(result->result_indices);
    free(result->result_distances);
    free(result->generated_filter);
    free(result->error_message);
    free(result);
}
