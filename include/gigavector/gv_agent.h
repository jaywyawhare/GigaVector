#ifndef GIGAVECTOR_GV_AGENT_H
#define GIGAVECTOR_GV_AGENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_agent.h
 * @brief Agentic interfaces for GigaVector.
 *
 * Provides LLM-powered agents that translate natural language into database
 * operations:
 *  - Query Agent:         NL query -> vector search with optional filtering.
 *  - Transformation Agent: NL instruction -> data mutations (delete/update).
 *  - Personalization Agent: NL query + user profile -> re-ranked results.
 *
 * Agents use the gv_llm.h API for LLM calls and gv_json.h for structured
 * response parsing.  All operations are thread-safe.
 */

/* ============================================================================
 * Agent Type Enumeration
 * ============================================================================ */

/**
 * @brief Agent type enumeration.
 */
typedef enum {
    GV_AGENT_QUERY       = 0,   /**< Natural language -> vector search. */
    GV_AGENT_TRANSFORM   = 1,   /**< Natural language -> data mutations. */
    GV_AGENT_PERSONALIZE = 2    /**< Natural language + user profile -> re-ranked results. */
} GV_AgentType;

/* ============================================================================
 * Agent Configuration
 * ============================================================================ */

/**
 * @brief Agent configuration structure.
 *
 * Defaults:
 *  - model:         "gpt-4o-mini"
 *  - temperature:   0.0
 *  - max_retries:   2
 *  - system_prompt_override: NULL (use built-in prompt for agent type)
 */
typedef struct {
    int agent_type;                     /**< GV_AgentType value. */
    const char *llm_provider;           /**< Provider string: "openai", "anthropic", or "google". */
    const char *api_key;                /**< API key for the LLM provider. */
    const char *model;                  /**< Model name (default: "gpt-4o-mini"). */
    float temperature;                  /**< Sampling temperature (default: 0.0). */
    int max_retries;                    /**< Maximum LLM call retries (default: 2). */
    const char *system_prompt_override; /**< Custom system prompt; NULL for built-in default. */
} GV_AgentConfig;

/* ============================================================================
 * Agent Handle (Opaque)
 * ============================================================================ */

/**
 * @brief Opaque agent handle.
 */
typedef struct GV_Agent GV_Agent;

/* ============================================================================
 * Agent Result
 * ============================================================================ */

/**
 * @brief Result structure returned by agent operations.
 *
 * For query/personalize agents, result_indices and result_distances contain
 * the search results.  For transform agents, result_count indicates the
 * number of affected rows.
 */
typedef struct {
    int success;                /**< 1 on success, 0 on failure. */
    char *response_text;        /**< Human-readable explanation from the LLM. */
    size_t *result_indices;     /**< Array of matching vector indices (query/personalize). */
    float *result_distances;    /**< Array of distances for each result (query/personalize). */
    size_t result_count;        /**< Number of results or affected rows. */
    char *generated_filter;     /**< Filter expression the agent chose (caller must not free). */
    char *error_message;        /**< Error description on failure; NULL on success. */
} GV_AgentResult;

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/**
 * @brief Create an agent bound to a database.
 *
 * The agent holds a read-only reference to @p db.  The database must remain
 * valid for the lifetime of the agent.
 *
 * @param db     Database handle (cast to const void* to avoid circular includes);
 *               must be non-NULL.
 * @param config Agent configuration; must be non-NULL.
 * @return Allocated agent handle, or NULL on failure.
 */
GV_Agent *gv_agent_create(const void *db, const GV_AgentConfig *config);

/**
 * @brief Destroy an agent and release all resources.
 *
 * Safe to call with NULL.
 *
 * @param agent Agent handle.
 */
void gv_agent_destroy(GV_Agent *agent);

/* ============================================================================
 * Agent Operations
 * ============================================================================ */

/**
 * @brief Execute a natural-language query.
 *
 * The agent sends the query to the LLM, which produces a structured response
 * containing a search vector (or text to embed), an optional filter expression,
 * the number of results, and a distance metric.  The agent then executes the
 * search against the bound database.
 *
 * Only valid for GV_AGENT_QUERY agents.
 *
 * @param agent                Agent handle; must be non-NULL.
 * @param natural_language_query  Free-form query string; must be non-NULL.
 * @param k                    Maximum number of results to return.
 * @return Allocated result (caller must free with gv_agent_free_result()),
 *         or NULL on allocation failure.
 */
GV_AgentResult *gv_agent_query(GV_Agent *agent, const char *natural_language_query, size_t k);

/**
 * @brief Execute a natural-language data transformation.
 *
 * The agent sends the instruction to the LLM, which produces an operation type
 * (delete or update) and a filter expression.  The agent then executes the
 * mutation against the bound database.
 *
 * Only valid for GV_AGENT_TRANSFORM agents.
 *
 * @param agent                      Agent handle; must be non-NULL.
 * @param natural_language_instruction  Free-form instruction; must be non-NULL.
 * @return Allocated result (caller must free with gv_agent_free_result()),
 *         or NULL on allocation failure.
 */
GV_AgentResult *gv_agent_transform(GV_Agent *agent, const char *natural_language_instruction);

/**
 * @brief Execute a personalized query.
 *
 * The agent sends the query together with the user profile to the LLM, which
 * returns per-attribute boost/demote factors.  The agent executes a base
 * search and re-ranks results using these factors.
 *
 * Only valid for GV_AGENT_PERSONALIZE agents.
 *
 * @param agent             Agent handle; must be non-NULL.
 * @param query             Natural-language query; must be non-NULL.
 * @param user_profile_json JSON string describing user preferences; must be non-NULL.
 * @param k                 Maximum number of results to return.
 * @return Allocated result (caller must free with gv_agent_free_result()),
 *         or NULL on allocation failure.
 */
GV_AgentResult *gv_agent_personalize(GV_Agent *agent, const char *query,
                                     const char *user_profile_json, size_t k);

/* ============================================================================
 * Result Cleanup
 * ============================================================================ */

/**
 * @brief Free an agent result and all owned memory.
 *
 * Safe to call with NULL.
 *
 * @param result Result to free.
 */
void gv_agent_free_result(GV_AgentResult *result);

/* ============================================================================
 * Schema Hints
 * ============================================================================ */

/**
 * @brief Provide a schema hint to the agent.
 *
 * The schema JSON describes available metadata fields (names, types, example
 * values).  This information is included in the system prompt so the LLM can
 * generate correct filter expressions and understand the data layout.
 *
 * Example schema_json:
 * @code
 * {
 *   "fields": [
 *     {"name": "category", "type": "string", "examples": ["news", "sports"]},
 *     {"name": "score",    "type": "float",  "range": [0.0, 1.0]}
 *   ]
 * }
 * @endcode
 *
 * @param agent       Agent handle; must be non-NULL.
 * @param schema_json JSON string describing metadata fields; must be non-NULL.
 *                    The string is copied internally.
 */
void gv_agent_set_schema_hint(GV_Agent *agent, const char *schema_json);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_AGENT_H */
