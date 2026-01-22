#ifndef GIGAVECTOR_GV_LLM_H
#define GIGAVECTOR_GV_LLM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief LLM provider enumeration.
 *
 * Supported providers:
 * - OPENAI: OpenAI GPT models (tested, recommended)
 * - GOOGLE: Google Gemini models (tested)
 * - CUSTOM: Custom OpenAI-compatible endpoints
 *
 * Internal/experimental (not exposed to end users):
 * - ANTHROPIC: Claude models (not yet tested due to API key unavailability)
 */
typedef enum {
    GV_LLM_PROVIDER_OPENAI = 0,
    GV_LLM_PROVIDER_ANTHROPIC = 1,      /* Internal: not yet tested, API keys unavailable */
    GV_LLM_PROVIDER_GOOGLE = 2,
    /* GV_LLM_PROVIDER_AZURE_OPENAI removed - use CUSTOM with Azure endpoint instead */
    GV_LLM_PROVIDER_CUSTOM = 3
} GV_LLMProvider;

/**
 * @brief LLM error codes.
 */
typedef enum {
    GV_LLM_SUCCESS = 0,
    GV_LLM_ERROR_NULL_POINTER = -1,
    GV_LLM_ERROR_INVALID_CONFIG = -2,
    GV_LLM_ERROR_INVALID_API_KEY = -3,
    GV_LLM_ERROR_INVALID_URL = -4,
    GV_LLM_ERROR_MEMORY_ALLOCATION = -5,
    GV_LLM_ERROR_CURL_INIT = -6,
    GV_LLM_ERROR_NETWORK = -7,
    GV_LLM_ERROR_TIMEOUT = -8,
    GV_LLM_ERROR_RESPONSE_TOO_LARGE = -9,
    GV_LLM_ERROR_PARSE_FAILED = -10,
    GV_LLM_ERROR_INVALID_RESPONSE = -11,
    GV_LLM_ERROR_CUSTOM_URL_REQUIRED = -12
} GV_LLMError;

/**
 * @brief LLM configuration structure.
 */
typedef struct {
    GV_LLMProvider provider;          /**< LLM provider type. */
    char *api_key;                    /**< API key for authentication. */
    char *model;                      /**< Model name (e.g., "gpt-4", "claude-3-opus"). */
    char *base_url;                   /**< Base URL for API (NULL for default). */
    double temperature;               /**< Temperature for generation (0.0-2.0). */
    int max_tokens;                   /**< Maximum tokens in response. */
    int timeout_seconds;              /**< Request timeout in seconds. */
    char *custom_prompt;               /**< Custom extraction prompt (NULL for default). */
} GV_LLMConfig;

/**
 * @brief LLM message structure.
 */
typedef struct {
    char *role;                       /**< Message role: "system", "user", "assistant". */
    char *content;                    /**< Message content. */
} GV_LLMMessage;

/**
 * @brief LLM response structure.
 */
typedef struct {
    char *content;                    /**< Generated text content. */
    int finish_reason;                 /**< Finish reason code. */
    int token_count;                  /**< Number of tokens used. */
} GV_LLMResponse;

/**
 * @brief LLM handle structure (opaque).
 */
typedef struct GV_LLM GV_LLM;

/**
 * @brief Create an LLM instance.
 *
 * @param config LLM configuration; must be non-NULL.
 * @return LLM handle or NULL on failure.
 */
GV_LLM *gv_llm_create(const GV_LLMConfig *config);

/**
 * @brief Destroy an LLM instance.
 *
 * @param llm LLM handle; safe to call with NULL.
 */
void gv_llm_destroy(GV_LLM *llm);

/**
 * @brief Generate a response from the LLM.
 *
 * @param llm LLM handle; must be non-NULL.
 * @param messages Array of messages; must be non-NULL.
 * @param message_count Number of messages.
 * @param response_format JSON format requirement (NULL for text, "json_object" for JSON).
 * @param response Output response structure; must be non-NULL.
 * @return GV_LLM_SUCCESS on success, error code on failure.
 */
int gv_llm_generate_response(GV_LLM *llm, const GV_LLMMessage *messages, size_t message_count,
                             const char *response_format, GV_LLMResponse *response);

/**
 * @brief Get the last error message.
 *
 * @param llm LLM handle; must be non-NULL.
 * @return Error message string, or NULL if no error.
 */
const char *gv_llm_get_last_error(GV_LLM *llm);

/**
 * @brief Get error code description.
 *
 * @param error_code Error code.
 * @return Human-readable error description.
 */
const char *gv_llm_error_string(int error_code);

/**
 * @brief Free LLM response structure.
 *
 * @param response Response to free; safe to call with NULL.
 */
void gv_llm_response_free(GV_LLMResponse *response);

/**
 * @brief Free LLM message structure.
 *
 * @param message Message to free; safe to call with NULL.
 */
void gv_llm_message_free(GV_LLMMessage *message);

/**
 * @brief Free array of LLM messages.
 *
 * @param messages Array of messages; can be NULL.
 * @param count Number of messages.
 */
void gv_llm_messages_free(GV_LLMMessage *messages, size_t count);

#ifdef __cplusplus
}
#endif

#endif

