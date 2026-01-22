#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdarg.h>
#include <ctype.h>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

#include "gigavector/gv_llm.h"
#include "gigavector/gv_json.h"

#define MAX_RESPONSE_SIZE (1024 * 1024)  // 1MB max response
#define DEFAULT_TIMEOUT 30
#define MAX_ERROR_MESSAGE 512

// Define struct before helper functions that use it
struct GV_LLM {
    GV_LLMConfig config;
    void *curl_handle;  // CURL handle for HTTP requests
    char *last_error;   // Last error message
};

// Secure memory clearing
static void secure_memclear(void *ptr, size_t len) {
    if (ptr == NULL) return;
    volatile unsigned char *p = (volatile unsigned char *)ptr;
    while (len--) {
        *p++ = 0;
    }
}

// Set error message
static void set_error(struct GV_LLM *llm, const char *format, ...) {
    if (llm == NULL) return;
    if (llm->last_error) {
        free(llm->last_error);
    }
    llm->last_error = (char *)malloc(MAX_ERROR_MESSAGE);
    if (llm->last_error == NULL) return;
    
    va_list args;
    va_start(args, format);
    vsnprintf(llm->last_error, MAX_ERROR_MESSAGE, format, args);
    va_end(args);
}

// Validate API key format
static int validate_api_key(const char *api_key, GV_LLMProvider provider) {
    if (api_key == NULL || strlen(api_key) < 10) {
        return 0;
    }
    
    switch (provider) {
        case GV_LLM_PROVIDER_OPENAI:
            // OpenAI keys start with "sk-"
            return (strncmp(api_key, "sk-", 3) == 0);
        case GV_LLM_PROVIDER_ANTHROPIC:
            // Anthropic keys start with "sk-ant-"
            return (strncmp(api_key, "sk-ant-", 7) == 0);
        case GV_LLM_PROVIDER_GOOGLE:
            // Google API keys are typically 39+ characters and may start with AIza
            return (strlen(api_key) >= 20);
        case GV_LLM_PROVIDER_CUSTOM:
        default:
            // For custom providers, just check minimum length
            return (strlen(api_key) >= 10);
    }
}

// Validate URL format
static int validate_url(const char *url) {
    if (url == NULL) return 0;
    // Basic URL validation: must start with http:// or https://
    return (strncmp(url, "http://", 7) == 0 || strncmp(url, "https://", 8) == 0);
}

#ifdef HAVE_CURL

struct ResponseBuffer {
    char *data;
    size_t size;
    size_t capacity;
};

static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct ResponseBuffer *buf = (struct ResponseBuffer *)userp;
    
    // Enforce MAX_RESPONSE_SIZE limit
    if (buf->size + realsize > MAX_RESPONSE_SIZE) {
        return 0;  // Stop reading
    }
    
    if (buf->size + realsize >= buf->capacity) {
        size_t new_capacity = buf->capacity * 2;
        if (new_capacity < buf->size + realsize + 1) {
            new_capacity = buf->size + realsize + 1;
        }
        // Don't exceed MAX_RESPONSE_SIZE
        if (new_capacity > MAX_RESPONSE_SIZE) {
            new_capacity = MAX_RESPONSE_SIZE;
        }
        char *new_data = (char *)realloc(buf->data, new_capacity);
        if (new_data == NULL) {
            return 0;
        }
        buf->data = new_data;
        buf->capacity = new_capacity;
    }
    
    memcpy(buf->data + buf->size, contents, realsize);
    buf->size += realsize;
    buf->data[buf->size] = '\0';
    
    return realsize;
}

static size_t json_escape_size(const char *str) {
    size_t size = 0;
    for (const char *p = str; *p; p++) {
        switch (*p) {
            case '"': case '\\': case '\n': case '\r': case '\t': case '\b': case '\f':
                size += 2;  // Escaped: \x
                break;
            default:
                if ((unsigned char)*p < 0x20) {
                    size += 6;  // Unicode escape: \uXXXX
                } else {
                    size += 1;
                }
                break;
        }
    }
    return size;
}

static void json_escape_append(char *dest, const char *src) {
    for (const char *p = src; *p; p++) {
        switch (*p) {
            case '"':
                *dest++ = '\\'; *dest++ = '"';
                break;
            case '\\':
                *dest++ = '\\'; *dest++ = '\\';
                break;
            case '\n':
                *dest++ = '\\'; *dest++ = 'n';
                break;
            case '\r':
                *dest++ = '\\'; *dest++ = 'r';
                break;
            case '\t':
                *dest++ = '\\'; *dest++ = 't';
                break;
            case '\b':
                *dest++ = '\\'; *dest++ = 'b';
                break;
            case '\f':
                *dest++ = '\\'; *dest++ = 'f';
                break;
            default:
                if ((unsigned char)*p < 0x20) {
                    // Unicode escape for control characters
                    snprintf(dest, 7, "\\u%04x", (unsigned char)*p);
                    dest += 6;
                } else {
                    *dest++ = *p;
                }
                break;
        }
    }
    *dest = '\0';
}

static char *build_openai_request(const GV_LLMConfig *config, const GV_LLMMessage *messages, 
                                   size_t message_count, const char *response_format) {
    // Calculate required size
    size_t total_size = 256;  // Base JSON structure
    const char *model = config->model ? config->model : "gpt-4o-mini";
    total_size += strlen(model);
    
    for (size_t i = 0; i < message_count; i++) {
        total_size += 32;  // Role and structure
        const char *role = messages[i].role ? messages[i].role : "user";
        total_size += strlen(role);
        const char *content = messages[i].content ? messages[i].content : "";
        total_size += json_escape_size(content);  // Escaped content size
    }
    
    total_size += 128;  // Temperature, max_tokens, etc.
    if (response_format && strcmp(response_format, "json_object") == 0) {
        total_size += 50;
    }
    
    char *json = (char *)malloc(total_size);
    if (json == NULL) return NULL;
    
    size_t pos = 0;
    int written = snprintf(json + pos, total_size - pos, "{\"model\":\"%s\",\"messages\":[", model);
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    pos += written;
    
    for (size_t i = 0; i < message_count; i++) {
        if (i > 0) {
            if (pos >= total_size - 1) {
                free(json);
                return NULL;
            }
            json[pos++] = ',';
            json[pos] = '\0';
        }
        const char *role = messages[i].role ? messages[i].role : "user";
        written = snprintf(json + pos, total_size - pos, "{\"role\":\"%s\",\"content\":\"", role);
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
        
        const char *content = messages[i].content ? messages[i].content : "";
        size_t content_escaped_size = json_escape_size(content);
        char *escaped = (char *)malloc(content_escaped_size + 1);
        if (escaped == NULL) {
            free(json);
            return NULL;
        }
        json_escape_append(escaped, content);
        written = snprintf(json + pos, total_size - pos, "%s", escaped);
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(escaped);
            free(json);
            return NULL;
        }
        pos += written;
        free(escaped);
        
        written = snprintf(json + pos, total_size - pos, "\"}");
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
    }
    
    char temp_str[32];
    snprintf(temp_str, sizeof(temp_str), "%.2f", config->temperature > 0 ? config->temperature : 0.7);
    written = snprintf(json + pos, total_size - pos, "],\"temperature\":%s", temp_str);
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    pos += written;
    
    if (response_format && strcmp(response_format, "json_object") == 0) {
        written = snprintf(json + pos, total_size - pos, ",\"response_format\":{\"type\":\"json_object\"}");
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
    }
    
    if (config->max_tokens > 0) {
        written = snprintf(json + pos, total_size - pos, ",\"max_tokens\":%d", config->max_tokens);
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
    }
    
    written = snprintf(json + pos, total_size - pos, "}");
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    return json;
}

static char *build_gemini_request(const GV_LLMConfig *config, const GV_LLMMessage *messages,
                                  size_t message_count) {
    // Gemini API format: {"contents": [{"parts": [{"text": "..."}]}]}
    // Calculate required size
    size_t total_size = 256;  // Base JSON structure
    
    for (size_t i = 0; i < message_count; i++) {
        total_size += 64;  // Structure overhead
        const char *content = messages[i].content ? messages[i].content : "";
        total_size += json_escape_size(content);
    }
    
    total_size += 128;  // Generation config
    
    char *json = (char *)malloc(total_size);
    if (json == NULL) return NULL;
    
    size_t pos = 0;
    int written = snprintf(json + pos, total_size - pos, "{\"contents\":[");
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    pos += written;
    
    for (size_t i = 0; i < message_count; i++) {
        if (i > 0) {
            if (pos >= total_size - 1) {
                free(json);
                return NULL;
            }
            json[pos++] = ',';
            json[pos] = '\0';
        }
        
        const char *role = messages[i].role ? messages[i].role : "user";
        // Gemini uses "user" or "model" roles
        const char *gemini_role = (strcmp(role, "assistant") == 0) ? "model" : "user";
        
        written = snprintf(json + pos, total_size - pos, "{\"role\":\"%s\",\"parts\":[{\"text\":\"", gemini_role);
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
        
        const char *content = messages[i].content ? messages[i].content : "";
        size_t content_escaped_size = json_escape_size(content);
        char *escaped = (char *)malloc(content_escaped_size + 1);
        if (escaped == NULL) {
            free(json);
            return NULL;
        }
        json_escape_append(escaped, content);
        written = snprintf(json + pos, total_size - pos, "%s", escaped);
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(escaped);
            free(json);
            return NULL;
        }
        pos += written;
        free(escaped);
        
        written = snprintf(json + pos, total_size - pos, "\"}]}");
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
    }
    
    written = snprintf(json + pos, total_size - pos, "]");
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    pos += written;
    
    // Add generation config
    char temp_str[32];
    snprintf(temp_str, sizeof(temp_str), "%.2f", config->temperature > 0 ? config->temperature : 0.7);
    written = snprintf(json + pos, total_size - pos, ",\"generationConfig\":{\"temperature\":%s", temp_str);
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    pos += written;
    
    if (config->max_tokens > 0) {
        written = snprintf(json + pos, total_size - pos, ",\"maxOutputTokens\":%d", config->max_tokens);
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
    }
    
    written = snprintf(json + pos, total_size - pos, "}}");
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    return json;
}

static char *build_anthropic_request(const GV_LLMConfig *config, const GV_LLMMessage *messages,
                                     size_t message_count) {
    // Calculate required size
    size_t total_size = 256;  // Base JSON structure
    const char *model = config->model ? config->model : "claude-3-haiku-20240307";
    total_size += strlen(model);
    
    for (size_t i = 0; i < message_count; i++) {
        total_size += 32;  // Role and structure
        const char *role = messages[i].role ? messages[i].role : "user";
        total_size += strlen(role);
        const char *content = messages[i].content ? messages[i].content : "";
        total_size += json_escape_size(content);  // Escaped content size
    }
    
    total_size += 64;  // max_tokens
    
    char *json = (char *)malloc(total_size);
    if (json == NULL) return NULL;
    
    size_t pos = 0;
    int written = snprintf(json + pos, total_size - pos, "{\"model\":\"%s\",\"max_tokens\":%d,\"messages\":[",
                    model, config->max_tokens > 0 ? config->max_tokens : 4096);
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    pos += written;
    
    for (size_t i = 0; i < message_count; i++) {
        if (i > 0) {
            if (pos >= total_size - 1) {
                free(json);
                return NULL;
            }
            json[pos++] = ',';
            json[pos] = '\0';
        }
        const char *role = messages[i].role ? messages[i].role : "user";
        written = snprintf(json + pos, total_size - pos, "{\"role\":\"%s\",\"content\":\"", role);
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
        
        const char *content = messages[i].content ? messages[i].content : "";
        size_t content_escaped_size = json_escape_size(content);
        char *escaped = (char *)malloc(content_escaped_size + 1);
        if (escaped == NULL) {
            free(json);
            return NULL;
        }
        json_escape_append(escaped, content);
        written = snprintf(json + pos, total_size - pos, "%s", escaped);
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(escaped);
            free(json);
            return NULL;
        }
        pos += written;
        free(escaped);
        
        written = snprintf(json + pos, total_size - pos, "\"}");
        if (written < 0 || (size_t)written >= total_size - pos) {
            free(json);
            return NULL;
        }
        pos += written;
    }
    
    written = snprintf(json + pos, total_size - pos, "]}");
    if (written < 0 || (size_t)written >= total_size - pos) {
        free(json);
        return NULL;
    }
    return json;
}

static int parse_openai_response(const char *response_json, GV_LLMResponse *out) {
    // Parse JSON using proper JSON parser
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(response_json, &err);
    if (root == NULL) {
        return -1;
    }

    // OpenAI format: {"choices": [{"message": {"content": "..."}}]}
    // Navigate to choices[0].message.content
    const char *content = gv_json_get_string_path(root, "choices.0.message.content");
    if (content == NULL) {
        gv_json_free(root);
        return -1;
    }

    out->content = strdup(content);
    if (out->content == NULL) {
        gv_json_free(root);
        return -1;
    }

    // Extract optional fields
    out->finish_reason = 0;
    GV_JsonValue *finish = gv_json_get_path(root, "choices.0.finish_reason");
    if (finish && gv_json_is_string(finish)) {
        const char *reason = gv_json_get_string(finish);
        if (reason && strcmp(reason, "stop") == 0) {
            out->finish_reason = 1;
        }
    }

    out->token_count = 0;
    GV_JsonValue *usage = gv_json_object_get(root, "usage");
    if (usage && gv_json_is_object(usage)) {
        GV_JsonValue *total = gv_json_object_get(usage, "total_tokens");
        if (total && gv_json_is_number(total)) {
            double tokens;
            if (gv_json_get_number(total, &tokens) == GV_JSON_OK) {
                out->token_count = (int)tokens;
            }
        }
    }

    gv_json_free(root);
    return 0;
}

static int parse_gemini_response(const char *response_json, GV_LLMResponse *out) {
    // Parse JSON using proper JSON parser
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(response_json, &err);
    if (root == NULL) {
        return -1;
    }

    // Gemini format: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
    // Navigate to candidates[0].content.parts[0].text
    const char *content = gv_json_get_string_path(root, "candidates.0.content.parts.0.text");
    if (content == NULL) {
        gv_json_free(root);
        return -1;
    }

    out->content = strdup(content);
    if (out->content == NULL) {
        gv_json_free(root);
        return -1;
    }

    // Extract optional fields
    out->finish_reason = 0;
    GV_JsonValue *finish = gv_json_get_path(root, "candidates.0.finishReason");
    if (finish && gv_json_is_string(finish)) {
        const char *reason = gv_json_get_string(finish);
        if (reason && strcmp(reason, "STOP") == 0) {
            out->finish_reason = 1;
        }
    }

    out->token_count = 0;
    GV_JsonValue *usage = gv_json_object_get(root, "usageMetadata");
    if (usage && gv_json_is_object(usage)) {
        GV_JsonValue *total = gv_json_object_get(usage, "totalTokenCount");
        if (total && gv_json_is_number(total)) {
            double tokens;
            if (gv_json_get_number(total, &tokens) == GV_JSON_OK) {
                out->token_count = (int)tokens;
            }
        }
    }

    gv_json_free(root);
    return 0;
}

static int parse_anthropic_response(const char *response_json, GV_LLMResponse *out) {
    // Parse JSON using proper JSON parser
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(response_json, &err);
    if (root == NULL) {
        return -1;
    }

    // Anthropic format: {"content": [{"type": "text", "text": "..."}]}
    // Navigate to content[0].text
    GV_JsonValue *content_array = gv_json_object_get(root, "content");
    if (content_array == NULL || !gv_json_is_array(content_array)) {
        gv_json_free(root);
        return -1;
    }

    // Find first text block in the content array
    const char *text_content = NULL;
    size_t array_len = gv_json_array_length(content_array);
    for (size_t i = 0; i < array_len; i++) {
        GV_JsonValue *block = gv_json_array_get(content_array, i);
        if (block && gv_json_is_object(block)) {
            GV_JsonValue *type_val = gv_json_object_get(block, "type");
            if (type_val && gv_json_is_string(type_val)) {
                const char *type = gv_json_get_string(type_val);
                if (type && strcmp(type, "text") == 0) {
                    text_content = gv_json_get_string(gv_json_object_get(block, "text"));
                    break;
                }
            }
        }
    }

    if (text_content == NULL) {
        gv_json_free(root);
        return -1;
    }

    out->content = strdup(text_content);
    if (out->content == NULL) {
        gv_json_free(root);
        return -1;
    }

    // Extract optional fields
    out->finish_reason = 0;
    GV_JsonValue *stop_reason = gv_json_object_get(root, "stop_reason");
    if (stop_reason && gv_json_is_string(stop_reason)) {
        const char *reason = gv_json_get_string(stop_reason);
        if (reason && strcmp(reason, "end_turn") == 0) {
            out->finish_reason = 1;
        }
    }

    out->token_count = 0;
    GV_JsonValue *usage = gv_json_object_get(root, "usage");
    if (usage && gv_json_is_object(usage)) {
        double input_tokens = 0, output_tokens = 0;
        GV_JsonValue *input = gv_json_object_get(usage, "input_tokens");
        GV_JsonValue *output = gv_json_object_get(usage, "output_tokens");
        if (input && gv_json_is_number(input)) {
            gv_json_get_number(input, &input_tokens);
        }
        if (output && gv_json_is_number(output)) {
            gv_json_get_number(output, &output_tokens);
        }
        out->token_count = (int)(input_tokens + output_tokens);
    }

    gv_json_free(root);
    return 0;
}

#endif  // HAVE_CURL

GV_LLM *gv_llm_create(const GV_LLMConfig *config) {
    if (config == NULL) {
        return NULL;
    }
    
    if (config->api_key == NULL) {
        return NULL;
    }
    
    // Validate API key format
    if (!validate_api_key(config->api_key, config->provider)) {
        return NULL;
    }
    
    // Validate base_url if provided
    if (config->base_url != NULL && !validate_url(config->base_url)) {
        return NULL;
    }

    // Custom provider requires base_url
    if (config->provider == GV_LLM_PROVIDER_CUSTOM && config->base_url == NULL) {
        return NULL;
    }

#ifdef HAVE_CURL
    GV_LLM *llm = (GV_LLM *)malloc(sizeof(GV_LLM));
    if (llm == NULL) {
        return NULL;
    }
    
    memset(llm, 0, sizeof(GV_LLM));
    llm->config = *config;
    llm->last_error = NULL;
    
    // Copy strings
    if (config->api_key) {
        llm->config.api_key = strdup(config->api_key);
        if (llm->config.api_key == NULL) {
            free(llm);
            return NULL;
        }
    }
    if (config->model) {
        llm->config.model = strdup(config->model);
        if (llm->config.model == NULL) {
            free(llm->config.api_key);
            free(llm);
            return NULL;
        }
    }
    if (config->base_url) {
        llm->config.base_url = strdup(config->base_url);
        if (llm->config.base_url == NULL) {
            free(llm->config.api_key);
            free(llm->config.model);
            free(llm);
            return NULL;
        }
    }
    if (config->custom_prompt) {
        llm->config.custom_prompt = strdup(config->custom_prompt);
        if (llm->config.custom_prompt == NULL) {
            free(llm->config.api_key);
            free(llm->config.model);
            free(llm->config.base_url);
            free(llm);
            return NULL;
        }
    }
    
    llm->curl_handle = curl_easy_init();
    if (llm->curl_handle == NULL) {
        set_error(llm, "Failed to initialize CURL");
        free(llm->config.api_key);
        free(llm->config.model);
        free(llm->config.base_url);
        free(llm->config.custom_prompt);
        free(llm->last_error);
        free(llm);
        return NULL;
    }
    
    return llm;
#else
    // No CURL support - return NULL
    return NULL;
#endif
}

void gv_llm_destroy(GV_LLM *llm) {
    if (llm == NULL) {
        return;
    }
    
#ifdef HAVE_CURL
    if (llm->curl_handle) {
        curl_easy_cleanup((CURL *)llm->curl_handle);
    }
#endif
    
    // Securely clear API key from memory
    if (llm->config.api_key) {
        secure_memclear(llm->config.api_key, strlen(llm->config.api_key));
        free(llm->config.api_key);
    }
    
    free(llm->config.model);
    free(llm->config.base_url);
    free(llm->config.custom_prompt);
    free(llm->last_error);
    free(llm);
}

int gv_llm_generate_response(GV_LLM *llm, const GV_LLMMessage *messages, size_t message_count,
                              const char *response_format, GV_LLMResponse *response) {
    (void)response_format;  // Used for OpenAI/Azure, not yet implemented for other providers
    if (llm == NULL) {
        return GV_LLM_ERROR_NULL_POINTER;
    }
    if (messages == NULL) {
        set_error(llm, "Messages array is NULL");
        return GV_LLM_ERROR_NULL_POINTER;
    }
    if (response == NULL) {
        set_error(llm, "Response pointer is NULL");
        return GV_LLM_ERROR_NULL_POINTER;
    }
    if (message_count == 0) {
        set_error(llm, "Message count is zero");
        return GV_LLM_ERROR_INVALID_CONFIG;
    }
    
    memset(response, 0, sizeof(GV_LLMResponse));
    
#ifdef HAVE_CURL
    CURL *curl = (CURL *)llm->curl_handle;
    if (curl == NULL) {
        set_error(llm, "CURL handle is NULL");
        return GV_LLM_ERROR_CURL_INIT;
    }
    
    char *request_json = NULL;
    const char *url = NULL;
    const char *auth_header = NULL;
    
    // Build request based on provider
    switch (llm->config.provider) {
        case GV_LLM_PROVIDER_OPENAI:
            request_json = build_openai_request(&llm->config, messages, message_count, response_format);
            url = llm->config.base_url ? llm->config.base_url : "https://api.openai.com/v1/chat/completions";
            auth_header = "Authorization: Bearer ";
            break;
            
        case GV_LLM_PROVIDER_ANTHROPIC:
            request_json = build_anthropic_request(&llm->config, messages, message_count);
            url = llm->config.base_url ? llm->config.base_url : "https://api.anthropic.com/v1/messages";
            auth_header = "x-api-key: ";
            break;
            
        case GV_LLM_PROVIDER_GOOGLE: {
            // Google Gemini API
            static char url_buf[512];  // Static to persist beyond case block
            const char *model = llm->config.model ? llm->config.model : "gemini-2.5-flash";
            if (llm->config.base_url) {
                url = llm->config.base_url;
            } else {
                snprintf(url_buf, sizeof(url_buf),
                        "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent",
                        model);
                url = url_buf;
            }
            request_json = build_gemini_request(&llm->config, messages, message_count);
            auth_header = "x-goog-api-key: ";
            break;
        }
            
        case GV_LLM_PROVIDER_CUSTOM:
            // Custom providers default to OpenAI-compatible endpoint
            // For custom, try OpenAI format first (most common)
            request_json = build_openai_request(&llm->config, messages, message_count, response_format);
            url = llm->config.base_url ? llm->config.base_url : "http://localhost:8000/v1/chat/completions";
            auth_header = "Authorization: Bearer ";
            break;
            
        default:
            set_error(llm, "Unknown provider type");
            return GV_LLM_ERROR_INVALID_CONFIG;
    }
    
    if (request_json == NULL) {
        set_error(llm, "Failed to build request JSON");
        return GV_LLM_ERROR_MEMORY_ALLOCATION;
    }
    
    struct ResponseBuffer buf;
    buf.data = (char *)malloc(4096);
    buf.size = 0;
    buf.capacity = 4096;
    if (buf.data == NULL) {
        free(request_json);
        set_error(llm, "Failed to allocate response buffer");
        return GV_LLM_ERROR_MEMORY_ALLOCATION;
    }
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    // Add provider-specific headers
    if (llm->config.provider == GV_LLM_PROVIDER_ANTHROPIC) {
        headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
    }
    
    // Add authentication header
    if (llm->config.api_key == NULL) {
        free(buf.data);
        free(request_json);
        curl_slist_free_all(headers);
        set_error(llm, "API key is NULL");
        return GV_LLM_ERROR_INVALID_API_KEY;
    }
    
    char auth_buf[1024];  // Increased size for longer API keys
    if (llm->config.provider == GV_LLM_PROVIDER_GOOGLE) {
        // Google uses x-goog-api-key header (no Bearer prefix)
        snprintf(auth_buf, sizeof(auth_buf), "%s%s", auth_header, llm->config.api_key);
    } else {
        snprintf(auth_buf, sizeof(auth_buf), "%s%s", auth_header, llm->config.api_key);
    }
    headers = curl_slist_append(headers, auth_buf);
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, llm->config.timeout_seconds > 0 ? 
                     llm->config.timeout_seconds : DEFAULT_TIMEOUT);
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(headers);
    free(request_json);
    
    if (res != CURLE_OK) {
        free(buf.data);
        if (res == CURLE_OPERATION_TIMEDOUT) {
            set_error(llm, "Request timeout: %s", curl_easy_strerror(res));
            return GV_LLM_ERROR_TIMEOUT;
        } else {
            set_error(llm, "Network error: %s", curl_easy_strerror(res));
            return GV_LLM_ERROR_NETWORK;
        }
    }
    
    // Check if response was truncated
    if (buf.size >= MAX_RESPONSE_SIZE) {
        free(buf.data);
        set_error(llm, "Response exceeds maximum size (%d bytes)", MAX_RESPONSE_SIZE);
        return GV_LLM_ERROR_RESPONSE_TOO_LARGE;
    }
    
    // Check for error responses first
    if (strstr(buf.data, "\"error\"") != NULL) {
        // Try to extract error message
        const char *error_msg = strstr(buf.data, "\"message\":\"");
        if (error_msg) {
            error_msg += 11;  // Skip past "\"message\":\""
            const char *error_end = strstr(error_msg, "\"");
            if (error_end) {
                size_t msg_len = error_end - error_msg;
                char *error_buf = (char *)malloc(msg_len + 1);
                if (error_buf) {
                    memcpy(error_buf, error_msg, msg_len);
                    error_buf[msg_len] = '\0';
                    set_error(llm, "API error: %s", error_buf);
                    free(error_buf);
                }
            }
        } else {
            set_error(llm, "API returned an error response");
        }
        free(buf.data);
        return GV_LLM_ERROR_INVALID_RESPONSE;
    }
    
    // Parse response based on provider
    int parse_result = GV_LLM_ERROR_PARSE_FAILED;
    switch (llm->config.provider) {
        case GV_LLM_PROVIDER_OPENAI:
        case GV_LLM_PROVIDER_CUSTOM:
            // OpenAI and custom providers use OpenAI-compatible format
            parse_result = parse_openai_response(buf.data, response);
            break;
        case GV_LLM_PROVIDER_ANTHROPIC:
            parse_result = parse_anthropic_response(buf.data, response);
            break;
        case GV_LLM_PROVIDER_GOOGLE:
            parse_result = parse_gemini_response(buf.data, response);
            break;
        default:
            set_error(llm, "Unsupported provider for parsing");
            break;
    }
    
    free(buf.data);
    
    if (parse_result != 0) {
        set_error(llm, "Failed to parse LLM response");
        return GV_LLM_ERROR_PARSE_FAILED;
    }
    
    if (response->content == NULL) {
        set_error(llm, "Response content is NULL");
        return GV_LLM_ERROR_INVALID_RESPONSE;
    }
    
    return GV_LLM_SUCCESS;
#else
    // No CURL support
    if (llm) {
        set_error(llm, "CURL support not compiled in");
    }
    return GV_LLM_ERROR_CURL_INIT;
#endif
}

const char *gv_llm_get_last_error(GV_LLM *llm) {
    if (llm == NULL) {
        return NULL;
    }
    return llm->last_error;
}

const char *gv_llm_error_string(int error_code) {
    switch (error_code) {
        case GV_LLM_SUCCESS:
            return "Success";
        case GV_LLM_ERROR_NULL_POINTER:
            return "Null pointer error";
        case GV_LLM_ERROR_INVALID_CONFIG:
            return "Invalid configuration";
        case GV_LLM_ERROR_INVALID_API_KEY:
            return "Invalid API key";
        case GV_LLM_ERROR_INVALID_URL:
            return "Invalid URL";
        case GV_LLM_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case GV_LLM_ERROR_CURL_INIT:
            return "CURL initialization failed";
        case GV_LLM_ERROR_NETWORK:
            return "Network error";
        case GV_LLM_ERROR_TIMEOUT:
            return "Request timeout";
        case GV_LLM_ERROR_RESPONSE_TOO_LARGE:
            return "Response too large";
        case GV_LLM_ERROR_PARSE_FAILED:
            return "Failed to parse response";
        case GV_LLM_ERROR_INVALID_RESPONSE:
            return "Invalid response";
        case GV_LLM_ERROR_CUSTOM_URL_REQUIRED:
            return "Custom provider requires base_url";
        default:
            return "Unknown error";
    }
}

void gv_llm_response_free(GV_LLMResponse *response) {
    if (response == NULL) {
        return;
    }
    free(response->content);
    memset(response, 0, sizeof(GV_LLMResponse));
}

void gv_llm_message_free(GV_LLMMessage *message) {
    if (message == NULL) {
        return;
    }
    free(message->role);
    free(message->content);
    memset(message, 0, sizeof(GV_LLMMessage));
}

void gv_llm_messages_free(GV_LLMMessage *messages, size_t count) {
    if (messages == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        gv_llm_message_free(&messages[i]);
    }
    free(messages);
}

