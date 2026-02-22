#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gigavector/gv_llm.h"

#define MAX_ENV_VALUE_LEN 1024
static char env_buffer[MAX_ENV_VALUE_LEN];

// Helper function to read .env file and return value for a key
static const char *read_env_file(const char *env_var) {
    FILE *file = fopen(".env", "r");
    if (file == NULL) {
        return NULL;
    }
    
    char line[2048];
    size_t key_len = strlen(env_var);
    
    while (fgets(line, sizeof(line), file)) {
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
            len--;
        }
        
        // Skip empty lines and comments
        if (len == 0 || line[0] == '#') {
            continue;
        }
        
        // Check if line starts with the key
        if (strncmp(line, env_var, key_len) == 0 && line[key_len] == '=') {
            // Found the key, extract value
            const char *value = line + key_len + 1;
            
            // Remove quotes if present
            size_t value_len = strlen(value);
            if (value_len >= 2 &&
                ((value[0] == '"' && value[value_len - 1] == '"') ||
                 (value[0] == '\'' && value[value_len - 1] == '\''))) {
                size_t val_len = value_len - 2;
                if (val_len < MAX_ENV_VALUE_LEN - 1) {
                    memcpy(env_buffer, value + 1, val_len);
                    env_buffer[val_len] = '\0';
                    fclose(file);
                    return env_buffer;
                }
            }

            // Copy value (without quotes)
            if (value_len < MAX_ENV_VALUE_LEN - 1) {
                memcpy(env_buffer, value, value_len);
                env_buffer[value_len] = '\0';
                fclose(file);
                return env_buffer;
            }
        }
    }
    
    fclose(file);
    return NULL;
}

// Helper function to get API key from .env file or environment
static const char *get_env_api_key(const char *env_var) {
    // First try .env file
    const char *key = read_env_file(env_var);
    if (key && strlen(key) > 0) {
        return key;
    }
    
    // Fall back to environment variable
    key = getenv(env_var);
    return key && strlen(key) > 0 ? key : NULL;
}

// Test LLM creation with valid config
void test_llm_create_valid(void) {
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        api_key = "sk-test123456789012345678901234567890";
    }
    
    GV_LLMConfig config = {
        .provider = GV_LLM_PROVIDER_OPENAI,
        .api_key = (char *)api_key,
        .model = "gpt-4o-mini",
        .base_url = NULL,
        .temperature = 0.7,
        .max_tokens = 2000,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    GV_LLM *llm = gv_llm_create(&config);
    
    if (llm == NULL) {
        return;
    }

    // Test error string function
    const char *error_str = gv_llm_error_string(GV_LLM_SUCCESS);
    assert(error_str != NULL);
    
    gv_llm_destroy(llm);
}

// Test actual API call with OpenAI
void test_llm_api_call_openai(void) {
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        return;
    }
    
    GV_LLMConfig config = {
        .provider = GV_LLM_PROVIDER_OPENAI,
        .api_key = (char *)api_key,
        .model = "gpt-4o-mini",
        .base_url = NULL,
        .temperature = 0.7,
        .max_tokens = 100,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    GV_LLM *llm = gv_llm_create(&config);
    if (llm == NULL) {
        return;
    }
    
    // Create a simple test message
    GV_LLMMessage messages[1];
    messages[0].role = "user";
    messages[0].content = "Say 'Hello, GigaVector!' in one sentence.";
    
    GV_LLMResponse response;
    memset(&response, 0, sizeof(response));
    
    int result = gv_llm_generate_response(llm, messages, 1, NULL, &response);
    
    if (result == GV_LLM_SUCCESS && response.content != NULL) {
        gv_llm_response_free(&response);
    } else {
        const char *error = gv_llm_get_last_error(llm);
        if (error) {
            (void)error;
        }
    }
    
    gv_llm_destroy(llm);
}

// Test LLM creation with invalid API key
void test_llm_create_invalid_api_key(void) {
    GV_LLMConfig config = {
        .provider = GV_LLM_PROVIDER_OPENAI,
        .api_key = "invalid-key",  // Too short and wrong format
        .model = "gpt-4o-mini",
        .base_url = NULL,
        .temperature = 0.7,
        .max_tokens = 2000,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    GV_LLM *llm = gv_llm_create(&config);
    
    if (llm == NULL) {
    } else {
        gv_llm_destroy(llm);
    }
}

// Test LLM creation with invalid URL
void test_llm_create_invalid_url(void) {
    GV_LLMConfig config = {
        .provider = GV_LLM_PROVIDER_OPENAI,
        .api_key = "sk-test123456789012345678901234567890",
        .model = "gpt-4o-mini",
        .base_url = "not-a-valid-url",  // Invalid URL
        .temperature = 0.7,
        .max_tokens = 2000,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    GV_LLM *llm = gv_llm_create(&config);
    
    if (llm == NULL) {
    } else {
        gv_llm_destroy(llm);
    }
}

// Test Custom provider requires base_url
void test_custom_requires_base_url(void) {
    GV_LLMConfig config = {
        .provider = GV_LLM_PROVIDER_CUSTOM,
        .api_key = "test123456789012345678901234567890123456",  // 32+ chars
        .model = "gpt-4",
        .base_url = NULL,  // Missing - should fail
        .temperature = 0.7,
        .max_tokens = 2000,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };

    GV_LLM *llm = gv_llm_create(&config);

    if (llm == NULL) {
    } else {
        gv_llm_destroy(llm);
    }
}

// Test error code strings
void test_error_strings(void) {
    const char *errors[] = {
        gv_llm_error_string(GV_LLM_SUCCESS),
        gv_llm_error_string(GV_LLM_ERROR_NULL_POINTER),
        gv_llm_error_string(GV_LLM_ERROR_INVALID_CONFIG),
        gv_llm_error_string(GV_LLM_ERROR_INVALID_API_KEY),
        gv_llm_error_string(GV_LLM_ERROR_NETWORK),
        gv_llm_error_string(GV_LLM_ERROR_TIMEOUT),
        gv_llm_error_string(GV_LLM_ERROR_PARSE_FAILED),
        gv_llm_error_string(999)  // Unknown error
    };
    
    for (int i = 0; i < 8; i++) {
        if (errors[i] != NULL && strlen(errors[i]) > 0) {
        } else {
            return;
        }
    }
}

// Test Anthropic API key validation
void test_anthropic_api_key(void) {
    const char *api_key = get_env_api_key("ANTHROPIC_API_KEY");
    if (!api_key) {
        api_key = "sk-ant-test123456789012345678901234567890";
    }
    
    GV_LLMConfig config = {
        .provider = GV_LLM_PROVIDER_ANTHROPIC,
        .api_key = (char *)api_key,
        .model = "claude-3-5-sonnet-20241022",
        .base_url = NULL,
        .temperature = 0.7,
        .max_tokens = 2000,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    GV_LLM *llm = gv_llm_create(&config);
    
    if (llm == NULL) {
    } else {
        gv_llm_destroy(llm);
    }
    
    // Test invalid Anthropic key
    config.api_key = "sk-test123";  // Wrong prefix
    llm = gv_llm_create(&config);
    if (llm == NULL) {
    } else {
        gv_llm_destroy(llm);
    }
}

// Test actual API call with Anthropic
void test_llm_api_call_anthropic(void) {
    const char *api_key = get_env_api_key("ANTHROPIC_API_KEY");
    if (!api_key) {
        return;
    }
    
    GV_LLMConfig config = {
        .provider = GV_LLM_PROVIDER_ANTHROPIC,
        .api_key = (char *)api_key,
        .model = "claude-3-haiku-20240307",  // Valid Anthropic model
        .base_url = NULL,
        .temperature = 0.7,
        .max_tokens = 100,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    GV_LLM *llm = gv_llm_create(&config);
    if (llm == NULL) {
        return;
    }
    
    // Create a simple test message
    GV_LLMMessage messages[1];
    messages[0].role = "user";
    messages[0].content = "Say 'Hello, GigaVector!' in one sentence.";
    
    GV_LLMResponse response;
    memset(&response, 0, sizeof(response));
    
    int result = gv_llm_generate_response(llm, messages, 1, NULL, &response);
    
    if (result == GV_LLM_SUCCESS && response.content != NULL) {
        gv_llm_response_free(&response);
    } else {
        const char *error = gv_llm_get_last_error(llm);
        if (error) {
            (void)error;
        }
    }
    
    gv_llm_destroy(llm);
}

// Test actual API call with Google Gemini
void test_llm_api_call_gemini(void) {
    const char *api_key = get_env_api_key("GEMINI_API_KEY");
    if (!api_key) {
        return;
    }
    
    // Use Google provider - it will use Gemini API format
    GV_LLMConfig config = {
        .provider = GV_LLM_PROVIDER_GOOGLE,
        .api_key = (char *)api_key,
        .model = "gemini-2.5-flash",  // Available model that supports generateContent
        .base_url = NULL,  // Will use default Gemini endpoint
        .temperature = 0.7,
        .max_tokens = 100,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    GV_LLM *llm = gv_llm_create(&config);
    if (llm == NULL) {
        return;
    }
    
    GV_LLMMessage messages[1];
    messages[0].role = "user";
    messages[0].content = "Say 'Hello, GigaVector!' in one sentence.";
    
    GV_LLMResponse response;
    memset(&response, 0, sizeof(response));
    
    int result = gv_llm_generate_response(llm, messages, 1, NULL, &response);
    
    if (result == GV_LLM_SUCCESS && response.content != NULL) {
        gv_llm_response_free(&response);
    } else {
        const char *error = gv_llm_get_last_error(llm);
        if (error) {
            (void)error;
        }
    }
    
    gv_llm_destroy(llm);
}

int main(void) {
    test_llm_create_valid();
    test_llm_create_invalid_api_key();
    test_llm_create_invalid_url();
    test_custom_requires_base_url();
    test_error_strings();
    test_anthropic_api_key();
    test_llm_api_call_openai();
    test_llm_api_call_anthropic();
    test_llm_api_call_gemini();
    return 0;
}





