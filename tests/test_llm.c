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
    printf("Testing LLM creation with valid config...\n");
    
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        printf("  [SKIP] Skipping: OPENAI_API_KEY not set (using test key for validation only)\n");
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
        printf("  [WARN]  LLM creation returned NULL (may be expected if libcurl not available)\n");
        return;
    }
    
    printf("  [OK] LLM created successfully\n");
    
    // Test error string function
    const char *error_str = gv_llm_error_string(GV_LLM_SUCCESS);
    assert(error_str != NULL);
    printf("  [OK] Error string function works: %s\n", error_str);
    
    gv_llm_destroy(llm);
    printf("  [OK] LLM destroyed successfully\n");
}

// Test actual API call with OpenAI
void test_llm_api_call_openai(void) {
    printf("Testing OpenAI API call...\n");
    
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        printf("  [SKIP] Skipping: OPENAI_API_KEY environment variable not set\n");
        printf("  Set OPENAI_API_KEY to run this test\n");
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
        printf("  [FAIL] Failed to create LLM instance\n");
        return;
    }
    
    // Create a simple test message
    GV_LLMMessage messages[1];
    messages[0].role = "user";
    messages[0].content = "Say 'Hello, GigaVector!' in one sentence.";
    
    GV_LLMResponse response;
    memset(&response, 0, sizeof(response));
    
    printf("  [INFO] Making API call to OpenAI...\n");
    int result = gv_llm_generate_response(llm, messages, 1, NULL, &response);
    
    if (result == GV_LLM_SUCCESS && response.content != NULL) {
        printf("  [OK] API call succeeded!\n");
        printf("  [RESPONSE] Response: %s\n", response.content);
        printf("  [TOKENS] Tokens: %d\n", response.token_count);
        gv_llm_response_free(&response);
    } else {
        printf("  [FAIL] API call failed: %s\n", gv_llm_error_string(result));
        const char *error = gv_llm_get_last_error(llm);
        if (error) {
            printf("  Error details: %s\n", error);
        }
    }
    
    gv_llm_destroy(llm);
}

// Test LLM creation with invalid API key
void test_llm_create_invalid_api_key(void) {
    printf("Testing LLM creation with invalid API key...\n");
    
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
        printf("  [OK] Correctly rejected invalid API key\n");
    } else {
        printf("  [FAIL] Should have rejected invalid API key\n");
        gv_llm_destroy(llm);
    }
}

// Test LLM creation with invalid URL
void test_llm_create_invalid_url(void) {
    printf("Testing LLM creation with invalid URL...\n");
    
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
        printf("  [OK] Correctly rejected invalid URL\n");
    } else {
        printf("  [FAIL] Should have rejected invalid URL\n");
        gv_llm_destroy(llm);
    }
}

// Test Custom provider requires base_url
void test_custom_requires_base_url(void) {
    printf("Testing Custom provider requires base_url...\n");

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
        printf("  [OK] Correctly rejected Custom provider config without base_url\n");
    } else {
        printf("  [FAIL] Should have rejected Custom provider config without base_url\n");
        gv_llm_destroy(llm);
    }
}

// Test error code strings
void test_error_strings(void) {
    printf("Testing error code strings...\n");
    
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
            printf("  [OK] Error %d: %s\n", i, errors[i]);
        } else {
            printf("  [FAIL] Error %d returned NULL or empty\n", i);
        }
    }
}

// Test Anthropic API key validation
void test_anthropic_api_key(void) {
    printf("Testing Anthropic API key validation...\n");
    
    const char *api_key = get_env_api_key("ANTHROPIC_API_KEY");
    if (!api_key) {
        printf("  [SKIP] Skipping: ANTHROPIC_API_KEY not set (using test key for validation only)\n");
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
        printf("  [WARN] LLM creation returned NULL (may be expected if libcurl not available)\n");
    } else {
        printf("  [OK] Anthropic API key format accepted\n");
        gv_llm_destroy(llm);
    }
    
    // Test invalid Anthropic key
    config.api_key = "sk-test123";  // Wrong prefix
    llm = gv_llm_create(&config);
    if (llm == NULL) {
        printf("  [OK] Correctly rejected invalid Anthropic API key format\n");
    } else {
        printf("  [FAIL] Should have rejected invalid Anthropic API key\n");
        gv_llm_destroy(llm);
    }
}

// Test actual API call with Anthropic
void test_llm_api_call_anthropic(void) {
    printf("Testing Anthropic API call...\n");
    
    const char *api_key = get_env_api_key("ANTHROPIC_API_KEY");
    if (!api_key) {
        printf("  [SKIP] Skipping: ANTHROPIC_API_KEY environment variable not set\n");
        printf("  Set ANTHROPIC_API_KEY to run this test\n");
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
        printf("  [FAIL] Failed to create LLM instance\n");
        return;
    }
    
    // Create a simple test message
    GV_LLMMessage messages[1];
    messages[0].role = "user";
    messages[0].content = "Say 'Hello, GigaVector!' in one sentence.";
    
    GV_LLMResponse response;
    memset(&response, 0, sizeof(response));
    
    printf("  [INFO] Making API call to Anthropic...\n");
    int result = gv_llm_generate_response(llm, messages, 1, NULL, &response);
    
    if (result == GV_LLM_SUCCESS && response.content != NULL) {
        printf("  [OK] API call succeeded!\n");
        printf("  [RESPONSE] Response: %s\n", response.content);
        printf("  [TOKENS] Tokens: %d\n", response.token_count);
        gv_llm_response_free(&response);
    } else {
        printf("  [FAIL] API call failed: %s\n", gv_llm_error_string(result));
        const char *error = gv_llm_get_last_error(llm);
        if (error) {
            printf("  Error details: %s\n", error);
        }
    }
    
    gv_llm_destroy(llm);
}

// Test actual API call with Google Gemini
void test_llm_api_call_gemini(void) {
    printf("Testing Google Gemini API call...\n");
    
    const char *api_key = get_env_api_key("GEMINI_API_KEY");
    if (!api_key) {
        printf("  [SKIP] Skipping: GEMINI_API_KEY environment variable not set\n");
        printf("  Set GEMINI_API_KEY to run this test\n");
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
        printf("  [FAIL] Failed to create LLM instance\n");
        return;
    }
    
    GV_LLMMessage messages[1];
    messages[0].role = "user";
    messages[0].content = "Say 'Hello, GigaVector!' in one sentence.";
    
    GV_LLMResponse response;
    memset(&response, 0, sizeof(response));
    
    printf("  [INFO] Making API call to Google Gemini...\n");
    
    int result = gv_llm_generate_response(llm, messages, 1, NULL, &response);
    
    if (result == GV_LLM_SUCCESS && response.content != NULL) {
        printf("  [OK] API call succeeded!\n");
        printf("  [RESPONSE] Response: %s\n", response.content);
        printf("  [TOKENS] Tokens: %d\n", response.token_count);
        gv_llm_response_free(&response);
    } else {
        printf("  [FAIL] API call failed: %s\n", gv_llm_error_string(result));
        const char *error = gv_llm_get_last_error(llm);
        if (error) {
            printf("  Error details: %s\n", error);
        }
    }
    
    gv_llm_destroy(llm);
}

int main(void) {
    printf("=== LLM Integration Tests ===\n\n");
    
    printf("Note: Set API keys in .env file or environment variables to enable real API call tests.\n");
    printf("      Create a .env file with:\n");
    printf("        OPENAI_API_KEY=sk-your-key\n");
    printf("        ANTHROPIC_API_KEY=sk-ant-your-key\n");
    printf("        GEMINI_API_KEY=your-gemini-key\n\n");
    
    test_llm_create_valid();
    printf("\n");
    
    test_llm_create_invalid_api_key();
    printf("\n");
    
    test_llm_create_invalid_url();
    printf("\n");
    
    test_custom_requires_base_url();
    printf("\n");
    
    test_error_strings();
    printf("\n");
    
    test_anthropic_api_key();
    printf("\n");
    
    printf("--- Real API Call Tests ---\n\n");
    test_llm_api_call_openai();
    printf("\n");
    
    test_llm_api_call_anthropic();
    printf("\n");
    
    test_llm_api_call_gemini();
    printf("\n");
    
    printf("=== Tests Complete ===\n");
    return 0;
}





