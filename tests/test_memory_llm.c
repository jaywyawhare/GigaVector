#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_memory_layer.h"
#include "gigavector/gv_memory_extraction.h"
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

// Test memory extraction with LLM (will fallback if LLM unavailable)
void test_memory_extraction_llm(void) {
    printf("Testing memory extraction with LLM...\n");
    
    // Create database
    GV_Database *db = gv_db_open("test_memory_llm.db", 384, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        printf("  [FAIL] Failed to create database\n");
        return;
    }
    printf("  [OK] Database created\n");
    
    // Get API key from environment or use test key
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        printf("  [WARN] OPENAI_API_KEY not set, using test key (validation only)\n");
        api_key = "sk-test123456789012345678901234567890";
    }
    
    // Create LLM config
    GV_LLMConfig llm_config = {
        .provider = GV_LLM_PROVIDER_OPENAI,
        .api_key = (char *)api_key,
        .model = "gpt-4o-mini",
        .base_url = NULL,
        .temperature = 0.7,
        .max_tokens = 2000,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    // Create memory layer config with LLM
    GV_MemoryLayerConfig mem_config = gv_memory_layer_config_default();
    mem_config.llm_config = &llm_config;
    mem_config.use_llm_extraction = 1;
    
    GV_MemoryLayer *layer = gv_memory_layer_create(db, &mem_config);
    if (layer == NULL) {
        printf("  [FAIL] Failed to create memory layer\n");
        gv_db_close(db);
        return;
    }
    printf("  [OK] Memory layer created\n");
    
    // Test conversation
    const char *conversation = 
        "User: Hi, my name is John. I'm a software engineer.\n"
        "Assistant: Nice to meet you, John!\n"
        "User: I love pizza, especially margherita.\n"
        "Assistant: I'll remember that!\n";
    
    GV_MemoryCandidate candidates[10];
    size_t actual_count = 0;
    
    // Try LLM extraction (will fallback to heuristics if LLM unavailable)
    if (layer->llm != NULL) {
        const char *real_api_key = get_env_api_key("OPENAI_API_KEY");
        if (real_api_key) {
            printf("  [INFO] LLM available with real API key, attempting LLM extraction...\n");
        } else {
            printf("  [INFO] LLM available (test key), attempting LLM extraction...\n");
        }
        
        int result = gv_memory_extract_candidates_from_conversation_llm(
            (GV_LLM *)layer->llm, conversation, "test_conv_001", 0, NULL,
            candidates, 10, &actual_count
        );
        
        if (result == 0) {
            if (actual_count > 0) {
                printf("  [OK] LLM extraction succeeded: %zu candidates\n", actual_count);
                for (size_t i = 0; i < actual_count; i++) {
                    printf("    - %s (score: %.2f, type: %d)\n",
                           candidates[i].content,
                           candidates[i].importance_score,
                           candidates[i].memory_type);
                }
            } else {
                printf("  [OK] LLM extraction succeeded but returned empty list, will fallback to heuristics\n");
            }
        } else {
            if (real_api_key) {
                printf("  [WARN] LLM extraction failed (result: %d), will fallback to heuristics\n", result);
            } else {
                printf("  [WARN] LLM extraction failed (test key - expected), will fallback to heuristics\n");
            }
        }
    } else {
        printf("  [INFO] LLM not available (libcurl not compiled or invalid config)\n");
    }
    
    // Test fallback to heuristics
    printf("  [INFO] Testing heuristic extraction fallback...\n");
    actual_count = 0;
    int result = gv_memory_extract_candidates_from_conversation(
        conversation, "test_conv_001", 0.5,
        candidates, 10, &actual_count
    );
    
    if (result == 0 && actual_count > 0) {
        printf("  [OK] Heuristic extraction succeeded: %zu candidates\n", actual_count);
        for (size_t i = 0; i < actual_count; i++) {
            printf("    - %s (score: %.2f)\n", 
                   candidates[i].content,
                   candidates[i].importance_score);
        }
    } else {
        printf("  [WARN] Heuristic extraction returned no candidates\n");
    }
    
    // Cleanup
    for (size_t i = 0; i < actual_count; i++) {
        gv_memory_candidate_free(&candidates[i]);
    }
    
    gv_memory_layer_destroy(layer);
    gv_db_close(db);
    printf("  [OK] Cleanup complete\n");
}

// Test input validation
void test_input_validation(void) {
    printf("Testing input validation...\n");
    
    // Test conversation length limit
    char long_conversation[100002];
    memset(long_conversation, 'A', 100001);
    long_conversation[100001] = '\0';
    
    GV_MemoryCandidate candidates[10];
    size_t actual_count = 0;
    
    // Get API key from environment or use test key
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        api_key = "sk-test123456789012345678901234567890";
    }
    
    // This should fail due to length limit
    GV_LLMConfig llm_config = {
        .provider = GV_LLM_PROVIDER_OPENAI,
        .api_key = (char *)api_key,
        .model = "gpt-4o-mini"
    };
    
    GV_LLM *llm = gv_llm_create(&llm_config);
    if (llm != NULL) {
        int result = gv_memory_extract_candidates_from_conversation_llm(
            llm, long_conversation, "test", 0, NULL,
            candidates, 10, &actual_count
        );
        
        if (result != 0) {
            printf("  [OK] Correctly rejected conversation exceeding length limit\n");
        } else {
            printf("  [WARN] Should have rejected long conversation\n");
        }
        
        gv_llm_destroy(llm);
    } else {
        printf("  [WARN] LLM not available for validation test\n");
    }
}

// Test real API call for memory extraction
void test_memory_extraction_real_api(void) {
    printf("Testing memory extraction with real API call...\n");
    
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        printf("  [SKIP] Skipping: OPENAI_API_KEY environment variable not set\n");
        printf("  Set OPENAI_API_KEY to run this test\n");
        return;
    }
    
    // Create database
    GV_Database *db = gv_db_open("test_memory_llm_real.db", 384, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        printf("  [FAIL] Failed to create database\n");
        return;
    }
    printf("  [OK] Database created\n");
    
    // Create LLM config with real API key
    GV_LLMConfig llm_config = {
        .provider = GV_LLM_PROVIDER_OPENAI,
        .api_key = (char *)api_key,
        .model = "gpt-4o-mini",
        .base_url = NULL,
        .temperature = 0.7,
        .max_tokens = 2000,
        .timeout_seconds = 30,
        .custom_prompt = NULL
    };
    
    // Create memory layer config with LLM
    GV_MemoryLayerConfig mem_config = gv_memory_layer_config_default();
    mem_config.llm_config = &llm_config;
    mem_config.use_llm_extraction = 1;
    
    GV_MemoryLayer *layer = gv_memory_layer_create(db, &mem_config);
    if (layer == NULL) {
        printf("  [FAIL] Failed to create memory layer\n");
        gv_db_close(db);
        return;
    }
    printf("  [OK] Memory layer created\n");
    
    // Test conversation with real content
    const char *conversation = 
        "User: Hi, my name is Alice. I'm a software engineer working at Google.\n"
        "Assistant: Nice to meet you, Alice! That's interesting that you work at Google.\n"
        "User: I love Italian food, especially pasta and pizza. My favorite is margherita pizza.\n"
        "Assistant: I'll remember your food preferences!\n"
        "User: I'm working on a machine learning project using Python and TensorFlow.\n"
        "Assistant: That sounds like an exciting project!\n";
    
    GV_MemoryCandidate candidates[10];
    size_t actual_count = 0;
    
    printf("  [INFO] Making real API call for memory extraction...\n");
    int result = gv_memory_extract_candidates_from_conversation_llm(
        (GV_LLM *)layer->llm, conversation, "test_conv_real_001", 0, NULL,
        candidates, 10, &actual_count
    );
    
    if (result == 0) {
        if (actual_count > 0) {
            printf("  [OK] Real API call succeeded! Extracted %zu memories:\n", actual_count);
            for (size_t i = 0; i < actual_count; i++) {
                printf("    [%zu] %s (score: %.2f, type: %d)\n",
                       i + 1,
                       candidates[i].content,
                       candidates[i].importance_score,
                       candidates[i].memory_type);
            }
        } else {
            printf("  [OK] Real API call succeeded but no facts extracted (LLM returned empty list)\n");
        }
    } else {
        printf("  [FAIL] Real API call failed (result: %d)\n", result);
    }
    
    // Cleanup
    for (size_t i = 0; i < actual_count; i++) {
        gv_memory_candidate_free(&candidates[i]);
    }
    
    gv_memory_layer_destroy(layer);
    gv_db_close(db);
    printf("  [OK] Cleanup complete\n");
}

int main(void) {
    printf("=== Memory LLM Integration Tests ===\n\n");
    
    printf("Note: Set API keys in .env file or environment variables to enable real API call tests.\n");
    printf("      Create a .env file with:\n");
    printf("        OPENAI_API_KEY=sk-your-key\n\n");
    
    test_memory_extraction_llm();
    printf("\n");
    
    test_input_validation();
    printf("\n");
    
    printf("--- Real API Call Test ---\n\n");
    test_memory_extraction_real_api();
    printf("\n");
    
    printf("=== Tests Complete ===\n");
    return 0;
}





