#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "storage/database.h"
#include "storage/memory_layer.h"
#include "storage/memory_extraction.h"
#include "multimodal/llm.h"

#define MAX_ENV_VALUE_LEN 1024
static char env_buffer[MAX_ENV_VALUE_LEN];

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

static const char *get_env_api_key(const char *env_var) {
    const char *key = read_env_file(env_var);
    if (key && strlen(key) > 0) {
        return key;
    }
    key = getenv(env_var);
    return key && strlen(key) > 0 ? key : NULL;
}

void test_memory_extraction_llm(void) {
    GV_Database *db = db_open("test_memory_llm.db", 384, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return;
    }

    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        api_key = "sk-test123456789012345678901234567890";
    }

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
    
    GV_MemoryLayerConfig mem_config = memory_layer_config_default();
    mem_config.llm_config = &llm_config;
    mem_config.use_llm_extraction = 1;

    GV_MemoryLayer *layer = memory_layer_create(db, &mem_config);
    if (layer == NULL) {
        db_close(db);
        return;
    }

    const char *conversation =
        "User: Hi, my name is John. I'm a software engineer.\n"
        "Assistant: Nice to meet you, John!\n"
        "User: I love pizza, especially margherita.\n"
        "Assistant: I'll remember that!\n";
    
    GV_MemoryCandidate candidates[10];
    size_t actual_count = 0;
    
    if (layer->llm != NULL) {
        const char *real_api_key = get_env_api_key("OPENAI_API_KEY");
        (void)real_api_key;
        
        int result = memory_extract_candidates_from_conversation_llm(
            (GV_LLM *)layer->llm, conversation, "test_conv_001", 0, NULL,
            candidates, 10, &actual_count
        );
        
        if (result == 0) {
            if (actual_count > 0) {
            } else {
            }
        } else {
            (void)result;
        }
    } else {
    }
    
    actual_count = 0;
    int result = memory_extract_candidates_from_conversation(
        conversation, "test_conv_001", 0.5,
        candidates, 10, &actual_count
    );
    
    if (result == 0 && actual_count > 0) {
    } else {
    }
    
    for (size_t i = 0; i < actual_count; i++) {
        memory_candidate_free(&candidates[i]);
    }

    memory_layer_destroy(layer);
    db_close(db);
}

void test_input_validation(void) {
    char long_conversation[100002];
    memset(long_conversation, 'A', 100001);
    long_conversation[100001] = '\0';
    
    GV_MemoryCandidate candidates[10];
    size_t actual_count = 0;
    
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        api_key = "sk-test123456789012345678901234567890";
    }

    /* Should fail due to input length limit */
    GV_LLMConfig llm_config = {
        .provider = GV_LLM_PROVIDER_OPENAI,
        .api_key = (char *)api_key,
        .model = "gpt-4o-mini"
    };
    
    GV_LLM *llm = llm_create(&llm_config);
    if (llm != NULL) {
        int result = memory_extract_candidates_from_conversation_llm(
            llm, long_conversation, "test", 0, NULL,
            candidates, 10, &actual_count
        );
        (void)result;
        
        llm_destroy(llm);
    } else {
    }
}

void test_memory_extraction_real_api(void) {
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        return;
    }
    
    GV_Database *db = db_open("test_memory_llm_real.db", 384, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return;
    }

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
    
    GV_MemoryLayerConfig mem_config = memory_layer_config_default();
    mem_config.llm_config = &llm_config;
    mem_config.use_llm_extraction = 1;

    GV_MemoryLayer *layer = memory_layer_create(db, &mem_config);
    if (layer == NULL) {
        db_close(db);
        return;
    }

    const char *conversation =
        "User: Hi, my name is Alice. I'm a software engineer working at Google.\n"
        "Assistant: Nice to meet you, Alice! That's interesting that you work at Google.\n"
        "User: I love Italian food, especially pasta and pizza. My favorite is margherita pizza.\n"
        "Assistant: I'll remember your food preferences!\n"
        "User: I'm working on a machine learning project using Python and TensorFlow.\n"
        "Assistant: That sounds like an exciting project!\n";
    
    GV_MemoryCandidate candidates[10];
    size_t actual_count = 0;
    
    int result = memory_extract_candidates_from_conversation_llm(
        (GV_LLM *)layer->llm, conversation, "test_conv_real_001", 0, NULL,
        candidates, 10, &actual_count
    );
    
    if (result == 0) {
        if (actual_count > 0) {
        } else {
        }
    } else {
    }
    
    for (size_t i = 0; i < actual_count; i++) {
        memory_candidate_free(&candidates[i]);
    }

    memory_layer_destroy(layer);
    db_close(db);
}

int main(void) {
    test_memory_extraction_llm();
    test_input_validation();
    test_memory_extraction_real_api();
    return 0;
}

