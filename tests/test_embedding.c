#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_embedding.h"

static void read_env_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return;
    
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        
        if (line[0] == '#' || line[0] == '\0') continue;
        
        
        char *eq = strchr(line, '=');
        if (!eq) continue;
        
        *eq = '\0';
        char *key = line;
        char *value = eq + 1;
        
        
        if ((value[0] == '"' && value[strlen(value) - 1] == '"') ||
            (value[0] == '\'' && value[strlen(value) - 1] == '\'')) {
            value[strlen(value) - 1] = '\0';
            value++;
        }
        
        
        if (!getenv(key)) {
            setenv(key, value, 0);
        }
    }
    
    fclose(fp);
}

static const char *get_env_api_key(const char *key) {
    const char *value = getenv(key);
    if (value && strlen(value) > 0) {
        return value;
    }
    return NULL;
}

void test_google_embedding(void) {
    const char *api_key = get_env_api_key("GEMINI_API_KEY");
    if (!api_key) {
        return;
    }
    
    GV_EmbeddingConfig config = {
        .provider = GV_EMBEDDING_PROVIDER_GOOGLE,
        .api_key = (char *)api_key,
        .model = "text-embedding-004",
        .base_url = NULL,
        .embedding_dimension = 768,
        .batch_size = 10,
        .enable_cache = 1,
        .timeout_seconds = 30
    };
    
    GV_EmbeddingService *service = gv_embedding_service_create(&config);
    if (service == NULL) {
        return;
    }

    const char *text = "Hello, GigaVector! This is a test of the embedding service.";
    size_t embedding_dim = 0;
    float *embedding = NULL;
    int result = gv_embedding_generate(service, text, &embedding_dim, &embedding);
    
    if (result == 0 && embedding != NULL) {
        free(embedding);
    }
    
    gv_embedding_service_destroy(service);
}

void test_google_embedding_batch(void) {
    const char *api_key = get_env_api_key("GEMINI_API_KEY");
    if (!api_key) {
        return;
    }
    
    GV_EmbeddingConfig config = {
        .provider = GV_EMBEDDING_PROVIDER_GOOGLE,
        .api_key = (char *)api_key,
        .model = "text-embedding-004",
        .base_url = NULL,
        .embedding_dimension = 768,
        .batch_size = 10,
        .enable_cache = 1,
        .timeout_seconds = 30
    };
    
    GV_EmbeddingService *service = gv_embedding_service_create(&config);
    if (service == NULL) {
        return;
    }
    
    const char *texts[] = {
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Embeddings are vector representations of text"
    };
    size_t text_count = 3;
    
    size_t *embedding_dims = NULL;
    float **embeddings = NULL;
    int result = gv_embedding_generate_batch(service, texts, text_count, 
                                             &embedding_dims, &embeddings);

    if (result >= 0 && result == (int)text_count) {
        for (size_t i = 0; i < text_count; i++) {
            if (embeddings[i] != NULL) {
                free(embeddings[i]);
            }
        }
        free(embedding_dims);
        free(embeddings);
    } else {
        free(embedding_dims);
        free(embeddings);
    }
    
    gv_embedding_service_destroy(service);
}

void test_openai_embedding(void) {
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        return;
    }
    
    GV_EmbeddingConfig config = {
        .provider = GV_EMBEDDING_PROVIDER_OPENAI,
        .api_key = (char *)api_key,
        .model = "text-embedding-3-small",
        .base_url = NULL,
        .embedding_dimension = 0,
        .batch_size = 10,
        .enable_cache = 1,
        .timeout_seconds = 30
    };
    
    GV_EmbeddingService *service = gv_embedding_service_create(&config);
    if (service == NULL) {
        return;
    }

    const char *text = "hello world";
    size_t embedding_dim = 0;
    float *embedding = NULL;
    int result = gv_embedding_generate(service, text, &embedding_dim, &embedding);
    
    if (result == 0 && embedding != NULL) {
        free(embedding);
    }
    
    gv_embedding_service_destroy(service);
}

void test_openai_embedding_batch(void) {
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        return;
    }
    
    GV_EmbeddingConfig config = {
        .provider = GV_EMBEDDING_PROVIDER_OPENAI,
        .api_key = (char *)api_key,
        .model = "text-embedding-3-small",
        .base_url = NULL,
        .embedding_dimension = 0,
        .batch_size = 10,
        .enable_cache = 1,
        .timeout_seconds = 30
    };
    
    GV_EmbeddingService *service = gv_embedding_service_create(&config);
    if (service == NULL) {
        return;
    }
    
    const char *texts[] = {
        "hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence"
    };
    size_t text_count = 3;
    
    size_t *embedding_dims = NULL;
    float **embeddings = NULL;
    
    int result = gv_embedding_generate_batch(service, texts, text_count, 
                                             &embedding_dims, &embeddings);

    if (result >= 0 && result == (int)text_count) {
        for (size_t i = 0; i < text_count; i++) {
            if (embeddings[i] != NULL) {
                free(embeddings[i]);
            }
        }
        free(embedding_dims);
        free(embeddings);
    } else {
        free(embedding_dims);
        free(embeddings);
    }
    
    gv_embedding_service_destroy(service);
}

int main(void) {
    
    read_env_file(".env");
    
    // test_openai_embedding();
    // test_openai_embedding_batch();
    test_google_embedding();
    test_google_embedding_batch();
    
    return 0;
}

