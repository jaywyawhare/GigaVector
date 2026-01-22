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
    printf("Testing Google Embedding API...\n");
    
    const char *api_key = get_env_api_key("GEMINI_API_KEY");
    if (!api_key) {
        printf("[SKIP] Skipping: GEMINI_API_KEY environment variable not set\n");
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
        printf("[FAIL] Failed to create embedding service\n");
        return;
    }
    
    printf("[OK] Embedding service created\n");
    
    const char *text = "Hello, GigaVector! This is a test of the embedding service.";
    size_t embedding_dim = 0;
    float *embedding = NULL;
    
    printf("[INFO] Generating embedding for: \"%s\"\n", text);
    int result = gv_embedding_generate(service, text, &embedding_dim, &embedding);
    
    if (result == 0 && embedding != NULL) {
        printf("[OK] Embedding generated successfully!\n");
        printf("[DIM] Dimension: %zu\n", embedding_dim);
        printf("[EMBEDDING] First 10 values: ");
        size_t print_count = embedding_dim < 10 ? embedding_dim : 10;
        for (size_t i = 0; i < print_count; i++) {
            printf("%.6f", embedding[i]);
            if (i < print_count - 1) printf(", ");
        }
        printf("\n");
        
        if (embedding_dim > 10) {
            printf("[EMBEDDING] Last 10 values: ");
            for (size_t i = embedding_dim - 10; i < embedding_dim; i++) {
                printf("%.6f", embedding[i]);
                if (i < embedding_dim - 1) printf(", ");
            }
            printf("\n");
        }
        
        
        float min_val = embedding[0];
        float max_val = embedding[0];
        float sum = 0.0f;
        for (size_t i = 0; i < embedding_dim; i++) {
            if (embedding[i] < min_val) min_val = embedding[i];
            if (embedding[i] > max_val) max_val = embedding[i];
            sum += embedding[i];
        }
        float mean = sum / embedding_dim;
        
        printf("[STATS] Min: %.6f, Max: %.6f, Mean: %.6f\n", min_val, max_val, mean);
        
        free(embedding);
    } else {
        printf("[FAIL] Failed to generate embedding\n");
    }
    
    gv_embedding_service_destroy(service);
}

void test_google_embedding_batch(void) {
    printf("\nTesting Google Embedding Batch API...\n");
    
    const char *api_key = get_env_api_key("GEMINI_API_KEY");
    if (!api_key) {
        printf("[SKIP] Skipping: GEMINI_API_KEY environment variable not set\n");
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
        printf("[FAIL] Failed to create embedding service\n");
        return;
    }
    
    const char *texts[] = {
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Embeddings are vector representations of text"
    };
    size_t text_count = 3;
    
    size_t *embedding_dims = (size_t *)malloc(text_count * sizeof(size_t));
    float **embeddings = (float **)malloc(text_count * sizeof(float *));
    
    printf("[INFO] Generating embeddings for %zu texts...\n", text_count);
    int result = gv_embedding_generate_batch(service, texts, text_count, 
                                             &embedding_dims, &embeddings);
    
    printf("[DEBUG] Batch result: %d (expected >= 0 for success)\n", result);
    
    if (result >= 0 && result == (int)text_count) {
        printf("[OK] Batch embedding generated successfully!\n");
        for (size_t i = 0; i < text_count; i++) {
            printf("[TEXT %zu] \"%s\"\n", i + 1, texts[i]);
            printf("    Dimension: %zu\n", embedding_dims[i]);
            if (embeddings[i] != NULL) {
                printf("    First 5 values: ");
                size_t print_count = embedding_dims[i] < 5 ? embedding_dims[i] : 5;
                for (size_t j = 0; j < print_count; j++) {
                    printf("%.6f", embeddings[i][j]);
                    if (j < print_count - 1) printf(", ");
                }
                printf("\n");
                free(embeddings[i]);
            }
        }
        free(embedding_dims);
        free(embeddings);
    } else {
        printf("[FAIL] Failed to generate batch embeddings\n");
        free(embedding_dims);
        free(embeddings);
    }
    
    gv_embedding_service_destroy(service);
}

void test_openai_embedding(void) {
    printf("\nTesting OpenAI Embedding API\n");
    
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        printf("[SKIP] Skipping: OPENAI_API_KEY environment variable not set\n");
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
        printf("[FAIL] Failed to create embedding service\n");
        return;
    }
    
    printf("[OK] Embedding service created\n");
    
    const char *text = "hello world";
    size_t embedding_dim = 0;
    float *embedding = NULL;
    
    printf("[INFO] Generating embedding for: \"%s\"\n", text);
    int result = gv_embedding_generate(service, text, &embedding_dim, &embedding);
    
    if (result == 0 && embedding != NULL) {
        printf("[OK] Embedding generated successfully!\n");
        printf("[DIM] Dimension: %zu\n", embedding_dim);
        printf("[EMBEDDING] First 10 values: ");
        size_t print_count = embedding_dim < 10 ? embedding_dim : 10;
        for (size_t i = 0; i < print_count; i++) {
            printf("%.6f", embedding[i]);
            if (i < print_count - 1) printf(", ");
        }
        printf("\n");
        
        if (embedding_dim > 10) {
            printf("[EMBEDDING] Last 10 values: ");
            for (size_t i = embedding_dim - 10; i < embedding_dim; i++) {
                printf("%.6f", embedding[i]);
                if (i < embedding_dim - 1) printf(", ");
            }
            printf("\n");
        }
        
        
        float min_val = embedding[0];
        float max_val = embedding[0];
        float sum = 0.0f;
        for (size_t i = 0; i < embedding_dim; i++) {
            if (embedding[i] < min_val) min_val = embedding[i];
            if (embedding[i] > max_val) max_val = embedding[i];
            sum += embedding[i];
        }
        float mean = sum / embedding_dim;
        
        printf("[STATS] Min: %.6f, Max: %.6f, Mean: %.6f\n", min_val, max_val, mean);
        
        free(embedding);
    } else {
        printf("[FAIL] Failed to generate embedding (result: %d)\n", result);
    }
    
    gv_embedding_service_destroy(service);
}

void test_openai_embedding_batch(void) {
    printf("\nTesting OpenAI Embedding Batch API\n");
    
    const char *api_key = get_env_api_key("OPENAI_API_KEY");
    if (!api_key) {
        printf("[SKIP] Skipping: OPENAI_API_KEY environment variable not set\n");
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
        printf("[FAIL] Failed to create embedding service\n");
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
    
    printf("[INFO] Generating embeddings for %zu texts...\n", text_count);
    int result = gv_embedding_generate_batch(service, texts, text_count, 
                                             &embedding_dims, &embeddings);
    
    printf("[DEBUG] Batch result: %d (expected >= 0 for success)\n", result);
    
    if (result >= 0 && result == (int)text_count) {
        printf("[OK] Batch embedding generated successfully!\n");
        for (size_t i = 0; i < text_count; i++) {
            printf("[TEXT %zu] \"%s\"\n", i + 1, texts[i]);
            printf("    Dimension: %zu\n", embedding_dims[i]);
            if (embeddings[i] != NULL) {
                printf("    First 5 values: ");
                size_t print_count = embedding_dims[i] < 5 ? embedding_dims[i] : 5;
                for (size_t j = 0; j < print_count; j++) {
                    printf("%.6f", embeddings[i][j]);
                    if (j < print_count - 1) printf(", ");
                }
                printf("\n");
                free(embeddings[i]);
            }
        }
        free(embedding_dims);
        free(embeddings);
    } else {
        printf("[FAIL] Failed to generate batch embeddings (result: %d)\n", result);
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

