#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_memory_layer.h"

int test_memory_layer_basic(void) {
    GV_Database *db = gv_db_open(NULL, 128, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        fprintf(stderr, "Failed to create database\n");
        return 1;
    }
    
    GV_MemoryLayerConfig config = gv_memory_layer_config_default();
    GV_MemoryLayer *layer = gv_memory_layer_create(db, &config);
    if (layer == NULL) {
        fprintf(stderr, "Failed to create memory layer\n");
        gv_db_close(db);
        return 1;
    }
    
    float embedding[128];
    for (int i = 0; i < 128; i++) {
        embedding[i] = (float)i / 128.0f;
    }
    
    GV_MemoryMetadata meta;
    memset(&meta, 0, sizeof(meta));
    meta.memory_type = GV_MEMORY_TYPE_FACT;
    meta.timestamp = time(NULL);
    meta.importance_score = 0.8;
    meta.consolidated = 0;
    
    char *memory_id = gv_memory_add(layer, "User prefers Python over Java", embedding, &meta);
    if (memory_id == NULL) {
        fprintf(stderr, "Failed to add memory\n");
        gv_memory_layer_destroy(layer);
        gv_db_close(db);
        return 1;
    }
    
    GV_MemoryResult result;
    int ret = gv_memory_get(layer, memory_id, &result);
    if (ret != 0) {
        fprintf(stderr, "Failed to get memory\n");
        free(memory_id);
        gv_memory_layer_destroy(layer);
        gv_db_close(db);
        return 1;
    }
    
    assert(strcmp(result.content, "User prefers Python over Java") == 0);
    assert(result.metadata != NULL);
    assert(result.metadata->memory_type == GV_MEMORY_TYPE_FACT);
    
    gv_memory_result_free(&result);
    free(memory_id);
    gv_memory_layer_destroy(layer);
    gv_db_close(db);
    
    printf("test_memory_layer_basic: PASSED\n");
    return 0;
}

int test_memory_search(void) {
    GV_Database *db = gv_db_open(NULL, 128, GV_INDEX_TYPE_HNSW);
    if (db == NULL) {
        return 1;
    }
    
    GV_MemoryLayer *layer = gv_memory_layer_create(db, NULL);
    if (layer == NULL) {
        gv_db_close(db);
        return 1;
    }
    
    float embedding1[128], embedding2[128], query[128];
    for (int i = 0; i < 128; i++) {
        embedding1[i] = (float)i / 128.0f;
        embedding2[i] = (float)(i + 1) / 128.0f;
        query[i] = (float)i / 128.0f;
    }
    
    gv_memory_add(layer, "Memory 1", embedding1, NULL);
    gv_memory_add(layer, "Memory 2", embedding2, NULL);
    
    GV_MemoryResult results[10];
    int count = gv_memory_search(layer, query, 10, results, GV_DISTANCE_COSINE);
    
    if (count < 0) {
        fprintf(stderr, "Search failed\n");
        gv_memory_layer_destroy(layer);
        gv_db_close(db);
        return 1;
    }
    
    for (int i = 0; i < count; i++) {
        gv_memory_result_free(&results[i]);
    }
    
    gv_memory_layer_destroy(layer);
    gv_db_close(db);
    
    printf("test_memory_search: PASSED\n");
    return 0;
}

int main(void) {
    int failures = 0;
    
    failures += test_memory_layer_basic();
    failures += test_memory_search();
    
    if (failures == 0) {
        printf("All memory layer tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}

