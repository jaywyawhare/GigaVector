#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "storage/database.h"
#include "storage/memory_layer.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static GV_MemoryLayer *make_layer(GV_Database **out_db) {
    GV_Database *db = db_open(NULL, 8, GV_INDEX_TYPE_FLAT);
    if (db == NULL) {
        return NULL;
    }
    GV_MemoryLayer *layer = memory_layer_create(db, NULL);
    if (layer == NULL) {
        db_close(db);
        return NULL;
    }
    *out_db = db;
    return layer;
}

static float *make_embedding(int seed) {
    float *embedding = (float *)gv_alloc(8 * sizeof(float));
    if (embedding == NULL) {
        return NULL;
    }
    for (int i = 0; i < 8; i++) {
        embedding[i] = (float)(seed + i) / 16.0f;
    }
    return embedding;
}

static int test_memory_get_update_delete(void) {
    GV_Database *db = NULL;
    GV_MemoryLayer *layer = make_layer(&db);
    ASSERT(layer != NULL, "create layer");

    float *embedding = make_embedding(1);
    ASSERT(embedding != NULL, "embedding");

    GV_MemoryMetadata meta;
    memset(&meta, 0, sizeof(meta));
    meta.memory_type = GV_MEMORY_TYPE_FACT;
    meta.importance_score = 0.7;
    meta.timestamp = time(NULL);

    char *memory_id = memory_add(layer, "alpha fact", embedding, &meta, NULL);
    ASSERT(memory_id != NULL, "add memory");

    GV_MemoryResult got;
    ASSERT(memory_get(layer, memory_id, &got) == 0, "get after add");
    ASSERT(got.content != NULL && strcmp(got.content, "alpha fact") == 0, "content matches");
    memory_result_free(&got);

    meta.importance_score = 0.95;
    ASSERT(memory_update(layer, memory_id, NULL, &meta) == 0, "update metadata");
    ASSERT(memory_get(layer, memory_id, &got) == 0, "get after update");
    ASSERT(got.metadata != NULL, "metadata present");
    ASSERT(got.metadata->importance_score > 0.94, "importance updated");
    memory_result_free(&got);

    ASSERT(memory_delete(layer, memory_id) == 0, "delete memory");
    ASSERT(memory_get(layer, memory_id, &got) != 0, "get after delete fails");

    gv_free(embedding);
    gv_free(memory_id);
    memory_layer_destroy(layer);
    db_close(db);
    return 0;
}

static int test_memory_typed_links(void) {
    GV_Database *db = NULL;
    GV_MemoryLayer *layer = make_layer(&db);
    ASSERT(layer != NULL, "create layer");

    float *emb_a = make_embedding(2);
    float *emb_b = make_embedding(5);
    ASSERT(emb_a != NULL && emb_b != NULL, "embeddings");

    char *id_a = memory_add(layer, "fact a", emb_a, NULL, NULL);
    char *id_b = memory_add(layer, "fact b", emb_b, NULL, NULL);
    ASSERT(id_a != NULL && id_b != NULL, "add memories");

    ASSERT(memory_link_create(layer, id_a, id_b, GV_LINK_CONTRADICTS, 1.0f, "test") == 0,
           "create link");

    GV_MemoryLink links[8];
    int link_count = memory_link_get(layer, id_a, links, 8);
    ASSERT(link_count == 1, "one forward link");
    ASSERT(links[0].target_memory_id != NULL &&
           strcmp(links[0].target_memory_id, id_b) == 0, "forward target");
    ASSERT(links[0].link_type == GV_LINK_CONTRADICTS, "link type");
    memory_link_free(&links[0]);

    link_count = memory_link_get(layer, id_b, links, 8);
    ASSERT(link_count == 1, "reciprocal link");
    ASSERT(links[0].target_memory_id != NULL &&
           strcmp(links[0].target_memory_id, id_a) == 0, "reciprocal target");
    memory_link_free(&links[0]);

    ASSERT(memory_link_remove(layer, id_a, id_b) == 0, "remove link");
    link_count = memory_link_get(layer, id_a, links, 8);
    ASSERT(link_count == 0, "forward link removed");

    gv_free(emb_a);
    gv_free(emb_b);
    gv_free(id_a);
    gv_free(id_b);
    memory_layer_destroy(layer);
    db_close(db);
    return 0;
}

static int test_memory_record_access_persists(void) {
    GV_Database *db = NULL;
    GV_MemoryLayer *layer = make_layer(&db);
    ASSERT(layer != NULL, "create layer");

    float *embedding = make_embedding(9);
    ASSERT(embedding != NULL, "embedding");

    char *memory_id = memory_add(layer, "access me", embedding, NULL, NULL);
    ASSERT(memory_id != NULL, "add memory");

    ASSERT(memory_record_access(layer, memory_id, 0.8f) == 0, "record access");

    GV_MemoryResult got;
    ASSERT(memory_get(layer, memory_id, &got) == 0, "get after access");
    ASSERT(got.metadata != NULL, "metadata present");
    ASSERT(got.metadata->access_count == 1, "access count persisted");
    ASSERT(got.metadata->last_accessed > 0, "last accessed persisted");
    memory_result_free(&got);

    gv_free(embedding);
    gv_free(memory_id);
    memory_layer_destroy(layer);
    db_close(db);
    return 0;
}

static int test_memory_hnsw_delete(void) {
    GV_Database *db = db_open(NULL, 8, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "create hnsw db");
    GV_MemoryLayer *layer = memory_layer_create(db, NULL);
    ASSERT(layer != NULL, "create layer");

    float *embedding = make_embedding(3);
    ASSERT(embedding != NULL, "embedding");

    char *memory_id = memory_add(layer, "delete me on hnsw", embedding, NULL, NULL);
    ASSERT(memory_id != NULL, "add memory");

    ASSERT(memory_delete(layer, memory_id) == 0, "delete memory");

    GV_MemoryResult got;
    ASSERT(memory_get(layer, memory_id, &got) != 0, "get after hnsw delete fails");

    gv_free(embedding);
    gv_free(memory_id);
    memory_layer_destroy(layer);
    db_close(db);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"memory_get_update_delete", test_memory_get_update_delete},
        {"memory_typed_links", test_memory_typed_links},
        {"memory_record_access_persists", test_memory_record_access_persists},
        {"memory_hnsw_delete", test_memory_hnsw_delete},
    };
    int n = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "Running %s...\n", tests[i].name);
        if (tests[i].fn() == 0) {
            passed++;
            fprintf(stderr, "PASS: %s\n", tests[i].name);
        } else {
            fprintf(stderr, "FAILED: %s\n", tests[i].name);
            return 1;
        }
    }
    fprintf(stderr, "%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
