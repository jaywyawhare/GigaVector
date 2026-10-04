#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/memory_layer.h"
#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_memory_layer_config_default(void) {
    GV_MemoryLayerConfig config = memory_layer_config_default();
    ASSERT(config.extraction_threshold > 0, "extraction_threshold is positive");
    ASSERT(config.consolidation_threshold > 0, "consolidation_threshold is positive");
    ASSERT(config.max_related_memories > 0, "max_related_memories is positive");
    return 0;
}

static int test_memory_layer_create_null_db(void) {
    GV_MemoryLayer *layer = memory_layer_create(NULL, NULL);
    ASSERT(layer == NULL, "create with null db fails");
    return 0;
}

static int test_memory_layer_create(void) {
    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create db");
    GV_MemoryLayer *layer = memory_layer_create(db, NULL);
    ASSERT(layer != NULL, "create memory layer");
    memory_layer_destroy(layer);
    db_close(db);
    return 0;
}

static int test_memory_layer_destroy_null(void) {
    memory_layer_destroy(NULL);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

static int test_memory_link_reciprocal(void) {
    /* Directional links invert; symmetric links map to themselves. */
    ASSERT(memory_link_reciprocal(GV_LINK_CAUSAL) == GV_LINK_CAUSES, "CAUSAL -> CAUSES");
    ASSERT(memory_link_reciprocal(GV_LINK_CAUSES) == GV_LINK_CAUSAL, "CAUSES -> CAUSAL");
    ASSERT(memory_link_reciprocal(GV_LINK_PREREQUISITE) == GV_LINK_DEPENDENT, "PREREQUISITE -> DEPENDENT");
    ASSERT(memory_link_reciprocal(GV_LINK_DEPENDENT) == GV_LINK_PREREQUISITE, "DEPENDENT -> PREREQUISITE");
    ASSERT(memory_link_reciprocal(GV_LINK_CONTRADICTS) == GV_LINK_CONTRADICTS, "CONTRADICTS symmetric");
    ASSERT(memory_link_reciprocal(GV_LINK_SIMILAR) == GV_LINK_SIMILAR, "SIMILAR symmetric");
    /* Reciprocal of reciprocal returns the original for directional pairs. */
    ASSERT(memory_link_reciprocal(memory_link_reciprocal(GV_LINK_CAUSAL)) == GV_LINK_CAUSAL,
           "CAUSAL round-trips");
    ASSERT(memory_link_reciprocal(memory_link_reciprocal(GV_LINK_PREREQUISITE)) == GV_LINK_PREREQUISITE,
           "PREREQUISITE round-trips");
    return 0;
}

int main(void) {
    TestCase tests[] = {
        {"memory_layer_config_default", test_memory_layer_config_default},
        {"memory_layer_create_null_db", test_memory_layer_create_null_db},
        {"memory_layer_create", test_memory_layer_create},
        {"memory_layer_destroy_null", test_memory_layer_destroy_null},
        {"memory_link_reciprocal", test_memory_link_reciprocal},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) passed++;
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}