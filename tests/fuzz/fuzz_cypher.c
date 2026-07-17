/**
 * @file fuzz_cypher.c
 * @brief libFuzzer harness for the Cypher query parser/executor
 *        (src/features/cypher.c).
 *
 * Build: make fuzz  (requires clang)
 * Run:   build/fuzz/fuzz_cypher tests/fuzz/corpus/cypher -runs=100000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "features/knowledge_graph.h"
#include "features/cypher.h"

#define FUZZ_MAX_INPUT (64 * 1024)

static GV_KnowledgeGraph *g_kg = NULL;
static GV_CypherEngine   *g_eng = NULL;

static void fuzz_setup_once(void) {
    if (g_eng) return;

    GV_KGConfig cfg;
    kg_config_init(&cfg);
    cfg.embedding_dimension = 0; /* no embeddings needed for parsing */
    g_kg = kg_create(&cfg);
    if (!g_kg) return;

    /* Seed a few nodes/edges so MATCH/RETURN paths have data to touch. */
    uint64_t a = kg_add_entity(g_kg, "Alice", "Person", NULL, 0);
    uint64_t b = kg_add_entity(g_kg, "Bob", "Person", NULL, 0);
    if (a && b) {
        kg_add_relation(g_kg, a, "KNOWS", b, 1.0f);
    }

    g_eng = cypher_create(g_kg);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;

    fuzz_setup_once();
    if (!g_eng) return 0;

    /* cypher_execute expects a NUL-terminated query string; copy and terminate. */
    char *query = (char *)gv_alloc(size + 1);
    if (!query) return 0;
    memcpy(query, data, size);
    query[size] = '\0';

    GV_CypherResult result;
    memset(&result, 0, sizeof(result));
    if (cypher_execute(g_eng, query, &result) == 0) {
        cypher_free_result(&result);
    }

    gv_free(query);
    return 0;
}

#ifdef GV_FUZZ_STANDALONE
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input-file>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint8_t buf[FUZZ_MAX_INPUT];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    return LLVMFuzzerTestOneInput(buf, n);
}
#endif
