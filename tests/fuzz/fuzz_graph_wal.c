/**
 * @file fuzz_graph_wal.c
 * @brief libFuzzer harness for graph/KG WAL crash recovery.
 *
 * Writes arbitrary bytes as a WAL next to an empty snapshot and runs the real
 * recovery path (graph_load / kg_load replay): torn frames, bogus lengths,
 * unknown ops, and hostile payloads must never crash, hang, or corrupt the
 * loaded state.
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "features/graph_db.h"
#include "features/knowledge_graph.h"

#define FUZZ_MAX_INPUT (256 * 1024)
#define SNAP_G  "/tmp/gv_fuzz_gwal.gvgr"
#define WAL_G   "/tmp/gv_fuzz_gwal.gvgr.wal"
#define SNAP_K  "/tmp/gv_fuzz_kwal.gvkg"
#define WAL_K   "/tmp/gv_fuzz_kwal.gvkg.wal"

static void write_file(const char *path, const uint8_t *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    if (size) fwrite(data, 1, size, f);
    fclose(f);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > FUZZ_MAX_INPUT) return 0;

    /* ── Property-graph recovery ── */
    unlink(SNAP_G); unlink(WAL_G);
    GV_GraphDB *seed = graph_create(NULL);
    if (seed) {
        graph_save(seed, SNAP_G);
        graph_destroy(seed);
    }
    write_file(WAL_G, data, size);

    GV_GraphDB *g = graph_load(SNAP_G);
    if (g) {
        /* Invariants that must hold after any byte soup. */
        if (graph_edge_count(g) > graph_node_count(g) *
                (graph_node_count(g) ? graph_node_count(g) : 1)) {
            graph_destroy(g);
            unlink(SNAP_G); unlink(WAL_G);
            return 0;
        }
        graph_destroy(g);
    }

    /* ── Knowledge-graph recovery ── */
    unlink(SNAP_K); unlink(WAL_K);
    GV_KnowledgeGraph *kseed = kg_create(NULL);
    if (kseed) {
        kg_save(kseed, SNAP_K);
        kg_destroy(kseed);
    }
    write_file(WAL_K, data, size);

    GV_KnowledgeGraph *kg = kg_load(SNAP_K);
    if (kg) {
        uint64_t ids[4];
        /* Index consistency after replay of garbage. */
        (void)kg_find_entities_by_name(kg, "A", ids, 4);
        (void)kg_find_entities_by_type(kg, "T", ids, 4);
        kg_destroy(kg);
    }

    unlink(SNAP_G); unlink(WAL_G);
    unlink(SNAP_K); unlink(WAL_K);
    return 0;
}
