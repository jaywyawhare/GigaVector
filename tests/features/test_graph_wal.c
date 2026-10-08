#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "features/graph_db.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

/* Build a small graph, save the snapshot, then mutate with the WAL attached
 * and never save again - simulating a crash after the mutations. */
static int test_wal_replay_after_crash(void) {
    char path[512];
    ASSERT(gv_test_make_temp_path(path, sizeof(path), "gv_graphwal", ".gvgr") == 0,
           "temp path");

    GV_GraphDB *g = graph_create(NULL);
    ASSERT(g != NULL, "create");
    ASSERT(graph_save(g, path) == 0, "initial snapshot save");
    ASSERT(graph_wal_attach(g, path) == 0, "wal attach");

    uint64_t a = graph_add_node(g, "Person");
    uint64_t b = graph_add_node(g, "Person");
    uint64_t c = graph_add_node(g, "Company");
    ASSERT(a && b && c, "add nodes");
    uint64_t e1 = graph_add_edge(g, a, b, "KNOWS", 2.0f);
    uint64_t e2 = graph_add_edge(g, a, c, "WORKS_AT", 1.0f);
    ASSERT(e1 && e2, "add edges");
    ASSERT(graph_set_node_prop(g, a, "name", "Alice") == 0, "node prop");
    ASSERT(graph_set_edge_prop(g, e1, "since", "2020") == 0, "edge prop");

    /* Mutations that must also survive replay. */
    uint64_t d = graph_add_node(g, "Person");
    ASSERT(d, "add node d");
    ASSERT(graph_remove_node(g, b) == 0, "remove b");   /* cascades e1 */
    ASSERT(graph_remove_edge(g, e2) == 0, "remove edge e2");

    /* Crash: destroy without saving. The WAL file survives on disk. */
    graph_destroy(g);

    /* Recovery: load snapshot + replay WAL. */
    GV_GraphDB *r = graph_load(path);
    ASSERT(r != NULL, "load + replay");

    ASSERT(graph_node_count(r) == 3, "replayed node count (a, c, d)");
    uint64_t ids[8];
    int n = graph_find_nodes_by_label(r, "Person", ids, 8);
    ASSERT(n == 2, "replayed Person count");
    ASSERT(graph_get_node_prop(r, a, "name") != NULL &&
           strcmp(graph_get_node_prop(r, a, "name"), "Alice") == 0,
           "replayed node prop");
    ASSERT(graph_in_degree(r, c) == 0 && graph_out_degree(r, c) == 0,
           "removed edges stayed removed");
    ASSERT(graph_edge_count(r) == 0, "all edges gone via cascade + removal");

    /* The recovered WAL stays attached: further mutations keep logging. */
    uint64_t f = graph_add_node(r, "Person");
    ASSERT(f > 0, "post-recovery mutation");
    graph_destroy(r);

    /* Reload once more: snapshot + grown WAL must include post-recovery op. */
    GV_GraphDB *r2 = graph_load(path);
    ASSERT(r2 != NULL, "second reload");
    ASSERT(graph_node_count(r2) == 4, "second reload includes new node");
    graph_destroy(r2);

    unlink(path);
    char wal_path[540];
    snprintf(wal_path, sizeof(wal_path), "%s.wal", path);
    unlink(wal_path);
    return 0;
}

/* A save to the attached snapshot path truncates the WAL (checkpoint). */
static int test_wal_checkpoint_on_save(void) {
    char path[512];
    ASSERT(gv_test_make_temp_path(path, sizeof(path), "gv_graphwal_ckpt", ".gvgr") == 0,
           "temp path");

    GV_GraphDB *g = graph_create(NULL);
    ASSERT(g != NULL, "create");
    ASSERT(graph_save(g, path) == 0, "save empty");
    ASSERT(graph_wal_attach(g, path) == 0, "attach");

    uint64_t a = graph_add_node(g, "X");
    ASSERT(a, "add node");

    /* Saving to the attached path checkpoints the WAL away. */
    ASSERT(graph_save(g, path) == 0, "checkpointing save");

    graph_destroy(g);

    /* No crash happened; the log was truncated at save time, so loading must
     * produce exactly the saved state either way. */
    GV_GraphDB *r = graph_load(path);
    ASSERT(r != NULL, "reload after checkpoint");
    ASSERT(graph_node_count(r) == 1, "state intact after checkpoint");
    graph_destroy(r);

    unlink(path);
    char wal_path[540];
    snprintf(wal_path, sizeof(wal_path), "%s.wal", path);
    unlink(wal_path);
    return 0;
}

int main(void) {
    if (test_wal_replay_after_crash() != 0) return 1;
    printf("ok: WAL replay after crash\n");
    if (test_wal_checkpoint_on_save() != 0) return 1;
    printf("ok: WAL checkpoint on save\n");
    printf("All graph-WAL tests PASSED\n");
    return 0;
}
