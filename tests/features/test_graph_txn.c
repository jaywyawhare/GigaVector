#include <stdio.h>
#include <string.h>

#include "features/graph_db.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

int main(void) {
    GV_GraphDB *g = graph_create(NULL);
    ASSERT(g != NULL, "create");

    uint64_t v0 = graph_version(g);
    ASSERT(v0 == 0, "fresh graph version is 0");

    uint64_t a = graph_add_node(g, "Person");
    ASSERT(graph_version(g) == v0 + 1, "version bumps on mutation");

    ASSERT(graph_set_node_prop(g, a, "name", "Alice") == 0, "set prop");
    ASSERT(graph_version(g) > v0 + 1, "prop set bumps too");

    /* Open a snapshot; mutations from other threads cannot commit under it,
     * and reads inside see a stable version. */
    GV_GraphReadTxn *txn = graph_read_txn_begin(g);
    ASSERT(txn != NULL, "begin txn");
    ASSERT(graph_read_txn_get_node(txn, a) != NULL, "txn get node");
    ASSERT(strcmp(graph_read_txn_get_node_prop(txn, a, "name"), "Alice") == 0,
           "txn prop");
    uint64_t ids[4];
    ASSERT(graph_read_txn_find_nodes_by_label(txn, "Person", ids, 4) == 1,
           "txn label search");

    /* Same-thread writes are not prevented by rwlock semantics; the txn
     * guarantees a consistent view only for threads that respect the lock
     * (all library mutations do). */
    graph_read_txn_end(txn);

    graph_destroy(g);
    printf("All graph read-txn tests PASSED\n");
    return 0;
}
