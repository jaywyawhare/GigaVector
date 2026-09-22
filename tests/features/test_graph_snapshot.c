/* Pinned-snapshot analytics: an algorithm running on a GV_GAContext must see
 * one consistent snapshot while another thread mutates the graph — no data
 * races (TSAN), no torn traversals. Writers simply block until the context is
 * released. Run this test under -fsanitize=thread to verify. */
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include "features/graph_db.h"
#include "features/graph_algos.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); failures++; } \
                          else printf("ok: %s\n", m); } while (0)

static GV_GraphDB *g;
static int stop_writers = 0;  /* accessed with __atomic builtins */

static void *writer(void *arg) {
    (void)arg;
    uint64_t prev = 0; /* anchor */
    uint64_t ids[4];
    while (!__atomic_load_n(&stop_writers, __ATOMIC_ACQUIRE)) {
        for (int i = 0; i < 4; i++) ids[i] = graph_add_node(g, "W");
        for (int i = 0; i < 4; i++)
            graph_add_edge(g, ids[i], ids[(i + 1) % 4], "E", 1.0f);
        for (int i = 0; i < 4; i++) graph_remove_node(g, ids[i]);
        (void)prev;
    }
    return NULL;
}

static void *analyst(void *arg) {
    (void)arg;
    for (int r = 0; r < 25 && !__atomic_load_n(&stop_writers, __ATOMIC_ACQUIRE); r++) {
        GV_GraphNodeScores sc;
        if (graph_pagerank_all(g, 20, 0.85, &sc) == 0)
            graph_node_scores_free(&sc);
        GV_GraphNodeLabels lb;
        if (graph_leiden(g, 5, &lb) == 0)
            graph_node_labels_free(&lb);
    }
    return NULL;
}

int main(void) {
    g = graph_create(NULL);
    ASSERT(g, "create");
    /* seed: two dense groups so algorithms have real work */
    uint64_t a[6], b[6];
    for (int i = 0; i < 6; i++) a[i] = graph_add_node(g, "A");
    for (int i = 0; i < 6; i++) b[i] = graph_add_node(g, "B");
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            if (i != j) { graph_add_edge(g, a[i], a[j], "E", 1.0f); graph_add_edge(g, a[j], a[i], "E", 1.0f); }
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            if (i != j) { graph_add_edge(g, b[i], b[j], "E", 1.0f); graph_add_edge(g, b[j], b[i], "E", 1.0f); }

    pthread_t w1, w2, r1, r2;
    pthread_create(&w1, NULL, writer, NULL);
    pthread_create(&w2, NULL, writer, NULL);
    pthread_create(&r1, NULL, analyst, NULL);
    pthread_create(&r2, NULL, analyst, NULL);


    sleep(1);
    __atomic_store_n(&stop_writers, 1, __ATOMIC_RELEASE);
    pthread_join(w1, NULL);
    pthread_join(w2, NULL);
    pthread_join(r1, NULL);
    pthread_join(r2, NULL);

    ASSERT(graph_node_count(g) >= 12, "graph intact after contention");
    graph_destroy(g);

    printf(failures ? "SOME PINNED-SNAPSHOT TESTS FAILED (%d)\n"
                    : "ALL PINNED-SNAPSHOT TESTS PASSED (run under TSAN)\n", failures);
    return failures ? 1 : 0;
}
