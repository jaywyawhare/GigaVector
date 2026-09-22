#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "features/graph_db.h"
#include "features/knowledge_graph.h"
#include "features/graph_algos.h"
#include "features/graph_csr.h"
#include "features/cypher.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_write_txn_commit_and_abort(void) {
    GV_GraphDB *g = graph_create(NULL);
    ASSERT(g != NULL, "create");
    uint64_t base = graph_add_node(g, "Base");
    ASSERT(base, "base node");

    /* Abort: staged mutations must not be visible. */
    GV_GraphWriteTxn *t = graph_write_txn_begin(g);
    ASSERT(t, "begin");
    uint64_t x = graph_write_txn_add_node(t, "TxnNode");
    ASSERT(x, "staged add");
    ASSERT(graph_write_txn_add_edge(t, base, x, "L", 1.0f), "staged edge");
    graph_write_txn_abort(t);
    ASSERT(graph_node_count(g) == 1, "abort left no nodes");
    ASSERT(graph_edge_count(g) == 0, "abort left no edges");

    /* Commit: everything lands atomically, including edge to staged node. */
    t = graph_write_txn_begin(g);
    ASSERT(t, "begin 2");
    uint64_t a = graph_write_txn_add_node(t, "Person");
    uint64_t b = graph_write_txn_add_node(t, "Person");
    uint64_t e = graph_write_txn_add_edge(t, a, b, "KNOWS", 2.0f);
    ASSERT(a && b && e, "staged adds");
    ASSERT(graph_write_txn_set_node_prop(t, a, "name", "Zed") == 0, "stage prop");
    ASSERT(graph_write_txn_remove_edge(t, e) == 0, "stage remove of staged edge");
    ASSERT(graph_write_txn_add_edge(t, b, a, "BACK", 1.0f) > 0, "restage edge");
    ASSERT(graph_version(g) == 1 || graph_version(g) == 1,
           "version unchanged until commit");
    ASSERT(graph_write_txn_commit(t) == 0, "commit");
    ASSERT(graph_node_count(g) == 3, "committed nodes (base,a,b)");
    ASSERT(graph_edge_count(g) == 1, "only BACK edge survived txn ops");
    const GV_GraphNode *na = graph_get_node(g, a);
    ASSERT(na && strcmp(graph_get_node_prop(g, a, "name"), "Zed") == 0,
           "committed prop visible");

    graph_destroy(g);
    return 0;
}

static int test_write_txn_wal_durability(void) {
    char path[512];
    ASSERT(gv_test_make_temp_path(path, sizeof(path), "gv_gtxn", ".gvgr") == 0,
           "temp path");

    GV_GraphDB *g = graph_create(NULL);
    graph_save(g, path);
    graph_wal_attach(g, path);

    GV_GraphWriteTxn *t = graph_write_txn_begin(g);
    uint64_t a = graph_write_txn_add_node(t, "P");
    uint64_t b = graph_write_txn_add_node(t, "P");
    graph_write_txn_add_edge(t, a, b, "E", 1.0f);
    graph_write_txn_set_node_prop(t, a, "k", "v");
    ASSERT(graph_write_txn_commit(t) == 0, "commit with WAL");

    graph_destroy(g);   /* crash before save */

    GV_GraphDB *r = graph_load(path);   /* snapshot + batched WAL */
    ASSERT(r != NULL, "reload");
    ASSERT(graph_node_count(r) == 2, "txn batch replayed");
    ASSERT(graph_edge_count(r) == 1, "txn edge replayed");
    ASSERT(strcmp(graph_get_node_prop(r, a, "k"), "v") == 0, "txn prop replayed");
    graph_destroy(r);

    unlink(path);
    char wal_path[540];
    snprintf(wal_path, sizeof(wal_path), "%s.wal", path);
    unlink(wal_path);
    return 0;
}

static int test_kg_expand_and_prop_remove(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    ASSERT(kg, "kg create");
    uint64_t a = kg_add_entity(kg, "A", "T", NULL, 0);
    uint64_t b = kg_add_entity(kg, "B", "T", NULL, 0);
    uint64_t c = kg_add_entity(kg, "C", "T", NULL, 0);
    ASSERT(a && b && c, "entities");
    ASSERT(kg_add_relation(kg, a, "r1", b, 1.0f), "rel1");
    ASSERT(kg_add_relation(kg, b, "r2", c, 1.0f), "rel2");

    /* radius 1 from A: only A->B triple. radius 2 reaches B->C too. */
    GV_KGTriple tr[8];
    int n = kg_expand_context(kg, &a, 1, 1, tr, 8);
    ASSERT(n == 1, "radius 1 expansion");
    kg_free_triples(tr, n);
    n = kg_expand_context(kg, &a, 1, 2, tr, 8);
    ASSERT(n == 2, "radius 2 multi-hop expansion");
    kg_free_triples(tr, n);

    /* Property delete removes outright (not empty-string). */
    ASSERT(kg_set_entity_prop(kg, a, "tmp", "x") == 0, "set prop");
    ASSERT(kg_get_entity_prop(kg, a, "tmp") != NULL, "prop present");
    ASSERT(kg_remove_entity_prop(kg, a, "tmp") == 0, "remove prop");
    ASSERT(kg_get_entity_prop(kg, a, "tmp") == NULL, "prop gone");

    uint64_t rid = kg_add_relation_with_chunk(kg, a, "cc", c, 1.0f, "d:1");
    ASSERT(rid, "chunk relation");
    int nchk = kg_query_triples_by_chunk(kg, "d:1", tr, 8);
    ASSERT(nchk == 1, "chunk indexed");
    kg_free_triples(tr, nchk);
    ASSERT(kg_remove_relation_prop(kg, rid, "chunk_id") == 0, "rm chunk prop");
    ASSERT(kg_query_triples_by_chunk(kg, "d:1", tr, 8) == 0,
           "chunk index updated on prop delete");
    kg_free_triples(tr, 0);
    kg_destroy(kg);
    return 0;
}

static int test_approx_betweenness(void) {
    GV_GraphDB *g = graph_create(NULL);
    ASSERT(g, "create");

    /* K=N must reproduce exact betweenness bit-for-bit (stride 1, scale 1). */
    uint64_t prev = 0;
    for (int i = 1; i <= 9; i++) {
        uint64_t v = graph_add_node(g, "N");
        if (prev) graph_add_edge(g, prev, v, "E", 1.0f);
        if (i == 5) graph_add_edge(g, v, prev, "BACK", 1.0f);
        prev = v;
    }
    GV_GraphNodeScores exact, full;
    ASSERT(graph_betweenness_centrality(g, 0, 1, &exact) == 0, "exact run");
    ASSERT(graph_betweenness_centrality_approx(g, 0, 1, graph_node_count(g),
                                               &full) == 0, "K=N run");
    size_t N = graph_node_count(g);
    for (size_t i = 0; i < N; i++)
        ASSERT(fabs(exact.scores[i] - full.scores[i]) < 1e-9,
               "K=N equals exact");

    /* Sampled variant runs and stays within scaled-exact bounds: no node's
     * estimate may exceed the exact maximum times the N/K scale factor. */
    GV_GraphNodeScores sampled2;
    ASSERT(graph_betweenness_centrality_approx(g, 0, 0, 5, &sampled2) == 0,
           "sampled run");
    double emaxv = 0.0;
    for (size_t i = 0; i < N; i++) {
        if (exact.scores[i] > emaxv) emaxv = exact.scores[i];
        ASSERT(sampled2.scores[i] >= 0.0 && sampled2.scores[i] < 1e12,
               "sampled scores finite and non-negative");
    }
    double scale = (double)N / 5.0;
    double bound = emaxv * scale + 1e-6;
    for (size_t i = 0; i < N; i++)
        ASSERT(sampled2.scores[i] <= bound, "estimate within scaled bound");
    graph_node_scores_free(&exact);
    graph_node_scores_free(&full);
    graph_node_scores_free(&sampled2);
    graph_destroy(g);
    return 0;
}

static int test_cypher_explain_and_triangles(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    ASSERT(kg, "kg create");
    uint64_t a = kg_add_entity(kg, "A", "Person", NULL, 0);
    uint64_t b = kg_add_entity(kg, "B", "Person", NULL, 0);
    kg_add_relation(kg, a, "knows", b, 1.0f);

    GV_CypherEngine *eng = cypher_create(kg);
    ASSERT(eng, "cypher create");

    GV_CypherResult res;
    memset(&res, 0, sizeof(res));
    ASSERT(cypher_execute(eng,
        "EXPLAIN MATCH (n:Person)-[:knows]->(m) RETURN n.name", &res) == 0,
        "EXPLAIN executes");
    ASSERT(res.row_count == 1 && res.column_count == 1, "explain one row");
    ASSERT(strstr(res.column_values[0], "MATCH") != NULL &&
           strstr(res.column_values[0], "type_index") != NULL,
           "plan mentions index lookup");
    cypher_free_result(&res);
    cypher_destroy(eng);
    kg_destroy(kg);

    /* Parallel triangle count must agree with serial semantics: a triangle
     * K3 has 1 triangle, each node participating once. */
    GV_GraphDB *g = graph_create(NULL);
    uint64_t v[3];
    for (int i = 0; i < 3; i++) v[i] = graph_add_node(g, "N");
    graph_add_edge(g, v[0], v[1], "E", 1.0f);
    graph_add_edge(g, v[1], v[2], "E", 1.0f);
    graph_add_edge(g, v[0], v[2], "E", 1.0f);
    /* undirected adjacency treats each directed edge both ways */
    graph_add_edge(g, v[1], v[0], "E", 1.0f);
    graph_add_edge(g, v[2], v[1], "E", 1.0f);
    graph_add_edge(g, v[2], v[0], "E", 1.0f);
    GV_GraphNodeScores scores;
    uint64_t total = 0;
    ASSERT(graph_triangle_count(g, &scores, &total) == 0, "triangle run");
    ASSERT(total == 1, "K3 has exactly one triangle");
    graph_node_scores_free(&scores);
    graph_destroy(g);
    return 0;
}

/* Multi-level Louvain / Leiden must recover the planted partition of a ring
 * of cliques joined by single bridge edges: each clique is far denser than the
 * bridges, so the exact communities are the cliques themselves. Also pins the
 * Leiden connectivity guarantee: every output community must induce a
 * connected subgraph (plain single-level Louvain has no such guarantee). */
/* ordinal of node `id` among entries of community `comm` (for visited flags) */
static size_t member_ordinal(const GV_GraphNodeLabels *lb, int64_t comm,
                             uint64_t id) {
    size_t o = 0;
    for (size_t i = 0; i < lb->count; i++) {
        if (lb->labels[i] != comm) continue;
        if (lb->node_ids[i] == id) return o;
        o++;
    }
    return o;
}

static int64_t label_of(const GV_GraphNodeLabels *l, uint64_t id) {
    for (size_t i = 0; i < l->count; i++)
        if (l->node_ids[i] == id) return l->labels[i];
    return -999;
}

static int test_multilevel_louvain_leiden(void) {
    enum { CLIQUES = 4, CSZ = 5 };
    GV_GraphDB *g = graph_create(NULL);
    ASSERT(g, "graph create");
    uint64_t ids[CLIQUES][CSZ];
    for (size_t c = 0; c < CLIQUES; c++)
        for (size_t i = 0; i < CSZ; i++) ids[c][i] = graph_add_node(g, "N");

    /* dense undirected cliques (both directions, like real usage) */
    for (size_t c = 0; c < CLIQUES; c++)
        for (size_t i = 0; i < CSZ; i++)
            for (size_t j = 0; j < CSZ; j++)
                if (i != j)
                    ASSERT(graph_add_edge(g, ids[c][i], ids[c][j], "E", 1.0f) != 0,
                           "clique edge");    /* ring bridges: last node of clique c -> first node of clique c+1 */
    for (size_t c = 0; c < CLIQUES; c++) {
        uint64_t u = ids[c][CSZ - 1];
        uint64_t v = ids[(c + 1) % CLIQUES][0];
        ASSERT(graph_add_edge(g, u, v, "B", 1.0f) != 0 &&
               graph_add_edge(g, v, u, "B", 1.0f) != 0,
               "bridge edge");
    }

    GV_GraphNodeLabels lb;
    ASSERT(graph_leiden(g, 10, &lb) == 0 && lb.count == CLIQUES * CSZ,
           "leiden runs");
    ASSERT(lb.num_labels == CLIQUES, "leiden finds exactly the 4 planted cliques");
    for (size_t c = 0; c < CLIQUES; c++) {
        int64_t lc = label_of(&lb, ids[c][0]);
        for (size_t i = 1; i < CSZ; i++)
            ASSERT(label_of(&lb, ids[c][i]) == lc, "clique stays one community");
        ASSERT(lc != label_of(&lb, ids[(c + 1) % CLIQUES][0]),
               "adjacent cliques stay separate");
    }
    /* connectivity guarantee: BFS restricted to each community reaches all
     * of its members */
    for (int64_t comm = 0; comm < (int64_t)lb.num_labels; comm++) {
        size_t members = 0;
        uint64_t start = 0;
        for (size_t i = 0; i < lb.count; i++) {
            if (lb.labels[i] != comm) continue;
            if (members == 0) start = lb.node_ids[i];
            members++;
        }
        ASSERT(members > 0, "community non-empty");

        /* BFS over graph edges, visiting only nodes labeled `comm`; visited
         * flags are keyed by the member's ordinal among lb entries. */
        uint8_t visit[CLIQUES * CSZ];
        memset(visit, 0, sizeof(visit));
        uint64_t queue[CLIQUES * CSZ];
        size_t qh = 0, qt = 0, reached = 1;
        queue[qt++] = start;
        visit[member_ordinal(&lb, comm, start)] = 1;
        while (qh < qt) {
            const GV_GraphNode *xn = graph_get_node(g, queue[qh++]);
            ASSERT(xn, "bfs node exists");
            for (size_t e = 0; e < xn->out_count + xn->in_count; e++) {
                uint64_t nb = e < xn->out_count
                                  ? xn->out_edges[e].neighbor_id
                                  : xn->in_edges[e - xn->out_count].neighbor_id;
                if (label_of(&lb, nb) != comm) continue;
                size_t o = member_ordinal(&lb, comm, nb);
                if (!visit[o]) {
                    visit[o] = 1;
                    queue[qt++] = nb;
                    reached++;
                }
            }
        }
        ASSERT(reached == members, "community is internally connected");
    }
    graph_node_labels_free(&lb);

    /* multi-level louvain on the same fixture: same exact partition */
    ASSERT(graph_louvain(g, 10, &lb) == 0 && lb.count == CLIQUES * CSZ,
           "multilevel louvain runs");
    ASSERT(lb.num_labels == CLIQUES, "louvain also finds the 4 cliques");
    for (size_t c = 0; c < CLIQUES; c++) {
        int64_t lc = label_of(&lb, ids[c][0]);
        for (size_t i = 1; i < CSZ; i++)
            ASSERT(label_of(&lb, ids[c][i]) == lc, "louvain clique intact");
    }
    graph_node_labels_free(&lb);

    graph_destroy(g);
    return 0;
}

/* Incremental CSR: after edge-only mutations the cached matrix must be
 * patched in place (same object identity) and stay bit-identical to a full
 * rebuild; node-adding mutations must invalidate instead of corrupting. */
static int test_incremental_csr(void) {
    GV_GraphDB *g = graph_create(NULL);
    ASSERT(g, "create");
    uint64_t v[12];
    for (int i = 0; i < 12; i++) v[i] = graph_add_node(g, "N");
    for (int i = 0; i < 11; i++)
        ASSERT(graph_add_edge(g, v[i], v[i + 1], "E", 1.0f) != 0, "path edge");

    GV_GAContext *ctx = gv_ga_build(g);
    ASSERT(ctx, "ctx build");
    GV_CSR *base = gv_csr_build_cached(g, ctx, GV_CSR_UNDIRECTED, 1);
    ASSERT(base, "initial cached build");
    gv_ga_free(ctx);   /* contexts pin the snapshot: release before mutating */

    /* edge-only churn: journal covers these -> in-place patch */
    for (int i = 0; i < 5; i++)
        ASSERT(graph_add_edge(g, v[i], v[(i + 3) % 12], "E", 2.0f) != 0,
               "chord add");
    ctx = gv_ga_build(g);   /* same node set -> dense indices unchanged */
    ASSERT(ctx, "ctx rebuild");
    GV_CSR *patched = gv_csr_build_cached(g, ctx, GV_CSR_UNDIRECTED, 1);
    ASSERT(patched == base, "edge-only mutation reuses the cached matrix");

    GV_CSR *ref = gv_csr_build(g, ctx, GV_CSR_UNDIRECTED, 1);
    ASSERT(ref && ref->n == patched->n && ref->nnz == patched->nnz,
           "patched shape matches rebuild");
    int identical = ref != NULL;
    for (size_t i = 0; i <= patched->n && identical; i++)
        if (patched->row_ptr[i] != ref->row_ptr[i]) identical = 0;
    for (size_t k = 0; k < patched->nnz && identical; k++)
        if (patched->col_idx[k] != ref->col_idx[k]) identical = 0;
    for (size_t k = 0; k < patched->nnz && identical; k++)
        if (patched->val[k] != ref->val[k]) identical = 0;
    ASSERT(identical, "patched CSR is bit-identical to full rebuild");
    gv_csr_free(ref);

    /* removals also patch */
    gv_ga_free(ctx);
    uint64_t victim = 0;
    {
        /* remove one chord via its id */
        const GV_GraphNode *n = graph_get_node(g, v[0]);
        ASSERT(n && n->out_count > 0, "node 0 has out edges");
        victim = n->out_edges[n->out_count - 1].edge_id;
    }
    ASSERT(graph_remove_edge(g, victim) == 0, "remove chord");
    ctx = gv_ga_build(g);
    ASSERT(ctx, "ctx rebuild 2");
    patched = gv_csr_build_cached(g, ctx, GV_CSR_UNDIRECTED, 1);
    ref = gv_csr_build(g, ctx, GV_CSR_UNDIRECTED, 1);
    ASSERT(ref && ref->nnz == patched->nnz, "post-remove shape matches");
    identical = ref != NULL;
    for (size_t i = 0; i <= patched->n && identical; i++)
        if (patched->row_ptr[i] != ref->row_ptr[i]) identical = 0;
    for (size_t k = 0; k < patched->nnz && identical; k++)
        if (patched->col_idx[k] != ref->col_idx[k]) identical = 0;
    ASSERT(identical, "post-remove CSR matches rebuild");
    gv_csr_free(ref);
    gv_ga_free(ctx);   /* release before the structural mutation below */

    /* structural change: node count moves -> cache must not go stale */
    uint64_t extra = graph_add_node(g, "N");
    ASSERT(extra != 0, "extra node");
    ctx = gv_ga_build(g);
    ASSERT(ctx && gv_ga_count(ctx) == graph_node_count(g), "fresh ctx");
    patched = gv_csr_build_cached(g, ctx, GV_CSR_UNDIRECTED, 1);
    ASSERT(patched->n == graph_node_count(g), "structural change invalidates");

    gv_ga_free(ctx);
    graph_destroy(g);
    return 0;
}

int main(void) {
    if (test_write_txn_commit_and_abort() != 0) return 1;
    printf("ok: write txn commit/abort\n");
    if (test_write_txn_wal_durability() != 0) return 1;
    printf("ok: write txn WAL durability\n");
    if (test_kg_expand_and_prop_remove() != 0) return 1;
    printf("ok: KG expand_context + prop remove\n");
    if (test_approx_betweenness() != 0) return 1;
    printf("ok: approximate betweenness\n");
    if (test_cypher_explain_and_triangles() != 0) return 1;
    printf("ok: Cypher EXPLAIN + parallel triangles\n");
    if (test_multilevel_louvain_leiden() != 0) return 1;
    printf("ok: multi-level Louvain + Leiden\n");
    if (test_incremental_csr() != 0) return 1;
    printf("ok: incremental CSR patching\n");
    printf("All gap-closure tests PASSED\n");
    return 0;
}
