/**
 * graph_algos_util.c — shared helpers for the graph-algorithms layer:
 * a dense node id<->index context and the result-container free functions.
 */
#include "features/graph_algos.h"
#include "core/memory.h"

#include <string.h>

struct GV_GAContext {
    uint64_t *ids;      /* N node ids, ids[i] at dense index i */
    size_t    N;
    /* open-addressing id -> index map */
    uint64_t *key;
    size_t   *val;
    uint8_t  *occ;
    size_t    cap;
};

static size_t ga_hash(uint64_t id, size_t cap) {
    return (size_t)((id * 11400714819323198485ULL) % cap);
}

GV_GAContext *gv_ga_build(const GV_GraphDB *g) {
    if (!g) return NULL;
    size_t N = graph_node_count(g);
    GV_GAContext *ctx = (GV_GAContext *)gv_calloc(1, sizeof(GV_GAContext));
    if (!ctx) return NULL;
    ctx->N = N;
    if (N == 0) return ctx;  /* valid empty context */

    ctx->ids = (uint64_t *)gv_alloc(N * sizeof(uint64_t));
    if (!ctx->ids) { gv_ga_free(ctx); return NULL; }
    int wrote = graph_get_all_node_ids(g, ctx->ids, N);
    if (wrote < 0 || (size_t)wrote != N) { gv_ga_free(ctx); return NULL; }

    ctx->cap = N * 2 + 1;
    ctx->key = (uint64_t *)gv_alloc(ctx->cap * sizeof(uint64_t));
    ctx->val = (size_t *)gv_alloc(ctx->cap * sizeof(size_t));
    ctx->occ = (uint8_t *)gv_calloc(ctx->cap, sizeof(uint8_t));
    if (!ctx->key || !ctx->val || !ctx->occ) { gv_ga_free(ctx); return NULL; }

    for (size_t i = 0; i < N; i++) {
        size_t h = ga_hash(ctx->ids[i], ctx->cap);
        for (size_t j = 0; j < ctx->cap; j++) {
            size_t p = (h + j) % ctx->cap;
            if (!ctx->occ[p]) {
                ctx->occ[p] = 1;
                ctx->key[p] = ctx->ids[i];
                ctx->val[p] = i;
                break;
            }
        }
    }
    return ctx;
}

size_t gv_ga_count(const GV_GAContext *ctx) { return ctx ? ctx->N : 0; }

uint64_t gv_ga_id(const GV_GAContext *ctx, size_t idx) {
    if (!ctx || idx >= ctx->N) return 0;
    return ctx->ids[idx];
}

size_t gv_ga_index(const GV_GAContext *ctx, uint64_t node_id) {
    if (!ctx || ctx->cap == 0) return (size_t)-1;
    size_t h = ga_hash(node_id, ctx->cap);
    for (size_t j = 0; j < ctx->cap; j++) {
        size_t p = (h + j) % ctx->cap;
        if (!ctx->occ[p]) return (size_t)-1;
        if (ctx->key[p] == node_id) return ctx->val[p];
    }
    return (size_t)-1;
}

void gv_ga_free(GV_GAContext *ctx) {
    if (!ctx) return;
    gv_free(ctx->ids);
    gv_free(ctx->key);
    gv_free(ctx->val);
    gv_free(ctx->occ);
    gv_free(ctx);
}

/* ── result containers ─────────────────────────────────────────────────────*/

void graph_node_scores_free(GV_GraphNodeScores *s) {
    if (!s) return;
    gv_free(s->node_ids);
    gv_free(s->scores);
    s->node_ids = NULL; s->scores = NULL; s->count = 0;
}

void graph_node_labels_free(GV_GraphNodeLabels *l) {
    if (!l) return;
    gv_free(l->node_ids);
    gv_free(l->labels);
    l->node_ids = NULL; l->labels = NULL; l->count = 0; l->num_labels = 0;
}

void graph_embeddings_free(GV_GraphEmbeddings *e) {
    if (!e) return;
    gv_free(e->node_ids);
    gv_free(e->vectors);
    e->node_ids = NULL; e->vectors = NULL; e->count = 0; e->dim = 0;
}

void graph_edge_set_free(GV_GraphEdgeSet *s) {
    if (!s) return;
    gv_free(s->edge_ids);
    s->edge_ids = NULL; s->count = 0; s->total_weight = 0.0;
}
