/**
 * @file graph_linkpred.c
 * @brief Topological similarity & link-prediction measures for the GigaVector
 *        property graph (graph_algos.h).
 *
 * All measures operate on UNDIRECTED neighborhoods: N(x) is the deduplicated set
 * of nodes adjacent to x via out_edges OR in_edges, excluding x itself. Each
 * neighborhood is materialized as a sorted, deduplicated array of node IDs so
 * that intersections and unions reduce to linear two-pointer merges.
 *
 * Read-only w.r.t. the graph; no leaks (ASAN/valgrind-clean); deterministic.
 */

#include "features/graph_algos.h"
#include "core/memory.h"

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* ── Neighborhood construction ──────────────────────────────────────────────*/

/** A sorted, deduplicated undirected neighborhood N(x). */
typedef struct {
    uint64_t *ids;   /**< Ascending, unique, excludes x itself (may be NULL if count==0). */
    size_t    count;
} GV_Nbhd;

static int cmp_u64(const void *pa, const void *pb) {
    uint64_t a = *(const uint64_t *)pa;
    uint64_t b = *(const uint64_t *)pb;
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

static void nbhd_free(GV_Nbhd *n) {
    if (!n) return;
    gv_free(n->ids);
    n->ids = NULL;
    n->count = 0;
}

/**
 * Build the undirected neighborhood of node `self` into `out`.
 * Returns 0 on success (including an empty neighborhood), -1 on allocation error.
 * On error `out` is left zeroed.
 */
static int nbhd_build(const GV_GraphNode *node, uint64_t self, GV_Nbhd *out) {
    out->ids = NULL;
    out->count = 0;
    if (!node) return 0;

    size_t raw = node->out_count + node->in_count;
    if (raw == 0) return 0;

    uint64_t *buf = gv_alloc(raw * sizeof(*buf));
    if (!buf) return -1;

    size_t n = 0;
    for (size_t i = 0; i < node->out_count; i++) {
        uint64_t nid = node->out_edges[i].neighbor_id;
        if (nid != self) buf[n++] = nid;
    }
    for (size_t i = 0; i < node->in_count; i++) {
        uint64_t nid = node->in_edges[i].neighbor_id;
        if (nid != self) buf[n++] = nid;
    }

    if (n == 0) {
        gv_free(buf);
        return 0;
    }

    qsort(buf, n, sizeof(*buf), cmp_u64);

    /* Deduplicate in place. */
    size_t uniq = 1;
    for (size_t i = 1; i < n; i++) {
        if (buf[i] != buf[uniq - 1]) buf[uniq++] = buf[i];
    }

    out->ids = buf;
    out->count = uniq;
    return 0;
}

/** |N(a) ∩ N(b)| over two sorted, deduplicated arrays. */
static size_t nbhd_intersection_size(const GV_Nbhd *a, const GV_Nbhd *b) {
    size_t i = 0, j = 0, c = 0;
    while (i < a->count && j < b->count) {
        if (a->ids[i] < b->ids[j]) {
            i++;
        } else if (a->ids[i] > b->ids[j]) {
            j++;
        } else {
            c++; i++; j++;
        }
    }
    return c;
}

/** |N(a) ∪ N(b)| = |N(a)| + |N(b)| - |N(a) ∩ N(b)|. */
static size_t nbhd_union_size(const GV_Nbhd *a, const GV_Nbhd *b, size_t inter) {
    return a->count + b->count - inter;
}

/** Undirected degree deg(c) = |N(c)|. */
static size_t undirected_degree(const GV_GraphDB *g, uint64_t id) {
    const GV_GraphNode *node = graph_get_node(g, id);
    if (!node) return 0;
    GV_Nbhd n;
    if (nbhd_build(node, id, &n) != 0) return 0; /* alloc failure → treat as 0 */
    size_t deg = n.count;
    nbhd_free(&n);
    return deg;
}

/* ── Pairwise measures ──────────────────────────────────────────────────────*/

/**
 * Common scaffold: fetch both nodes, build both neighborhoods.
 * Returns 0 on success (na/nb populated), -1 if a or b is absent (caller returns
 * -1.0), -2 on allocation failure (caller returns 0.0 — read-only best effort).
 */
static int prep_pair(const GV_GraphDB *g, uint64_t a, uint64_t b,
                     GV_Nbhd *na, GV_Nbhd *nb) {
    na->ids = NULL; na->count = 0;
    nb->ids = NULL; nb->count = 0;

    const GV_GraphNode *pa = graph_get_node(g, a);
    const GV_GraphNode *pb = graph_get_node(g, b);
    if (!pa || !pb) return -1;

    if (nbhd_build(pa, a, na) != 0) return -2;
    if (nbhd_build(pb, b, nb) != 0) { nbhd_free(na); return -2; }
    return 0;
}

double graph_common_neighbors(const GV_GraphDB *g, uint64_t a, uint64_t b) {
    GV_Nbhd na, nb;
    int rc = prep_pair(g, a, b, &na, &nb);
    if (rc == -1) return -1.0;
    if (rc == -2) return 0.0;
    double v = (double)nbhd_intersection_size(&na, &nb);
    nbhd_free(&na); nbhd_free(&nb);
    return v;
}

double graph_total_neighbors(const GV_GraphDB *g, uint64_t a, uint64_t b) {
    GV_Nbhd na, nb;
    int rc = prep_pair(g, a, b, &na, &nb);
    if (rc == -1) return -1.0;
    if (rc == -2) return 0.0;
    size_t inter = nbhd_intersection_size(&na, &nb);
    double v = (double)nbhd_union_size(&na, &nb, inter);
    nbhd_free(&na); nbhd_free(&nb);
    return v;
}

double graph_preferential_attachment(const GV_GraphDB *g, uint64_t a, uint64_t b) {
    GV_Nbhd na, nb;
    int rc = prep_pair(g, a, b, &na, &nb);
    if (rc == -1) return -1.0;
    if (rc == -2) return 0.0;
    double v = (double)na.count * (double)nb.count;
    nbhd_free(&na); nbhd_free(&nb);
    return v;
}

double graph_jaccard_similarity(const GV_GraphDB *g, uint64_t a, uint64_t b) {
    GV_Nbhd na, nb;
    int rc = prep_pair(g, a, b, &na, &nb);
    if (rc == -1) return -1.0;
    if (rc == -2) return 0.0;
    size_t inter = nbhd_intersection_size(&na, &nb);
    size_t uni = nbhd_union_size(&na, &nb, inter);
    double v = (uni == 0) ? 0.0 : (double)inter / (double)uni;
    nbhd_free(&na); nbhd_free(&nb);
    return v;
}

double graph_cosine_neighborhood(const GV_GraphDB *g, uint64_t a, uint64_t b) {
    GV_Nbhd na, nb;
    int rc = prep_pair(g, a, b, &na, &nb);
    if (rc == -1) return -1.0;
    if (rc == -2) return 0.0;
    double v = 0.0;
    if (na.count != 0 && nb.count != 0) {
        size_t inter = nbhd_intersection_size(&na, &nb);
        v = (double)inter / sqrt((double)na.count * (double)nb.count);
    }
    nbhd_free(&na); nbhd_free(&nb);
    return v;
}

/**
 * Walk the intersection of N(a) and N(b), invoking a weighting term for each
 * common neighbor c. `weight_for_deg` returns the contribution given deg(c);
 * a return of 0.0 (used for skipped nodes) simply adds nothing.
 */
static double intersection_degree_sum(const GV_GraphDB *g,
                                      const GV_Nbhd *a, const GV_Nbhd *b,
                                      double (*weight_for_deg)(size_t deg)) {
    size_t i = 0, j = 0;
    double acc = 0.0;
    while (i < a->count && j < b->count) {
        if (a->ids[i] < b->ids[j]) {
            i++;
        } else if (a->ids[i] > b->ids[j]) {
            j++;
        } else {
            size_t deg = undirected_degree(g, a->ids[i]);
            acc += weight_for_deg(deg);
            i++; j++;
        }
    }
    return acc;
}

static double adamic_adar_term(size_t deg) {
    if (deg <= 1) return 0.0;      /* log(deg) <= 0 → skip */
    double l = log((double)deg);
    if (l <= 0.0) return 0.0;
    return 1.0 / l;
}

static double resource_alloc_term(size_t deg) {
    if (deg == 0) return 0.0;      /* skip deg==0 */
    return 1.0 / (double)deg;
}

double graph_adamic_adar(const GV_GraphDB *g, uint64_t a, uint64_t b) {
    GV_Nbhd na, nb;
    int rc = prep_pair(g, a, b, &na, &nb);
    if (rc == -1) return -1.0;
    if (rc == -2) return 0.0;
    double v = intersection_degree_sum(g, &na, &nb, adamic_adar_term);
    nbhd_free(&na); nbhd_free(&nb);
    return v;
}

double graph_resource_allocation(const GV_GraphDB *g, uint64_t a, uint64_t b) {
    GV_Nbhd na, nb;
    int rc = prep_pair(g, a, b, &na, &nb);
    if (rc == -1) return -1.0;
    if (rc == -2) return 0.0;
    double v = intersection_degree_sum(g, &na, &nb, resource_alloc_term);
    nbhd_free(&na); nbhd_free(&nb);
    return v;
}

/* ── Top-k similar ──────────────────────────────────────────────────────────*/

/** Dispatch a measure by enum over an already-existing pair of node IDs. */
static double measure_pair(const GV_GraphDB *g, uint64_t a, uint64_t b,
                           GV_GraphSimMeasure measure) {
    switch (measure) {
        case GV_GSIM_JACCARD:                return graph_jaccard_similarity(g, a, b);
        case GV_GSIM_COSINE:                 return graph_cosine_neighborhood(g, a, b);
        case GV_GSIM_COMMON_NEIGHBORS:       return graph_common_neighbors(g, a, b);
        case GV_GSIM_ADAMIC_ADAR:            return graph_adamic_adar(g, a, b);
        case GV_GSIM_RESOURCE_ALLOCATION:    return graph_resource_allocation(g, a, b);
        case GV_GSIM_PREFERENTIAL_ATTACHMENT:return graph_preferential_attachment(g, a, b);
        default:                             return graph_jaccard_similarity(g, a, b);
    }
}

typedef struct {
    uint64_t node_id;
    double   score;
} GV_ScoredNode;

/* Sort by score descending; ties broken by smaller node_id (ascending). */
static int cmp_scored_desc(const void *pa, const void *pb) {
    const GV_ScoredNode *a = pa;
    const GV_ScoredNode *b = pb;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    if (a->node_id < b->node_id) return -1;
    if (a->node_id > b->node_id) return 1;
    return 0;
}

/**
 * Build the 2-hop candidate set: all distinct nodes reachable via one of `node`'s
 * neighbors, excluding `node` itself. Returns a sorted, deduplicated array in
 * `*out_ids`/`*out_count`. Returns 0 on success (possibly empty), -1 on alloc
 * failure.
 */
static int build_candidates(const GV_GraphDB *g, uint64_t node,
                            const GV_Nbhd *nself,
                            uint64_t **out_ids, size_t *out_count) {
    *out_ids = NULL;
    *out_count = 0;

    /* Upper bound on raw candidate count = sum of neighbor degrees (out+in). */
    size_t cap = 0;
    for (size_t i = 0; i < nself->count; i++) {
        const GV_GraphNode *nb = graph_get_node(g, nself->ids[i]);
        if (nb) cap += nb->out_count + nb->in_count;
    }
    if (cap == 0) return 0;

    uint64_t *buf = gv_alloc(cap * sizeof(*buf));
    if (!buf) return -1;

    size_t n = 0;
    for (size_t i = 0; i < nself->count; i++) {
        const GV_GraphNode *nb = graph_get_node(g, nself->ids[i]);
        if (!nb) continue;
        for (size_t k = 0; k < nb->out_count; k++) {
            uint64_t cid = nb->out_edges[k].neighbor_id;
            if (cid != node) buf[n++] = cid;
        }
        for (size_t k = 0; k < nb->in_count; k++) {
            uint64_t cid = nb->in_edges[k].neighbor_id;
            if (cid != node) buf[n++] = cid;
        }
    }

    if (n == 0) {
        gv_free(buf);
        return 0;
    }

    qsort(buf, n, sizeof(*buf), cmp_u64);
    size_t uniq = 1;
    for (size_t i = 1; i < n; i++) {
        if (buf[i] != buf[uniq - 1]) buf[uniq++] = buf[i];
    }

    *out_ids = buf;
    *out_count = uniq;
    return 0;
}

int graph_topk_similar(const GV_GraphDB *g, uint64_t node, GV_GraphSimMeasure measure,
                       size_t k, GV_GraphNodeScores *out) {
    if (!out) return -1;
    out->node_ids = NULL;
    out->scores = NULL;
    out->count = 0;

    const GV_GraphNode *self = graph_get_node(g, node);
    if (!self) return 0;

    /* Self neighborhood drives 2-hop candidate generation. */
    GV_Nbhd nself;
    if (nbhd_build(self, node, &nself) != 0) return -1;
    if (nself.count == 0) {
        nbhd_free(&nself);
        return 0; /* no neighbors → no candidates */
    }

    uint64_t *cands = NULL;
    size_t ncands = 0;
    int rc = build_candidates(g, node, &nself, &cands, &ncands);
    nbhd_free(&nself);
    if (rc != 0) return -1;
    if (ncands == 0) return 0;

    GV_ScoredNode *scored = gv_alloc(ncands * sizeof(*scored));
    if (!scored) { gv_free(cands); return -1; }

    size_t found = 0;
    for (size_t i = 0; i < ncands; i++) {
        double s = measure_pair(g, node, cands[i], measure);
        /* Candidates all share a neighbor with `node`; -1.0 (absent) shouldn't
         * occur here, but guard anyway. */
        if (s < 0.0) continue;
        scored[found].node_id = cands[i];
        scored[found].score = s;
        found++;
    }
    gv_free(cands);

    if (found == 0) {
        gv_free(scored);
        return 0;
    }

    qsort(scored, found, sizeof(*scored), cmp_scored_desc);

    size_t take = (k < found) ? k : found;
    if (take == 0) {
        gv_free(scored);
        return 0;
    }

    uint64_t *ids = gv_alloc(take * sizeof(*ids));
    double   *scr = gv_alloc(take * sizeof(*scr));
    if (!ids || !scr) {
        gv_free(ids);
        gv_free(scr);
        gv_free(scored);
        return -1;
    }

    for (size_t i = 0; i < take; i++) {
        ids[i] = scored[i].node_id;
        scr[i] = scored[i].score;
    }
    gv_free(scored);

    out->node_ids = ids;
    out->scores = scr;
    out->count = take;
    return 0;
}
