/**
 * @file sparse_dense_fusion.c
 * @brief Learned sparse + dense fusion implementation.
 *
 * Implements four fusion strategies for combining learned-sparse (SPLADE-style)
 * retrieval results with dense vector search results:
 *   - RRF (Reciprocal Rank Fusion)
 *   - Weighted RRF
 *   - Linear (normalised score combination)
 *   - Convex (alpha-weighted combination)
 *
 * Deduplication is performed by vector_id; when a document appears in both
 * result lists the entry with the higher individual source score is kept and
 * receives the fused score.
 */

#include "search/sparse_dense_fusion.h"
#include "core/memory.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

static const GV_SparseDenseFusionConfig DEFAULT_CONFIG = {
    .type          = GV_SPARSE_DENSE_FUSION_RRF,
    .sparse_weight = 0.5f,
    .dense_weight  = 0.5f,
    .rrf_k         = 60
};

void sparse_dense_fusion_config_init(GV_SparseDenseFusionConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ── Internal candidate ────────────────────────────────────────────────── */

typedef struct {
    size_t id;
    float  sparse_score;
    float  dense_score;
    size_t sparse_rank;  /* 1-based, 0 if not in sparse list */
    size_t dense_rank;   /* 1-based, 0 if not in dense list */
    float  fused_score;
} SDFCandidate;

static int compare_candidates_desc(const void *a, const void *b) {
    const SDFCandidate *ca = (const SDFCandidate *)a;
    const SDFCandidate *cb = (const SDFCandidate *)b;
    if (cb->fused_score > ca->fused_score) return  1;
    if (cb->fused_score < ca->fused_score) return -1;
    return 0;
}

/**
 * @brief Find or create a candidate entry by id.
 */
static SDFCandidate *find_or_add(SDFCandidate *cands, size_t *count,
                                  size_t capacity, size_t id) {
    for (size_t i = 0; i < *count; i++) {
        if (cands[i].id == id) return &cands[i];
    }
    if (*count >= capacity) return NULL;

    SDFCandidate *e = &cands[*count];
    memset(e, 0, sizeof(*e));
    e->id = id;
    (*count)++;
    return e;
}

/**
 * @brief Normalise a value to [0, 1] given min/max bounds.
 */
static float normalise(float val, float lo, float hi) {
    if (hi <= lo) return 0.5f;
    return (val - lo) / (hi - lo);
}

/* ── Public: sparse_dense_fuse ─────────────────────────────────────────── */

int sparse_dense_fuse(const GV_SearchResult *sparse_results, size_t sparse_count,
                      const GV_SearchResult *dense_results, size_t dense_count,
                      GV_SearchResult *fused, size_t max_fused,
                      const GV_SparseDenseFusionConfig *config) {
    if (!fused || max_fused == 0) return -1;
    if (sparse_count == 0 && dense_count == 0) return 0;

    GV_SparseDenseFusionConfig cfg = config ? *config : DEFAULT_CONFIG;

    /* Maximum candidates = sparse + dense (before dedup) */
    size_t max_cands = sparse_count + dense_count;
    if (max_cands > max_fused) max_cands = max_fused;

    SDFCandidate *cands = (SDFCandidate *)gv_calloc(max_cands, sizeof(SDFCandidate));
    if (!cands) return -1;
    size_t cand_count = 0;

    /* ── Populate from sparse results ──────────────────────────────────── */
    float sp_min = 1e30f, sp_max = -1e30f;
    for (size_t i = 0; i < sparse_count; i++) {
        if (sparse_results[i].distance < sp_min) sp_min = sparse_results[i].distance;
        if (sparse_results[i].distance > sp_max) sp_max = sparse_results[i].distance;
    }

    for (size_t i = 0; i < sparse_count; i++) {
        SDFCandidate *e = find_or_add(cands, &cand_count, max_cands,
                                       sparse_results[i].id);
        if (!e) break;
        e->sparse_score = sparse_results[i].distance;
        e->sparse_rank  = i + 1;
    }

    /* ── Populate from dense results ───────────────────────────────────── */
    float dn_min = 1e30f, dn_max = -1e30f;
    for (size_t i = 0; i < dense_count; i++) {
        if (dense_results[i].distance < dn_min) dn_min = dense_results[i].distance;
        if (dense_results[i].distance > dn_max) dn_max = dense_results[i].distance;
    }

    for (size_t i = 0; i < dense_count; i++) {
        SDFCandidate *e = find_or_add(cands, &cand_count, max_cands,
                                       dense_results[i].id);
        if (!e) break;
        e->dense_score = dense_results[i].distance;
        e->dense_rank  = i + 1;
    }

    /* ── Compute fused scores ──────────────────────────────────────────── */
    for (size_t i = 0; i < cand_count; i++) {
        SDFCandidate *c = &cands[i];
        float ns = 0.0f, nd = 0.0f;

        if (c->sparse_rank > 0)
            ns = normalise(c->sparse_score, sp_min, sp_max);
        if (c->dense_rank > 0)
            nd = normalise(c->dense_score, dn_min, dn_max);

        double k = (double)cfg.rrf_k;

        switch (cfg.type) {
        case GV_SPARSE_DENSE_FUSION_RRF: {
            double score = 0.0;
            if (c->sparse_rank > 0)
                score += 1.0 / (k + (double)c->sparse_rank);
            if (c->dense_rank > 0)
                score += 1.0 / (k + (double)c->dense_rank);
            c->fused_score = (float)score;
            break;
        }

        case GV_SPARSE_DENSE_FUSION_WEIGHTED_RRF: {
            double sw = (double)cfg.sparse_weight;
            double dw = (double)cfg.dense_weight;
            double score = 0.0;
            if (c->sparse_rank > 0)
                score += sw / (k + (double)c->sparse_rank);
            if (c->dense_rank > 0)
                score += dw / (k + (double)c->dense_rank);
            c->fused_score = (float)score;
            break;
        }

        case GV_SPARSE_DENSE_FUSION_LINEAR: {
            double sw = (double)cfg.sparse_weight;
            double dw = (double)cfg.dense_weight;
            double sum = sw + dw;
            if (sum <= 0.0) sum = 1.0;
            c->fused_score = (float)((sw * ns + dw * nd) / sum);
            break;
        }

        case GV_SPARSE_DENSE_FUSION_CONVEX: {
            double alpha = (double)cfg.dense_weight;
            c->fused_score = (float)(alpha * nd + (1.0 - alpha) * ns);
            break;
        }
        }
    }

    /* ── Sort and copy ─────────────────────────────────────────────────── */
    qsort(cands, cand_count, sizeof(SDFCandidate), compare_candidates_desc);

    size_t out_count = cand_count < max_fused ? cand_count : max_fused;
    for (size_t i = 0; i < out_count; i++) {
        memset(&fused[i], 0, sizeof(GV_SearchResult));
        fused[i].id       = cands[i].id;
        fused[i].distance = cands[i].fused_score;
    }

    gv_free(cands);
    return (int)out_count;
}

/* ── Public: sparse_dense_search (convenience) ─────────────────────────── */

int sparse_dense_search(void *sparse_index, void *dense_index,
                        const char **sparse_terms, const float *sparse_weights,
                        size_t sparse_term_count,
                        const float *dense_query, size_t dimension,
                        size_t k, GV_SearchResult *fused,
                        const GV_SparseDenseFusionConfig *config) {
    (void)sparse_index;
    (void)dense_index;
    (void)sparse_terms;
    (void)sparse_weights;
    (void)sparse_term_count;
    (void)dense_query;
    (void)dimension;
    (void)k;
    (void)fused;
    (void)config;

    /* This is a thin adapter that would call the respective index search
     * functions and then fuse. The actual index API varies, so this
     * provides the integration point.  Returns -1 until wired to concrete
     * index implementations. */
    return -1;
}
