/*
 * CBMC harness: result-ordering contracts across the search pipeline
 * (search/ranking.c, mmr.c, score_threshold.c, phased_ranking.c,
 *  group_search.c, hybrid_search.c, quant_rerank.c).
 *
 * Ranking QUALITY is statistical. Ranking STRUCTURE is not: a reranker must
 * return a subset of its candidates, must not duplicate, must respect k, and
 * must not emit anything below a configured threshold. Those are the errors
 * that show up as "a result appeared twice" or "we returned 11 for k=10",
 * and they are provable.
  *
 * CBMC-SOURCES: src/search/mmr.c src/search/distance.c src/core/scope.c src/core/arena.c src/core/memory.c
 * CBMC-UNWIND: 4
 * CBMC-SKIP: intractable for the same structural reason as harness_distance.c.
 *   Its earlier FAILED verdict was a missing source -- mmr_rerank calls
 *   gv_tls_alloc_or_heap (core/scope.c), and an unlinked function is modelled
 *   as returning an arbitrary pointer, so every downstream result was garbage.
 *   With the closure complete it no longer fails; it simply does not finish,
 *   even at DIM 2 / NCAND 2 / K 1. MMR defaults to COSINE, whose division and
 *   score normalisation put exact IEEE-754 arithmetic on the critical path.
 *   Concrete smoke-test only; the ranking files stay `cbmc-pending`.
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "search/mmr.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
size_t   nondet_size(void);
float    nondet_float(void);
unsigned char nondet_uchar(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
static size_t nondet_size(void) { return (size_t)(nondet_u64() % 8); }
static float nondet_float(void) {
    return (float)((int)(nondet_u64() % 2000) - 1000) / 100.0f;
}
static unsigned char nondet_uchar(void) { return (unsigned char)(nondet_u64() & 0xFF); }
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define DIM   2
#define NCAND 2
#define K     1

int main(void) {
    float qd[DIM];
    for (unsigned j = 0; j < DIM; j++) {
        qd[j] = nondet_float();
        ASSUME(qd[j] > -1.0e18f && qd[j] < 1.0e18f);
    }

    float cand[NCAND * DIM];
    size_t cidx[NCAND];
    float cdist[NCAND];
    for (unsigned i = 0; i < NCAND; i++) {
        for (unsigned j = 0; j < DIM; j++) {
            cand[i * DIM + j] = nondet_float();
            /* FINITE, not merely non-NaN: an unconstrained float ranges over
             * +/-Infinity, and inf arithmetic yields NaN, which makes the
             * score comparisons below vacuously false. */
            ASSUME(cand[i * DIM + j] > -1.0e18f && cand[i * DIM + j] < 1.0e18f);
        }
        cidx[i] = i;                       /* distinct candidate ids */
        cdist[i] = nondet_float();
        ASSUME(cdist[i] == cdist[i] && cdist[i] >= 0.0f && cdist[i] <= 100.0f);
    }

    GV_MMRConfig cfg;
    mmr_config_init(&cfg);

    GV_MMRResult out[K];
    memset(out, 0, sizeof(out));
    int n = mmr_rerank(qd, DIM, cand, cidx, cdist, NCAND, K, &cfg, out);

    if (n <= 0) return 0;

    assert(n <= K);                        /* respects the requested k */

    for (int i = 0; i < n; i++) {
        /* SUBSET: every selected item came from the candidate set. */
        int found = 0;
        for (unsigned c = 0; c < NCAND; c++)
            if (out[i].index == cidx[c]) found = 1;
        assert(found);

        /* Scores are well-formed, not NaN -- a NaN score silently corrupts
         * every downstream comparison. */
        assert(out[i].score == out[i].score);
        assert(out[i].relevance == out[i].relevance);
        assert(out[i].diversity == out[i].diversity);
    }

    /* NO DUPLICATES: MMR selects without replacement. */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            assert(out[i].index != out[j].index);

    /* MMR emits in selection order, which is non-increasing by score. */
    for (int i = 1; i < n; i++)
        assert(out[i - 1].score >= out[i].score - 1e-3f);

    return 0;
}
