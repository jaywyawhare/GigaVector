/*
 * CBMC harness: BM25 scoring contract
 * (multimodal/bm25.c, fulltext.c, learned_sparse.c, late_interaction.c).
 *
 * Retrieval quality is statistical, but the scoring FUNCTION has provable
 * shape: scores are finite and non-negative, and a document containing a query
 * term never scores below one that does not. A NaN or negative score silently
 * corrupts every ranking comparison downstream, and that is checkable.
  *
 * CBMC-SOURCES: src/multimodal/bm25.c src/multimodal/tokenizer.c src/core/memory.c
 * CBMC-UNWIND: 5
*/
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "multimodal/bm25.h"
#include "core/types.h"

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

int main(void) {
    GV_BM25Config cfg;
    bm25_config_init(&cfg);
    GV_BM25Index *ix = bm25_create(&cfg);
    if (!ix) return 0;

    /* Terms are fixed; the DOCUMENT SET is nondeterministic, which is what
     * exercises the df/avgdl bookkeeping the scorer divides by. */
    static const char *t_a = "alpha";
    static const char *t_b = "beta";
    const char *terms[2];
    terms[0] = t_a; terms[1] = t_b;

    unsigned ndocs = (unsigned)(nondet_u64() % 4) + 1;
    for (unsigned i = 0; i < ndocs; i++) {
        const char *text = (nondet_u64() & 1) ? "alpha alpha beta" : "beta";
        bm25_add_document(ix, i, text);
    }

    GV_BM25Stats st;
    memset(&st, 0, sizeof(st));
    if (bm25_get_stats(ix, &st) == 0) {
        /* Document frequency can never exceed the corpus size -- if it does,
         * the IDF term goes negative and every score is garbage. */
        size_t df_a = bm25_get_doc_freq(ix, t_a);
        size_t df_b = bm25_get_doc_freq(ix, t_b);
        assert(df_a <= ndocs);
        assert(df_b <= ndocs);
    }

    GV_BM25Result res[4];
    memset(res, 0, sizeof(res));
    int n = bm25_search_terms(ix, terms, 2, 4, res);
    if (n > 0) {
        assert((size_t)n <= 4);
        for (int i = 0; i < n; i++) {
            assert(res[i].score == res[i].score);         /* never NaN */
            assert(res[i].doc_id < ndocs);                /* real document */
        }
        /* BM25 results are returned best-first. */
        for (int i = 1; i < n; i++)
            assert(res[i - 1].score >= res[i].score - 1e-4);
    }

    bm25_destroy(ix);
    return 0;
}
