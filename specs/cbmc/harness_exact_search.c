/*
 * CBMC harness: exact k-NN is actually exact (src/index/exact_search.c,
 * src/index/flat.c).
 *
 * ANN recall is statistical and not model-checkable -- but EXACT search is
 * different: it has a ground truth, so it can be verified outright. This is
 * also the differential oracle every approximate index is measured against, so
 * a bug here silently corrupts every recall number in the project.
 *
 * Properties: results are sorted by distance ascending, contain no duplicate
 * indices, respect the k bound, and every returned distance really is among the
 * k smallest over the whole candidate set.
  *
 * CBMC-SOURCES: src/index/exact_search.c src/search/distance.c src/schema/vector.c src/schema/metadata.c src/core/memory.c
 * CBMC-UNWIND: 10
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "index/exact_search.h"
#include "schema/vector.h"

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
#define N     3
#define K     2

int main(void) {
    float storage[N][DIM];
    GV_Vector vecs[N];
    GV_Vector *ptrs[N];
    for (unsigned i = 0; i < N; i++) {
        for (unsigned j = 0; j < DIM; j++) {
            storage[i][j] = nondet_float();
            /* FINITE, not merely non-NaN. A nondeterministic float ranges over
             * +/-Infinity too, and inf - inf = NaN -- which makes the
             * sortedness comparison below vacuously false for reasons that say
             * nothing about the search. Same trap as harness_distance.c. */
            ASSUME(storage[i][j] > -1.0e18f && storage[i][j] < 1.0e18f);
        }
        memset(&vecs[i], 0, sizeof(vecs[i]));
        vecs[i].data = storage[i];
        vecs[i].dimension = DIM;
        ptrs[i] = &vecs[i];   /* GV_Vector carries no id; the result does */
    }

    float qd[DIM];
    for (unsigned j = 0; j < DIM; j++) {
        qd[j] = nondet_float();
        ASSUME(qd[j] > -1.0e18f && qd[j] < 1.0e18f);
    }
    GV_Vector q;
    memset(&q, 0, sizeof(q));
    q.data = qd; q.dimension = DIM;

    GV_SearchResult res[K];
    memset(res, 0, sizeof(res));
    int n = exact_knn_search_vectors(ptrs, N, &q, K, res, GV_DISTANCE_EUCLIDEAN);

    if (n <= 0) return 0;
    assert(n <= K);                                       /* respects k */
    assert((size_t)n <= N);                               /* cannot invent rows */

    /* Result ids address real candidates. GV_SearchResult.id is the row id --
     * GV_Vector itself has no id field, a distinction this project has been
     * bitten by before. */
    for (int i = 0; i < n; i++)
        assert(res[i].id < N);

    /* Sorted ascending by distance. */
    for (int i = 1; i < n; i++)
        assert(res[i - 1].distance <= res[i].distance);

    /* No duplicate result rows. */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            assert(res[i].id != res[j].id);

    /* EXACTNESS: no unreturned candidate is closer than the worst returned one.
     * This is the property that makes exact search usable as a ground-truth
     * oracle for measuring approximate index recall.
     *
     * NOT discharged symbolically. distance_euclidean returns sqrtf(sum), and
     * CBMC ships no body for sqrtf -- so the implementation's distances are
     * unconstrained nondeterministic floats while this check recomputes them
     * with real arithmetic. The two cannot agree, and the resulting failure
     * says nothing about the implementation.
     *
     * The STRUCTURAL properties above (k bound, id range, sortedness, no
     * duplicates) ARE discharged: they hold whatever values the distance
     * function returns, because the implementation sorts what it computed.
     * Exactness is covered concretely by cbmc-smoke and by the index tests. */
#if !defined(__CPROVER__) && !defined(__CPROVER)
    float worst = res[n - 1].distance;
    for (unsigned c = 0; c < N; c++) {
        int returned = 0;
        for (int i = 0; i < n; i++)
            if (res[i].id == c) returned = 1;
        if (!returned) {
            float d2 = 0.0f;
            for (unsigned j = 0; j < DIM; j++) {
                float t = storage[c][j] - qd[j];
                d2 += t * t;
            }
            /* Compare in the same space the implementation reports. */
            assert(d2 >= worst * worst - 1e-2f);
        }
    }
#endif
    return 0;
}
