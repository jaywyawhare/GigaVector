/* Parallel HNSW construction + runtime efConstruction knob.
 *
 * Verifies that db_add_vectors_parallel() builds a graph whose recall matches a
 * serial insert build (the batched parallel-search + serial-link strategy must
 * not degrade quality), and that db_set_ef_construction() is honoured. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "storage/database.h"
#include "schema/vector.h"

static int failures = 0;
#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", (msg)); failures++; } \
    else { printf("ok: %s\n", (msg)); } \
} while (0)

/* recall@K of `idx` measured against exact ground truth `gt`, averaged over Q queries. */
static double recall_at_k(GV_Database *idx, GV_Database *gt, const float *v,
                          size_t N, size_t Q, size_t D, size_t K) {
    double acc = 0.0;
    GV_SearchResult a[64], b[64];
    for (size_t q = 0; q < Q; ++q) {
        const float *qv = v + (q * 131 % N) * D;
        int ng = db_search(gt, qv, K, a, GV_DISTANCE_EUCLIDEAN);
        int ni = db_search(idx, qv, K, b, GV_DISTANCE_EUCLIDEAN);
        int hit = 0;
        for (int i = 0; i < ng; ++i)
            for (int j = 0; j < ni; ++j)
                if (a[i].id == b[j].id) { hit++; break; }
        for (int i = 0; i < ng; ++i) if (a[i].vector) vector_destroy((GV_Vector *)a[i].vector);
        for (int i = 0; i < ni; ++i) if (b[i].vector) vector_destroy((GV_Vector *)b[i].vector);
        acc += ng ? (double)hit / ng : 0.0;
    }
    return 100.0 * acc / (double)Q;
}

int main(void) {
    const size_t N = 6000, Q = 200, D = 64, K = 10;
    srand(7);
    float *v = (float *)malloc(N * D * sizeof(float));
    for (size_t i = 0; i < N * D; ++i) v[i] = (float)rand() / (float)RAND_MAX;

    /* exact ground truth */
    GV_Database *gt = db_open(NULL, D, GV_INDEX_TYPE_FLAT);
    ASSERT(gt != NULL, "open flat gt");
    for (size_t i = 0; i < N; ++i) { int _r = db_add_vector(gt, v + i * D, D); (void)_r; }

    /* serial HNSW build */
    GV_Database *s = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    ASSERT(s != NULL, "open serial hnsw");
    for (size_t i = 0; i < N; ++i) { int _r = db_add_vector(s, v + i * D, D); (void)_r; }

    /* parallel HNSW build over the same data */
    GV_Database *p = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    ASSERT(p != NULL, "open parallel hnsw");
    int rc = db_add_vectors_parallel(p, v, N, D, 4);
    ASSERT(rc == 0, "db_add_vectors_parallel rc==0");
    { GV_DBStats st; db_get_stats(p, &st); ASSERT(st.total_inserts == N, "parallel build inserted all vectors"); }

    double sr = recall_at_k(s, gt, v, N, Q, D, K);
    double pr = recall_at_k(p, gt, v, N, Q, D, K);
    printf("recall@%zu: serial %.1f%%  parallel %.1f%%\n", K, sr, pr);

    /* parallel recall must be healthy on its own... */
    ASSERT(pr >= 80.0, "parallel recall >= 80%");
    /* ...and within 3pp of serial - i.e. no meaningful quality loss from parallelism. */
    ASSERT(fabs(sr - pr) <= 3.0, "parallel recall within 3pp of serial");

    /* efConstruction knob is accepted before a build */
    GV_Database *e = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    ASSERT(db_set_ef_construction(e, 32) == 0, "set_ef_construction(32) ok");
    ASSERT(db_add_vectors_parallel(e, v, N, D, 4) == 0, "parallel build with ef=32 ok");
    { GV_DBStats st; db_get_stats(e, &st); ASSERT(st.total_inserts == N, "ef-tuned build inserted all vectors"); }

    db_close(gt); db_close(s); db_close(p); db_close(e);
    free(v);

    if (failures == 0) printf("ALL HNSW-PARALLEL TESTS PASSED\n");
    else printf("%d FAILURE(S)\n", failures);
    return failures ? 1 : 0;
}
