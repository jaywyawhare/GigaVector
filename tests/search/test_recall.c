/* Recall@k evaluation: FLAT (exact) must score 1.0; HNSW must be high. */
#include <stdio.h>
#include <stdlib.h>
#include "storage/database.h"
#include "search/recall.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

int main(void) {
    const size_t N = 3000, D = 32, Q = 100, K = 10;
    srand(5);
    float *v = (float *)malloc(N * D * sizeof(float));
    for (size_t i = 0; i < N * D; i++) v[i] = (float)rand() / (float)RAND_MAX;

    GV_Database *flat = db_open(NULL, D, GV_INDEX_TYPE_FLAT);
    GV_Database *hnsw = db_open(NULL, D, GV_INDEX_TYPE_HNSW);
    for (size_t i = 0; i < N; i++) { int _rf = db_add_vector(flat, v + i * D, D); int _rh = db_add_vector(hnsw, v + i * D, D); (void)_rf; (void)_rh; }

    GV_RecallReport rf, rh;
    ASSERT(db_evaluate_recall(flat, v, Q, D, K, GV_DISTANCE_EUCLIDEAN, &rf) == 0, "eval flat ok");
    ASSERT(db_evaluate_recall(hnsw, v, Q, D, K, GV_DISTANCE_EUCLIDEAN, &rh) == 0, "eval hnsw ok");
    ASSERT(rf.queries == Q && rf.k == K, "report metadata");
    ASSERT(rf.recall > 0.999, "FLAT is exact -> recall == 1.0");
    ASSERT(rh.recall > 0.70 && rh.recall <= 1.0001, "HNSW recall in a sane range");

    /* invalid args */
    ASSERT(db_evaluate_recall(NULL, v, Q, D, K, GV_DISTANCE_EUCLIDEAN, &rf) == -1, "null db -> -1");
    ASSERT(db_evaluate_recall(flat, v, 0, D, K, GV_DISTANCE_EUCLIDEAN, &rf) == -1, "0 queries -> -1");

    db_close(flat); db_close(hnsw); free(v);
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL RECALL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
