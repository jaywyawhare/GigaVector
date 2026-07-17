/* Standalone HNSW build-time benchmark.
 * Builds an index of N random vectors and reports insert throughput.
 * Deterministic vector data (fixed PRNG seed) so timings are comparable. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "index/hnsw.h"
#include "core/types.h"

static uint64_t xs = 0x12345678abcdefULL;
static inline float frand(void) {
    xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17;
    return (float)((xs >> 11) & 0xFFFFFF) / (float)0xFFFFFF;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t n   = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 20000;
    size_t dim = argc > 2 ? (size_t)strtoull(argv[2], NULL, 10) : 128;

    float *data = malloc(n * dim * sizeof(float));
    for (size_t i = 0; i < n * dim; ++i) data[i] = frand();

    GV_HNSWConfig cfg = {0};
    cfg.M = 16; cfg.efConstruction = 200; cfg.efSearch = 50;
    cfg.distance_type = GV_DISTANCE_EUCLIDEAN;

    void *idx = gv_hnsw_create(dim, &cfg, NULL);
    if (!idx) { fprintf(stderr, "create failed\n"); return 1; }
    gv_hnsw_reserve(idx, n);

    double t0 = now_s();
    for (size_t i = 0; i < n; ++i) {
        if (gv_hnsw_insert_raw(idx, data + i * dim, dim) != 0) {
            fprintf(stderr, "insert %zu failed\n", i); return 1;
        }
    }
    double t1 = now_s();

    double secs = t1 - t0;
    printf("build: n=%zu dim=%zu  %.3f s  %.0f ins/s  %.1f us/ins\n",
           n, dim, secs, (double)n / secs, secs / (double)n * 1e6);

    /* Recall@10 vs brute force over a fixed query subset. */
    size_t nq = n < 200 ? n : 200;
    size_t k = 10;
    size_t hit = 0, tot = 0;
    GV_SearchResult res[16];
    size_t *bf = malloc(n * sizeof(size_t));
    float  *bd = malloc(n * sizeof(float));
    for (size_t q = 0; q < nq; ++q) {
        const float *qv = data + q * dim; /* query = an inserted point */
        for (size_t i = 0; i < n; ++i) {
            float s = 0; const float *v = data + i * dim;
            for (size_t d = 0; d < dim; ++d) { float e = qv[d]-v[d]; s += e*e; }
            bd[i] = s; bf[i] = i;
        }
        /* partial selection of k smallest brute-force ids */
        for (size_t a = 0; a < k; ++a) {
            size_t m = a;
            for (size_t b = a+1; b < n; ++b) if (bd[b] < bd[m]) m = b;
            float td = bd[a]; bd[a] = bd[m]; bd[m] = td;
            size_t ti = bf[a]; bf[a] = bf[m]; bf[m] = ti;
        }
        GV_Vector qq = { .dimension = dim, .data = (float *)qv, .metadata = NULL };
        int got = gv_hnsw_search(idx, &qq, k, res, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
        for (int r = 0; r < got; ++r) {
            for (size_t a = 0; a < k; ++a)
                if (res[r].id == bf[a]) { hit++; break; }
        }
        tot += k;
    }
    printf("recall@%zu: %.4f  (%zu/%zu over %zu queries)\n",
           k, (double)hit/(double)tot, hit, tot, nq);
    free(bf); free(bd);

    gv_hnsw_destroy(idx);
    free(data);
    return 0;
}
