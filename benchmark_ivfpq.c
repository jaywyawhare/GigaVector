#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "include/gigavector/gigavector.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void fill_random(float *data, size_t n, size_t dim) {
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc, char **argv) {
    size_t dim = 64;
    size_t train = 2000;
    size_t n = 10000;
    size_t q = 200;
    size_t k = 10;
    size_t nlist = 256;
    size_t m = 8;
    uint8_t nbits = 8;
    size_t nprobe = 16;
    size_t rerank = 32;
    int use_cosine = 0;
    if (argc > 1) n = (size_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) q = (size_t)strtoul(argv[2], NULL, 10);
    if (argc > 3) nlist = (size_t)strtoul(argv[3], NULL, 10);
    if (argc > 4) m = (size_t)strtoul(argv[4], NULL, 10);
    if (argc > 5) nbits = (uint8_t)strtoul(argv[5], NULL, 10);
    if (argc > 6) nprobe = (size_t)strtoul(argv[6], NULL, 10);
    if (argc > 7) rerank = (size_t)strtoul(argv[7], NULL, 10);
    if (argc > 8) use_cosine = atoi(argv[8]);

    srand(42);
    float *data = (float *)malloc(n * dim * sizeof(float));
    float *queries = (float *)malloc(q * dim * sizeof(float));
    if (!data || !queries) return 1;
    fill_random(data, n, dim);
    fill_random(queries, q, dim);

    GV_Database *db = gv_db_open(NULL, dim, GV_INDEX_TYPE_IVFPQ);
    if (!db) {
        fprintf(stderr, "db open failed\n");
        return 1;
    }
    /* train */
    if (gv_ivfpq_train(db->hnsw_index, data, train) != 0) {
        fprintf(stderr, "train failed\n");
        return 1;
    }
    for (size_t i = 0; i < n; ++i) {
        if (gv_db_add_vector(db, data + i * dim, dim) != 0) {
            fprintf(stderr, "insert failed at %zu\n", i);
            return 1;
        }
    }

    GV_SearchResult *res = (GV_SearchResult *)malloc(k * sizeof(GV_SearchResult));
    double t0 = now_ms();
    for (size_t qi = 0; qi < q; ++qi) {
        int found = gv_db_search_ivfpq_opts(db, queries + qi * dim, k, res,
                                            use_cosine ? GV_DISTANCE_COSINE : GV_DISTANCE_EUCLIDEAN,
                                            nprobe, rerank);
        if (found < 0) {
            fprintf(stderr, "search failed\n");
            return 1;
        }
    }
    double t1 = now_ms();
    double qps = (double)q / ((t1 - t0) / 1000.0);
    printf("IVF-PQ benchmark: n=%zu dim=%zu q=%zu k=%zu nlist=%zu m=%zu nbits=%u nprobe=%zu rerank=%zu cosine=%d time=%.2fms qps=%.1f\n",
           n, dim, q, k, nlist, m, nbits, nprobe, rerank, use_cosine, t1 - t0, qps);

    free(res);
    gv_db_close(db);
    free(data);
    free(queries);
    return 0;
}

