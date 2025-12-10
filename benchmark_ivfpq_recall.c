#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

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

static void brute_force(const float *queries, size_t qcount,
                        const float *base, size_t n, size_t dim, size_t k,
                        size_t *hits) {
    for (size_t qi = 0; qi < qcount; ++qi) {
        const float *q = queries + qi * dim;
        float bestd = INFINITY;
        size_t besti = (size_t)-1;
        for (size_t i = 0; i < n; ++i) {
            const float *v = base + i * dim;
            float d = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                float diff = q[j] - v[j];
                d += diff * diff;
            }
            if (d < bestd) {
                bestd = d;
                besti = i;
            }
        }
        hits[qi * k + 0] = besti;
    }
}

int main(int argc, char **argv) {
    size_t dim = 64;
    size_t train = 4000;
    size_t n = 20000;
    size_t q = 200;
    size_t k = 10;
    size_t nprobe = 0;
    size_t rerank = 0;
    int use_cosine = 0;
    if (argc > 1) n = (size_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) q = (size_t)strtoul(argv[2], NULL, 10);
    if (argc > 3) nprobe = (size_t)strtoul(argv[3], NULL, 10);
    if (argc > 4) rerank = (size_t)strtoul(argv[4], NULL, 10);
    if (argc > 5) use_cosine = atoi(argv[5]);

    srand(123);
    float *data = (float *)malloc(n * dim * sizeof(float));
    float *queries = (float *)malloc(q * dim * sizeof(float));
    size_t *gt = (size_t *)malloc(q * k * sizeof(size_t));
    if (!data || !queries || !gt) return 1;
    fill_random(data, n, dim);
    fill_random(queries, q, dim);

    GV_Database *db = gv_db_open(NULL, dim, GV_INDEX_TYPE_IVFPQ);
    if (!db) {
        fprintf(stderr, "db open failed\n");
        return 1;
    }
    if (gv_ivfpq_train(db->hnsw_index, data, train) != 0) {
        fprintf(stderr, "train failed\n");
        return 1;
    }
    for (size_t i = 0; i < n; ++i) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu", i);
        const char *key = "id";
        const char *val = buf;
        if (gv_db_add_vector_with_metadata(db, data + i * dim, dim, key, val) != 0) {
            fprintf(stderr, "insert failed at %zu\n", i);
            return 1;
        }
    }

    brute_force(queries, q, data, n, dim, k, gt);

    GV_SearchResult *res = (GV_SearchResult *)malloc(k * sizeof(GV_SearchResult));
    size_t correct = 0;
    double t0 = now_ms();
    for (size_t qi = 0; qi < q; ++qi) {
        int found = gv_db_search_ivfpq_opts(db, queries + qi * dim, k, res,
                                            use_cosine ? GV_DISTANCE_COSINE : GV_DISTANCE_EUCLIDEAN,
                                            nprobe, rerank);
        if (found < 0) {
            fprintf(stderr, "search failed\n");
            return 1;
        }
        size_t gt_id = gt[qi * k + 0];
        const char *hit_id = NULL;
        for (size_t j = 0; j < k; ++j) {
            if (!res[j].vector) continue;
            const GV_Metadata *m = res[j].vector->metadata;
            while (m) {
                if (strcmp(m->key, "id") == 0) {
                    hit_id = m->value;
                    break;
                }
                m = m->next;
            }
            if (hit_id) break;
        }
        if (hit_id && (size_t)strtoul(hit_id, NULL, 10) == gt_id) {
            correct++;
        }
    }
    double t1 = now_ms();
    double recall = (double)correct / (double)q;
    double qps = (double)q / ((t1 - t0) / 1000.0);
    printf("IVF-PQ recall@1=%.3f q=%zu k=%zu time=%.2fms qps=%.1f\n", recall, q, k, t1 - t0, qps);

    free(res);
    gv_db_close(db);
    free(data);
    free(queries);
    free(gt);
    return 0;
}

