/**
 * IVFDisk larger-than-RAM benchmark.
 *
 * Usage: bench_ivfdisk [n] [dim] [nlist] [nprobe] [queries] [mode]
 *   mode=1  smoke (10k vectors, tuned nlist/nprobe)
 *   mode=0  full  (default 1M vectors; brute-force recall, no Flat index)
 *
 * Environment overrides (optional):
 *   BENCH_IVFDISK_N, BENCH_IVFDISK_DIM, BENCH_IVFDISK_NLIST,
 *   BENCH_IVFDISK_NPROBE, BENCH_IVFDISK_QUERIES, BENCH_IVFDISK_CACHE_MB
 *
 * Reports: insert time, QPS, recall@10 vs brute-force, p50/p99 latency, RSS.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/resource.h>
#endif

#include "gigavector.h"
#include "index/flat.h"
#include "index/ivfdisk.h"

#define BENCH_RECALL_K 10u

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static size_t rss_kb(void)
{
#if defined(__linux__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return (size_t)ru.ru_maxrss;
    }
#endif
    return 0;
}

static size_t env_size_t(const char *name, size_t fallback)
{
    const char *v = getenv(name);
    if (!v || !*v) return fallback;
    return (size_t)strtoul(v, NULL, 10);
}

static void fill_random(float *data, size_t n, size_t dim, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static float euclid_sq(const float *a, const float *b, size_t dim)
{
    float s = 0.f;
    for (size_t d = 0; d < dim; ++d) {
        float diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
}

/** Top-k by squared Euclidean distance; writes vector ids 0..n-1. */
static void brute_topk(const float *query, const float *data, size_t n, size_t dim, size_t k,
                       size_t *out_ids)
{
    float *best_dist = (float *)malloc(k * sizeof(float));
    if (!best_dist) return;

    for (size_t i = 0; i < k; ++i) {
        out_ids[i] = (size_t)-1;
        best_dist[i] = INFINITY;
    }

    for (size_t i = 0; i < n; ++i) {
        float d = euclid_sq(query, data + i * dim, dim);
        size_t worst = 0;
        for (size_t j = 1; j < k; ++j) {
            if (best_dist[j] > best_dist[worst]) worst = j;
        }
        if (d < best_dist[worst]) {
            best_dist[worst] = d;
            out_ids[worst] = i;
        }
    }
    free(best_dist);
}

static int id_in_topk(size_t id, const size_t *ids, size_t k)
{
    for (size_t i = 0; i < k; ++i) {
        if (ids[i] == id) return 1;
    }
    return 0;
}

static double recall_at_k_brute(const float *queries, size_t qcount,
                                const float *data, size_t n, size_t dim, size_t k,
                                GV_IVFDiskIndex *idx)
{
    size_t hits = 0;
    size_t total = 0;
    GV_SearchResult *res = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    size_t *gt = (size_t *)calloc(k, sizeof(size_t));
    if (!res || !gt) {
        free(res);
        free(gt);
        return 0.0;
    }

    for (size_t qi = 0; qi < qcount; ++qi) {
        const float *query = queries + qi * dim;
        brute_topk(query, data, n, dim, k, gt);

        int found = ivfdisk_search(idx, query, k, res, GV_DISTANCE_EUCLIDEAN);
        if (found <= 0) continue;

        for (int i = 0; i < found; ++i) {
            if (id_in_topk(res[i].id, gt, k)) hits++;
        }
        total += (size_t)found;
    }

    free(res);
    free(gt);
    return total > 0 ? (double)hits / (double)total : 0.0;
}

static double recall_at_k_flat(const float *queries, size_t qcount,
                               void *flat, size_t dim, size_t k,
                               GV_IVFDiskIndex *idx)
{
    size_t hits = 0;
    size_t total = 0;
    GV_SearchResult *res = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    GV_SearchResult *gt = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    if (!res || !gt) {
        free(res);
        free(gt);
        return 0.0;
    }

    for (size_t qi = 0; qi < qcount; ++qi) {
        const float *query = queries + qi * dim;
        GV_Vector qv;
        qv.dimension = dim;
        qv.data = (float *)query;
        qv.metadata = NULL;

        int found = ivfdisk_search(idx, query, k, res, GV_DISTANCE_EUCLIDEAN);
        int ngt = flat_search(flat, &qv, k, gt, GV_DISTANCE_EUCLIDEAN, NULL, NULL);
        if (found <= 0 || ngt <= 0) continue;

        for (int i = 0; i < found; ++i) {
            for (int j = 0; j < ngt; ++j) {
                if (res[i].id == gt[j].id) {
                    hits++;
                    break;
                }
            }
        }
        total += (size_t)found;
        for (int j = 0; j < ngt; ++j) {
            if (gt[j].vector) vector_destroy((GV_Vector *)gt[j].vector);
        }
    }

    free(res);
    free(gt);
    return total > 0 ? (double)hits / (double)total : 0.0;
}

static void latency_percentiles(GV_IVFDiskIndex *idx, const float *queries,
                                size_t qcount, size_t dim, size_t k,
                                double *p50_ms, double *p99_ms)
{
    size_t samples = qcount * 5;
    double *lat = (double *)malloc(samples * sizeof(double));
    if (!lat) return;

    GV_SearchResult *res = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    if (!res) {
        free(lat);
        return;
    }

    for (size_t i = 0; i < samples; ++i) {
        const float *q = queries + (i % qcount) * dim;
        double t0 = now_ms();
        ivfdisk_search(idx, q, k, res, GV_DISTANCE_EUCLIDEAN);
        lat[i] = now_ms() - t0;
    }

    qsort(lat, samples, sizeof(double), cmp_double);

    *p50_ms = lat[samples / 2];
    *p99_ms = lat[(samples * 99) / 100];
    free(res);
    free(lat);
}

int main(int argc, char **argv)
{
    size_t n = env_size_t("BENCH_IVFDISK_N", 1000000);
    size_t dim = env_size_t("BENCH_IVFDISK_DIM", 128);
    size_t nlist = env_size_t("BENCH_IVFDISK_NLIST", 1024);
    size_t nprobe = env_size_t("BENCH_IVFDISK_NPROBE", 64);
    size_t queries = env_size_t("BENCH_IVFDISK_QUERIES", 100);
    size_t cache_mb = env_size_t("BENCH_IVFDISK_CACHE_MB", 128);
    int smoke = 0;

    if (argc > 1) n = (size_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) dim = (size_t)strtoul(argv[2], NULL, 10);
    if (argc > 3) nlist = (size_t)strtoul(argv[3], NULL, 10);
    if (argc > 4) nprobe = (size_t)strtoul(argv[4], NULL, 10);
    if (argc > 5) queries = (size_t)strtoul(argv[5], NULL, 10);
    if (argc > 6) smoke = atoi(argv[6]);

    if (smoke) {
        n = 10000;
        queries = 30;
        nlist = 64;
        nprobe = 48;
        cache_mb = 64;
    } else if (cache_mb == 128 && n >= 500000) {
        cache_mb = 256;
    }

    const int use_brute_gt = (n > 50000);
    const size_t recall_k = BENCH_RECALL_K;

    size_t train_n = nlist * 40;
    if (train_n > n) train_n = n;
    if (train_n < nlist) train_n = nlist;

    char dir[512];
    const char *tmpdir = getenv("TMPDIR");
    if (tmpdir == NULL || !*tmpdir) tmpdir = getenv("TEMP");
    if (tmpdir == NULL || !*tmpdir) tmpdir = "/tmp";
    snprintf(dir, sizeof(dir), "%s/gv_bench_ivfdisk_XXXXXX", tmpdir);
    if (!mkdtemp(dir)) {
        fprintf(stderr, "mkdtemp failed\n");
        return 1;
    }

    fprintf(stderr, "IVFDisk bench: n=%zu dim=%zu nlist=%zu nprobe=%zu queries=%zu mode=%s\n",
            n, dim, nlist, nprobe, queries, smoke ? "smoke" : "full");
    fprintf(stderr, "  data_dir=%s gt=%s cache_mb=%zu\n",
            dir, use_brute_gt ? "brute_force" : "flat_index", cache_mb);

    float *data = (float *)malloc(n * dim * sizeof(float));
    float *train = (float *)malloc(train_n * dim * sizeof(float));
    float *query = (float *)malloc(queries * dim * sizeof(float));
    if (!data || !train || !query) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    fill_random(data, n, dim, 42);
    memcpy(train, data, train_n * dim * sizeof(float));
    fill_random(query, queries, dim, 99);

    GV_IVFDiskConfig cfg;
    ivfdisk_config_init(&cfg);
    cfg.nlist = nlist;
    cfg.nprobe = nprobe;
    cfg.train_iters = 15;
    cfg.cache_size_mb = cache_mb;
    cfg.use_hnsw_head = (nlist >= 256) ? 1 : 0;
    cfg.use_sq8 = 0;
    cfg.data_dir = dir;

    GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
    if (!idx) {
        fprintf(stderr, "ivfdisk_create failed\n");
        return 1;
    }

    double t_train = now_ms();
    if (ivfdisk_train(idx, train, train_n) != 0) {
        fprintf(stderr, "train failed\n");
        return 1;
    }
    t_train = now_ms() - t_train;
    free(train);
    train = NULL;

    double t_ins = now_ms();
    for (size_t i = 0; i < n; ++i) {
        if (ivfdisk_insert(idx, data + i * dim, dim, i) != 0) {
            fprintf(stderr, "insert failed at %zu\n", i);
            return 1;
        }
        if ((i + 1) % 100000 == 0) {
            fprintf(stderr, "  inserted %zu / %zu\n", i + 1, n);
        }
    }
    t_ins = now_ms() - t_ins;

    size_t rss_after_insert = rss_kb();
    size_t dataset_bytes = n * dim * sizeof(float);
    size_t expected_ram = nlist * dim * sizeof(float) + cfg.cache_size_mb * 1024u * 1024u;

    double recall = 0.0;
    void *flat = NULL;

    if (use_brute_gt) {
        fprintf(stderr, "  computing recall@%zu vs brute-force (%zu queries)...\n",
                recall_k, queries);
        double t_recall = now_ms();
        recall = recall_at_k_brute(query, queries, data, n, dim, recall_k, idx);
        fprintf(stderr, "  recall_ms=%.1f\n", now_ms() - t_recall);
    } else {
        flat = flat_create(dim, NULL, NULL);
        if (!flat) {
            fprintf(stderr, "flat_create failed\n");
            return 1;
        }
        for (size_t i = 0; i < n; ++i) {
            GV_Vector *v = vector_create_from_data(dim, data + i * dim);
            if (!v || flat_insert(flat, v) != 0) {
                fprintf(stderr, "flat insert failed at %zu\n", i);
                return 1;
            }
        }
        recall = recall_at_k_flat(query, queries, flat, dim, recall_k, idx);
    }

    double p50 = 0.0, p99 = 0.0;
    latency_percentiles(idx, query, queries, dim, recall_k, &p50, &p99);

    GV_IVFDiskStats stats;
    ivfdisk_get_stats(idx, &stats);

    printf("IVFDisk benchmark\n");
    printf("  mode=%s vectors=%zu dim=%zu nlist=%zu nprobe=%zu queries=%zu\n",
           smoke ? "smoke" : "full", n, dim, nlist, nprobe, queries);
    printf("  train_ms=%.1f insert_ms=%.1f insert_ips=%.0f\n",
           t_train, t_ins, (double)n / (t_ins / 1000.0));
    printf("  recall@%zu=%.3f (target >= 0.90) gt=%s\n",
           recall_k, recall, use_brute_gt ? "brute_force" : "flat");
    printf("  search_p50_ms=%.3f search_p99_ms=%.3f\n", p50, p99);
    printf("  rss_kb=%zu (after insert) expected_cap~%zu dataset_bytes=%zu\n",
           rss_after_insert, expected_ram / 1024, dataset_bytes);
    printf("  ram_ratio=%.3f (rss/expected_cap) cache_hits=%zu cache_misses=%zu segments=%zu\n",
           (double)(rss_after_insert * 1024) / (double)expected_ram,
           stats.cache_hits, stats.cache_misses, stats.segment_count);
    printf("  hnsw_head=%d sq8=%d cache_mb=%zu\n",
           cfg.use_hnsw_head, cfg.use_sq8, cache_mb);

    int pass = 1;
    if (recall < 0.90) {
        fprintf(stderr, "FAIL: recall@%zu %.3f < 0.90\n", recall_k, recall);
        pass = 0;
    }
    if (rss_after_insert * 1024 > dataset_bytes / 2) {
        fprintf(stderr, "NOTE: RSS includes libc/index; vectors live on disk (%zu bytes)\n",
                dataset_bytes);
    }
    if (rss_after_insert * 1024 > expected_ram * 3) {
        fprintf(stderr, "FAIL: RSS exceeds 3x head+cache budget\n");
        pass = 0;
    }

    ivfdisk_destroy(idx);
    if (flat) flat_destroy(flat);
    free(data);
    free(query);

    return pass ? 0 : 1;
}
