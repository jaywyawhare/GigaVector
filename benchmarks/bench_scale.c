/* bench_scale — reproducible scale harness for the "vector db killer" claim.
 *
 * Measures, for a configurable corpus size N (default 100k; pass 10000000 for
 * the 10M run):
 *   - indexed build throughput (vecs/sec), using the bulk path:
 *     gv_hnsw_insert_raw + db_hnsw_build_parallel (multi-threaded linking)
 *   - peak RSS after build
 *   - search latency p50/p95 and QPS at k=10
 *   - recall@k against exact brute-force ground truth (sampled queries)
 *
 * Metadata/WAL overhead is deliberately excluded: these are index-level
 * scale numbers. Deterministic PRNG so runs are comparable across versions.
 * NOTE: uniform random vectors are the hardest case for ANN graphs (no
 * manifold structure) — treat recall here as a floor, not a headline.
 *
 * Usage: bench_scale [N] [dim] [queries] [k]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "storage/database.h"
#include "index/hnsw.h"
#include "core/memory.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* xorshift64* — fast deterministic fills without rand() range limits. */
static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint64_t next_u64(void) {
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return rng * 2685821657736338717ULL;
}
static float next_unit(void) {
    return (float)((next_u64() >> 40) / (double)(1 << 24)) - 0.5f;
}

static size_t rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "VmRSS: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    size_t n      = argc > 1 ? strtoull(argv[1], NULL, 10) : 100000;
    size_t dim    = argc > 2 ? strtoull(argv[2], NULL, 10) : 128;
    size_t qn     = argc > 3 ? strtoull(argv[3], NULL, 10) : 200;
    size_t k      = argc > 4 ? strtoull(argv[4], NULL, 10) : 10;

    printf("bench_scale: N=%zu dim=%zu queries=%zu k=%zu\n", n, dim, qn, k);

    float *data = (float *)gv_alloc(n * dim * sizeof(float));
    float *queries = (float *)gv_alloc(qn * dim * sizeof(float));
    if (!data || !queries) { fprintf(stderr, "oom\n"); return 1; }
    for (size_t i = 0; i < n * dim; i++) data[i] = next_unit();
    for (size_t i = 0; i < qn * dim; i++) queries[i] = next_unit();

    GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_HNSW);
    if (!db) { fprintf(stderr, "db open failed\n"); return 1; }

    /* ---- build: staged SoA load + multi-threaded graph construction ---- */
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 1;
    double t0 = now_ms();
    if (db_add_vectors_parallel(db, data, n, dim, (size_t)ncpu) != 0) {
        fprintf(stderr, "parallel bulk add failed\n");
        return 1;
    }
    double total_s = (now_ms() - t0) / 1000.0;
    printf("build: %.1fs (%.0f vec/s, %ld threads), RSS=%zuMB\n",
           total_s, n / total_s, ncpu, rss_kb() / 1024);

    /* ---- ground truth on a sample (exact scan over stored vectors) ---- */
    size_t gt_sample = qn < 20 ? qn : 20;   /* exact scan is O(N); keep bounded */
    size_t *gt = (size_t *)gv_alloc(gt_sample * sizeof(size_t));
    if (!gt) return 1;
    t0 = now_ms();
    for (size_t qi = 0; qi < gt_sample; qi++) {
        double best = INFINITY; size_t best_i = 0;
        for (size_t i = 0; i < n; i++) {
            double d = 0;
            const float *a = queries + qi * dim, *b = data + i * dim;
            for (size_t c = 0; c < dim; c++) { double e = a[c] - b[c]; d += e * e; }
            if (d < best) { best = d; best_i = i; }
        }
        gt[qi] = best_i;
    }
    printf("ground truth: %.1fs (%zu exact scans)\n",
           (now_ms() - t0) / 1000.0, gt_sample);

    /* ---- latency + recall@k ---- */
    GV_SearchResult *res =
        (GV_SearchResult *)gv_alloc(k * sizeof(GV_SearchResult));
    double *lat = (double *)gv_alloc(qn * sizeof(double));
    if (!res || !lat) return 1;

    size_t hits = 0, scored = 0;
    for (size_t qi = 0; qi < qn; qi++) {
        double s = now_ms();
        int found = db_search(db, queries + qi * dim, k, res, GV_DISTANCE_EUCLIDEAN);
        lat[qi] = now_ms() - s;
        if (found < 0) continue;
        if (qi < gt_sample) {
            scored++;
            /* internal ids follow insertion order */
            for (int j = 0; j < found; j++) {
                if ((size_t)res[j].id == gt[qi]) { hits++; break; }
            }
        }
    }

    qsort(lat, qn, sizeof(double), cmp_double);
    double p50 = lat[qn / 2], p95 = lat[(qn * 95) / 100];
    double search_s = 0;
    for (size_t qi = 0; qi < qn; qi++) search_s += lat[qi];
    search_s /= 1000.0;

    printf("search: p50=%.3fms p95=%.3fms qps=%.0f recall@%zu=%.3f (%zu/%zu)\n",
           p50, p95, (double)qn / search_s, k,
           scored ? (double)hits / (double)scored : 0.0, hits, scored);

    gv_free(lat); gv_free(res); gv_free(gt);
    gv_free(queries); gv_free(data);
    db_close(db);
    return 0;
}
