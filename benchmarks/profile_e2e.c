/*
 * profile_e2e - granular end-to-end profiler for the GigaVector stack.
 *
 * Walks the same four subsystems as tests/test_e2e.c (vector DB, graph+Cypher,
 * SPLADE learned-sparse, hybrid dense+sparse) but at a configurable scale, and
 * times every sub-operation with a nestable scope profiler that reports, per
 * named scope:
 *
 *   calls | total(ms) | self(ms) | avg(ms) | max(ms) | self% | allocs | net(KB)
 *
 * "self" excludes nested scopes, so the report ranks the true hot spots rather
 * than the outer phase that contains them. Allocation columns come from the
 * gv_alloc choke point (gv_alloc_stats): build the library with
 * -DGV_PROFILE_ALLOC to populate them (the `make profile-e2e` target does).
 *
 * Set GV_PROF_TRACE=path to also emit a chrome://tracing / Perfetto JSON of
 * every scope span.
 *
 * Covers: the vector-DB lifecycle (FLAT), graph+Cypher, SPLADE, hybrid fusion,
 * the full dense index matrix (FLAT/HNSW/KDTREE/LSH/RABITQ + trained
 * IVFFLAT/IVFSQ8/IVFTURBOQUANT/PQ/IVFPQ: train+build+search), every distance
 * metric, persistence (save+reopen), and transactions (commit+rollback).
 *
 * Usage: profile_e2e [N_vectors] [dim] [queries] [N_docs] [text_queries] [idx_N]
 *   defaults: 20000 128 500 3000 500 (idx_N auto = min(N/4, 5000))
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "gigavector.h"
#include "admin/raft_log.h"
#include "admin/repl_sim.h"
#include "api/server.h"
#include "api/grpc.h"
#include "core/memory.h"
#include "features/cypher.h"
#include "features/knowledge_graph.h"
#include "index/diskann.h"
#include "index/ivfdisk.h"
#include "index/ivfflat.h"
#include "index/ivfpq.h"
#include "index/ivfsq8.h"
#include "index/ivfturboquant.h"
#include "index/pq.h"
#include "index/sparse_index.h"
#include "storage/sparse_vector.h"
#include "multimodal/bm25.h"
#include "multimodal/learned_sparse.h"
#include "multimodal/splade.h"
#include "search/distance.h"
#include "search/hybrid_search.h"
#include "storage/database.h"
#include "storage/transaction.h"

/* ------------------------------------------------------------- scope profiler */

#define PROF_MAX_SCOPES 256
#define PROF_MAX_DEPTH  32
#define PROF_MAX_SPANS  (1 << 20)

typedef struct {
    const char *name;
    uint64_t    calls;
    double      total_ms;  /* inclusive */
    double      self_ms;   /* inclusive minus direct children */
    double      min_ms, max_ms;
    uint64_t    allocs;    /* gv_alloc/calloc calls charged to this scope */
    long long   net_bytes; /* live-byte delta charged to this scope */
} ProfStat;

typedef struct {
    int      stat;       /* index into g_stats */
    double   start_ms;
    double   child_ms;   /* inclusive time of direct children */
    GV_AllocStats a0;    /* alloc snapshot at scope entry */
    double   span_t0;    /* for the chrome trace */
} ProfFrame;

typedef struct {
    const char *name;
    double      ts_us, dur_us;
    int         depth;
} ProfSpan;

static ProfStat  g_stats[PROF_MAX_SCOPES];
static int       g_stat_count;
static ProfFrame g_stack[PROF_MAX_DEPTH];
static int       g_depth;
static ProfSpan *g_spans;
static size_t    g_span_count;
static double    g_t0_ms; /* program start, so trace timestamps begin near 0 */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int prof_stat_for(const char *name) {
    for (int i = 0; i < g_stat_count; i++)
        if (g_stats[i].name == name || strcmp(g_stats[i].name, name) == 0)
            return i;
    if (g_stat_count >= PROF_MAX_SCOPES) {
        fprintf(stderr, "profiler: scope table full (%d); raise PROF_MAX_SCOPES\n",
                PROF_MAX_SCOPES);
        return PROF_MAX_SCOPES - 1; /* fold extras into the last slot, never OOB */
    }
    int i = g_stat_count++;
    g_stats[i].name = name;
    g_stats[i].min_ms = 1e30;
    return i;
}

static void prof_begin(const char *name) {
    if (g_depth >= PROF_MAX_DEPTH) return;
    ProfFrame *f = &g_stack[g_depth++];
    f->stat = prof_stat_for(name);
    f->child_ms = 0.0;
    gv_alloc_stats(&f->a0);
    f->span_t0 = now_ms();
    f->start_ms = f->span_t0; /* read clock last so setup above is not timed */
}

static void prof_end(void) {
    double end_ms = now_ms(); /* read clock first */
    if (g_depth <= 0) return;
    ProfFrame *f = &g_stack[--g_depth];
    double incl = end_ms - f->start_ms;

    ProfStat *s = &g_stats[f->stat];
    s->calls++;
    s->total_ms += incl;
    double self = incl - f->child_ms;
    if (self < 0) self = 0;
    s->self_ms += self;
    if (incl < s->min_ms) s->min_ms = incl;
    if (incl > s->max_ms) s->max_ms = incl;

    GV_AllocStats a1;
    gv_alloc_stats(&a1);
    s->allocs    += a1.alloc_count - f->a0.alloc_count;
    s->net_bytes += (long long)a1.live_bytes - (long long)f->a0.live_bytes;

    if (g_depth > 0) g_stack[g_depth - 1].child_ms += incl;

    if (g_spans && g_span_count < PROF_MAX_SPANS) {
        ProfSpan *sp = &g_spans[g_span_count++];
        sp->name = s->name;
        sp->ts_us = (f->span_t0 - g_t0_ms) * 1000.0;
        sp->dur_us = incl * 1000.0;
        sp->depth = g_depth;
    }
}

/* cleanup attribute closes the scope even on an early return/break inside the
 * body, so prof_end() always pairs with prof_begin(). */
static inline void prof_scope_end_(int *unused) { (void)unused; prof_end(); }
#define PROF_SCOPE(name) \
    for (int _p __attribute__((cleanup(prof_scope_end_))) = (prof_begin(name), 0); \
         _p == 0; _p = 1)

static int prof_cmp_self(const void *a, const void *b) {
    double x = ((const ProfStat *)a)->self_ms, y = ((const ProfStat *)b)->self_ms;
    return (x < y) - (x > y);
}

static void prof_report(void) {
    ProfStat sorted[PROF_MAX_SCOPES];
    memcpy(sorted, g_stats, sizeof(ProfStat) * (size_t)g_stat_count);
    qsort(sorted, (size_t)g_stat_count, sizeof(ProfStat), prof_cmp_self);

    double total_self = 0;
    for (int i = 0; i < g_stat_count; i++) total_self += sorted[i].self_ms;
    if (total_self <= 0) total_self = 1;

    printf("\n%-30s %8s %10s %10s %9s %9s %6s %9s %10s\n", "scope", "calls",
           "total ms", "self ms", "avg ms", "max ms", "self%", "allocs", "net KB");
    printf("------------------------------------------------------------------"
           "------------------------------------------------------\n");
    for (int i = 0; i < g_stat_count; i++) {
        ProfStat *s = &sorted[i];
        printf("%-30s %8llu %10.2f %10.2f %9.4f %9.4f %5.1f%% %9llu %10.1f\n",
               s->name, (unsigned long long)s->calls, s->total_ms, s->self_ms,
               s->calls ? s->total_ms / (double)s->calls : 0.0, s->max_ms,
               100.0 * s->self_ms / total_self, (unsigned long long)s->allocs,
               s->net_bytes / 1024.0);
    }
    printf("------------------------------------------------------------------"
           "------------------------------------------------------\n");
    printf("total self ms: %.2f\n", total_self);
    if (!gv_alloc_stats_enabled())
        printf("(alloc columns are zero: rebuild the library with -DGV_PROFILE_ALLOC)\n");
}

static void prof_write_trace(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "could not open trace %s\n", path); return; }
    fputs("{\"displayTimeUnit\":\"ns\",\"traceEvents\":[", f);
    for (size_t i = 0; i < g_span_count; i++) {
        ProfSpan *s = &g_spans[i];
        fprintf(f, "%s{\"name\":\"%s\",\"ph\":\"X\",\"pid\":1,\"tid\":%d,"
                   "\"ts\":%.3f,\"dur\":%.3f}",
                i ? "," : "", s->name, s->depth, s->ts_us, s->dur_us);
    }
    fputs("]}\n", f);
    fclose(f);
    printf("wrote chrome trace: %s (%zu spans)\n", path, g_span_count);
}

/* ------------------------------------------------------------------ workload */

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

static const char *WORDS[] = {
    "vector", "database", "search", "neural", "network", "embedding", "graph",
    "raft", "consensus", "replication", "index", "query", "sparse", "dense",
    "fusion", "ranking", "distance", "cosine", "cluster", "quantize", "token",
    "semantic", "retrieval", "cache", "shard", "latency", "throughput", "kernel",
};
#define NWORDS (sizeof(WORDS) / sizeof(WORDS[0]))

static void make_sentence(char *buf, size_t cap, int nwords) {
    size_t off = 0;
    for (int i = 0; i < nwords && off + 20 < cap; i++) {
        const char *w = WORDS[next_u64() % NWORDS];
        off += (size_t)snprintf(buf + off, cap - off, "%s%s", i ? " " : "", w);
    }
}

static void phase_vector_db(size_t n, size_t dim, size_t queries) {
    PROF_SCOPE("1.vectordb") {
        GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_FLAT);
        if (!db) { fprintf(stderr, "db_open failed\n"); return; }

        float *vec = (float *)malloc(dim * sizeof(float));
        PROF_SCOPE("1.vectordb/add") {
            for (size_t i = 0; i < n; i++) {
                for (size_t d = 0; d < dim; d++) vec[d] = next_unit();
                char val[16];
                snprintf(val, sizeof(val), "b%zu", i % 8);
                (void)db_add_vector_with_metadata(db, vec, dim, "bucket", val);
            }
        }

        GV_SearchResult res[16];
        PROF_SCOPE("1.vectordb/search") {
            for (size_t q = 0; q < queries; q++) {
                for (size_t d = 0; d < dim; d++) vec[d] = next_unit();
                int nn = db_search(db, vec, 10, res, GV_DISTANCE_EUCLIDEAN);
                if (nn > 0) gv_search_results_free(res, (size_t)nn);
            }
        }
        PROF_SCOPE("1.vectordb/filter") {
            for (size_t q = 0; q < queries / 4; q++) {
                for (size_t d = 0; d < dim; d++) vec[d] = next_unit();
                int nn = db_search_with_filter_expr(db, vec, 10, res,
                                                    GV_DISTANCE_EUCLIDEAN,
                                                    "bucket == \"b3\"");
                if (nn > 0) gv_search_results_free(res, (size_t)nn);
            }
        }
        PROF_SCOPE("1.vectordb/range") {
            for (size_t q = 0; q < queries / 4; q++) {
                for (size_t d = 0; d < dim; d++) vec[d] = next_unit();
                int nn = db_range_search(db, vec, 5.0f, res, 16, GV_DISTANCE_EUCLIDEAN);
                if (nn > 0) gv_search_results_free(res, (size_t)nn);
            }
        }
        free(vec);
        db_close(db);
    }
}

static void phase_graph_cypher(size_t chain, size_t matches) {
    PROF_SCOPE("2.graph") {
        GV_KnowledgeGraph *kg = kg_create(NULL);
        GV_CypherEngine *cy = kg ? cypher_create(kg) : NULL;
        if (!cy) { if (kg) kg_destroy(kg); return; }
        GV_CypherResult r;

        PROF_SCOPE("2.graph/create") {
            char q[256];
            for (size_t i = 0; i + 1 < chain; i++) {
                snprintf(q, sizeof(q),
                         "CREATE (a:Doc {name:'n%zu', emb:'[%d,0,0]'})"
                         "-[:CITES]->(b:Doc {name:'n%zu', emb:'[0,%d,0]'})",
                         i, (int)(i % 3), i + 1, (int)(i % 3));
                if (cypher_execute(cy, q, &r) == 0) cypher_free_result(&r);
            }
        }
        PROF_SCOPE("2.graph/match") {
            for (size_t m = 0; m < matches; m++) {
                if (cypher_execute(cy, "MATCH (d:Doc) RETURN count(*)", &r) == 0)
                    cypher_free_result(&r);
            }
        }
        PROF_SCOPE("2.graph/varlen+vecpred") {
            for (size_t m = 0; m < matches; m++) {
                if (cypher_execute(cy,
                        "MATCH (a:Doc {name:'n0'})-[:CITES*1..3]->(x) "
                        "WHERE vector_distance(x.emb, '[1,0,0]') < 0.9 "
                        "RETURN x.name ORDER BY x.name", &r) == 0)
                    cypher_free_result(&r);
            }
        }
        cypher_destroy(cy);
        kg_destroy(kg);
    }
}

static void phase_splade(size_t ndocs, size_t queries) {
    PROF_SCOPE("3.splade") {
        GV_SpladeConfig scfg;
        splade_config_init(&scfg);
        GV_SpladeEncoder *enc = splade_create(&scfg);
        GV_LearnedSparseConfig icfg;
        ls_config_init(&icfg);
        icfg.vocab_size = scfg.vocab_size;
        GV_LearnedSparseIndex *idx = ls_create(&icfg);
        if (!enc || !idx) { if (enc) splade_destroy(enc); if (idx) ls_destroy(idx); return; }

        char text[256];
        PROF_SCOPE("3.splade/index") {
            for (size_t i = 0; i < ndocs; i++) {
                make_sentence(text, sizeof(text), 8);
                (void)splade_index_add(enc, idx, text);
            }
        }
        GV_LearnedSparseResult res[10];
        PROF_SCOPE("3.splade/search") {
            for (size_t q = 0; q < queries; q++) {
                make_sentence(text, sizeof(text), 6);
                (void)splade_index_search(enc, idx, text, 10, res);
            }
        }
        ls_destroy(idx);
        splade_destroy(enc);
    }
}

static void phase_hybrid(size_t ndocs, size_t queries) {
    PROF_SCOPE("4.hybrid") {
        GV_Database *db = db_open(NULL, 16, GV_INDEX_TYPE_FLAT);
        GV_BM25Config bcfg;
        bm25_config_init(&bcfg);
        GV_BM25Index *bm = bm25_create(&bcfg);
        if (!db || !bm) { if (db) db_close(db); if (bm) bm25_destroy(bm); return; }

        float vec[16];
        char text[256];
        PROF_SCOPE("4.hybrid/index") {
            for (size_t i = 0; i < ndocs; i++) {
                for (int d = 0; d < 16; d++) vec[d] = next_unit();
                if (db_add_vector(db, vec, 16) != 0) {}
                make_sentence(text, sizeof(text), 8);
                (void)bm25_add_document(bm, i, text);
            }
        }
        GV_HybridConfig hcfg;
        hybrid_config_init(&hcfg);
        GV_HybridSearcher *hs = hybrid_create(db, bm, &hcfg);
        if (hs) {
            GV_HybridResult res[10];
            PROF_SCOPE("4.hybrid/search") {
                for (size_t q = 0; q < queries; q++) {
                    for (int d = 0; d < 16; d++) vec[d] = next_unit();
                    make_sentence(text, sizeof(text), 4);
                    (void)hybrid_search(hs, vec, text, 10, res);
                }
            }
            hybrid_destroy(hs);
        }
        bm25_destroy(bm);
        db_close(db);
    }
}

/* ---- index matrix: build + search for every reachable dense index --------- */

typedef int (*train_fn)(void *index, const float *data, size_t count);

/* One scoped build+search per index type. Trained types (IVF and PQ) are
 * trained on the first train_n vectors first; incremental types pass NULL. */
static void profile_one_index(const char *name, int index_type, train_fn train,
                              size_t n, size_t dim, size_t queries, size_t train_n) {
    char sb[64], ss[64], st[64];
    snprintf(sb, sizeof(sb), "idx/%s/build", name);
    snprintf(ss, sizeof(ss), "idx/%s/search", name);
    snprintf(st, sizeof(st), "idx/%s/train", name);

    GV_Database *db = db_open(NULL, dim, index_type);
    if (!db) { fprintf(stderr, "  %s: open failed\n", name); return; }

    float *data = (float *)malloc(n * dim * sizeof(float));
    for (size_t i = 0; i < n * dim; i++) data[i] = next_unit();

    if (train) {
        /* interned scope name: the table keys by string identity or value */
        prof_begin(strdup(st));
        if (train(db->hnsw_index, data, train_n < n ? train_n : n) != 0)
            fprintf(stderr, "  %s: train failed\n", name);
        prof_end();
    }

    int added = 0;
    prof_begin(strdup(sb));
    for (size_t i = 0; i < n; i++)
        if (db_add_vector(db, data + i * dim, dim) == 0) added++;
    prof_end();

    GV_SearchResult res[16];
    int last = 0;
    prof_begin(strdup(ss));
    for (size_t q = 0; q < queries; q++) {
        const float *qv = data + (q % n) * dim;
        int nn = db_search(db, qv, 10, res, GV_DISTANCE_EUCLIDEAN);
        if (nn > 0) { gv_search_results_free(res, (size_t)nn); last = nn; }
    }
    prof_end();

    if (added == 0 || last == 0)
        fprintf(stderr, "  %s: added=%d last_search=%d (skipped/unsupported)\n",
                name, added, last);
    free(data);
    db_close(db);
}

static void phase_index_matrix(size_t n, size_t dim, size_t queries) {
    PROF_SCOPE("5.index_matrix") {
        profile_one_index("flat",   GV_INDEX_TYPE_FLAT,   NULL, n, dim, queries, 0);
        profile_one_index("hnsw",   GV_INDEX_TYPE_HNSW,   NULL, n, dim, queries, 0);
        profile_one_index("kdtree", GV_INDEX_TYPE_KDTREE, NULL, n, dim, queries, 0);
        profile_one_index("lsh",    GV_INDEX_TYPE_LSH,    NULL, n, dim, queries, 0);
        profile_one_index("rabitq", GV_INDEX_TYPE_RABITQ, NULL, n, dim, queries, 0);
        size_t tn = n / 4 + 256;
        profile_one_index("ivfflat", GV_INDEX_TYPE_IVFFLAT, ivfflat_train, n, dim, queries, tn);
        profile_one_index("ivfsq8",  GV_INDEX_TYPE_IVFSQ8,  ivfsq8_train,  n, dim, queries, tn);
        profile_one_index("ivfturbo", GV_INDEX_TYPE_IVFTURBOQUANT, ivfturboquant_train, n, dim, queries, tn);
        profile_one_index("pq",     GV_INDEX_TYPE_PQ,     pq_train,     n, dim, queries, tn);
        profile_one_index("ivfpq",  GV_INDEX_TYPE_IVFPQ,  gv_ivfpq_train, n, dim, queries, tn);
    }
}

/* ---- distance metrics: same FLAT corpus, each metric ---------------------- */

static void phase_metrics(size_t n, size_t dim, size_t queries) {
    PROF_SCOPE("6.metrics") {
        GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_FLAT);
        if (!db) return;
        float *data = (float *)malloc(n * dim * sizeof(float));
        for (size_t i = 0; i < n * dim; i++) data[i] = next_unit();
        for (size_t i = 0; i < n; i++) if (db_add_vector(db, data + i * dim, dim)) {}

        static const struct { const char *n; GV_DistanceType t; } M[] = {
            {"euclidean", GV_DISTANCE_EUCLIDEAN}, {"cosine", GV_DISTANCE_COSINE},
            {"dot", GV_DISTANCE_DOT_PRODUCT}, {"manhattan", GV_DISTANCE_MANHATTAN},
            {"hamming", GV_DISTANCE_HAMMING}, {"jaccard", GV_DISTANCE_JACCARD},
        };
        GV_SearchResult res[16];
        for (size_t m = 0; m < sizeof(M) / sizeof(M[0]); m++) {
            char s[48]; snprintf(s, sizeof(s), "metric/%s", M[m].n);
            int last = 0;
            prof_begin(strdup(s));
            for (size_t q = 0; q < queries; q++) {
                int nn = db_search(db, data + (q % n) * dim, 10, res, M[m].t);
                if (nn > 0) { gv_search_results_free(res, (size_t)nn); last = nn; }
            }
            prof_end();
            if (last == 0) fprintf(stderr, "  metric %s unsupported on this data\n", M[m].n);
        }
        free(data);
        db_close(db);
    }
}

/* ---- persistence: save + reopen ------------------------------------------- */

static void phase_persistence(size_t n, size_t dim, size_t queries) {
    PROF_SCOPE("7.persist") {
        char path[512];
        snprintf(path, sizeof(path), "/tmp/gv_prof_%d.gvdb", (int)getpid());
        GV_Database *db = db_open(path, dim, GV_INDEX_TYPE_FLAT);
        if (!db) return;
        float *data = (float *)malloc(n * dim * sizeof(float));
        for (size_t i = 0; i < n * dim; i++) data[i] = next_unit();
        for (size_t i = 0; i < n; i++) if (db_add_vector(db, data + i * dim, dim)) {}

        PROF_SCOPE("persist/save") { if (db_save(db, path)) {} }
        db_close(db);

        GV_Database *db2 = NULL;
        PROF_SCOPE("persist/reopen") { db2 = db_open(path, dim, GV_INDEX_TYPE_FLAT); }
        if (db2) {
            GV_SearchResult res[16];
            PROF_SCOPE("persist/search") {
                for (size_t q = 0; q < queries; q++) {
                    int nn = db_search(db2, data + (q % n) * dim, 10, res, GV_DISTANCE_EUCLIDEAN);
                    if (nn > 0) gv_search_results_free(res, (size_t)nn);
                }
            }
            db_close(db2);
        }
        free(data);
        remove(path);
    }
}

/* ---- transactions: commit + rollback -------------------------------------- */

static void phase_transactions(size_t ops, size_t dim) {
    PROF_SCOPE("8.txn") {
        GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_FLAT);
        if (!db) return;
        float *v = (float *)malloc(dim * sizeof(float));

        PROF_SCOPE("txn/commit") {
            GV_DBTxn *tx = db_begin(db);
            if (tx) {
                for (size_t i = 0; i < ops; i++) {
                    for (size_t d = 0; d < dim; d++) v[d] = next_unit();
                    (void)db_txn_add_vector(tx, v, dim);
                }
                (void)db_commit(tx);
            }
        }
        PROF_SCOPE("txn/rollback") {
            GV_DBTxn *tx = db_begin(db);
            if (tx) {
                for (size_t i = 0; i < ops; i++) {
                    for (size_t d = 0; d < dim; d++) v[d] = next_unit();
                    (void)db_txn_add_vector(tx, v, dim);
                }
                (void)db_rollback(tx);
            }
        }
        free(v);
        db_close(db);
    }
}

/* ---- sparse index (raw API: not reachable via dense db_add_vector) -------- */

static void phase_sparse(size_t n, size_t dim, size_t nnz, size_t queries) {
    PROF_SCOPE("9.sparse") {
        GV_SparseIndex *idx = sparse_index_create(dim);
        if (!idx) return;
        uint32_t *ix = (uint32_t *)malloc(nnz * sizeof(uint32_t));
        float *vals = (float *)malloc(nnz * sizeof(float));

        PROF_SCOPE("sparse/build") {
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < nnz; j++) {
                    ix[j] = (uint32_t)(next_u64() % dim);
                    vals[j] = next_unit() + 0.5f;
                }
                GV_SparseVector *sv = sparse_vector_create(dim, ix, vals, nnz);
                if (sv && sparse_index_add(idx, sv) != 0) sparse_vector_destroy(sv);
            }
        }
        GV_SearchResult res[16];
        PROF_SCOPE("sparse/search") {
            for (size_t q = 0; q < queries; q++) {
                for (size_t j = 0; j < nnz; j++) {
                    ix[j] = (uint32_t)(next_u64() % dim);
                    vals[j] = next_unit() + 0.5f;
                }
                GV_SparseVector *qv = sparse_vector_create(dim, ix, vals, nnz);
                if (qv) {
                    int nn = sparse_index_search(idx, qv, 10, res, GV_DISTANCE_DOT_PRODUCT);
                    if (nn > 0) gv_search_results_free(res, (size_t)nn);
                    sparse_vector_destroy(qv);
                }
            }
        }
        free(ix); free(vals);
        sparse_index_destroy(idx);
    }
}

/* ---- disk-backed indexes (raw API, own on-disk storage) ------------------- */

static void phase_diskann(size_t n, size_t dim, size_t queries) {
    PROF_SCOPE("10.diskann") {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/gv_prof_diskann_%d.dat", (int)getpid());
        GV_DiskANNConfig cfg;
        diskann_config_init(&cfg);
        cfg.data_path = path; /* DiskANN wants a file path, not a directory */
        GV_DiskANNIndex *idx = diskann_create(dim, &cfg);
        if (!idx) return;

        float *data = (float *)malloc(n * dim * sizeof(float));
        for (size_t i = 0; i < n * dim; i++) data[i] = next_unit();

        int built = -1;
        PROF_SCOPE("diskann/build") { built = diskann_build(idx, data, n, dim); }
        if (built == 0) {
            GV_DiskANNResult res[16];
            PROF_SCOPE("diskann/search") {
                for (size_t q = 0; q < queries; q++)
                    (void)diskann_search(idx, data + (q % n) * dim, dim, 10, res);
            }
        }
        free(data);
        diskann_destroy(idx);
        remove(path);
    }
}

static void phase_ivfdisk(size_t n, size_t dim, size_t queries) {
    PROF_SCOPE("11.ivfdisk") {
        char dir[256];
        snprintf(dir, sizeof(dir), "/tmp/gv_prof_ivfdisk_%d", (int)getpid());
        mkdir(dir, 0700);
        GV_IVFDiskConfig cfg;
        ivfdisk_config_init(&cfg);
        cfg.nlist = 64;
        cfg.data_dir = dir;
        GV_IVFDiskIndex *idx = ivfdisk_create(dim, &cfg);
        if (!idx) return;

        float *data = (float *)malloc(n * dim * sizeof(float));
        for (size_t i = 0; i < n * dim; i++) data[i] = next_unit();

        PROF_SCOPE("ivfdisk/train") {
            (void)ivfdisk_train(idx, data, n / 2 < n ? (n / 2 ? n / 2 : n) : n);
        }
        PROF_SCOPE("ivfdisk/insert") {
            for (size_t i = 0; i < n; i++) (void)ivfdisk_insert(idx, data + i * dim, dim, i);
        }
        GV_SearchResult res[16];
        PROF_SCOPE("ivfdisk/search") {
            for (size_t q = 0; q < queries; q++) {
                int nn = ivfdisk_search(idx, data + (q % n) * dim, 10, res, GV_DISTANCE_EUCLIDEAN);
                if (nn > 0) gv_search_results_free(res, (size_t)nn);
            }
        }
        free(data);
        ivfdisk_destroy(idx);
    }
}

/* ---- REST server (end-to-end HTTP: parse -> dispatch -> engine -> JSON) --- */

/* Minimal blocking HTTP POST to localhost; reads and discards the response. */
static int http_post(uint16_t port, const char *path, const char *body) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) { close(fd); return -1; }
    char req[8192];
    int blen = (int)strlen(body);
    int n = snprintf(req, sizeof(req),
                     "POST %s HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                     "Content-Length: %d\r\nConnection: close\r\n\r\n%s",
                     path, blen, body);
    if (write(fd, req, (size_t)n) != n) { close(fd); return -1; }
    char buf[4096];
    while (read(fd, buf, sizeof(buf)) > 0) { /* drain response */ }
    close(fd);
    return 0;
}

static void phase_server(size_t n, size_t dim, size_t queries) {
    PROF_SCOPE("12.server") {
        GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_FLAT);
        if (!db) return;
        float *v = (float *)malloc(dim * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            for (size_t d = 0; d < dim; d++) v[d] = next_unit();
            if (db_add_vector(db, v, dim)) {}
        }

        GV_ServerConfig cfg;
        server_config_init(&cfg);
        cfg.port = 18080;
        cfg.enable_logging = 0;
        GV_Server *srv = server_create(db, &cfg);
        if (srv && server_start(srv) == 0) {
            uint16_t port = server_get_port(srv);

            /* Build a JSON search body for this dimension. */
            char body[8192];
            size_t off = (size_t)snprintf(body, sizeof(body), "{\"query\":[");
            for (size_t d = 0; d < dim && off + 24 < sizeof(body); d++)
                off += (size_t)snprintf(body + off, sizeof(body) - off, "%s%.4f", d ? "," : "", (double)next_unit());
            snprintf(body + off, sizeof(body) - off, "],\"k\":10}");

            PROF_SCOPE("server/search_http") {
                for (size_t q = 0; q < queries; q++)
                    (void)http_post(port, "/search", body);
            }
            PROF_SCOPE("server/health_http") {
                for (size_t q = 0; q < queries; q++)
                    (void)http_post(port, "/health", "");
            }
            server_stop(srv);
        } else {
            fprintf(stderr, "  server: not started (lib built without microhttpd) - skipped\n");
        }
        if (srv) server_destroy(srv);
        free(v);
        db_close(db);
    }
}

/* ---- replication: durable Raft log + in-process WAL transport ------------- */

static void phase_replication(size_t ops, size_t rec_len) {
    PROF_SCOPE("13.replication") {
        uint8_t *rec = (uint8_t *)malloc(rec_len);
        memset(rec, 0xAB, rec_len);

        /* Durable Raft log append (the replication durability hot path). */
        char path[256];
        snprintf(path, sizeof(path), "/tmp/gv_prof_raft_%d.log", (int)getpid());
        GV_RaftLog *log = raft_log_open(path);
        if (log) {
            PROF_SCOPE("repl/raft_log_append") {
                for (size_t i = 0; i < ops; i++)
                    (void)raft_log_append(log, i + 1, 1, rec, rec_len);
            }
            raft_log_close(log);
            remove(path);
        }

        /* In-process WAL replication transport (no network): enqueue to a
         * follower, then deliver each message. */
        GV_ReplSim *sim = repl_sim_create(42);
        if (sim) {
            const char *follower = "follower-1";
            PROF_SCOPE("repl/enqueue") {
                for (size_t i = 0; i < ops; i++)
                    (void)repl_sim_enqueue_wal(sim, follower, i + 1, rec, rec_len);
            }
            PROF_SCOPE("repl/deliver") {
                uint64_t idx;
                uint8_t *out;
                size_t olen;
                while (repl_sim_deliver_wal(sim, follower, &idx, &out, &olen) == 0)
                    gv_free(out); /* deliver transfers record ownership */
            }
            repl_sim_destroy(sim);
        }
        free(rec);
    }
}

/* ---- gRPC: in-process request dispatch (no HTTP/2 socket transport) -------
 *
 * grpc_fuzz_dispatch_message drives the full server-side request path -
 * dispatch -> protobuf-style payload decode -> engine op -> response encode ->
 * send_message framing to a real fd - bypassing only the TCP/recv_message
 * ingress (which is the kernel's job, not ours). The response fd is one end of
 * a socketpair, drained after each call so send_message never blocks. This is
 * the same harness the grpc fuzzer uses, so it exercises the real code path. */
static void phase_grpc(size_t n, size_t dim, size_t queries) {
    PROF_SCOPE("14.grpc") {
        GV_Database *db = db_open(NULL, dim, GV_INDEX_TYPE_FLAT);
        if (!db) return;
        float *v = (float *)malloc(dim * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            for (size_t d = 0; d < dim; d++) v[d] = next_unit();
            if (db_add_vector(db, v, dim)) {}
        }

        GV_GrpcConfig cfg;
        grpc_config_init(&cfg);
        GV_GrpcServer *srv = grpc_create(db, &cfg);
        int fds[2] = {-1, -1};
        if (srv && socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0) {
            /* Encode a SEARCH request payload once (same wire layout a client
             * sends): [dim u32][k u32][distance u32][dim * float32]. */
            size_t pbuf_sz = 12 + dim * sizeof(float);
            uint8_t *pbuf = (uint8_t *)malloc(pbuf_sz);
            uint8_t drain[4096];

            PROF_SCOPE("grpc/search_dispatch") {
                for (size_t q = 0; q < queries; q++) {
                    for (size_t d = 0; d < dim; d++) v[d] = next_unit();
                    size_t plen = 0;
                    if (grpc_encode_search_request(v, dim, 10, GV_DISTANCE_EUCLIDEAN,
                                                   pbuf, pbuf_sz, &plen) != GV_GRPC_OK)
                        continue;
                    GV_GrpcMessage msg;
                    memset(&msg, 0, sizeof(msg));
                    msg.msg_type = GV_MSG_SEARCH;
                    msg.request_id = (uint32_t)q;
                    msg.payload = pbuf;
                    msg.payload_len = plen;
                    msg.length = (uint32_t)(1 + 4 + plen);
                    (void)grpc_fuzz_dispatch_message(srv, fds[1], &msg);
                    while (recv(fds[0], drain, sizeof(drain), MSG_DONTWAIT) > 0) {}
                }
            }

            /* Cheap framing-only ops (health/stats): isolate the dispatch +
             * encode + send cost from the engine search cost above. */
            PROF_SCOPE("grpc/health_dispatch") {
                for (size_t q = 0; q < queries; q++) {
                    GV_GrpcMessage msg;
                    memset(&msg, 0, sizeof(msg));
                    msg.msg_type = GV_MSG_HEALTH;
                    msg.request_id = (uint32_t)q;
                    (void)grpc_fuzz_dispatch_message(srv, fds[1], &msg);
                    while (recv(fds[0], drain, sizeof(drain), MSG_DONTWAIT) > 0) {}
                }
            }

            free(pbuf);
            close(fds[0]);
            close(fds[1]);
        }
        if (srv) grpc_destroy(srv);
        free(v);
        db_close(db);
    }
}

int main(int argc, char **argv) {
    size_t n_vec   = argc > 1 ? strtoul(argv[1], NULL, 10) : 20000;
    size_t dim     = argc > 2 ? strtoul(argv[2], NULL, 10) : 128;
    size_t queries = argc > 3 ? strtoul(argv[3], NULL, 10) : 500;
    size_t n_docs  = argc > 4 ? strtoul(argv[4], NULL, 10) : 3000;
    size_t q_text  = argc > 5 ? strtoul(argv[5], NULL, 10) : 500;

    g_spans = (ProfSpan *)malloc(sizeof(ProfSpan) * PROF_MAX_SPANS);
    g_t0_ms = now_ms();

    printf("GigaVector e2e profiler\n");
    printf("  vectors=%zu dim=%zu queries=%zu docs=%zu text_queries=%zu\n",
           n_vec, dim, queries, n_docs, q_text);
    printf("  alloc tracking: %s\n", gv_alloc_stats_enabled() ? "on" : "off");

    /* The index matrix builds 10 indexes (HNSW build is superlinear), so scale
     * it down from the search corpus unless overridden. */
    size_t idx_n = argc > 6 ? strtoul(argv[6], NULL, 10) : (n_vec / 4 < 5000 ? n_vec / 4 : 5000);
    if (idx_n < 512) idx_n = 512;

    double wall0 = now_ms();
    phase_vector_db(n_vec, dim, queries);
    phase_graph_cypher(n_docs / 20 + 2, queries / 4 + 1);
    phase_splade(n_docs, q_text);
    phase_hybrid(n_docs, q_text);
    phase_index_matrix(idx_n, dim, queries);
    phase_metrics(idx_n, dim, queries);
    phase_persistence(idx_n, dim, queries);
    phase_transactions(1000, dim);
    phase_sparse(idx_n, dim < 1000 ? 1000 : dim, 16, queries);
    phase_diskann(idx_n, dim, queries);
    phase_ivfdisk(idx_n, dim, queries);
    phase_server(idx_n, dim, queries);
    phase_replication(idx_n * 2, 256);
    phase_grpc(idx_n, dim, queries);
    double wall = now_ms() - wall0;

    prof_report();
    printf("wall: %.2f ms   peak RSS: %.1f MB\n", wall, rss_kb() / 1024.0);

    const char *trace = getenv("GV_PROF_TRACE");
    if (trace && *trace) prof_write_trace(trace);

    free(g_spans);
    return 0;
}
