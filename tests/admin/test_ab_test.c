/**
 * @file test_ab_test.c
 * @brief Tests for index-level A/B testing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include "admin/ab_test.h"
#include "storage/database.h"
#include "core/compat.h"   /* gv_rand_r (portable PRNG) */

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
            return -1; \
        } \
    } while (0)

#define DIM 8
#define N_TRAIN 50
#define N_SEARCH 100

static void make_random_vec(float *v, size_t dim, unsigned int *seed) {
    for (size_t i = 0; i < dim; i++) {
        v[i] = ((float)(gv_rand_r(seed) % 10000) / 10000.0f) - 0.5f;
    }
}

static int test_ab_test_start_stop(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "db_open should succeed");
    ASSERT(db->ab_test == NULL, "ab_test should be NULL initially");

    int rc = gv_db_ab_test_start(db, "test1", GV_INDEX_TYPE_FLAT, 0.5f);
    ASSERT(rc == 0, "gv_db_ab_test_start should succeed");
    ASSERT(db->ab_test != NULL, "ab_test should be set after start");

    /* Starting a second test while one is active should fail. */
    int rc2 = gv_db_ab_test_start(db, "test2", GV_INDEX_TYPE_FLAT, 0.5f);
    ASSERT(rc2 == -1, "starting second test should fail");

    gv_db_ab_test_stop(db);
    ASSERT(db->ab_test == NULL, "ab_test should be NULL after stop");

    db_close(db);
    return 0;
}

static int test_ab_test_copies_vectors(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 42;
    float vec[DIM];
    for (int i = 0; i < N_TRAIN; i++) {
        make_random_vec(vec, DIM, &seed);
        ASSERT(db_add_vector(db, vec, DIM) == 0, "db_add_vector should succeed");
    }
    ASSERT(database_count(db) == N_TRAIN, "primary count should match inserts");

    int rc = gv_db_ab_test_start(db, "shadow_copy_test", GV_INDEX_TYPE_FLAT, 0.5f);
    ASSERT(rc == 0, "ab test start should succeed");

    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    ASSERT(test != NULL, "test struct should be set");
    ASSERT(test->shadow_db != NULL, "shadow db should exist");
    ASSERT(database_count(test->shadow_db) == N_TRAIN, "shadow count should match primary");

    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

static int test_ab_test_routing(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 99;
    float vec[DIM];
    for (int i = 0; i < N_TRAIN; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }

    /* Use 50% split. */
    int rc = gv_db_ab_test_start(db, "routing_test", GV_INDEX_TYPE_FLAT, 0.5f);
    ASSERT(rc == 0, "ab test start should succeed");

    /* Issue N_SEARCH searches and check the traffic split is approximate. */
    float query[DIM];
    GV_SearchResult results[5];
    for (int i = 0; i < N_SEARCH; i++) {
        make_random_vec(query, DIM, &seed);
        gv_db_ab_test_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
        /* Free result vectors. */
        gv_search_results_free(results, 5);
    }

    char report[512];
    int rep_rc = gv_db_ab_test_report(db, report, sizeof(report));
    ASSERT(rep_rc == 0, "report should succeed");

    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    ASSERT(test != NULL, "test should still be active");

    uint64_t total = test->queries_a + test->queries_b;
    ASSERT(total == N_SEARCH, "total routed queries should equal search count");

    /* With 50% split and 100 queries, expect between 30 and 70 each side. */
    ASSERT(test->queries_b >= 30 && test->queries_b <= 70,
           "shadow queries should be roughly 50% (30-70 range for 100 searches)");

    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

static int test_ab_test_report_fields(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 7;
    float vec[DIM];
    for (int i = 0; i < 10; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }

    int rc = gv_db_ab_test_start(db, "mytest", GV_INDEX_TYPE_FLAT, 0.3f);
    ASSERT(rc == 0, "ab test start should succeed");

    float query[DIM];
    GV_SearchResult results[3];
    for (int i = 0; i < 20; i++) {
        make_random_vec(query, DIM, &seed);
        gv_db_ab_test_search(db, query, 3, results, GV_DISTANCE_EUCLIDEAN);
        gv_search_results_free(results, 3);
    }

    char buf[1024];
    int r = gv_db_ab_test_report(db, buf, sizeof(buf));
    ASSERT(r == 0, "report should succeed");

    /* Verify required JSON fields are present. */
    ASSERT(strstr(buf, "\"name\"") != NULL, "report should contain name");
    ASSERT(strstr(buf, "mytest") != NULL, "report should contain test name value");
    ASSERT(strstr(buf, "\"queries_a\"") != NULL, "report should contain queries_a");
    ASSERT(strstr(buf, "\"queries_b\"") != NULL, "report should contain queries_b");
    ASSERT(strstr(buf, "\"avg_latency_a_us\"") != NULL, "report should contain avg_latency_a_us");
    ASSERT(strstr(buf, "\"avg_latency_b_us\"") != NULL, "report should contain avg_latency_b_us");
    ASSERT(strstr(buf, "\"traffic_split\"") != NULL, "report should contain traffic_split");

    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

static int test_ab_test_no_active_test(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    /* report with no active test should fail. */
    char buf[256];
    int rc = gv_db_ab_test_report(db, buf, sizeof(buf));
    ASSERT(rc == -1, "report with no active test should return -1");

    /* stop with no active test should be a no-op. */
    gv_db_ab_test_stop(db);

    db_close(db);
    return 0;
}

static int test_ab_test_zero_split(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 21;
    float vec[DIM];
    for (int i = 0; i < 10; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }

    /* 0% split: all traffic goes to A. */
    int rc = gv_db_ab_test_start(db, "zero_split", GV_INDEX_TYPE_FLAT, 0.0f);
    ASSERT(rc == 0, "start should succeed");

    float query[DIM];
    GV_SearchResult results[3];
    for (int i = 0; i < 20; i++) {
        make_random_vec(query, DIM, &seed);
        gv_db_ab_test_search(db, query, 3, results, GV_DISTANCE_EUCLIDEAN);
        gv_search_results_free(results, 3);
    }

    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    ASSERT(test->queries_b == 0, "zero split should route no traffic to B");
    ASSERT(test->queries_a == 20, "zero split should route all traffic to A");

    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

static int test_ab_test_full_split(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 55;
    float vec[DIM];
    for (int i = 0; i < 10; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }

    /* 100% split: all traffic goes to B. */
    int rc = gv_db_ab_test_start(db, "full_split", GV_INDEX_TYPE_FLAT, 1.0f);
    ASSERT(rc == 0, "start should succeed");

    float query[DIM];
    GV_SearchResult results[3];
    for (int i = 0; i < 20; i++) {
        make_random_vec(query, DIM, &seed);
        gv_db_ab_test_search(db, query, 3, results, GV_DISTANCE_EUCLIDEAN);
        gv_search_results_free(results, 3);
    }

    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    ASSERT(test->queries_a == 0, "full split should route no traffic to A");
    ASSERT(test->queries_b == 20, "full split should route all traffic to B");

    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

/* A=FLAT (exact), B=HNSW (approximate): both variants must serve queries and
 * return the requested k over a populated corpus. */
static int test_ab_test_heterogeneous(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 7;
    float vec[DIM];
    for (int i = 0; i < 100; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }

    int rc = gv_db_ab_test_start(db, "flat_vs_hnsw", GV_INDEX_TYPE_HNSW, 0.5f);
    ASSERT(rc == 0, "heterogeneous start should succeed");
    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    ASSERT(test->index_type_a == GV_INDEX_TYPE_FLAT, "A mirrors primary (FLAT)");
    ASSERT(test->index_type_b == GV_INDEX_TYPE_HNSW, "B is the shadow (HNSW)");
    ASSERT(database_count(test->shadow_db) == 100, "shadow HNSW copied all vectors");

    float query[DIM];
    GV_SearchResult results[5];
    int min_found = 5;
    for (int i = 0; i < 200; i++) {
        make_random_vec(query, DIM, &seed);
        int found = gv_db_ab_test_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
        if (found < min_found) min_found = found;
        if (found > 0) gv_search_results_free(results, (size_t)found);
    }
    ASSERT(test->queries_a > 0 && test->queries_b > 0, "both variants served traffic");
    ASSERT(test->queries_a + test->queries_b == 200, "all queries accounted for");
    ASSERT(min_found == 5, "every search returned k=5 from both indexes");

    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

/* The routing hash should track the configured split closely over a batch. */
static int test_ab_test_split_distribution(void) {
    const float splits[] = {0.25f, 0.5f, 0.75f};
    for (size_t s = 0; s < sizeof(splits) / sizeof(splits[0]); s++) {
        GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
        ASSERT(db != NULL, "db_open should succeed");
        unsigned int seed = 99 + (unsigned)s;
        float vec[DIM];
        for (int i = 0; i < 10; i++) {
            make_random_vec(vec, DIM, &seed);
            { int _r = db_add_vector(db, vec, DIM); (void)_r; }
        }
        ASSERT(gv_db_ab_test_start(db, "split", GV_INDEX_TYPE_FLAT, splits[s]) == 0,
               "start should succeed");
        float query[DIM];
        GV_SearchResult results[3];
        const int total = 3000;
        for (int i = 0; i < total; i++) {
            make_random_vec(query, DIM, &seed);
            int f = gv_db_ab_test_search(db, query, 3, results, GV_DISTANCE_EUCLIDEAN);
            if (f > 0) gv_search_results_free(results, (size_t)f);
        }
        GV_ABTest *test = (GV_ABTest *)db->ab_test;
        double frac_b = (double)test->queries_b / (double)total;
        ASSERT(fabs(frac_b - (double)splits[s]) < 0.07, "observed split within tolerance");
        gv_db_ab_test_stop(db);
        db_close(db);
    }
    return 0;
}

/* Latency sums accumulate and the report's averages are finite and non-negative. */
static int test_ab_test_latency_recorded(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");
    unsigned int seed = 3;
    float vec[DIM];
    for (int i = 0; i < 40; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }
    ASSERT(gv_db_ab_test_start(db, "lat", GV_INDEX_TYPE_FLAT, 0.5f) == 0, "start ok");
    float query[DIM];
    GV_SearchResult results[3];
    for (int i = 0; i < 100; i++) {
        make_random_vec(query, DIM, &seed);
        int f = gv_db_ab_test_search(db, query, 3, results, GV_DISTANCE_EUCLIDEAN);
        if (f > 0) gv_search_results_free(results, (size_t)f);
    }
    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    ASSERT(test->latency_sum_a >= 0.0 && test->latency_sum_b >= 0.0, "latency sums non-negative");
    ASSERT(test->creation_time_us > 0, "creation timestamp recorded");

    char buf[512];
    ASSERT(gv_db_ab_test_report(db, buf, sizeof(buf)) == 0, "report ok");
    ASSERT(strstr(buf, "\"creation_time_us\"") != NULL, "report has creation_time_us");
    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

/* A stopped test can be restarted; a too-small report buffer fails cleanly. */
static int test_ab_test_restart_and_small_buffer(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");
    unsigned int seed = 11;
    float vec[DIM];
    for (int i = 0; i < 10; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }
    ASSERT(gv_db_ab_test_start(db, "first", GV_INDEX_TYPE_FLAT, 0.5f) == 0, "first start ok");

    char tiny[5];
    ASSERT(gv_db_ab_test_report(db, tiny, sizeof(tiny)) == -1, "too-small buffer -> -1");

    gv_db_ab_test_stop(db);
    ASSERT(db->ab_test == NULL, "stopped");
    ASSERT(gv_db_ab_test_start(db, "second", GV_INDEX_TYPE_HNSW, 0.3f) == 0, "restart ok");
    ASSERT(db->ab_test != NULL, "restarted test active");
    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

/* Names longer than the fixed buffer are truncated, not overflowed. */
static int test_ab_test_name_truncation(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");
    char longname[200];
    memset(longname, 'x', sizeof(longname) - 1);
    longname[sizeof(longname) - 1] = '\0';
    ASSERT(gv_db_ab_test_start(db, longname, GV_INDEX_TYPE_FLAT, 0.5f) == 0, "start ok");
    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    ASSERT(strlen(test->name) == sizeof(test->name) - 1, "name truncated to buffer-1");
    ASSERT(strncmp(test->name, longname, sizeof(test->name) - 1) == 0, "prefix preserved");
    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

/* With no active test, the routed search transparently falls back to primary. */
static int test_ab_test_search_fallback(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");
    unsigned int seed = 44;
    float vec[DIM];
    for (int i = 0; i < 20; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }
    float query[DIM];
    make_random_vec(query, DIM, &seed);
    GV_SearchResult ab[5], plain[5];
    int na = gv_db_ab_test_search(db, query, 5, ab, GV_DISTANCE_EUCLIDEAN);
    int np = db_search(db, query, 5, plain, GV_DISTANCE_EUCLIDEAN);
    ASSERT(na == np && na == 5, "fallback matches plain db_search count");
    ASSERT(ab[0].id == plain[0].id, "fallback returns same top hit");
    if (na > 0) gv_search_results_free(ab, (size_t)na);
    if (np > 0) gv_search_results_free(plain, (size_t)np);
    db_close(db);
    return 0;
}

/* Starting a test on an empty collection is safe; searches return nothing. */
static int test_ab_test_empty_db(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");
    ASSERT(gv_db_ab_test_start(db, "empty", GV_INDEX_TYPE_FLAT, 0.5f) == 0, "start on empty ok");
    GV_ABTest *test = (GV_ABTest *)db->ab_test;
    ASSERT(database_count(test->shadow_db) == 0, "empty shadow");
    float query[DIM];
    unsigned int seed = 1;
    make_random_vec(query, DIM, &seed);
    GV_SearchResult results[3];
    int f = gv_db_ab_test_search(db, query, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(f == 0, "search on empty returns 0");
    gv_db_ab_test_stop(db);
    db_close(db);
    return 0;
}

/* Routing depends only on the counter and split, so two independent tests with
 * the same split produce an identical routing sequence (deterministic). */
static int test_ab_test_routing_deterministic(void) {
    GV_Database *d1 = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    GV_Database *d2 = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(d1 && d2, "db_open should succeed");
    ASSERT(gv_db_ab_test_start(d1, "det1", GV_INDEX_TYPE_FLAT, 0.5f) == 0, "start d1");
    ASSERT(gv_db_ab_test_start(d2, "det2", GV_INDEX_TYPE_FLAT, 0.5f) == 0, "start d2");
    int diffs = 0;
    for (int i = 0; i < 500; i++) {
        if (ab_test_route(d1) != ab_test_route(d2)) diffs++;
    }
    ASSERT(diffs == 0, "same split yields identical routing sequence");
    gv_db_ab_test_stop(d1);
    gv_db_ab_test_stop(d2);
    db_close(d1);
    db_close(d2);
    return 0;
}

typedef struct {
    GV_Database *db;
    int iters;
    volatile int *stop;
} ABWorker;

static void *ab_search_worker(void *arg) {
    ABWorker *w = (ABWorker *)arg;
    unsigned int seed = 123;
    float query[DIM];
    GV_SearchResult results[5];
    for (int i = 0; i < w->iters && !*w->stop; i++) {
        make_random_vec(query, DIM, &seed);
        int f = gv_db_ab_test_search(w->db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
        if (f > 0) gv_search_results_free(results, (size_t)f);
    }
    return NULL;
}

/* Concurrent searches while the test is stopped must drain in-flight work before
 * the shadow is freed (the UAF guard) - clean under TSAN/ASan. */
static int test_ab_test_concurrent_stop(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");
    unsigned int seed = 77;
    float vec[DIM];
    for (int i = 0; i < 100; i++) {
        make_random_vec(vec, DIM, &seed);
        { int _r = db_add_vector(db, vec, DIM); (void)_r; }
    }
    ASSERT(gv_db_ab_test_start(db, "concurrent", GV_INDEX_TYPE_HNSW, 0.5f) == 0, "start ok");

    volatile int stop = 0;
    const int NT = 4;
    pthread_t th[4];
    ABWorker w[4];
    for (int i = 0; i < NT; i++) {
        w[i] = (ABWorker){db, 100000, &stop};
        ASSERT(pthread_create(&th[i], NULL, ab_search_worker, &w[i]) == 0, "spawn worker");
    }
    /* Let searches get in flight, then stop concurrently. */
    gv_db_ab_test_stop(db);
    stop = 1;
    for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
    ASSERT(db->ab_test == NULL, "test detached after stop");

    /* Searches after stop fall back to primary without crashing. */
    float query[DIM];
    make_random_vec(query, DIM, &seed);
    GV_SearchResult results[5];
    int f = gv_db_ab_test_search(db, query, 5, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(f >= 0, "post-stop search is safe");
    if (f > 0) gv_search_results_free(results, (size_t)f);
    db_close(db);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"test_ab_test_start_stop",       test_ab_test_start_stop},
        {"test_ab_test_copies_vectors",   test_ab_test_copies_vectors},
        {"test_ab_test_routing",          test_ab_test_routing},
        {"test_ab_test_report_fields",    test_ab_test_report_fields},
        {"test_ab_test_no_active_test",   test_ab_test_no_active_test},
        {"test_ab_test_zero_split",       test_ab_test_zero_split},
        {"test_ab_test_full_split",       test_ab_test_full_split},
        {"test_ab_test_heterogeneous",    test_ab_test_heterogeneous},
        {"test_ab_test_split_distribution", test_ab_test_split_distribution},
        {"test_ab_test_latency_recorded", test_ab_test_latency_recorded},
        {"test_ab_test_restart_and_small_buffer", test_ab_test_restart_and_small_buffer},
        {"test_ab_test_name_truncation",  test_ab_test_name_truncation},
        {"test_ab_test_search_fallback",  test_ab_test_search_fallback},
        {"test_ab_test_empty_db",         test_ab_test_empty_db},
        {"test_ab_test_routing_deterministic", test_ab_test_routing_deterministic},
        {"test_ab_test_concurrent_stop",  test_ab_test_concurrent_stop},
    };
    int n = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;
    for (int i = 0; i < n; i++) {
        fprintf(stdout, "Running %s ... ", tests[i].name);
        fflush(stdout);
        int r = tests[i].fn();
        if (r == 0) {
            fprintf(stdout, "OK\n");
            passed++;
        } else {
            fprintf(stdout, "FAILED\n");
        }
    }
    fprintf(stdout, "%d/%d passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}
