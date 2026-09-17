/**
 * @file test_ab_test.c
 * @brief Tests for index-level A/B testing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* -------------------------------------------------------------------------
 * test_ab_test_start_stop
 * ---------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * test_ab_test_copies_vectors
 * ---------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * test_ab_test_routing
 * ---------------------------------------------------------------------- */
static int test_ab_test_routing(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_HNSW);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 99;
    float vec[DIM];
    for (int i = 0; i < N_TRAIN; i++) {
        make_random_vec(vec, DIM, &seed);
        db_add_vector(db, vec, DIM);
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

/* -------------------------------------------------------------------------
 * test_ab_test_report_fields
 * ---------------------------------------------------------------------- */
static int test_ab_test_report_fields(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 7;
    float vec[DIM];
    for (int i = 0; i < 10; i++) {
        make_random_vec(vec, DIM, &seed);
        db_add_vector(db, vec, DIM);
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

/* -------------------------------------------------------------------------
 * test_ab_test_no_active_test
 * ---------------------------------------------------------------------- */
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

/* -------------------------------------------------------------------------
 * test_ab_test_zero_split
 * ---------------------------------------------------------------------- */
static int test_ab_test_zero_split(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 21;
    float vec[DIM];
    for (int i = 0; i < 10; i++) {
        make_random_vec(vec, DIM, &seed);
        db_add_vector(db, vec, DIM);
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

/* -------------------------------------------------------------------------
 * test_ab_test_full_split
 * ---------------------------------------------------------------------- */
static int test_ab_test_full_split(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    unsigned int seed = 55;
    float vec[DIM];
    for (int i = 0; i < 10; i++) {
        make_random_vec(vec, DIM, &seed);
        db_add_vector(db, vec, DIM);
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

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */

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
