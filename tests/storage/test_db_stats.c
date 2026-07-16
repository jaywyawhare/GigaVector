#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "storage/database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 4

static GV_Database *make_db(void) {
    return db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
}

/* Latency histogram uses logarithmic buckets with boundaries pow(10, i+1):
   bucket 0 <= 10us, bucket 1 <= 100us, bucket 2 <= 1000us, ... (8 buckets). */
static int test_record_latency_buckets(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    /* Insert-path samples. */
    db_record_latency(db, 5, 1);     /* -> bucket 0 (<= 10us) */
    db_record_latency(db, 5, 1);     /* -> bucket 0 */
    db_record_latency(db, 50, 1);    /* -> bucket 1 (<= 100us) */

    /* Search-path samples. */
    db_record_latency(db, 500, 0);   /* -> bucket 2 (<= 1000us) */

    GV_DetailedStats stats;
    ASSERT(db_get_detailed_stats(db, &stats) == 0, "detailed stats should succeed");

    ASSERT(stats.insert_latency.buckets != NULL, "insert histogram allocated");
    ASSERT(stats.insert_latency.bucket_count == 8, "insert histogram has 8 buckets");
    ASSERT(stats.insert_latency.total_samples == 3, "3 insert samples recorded");
    ASSERT(stats.insert_latency.sum_latency_us == 60, "insert latency sum is 60us");
    ASSERT(stats.insert_latency.buckets[0] == 2, "two insert samples in bucket 0");
    ASSERT(stats.insert_latency.buckets[1] == 1, "one insert sample in bucket 1");

    ASSERT(stats.search_latency.buckets != NULL, "search histogram allocated");
    ASSERT(stats.search_latency.total_samples == 1, "1 search sample recorded");
    ASSERT(stats.search_latency.sum_latency_us == 500, "search latency sum is 500us");
    ASSERT(stats.search_latency.buckets[2] == 1, "one search sample in bucket 2");

    db_free_detailed_stats(&stats);
    /* Freeing twice must be safe (idempotent). */
    db_free_detailed_stats(&stats);
    db_free_detailed_stats(NULL);

    db_close(db);
    return 0;
}

static int test_detailed_stats_null_args(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    GV_DetailedStats stats;
    ASSERT(db_get_detailed_stats(NULL, &stats) == -1, "NULL db should fail");
    ASSERT(db_get_detailed_stats(db, NULL) == -1, "NULL out should fail");

    db_close(db);
    return 0;
}

static int test_detailed_stats_no_latency(void) {
    /* With no recorded latency, histograms are absent but the call still works. */
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    GV_DetailedStats stats;
    ASSERT(db_get_detailed_stats(db, &stats) == 0, "detailed stats should succeed");
    ASSERT(stats.insert_latency.buckets == NULL, "no insert histogram without samples");
    ASSERT(stats.search_latency.buckets == NULL, "no search histogram without samples");
    ASSERT(stats.health_status == 0, "empty db is healthy");
    ASSERT(stats.deleted_vector_count == 0, "empty db has no deleted vectors");

    db_free_detailed_stats(&stats);
    db_close(db);
    return 0;
}

static int test_health_check(void) {
    /* NULL db is unhealthy. */
    ASSERT(db_health_check(NULL) == -2, "NULL db is unhealthy");

    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    /* Fresh db with no deletions is healthy. */
    ASSERT(db_health_check(db) == 0, "fresh db is healthy");

    /* Add vectors; still healthy (no deletions). */
    for (int i = 0; i < 10; i++) {
        float v[DIM] = {(float)i, 0.0f, 0.0f, 0.0f};
        db_add_vector(db, v, DIM);
    }
    ASSERT(db_health_check(db) == 0, "db with only live vectors is healthy");

    db_close(db);
    return 0;
}

static int test_basic_stats_reflected(void) {
    GV_Database *db = make_db();
    ASSERT(db != NULL, "make_db should succeed");

    for (int i = 0; i < 5; i++) {
        float v[DIM] = {(float)i, 1.0f, 2.0f, 3.0f};
        db_add_vector(db, v, DIM);
    }

    GV_DetailedStats stats;
    ASSERT(db_get_detailed_stats(db, &stats) == 0, "detailed stats should succeed");
    ASSERT(stats.basic_stats.total_inserts == 5, "5 inserts reflected in basic stats");

    db_free_detailed_stats(&stats);
    db_close(db);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing latency histogram buckets...",   test_record_latency_buckets},
        {"Testing detailed stats null args...",    test_detailed_stats_null_args},
        {"Testing detailed stats no latency...",   test_detailed_stats_no_latency},
        {"Testing db health check...",             test_health_check},
        {"Testing basic stats reflected...",       test_basic_stats_reflected},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
