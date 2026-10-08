/*
 * test_tiered_storage.c - Unit tests for the tiered hot/warm/cold storage feature.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

#include "storage/database.h"
#include "storage/tiered_storage.h"
#include "schema/vector.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
            return -1; \
        } \
    } while (0)

static GV_Database *make_db(size_t dim) {
    return db_open(NULL, dim, GV_INDEX_TYPE_FLAT);
}

static int insert_n(GV_Database *db, size_t n, size_t dim) {
    float *v = (float *)calloc(dim, sizeof(float));
    if (!v) return -1;
    for (size_t i = 0; i < n; ++i) {
        v[0] = (float)i;
        if (db_add_vector(db, v, dim) != 0) {
            free(v);
            return -1;
        }
    }
    free(v);
    return 0;
}

static int test_manager_lifecycle(void) {
    GV_TieredStorageManager *mgr = tiered_storage_create(16);
    ASSERT(mgr != NULL, "tiered_storage_create returns non-NULL");

    /* Record an insert timestamp */
    ASSERT(tiered_storage_record_insert(mgr, 0, 1000000ULL) == 0,
           "record_insert slot 0");
    ASSERT(tiered_storage_record_insert(mgr, 15, 2000000ULL) == 0,
           "record_insert slot 15 (capacity boundary)");

    /* Force grow beyond initial capacity */
    ASSERT(tiered_storage_record_insert(mgr, 100, 3000000ULL) == 0,
           "record_insert slot 100 triggers grow");

    tiered_storage_destroy(mgr);
    tiered_storage_destroy(NULL); /* safe with NULL */
    return 0;
}

static int test_set_insert_time(void) {
    GV_TieredStorageManager *mgr = tiered_storage_create(4);
    ASSERT(mgr != NULL, "create");

    ASSERT(tiered_storage_record_insert(mgr, 0, 500ULL) == 0, "record slot 0");
    ASSERT(tiered_storage_set_insert_time(mgr, 0, 999ULL) == 0, "set slot 0");

    /* Slot out of range should fail */
    ASSERT(tiered_storage_set_insert_time(mgr, 100, 1ULL) == -1,
           "out-of-range slot should return -1");

    tiered_storage_destroy(mgr);
    return 0;
}

static int test_tier_classification(void) {
    const size_t DIM = 4;
    GV_Database *db = make_db(DIM);
    ASSERT(db != NULL, "create db");

    /* Insert 3 vectors */
    ASSERT(insert_n(db, 3, DIM) == 0, "insert 3 vectors");

    /*
     * Configure tiering: hot_max_age = 1 s, warm_max_age = 5 s.
     * Then manually backdate the insertion timestamps to test classification.
     *
     * Vector 0: inserted 10 seconds ago  -> COLD
     * Vector 1: inserted  3 seconds ago  -> WARM
     * Vector 2: inserted  0 seconds ago  -> HOT (timestamp = now)
     */
    ASSERT(gv_db_set_tiering_config(db, 1, 5, 0) == 0, "set tiering config");
    ASSERT(db->tiering_enabled != 0, "tiering_enabled flag set");
    ASSERT(db->tiered_storage != NULL, "tiered_storage created");

    /* Get current time in us and backdate */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t now_us = (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;

    ASSERT(tiered_storage_set_insert_time(db->tiered_storage, 0,
                                           now_us - 10000000ULL) == 0,
           "backdate vec 0 by 10s");
    ASSERT(tiered_storage_set_insert_time(db->tiered_storage, 1,
                                           now_us - 3000000ULL) == 0,
           "backdate vec 1 by 3s");
    ASSERT(tiered_storage_set_insert_time(db->tiered_storage, 2, now_us) == 0,
           "keep vec 2 at now");

    GV_StorageTier tier;

    ASSERT(gv_db_get_vector_tier(db, 0, &tier) == 0, "get tier for vec 0");
    ASSERT(tier == GV_TIER_COLD, "vec 0 should be COLD (age=10s > warm=5s)");

    ASSERT(gv_db_get_vector_tier(db, 1, &tier) == 0, "get tier for vec 1");
    ASSERT(tier == GV_TIER_WARM, "vec 1 should be WARM (age=3s > hot=1s, < warm=5s)");

    ASSERT(gv_db_get_vector_tier(db, 2, &tier) == 0, "get tier for vec 2");
    ASSERT(tier == GV_TIER_HOT, "vec 2 should be HOT (age~0)");

    db_close(db);
    return 0;
}

static int test_tiering_stats(void) {
    const size_t DIM = 4;
    GV_Database *db = make_db(DIM);
    ASSERT(db != NULL, "create db");

    ASSERT(insert_n(db, 4, DIM) == 0, "insert 4 vectors");
    ASSERT(gv_db_set_tiering_config(db, 1, 5, 0) == 0, "configure tiering");

    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t now_us = (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;

    /* 2 HOT, 1 WARM, 1 COLD */
    tiered_storage_set_insert_time(db->tiered_storage, 0, now_us);
    tiered_storage_set_insert_time(db->tiered_storage, 1, now_us);
    tiered_storage_set_insert_time(db->tiered_storage, 2, now_us - 3000000ULL);
    tiered_storage_set_insert_time(db->tiered_storage, 3, now_us - 10000000ULL);

    size_t h = 0, w = 0, c = 0;
    ASSERT(gv_db_tiering_stats(db, &h, &w, &c) == 0, "get tiering stats");
    ASSERT(h == 2, "hot count == 2");
    ASSERT(w == 1, "warm count == 1");
    ASSERT(c == 1, "cold count == 1");

    db_close(db);
    return 0;
}

static int test_tiering_disabled(void) {
    const size_t DIM = 4;
    GV_Database *db = make_db(DIM);
    ASSERT(db != NULL, "create db");

    ASSERT(insert_n(db, 2, DIM) == 0, "insert 2 vectors");

    /* Tiering is NOT enabled */
    GV_StorageTier tier;
    ASSERT(gv_db_get_vector_tier(db, 0, &tier) == -1,
           "get_vector_tier returns -1 when tiering disabled");

    size_t h, w, c;
    ASSERT(gv_db_tiering_stats(db, &h, &w, &c) == -1,
           "tiering_stats returns -1 when tiering disabled");

    db_close(db);
    return 0;
}

static int test_insert_records_timestamp(void) {
    const size_t DIM = 4;
    GV_Database *db = make_db(DIM);
    ASSERT(db != NULL, "create db");

    /* Enable tiering BEFORE inserting */
    ASSERT(gv_db_set_tiering_config(db, 60, 3600, 0) == 0,
           "set tiering config");

    float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT(db_add_vector(db, v, DIM) == 0, "insert vector");

    /* The first vector (slot 0) should have a non-zero timestamp. */
    GV_StorageTier tier;
    ASSERT(gv_db_get_vector_tier(db, 0, &tier) == 0,
           "get tier for slot 0");
    /* Just inserted, age is < 60s, so must be HOT */
    ASSERT(tier == GV_TIER_HOT, "fresh vector is HOT");

    db_close(db);
    return 0;
}

static int test_db_close_cleanup(void) {
    const size_t DIM = 4;
    GV_Database *db = make_db(DIM);
    ASSERT(db != NULL, "create db");
    ASSERT(gv_db_set_tiering_config(db, 1, 5, 0) == 0, "set tiering config");
    ASSERT(insert_n(db, 10, DIM) == 0, "insert 10 vectors");
    /* db_close should not crash */
    db_close(db);
    return 0;
}

static uint64_t now_us_(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/* Phase 4: a recently-accessed but old vector is promoted back to HOT. */
static int test_recency_promotion(void) {
    const size_t DIM = 4;
    GV_Database *db = make_db(DIM);
    ASSERT(db != NULL, "create db");
    ASSERT(insert_n(db, 2, DIM) == 0, "insert 2 vectors");
    ASSERT(gv_db_set_tiering_config(db, 1, 5, 0) == 0, "age thresholds hot=1s warm=5s");
    ASSERT(gv_db_set_access_tiering_policy(db, 30, 0) == 0, "recency window 30s");

    uint64_t now = now_us_();
    /* Both vectors are old enough to be COLD by age alone. */
    tiered_storage_set_insert_time(db->tiered_storage, 0, now - 10000000ULL);
    tiered_storage_set_insert_time(db->tiered_storage, 1, now - 10000000ULL);

    GV_StorageTier tier;
    ASSERT(gv_db_get_vector_tier(db, 0, &tier) == 0 && tier == GV_TIER_COLD, "vec 0 COLD by age");

    /* Access vec 0 now -> recency promotion to HOT; vec 1 untouched stays COLD. */
    ASSERT(gv_db_record_vector_access(db, 0) == 0, "record access vec 0");
    ASSERT(gv_db_get_vector_tier(db, 0, &tier) == 0 && tier == GV_TIER_HOT, "vec 0 promoted HOT by recency");
    ASSERT(gv_db_get_vector_tier(db, 1, &tier) == 0 && tier == GV_TIER_COLD, "vec 1 still COLD");

    db_close(db);
    return 0;
}

/* Phase 4: a frequently-accessed vector is promoted one tier warmer than its age. */
static int test_frequency_promotion(void) {
    const size_t DIM = 4;
    GV_Database *db = make_db(DIM);
    ASSERT(db != NULL, "create db");
    ASSERT(insert_n(db, 2, DIM) == 0, "insert 2 vectors");
    ASSERT(gv_db_set_tiering_config(db, 1, 5, 0) == 0, "age thresholds hot=1s warm=5s");
    /* Frequency threshold 3; NO recency window (so recency doesn't mask the test). */
    ASSERT(gv_db_set_access_tiering_policy(db, 0, 3) == 0, "freq threshold 3, no recency");

    uint64_t now = now_us_();
    tiered_storage_set_insert_time(db->tiered_storage, 0, now - 10000000ULL); /* COLD by age */
    tiered_storage_set_insert_time(db->tiered_storage, 1, now - 10000000ULL);

    /* Record 3 accesses but backdate last-access outside any recency window (there is none). */
    for (int i = 0; i < 3; i++) ASSERT(gv_db_record_vector_access(db, 0) == 0, "access vec 0");

    GV_StorageTier tier;
    ASSERT(gv_db_get_vector_tier(db, 0, &tier) == 0 && tier == GV_TIER_WARM,
           "vec 0 COLD->WARM by frequency (>=3 accesses)");
    ASSERT(gv_db_get_vector_tier(db, 1, &tier) == 0 && tier == GV_TIER_COLD, "vec 1 stays COLD");

    uint32_t cnt = 0; uint64_t last = 0;
    ASSERT(tiered_storage_get_access(db->tiered_storage, 0, &cnt, &last) == 0 && cnt == 3 && last > 0,
           "access stats recorded (count=3)");

    db_close(db);
    return 0;
}

/* Phase 4: db_search records accesses for returned vectors (read-path hook). */
static int test_search_records_access(void) {
    const size_t DIM = 4;
    GV_Database *db = make_db(DIM);
    ASSERT(db != NULL, "create db");
    ASSERT(gv_db_set_tiering_config(db, 3600, 86400, 0) == 0, "enable tiering");
    ASSERT(gv_db_set_access_tiering_policy(db, 60, 0) == 0, "recency 60s");
    ASSERT(insert_n(db, 5, DIM) == 0, "insert 5 vectors");

    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GV_SearchResult results[3];
    int n = db_search(db, q, 3, results, GV_DISTANCE_EUCLIDEAN);
    ASSERT(n > 0, "search returns hits");

    /* At least one returned vector now has a non-zero access count. */
    int any = 0;
    for (int i = 0; i < n; i++) {
        uint32_t cnt = 0;
        if (tiered_storage_get_access(db->tiered_storage, results[i].id, &cnt, NULL) == 0 && cnt > 0) any = 1;
        if (results[i].vector) vector_destroy((GV_Vector *)results[i].vector);
    }
    ASSERT(any, "search recorded at least one access");

    db_close(db);
    return 0;
}

typedef struct {
    const char *name;
    int (*fn)(void);
} Test;

int main(void) {
    Test tests[] = {
        {"manager_lifecycle",       test_manager_lifecycle},
        {"set_insert_time",         test_set_insert_time},
        {"tier_classification",     test_tier_classification},
        {"tiering_stats",           test_tiering_stats},
        {"tiering_disabled",        test_tiering_disabled},
        {"insert_records_timestamp",test_insert_records_timestamp},
        {"db_close_cleanup",        test_db_close_cleanup},
        {"recency_promotion",       test_recency_promotion},
        {"frequency_promotion",     test_frequency_promotion},
        {"search_records_access",   test_search_records_access},
    };
    size_t ntests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0, failed = 0;

    for (size_t i = 0; i < ntests; ++i) {
        int rc = tests[i].fn();
        if (rc == 0) {
            printf("PASS: %s\n", tests[i].name);
            ++passed;
        } else {
            printf("FAIL: %s\n", tests[i].name);
            ++failed;
        }
    }
    printf("\n%d/%zu passed\n", passed, (size_t)ntests);
    return (failed == 0) ? 0 : 1;
}
