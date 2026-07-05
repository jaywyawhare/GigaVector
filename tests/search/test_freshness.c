/**
 * @file test_freshness.c
 * @brief Tests for Vector Freshness Scoring.
 *
 * Inserts "old" and "new" vectors with the same embedding, then verifies that
 * a freshness-weighted search ranks the newer vectors higher.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "search/freshness.h"
#include "storage/database.h"
#include "storage/soa_storage.h"
#include "core/sim_time.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
            return -1; \
        } \
    } while (0)

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------ */

static float vec[4] = {1.0f, 0.0f, 0.0f, 0.0f};

/* ---------------------------------------------------------------------------
 * Test: freshness_score() — basic decay math
 * ------------------------------------------------------------------------ */

static int test_freshness_score_exp(void)
{
    /* A vector inserted at time 0 with now = half_life should score ~0.5 */
    double half_life_ms = 1000.0; /* 1 second */
    float score = freshness_score(0, (uint64_t)half_life_ms,
                                  half_life_ms, GV_RANK_DECAY_EXP);
    ASSERT(fabsf(score - 0.5f) < 0.01f, "exp decay at 1 half-life should be ~0.5");

    /* A vector inserted right now should score ~1.0 */
    float score_new = freshness_score(1000, 1000, half_life_ms, GV_RANK_DECAY_EXP);
    ASSERT(fabsf(score_new - 1.0f) < 0.01f, "freshness at age 0 should be ~1.0");

    return 0;
}

static int test_freshness_score_gauss(void)
{
    double half_life_ms = 1000.0;
    float score = freshness_score(0, (uint64_t)half_life_ms,
                                  half_life_ms, GV_RANK_DECAY_GAUSS);
    ASSERT(score > 0.0f && score <= 1.0f, "gauss score must be in (0,1]");

    float score_new = freshness_score(500, 500, half_life_ms, GV_RANK_DECAY_GAUSS);
    ASSERT(fabsf(score_new - 1.0f) < 0.01f, "gauss freshness at age 0 should be ~1.0");
    return 0;
}

static int test_freshness_score_linear(void)
{
    double half_life_ms = 1000.0;
    float score_new = freshness_score(0, 0, half_life_ms, GV_RANK_DECAY_LINEAR);
    ASSERT(fabsf(score_new - 1.0f) < 0.01f, "linear freshness at age 0 should be ~1.0");

    /* Very old vector should score 0 */
    float score_old = freshness_score(0, 1000000, half_life_ms, GV_RANK_DECAY_LINEAR);
    ASSERT(score_old == 0.0f, "very old vector should score 0 with linear decay");
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test: freshness_blend()
 * ------------------------------------------------------------------------ */

static int test_freshness_blend(void)
{
    float blended = freshness_blend(0.8f, 0.6f, 0.7f, 0.3f);
    /* 0.7*0.8 + 0.3*0.6 = 0.56 + 0.18 = 0.74 */
    ASSERT(fabsf(blended - 0.74f) < 0.001f, "blend calculation should be correct");
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test: db_search_with_freshness() — newer vectors rank higher
 * ------------------------------------------------------------------------ */

static int test_newer_vectors_rank_higher(void)
{
    /* Use simulated time for deterministic results. */
    gv_sim_time_set_mode(GV_TIME_SIM);
    gv_sim_time_reset(1700000000ULL);

    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    /* Insert 5 "old" vectors at t=0 (same direction as query). */
    for (int i = 0; i < 5; ++i) {
        float v[4] = {1.0f, (float)i * 0.01f, 0.0f, 0.0f};
        ASSERT(db_add_vector(db, v, 4) == 0, "inserting old vector should succeed");
    }

    /* Advance clock by 1 day (86400 seconds). */
    gv_sim_time_advance_sec(86400ULL);

    /* Insert 5 "new" vectors at t=1day (same direction as query). */
    for (int i = 0; i < 5; ++i) {
        float v[4] = {1.0f, (float)i * 0.01f, 0.0f, 0.0f};
        ASSERT(db_add_vector(db, v, 4) == 0, "inserting new vector should succeed");
    }

    /* Advance a tiny bit so now > insert time. */
    gv_sim_time_advance_ms(1);

    GV_FreshnessParams params = {
        .score_weight     = 0.3,
        .freshness_weight = 0.7,
        .half_life_sec    = 43200.0, /* 12 hours */
        .decay_type       = GV_RANK_DECAY_EXP
    };

    float query[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_FreshnessResult results[10];
    int n = db_search_with_freshness(db, query, 10, &params, results);
    ASSERT(n == 10, "should return 10 results");

    /* New vectors (index >= 5) should rank before old vectors (index < 5)
       because freshness_weight is high. */
    int first_new_pos = -1;
    int first_old_pos = -1;
    for (int i = 0; i < n; ++i) {
        if (results[i].index >= 5 && first_new_pos < 0) first_new_pos = i;
        if (results[i].index < 5  && first_old_pos < 0) first_old_pos = i;
    }
    ASSERT(first_new_pos >= 0, "at least one new vector should appear in results");
    ASSERT(first_old_pos >= 0, "at least one old vector should appear in results");
    ASSERT(first_new_pos < first_old_pos,
           "new vectors should rank higher than old vectors with freshness_weight=0.7");

    /* Verify scores are in descending order. */
    for (int i = 1; i < n; ++i) {
        ASSERT(results[i].final_score <= results[i-1].final_score,
               "results should be sorted by final_score descending");
    }

    db_close(db);
    gv_sim_time_set_mode(GV_TIME_WALL);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test: freshness_weight=0 should behave like normal search
 * ------------------------------------------------------------------------ */

static int test_zero_freshness_weight(void)
{
    gv_sim_time_set_mode(GV_TIME_SIM);
    gv_sim_time_reset(1700000000ULL);

    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    for (int i = 0; i < 5; ++i) {
        float v[4] = {(float)(i + 1), 0.0f, 0.0f, 0.0f};
        ASSERT(db_add_vector(db, v, 4) == 0, "insert should succeed");
    }

    GV_FreshnessParams params = {
        .score_weight     = 1.0,
        .freshness_weight = 0.0,
        .half_life_sec    = 86400.0,
        .decay_type       = GV_RANK_DECAY_EXP
    };

    float query[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_FreshnessResult results[5];
    int n = db_search_with_freshness(db, query, 5, &params, results);
    ASSERT(n > 0, "should return results");

    db_close(db);
    gv_sim_time_set_mode(GV_TIME_WALL);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test: filtered search with freshness
 * ------------------------------------------------------------------------ */

static int test_filtered_freshness(void)
{
    gv_sim_time_set_mode(GV_TIME_SIM);
    gv_sim_time_reset(1700000000ULL);

    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    /* Insert vectors with "type=A" and "type=B". */
    for (int i = 0; i < 3; ++i) {
        float v[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        ASSERT(db_add_vector_with_metadata(db, v, 4, "type", "A") == 0,
               "insert type=A should succeed");
    }
    for (int i = 0; i < 3; ++i) {
        float v[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        ASSERT(db_add_vector_with_metadata(db, v, 4, "type", "B") == 0,
               "insert type=B should succeed");
    }

    GV_FreshnessParams params = {
        .score_weight     = 0.5,
        .freshness_weight = 0.5,
        .half_life_sec    = 86400.0,
        .decay_type       = GV_RANK_DECAY_EXP
    };

    float query[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    GV_FreshnessResult results[6];
    int n = db_search_filtered_with_freshness(db, query, 3, &params,
                                               "type", "A", results);
    ASSERT(n > 0, "filtered freshness search should return results");
    /* All results should be type=A vectors (indices 0-2). */
    for (int i = 0; i < n; ++i) {
        ASSERT(results[i].index < 3, "filtered results should only be type=A vectors");
    }

    db_close(db);
    gv_sim_time_set_mode(GV_TIME_WALL);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test: timestamp is refreshed on update
 * ------------------------------------------------------------------------ */

static int test_timestamp_refresh_on_update(void)
{
    gv_sim_time_set_mode(GV_TIME_SIM);
    gv_sim_time_reset(1700000000ULL);

    GV_Database *db = db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open should succeed");

    float v[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    ASSERT(db_add_vector(db, v, 4) == 0, "insert should succeed");

    uint64_t ts_before = db->soa_storage->insert_timestamps[0];

    /* Advance time and update the vector. */
    gv_sim_time_advance_sec(3600ULL);
    float v2[4] = {0.9f, 0.1f, 0.0f, 0.0f};
    ASSERT(db_update_vector(db, 0, v2, 4) == 0, "update should succeed");

    /* The timestamp should be newer now. */
    uint64_t ts_after = db->soa_storage->insert_timestamps[0];
    ASSERT(ts_after > ts_before, "timestamp should be refreshed after update");

    db_close(db);
    gv_sim_time_set_mode(GV_TIME_WALL);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------ */

typedef int (*test_fn)(void);

static struct { const char *name; test_fn fn; } tests[] = {
    { "freshness_score_exp",        test_freshness_score_exp       },
    { "freshness_score_gauss",      test_freshness_score_gauss     },
    { "freshness_score_linear",     test_freshness_score_linear    },
    { "freshness_blend",            test_freshness_blend           },
    { "newer_vectors_rank_higher",  test_newer_vectors_rank_higher },
    { "zero_freshness_weight",      test_zero_freshness_weight     },
    { "filtered_freshness",         test_filtered_freshness        },
    { "timestamp_refresh_on_update",test_timestamp_refresh_on_update},
};

int main(void)
{
    int passed = 0, failed = 0;
    size_t n = sizeof(tests) / sizeof(tests[0]);

    for (size_t i = 0; i < n; ++i) {
        printf("  [%zu/%zu] %s ... ", i + 1, n, tests[i].name);
        fflush(stdout);
        if (tests[i].fn() == 0) {
            printf("OK\n");
            passed++;
        } else {
            printf("FAILED\n");
            failed++;
        }
    }

    printf("\nResults: %d/%d passed\n", passed, passed + failed);
    return failed == 0 ? 0 : 1;
}
