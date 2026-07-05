#ifndef GIGAVECTOR_GV_AB_TEST_H
#define GIGAVECTOR_GV_AB_TEST_H

#include <stdint.h>
#include <stddef.h>

#include "core/types.h"
#include "search/distance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file ab_test.h
 * @brief Index-level A/B testing for GigaVector.
 *
 * Enables running a shadow index alongside the primary index, routing a
 * configurable fraction of queries to each, and collecting per-variant
 * latency and query-count statistics.
 */

/* Forward declaration to avoid circular includes. */
typedef struct GV_Database GV_Database;

/**
 * @brief A/B test state embedded in a GV_Database instance.
 *
 * Tracks a shadow index (type_b) alongside the primary index (type_a).
 * The route_counter is used to deterministically split traffic.
 */
typedef struct GV_ABTest {
    char     name[64];          /**< Human-readable test name. */
    int      index_type_a;      /**< Primary index type (mirrors db->index_type). */
    int      index_type_b;      /**< Shadow index type. */
    float    traffic_split;     /**< Fraction [0,1] of queries routed to shadow (B). */
    uint64_t queries_a;         /**< Total queries served by primary index A. */
    uint64_t queries_b;         /**< Total queries served by shadow index B. */
    double   latency_sum_a;     /**< Accumulated query latency for A (microseconds). */
    double   latency_sum_b;     /**< Accumulated query latency for B (microseconds). */
    uint64_t creation_time_us;  /**< Monotonic creation timestamp (microseconds). */
    uint64_t route_counter;     /**< Monotonically increasing counter used for routing. */
    GV_Database *shadow_db;     /**< Shadow database (owned; freed on destroy). */
} GV_ABTest;

/* -------------------------------------------------------------------------
 * Internal helpers — used by ab_test.c and callable from tests.
 * ---------------------------------------------------------------------- */

/**
 * @brief Allocate and initialise a GV_ABTest, building the shadow index.
 *
 * Copies all existing vectors from @p db into the new shadow database of
 * type @p type_b.  The caller (usually gv_db_ab_test_start) stores the
 * returned pointer in db->ab_test.
 *
 * @param db     Primary database (must be non-NULL).
 * @param name   Test name (truncated to 63 characters).
 * @param type_b Shadow index type.
 * @param split  Traffic fraction for shadow [0.0, 1.0].
 * @return New GV_ABTest instance, or NULL on failure.
 */
GV_ABTest *ab_test_create(GV_Database *db, const char *name, int type_b, float split);

/**
 * @brief Decide which variant (0=A, 1=B) should serve the next query.
 *
 * Increments the internal route_counter and routes to B when
 * (counter % 1000) < traffic_split * 1000.
 * Must be called with db->ab_mutex held.
 *
 * @param db Database whose ab_test is consulted (must be non-NULL).
 * @return 0 to route to primary (A), 1 to route to shadow (B).
 */
int ab_test_route(GV_Database *db);

/**
 * @brief Record query latency for variant @p which.
 *
 * Must be called with db->ab_mutex held.
 *
 * @param db         Database whose ab_test receives the sample.
 * @param which      0 for variant A, 1 for variant B.
 * @param latency_us Query latency in microseconds.
 */
void ab_test_record(GV_Database *db, int which, double latency_us);

/**
 * @brief Render a JSON report of A/B test statistics into @p buf.
 *
 * @param db  Database to report on (must have ab_test set).
 * @param buf Output buffer.
 * @param len Buffer capacity in bytes.
 * @return 0 on success, -1 on error or if no test is active.
 */
int ab_test_report(const GV_Database *db, char *buf, size_t len);

/**
 * @brief Free a GV_ABTest and its owned shadow database.
 *
 * @param test Test to destroy (NULL is safe).
 */
void ab_test_destroy(GV_ABTest *test);

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

/**
 * @brief Start an A/B test on @p db.
 *
 * Creates a shadow index of type @p type_b populated with all currently
 * stored vectors.  Subsequent calls to gv_db_ab_test_search() will route
 * queries between the primary and shadow indexes according to @p split.
 *
 * @param db     Primary database (must be non-NULL).
 * @param name   Descriptive test name (max 63 chars).
 * @param type_b Shadow index type (e.g. GV_INDEX_TYPE_FLAT).
 * @param split  Fraction of queries routed to shadow [0.0, 1.0].
 * @return 0 on success, -1 on failure or if a test is already running.
 */
int gv_db_ab_test_start(GV_Database *db, const char *name, int type_b, float split);

/**
 * @brief Execute a search routed through the active A/B test.
 *
 * Routes the query to either the primary or shadow index according to the
 * configured traffic_split, times the search, and records latency.
 *
 * @param db            Database with an active A/B test.
 * @param query         Query vector (dimension must match db->dimension).
 * @param k             Number of nearest neighbors.
 * @param results       Output array of at least @p k elements.
 * @param distance_type Distance metric.
 * @return Number of results found (0..k), or -1 on error.
 */
int gv_db_ab_test_search(GV_Database *db, const float *query, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type);

/**
 * @brief Write a JSON summary of the active A/B test into @p buf.
 *
 * The JSON object contains: name, index_type_a, index_type_b,
 * traffic_split, queries_a, queries_b, avg_latency_a_us,
 * avg_latency_b_us, creation_time_us.
 *
 * @param db  Database (must have an active test).
 * @param buf Output buffer.
 * @param len Buffer capacity in bytes.
 * @return 0 on success, -1 if no test is active or arguments are invalid.
 */
int gv_db_ab_test_report(const GV_Database *db, char *buf, size_t len);

/**
 * @brief Stop the active A/B test and free all associated resources.
 *
 * After this call db->ab_test is NULL. The primary database is unaffected.
 *
 * @param db Database to stop the test on (must be non-NULL).
 */
void gv_db_ab_test_stop(GV_Database *db);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_AB_TEST_H */
