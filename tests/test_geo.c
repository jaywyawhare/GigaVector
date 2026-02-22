#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_geo.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* --- Test: create and destroy --- */
static int test_create_destroy(void) {
    GV_GeoIndex *idx = gv_geo_create();
    ASSERT(idx != NULL, "gv_geo_create should return non-NULL");
    ASSERT(gv_geo_count(idx) == 0, "newly created index should have count 0");
    gv_geo_destroy(idx);
    /* destroying NULL should be safe */
    gv_geo_destroy(NULL);
    return 0;
}

/* --- Test: insert and count --- */
static int test_insert_count(void) {
    GV_GeoIndex *idx = gv_geo_create();
    ASSERT(idx != NULL, "gv_geo_create should return non-NULL");

    int rc = gv_geo_insert(idx, 0, 40.7128, -74.0060);   /* New York */
    ASSERT(rc == 0, "insert point 0 should succeed");
    ASSERT(gv_geo_count(idx) == 1, "count should be 1 after one insert");

    rc = gv_geo_insert(idx, 1, 34.0522, -118.2437);       /* Los Angeles */
    ASSERT(rc == 0, "insert point 1 should succeed");
    ASSERT(gv_geo_count(idx) == 2, "count should be 2 after two inserts");

    rc = gv_geo_insert(idx, 2, 51.5074, -0.1278);         /* London */
    ASSERT(rc == 0, "insert point 2 should succeed");
    ASSERT(gv_geo_count(idx) == 3, "count should be 3 after three inserts");

    gv_geo_destroy(idx);
    return 0;
}

/* --- Test: update a point --- */
static int test_update(void) {
    GV_GeoIndex *idx = gv_geo_create();
    ASSERT(idx != NULL, "gv_geo_create should return non-NULL");

    gv_geo_insert(idx, 0, 40.7128, -74.0060);
    int rc = gv_geo_update(idx, 0, 48.8566, 2.3522);      /* Move to Paris */
    ASSERT(rc == 0, "update should succeed");
    ASSERT(gv_geo_count(idx) == 1, "count should remain 1 after update");

    gv_geo_destroy(idx);
    return 0;
}

/* --- Test: remove a point --- */
static int test_remove(void) {
    GV_GeoIndex *idx = gv_geo_create();
    ASSERT(idx != NULL, "gv_geo_create should return non-NULL");

    gv_geo_insert(idx, 0, 40.7128, -74.0060);
    gv_geo_insert(idx, 1, 34.0522, -118.2437);
    ASSERT(gv_geo_count(idx) == 2, "count should be 2 before remove");

    int rc = gv_geo_remove(idx, 0);
    ASSERT(rc == 0, "remove should succeed for existing point");
    ASSERT(gv_geo_count(idx) == 1, "count should be 1 after remove");

    gv_geo_destroy(idx);
    return 0;
}

/* --- Test: radius search --- */
static int test_radius_search(void) {
    GV_GeoIndex *idx = gv_geo_create();
    ASSERT(idx != NULL, "gv_geo_create should return non-NULL");

    /* Insert cities */
    gv_geo_insert(idx, 0, 40.7128, -74.0060);   /* New York */
    gv_geo_insert(idx, 1, 40.7580, -73.9855);   /* Midtown Manhattan (~5km from NYC center) */
    gv_geo_insert(idx, 2, 34.0522, -118.2437);  /* Los Angeles (~3940km away) */

    GV_GeoResult results[10];
    int n = gv_geo_radius_search(idx, 40.7128, -74.0060, 50.0, results, 10);
    /* Should find NYC center and Midtown within 50km, not LA */
    ASSERT(n >= 1, "radius search should find at least 1 point within 50km");
    ASSERT(n <= 2, "radius search should find at most 2 points within 50km");

    gv_geo_destroy(idx);
    return 0;
}

/* --- Test: bounding box search --- */
static int test_bbox_search(void) {
    GV_GeoIndex *idx = gv_geo_create();
    ASSERT(idx != NULL, "gv_geo_create should return non-NULL");

    gv_geo_insert(idx, 0, 40.7128, -74.0060);   /* New York */
    gv_geo_insert(idx, 1, 34.0522, -118.2437);  /* Los Angeles */
    gv_geo_insert(idx, 2, 51.5074, -0.1278);    /* London */

    /* BBox covering northeastern US */
    GV_GeoBBox bbox = {
        .min = { .lat = 39.0, .lng = -76.0 },
        .max = { .lat = 42.0, .lng = -72.0 }
    };

    GV_GeoResult results[10];
    int n = gv_geo_bbox_search(idx, &bbox, results, 10);
    ASSERT(n >= 1, "bbox search should find at least 1 point in NE US");

    gv_geo_destroy(idx);
    return 0;
}

/* --- Test: haversine distance --- */
static int test_haversine_distance(void) {
    /* Distance from NYC to London is approximately 5570 km */
    double dist = gv_geo_distance_km(40.7128, -74.0060, 51.5074, -0.1278);
    ASSERT(dist > 5000.0, "NYC-London distance should be > 5000 km");
    ASSERT(dist < 6000.0, "NYC-London distance should be < 6000 km");

    /* Distance from a point to itself should be 0 */
    double self_dist = gv_geo_distance_km(40.0, -74.0, 40.0, -74.0);
    ASSERT(self_dist < 0.001, "distance from point to itself should be ~0");

    return 0;
}

/* --- Test: get candidates for pre-filtering --- */
static int test_get_candidates(void) {
    GV_GeoIndex *idx = gv_geo_create();
    ASSERT(idx != NULL, "gv_geo_create should return non-NULL");

    gv_geo_insert(idx, 0, 40.7128, -74.0060);
    gv_geo_insert(idx, 1, 40.7580, -73.9855);
    gv_geo_insert(idx, 2, 34.0522, -118.2437);

    size_t indices[10];
    int n = gv_geo_get_candidates(idx, 40.7128, -74.0060, 50.0, indices, 10);
    ASSERT(n >= 1, "get_candidates should return at least 1 candidate");

    gv_geo_destroy(idx);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing geo create/destroy...", test_create_destroy},
        {"Testing geo insert/count...", test_insert_count},
        {"Testing geo update...", test_update},
        {"Testing geo remove...", test_remove},
        {"Testing geo radius search...", test_radius_search},
        {"Testing geo bbox search...", test_bbox_search},
        {"Testing geo haversine distance...", test_haversine_distance},
        {"Testing geo get candidates...", test_get_candidates},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
