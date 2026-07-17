/**
 * test_ivf_base.c — direct tests for src/index/ivf_base.c kmeans/assignment.
 *
 * Covers ivf_train_centroids and ivf_assign_to_list:
 *   - empty input / invalid args
 *   - count < nlist (rejected)
 *   - degenerate / duplicate points
 *   - a normal assignment sanity check (well-separated clusters).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "index/ivf_base.h"

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); return -1; } \
} while (0)

/* count < nlist and NULL args are rejected. */
static int test_invalid_args(void) {
    float centroids[4 * 2];
    float data[2 * 2] = {0, 0, 1, 1};

    /* count (2) < nlist (4) -> reject. */
    ASSERT(ivf_train_centroids(data, 2, 2, 4, 5, centroids) == -1, "count<nlist rejected");
    /* NULL data / out. */
    ASSERT(ivf_train_centroids(NULL, 4, 2, 2, 5, centroids) == -1, "NULL data rejected");
    ASSERT(ivf_train_centroids(data, 4, 2, 2, 5, NULL) == -1, "NULL out rejected");
    return 0;
}

/* Empty input (count == 0): with nlist == 0 this is count == nlist so it is a
 * defined no-op success; with nlist > 0 it is count < nlist -> rejected. */
static int test_empty_input(void) {
    float centroids[1];
    float dummy = 0.0f;
    ASSERT(ivf_train_centroids(&dummy, 0, 2, 0, 5, centroids) == 0, "count==nlist==0 ok");
    ASSERT(ivf_train_centroids(&dummy, 0, 2, 1, 5, centroids) == -1, "0<1 rejected");
    return 0;
}

/* Degenerate: all points identical. Every centroid collapses to that point and
 * assignment is stable (all to some valid list). */
static int test_duplicate_points(void) {
    const size_t dim = 3, count = 6, nlist = 2;
    float data[6 * 3];
    for (size_t i = 0; i < count; i++) {
        data[i * dim + 0] = 5.0f;
        data[i * dim + 1] = -2.0f;
        data[i * dim + 2] = 7.0f;
    }
    float centroids[2 * 3];
    ASSERT(ivf_train_centroids(data, count, dim, nlist, 10, centroids) == 0, "train dup ok");

    /* Seeded centroid 0 equals the point; after averaging it stays the point. */
    ASSERT(fabsf(centroids[0] - 5.0f) < 1e-4f, "centroid0 x");
    ASSERT(fabsf(centroids[1] + 2.0f) < 1e-4f, "centroid0 y");
    ASSERT(fabsf(centroids[2] - 7.0f) < 1e-4f, "centroid0 z");

    int assign[6];
    ivf_assign_to_list(data, count, dim, centroids, nlist, assign);
    for (size_t i = 0; i < count; i++) {
        ASSERT(assign[i] >= 0 && assign[i] < (int)nlist, "assign in range");
    }
    return 0;
}

/* Normal case: two well-separated clusters. After training, points assign to
 * the centroid nearest their own cluster, and the two learned centroids sit
 * near the true cluster means. */
static int test_normal_assignment(void) {
    const size_t dim = 2, nlist = 2;
    /* Cluster A around (0,0), cluster B around (100,100). */
    float data[8 * 2] = {
        0.0f, 0.0f,   0.1f, -0.1f,   -0.1f, 0.1f,   0.2f, 0.2f,
        100.0f, 100.0f, 100.1f, 99.9f, 99.9f, 100.1f, 100.2f, 100.2f
    };
    const size_t count = 8;
    float centroids[2 * 2];
    ASSERT(ivf_train_centroids(data, count, dim, nlist, 25, centroids) == 0, "train ok");

    int assign[8];
    ivf_assign_to_list(data, count, dim, centroids, nlist, assign);

    /* First 4 points share a list, last 4 share the other list, and the two
     * groups differ. */
    for (size_t i = 1; i < 4; i++) ASSERT(assign[i] == assign[0], "cluster A coherent");
    for (size_t i = 5; i < 8; i++) ASSERT(assign[i] == assign[4], "cluster B coherent");
    ASSERT(assign[0] != assign[4], "clusters separated");

    /* Learned centroids near (0,0) and (100,100) in some order. */
    int a_list = assign[0];
    const float *ca = centroids + (size_t)a_list * dim;
    const float *cb = centroids + (size_t)(1 - a_list) * dim;
    ASSERT(fabsf(ca[0]) < 1.0f && fabsf(ca[1]) < 1.0f, "cluster A centroid near origin");
    ASSERT(fabsf(cb[0] - 100.0f) < 1.0f && fabsf(cb[1] - 100.0f) < 1.0f, "cluster B centroid near 100");

    /* A query at (0,0) assigns to A's list; at (100,100) to B's. */
    float qa[2] = {0.0f, 0.0f}, qb[2] = {100.0f, 100.0f};
    int ra, rb;
    ivf_assign_to_list(qa, 1, dim, centroids, nlist, &ra);
    ivf_assign_to_list(qb, 1, dim, centroids, nlist, &rb);
    ASSERT(ra == a_list, "query A -> list A");
    ASSERT(rb == (1 - a_list), "query B -> list B");
    return 0;
}

/* nlist == count: each point seeds its own centroid; assignment is a
 * permutation (each point nearest to itself). */
static int test_nlist_equals_count(void) {
    const size_t dim = 2, count = 3, nlist = 3;
    float data[3 * 2] = {0, 0, 10, 10, -10, -10};
    float centroids[3 * 2];
    ASSERT(ivf_train_centroids(data, count, dim, nlist, 5, centroids) == 0, "train ok");

    int assign[3];
    ivf_assign_to_list(data, count, dim, centroids, nlist, assign);
    /* Each point is its own seed centroid -> assigns to a distinct list. */
    ASSERT(assign[0] != assign[1] && assign[1] != assign[2] && assign[0] != assign[2],
           "distinct assignments");
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_invalid_args();
    rc |= test_empty_input();
    rc |= test_duplicate_points();
    rc |= test_normal_assignment();
    rc |= test_nlist_equals_count();
    if (rc == 0) printf("All ivf_base tests PASSED.\n");
    return rc != 0;
}
