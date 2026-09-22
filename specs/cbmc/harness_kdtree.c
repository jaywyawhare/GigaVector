/*
 * CBMC harness: K-D tree structural invariants (src/index/kdtree.c).
 *
 * A k-d tree is only a correct search structure if the split invariant holds:
 * everything in the left subtree is <= the node on its axis, everything right
 * is >=. Break it and search silently prunes the wrong branch -- which shows up
 * as "recall regression", not as a crash, so it can hide for a long time.
 *
 * Also pins the axis-cycling and index-range invariants that
 * exact_knn_search_kdtree relies on.
  *
 * CBMC-SOURCES: src/index/kdtree.c src/storage/soa_storage.c src/search/distance.c src/core/memory.c
 * CBMC-UNWIND: 5
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/types.h"
#include "index/kdtree.h"
#include "storage/soa_storage.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
size_t   nondet_size(void);
float    nondet_float(void);
unsigned char nondet_uchar(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
static size_t nondet_size(void) { return (size_t)(nondet_u64() % 8); }
static float nondet_float(void) {
    return (float)((int)(nondet_u64() % 2000) - 1000) / 100.0f;
}
static unsigned char nondet_uchar(void) { return (unsigned char)(nondet_u64() & 0xFF); }
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define DIM 2
#define N   2

/* Recursively check the split invariant and that every node index is in range. */
static void check_node(const GV_KDNode *node, const GV_SoAStorage *st,
                       size_t count, unsigned depth) {
    if (!node || depth > N) return;
    assert(node->vector_index < count);      /* index in range */
    assert(node->axis < DIM);                /* axis cycles within the dimension */

    const float *here = soa_storage_get_data(st, node->vector_index);
    assert(here != NULL);

    if (node->left) {
        assert(node->left->vector_index < count);
        const float *l = soa_storage_get_data(st, node->left->vector_index);
        if (l) assert(l[node->axis] <= here[node->axis]);   /* split invariant */
        check_node(node->left, st, count, depth + 1);
    }
    if (node->right) {
        assert(node->right->vector_index < count);
        const float *r = soa_storage_get_data(st, node->right->vector_index);
        if (r) assert(r[node->axis] >= here[node->axis]);
        check_node(node->right, st, count, depth + 1);
    }
}

int main(void) {
    GV_SoAStorage *st = soa_storage_create(DIM, N);
    if (!st) return 0;

    GV_KDNode *root = NULL;
    for (unsigned i = 0; i < N; i++) {
        float v[DIM];
        for (unsigned j = 0; j < DIM; j++) {
            v[j] = nondet_float();
            ASSUME(v[j] == v[j]);
        }
        size_t idx = soa_storage_add(st, v, NULL);
        (void)idx;
        kdtree_insert(&root, st, soa_storage_count(st) - 1, 0);
    }

    check_node(root, st, soa_storage_count(st), 0);

    kdtree_destroy_recursive(root);
    soa_storage_destroy(st);
    return 0;
}
