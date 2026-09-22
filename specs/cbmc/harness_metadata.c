/*
 * CBMC harness: metadata key/value list operations
 * (schema/metadata.c, schema/vector.c, api/schema.c, api/typed_metadata.c,
 *  specialized/named_vectors.c).
 *
 * Metadata is an intrusive linked list threaded through every vector, copied on
 * migration and rebuilt on WAL replay. The contract is small and entirely
 * provable: set-then-get round-trips, keys are unique (a second set REPLACES),
 * remove actually removes, and a missing key returns NULL rather than reading
 * off the end of the list.
  *
 * CBMC-SOURCES: src/schema/metadata.c src/core/memory.c
 * CBMC-UNWIND: 6
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/types.h"
#include "schema/metadata.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
size_t   nondet_size(void);
float    nondet_float(void);
double   nondet_double(void);
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
static double nondet_double(void) { return (double)nondet_float(); }
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define DIM 2

int main(void) {
    float data[DIM] = { 1.0f, 2.0f };
    GV_Vector v;
    memset(&v, 0, sizeof(v));
    v.data = data; v.dimension = DIM; v.metadata = NULL;

    /* A key that is absent must read back as NULL, not as garbage. */
    assert(vector_get_metadata(&v, "absent") == NULL);

    if (vector_set_metadata(&v, "k1", "v1") != 0) return 0;

    /* ROUND-TRIP: what was set is what is read back. */
    const char *got = vector_get_metadata(&v, "k1");
    assert(got != NULL);
    assert(strcmp(got, "v1") == 0);

    /* KEY UNIQUENESS: setting the same key replaces rather than appending a
     * shadowing duplicate -- a duplicate would make get/remove order-dependent. */
    if (vector_set_metadata(&v, "k1", "v2") == 0) {
        const char *g2 = vector_get_metadata(&v, "k1");
        assert(g2 != NULL);
        assert(strcmp(g2, "v2") == 0);

        size_t count = 0;
        for (GV_Metadata *m = v.metadata; m; m = m->next)
            if (m->key && strcmp(m->key, "k1") == 0) count++;
        assert(count == 1);
    }

    /* A second, distinct key coexists. The set can fail on allocation, so
     * everything downstream that depends on k2 must be guarded by whether it
     * actually landed -- CBMC explores the allocation-failure path, and an
     * unconditional assertion here is simply wrong. */
    int have_k2 = (vector_set_metadata(&v, "k2", "v3") == 0);
    if (have_k2) {
        assert(vector_get_metadata(&v, "k1") != NULL);
        const char *g3 = vector_get_metadata(&v, "k2");
        assert(g3 != NULL && strcmp(g3, "v3") == 0);
    }

    /* REMOVE actually removes, and leaves the other key intact. */
    if (vector_remove_metadata(&v, "k1") == 0) {
        assert(vector_get_metadata(&v, "k1") == NULL);
        if (have_k2) assert(vector_get_metadata(&v, "k2") != NULL);
    }

    /* Removing an absent key is a no-op, not a corruption. */
    (void)vector_remove_metadata(&v, "never-present");
    if (have_k2) assert(vector_get_metadata(&v, "k2") != NULL);

    vector_clear_metadata(&v);
    assert(v.metadata == NULL);
    assert(vector_get_metadata(&v, "k2") == NULL);
    assert(vector_get_metadata(&v, "k1") == NULL);
    return 0;
}
