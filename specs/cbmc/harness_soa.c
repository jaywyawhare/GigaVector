/*
 * CBMC harness: struct-of-arrays slot bookkeeping
 * (storage/soa_storage.c, compact_store.c, disk_layout.c, mmap.c).
 *
 * SoA storage hands out stable indices and marks deletes with tombstones
 * rather than compacting -- which is precisely why TTL cleanup and scroll
 * cursors are safe (see specs/quint/ttl.qnt, scroll_cursor.qnt). This harness
 * pins that behaviour down: indices stay valid, tombstones do not renumber, and
 * out-of-range access is rejected rather than read.
  *
 * CBMC-SOURCES: src/storage/soa_storage.c src/core/memory.c
 * CBMC-UNWIND: 5
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "storage/soa_storage.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
size_t   nondet_size(void);
unsigned char nondet_uchar(void);
float    nondet_float(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
static size_t nondet_size(void) { return (size_t)(nondet_u64() % 8); }
static unsigned char nondet_uchar(void) { return (unsigned char)(nondet_u64() & 0xFF); }
static float nondet_float(void) {
    return (float)((int)(nondet_u64() % 2000) - 1000) / 100.0f;
}
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

#define DIM 2
#define N   2

int main(void) {
    GV_SoAStorage *s = soa_storage_create(DIM, N);
    if (!s) return 0;

    float v[DIM];
    size_t idx[N];
    for (unsigned i = 0; i < N; i++) {
        for (unsigned j = 0; j < DIM; j++) v[j] = nondet_float();
        idx[i] = soa_storage_add(s, v, NULL);
    }

    /* Indices are handed out densely and in order. */
    for (unsigned i = 1; i < N; i++)
        assert(idx[i] > idx[i - 1]);
    assert(soa_storage_count(s) == N);

    /* Deleting is a TOMBSTONE: the count of slots does not shrink and the
     * surviving indices keep pointing at the same rows. This is the property
     * TTL cleanup and pagination depend on. */
    size_t before = soa_storage_count(s);
    if (soa_storage_mark_deleted(s, idx[0]) == 0) {
        assert(soa_storage_is_deleted(s, idx[0]) == 1);
        assert(soa_storage_count(s) == before);        /* no renumbering */
        for (unsigned i = 1; i < N; i++) {
            assert(soa_storage_is_deleted(s, idx[i]) == 0);
            assert(soa_storage_get_data(s, idx[i]) != NULL);
        }
    }

    /* Out-of-range access is rejected, not read. */
    assert(soa_storage_get_data(s, before + 100) == NULL);
    assert(soa_storage_mark_deleted(s, before + 100) != 0);

    soa_storage_destroy(s);
    return 0;
}
