/*
 * CBMC harness: memory safety of the WAL record decoder
 * (wal_apply_record_buffer, src/storage/wal.c).
 *
 * This is the same entry point tests/fuzz/fuzz_wal_apply.c drives, but fed a
 * fully nondeterministic buffer instead of a corpus. It is the natural CBMC
 * target because the parser reads attacker-influenced length fields.
 *
 *   cbmc --bounds-check --pointer-check --conversion-check \
 *        --unwind 16 -I include specs/cbmc/harness_wal_record.c src/storage/wal.c ...
  *
 * CBMC-SOURCES: src/storage/wal.c src/core/memory.c
 * CBMC-UNWIND: 10
*/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "storage/wal.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint8_t nondet_u8(void);
size_t  nondet_size(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
static uint8_t nondet_u8(void) { return 0; }
static size_t  nondet_size(void) { return 0; }
#endif

#define BUF_BOUND 6
#define WAL_DIM    4

static int noop_insert(void *ctx, const float *data, size_t dimension,
                       const char *const *keys, const char *const *vals,
                       size_t n) {
    (void)ctx; (void)data; (void)dimension; (void)keys; (void)vals; (void)n;
    return 0;
}
static int noop_delete(void *ctx, size_t idx) { (void)ctx; (void)idx; return 0; }
static int noop_update(void *ctx, size_t idx, const float *data, size_t dim,
                       const char *const *keys, const char *const *vals, size_t n) {
    (void)ctx; (void)idx; (void)data; (void)dim; (void)keys; (void)vals; (void)n;
    return 0;
}
static int noop_ivfdisk(void *ctx, uint64_t head, uint64_t vid,
                        const float *data, size_t dim) {
    (void)ctx; (void)head; (void)vid; (void)data; (void)dim;
    return 0;
}

int main(void) {
    size_t len = nondet_size();
    ASSUME(len <= BUF_BOUND);

    uint8_t buf[BUF_BOUND];
    for (size_t i = 0; i < BUF_BOUND; i++) buf[i] = nondet_u8();

    /* The contract: for ANY byte string the decoder either reports an error or
     * consumes it safely. It must never read or write out of bounds -- that is
     * what --bounds-check and --pointer-check assert here. There is no
     * functional postcondition: garbage in, defined error out. */
    (void)wal_apply_record_buffer(buf, len, /*has_crc=*/1, WAL_DIM,
                                  noop_insert, noop_delete, noop_update,
                                  noop_ivfdisk, NULL);
    (void)wal_apply_record_buffer(buf, len, /*has_crc=*/0, WAL_DIM,
                                  noop_insert, noop_delete, noop_update,
                                  noop_ivfdisk, NULL);
    return 0;
}
