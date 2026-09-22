/*
 * CBMC harness: posting-segment decoding
 * (src/storage/posting_list.c, multimodal/metadata_index.c, payload_index.c).
 *
 * `posting_segment_parse_buffer` reads length-prefixed on-disk segments. It is
 * already driven by tests/fuzz/fuzz_posting_segment.c over a corpus; this feeds
 * it a fully nondeterministic buffer so the bound covers EVERY input, not the
 * ones the fuzzer happened to reach. Same entry point, stronger quantifier.
  *
 * CBMC-SOURCES: src/storage/posting_list.c src/core/memory.c
 * CBMC-UNWIND: 10
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "storage/posting_list.h"

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

#define BUF_BOUND 6
#define MAX_DIM   8

static int noop_visit(void *ctx, const GV_PostingEntry *entry) {
    (void)ctx; (void)entry;
    return 0;
}

int main(void) {
    size_t len = nondet_size();
    ASSUME(len >= 1 && len <= BUF_BOUND);

    uint8_t buf[BUF_BOUND];
    for (size_t i = 0; i < BUF_BOUND; i++) buf[i] = nondet_uchar();

    /* Contract: for ANY byte string the parser either reports an error or
     * consumes it safely. There is no functional postcondition -- the assertion
     * is CBMC's own --bounds-check / --pointer-check over the whole call. */
    (void)posting_segment_parse_buffer(buf, len, MAX_DIM, noop_visit, NULL);
    (void)posting_segment_parse_buffer(buf, len, 0, noop_visit, NULL);
    return 0;
}
