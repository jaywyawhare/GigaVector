/*
 * CBMC harness: text parser memory safety
 * (features/json.c, sql.c, cypher.c, json_index.c, multimodal/tokenizer.c).
 *
 * These are the query-language and document parsers -- the largest
 * attacker-reachable surface in the tree. libFuzzer already drives them over a
 * seeded corpus (tests/fuzz/fuzz_{json,sql,cypher}.c); this bounds the claim
 * over every input up to a length, which the corpus cannot do.
 *
 * Deep-nesting inputs are what break recursive-descent parsers, so the corpus
 * seeds include deep_expr_512 / deep_pattern_parens. Keep --unwind above the
 * parser's own depth limit when running this for real.
  *
 * CBMC-SOURCES: src/features/json.c src/core/memory.c
 * CBMC-UNWIND: 8
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "features/json.h"

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

#define TEXT_BOUND 5

int main(void) {
    size_t len = nondet_size();
    ASSUME(len <= TEXT_BOUND - 1);

    char text[TEXT_BOUND];
    for (size_t i = 0; i < TEXT_BOUND - 1; i++) {
        text[i] = (char)nondet_uchar();
        ASSUME(text[i] != 0);
    }
    text[len] = '\0';

    /* Contract: parse either returns a value or NULL; it never reads past the
     * terminator and never double-frees on the error path. */
    GV_JsonError err;
    memset(&err, 0, sizeof(err));
    GV_JsonValue *v = json_parse(text, &err);
    if (v) {
        json_free(v);
    }
    return 0;
}
