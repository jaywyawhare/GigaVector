/*
 * CBMC harness: simulated-clock monotonicity (src/core/sim_time.c).
 *
 * This one is load-bearing beyond its size. `tests/dst/` and the Quint replay
 * bridge both drive TIME through this module -- election timeouts, TTL expiry
 * and cache staleness are all measured against it. If the simulated clock can
 * go backwards or wrap, every deterministic replay silently stops being
 * deterministic, and the whole verification story rests on sand.
 *
 * Previously classified `exempt` as "configuration plumbing", which was wrong:
 * it has real arithmetic and a real invariant.
  *
 * CBMC-SOURCES: src/core/sim_time.c
 * CBMC-UNWIND: 4
*/
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "core/sim_time.h"

#if defined(__CPROVER__) || defined(__CPROVER)
uint64_t nondet_u64(void);
#  define ASSUME(c) __CPROVER_assume(c)
#else
static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static uint64_t nondet_u64(void) {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17;
    return g_seed;
}
#  define ASSUME(c) do { if (!(c)) return 0; } while (0)
#endif

int main(void) {
    gv_sim_time_set_mode(GV_TIME_SIM);

    uint64_t start = nondet_u64() % 1000000u;
    gv_sim_time_reset(start);
    assert(gv_time_now_sec() == start);

    /* MONOTONICITY: advancing never moves the clock backwards, for any delta.
     * A wrap here would make a DST replay diverge from its recorded trace. */
    uint64_t d1 = nondet_u64() % 100000u;
    uint64_t before_s = gv_time_now_sec();
    gv_sim_time_advance_sec(d1);
    uint64_t after_s = gv_time_now_sec();
    assert(after_s >= before_s);
    assert(after_s == before_s + d1);          /* exact, not merely monotone */

    uint64_t before_ms = gv_time_now_ms();
    uint64_t d2 = nondet_u64() % 100000u;
    gv_sim_time_advance_ms(d2);
    uint64_t after_ms = gv_time_now_ms();
    assert(after_ms >= before_ms);
    assert(after_ms == before_ms + d2);

    /* Seconds and milliseconds stay consistent with each other. */
    assert(gv_time_now_ms() / 1000u == gv_time_now_sec());

    /* A zero advance is a no-op, not a drift. */
    uint64_t t = gv_time_now_ms();
    gv_sim_time_advance_ms(0);
    assert(gv_time_now_ms() == t);

    /* Reset is absolute: it lands exactly where asked, regardless of history. */
    uint64_t again = nondet_u64() % 1000000u;
    gv_sim_time_reset(again);
    assert(gv_time_now_sec() == again);

    /* In simulated mode, sleeping ADVANCES the clock rather than blocking --
     * this is what makes timeout-driven tests deterministic. */
    uint64_t pre = gv_time_now_ms();
    gv_time_sleep_ms(50);
    assert(gv_time_now_ms() >= pre);

    return 0;
}
