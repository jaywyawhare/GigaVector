/* Durable Raft log: persistence across a simulated restart. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "admin/raft_log.h"
#include "../test_tmp.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

int main(void) {
    char path[512];
    gv_test_make_temp_path(path, sizeof(path), "raftlog", ".bin");
    remove(path);

    /* Session 1: write vote + three entries. */
    GV_RaftLog *log = raft_log_open(path);
    ASSERT(log != NULL, "open fresh log");
    ASSERT(raft_log_count(log) == 0, "fresh log empty");
    ASSERT(raft_log_set_meta(log, 7, 2) == 0, "set meta term=7 vote=2");
    uint64_t a = 0xAA, b = 0xBB, c = 0xCC;
    ASSERT(raft_log_append(log, 1, 7, &a, sizeof(a)) == 0, "append entry 1");
    ASSERT(raft_log_append(log, 2, 7, &b, sizeof(b)) == 0, "append entry 2");
    ASSERT(raft_log_append(log, 3, 7, &c, sizeof(c)) == 0, "append entry 3");
    ASSERT(raft_log_count(log) == 3, "three entries live");
    raft_log_close(log); /* simulate crash/restart */

    /* Session 2: reopen and verify everything recovered. */
    log = raft_log_open(path);
    ASSERT(log != NULL, "reopen log");
    ASSERT(raft_log_term(log) == 7, "term recovered");
    ASSERT(raft_log_voted_for(log) == 2, "vote recovered");
    ASSERT(raft_log_count(log) == 3, "entries recovered (not lost)");
    const GV_RaftEntry *e = raft_log_entries(log);
    ASSERT(e && e[2].len == sizeof(c) && memcmp(e[2].data, &c, sizeof(c)) == 0,
           "entry 3 payload intact");

    /* Conflict truncation then re-append (follower overwrite path). */
    ASSERT(raft_log_truncate(log, 2) == 0, "truncate to index 2");
    ASSERT(raft_log_count(log) == 2, "two entries after truncate");
    uint64_t d = 0xDD;
    ASSERT(raft_log_append(log, 3, 9, &d, sizeof(d)) == 0, "re-append new entry 3");
    raft_log_close(log);

    /* Session 3: the overwrite survived and compacted. */
    log = raft_log_open(path);
    ASSERT(raft_log_count(log) == 3, "count 3 after truncate+reappend restart");
    e = raft_log_entries(log);
    ASSERT(e && e[2].term == 9 && e[2].len == sizeof(d) &&
           memcmp(e[2].data, &d, sizeof(d)) == 0, "overwritten entry 3 recovered");
    raft_log_close(log);
    remove(path);

    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL RAFT-LOG TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
