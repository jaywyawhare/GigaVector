/*
 * test_vlog.c — WiscKey-style value log: append/read, durability, GC, CRC.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "storage/vlog.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

#define PATH "/tmp/gv_test_vlog.bin"

/* Liveness callback: live if the offset is present in a caller-supplied set. */
typedef struct { const uint64_t *live; size_t n; } LiveSet;
static int live_in_set(uint64_t off, void *ctx) {
    LiveSet *s = (LiveSet *)ctx;
    for (size_t i = 0; i < s->n; i++) if (s->live[i] == off) return 1;
    return 0;
}

int main(void) {
    remove(PATH);

    /* ---- append + read round-trip (incl. zero-length and large values) ---- */
    GV_VLog *vl = vlog_open(PATH);
    ASSERT(vl != NULL, "vlog_open creates file");

    const char *vals[] = { "alpha", "", "a longer value with spaces", "x" };
    size_t lens[] = { 5, 0, 26, 1 };
    uint64_t offs[4];
    for (int i = 0; i < 4; i++)
        ASSERT(vlog_append(vl, vals[i], lens[i], &offs[i]) == 0, "append value");
    ASSERT(offs[0] >= 16, "first offset past header");
    ASSERT(offs[0] != offs[2] && offs[1] != offs[3], "offsets distinct");
    ASSERT(vlog_record_count(vl) == 4, "record count == 4");

    for (int i = 0; i < 4; i++) {
        void *buf = NULL; size_t len = 0;
        ASSERT(vlog_read(vl, offs[i], &buf, &len) == 0, "read value back");
        ASSERT(len == lens[i], "length matches");
        ASSERT(len == 0 || memcmp(buf, vals[i], len) == 0, "content matches");
        free(buf);
    }
    /* A large value to exercise multi-KB records. */
    size_t big = 100000; char *bigbuf = (char *)malloc(big);
    for (size_t i = 0; i < big; i++) bigbuf[i] = (char)(i * 31 + 7);
    uint64_t big_off;
    ASSERT(vlog_append(vl, bigbuf, big, &big_off) == 0, "append 100KB value");
    { void *b = NULL; size_t l = 0; ASSERT(vlog_read(vl, big_off, &b, &l) == 0 && l == big && memcmp(b, bigbuf, big) == 0, "read 100KB back"); free(b); }
    ASSERT(vlog_sync(vl) == 0, "vlog_sync");
    vlog_close(vl);

    /* ---- reopen durability: values survive close/reopen ---- */
    vl = vlog_open(PATH);
    ASSERT(vl != NULL, "reopen existing log");
    ASSERT(vlog_record_count(vl) == 5, "reopened record count == 5 (scan)");
    for (int i = 0; i < 4; i++) {
        void *buf = NULL; size_t len = 0;
        ASSERT(vlog_read(vl, offs[i], &buf, &len) == 0 && len == lens[i]
               && (len == 0 || memcmp(buf, vals[i], len) == 0), "value survives reopen");
        free(buf);
    }

    /* ---- GC compaction: keep only offs[0] and offs[2]; drop the rest ---- */
    uint64_t keep[] = { offs[0], offs[2], big_off };
    LiveSet set = { keep, 3 };
    uint64_t *old_off = NULL, *new_off = NULL; size_t nrem = 0;
    uint64_t size_before = vlog_size(vl);
    ASSERT(vlog_gc(vl, live_in_set, &set, &old_off, &new_off, &nrem) == 0, "vlog_gc runs");
    ASSERT(nrem == 3, "3 live records survived GC");
    ASSERT(vlog_size(vl) < size_before, "log shrank after GC");
    ASSERT(vlog_record_count(vl) == 3, "record count == 3 after GC");

    /* Every surviving record is readable at its NEW offset with identical content. */
    for (size_t i = 0; i < nrem; i++) {
        void *buf = NULL; size_t len = 0;
        ASSERT(vlog_read(vl, new_off[i], &buf, &len) == 0, "read at remapped offset");
        /* Match old content by comparing against what we appended at old_off[i]. */
        const char *expect = NULL; size_t elen = 0;
        if (old_off[i] == offs[0]) { expect = vals[0]; elen = lens[0]; }
        else if (old_off[i] == offs[2]) { expect = vals[2]; elen = lens[2]; }
        else { expect = bigbuf; elen = big; }
        ASSERT(len == elen && (elen == 0 || memcmp(buf, expect, elen) == 0), "remapped content matches");
        free(buf);
    }
    free(old_off); free(new_off);
    vlog_close(vl);

    /* ---- reopen after GC still valid ---- */
    vl = vlog_open(PATH);
    ASSERT(vl != NULL && vlog_record_count(vl) == 3, "reopen after GC, 3 records");
    vlog_close(vl);

    /* ---- CRC corruption detection ---- */
    {
        /* Flip a byte inside the first value's payload region and expect read failure. */
        FILE *f = fopen(PATH, "r+b");
        /* First record after header: offset 16, header 8 bytes, payload starts at 24. */
        fseek(f, 24, SEEK_SET);
        int c = fgetc(f);
        fseek(f, 24, SEEK_SET);
        fputc(c ^ 0xFF, f);
        fclose(f);
        vl = vlog_open(PATH);
        void *buf = NULL; size_t len = 0;
        int rc = vlog_read(vl, 16, &buf, &len);
        ASSERT(rc == -1, "CRC mismatch detected on corrupted value");
        free(buf);
        vlog_close(vl);
    }

    free(bigbuf);
    remove(PATH);
    printf(failures ? "\nSOME VLOG TESTS FAILED (%d)\n" : "\nALL VLOG TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
