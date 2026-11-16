/**
 * @file fuzz_vlog_open.c
 * @brief libFuzzer harness for the value-log reopen/parse path.
 *
 * A value-log file (and the WiscKey value store layered on it) is durable,
 * on-disk, and therefore untrusted at reopen. This writes RAW fuzz bytes to a
 * temp file and reopens it through the real parsers:
 *   - vlog_open()  + vlog_scan()      : header + per-record [len][crc][value] walk
 *   - value_store_open()              : rebuilds the key index from the log
 * Bogus magic/version, torn records, absurd length fields and CRC mismatches
 * must never crash, hang or over-read. Whatever opened is closed; temp unlinked.
 *
 * Build: make fuzz  (requires clang)
 * Run:   build/fuzz/fuzz_vlog_open tests/fuzz/corpus/vlog -runs=10000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "storage/vlog.h"
#include "storage/value_store.h"
#include "../test_tmp.h"

#define FUZZ_MAX_INPUT (256 * 1024)

static int write_file(const char *path, const uint8_t *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (size) fwrite(data, 1, size, f);
    fclose(f);
    return 0;
}

/* Walk every record; return 0 so the scan runs the parser over the whole file.
 * The value buffer is owned by vlog_scan (temporary), so nothing to free here. */
static int scan_cb(uint64_t offset, const void *value, size_t len, void *ctx) {
    (void)offset; (void)value; (void)len; (void)ctx;
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > FUZZ_MAX_INPUT) return 0;

    char path[256];
    if (gv_test_make_temp_path(path, sizeof(path), "gv_fuzz_vlog", ".gvvl") != 0) {
        return 0;
    }

    /* ── raw value-log reopen + record walk ── */
    if (write_file(path, data, size) == 0) {
        GV_VLog *vl = vlog_open(path);
        if (vl) {
            (void)vlog_scan(vl, scan_cb, NULL);
            (void)vlog_size(vl);
            (void)vlog_record_count(vl);
            vlog_close(vl);
        }
    }

    /* ── value-store reopen (rebuilds the key index from the same bytes) ── */
    if (write_file(path, data, size) == 0) {
        GV_ValueStore *vs = value_store_open(path);
        if (vs) {
            (void)value_store_count(vs);
            value_store_close(vs);
        }
    }

    unlink(path);
    return 0;
}

#ifdef GV_FUZZ_STANDALONE
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input-file>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint8_t buf[FUZZ_MAX_INPUT];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    return LLVMFuzzerTestOneInput(buf, n);
}
#endif
