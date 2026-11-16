/**
 * @file fuzz_backup_restore.c
 * @brief libFuzzer harness for backup-file header/integrity parsing.
 *
 * A backup file is untrusted input on the restore path. This writes RAW fuzz
 * bytes to a temp file and runs the lightweight parsers that read a backup
 * without side-effecting the filesystem with a restored DB:
 *   - backup_read_header()  : "read header without full restore"
 *   - backup_verify()       : parse + walk + checksum the whole file
 * Both take only a path and must stay crash-safe on arbitrary bytes. The
 * GV_BackupResult from backup_verify is freed; the temp file is unlinked.
 *
 * Build: make fuzz  (requires clang)
 * Run:   build/fuzz/fuzz_backup_restore tests/fuzz/corpus/backup -runs=10000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "storage/backup.h"
#include "../test_tmp.h"

#define FUZZ_MAX_INPUT (256 * 1024)

static int write_file(const char *path, const uint8_t *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (size) fwrite(data, 1, size, f);
    fclose(f);
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > FUZZ_MAX_INPUT) return 0;

    char path[256];
    if (gv_test_make_temp_path(path, sizeof(path), "gv_fuzz_backup", ".gvbk") != 0) {
        return 0;
    }
    if (write_file(path, data, size) != 0) {
        return 0;
    }

    /* Header parser: fixed-size struct output, no allocation to leak. */
    GV_BackupHeader header;
    (void)backup_read_header(path, &header);

    /* Full-file integrity parser: walks the body and checksums it. Returns an
     * owned result that must be freed regardless of success/failure. */
    GV_BackupResult *res = backup_verify(path, NULL);
    if (res) {
        backup_result_free(res);
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
