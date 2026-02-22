#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "gigavector/gv_backup.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define TMP_BACKUP_PATH "/tmp/gv_test_backup.bak"
#define TMP_RESTORE_PATH "/tmp/gv_test_restore.db"
#define TMP_INCR_PATH "/tmp/gv_test_incr.bak"
#define TMP_MERGED_PATH "/tmp/gv_test_merged.bak"

static void cleanup_temp_files(void) {
    unlink(TMP_BACKUP_PATH);
    unlink(TMP_RESTORE_PATH);
    unlink(TMP_INCR_PATH);
    unlink(TMP_MERGED_PATH);
}

/* ── Test: backup options init ─────────────────────────────────────────── */
static int test_backup_options_init(void) {
    GV_BackupOptions opts;
    memset(&opts, 0xFF, sizeof(opts));
    gv_backup_options_init(&opts);
    ASSERT(opts.compression == GV_BACKUP_COMPRESS_NONE, "default compression should be NONE");
    ASSERT(opts.include_wal == 1, "default include_wal should be 1");
    ASSERT(opts.include_metadata == 1, "default include_metadata should be 1");
    ASSERT(opts.verify_after == 1, "default verify_after should be 1");
    ASSERT(opts.encryption_key == NULL, "default encryption_key should be NULL");
    return 0;
}

/* ── Test: restore options init ────────────────────────────────────────── */
static int test_restore_options_init(void) {
    GV_RestoreOptions opts;
    memset(&opts, 0xFF, sizeof(opts));
    gv_restore_options_init(&opts);
    ASSERT(opts.overwrite == 0, "default overwrite should be 0");
    ASSERT(opts.verify_checksum == 1, "default verify_checksum should be 1");
    ASSERT(opts.decryption_key == NULL, "default decryption_key should be NULL");
    return 0;
}

/* ── Test: compression string ──────────────────────────────────────────── */
static int test_compression_string(void) {
    const char *s;

    s = gv_backup_compression_string(GV_BACKUP_COMPRESS_NONE);
    ASSERT(s != NULL, "compression string for NONE should not be NULL");

    s = gv_backup_compression_string(GV_BACKUP_COMPRESS_ZLIB);
    ASSERT(s != NULL, "compression string for ZLIB should not be NULL");

    s = gv_backup_compression_string(GV_BACKUP_COMPRESS_LZ4);
    ASSERT(s != NULL, "compression string for LZ4 should not be NULL");

    return 0;
}

/* ── Test: result free with NULL ───────────────────────────────────────── */
static int test_result_free_null(void) {
    /* Should not crash */
    gv_backup_result_free(NULL);
    return 0;
}

/* ── Test: backup create from file (non-existent source) ───────────────── */
static int test_backup_create_nonexistent(void) {
    cleanup_temp_files();
    GV_BackupOptions opts;
    gv_backup_options_init(&opts);

    GV_BackupResult *result = gv_backup_create_from_file(
        "/tmp/gv_nonexistent_db_file.db", TMP_BACKUP_PATH, &opts, NULL, NULL);
    /* With a non-existent source, we expect either NULL or result->success == 0 */
    if (result != NULL) {
        ASSERT(result->success == 0, "backup of non-existent file should fail");
        gv_backup_result_free(result);
    }
    cleanup_temp_files();
    return 0;
}

/* ── Test: read header on non-existent backup ──────────────────────────── */
static int test_read_header_nonexistent(void) {
    GV_BackupHeader header;
    memset(&header, 0, sizeof(header));
    int rc = gv_backup_read_header("/tmp/gv_no_such_backup.bak", &header);
    ASSERT(rc == -1, "read header on non-existent file should return -1");
    return 0;
}

/* ── Test: verify on non-existent backup ───────────────────────────────── */
static int test_verify_nonexistent(void) {
    GV_BackupResult *result = gv_backup_verify("/tmp/gv_no_such_backup.bak", NULL);
    if (result != NULL) {
        ASSERT(result->success == 0, "verify non-existent backup should fail");
        gv_backup_result_free(result);
    }
    return 0;
}

/* ── Test: backup header struct sizes ──────────────────────────────────── */
static int test_header_struct(void) {
    GV_BackupHeader header;
    memset(&header, 0, sizeof(header));

    header.version = GV_BACKUP_VERSION;
    ASSERT(header.version == 1, "backup version should be 1");

    header.vector_count = 1000;
    header.dimension = 128;
    ASSERT(header.vector_count == 1000, "vector_count should be 1000");
    ASSERT(header.dimension == 128, "dimension should be 128");
    ASSERT(sizeof(header.checksum) == 65, "checksum buffer should be 65 bytes");

    return 0;
}

/* Main */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing backup options init...",          test_backup_options_init},
        {"Testing restore options init...",         test_restore_options_init},
        {"Testing compression string...",           test_compression_string},
        {"Testing result free NULL...",             test_result_free_null},
        {"Testing backup create non-existent...",   test_backup_create_nonexistent},
        {"Testing read header non-existent...",     test_read_header_nonexistent},
        {"Testing verify non-existent...",          test_verify_nonexistent},
        {"Testing header struct...",                test_header_struct},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    cleanup_temp_files();
    return passed == n ? 0 : 1;
}
