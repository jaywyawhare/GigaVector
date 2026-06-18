#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "storage/backup.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static char tmp_backup_path[512];
static char tmp_restore_path[512];
static char tmp_incr_path[512];
static char tmp_merged_path[512];
static char tmp_missing_db_path[512];
static char tmp_missing_bak_path[512];
static int tmp_paths_ready;

static int ensure_temp_paths(void) {
    if (tmp_paths_ready) {
        return 0;
    }
    if (gv_test_make_temp_path(tmp_backup_path, sizeof(tmp_backup_path),
                               "gv_test_backup", ".bak") != 0) {
        return -1;
    }
    if (gv_test_make_temp_path(tmp_restore_path, sizeof(tmp_restore_path),
                               "gv_test_restore", ".db") != 0) {
        return -1;
    }
    if (gv_test_make_temp_path(tmp_incr_path, sizeof(tmp_incr_path),
                               "gv_test_incr", ".bak") != 0) {
        return -1;
    }
    if (gv_test_make_temp_path(tmp_merged_path, sizeof(tmp_merged_path),
                               "gv_test_merged", ".bak") != 0) {
        return -1;
    }
    if (gv_test_make_temp_path(tmp_missing_db_path, sizeof(tmp_missing_db_path),
                               "gv_test_missing_db", ".db") != 0) {
        return -1;
    }
    if (gv_test_make_temp_path(tmp_missing_bak_path, sizeof(tmp_missing_bak_path),
                               "gv_test_missing_bak", ".bak") != 0) {
        return -1;
    }
    tmp_paths_ready = 1;
    return 0;
}

static void cleanup_temp_files(void) {
    if (!tmp_paths_ready) {
        return;
    }
    unlink(tmp_backup_path);
    unlink(tmp_restore_path);
    unlink(tmp_incr_path);
    unlink(tmp_merged_path);
    unlink(tmp_missing_db_path);
    unlink(tmp_missing_bak_path);
}

static int test_backup_options_init(void) {
    GV_BackupOptions opts;
    memset(&opts, 0xFF, sizeof(opts));
    backup_options_init(&opts);
    ASSERT(opts.compression == GV_BACKUP_COMPRESS_NONE, "default compression should be NONE");
    ASSERT(opts.include_wal == 1, "default include_wal should be 1");
    ASSERT(opts.include_metadata == 1, "default include_metadata should be 1");
    ASSERT(opts.verify_after == 1, "default verify_after should be 1");
    ASSERT(opts.encryption_key == NULL, "default encryption_key should be NULL");
    return 0;
}

static int test_restore_options_init(void) {
    GV_RestoreOptions opts;
    memset(&opts, 0xFF, sizeof(opts));
    restore_options_init(&opts);
    ASSERT(opts.overwrite == 0, "default overwrite should be 0");
    ASSERT(opts.verify_checksum == 1, "default verify_checksum should be 1");
    ASSERT(opts.decryption_key == NULL, "default decryption_key should be NULL");
    return 0;
}

static int test_compression_string(void) {
    const char *s;

    s = backup_compression_string(GV_BACKUP_COMPRESS_NONE);
    ASSERT(s != NULL, "compression string for NONE should not be NULL");

    s = backup_compression_string(GV_BACKUP_COMPRESS_ZLIB);
    ASSERT(s != NULL, "compression string for ZLIB should not be NULL");

    s = backup_compression_string(GV_BACKUP_COMPRESS_LZ4);
    ASSERT(s != NULL, "compression string for LZ4 should not be NULL");

    return 0;
}

static int test_result_free_null(void) {
    backup_result_free(NULL);
    return 0;
}

static int test_backup_create_nonexistent(void) {
    cleanup_temp_files();
    ASSERT(ensure_temp_paths() == 0, "temp paths");
    GV_BackupOptions opts;
    backup_options_init(&opts);

    GV_BackupResult *result = backup_create_from_file(
        tmp_missing_db_path, tmp_backup_path, &opts, NULL, NULL);
    /* With a non-existent source, we expect either NULL or result->success == 0 */
    if (result != NULL) {
        ASSERT(result->success == 0, "backup of non-existent file should fail");
        backup_result_free(result);
    }
    cleanup_temp_files();
    return 0;
}

static int test_read_header_nonexistent(void) {
    ASSERT(ensure_temp_paths() == 0, "temp paths");
    GV_BackupHeader header;
    memset(&header, 0, sizeof(header));
    int rc = backup_read_header(tmp_missing_bak_path, &header);
    ASSERT(rc == -1, "read header on non-existent file should return -1");
    return 0;
}

static int test_verify_nonexistent(void) {
    ASSERT(ensure_temp_paths() == 0, "temp paths");
    GV_BackupResult *result = backup_verify(tmp_missing_bak_path, NULL);
    if (result != NULL) {
        ASSERT(result->success == 0, "verify non-existent backup should fail");
        backup_result_free(result);
    }
    return 0;
}

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
