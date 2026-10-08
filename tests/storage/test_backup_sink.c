/* Object-storage backup sink + regression for the verify-after checksum bug. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "storage/database.h"
#include "storage/backup.h"
#include "storage/backup_sink.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static int captured = 0;
static char cap_obj[256];
static int my_put(void *ctx, const char *obj, const char *path) {
    (void)ctx; captured = 1; snprintf(cap_obj, sizeof cap_obj, "%s", obj);
    FILE *f = fopen(path, "rb"); if (!f) return 1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fclose(f);
    return sz > 0 ? 0 : 1;
}

int main(void) {
    GV_Database *db = db_open(NULL, 8, GV_INDEX_TYPE_HNSW);
    float v[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < 100; i++) { int _r = db_add_vector(db, v, 8); (void)_r; }

    /* Regression: verify_after=1 (default) must SUCCEED - was always "corrupted". */
    GV_BackupResult *r = backup_create(db, "/tmp/gv_vok.gvbak", NULL, NULL, NULL);
    ASSERT(r && r->success, "backup with default verify_after passes");
    if (r) backup_result_free(r);

    /* Custom sink receives the backup artifact under the object name. */
    GV_BackupSink sink = { my_put, NULL };
    ASSERT(db_backup_to_sink(db, "snap/2026.gvbak", NULL, &sink) == 0, "backup_to_sink ok");
    ASSERT(captured && strcmp(cap_obj, "snap/2026.gvbak") == 0, "sink saw the object name");

    /* Command-template sink (emulates `aws s3 cp` with local cp). */
    GV_BackupSink csink = { gv_backup_command_put, (void *)"cp %f /tmp/gv_out_%o" };
    ASSERT(db_backup_to_sink(db, "cloud.bak", NULL, &csink) == 0, "command sink uploads");

    /* Injection guard: object name with shell metachars is rejected. */
    ASSERT(gv_backup_command_put((void *)"echo %o", "a;rm -rf x", "/tmp/x") != 0, "injection rejected");

    /* Bad args */
    ASSERT(db_backup_to_sink(NULL, "o", NULL, &sink) == -1, "null db -> -1");
    ASSERT(db_backup_to_sink(db, "o", NULL, NULL) == -1, "null sink -> -1");

    db_close(db);
    remove("/tmp/gv_vok.gvbak"); remove("/tmp/gv_out_cloud.bak");
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL BACKUP-SINK TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
