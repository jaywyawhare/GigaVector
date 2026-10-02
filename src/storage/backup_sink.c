#define _GNU_SOURCE
#include "storage/backup_sink.h"
#include "core/memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif

int db_backup_to_sink(GV_Database *db, const char *object_name,
                      const GV_BackupOptions *options, const GV_BackupSink *sink) {
    if (!db || !object_name || !sink || !sink->put) return -1;

    char tmpl[] = "/tmp/gv_backup_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return -1;
    close(fd);
    unlink(tmpl);   /* reserve a unique name; backup_create creates the file itself */

    GV_BackupResult *res = backup_create(db, tmpl, options, NULL, NULL);
    int ok = res && res->success;
    if (res) backup_result_free(res);
    if (!ok) { unlink(tmpl); return -1; }

    int up = sink->put(sink->ctx, object_name, tmpl);
    unlink(tmpl);
    return up == 0 ? 0 : -1;
}

/* Reject object names that could break out of the command template. Operators
 * supply trusted keys, but this is a cheap guard against accidental injection. */
static int name_is_safe(const char *s) {
    if (!s || !*s) return 0;
    for (const char *p = s; *p; p++) {
        char c = *p;
        int ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                 (c >= '0' && c <= '9') || c == '/' || c == '.' ||
                 c == '-' || c == '_' || c == ':';
        if (!ok) return 0;
    }
    return 1;
}

/* Expand a "%f"/"%o" template into out; returns 0 on success, -1 if it overflows. */
static int expand_template(const char *tmpl, const char *file, const char *object,
                           char *out, size_t out_size) {
    size_t o = 0;
    for (const char *p = tmpl; *p; p++) {
        const char *sub = NULL;
        if (p[0] == '%' && p[1] == 'f') { sub = file; p++; }
        else if (p[0] == '%' && p[1] == 'o') { sub = object; p++; }
        if (sub) {
            size_t n = strlen(sub);
            if (o + n >= out_size) return -1;
            memcpy(out + o, sub, n); o += n;
        } else {
            if (o + 1 >= out_size) return -1;
            out[o++] = *p;
        }
    }
    out[o] = '\0';
    return 0;
}

int gv_backup_command_put(void *ctx, const char *object_name, const char *local_path) {
    const char *tmpl = (const char *)ctx;
    if (!tmpl || !object_name || !local_path) return -1;
    if (!name_is_safe(object_name)) return -1;

    char cmd[4096];
    if (expand_template(tmpl, local_path, object_name, cmd, sizeof(cmd)) != 0) return -1;

    /* NOLINTNEXTLINE(cert-env33-c): operator-configured backup command; object_name is validated by name_is_safe(). */
    int rc = system(cmd);
    if (rc == -1) return -1;
#ifdef WIFEXITED
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
#endif
    /* cppcheck-suppress identicalConditionAfterEarlyExit */
    return rc;
}
