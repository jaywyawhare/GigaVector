#ifndef GIGAVECTOR_GV_BACKUP_SINK_H
#define GIGAVECTOR_GV_BACKUP_SINK_H

#include <stddef.h>

#include "storage/backup.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file backup_sink.h
 * @brief Pluggable backup destination so backups can target object storage
 *        (S3 / GCS / Azure / any HTTP store) instead of only the local disk.
 */

/**
 * @brief Upload callback: publish the local backup file @p local_path under the
 *        logical name @p object_name. Return 0 on success, non-zero on failure.
 */
typedef int (*GV_BackupPutFn)(void *ctx, const char *object_name, const char *local_path);

typedef struct {
    GV_BackupPutFn put;   /**< Uploader; must be non-NULL. */
    void          *ctx;   /**< Opaque context passed to @ref put. */
} GV_BackupSink;

/**
 * @brief Back up a database and hand the artifact to an object-storage sink.
 *
 * Writes a normal backup to a temporary local file, invokes @p sink to upload it
 * under @p object_name, then removes the temp file. This keeps the (SDK-specific,
 * network) upload out of the core while still letting backups reach the cloud.
 *
 * @param db          Database to back up.
 * @param object_name Destination object key (e.g. "snapshots/2026-07-25.gvbak").
 * @param options     Backup options (NULL for defaults).
 * @param sink        Upload sink; @p sink->put must be non-NULL.
 * @return 0 on success, -1 on backup or upload failure.
 */
int db_backup_to_sink(GV_Database *db, const char *object_name,
                      const GV_BackupOptions *options, const GV_BackupSink *sink);

/**
 * @brief Built-in command-template sink for @ref GV_BackupSink.
 *
 * @p ctx is a command-template string where "%f" expands to the local file path
 * and "%o" to the object name, e.g. "aws s3 cp %f s3://bucket/%o" or
 * "gsutil cp %f gs://bucket/%o". Runs via the shell. For safety the object name
 * is rejected if it contains shell metacharacters - operators supply trusted
 * templates and keys, not end-user input.
 *
 * @return 0 if the command exits 0, non-zero otherwise.
 */
int gv_backup_command_put(void *ctx, const char *object_name, const char *local_path);

#ifdef __cplusplus
}
#endif

#endif
