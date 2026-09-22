#ifndef GIGAVECTOR_VLOG_H
#define GIGAVECTOR_VLOG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file vlog.h
 * @brief WiscKey-style append-only value log with garbage collection.
 *
 * Key-value separation (WiscKey): an index keeps small keys mapped to compact
 * value-log offsets, while the (potentially large) values live in this
 * append-only log. That keeps the index dense and cheap to scan/compact while
 * values are written sequentially. Dead values accumulate as keys are updated
 * or deleted and are reclaimed by vlog_gc(), which rewrites only the live
 * records into a fresh log and returns an old->new offset remap so the caller
 * can fix up its index.
 *
 * On-disk layout:
 *   [16-byte header: magic "GVVL", version u32, flags u32, reserved u32]
 *   record* where each record = [u32 length][u32 crc32(value)][value bytes]
 * A record's offset is the byte position of its length field. Offsets are
 * always >= 16, so 0 is a safe "no value" sentinel.
 */

typedef struct GV_VLog GV_VLog;

/** Liveness callback for GC: return non-zero if the record at @p offset is live. */
typedef int (*GV_VLogLiveFn)(uint64_t offset, void *ctx);

/**
 * @brief Open (creating if absent) a value log at @p path.
 * @return Log handle, or NULL on error. Positions the write cursor at EOF.
 */
GV_VLog *vlog_open(const char *path);

/**
 * @brief Append a value; returns its offset for later reads.
 * @param vl Log handle.
 * @param value Value bytes (may be NULL only if len == 0).
 * @param len Value length in bytes.
 * @param offset_out Output: the record offset (>= 16). Must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int vlog_append(GV_VLog *vl, const void *value, size_t len, uint64_t *offset_out);

/**
 * @brief Read the value stored at @p offset, validating its CRC.
 * @param vl Log handle.
 * @param offset Record offset previously returned by vlog_append.
 * @param value_out Output: newly allocated buffer (caller frees with free()); may be NULL for a zero-length value.
 * @param len_out Output: value length. Must be non-NULL.
 * @return 0 on success, -1 on error or CRC mismatch.
 */
int vlog_read(GV_VLog *vl, uint64_t offset, void **value_out, size_t *len_out);

/** @brief Flush and fsync the log to disk. @return 0 on success, -1 on error. */
int vlog_sync(GV_VLog *vl);

/**
 * @brief Iterator callback for vlog_scan. @p value points at a temporary buffer
 * valid only during the call. Return non-zero to stop the scan early.
 */
typedef int (*GV_VLogScanFn)(uint64_t offset, const void *value, size_t len, void *ctx);

/**
 * @brief Walk every record in offset order, invoking @p cb for each.
 *
 * Used to rebuild an external key index after reopening the log (the log is the
 * durable source of truth; the in-memory index is derived from it).
 *
 * @return 0 if the whole log was scanned, 1 if @p cb requested early stop, -1 on error.
 */
int vlog_scan(GV_VLog *vl, GV_VLogScanFn cb, void *ctx);

/** @brief Current total size of the log in bytes (including header). */
uint64_t vlog_size(const GV_VLog *vl);

/** @brief Number of records appended over the life of the log file. */
size_t vlog_record_count(const GV_VLog *vl);

/**
 * @brief Garbage-collect: rewrite only live records into a fresh log.
 *
 * Walks every record, calls @p live(offset, ctx), and copies the live ones into
 * a compacted replacement that atomically supersedes the current file. The
 * old->new offset remap is returned so the caller can update its index.
 *
 * @param vl Log handle (updated in place to point at the compacted file).
 * @param live Liveness predicate; must be non-NULL.
 * @param ctx Opaque context passed to @p live.
 * @param old_offsets_out Output: array of old offsets of surviving records (caller frees); may be NULL.
 * @param new_offsets_out Output: array of their new offsets (caller frees); may be NULL.
 * @param n_remapped_out  Output: number of surviving records. Must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int vlog_gc(GV_VLog *vl, GV_VLogLiveFn live, void *ctx,
            uint64_t **old_offsets_out, uint64_t **new_offsets_out,
            size_t *n_remapped_out);

/** @brief Close the log and free resources. Safe with NULL. */
void vlog_close(GV_VLog *vl);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_VLOG_H */
