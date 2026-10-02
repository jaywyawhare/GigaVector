/*
 * vlog.c — WiscKey-style append-only value log with garbage collection.
 *
 * See include/storage/vlog.h for the on-disk layout and API contract.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "storage/vlog.h"
#include "core/utils.h"
#include "core/compat.h"   /* gv_rename_replace (Windows-safe atomic publish) */

#if defined(_WIN32)
#include <io.h>
#define vlog_fsync(f) _commit(_fileno(f))
#else
#include <unistd.h>
#define vlog_fsync(f) fsync(fileno(f))
#endif

#define VLOG_MAGIC       "GVVL"
#define VLOG_VERSION     1u
#define VLOG_HEADER_SIZE 16u
#define VLOG_RECHDR_SIZE 8u   /* [u32 length][u32 crc] */
/* Sanity cap on a single value to reject corrupt length fields (256 MiB). */
#define VLOG_MAX_VALUE   (256u * 1024u * 1024u)

struct GV_VLog {
    FILE       *fp;
    char       *path;
    uint64_t    write_pos;   /* offset of EOF (next append lands here) */
    size_t      record_count;
    pthread_mutex_t mutex;
};

static uint32_t vlog_crc(const void *data, size_t len) {
    uint32_t c = gv_crc32_init();
    c = gv_crc32_update(c, data, len);
    return gv_crc32_finish(c);
}

/* Write the 16-byte header at the start of a freshly created file. */
static int vlog_write_header(FILE *fp) {
    unsigned char hdr[VLOG_HEADER_SIZE];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, VLOG_MAGIC, 4);
    gv_put_u32(hdr + 4, VLOG_VERSION);
    if (fseek(fp, 0, SEEK_SET) != 0) return -1;
    if (fwrite(hdr, 1, VLOG_HEADER_SIZE, fp) != VLOG_HEADER_SIZE) return -1;
    return 0;
}

/* Validate an existing file's header; returns 0 if valid. */
static int vlog_check_header(FILE *fp) {
    unsigned char hdr[VLOG_HEADER_SIZE];
    if (fseek(fp, 0, SEEK_SET) != 0) return -1;
    if (fread(hdr, 1, VLOG_HEADER_SIZE, fp) != VLOG_HEADER_SIZE) return -1;
    if (memcmp(hdr, VLOG_MAGIC, 4) != 0) return -1;
    if (gv_get_u32(hdr + 4) != VLOG_VERSION) return -1;
    return 0;
}

/* Count valid records by walking the file; also detects a truncated tail. */
static size_t vlog_scan_count(FILE *fp, uint64_t file_size, uint64_t *valid_end_out) {
    size_t count = 0;
    uint64_t pos = VLOG_HEADER_SIZE;
    unsigned char rh[VLOG_RECHDR_SIZE];
    while (pos + VLOG_RECHDR_SIZE <= file_size) {
        if (fseek(fp, (long)pos, SEEK_SET) != 0) break;
        if (fread(rh, 1, VLOG_RECHDR_SIZE, fp) != VLOG_RECHDR_SIZE) break;
        uint32_t len = gv_get_u32(rh);
        if (len > VLOG_MAX_VALUE) break;
        uint64_t next = pos + VLOG_RECHDR_SIZE + len;
        if (next > file_size) break;   /* truncated tail record */
        count++;
        pos = next;
    }
    /* `pos` is the end of the last fully-valid record — the position new appends
     * must start from so a torn trailing record is overwritten rather than left
     * mid-log (which a later scan would misparse against following data). */
    if (valid_end_out) *valid_end_out = pos;
    return count;
}

GV_VLog *vlog_open(const char *path) {
    if (!path) return NULL;
    GV_VLog *vl = (GV_VLog *)calloc(1, sizeof(GV_VLog));
    if (!vl) return NULL;
    vl->path = gv_dup_cstr(path);
    if (!vl->path) { free(vl); return NULL; }

    FILE *fp = fopen(path, "r+b");
    if (fp) {
        if (vlog_check_header(fp) != 0) { fclose(fp); free(vl->path); free(vl); return NULL; }
        if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); free(vl->path); free(vl); return NULL; }
        long sz = ftell(fp);
        if (sz < (long)VLOG_HEADER_SIZE) sz = (long)VLOG_HEADER_SIZE;
        uint64_t valid_end = VLOG_HEADER_SIZE;
        vl->record_count = vlog_scan_count(fp, (uint64_t)sz, &valid_end);
        /* Start appends at the last valid record boundary, not the raw file end:
         * a torn trailing record must be overwritten. Drop it durably too so a
         * later scan can't misparse the garbage against records appended after. */
        if (valid_end < (uint64_t)sz) {
            fflush(fp);
            if (ftruncate(fileno(fp), (off_t)valid_end) != 0) {
                /* Truncate failed (e.g. read-only fs): still position appends at
                 * valid_end so this session overwrites the torn tail. */
            }
            (void)fseek(fp, (long)valid_end, SEEK_SET);
        }
        vl->write_pos = valid_end;
    } else {
        fp = fopen(path, "w+b");
        if (!fp) { free(vl->path); free(vl); return NULL; }
        if (vlog_write_header(fp) != 0) { fclose(fp); free(vl->path); free(vl); return NULL; }
        vl->write_pos = VLOG_HEADER_SIZE;
        vl->record_count = 0;
    }
    vl->fp = fp;
    pthread_mutex_init(&vl->mutex, NULL);
    return vl;
}

int vlog_append(GV_VLog *vl, const void *value, size_t len, uint64_t *offset_out) {
    if (!vl || !vl->fp || !offset_out || (len && !value) || len > VLOG_MAX_VALUE) return -1;
    unsigned char rh[VLOG_RECHDR_SIZE];
    gv_put_u32(rh, (uint32_t)len);
    gv_put_u32(rh + 4, len ? vlog_crc(value, len) : 0u);

    pthread_mutex_lock(&vl->mutex);
    if (fseek(vl->fp, (long)vl->write_pos, SEEK_SET) != 0) { pthread_mutex_unlock(&vl->mutex); return -1; }
    if (fwrite(rh, 1, VLOG_RECHDR_SIZE, vl->fp) != VLOG_RECHDR_SIZE) { pthread_mutex_unlock(&vl->mutex); return -1; }
    if (len && fwrite(value, 1, len, vl->fp) != len) { pthread_mutex_unlock(&vl->mutex); return -1; }
    if (fflush(vl->fp) != 0) { pthread_mutex_unlock(&vl->mutex); return -1; }
    *offset_out = vl->write_pos;
    vl->write_pos += VLOG_RECHDR_SIZE + len;
    vl->record_count++;
    pthread_mutex_unlock(&vl->mutex);
    return 0;
}

int vlog_read(GV_VLog *vl, uint64_t offset, void **value_out, size_t *len_out) {
    if (!vl || !vl->fp || !len_out || offset < VLOG_HEADER_SIZE) return -1;
    pthread_mutex_lock(&vl->mutex);
    if (offset + VLOG_RECHDR_SIZE > vl->write_pos) { pthread_mutex_unlock(&vl->mutex); return -1; }
    unsigned char rh[VLOG_RECHDR_SIZE];
    if (fseek(vl->fp, (long)offset, SEEK_SET) != 0) { pthread_mutex_unlock(&vl->mutex); return -1; }
    if (fread(rh, 1, VLOG_RECHDR_SIZE, vl->fp) != VLOG_RECHDR_SIZE) { pthread_mutex_unlock(&vl->mutex); return -1; }
    uint32_t len = gv_get_u32(rh);
    uint32_t crc = gv_get_u32(rh + 4);
    if (len > VLOG_MAX_VALUE || offset + VLOG_RECHDR_SIZE + len > vl->write_pos) {
        pthread_mutex_unlock(&vl->mutex); return -1;
    }
    void *buf = NULL;
    if (len) {
        buf = malloc(len);
        if (!buf) { pthread_mutex_unlock(&vl->mutex); return -1; }
        if (fread(buf, 1, len, vl->fp) != len) { free(buf); pthread_mutex_unlock(&vl->mutex); return -1; }
        if (vlog_crc(buf, len) != crc) { free(buf); pthread_mutex_unlock(&vl->mutex); return -1; }
    }
    pthread_mutex_unlock(&vl->mutex);
    if (value_out) *value_out = buf; else free(buf);
    *len_out = len;
    return 0;
}

int vlog_sync(GV_VLog *vl) {
    if (!vl || !vl->fp) return -1;
    pthread_mutex_lock(&vl->mutex);
    int rc = (fflush(vl->fp) == 0 && vlog_fsync(vl->fp) == 0) ? 0 : -1;
    pthread_mutex_unlock(&vl->mutex);
    return rc;
}

uint64_t vlog_size(const GV_VLog *vl) { return vl ? vl->write_pos : 0; }
size_t vlog_record_count(const GV_VLog *vl) { return vl ? vl->record_count : 0; }

int vlog_scan(GV_VLog *vl, GV_VLogScanFn cb, void *ctx) {
    if (!vl || !vl->fp || !cb) return -1;
    pthread_mutex_lock(&vl->mutex);
    int stopped = 0;
    uint64_t pos = VLOG_HEADER_SIZE;
    unsigned char rh[VLOG_RECHDR_SIZE];
    while (pos + VLOG_RECHDR_SIZE <= vl->write_pos) {
        if (fseek(vl->fp, (long)pos, SEEK_SET) != 0) { pthread_mutex_unlock(&vl->mutex); return -1; }
        if (fread(rh, 1, VLOG_RECHDR_SIZE, vl->fp) != VLOG_RECHDR_SIZE) { pthread_mutex_unlock(&vl->mutex); return -1; }
        uint32_t len = gv_get_u32(rh);
        uint32_t crc = gv_get_u32(rh + 4);
        if (len > VLOG_MAX_VALUE || pos + VLOG_RECHDR_SIZE + len > vl->write_pos) break;
        void *buf = NULL;
        if (len) {
            buf = malloc(len);
            if (!buf) { pthread_mutex_unlock(&vl->mutex); return -1; }
            if (fread(buf, 1, len, vl->fp) != len || vlog_crc(buf, len) != crc) { free(buf); pthread_mutex_unlock(&vl->mutex); return -1; }
        }
        int rc = cb(pos, buf, len, ctx);
        free(buf);
        if (rc) { stopped = 1; break; }
        pos += VLOG_RECHDR_SIZE + len;
    }
    pthread_mutex_unlock(&vl->mutex);
    return stopped ? 1 : 0;
}

int vlog_gc(GV_VLog *vl, GV_VLogLiveFn live, void *ctx,
            uint64_t **old_offsets_out, uint64_t **new_offsets_out,
            size_t *n_remapped_out) {
    if (!vl || !vl->fp || !live || !n_remapped_out) return -1;

    pthread_mutex_lock(&vl->mutex);

    char *tmp_path = (char *)malloc(strlen(vl->path) + 5);
    if (!tmp_path) { pthread_mutex_unlock(&vl->mutex); return -1; }
    sprintf(tmp_path, "%s.gc", vl->path);

    FILE *out = fopen(tmp_path, "w+b");
    if (!out) { free(tmp_path); pthread_mutex_unlock(&vl->mutex); return -1; }
    if (vlog_write_header(out) != 0) { fclose(out); remove(tmp_path); free(tmp_path); pthread_mutex_unlock(&vl->mutex); return -1; }

    uint64_t out_pos = VLOG_HEADER_SIZE;
    size_t cap = 0, n = 0;
    uint64_t *old_arr = NULL, *new_arr = NULL;
    size_t new_count = 0;
    int err = 0;

    uint64_t pos = VLOG_HEADER_SIZE;
    unsigned char rh[VLOG_RECHDR_SIZE];
    while (pos + VLOG_RECHDR_SIZE <= vl->write_pos) {
        if (fseek(vl->fp, (long)pos, SEEK_SET) != 0) { err = 1; break; }
        if (fread(rh, 1, VLOG_RECHDR_SIZE, vl->fp) != VLOG_RECHDR_SIZE) { err = 1; break; }
        uint32_t len = gv_get_u32(rh);
        if (len > VLOG_MAX_VALUE || pos + VLOG_RECHDR_SIZE + len > vl->write_pos) break;

        if (live(pos, ctx)) {
            /* Copy the whole record (header + payload) verbatim to the new log. */
            void *buf = malloc(VLOG_RECHDR_SIZE + len);
            if (!buf) { err = 1; break; }
            memcpy(buf, rh, VLOG_RECHDR_SIZE);
            if (len && fread((unsigned char *)buf + VLOG_RECHDR_SIZE, 1, len, vl->fp) != len) { free(buf); err = 1; break; }
            if (fwrite(buf, 1, VLOG_RECHDR_SIZE + len, out) != VLOG_RECHDR_SIZE + len) { free(buf); err = 1; break; }
            free(buf);

            if (n == cap) {
                size_t ncap = cap ? cap * 2 : 16;
                uint64_t *o2 = (uint64_t *)realloc(old_arr, ncap * sizeof(uint64_t));
                uint64_t *n2 = (uint64_t *)realloc(new_arr, ncap * sizeof(uint64_t));
                if (!o2 || !n2) { free(o2 ? o2 : old_arr); free(n2 ? n2 : new_arr); old_arr = NULL; new_arr = NULL; err = 1; break; }
                old_arr = o2; new_arr = n2; cap = ncap;
            }
            old_arr[n] = pos;
            new_arr[n] = out_pos;
            n++;
            out_pos += VLOG_RECHDR_SIZE + len;
            new_count++;
        }
        pos += VLOG_RECHDR_SIZE + len;
    }

    if (err) {
        fclose(out); remove(tmp_path); free(tmp_path);
        free(old_arr); free(new_arr);
        pthread_mutex_unlock(&vl->mutex);
        return -1;
    }

    /* Atomically swap the compacted file in for the old one. fsync the new file's
     * data before the rename so a crash can't leave a renamed-but-unflushed log
     * with zero/garbage records where preserved values should be. */
    if (fflush(out) != 0 || vlog_fsync(out) != 0) { fclose(out); remove(tmp_path); free(tmp_path); free(old_arr); free(new_arr); pthread_mutex_unlock(&vl->mutex); return -1; }
    fclose(out);
    fclose(vl->fp);
    if (gv_rename_replace(tmp_path, vl->path) != 0) {
        remove(tmp_path); free(tmp_path);
        /* Best effort: reopen the original so the handle stays usable. */
        vl->fp = fopen(vl->path, "r+b");
        free(old_arr); free(new_arr);
        pthread_mutex_unlock(&vl->mutex);
        return -1;
    }
    free(tmp_path);

    vl->fp = fopen(vl->path, "r+b");
    if (!vl->fp) { free(old_arr); free(new_arr); pthread_mutex_unlock(&vl->mutex); return -1; }
    vl->write_pos = out_pos;
    vl->record_count = new_count;

    pthread_mutex_unlock(&vl->mutex);

    if (old_offsets_out) *old_offsets_out = old_arr; else free(old_arr);
    if (new_offsets_out) *new_offsets_out = new_arr; else free(new_arr);
    *n_remapped_out = n;
    return 0;
}

void vlog_close(GV_VLog *vl) {
    if (!vl) return;
    if (vl->fp) fclose(vl->fp);
    pthread_mutex_destroy(&vl->mutex);
    free(vl->path);
    free(vl);
}
