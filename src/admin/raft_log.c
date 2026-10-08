/*
 * Crash-safe on-disk Raft log (see raft_log.h).
 *
 * File format: a sequence of records, each a 1-byte tag followed by a
 * fixed/variable payload, little-endian:
 *   META  tag=0 : term(u64) voted_for(i32)
 *   ENTRY tag=1 : index(u64) term(u64) len(u64) data[len]
 *   TRUNC tag=2 : keep_upto(u64)
 * open() replays records to rebuild the in-memory state, then rewrites the file
 * compacted (one META + current entries) so it does not grow without bound.
 */

#include "admin/raft_log.h"
#include "core/compat.h"   /* gv_rename_replace (Windows-safe atomic replace) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#define fsync(fd) _commit(fd)
#else
#include <unistd.h>
#endif

struct GV_RaftLog {
    char         *path;
    FILE         *fp;         /* append handle */
    uint64_t      term;
    int           voted_for;
    GV_RaftEntry *entries;    /* index k -> raft index k+1 */
    size_t        count;
    size_t        cap;
    int           failed;     /* a durable write failed; reject further writes */
};

enum { REC_META = 0, REC_ENTRY = 1, REC_TRUNC = 2 };

/* Reject an absurd entry length read from a (possibly corrupt) log file before
 * allocating for it. */
#define RAFT_LOG_MAX_ENTRY (64u * 1024u * 1024u)

/* ---- little-endian helpers ---- */
static int w_u64(FILE *f, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    return fwrite(b, 1, 8, f) == 8 ? 0 : -1;
}
static int w_i32(FILE *f, int32_t v) {
    unsigned char b[4];
    uint32_t u = (uint32_t)v;
    for (int i = 0; i < 4; i++) b[i] = (unsigned char)(u >> (8 * i));
    return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}
static int r_u64(FILE *f, uint64_t *out) {
    unsigned char b[8];
    if (fread(b, 1, 8, f) != 8) return -1;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    *out = v;
    return 0;
}
static int r_i32(FILE *f, int32_t *out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)b[i] << (8 * i);
    *out = (int32_t)v;
    return 0;
}

static int sync_file(FILE *f) {
    if (fflush(f) != 0) return -1;
    return fsync(fileno(f));
}

/* Append an entry to the in-memory vector (takes a copy of data). */
static int mem_append(GV_RaftLog *log, uint64_t term, const void *data, size_t len) {
    if (log->count == log->cap) {
        size_t nc = log->cap ? log->cap * 2 : 16;
        GV_RaftEntry *ne = (GV_RaftEntry *)realloc(log->entries, nc * sizeof(GV_RaftEntry));
        if (!ne) return -1;
        log->entries = ne;
        log->cap = nc;
    }
    void *copy = NULL;
    if (len) {
        copy = malloc(len);
        if (!copy) return -1;
        memcpy(copy, data, len);
    }
    log->entries[log->count].term = term;
    log->entries[log->count].data = copy;
    log->entries[log->count].len = len;
    log->count++;
    return 0;
}

static void mem_truncate(GV_RaftLog *log, uint64_t keep) {
    while (log->count > keep) {
        log->count--;
        free(log->entries[log->count].data);
        log->entries[log->count].data = NULL;
    }
}

static void mem_clear(GV_RaftLog *log) {
    for (size_t i = 0; i < log->count; i++) free(log->entries[i].data);
    free(log->entries);
    log->entries = NULL;
    log->count = 0;
    log->cap = 0;
}

/* Replay every record in `fp` (positioned at start) into the in-memory state. */
static void replay(GV_RaftLog *log, FILE *fp) {
    int tag;
    while ((tag = fgetc(fp)) != EOF) {
        if (tag == REC_META) {
            uint64_t t; int32_t v;
            if (r_u64(fp, &t) != 0 || r_i32(fp, &v) != 0) break;
            log->term = t; log->voted_for = (int)v;
        } else if (tag == REC_ENTRY) {
            uint64_t idx, term, len;
            if (r_u64(fp, &idx) != 0 || r_u64(fp, &term) != 0 || r_u64(fp, &len) != 0) break;
            /* Entries are appended with monotonically increasing indices; a gap
             * or an out-of-range length means the tail is corrupt — stop and
             * keep the valid prefix. Bound len before malloc to resist a
             * corrupt/hostile log driving a huge or overflowing allocation. */
            if (idx != log->count + 1 || len > RAFT_LOG_MAX_ENTRY) break;
            void *buf = NULL;
            if (len) {
                buf = malloc(len);
                if (!buf) break;
                if (fread(buf, 1, len, fp) != len) { free(buf); break; }
            }
            if (mem_append(log, term, buf, len) != 0) { free(buf); break; }
            free(buf);
        } else if (tag == REC_TRUNC) {
            uint64_t keep;
            if (r_u64(fp, &keep) != 0) break;
            mem_truncate(log, keep);
        } else {
            break; /* unknown/corrupt tag: stop, keep what we have */
        }
    }
}

/* Rewrite `path` with a compacted image (one META + all current entries). */
static int compact_rewrite(GV_RaftLog *log) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp", log->path);
    FILE *out = fopen(tmp, "wb");
    if (!out) return -1;
    int ok = 1;
    if (fputc(REC_META, out) == EOF || w_u64(out, log->term) != 0 || w_i32(out, log->voted_for) != 0)
        ok = 0;
    for (size_t i = 0; ok && i < log->count; i++) {
        if (fputc(REC_ENTRY, out) == EOF ||
            w_u64(out, i + 1) != 0 ||
            w_u64(out, log->entries[i].term) != 0 ||
            w_u64(out, log->entries[i].len) != 0 ||
            (log->entries[i].len &&
             fwrite(log->entries[i].data, 1, log->entries[i].len, out) != log->entries[i].len))
            ok = 0;
    }
    if (ok && sync_file(out) != 0) ok = 0;
    fclose(out);
    if (!ok) { remove(tmp); return -1; }
    /* gv_rename_replace overwrites an existing destination; plain rename() fails
     * on Windows when log->path already exists, which broke reopen there. */
    if (gv_rename_replace(tmp, log->path) != 0) { remove(tmp); return -1; }
    return 0;
}

GV_RaftLog *raft_log_open(const char *path) {
    if (!path) return NULL;
    GV_RaftLog *log = (GV_RaftLog *)calloc(1, sizeof(*log));
    if (!log) return NULL;
    log->voted_for = -1;
    log->path = strdup(path);
    if (!log->path) { free(log); return NULL; }

    FILE *in = fopen(path, "rb");
    if (in) {
        replay(log, in);
        fclose(in);
    }
    /* Compact the on-disk image so truncations do not accumulate forever. */
    if (compact_rewrite(log) != 0) {
        mem_clear(log);
        free(log->path);
        free(log);
        return NULL;
    }
    /* Open an append handle for subsequent records. */
    log->fp = fopen(path, "ab");
    if (!log->fp) {
        mem_clear(log);
        free(log->path);
        free(log);
        return NULL;
    }
    return log;
}

void raft_log_close(GV_RaftLog *log) {
    if (!log) return;
    if (log->fp) fclose(log->fp);
    mem_clear(log);
    free(log->path);
    free(log);
}

/* Durability ordering: persist the record (write + fsync) BEFORE mutating the
 * in-memory state, so a failed write never leaves memory and disk disagreeing.
 * Any write failure latches `failed`, after which every write is rejected (the
 * on-disk tail may be torn and must not be appended past). */
int raft_log_set_meta(GV_RaftLog *log, uint64_t term, int voted_for) {
    if (!log) return -1;
    if (log->failed) return -1;
    if (fputc(REC_META, log->fp) == EOF || w_u64(log->fp, term) != 0 ||
        w_i32(log->fp, voted_for) != 0 || sync_file(log->fp) != 0) {
        log->failed = 1;
        return -1;
    }
    log->term = term;
    log->voted_for = voted_for;
    return 0;
}

int raft_log_append(GV_RaftLog *log, uint64_t index, uint64_t term,
                    const void *data, size_t len) {
    if (!log) return -1;
    if (log->failed) return -1;
    if (fputc(REC_ENTRY, log->fp) == EOF || w_u64(log->fp, index) != 0 ||
        w_u64(log->fp, term) != 0 || w_u64(log->fp, len) != 0 ||
        (len && fwrite(data, 1, len, log->fp) != len) || sync_file(log->fp) != 0) {
        log->failed = 1;
        return -1;
    }
    return mem_append(log, term, data, len);
}

int raft_log_truncate(GV_RaftLog *log, uint64_t keep_upto) {
    if (!log) return -1;
    if (log->failed) return -1;
    if (fputc(REC_TRUNC, log->fp) == EOF || w_u64(log->fp, keep_upto) != 0 ||
        sync_file(log->fp) != 0) {
        log->failed = 1;
        return -1;
    }
    mem_truncate(log, keep_upto);
    return 0;
}

uint64_t raft_log_term(const GV_RaftLog *log) { return log ? log->term : 0; }
int      raft_log_voted_for(const GV_RaftLog *log) { return log ? log->voted_for : -1; }
size_t   raft_log_count(const GV_RaftLog *log) { return log ? log->count : 0; }
const GV_RaftEntry *raft_log_entries(const GV_RaftLog *log) { return log ? log->entries : NULL; }
