#!/usr/bin/env bash
# Generate libFuzzer seed corpora from known-good encodings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/tests/fuzz/corpus/grpc" "$ROOT/tests/fuzz/corpus/wal" \
         "$ROOT/tests/fuzz/corpus/repl" "$ROOT/tests/fuzz/corpus/wal_files" \
         "$ROOT/tests/fuzz/corpus/posting" "$ROOT/tests/fuzz/corpus/compression" \
         "$ROOT/tests/fuzz/corpus/backup" "$ROOT/tests/fuzz/corpus/rest" \
         "$ROOT/tests/fuzz/corpus/vlog"

make -C "$ROOT" lib >/dev/null

TMP_ROOT="${TMPDIR:-${TEMP:-/tmp}}"
GEN_SRC="$(mktemp "$TMP_ROOT/gv_gen_corpus.XXXXXX.c")"
GEN_BIN="$(mktemp "$TMP_ROOT/gv_gen_corpus.XXXXXX")"
trap 'rm -f "$GEN_SRC" "$GEN_BIN"' EXIT

cat > "$GEN_SRC" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "core/memory.h"
#include "api/grpc.h"
#include "admin/repl_transport.h"
#include "storage/wal.h"
#include "storage/posting_list.h"
#include "storage/compression.h"
#include "storage/backup.h"
#include "storage/vlog.h"
#include "storage/database.h"

static void write_u32_be(uint8_t *buf, uint32_t val) {
    buf[0] = (uint8_t)(val >> 24);
    buf[1] = (uint8_t)(val >> 16);
    buf[2] = (uint8_t)(val >> 8);
    buf[3] = (uint8_t)(val);
}

static void write_u64_be(uint8_t *buf, uint64_t val) {
    write_u32_be(buf, (uint32_t)(val >> 32));
    write_u32_be(buf + 4, (uint32_t)(val & 0xFFFFFFFFu));
}

static void write_file(const char *path, const uint8_t *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(data, 1, len, f);
    fclose(f);
}

/* Copy an existing file's raw bytes into a corpus path (libc malloc/free only;
 * these are our own buffers, never gv_alloc'd). */
static void copy_file(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) return;
    fseek(in, 0, SEEK_END);
    long sz = ftell(in);
    fseek(in, 0, SEEK_SET);
    if (sz > 0) {
        uint8_t *data = (uint8_t *)malloc((size_t)sz);
        if (data && fread(data, 1, (size_t)sz, in) == (size_t)sz) {
            write_file(dst, data, (size_t)sz);
        }
        free(data);
    }
    fclose(in);
}

static void make_wal_tmp(char *buf, size_t size) {
    const char *root = getenv("TMPDIR");
    if (!root || !*root) root = getenv("TEMP");
    if (!root || !*root) root = "/tmp";
    snprintf(buf, size, "%s/gv_corpus_seed_%lu.wal", root, (unsigned long)getpid());
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    const char *root = argv[1];

    char grpc_path[512];
    snprintf(grpc_path, sizeof(grpc_path), "%s/tests/fuzz/corpus/grpc/search_valid.bin", root);

    float query[4] = {1.f, 0.f, 0.f, 0.f};
    uint8_t buf[256];
    size_t out_len = 0;
    if (grpc_encode_search_request(query, 4, 10, 0, buf, sizeof(buf), &out_len) == 0) {
        write_file(grpc_path, buf, out_len);
    }

    char grpc_frame_path[512];
    snprintf(grpc_frame_path, sizeof(grpc_frame_path),
             "%s/tests/fuzz/corpus/grpc/search_frame.bin", root);
    if (out_len > 0) {
        uint32_t frame_len = (uint32_t)(5 + out_len);
        uint8_t frame[512];
        write_u32_be(frame, frame_len);
        frame[4] = 2; /* GV_MSG_SEARCH */
        write_u32_be(frame + 5, 1);
        memcpy(frame + 9, buf, out_len);
        write_file(grpc_frame_path, frame, 4 + frame_len);
    }

    char wal_tmp[512];
    make_wal_tmp(wal_tmp, sizeof(wal_tmp));
    remove(wal_tmp);
    GV_WAL *wal = wal_open(wal_tmp, 4, 0);
    if (wal) {
        float v[4] = {0.5f, 0.5f, 0.f, 0.f};
        wal_append_insert(wal, v, 4, "tag", "seed");
        wal_close(wal);

        uint8_t type = 0;
        uint8_t *record = NULL;
        size_t record_len = 0;
        if (wal_read_entry_at(wal_tmp, 0, &type, &record, &record_len) == 0 && record) {
            char repl_path[512];
            snprintf(repl_path, sizeof(repl_path),
                     "%s/tests/fuzz/corpus/repl/wal_frame.bin", root);
            size_t payload_len = 12 + record_len;
            uint8_t *payload = (uint8_t *)malloc(payload_len);
            if (payload) {
                write_u64_be(payload, 0);
                write_u32_be(payload + 8, (uint32_t)record_len);
                memcpy(payload + 12, record, record_len);
                uint32_t frame_len = (uint32_t)(5 + payload_len);
                uint8_t *frame = (uint8_t *)malloc(4 + frame_len);
                if (frame) {
                    write_u32_be(frame, frame_len);
                    frame[4] = 2; /* REPL_MSG_WAL */
                    write_u32_be(frame + 5, 1);
                    memcpy(frame + 9, payload, payload_len);
                    write_file(repl_path, frame, 4 + frame_len);
                    free(frame);
                }
                free(payload);
            }
            gv_free(record); /* wal_read_entry_at returns a gv_alloc'd buffer */
        }

        char wal_path[512];
        snprintf(wal_path, sizeof(wal_path), "%s/tests/fuzz/corpus/wal_files/minimal.wal", root);
        FILE *in = fopen(wal_tmp, "rb");
        if (in) {
            fseek(in, 0, SEEK_END);
            long sz = ftell(in);
            fseek(in, 0, SEEK_SET);
            uint8_t *data = (uint8_t *)malloc((size_t)sz);
            if (data && sz > 0) {
                fread(data, 1, (size_t)sz, in);
                write_file(wal_path, data, (size_t)sz);
            }
            free(data);
            fclose(in);
        }
        remove(wal_tmp);
    }

    {
        float v[4] = {0.25f, 0.5f, 0.75f, 1.f};
        GV_PostingWriteEntry pe = { .vector_id = 1, .version = 1, .flags = 0, .data = v };
        uint8_t *seg = NULL;
        size_t seg_len = 0;
        if (posting_segment_encode(0, 0, &pe, 1, 4, 4096, &seg, &seg_len) == 0 && seg) {
            char posting_path[512];
            snprintf(posting_path, sizeof(posting_path),
                     "%s/tests/fuzz/corpus/posting/valid_segment.bin", root);
            write_file(posting_path, seg, seg_len);
            gv_free(seg); /* posting_segment_encode returns a gv_alloc'd buffer */
        }

        GV_PostingSegmentParams sq8 = { .payload_type = GV_POSTING_PAYLOAD_SQ8 };
        if (posting_segment_encode_ex(0, 1, &pe, 1, 4, 4096, &sq8, 0, &seg, &seg_len) == 0 && seg) {
            char sq8_path[512];
            snprintf(sq8_path, sizeof(sq8_path),
                     "%s/tests/fuzz/corpus/posting/valid_sq8_segment.bin", root);
            write_file(sq8_path, seg, seg_len);
            gv_free(seg);
        }
    }

    /* compression: a real compressor-produced blob decompress() accepts. */
    {
        GV_CompressionConfig ccfg;
        compression_config_init(&ccfg);
        ccfg.type = GV_COMPRESS_LZ4;
        GV_Compressor *comp = compression_create(&ccfg);
        if (comp) {
            uint8_t in[512];
            for (size_t i = 0; i < sizeof(in); i++) in[i] = (uint8_t)(i & 0x3f);
            uint8_t outbuf[1024];
            size_t clen = compress(comp, in, sizeof(in), outbuf, sizeof(outbuf));
            if (clen > 0) {
                char cpath[512];
                snprintf(cpath, sizeof(cpath),
                         "%s/tests/fuzz/corpus/compression/valid_blob.bin", root);
                write_file(cpath, outbuf, clen);
            }
            compression_destroy(comp);
        }
    }

    /* backup: a real backup file (valid header + body) via backup_create. */
    {
        char bdb[512], bbk[512];
        snprintf(bdb, sizeof(bdb), "%s/gv_corpus_backup_src_%lu.gv",
                 getenv("TMPDIR") && *getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp",
                 (unsigned long)getpid());
        snprintf(bbk, sizeof(bbk), "%s/gv_corpus_backup_%lu.gvbk",
                 getenv("TMPDIR") && *getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp",
                 (unsigned long)getpid());
        remove(bdb); remove(bbk);
        GV_Database *bdbh = db_open(bdb, 4, GV_INDEX_TYPE_FLAT);
        if (bdbh) {
            GV_BackupResult *br = backup_create(bdbh, bbk, NULL, NULL, NULL);
            if (br) backup_result_free(br);
            db_close(bdbh);
            char bpath[512];
            snprintf(bpath, sizeof(bpath),
                     "%s/tests/fuzz/corpus/backup/valid_backup.bin", root);
            copy_file(bbk, bpath);
        }
        remove(bdb); remove(bbk);
    }

    /* vlog: a tiny valid value log (header + one record). */
    {
        char vlp[512];
        snprintf(vlp, sizeof(vlp), "%s/gv_corpus_vlog_%lu.gvvl",
                 getenv("TMPDIR") && *getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp",
                 (unsigned long)getpid());
        remove(vlp);
        GV_VLog *vl = vlog_open(vlp);
        if (vl) {
            uint64_t off = 0;
            vlog_append(vl, "seed-value", 10, &off);
            vlog_sync(vl);
            vlog_close(vl);
            char vpath[512];
            snprintf(vpath, sizeof(vpath),
                     "%s/tests/fuzz/corpus/vlog/valid.bin", root);
            copy_file(vlp, vpath);
        }
        remove(vlp);
    }

    return 0;
}
EOF

# ── graph WAL seed corpus: valid frames + torn/junk tails ──────────────────
mkdir -p "$ROOT/tests/fuzz/corpus/graph_wal"
python3 - "$ROOT/tests/fuzz/corpus/graph_wal" << 'PYEOF'
import struct, sys, os
out = sys.argv[1]
def frame(op, payload): return bytes([op]) + struct.pack("<I", len(payload)) + payload
def s(x):
    b = x.encode(); return struct.pack("<I", len(b)) + b
open(os.path.join(out, "a_add_node"), "wb").write(frame(1, struct.pack("<Q", 99) + s("Fuzz") + s("T")))
open(os.path.join(out, "b_remove"), "wb").write(frame(2, struct.pack("<Q", 1)))
open(os.path.join(out, "c_add_edge"), "wb").write(frame(3, struct.pack("<QQ", 10, 11) + s("E") + struct.pack("<f", 2.0)))
open(os.path.join(out, "d_set_prop"), "wb").write(frame(5, struct.pack("<Q", 1) + s("k") + s("v")))
open(os.path.join(out, "e_torn"), "wb").write(bytes([5]) + struct.pack("<I", 500) + b"x" * 10)
open(os.path.join(out, "f_junk"), "wb").write(bytes(range(64)))
print("graph WAL corpus written")
PYEOF

# ── REST seed corpus: method byte + URL, optional '\n' + JSON body ─────────
python3 - "$ROOT/tests/fuzz/corpus/rest" << 'PYEOF'
import sys, os
out = sys.argv[1]
# byte 0 = method index mod 6 (GET=0, POST=1); then URL, then '\n' + body.
open(os.path.join(out, "a_health"), "wb").write(b"\x00/health")
open(os.path.join(out, "b_stats"), "wb").write(b"\x00/stats")
open(os.path.join(out, "c_search"), "wb").write(b"\x01/search\n" + b'{"vector":[0,0,0,0],"k":1}')
open(os.path.join(out, "d_vectors"), "wb").write(b"\x01/vectors\n" + b'{"vector":[1,2,3,4]}')
open(os.path.join(out, "e_vec_get"), "wb").write(b"\x00/vectors/0")
print("REST corpus written")
PYEOF

gcc -O2 -I"$ROOT/include" "$GEN_SRC" -L"$ROOT/build/lib" -lGigaVector -lm -pthread \
    -Wl,-rpath,"$ROOT/build/lib" -o "$GEN_BIN"
"$GEN_BIN" "$ROOT"
echo "Corpus written to $ROOT/tests/fuzz/corpus/"
