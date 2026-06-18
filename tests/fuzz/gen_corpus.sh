#!/usr/bin/env bash
# Generate libFuzzer seed corpora from known-good encodings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/tests/fuzz/corpus/grpc" "$ROOT/tests/fuzz/corpus/wal" \
         "$ROOT/tests/fuzz/corpus/repl" "$ROOT/tests/fuzz/corpus/wal_files" \
         "$ROOT/tests/fuzz/corpus/posting"

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
#include "api/grpc.h"
#include "admin/repl_transport.h"
#include "storage/wal.h"
#include "storage/posting_list.h"

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
            free(record);
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
            free(seg);
        }

        GV_PostingSegmentParams sq8 = { .payload_type = GV_POSTING_PAYLOAD_SQ8 };
        if (posting_segment_encode_ex(0, 1, &pe, 1, 4, 4096, &sq8, &seg, &seg_len) == 0 && seg) {
            char sq8_path[512];
            snprintf(sq8_path, sizeof(sq8_path),
                     "%s/tests/fuzz/corpus/posting/valid_sq8_segment.bin", root);
            write_file(sq8_path, seg, seg_len);
            free(seg);
        }
    }
    return 0;
}
EOF

gcc -O2 -I"$ROOT/include" "$GEN_SRC" -L"$ROOT/build/lib" -lGigaVector -lm -pthread \
    -Wl,-rpath,"$ROOT/build/lib" -o "$GEN_BIN"
"$GEN_BIN" "$ROOT"
echo "Corpus written to $ROOT/tests/fuzz/corpus/"
