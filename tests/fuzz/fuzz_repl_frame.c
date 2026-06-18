/**
 * @file fuzz_repl_frame.c
 * @brief libFuzzer harness for TCP replication wire frames.
 */

#include <stddef.h>
#include "core/memory.h"
#include <stdint.h>
#include <stdlib.h>

#include "admin/repl_transport.h"
#include "storage/wal.h"

#define FUZZ_MAX_INPUT (256 * 1024)
#define FUZZ_MAX_FRAME (16 * 1024 * 1024)
#define FUZZ_DIM 8

static int noop_insert(void *ctx, const float *data, size_t dimension,
                       const char *const *metadata_keys,
                       const char *const *metadata_values,
                       size_t metadata_count) {
    (void)ctx;
    (void)data;
    (void)dimension;
    (void)metadata_keys;
    (void)metadata_values;
    (void)metadata_count;
    return 0;
}

static int noop_delete(void *ctx, size_t vector_index) {
    (void)ctx;
    (void)vector_index;
    return 0;
}

static int noop_update(void *ctx, size_t vector_index, const float *data, size_t dimension,
                       const char *const *metadata_keys, const char *const *metadata_values,
                       size_t metadata_count) {
    (void)ctx;
    (void)vector_index;
    (void)data;
    (void)dimension;
    (void)metadata_keys;
    (void)metadata_values;
    (void)metadata_count;
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;

    uint8_t msg_type = 0;
    uint32_t request_id = 0;
    uint8_t *payload = NULL;
    size_t payload_len = 0;

    if (repl_parse_frame_buffer(data, size, FUZZ_MAX_FRAME,
                                &msg_type, &request_id, &payload, &payload_len) != 0) {
        return 0;
    }

    if (msg_type == REPL_MSG_WAL && payload_len >= 12) {
        uint32_t record_len = ((uint32_t)payload[8] << 24) | ((uint32_t)payload[9] << 16) |
                              ((uint32_t)payload[10] << 8) | (uint32_t)payload[11];
        if (record_len <= 1024 * 1024 &&
            (size_t)(12 + record_len) <= payload_len) {
            (void)wal_apply_record_buffer(payload + 12, record_len, 1, FUZZ_DIM,
                                          noop_insert, noop_delete, noop_update, NULL, NULL);
        }
    }

    gv_free(payload);
    (void)request_id;
    return 0;
}

#ifdef GV_FUZZ_STANDALONE
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input-file>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint8_t buf[FUZZ_MAX_INPUT];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    return LLVMFuzzerTestOneInput(buf, n);
}
#endif
