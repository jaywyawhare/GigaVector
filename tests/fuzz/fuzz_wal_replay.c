/**
 * @file fuzz_wal_replay.c
 * @brief libFuzzer harness for wal_replay_rich (full WAL file parser).
 *
 * Writes arbitrary input to a temp file and attempts replay with noop callbacks.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/database.h"
#include "storage/wal.h"
#include "../test_tmp.h"

#define FUZZ_MAX_INPUT (64 * 1024)
#define FUZZ_DIM 4

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

    char path[256];
    if (gv_test_make_temp_path(path, sizeof(path), "gv_fuzz_wal", ".wal") != 0) return 0;

    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    if (fwrite(data, 1, size, f) != size) {
        fclose(f);
        remove(path);
        return 0;
    }
    fclose(f);

    (void)wal_replay_rich(path, FUZZ_DIM, noop_insert, noop_delete, noop_update,
                          NULL, NULL, (uint32_t)GV_INDEX_TYPE_FLAT);
    remove(path);
    return 0;
}

#ifdef GV_FUZZ_STANDALONE
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input-file>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint8_t *buf = (uint8_t *)malloc(FUZZ_MAX_INPUT);
    if (!buf) {
        fclose(f);
        return 1;
    }
    size_t n = fread(buf, 1, FUZZ_MAX_INPUT, f);
    fclose(f);
    int rc = LLVMFuzzerTestOneInput(buf, n);
    free(buf);
    return rc;
}
#endif
