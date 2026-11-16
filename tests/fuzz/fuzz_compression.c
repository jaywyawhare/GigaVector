/**
 * @file fuzz_compression.c
 * @brief libFuzzer harness for the DECOMPRESS path (storage/compression.h).
 *
 * The decompressor runs on attacker-controlled compressed bytes whenever the
 * WAL / value-log / snapshot read path restores a compressed block. This feeds
 * RAW fuzz bytes straight into decompress() (no round-trip) so the real codec
 * dispatch -- marker byte, LZ4 / ZSTD / Snappy frame parsers via
 * decompress_core -- is exercised on arbitrary input. Every return code must be
 * handled without UB; no over-read past the bounded output buffer.
 *
 * Build: make fuzz  (requires clang)
 * Run:   build/fuzz/fuzz_compression tests/fuzz/corpus/compression -runs=10000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "storage/compression.h"

#define FUZZ_MAX_INPUT (256 * 1024)
#define FUZZ_OUT_CAP   (1024 * 1024)

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;

    /* Bounded scratch output buffer: a malformed frame that claims a larger
     * decompressed size must be rejected (return 0), never overflow this. */
    uint8_t *out = (uint8_t *)malloc(FUZZ_OUT_CAP);
    if (!out) return 0;

    /* Exercise each codec's decoder: the compressor's configured type selects
     * which decompress_core branch runs on the marker==0x01 path. Cycle the
     * three real codecs plus NONE, derived from the first input byte. */
    const GV_CompressionType types[4] = {
        GV_COMPRESS_NONE, GV_COMPRESS_LZ4, GV_COMPRESS_ZSTD, GV_COMPRESS_SNAPPY
    };

    for (int i = 0; i < 4; i++) {
        GV_CompressionConfig cfg;
        compression_config_init(&cfg);
        cfg.type = types[(data[0] + (unsigned)i) & 3u];

        GV_Compressor *comp = compression_create(&cfg);
        if (!comp) continue;

        /* RAW fuzz bytes as the "compressed" input -- that is the point. */
        (void)decompress(comp, data, size, out, FUZZ_OUT_CAP);

        compression_destroy(comp);
    }

    free(out);
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
    uint8_t buf[FUZZ_MAX_INPUT];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    return LLVMFuzzerTestOneInput(buf, n);
}
#endif
