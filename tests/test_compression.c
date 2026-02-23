#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_compression.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Generate repeatable test data with some redundancy for compression. */
static void fill_test_data(char *buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        buf[i] = (char)('A' + (i % 26));
    }
}

/* 1. test_compression_config_init                                     */
static int test_compression_config_init(void) {
    GV_CompressionConfig config;
    memset(&config, 0xFF, sizeof(config));

    gv_compression_config_init(&config);

    ASSERT(config.type == GV_COMPRESS_LZ4, "default type should be GV_COMPRESS_LZ4");
    ASSERT(config.level == 1, "default level should be 1");
    ASSERT(config.min_size == 64, "default min_size should be 64");

    return 0;
}

/* 2. test_compression_create_destroy                                  */
static int test_compression_create_destroy(void) {
    GV_CompressionConfig config;
    gv_compression_config_init(&config);
    config.type = GV_COMPRESS_LZ4;

    GV_Compressor *comp = gv_compression_create(&config);
    ASSERT(comp != NULL, "gv_compression_create returned NULL");

    gv_compression_destroy(comp);
    return 0;
}

/* 3. test_compress_decompress_lz4                                     */
static int test_compress_decompress_lz4(void) {
    GV_CompressionConfig config;
    gv_compression_config_init(&config);
    config.type = GV_COMPRESS_LZ4;

    GV_Compressor *comp = gv_compression_create(&config);
    ASSERT(comp != NULL, "create failed");

    char input[256];
    fill_test_data(input, sizeof(input));

    size_t bound = gv_compress_bound(comp, sizeof(input));
    ASSERT(bound > 0, "compress_bound should be > 0");

    char *compressed = (char *)malloc(bound);
    ASSERT(compressed != NULL, "malloc failed");

    size_t comp_size = gv_compress(comp, input, sizeof(input), compressed, bound);
    ASSERT(comp_size > 0, "compression failed");

    char decompressed[256];
    size_t decomp_size = gv_decompress(comp, compressed, comp_size,
                                        decompressed, sizeof(decompressed));
    ASSERT(decomp_size == sizeof(input), "decompressed size mismatch");
    ASSERT(memcmp(input, decompressed, sizeof(input)) == 0,
           "decompressed data does not match original");

    free(compressed);
    gv_compression_destroy(comp);
    return 0;
}

/* 4. test_compress_decompress_zstd                                    */
static int test_compress_decompress_zstd(void) {
    GV_CompressionConfig config;
    gv_compression_config_init(&config);
    config.type = GV_COMPRESS_ZSTD;
    config.level = 3;

    GV_Compressor *comp = gv_compression_create(&config);
    ASSERT(comp != NULL, "create failed");

    char input[512];
    fill_test_data(input, sizeof(input));

    size_t bound = gv_compress_bound(comp, sizeof(input));
    ASSERT(bound > 0, "compress_bound should be > 0");

    char *compressed = (char *)malloc(bound);
    ASSERT(compressed != NULL, "malloc failed");

    size_t comp_size = gv_compress(comp, input, sizeof(input), compressed, bound);
    ASSERT(comp_size > 0, "compression failed");

    char decompressed[512];
    size_t decomp_size = gv_decompress(comp, compressed, comp_size,
                                        decompressed, sizeof(decompressed));
    ASSERT(decomp_size == sizeof(input), "decompressed size mismatch");
    ASSERT(memcmp(input, decompressed, sizeof(input)) == 0,
           "decompressed data does not match original");

    free(compressed);
    gv_compression_destroy(comp);
    return 0;
}

/* 5. test_compress_bound */
static int test_compress_bound(void) {
    GV_CompressionConfig config;
    gv_compression_config_init(&config);
    config.type = GV_COMPRESS_LZ4;

    GV_Compressor *comp = gv_compression_create(&config);
    ASSERT(comp != NULL, "create failed");

    /* Bound must be at least the input size (worst case no compression) */
    size_t bound = gv_compress_bound(comp, 1024);
    ASSERT(bound >= 1024, "compress bound should be >= input size");

    gv_compression_destroy(comp);
    return 0;
}

/* 6. test_compression_stats                                           */
static int test_compression_stats(void) {
    GV_CompressionConfig config;
    gv_compression_config_init(&config);
    config.type = GV_COMPRESS_LZ4;

    GV_Compressor *comp = gv_compression_create(&config);
    ASSERT(comp != NULL, "create failed");

    /* Perform a round of compression to generate stats */
    char input[256];
    fill_test_data(input, sizeof(input));

    size_t bound = gv_compress_bound(comp, sizeof(input));
    char *compressed = (char *)malloc(bound);
    ASSERT(compressed != NULL, "malloc failed");

    size_t comp_size = gv_compress(comp, input, sizeof(input), compressed, bound);
    ASSERT(comp_size > 0, "compression failed");

    GV_CompressionStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = gv_compression_get_stats(comp, &stats);
    ASSERT(rc == 0, "get_stats failed");
    ASSERT(stats.total_compressed >= 1, "total_compressed should be >= 1");

    free(compressed);
    gv_compression_destroy(comp);
    return 0;
}

/* 7. test_compress_snappy                                             */
static int test_compress_snappy(void) {
    GV_CompressionConfig config;
    gv_compression_config_init(&config);
    config.type = GV_COMPRESS_SNAPPY;

    GV_Compressor *comp = gv_compression_create(&config);
    ASSERT(comp != NULL, "create snappy compressor failed");

    char input[256];
    fill_test_data(input, sizeof(input));

    size_t bound = gv_compress_bound(comp, sizeof(input));
    char *compressed = (char *)malloc(bound);
    ASSERT(compressed != NULL, "malloc failed");

    size_t comp_size = gv_compress(comp, input, sizeof(input), compressed, bound);
    ASSERT(comp_size > 0, "snappy compression failed");

    char decompressed[256];
    size_t decomp_size = gv_decompress(comp, compressed, comp_size,
                                        decompressed, sizeof(decompressed));
    ASSERT(decomp_size == sizeof(input), "snappy decompressed size mismatch");
    ASSERT(memcmp(input, decompressed, sizeof(input)) == 0,
           "snappy decompressed data mismatch");

    free(compressed);
    gv_compression_destroy(comp);
    return 0;
}

/* 8. test_compress_destroy_null                                       */
static int test_compress_destroy_null(void) {
    /* Should be safe to call with NULL */
    gv_compression_destroy(NULL);
    return 0;
}

/* main */
typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing compression config init...",           test_compression_config_init},
        {"Testing compression create/destroy...",        test_compression_create_destroy},
        {"Testing compress/decompress LZ4...",           test_compress_decompress_lz4},
        {"Testing compress/decompress ZSTD...",          test_compress_decompress_zstd},
        {"Testing compress bound...",                    test_compress_bound},
        {"Testing compression stats...",                 test_compression_stats},
        {"Testing compress/decompress Snappy...",        test_compress_snappy},
        {"Testing compression destroy null...",          test_compress_destroy_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
