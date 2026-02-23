#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_mmap.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static const char *TMP_FILE = "tmp_test_mmap.bin";
static const char *TMP_FILE2 = "tmp_test_mmap2.bin";
static const char *TMP_FILE_LARGE = "tmp_test_mmap_large.bin";

static void cleanup(void) {
    remove(TMP_FILE);
    remove(TMP_FILE2);
    remove(TMP_FILE_LARGE);
}

/* Helper: write known data to a file */
static int write_file(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t written = fwrite(data, 1, len, f);
    fclose(f);
    return written == len ? 0 : -1;
}

static int test_open_readonly(void) {
    cleanup();
    const char *payload = "Hello, mmap world!";
    size_t payload_len = strlen(payload);
    ASSERT(write_file(TMP_FILE, payload, payload_len) == 0, "write test file");

    GV_MMap *mm = gv_mmap_open_readonly(TMP_FILE);
    ASSERT(mm != NULL, "gv_mmap_open_readonly should succeed");

    size_t sz = gv_mmap_size(mm);
    ASSERT(sz == payload_len, "mmap size should match file size");

    const void *data = gv_mmap_data(mm);
    ASSERT(data != NULL, "mmap data should not be NULL");
    ASSERT(memcmp(data, payload, payload_len) == 0, "mmap data should match written content");

    gv_mmap_close(mm);
    cleanup();
    return 0;
}

static int test_mmap_size(void) {
    cleanup();
    /* Write exactly 256 bytes */
    unsigned char buf[256];
    for (int i = 0; i < 256; i++) buf[i] = (unsigned char)i;
    ASSERT(write_file(TMP_FILE, buf, sizeof(buf)) == 0, "write 256 bytes");

    GV_MMap *mm = gv_mmap_open_readonly(TMP_FILE);
    ASSERT(mm != NULL, "open");
    ASSERT(gv_mmap_size(mm) == 256, "size should be 256");

    gv_mmap_close(mm);
    cleanup();
    return 0;
}

static int test_mmap_data_contents(void) {
    cleanup();
    /* Write a pattern: 0x00, 0x01, ..., 0xFF repeated */
    unsigned char buf[512];
    for (int i = 0; i < 512; i++) buf[i] = (unsigned char)(i & 0xFF);
    ASSERT(write_file(TMP_FILE, buf, sizeof(buf)) == 0, "write pattern");

    GV_MMap *mm = gv_mmap_open_readonly(TMP_FILE);
    ASSERT(mm != NULL, "open");

    const unsigned char *data = (const unsigned char *)gv_mmap_data(mm);
    ASSERT(data != NULL, "data not NULL");

    for (int i = 0; i < 512; i++) {
        ASSERT(data[i] == (unsigned char)(i & 0xFF), "data byte mismatch");
    }

    gv_mmap_close(mm);
    cleanup();
    return 0;
}

static int test_close_null(void) {
    /* Closing NULL must not crash */
    gv_mmap_close(NULL);
    return 0;
}

static int test_double_close(void) {
    cleanup();
    const char *payload = "double close test";
    ASSERT(write_file(TMP_FILE, payload, strlen(payload)) == 0, "write");

    GV_MMap *mm = gv_mmap_open_readonly(TMP_FILE);
    ASSERT(mm != NULL, "open");

    gv_mmap_close(mm);
    /* Second close on the same pointer is undefined, but closing NULL is safe */
    gv_mmap_close(NULL);

    cleanup();
    return 0;
}

static int test_open_nonexistent(void) {
    remove("tmp_test_mmap_noexist.bin");
    GV_MMap *mm = gv_mmap_open_readonly("tmp_test_mmap_noexist.bin");
    ASSERT(mm == NULL, "open non-existent file should return NULL");
    return 0;
}

static int test_open_empty_file(void) {
    cleanup();
    /* Create an empty file */
    FILE *f = fopen(TMP_FILE, "wb");
    ASSERT(f != NULL, "create empty file");
    fclose(f);

    GV_MMap *mm = gv_mmap_open_readonly(TMP_FILE);
    /* An empty file may return NULL or a valid mapping with size 0;
     * both are acceptable behaviors. */
    if (mm != NULL) {
        ASSERT(gv_mmap_size(mm) == 0, "empty file size should be 0");
        gv_mmap_close(mm);
    }

    cleanup();
    return 0;
}

static int test_mmap_large_file(void) {
    cleanup();
    /* Write 1 MB of data */
    size_t file_size = 1024 * 1024;
    unsigned char *buf = (unsigned char *)malloc(file_size);
    ASSERT(buf != NULL, "allocate 1MB buffer");

    for (size_t i = 0; i < file_size; i++) {
        buf[i] = (unsigned char)(i % 251); /* prime modulus for varied pattern */
    }

    ASSERT(write_file(TMP_FILE_LARGE, buf, file_size) == 0, "write 1MB file");

    GV_MMap *mm = gv_mmap_open_readonly(TMP_FILE_LARGE);
    ASSERT(mm != NULL, "open large file");
    ASSERT(gv_mmap_size(mm) == file_size, "large file size should be 1MB");

    const unsigned char *data = (const unsigned char *)gv_mmap_data(mm);
    ASSERT(data != NULL, "large file data not NULL");

    /* Spot-check some values */
    ASSERT(data[0] == 0, "first byte");
    ASSERT(data[250] == 250, "byte at 250");
    ASSERT(data[251] == 0, "byte at 251 wraps");
    ASSERT(data[file_size - 1] == (unsigned char)((file_size - 1) % 251), "last byte");

    /* Verify first 1024 bytes fully */
    for (size_t i = 0; i < 1024; i++) {
        ASSERT(data[i] == (unsigned char)(i % 251), "verify first 1024 bytes");
    }

    gv_mmap_close(mm);
    free(buf);
    cleanup();
    return 0;
}

static int test_open_null_path(void) {
    GV_MMap *mm = gv_mmap_open_readonly(NULL);
    ASSERT(mm == NULL, "open NULL path should return NULL");
    return 0;
}

static int test_binary_data(void) {
    cleanup();
    /* Write floats as binary */
    float floats[] = {1.0f, 2.5f, -3.14f, 0.0f, 100.0f};
    size_t len = sizeof(floats);
    ASSERT(write_file(TMP_FILE2, floats, len) == 0, "write float data");

    GV_MMap *mm = gv_mmap_open_readonly(TMP_FILE2);
    ASSERT(mm != NULL, "open binary file");
    ASSERT(gv_mmap_size(mm) == len, "binary file size");

    const float *data = (const float *)gv_mmap_data(mm);
    ASSERT(data != NULL, "binary data not NULL");

    for (int i = 0; i < 5; i++) {
        ASSERT(data[i] == floats[i], "float value mismatch");
    }

    gv_mmap_close(mm);
    cleanup();
    return 0;
}

static int test_multiple_mmaps(void) {
    cleanup();
    const char *data1 = "file one contents";
    const char *data2 = "file two data here";
    ASSERT(write_file(TMP_FILE, data1, strlen(data1)) == 0, "write file 1");
    ASSERT(write_file(TMP_FILE2, data2, strlen(data2)) == 0, "write file 2");

    GV_MMap *mm1 = gv_mmap_open_readonly(TMP_FILE);
    GV_MMap *mm2 = gv_mmap_open_readonly(TMP_FILE2);
    ASSERT(mm1 != NULL, "open file 1");
    ASSERT(mm2 != NULL, "open file 2");

    ASSERT(gv_mmap_size(mm1) == strlen(data1), "file 1 size");
    ASSERT(gv_mmap_size(mm2) == strlen(data2), "file 2 size");

    ASSERT(memcmp(gv_mmap_data(mm1), data1, strlen(data1)) == 0, "file 1 contents");
    ASSERT(memcmp(gv_mmap_data(mm2), data2, strlen(data2)) == 0, "file 2 contents");

    gv_mmap_close(mm1);
    gv_mmap_close(mm2);
    cleanup();
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    cleanup();
    TestCase tests[] = {
        {"Testing open_readonly...",     test_open_readonly},
        {"Testing mmap_size...",         test_mmap_size},
        {"Testing mmap_data_contents..", test_mmap_data_contents},
        {"Testing close_null...",        test_close_null},
        {"Testing double_close...",      test_double_close},
        {"Testing open_nonexistent...",  test_open_nonexistent},
        {"Testing open_empty_file...",   test_open_empty_file},
        {"Testing mmap_large_file...",   test_mmap_large_file},
        {"Testing open_null_path...",    test_open_null_path},
        {"Testing binary_data...",       test_binary_data},
        {"Testing multiple_mmaps...",    test_multiple_mmaps},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    cleanup();
    return passed == n ? 0 : 1;
}
