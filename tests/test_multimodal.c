#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_multimodal.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static const char *TEST_STORAGE_DIR = "/tmp/gv_test_multimodal";

static int test_media_config_init(void) {
    GV_MediaConfig config;
    memset(&config, 0, sizeof(config));
    gv_media_config_init(&config);

    ASSERT(config.storage_dir == NULL, "default storage_dir should be NULL");
    ASSERT(config.max_blob_size_mb == 100, "default max_blob_size_mb should be 100");
    ASSERT(config.deduplicate == 1, "default deduplicate should be 1");
    ASSERT(config.compress_blobs == 0, "default compress_blobs should be 0");

    return 0;
}

static int test_media_create_destroy(void) {
    GV_MediaConfig config;
    gv_media_config_init(&config);
    config.storage_dir = TEST_STORAGE_DIR;

    GV_MediaStore *store = gv_media_create(&config);
    ASSERT(store != NULL, "media store creation should succeed");

    gv_media_destroy(store);

    /* Destroying NULL should be safe */
    gv_media_destroy(NULL);
    return 0;
}

static int test_media_store_blob(void) {
    GV_MediaConfig config;
    gv_media_config_init(&config);
    config.storage_dir = TEST_STORAGE_DIR;

    GV_MediaStore *store = gv_media_create(&config);
    ASSERT(store != NULL, "media store creation");

    /* Store a small test blob */
    const unsigned char blob_data[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                                       0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
    int rc = gv_media_store_blob(store, 0, GV_MEDIA_IMAGE, blob_data, sizeof(blob_data),
                                  "test.png", "image/png");
    ASSERT(rc == 0, "storing blob should succeed");

    ASSERT(gv_media_count(store) == 1, "should have 1 media entry");

    gv_media_destroy(store);
    return 0;
}

static int test_media_retrieve(void) {
    GV_MediaConfig config;
    gv_media_config_init(&config);
    config.storage_dir = TEST_STORAGE_DIR;

    GV_MediaStore *store = gv_media_create(&config);
    ASSERT(store != NULL, "media store creation");

    const unsigned char original[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE};
    int rc = gv_media_store_blob(store, 10, GV_MEDIA_BLOB, original, sizeof(original),
                                  "data.bin", "application/octet-stream");
    ASSERT(rc == 0, "storing blob should succeed");

    unsigned char buffer[256];
    size_t actual_size = 0;
    rc = gv_media_retrieve(store, 10, buffer, sizeof(buffer), &actual_size);
    ASSERT(rc == 0, "retrieving blob should succeed");
    ASSERT(actual_size == sizeof(original), "retrieved size should match original");
    ASSERT(memcmp(buffer, original, sizeof(original)) == 0, "retrieved data should match original");

    gv_media_destroy(store);
    return 0;
}

static int test_media_get_info(void) {
    GV_MediaConfig config;
    gv_media_config_init(&config);
    config.storage_dir = TEST_STORAGE_DIR;

    GV_MediaStore *store = gv_media_create(&config);
    ASSERT(store != NULL, "media store creation");

    const unsigned char data[] = {0x01, 0x02, 0x03, 0x04};
    int rc = gv_media_store_blob(store, 5, GV_MEDIA_AUDIO, data, sizeof(data),
                                  "clip.wav", "audio/wav");
    ASSERT(rc == 0, "storing blob should succeed");

    GV_MediaEntry entry;
    memset(&entry, 0, sizeof(entry));
    rc = gv_media_get_info(store, 5, &entry);
    ASSERT(rc == 0, "getting info should succeed");
    ASSERT(entry.vector_index == 5, "vector_index should be 5");
    ASSERT(entry.type == GV_MEDIA_AUDIO, "type should be AUDIO");
    ASSERT(entry.file_size == sizeof(data), "file_size should match data size");
    ASSERT(strlen(entry.hash) == 64, "hash should be 64 hex chars");

    if (entry.filename) {
        ASSERT(strcmp(entry.filename, "clip.wav") == 0, "filename should match");
        free(entry.filename);
    }
    if (entry.mime_type) {
        ASSERT(strcmp(entry.mime_type, "audio/wav") == 0, "mime_type should match");
        free(entry.mime_type);
    }

    gv_media_destroy(store);
    return 0;
}

static int test_media_exists_and_delete(void) {
    GV_MediaConfig config;
    gv_media_config_init(&config);
    config.storage_dir = TEST_STORAGE_DIR;

    GV_MediaStore *store = gv_media_create(&config);
    ASSERT(store != NULL, "media store creation");

    const unsigned char data[] = {0xAA, 0xBB, 0xCC};
    gv_media_store_blob(store, 20, GV_MEDIA_DOCUMENT, data, sizeof(data),
                        "doc.pdf", "application/pdf");

    ASSERT(gv_media_exists(store, 20) == 1, "blob should exist at index 20");
    ASSERT(gv_media_exists(store, 99) == 0, "blob should not exist at index 99");

    int rc = gv_media_delete(store, 20);
    ASSERT(rc == 0, "deleting blob should succeed");
    ASSERT(gv_media_exists(store, 20) == 0, "blob should not exist after deletion");
    ASSERT(gv_media_count(store) == 0, "count should be 0 after deletion");

    gv_media_destroy(store);
    return 0;
}

static int test_media_total_size(void) {
    GV_MediaConfig config;
    gv_media_config_init(&config);
    config.storage_dir = TEST_STORAGE_DIR;

    GV_MediaStore *store = gv_media_create(&config);
    ASSERT(store != NULL, "media store creation");

    ASSERT(gv_media_total_size(store) == 0, "empty store should have 0 total size");

    const unsigned char data1[] = {0x01, 0x02, 0x03, 0x04, 0x05};
    const unsigned char data2[] = {0x10, 0x20, 0x30};

    gv_media_store_blob(store, 0, GV_MEDIA_BLOB, data1, sizeof(data1), NULL, NULL);
    gv_media_store_blob(store, 1, GV_MEDIA_BLOB, data2, sizeof(data2), NULL, NULL);

    size_t total = gv_media_total_size(store);
    ASSERT(total == sizeof(data1) + sizeof(data2),
           "total size should equal sum of stored blob sizes");

    gv_media_destroy(store);
    return 0;
}

static int test_media_get_path(void) {
    GV_MediaConfig config;
    gv_media_config_init(&config);
    config.storage_dir = TEST_STORAGE_DIR;

    GV_MediaStore *store = gv_media_create(&config);
    ASSERT(store != NULL, "media store creation");

    const unsigned char data[] = {0xFF, 0xFE, 0xFD};
    gv_media_store_blob(store, 7, GV_MEDIA_IMAGE, data, sizeof(data),
                        "img.jpg", "image/jpeg");

    char path[512];
    int rc = gv_media_get_path(store, 7, path, sizeof(path));
    ASSERT(rc == 0, "getting path should succeed");
    ASSERT(strlen(path) > 0, "path should be non-empty");
    ASSERT(strstr(path, TEST_STORAGE_DIR) != NULL, "path should contain storage_dir");

    gv_media_destroy(store);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing media config init...", test_media_config_init},
        {"Testing media create/destroy...", test_media_create_destroy},
        {"Testing media store blob...", test_media_store_blob},
        {"Testing media retrieve...", test_media_retrieve},
        {"Testing media get info...", test_media_get_info},
        {"Testing media exists and delete...", test_media_exists_and_delete},
        {"Testing media total size...", test_media_total_size},
        {"Testing media get path...", test_media_get_path},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
