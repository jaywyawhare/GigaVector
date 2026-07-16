#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "storage/disk_page_cache.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Helper: build a byte buffer of length len filled with a marker byte. */
static void fill_buf(uint8_t *buf, size_t len, uint8_t marker) {
    for (size_t i = 0; i < len; i++) buf[i] = marker;
}

static int test_create_destroy(void) {
    GV_DiskPageCache *cache = gv_disk_page_cache_create(1024);
    ASSERT(cache != NULL, "create should succeed");

    GV_DiskPageCacheStats st;
    gv_disk_page_cache_get_stats(cache, &st);
    ASSERT(st.max_bytes == 1024, "max_bytes should be 1024");
    ASSERT(st.cached_entries == 0, "empty cache has no entries");
    ASSERT(st.used_bytes == 0, "empty cache uses no bytes");

    gv_disk_page_cache_destroy(cache);
    gv_disk_page_cache_destroy(NULL); /* must be safe */
    return 0;
}

static int test_insert_lookup(void) {
    GV_DiskPageCache *cache = gv_disk_page_cache_create(1024);
    ASSERT(cache != NULL, "create should succeed");

    uint8_t data[64];
    fill_buf(data, sizeof(data), 0xAB);

    ASSERT(gv_disk_page_cache_insert(cache, "page-a", data, sizeof(data)) == 0,
           "insert should succeed");

    size_t len = 0;
    const uint8_t *got = gv_disk_page_cache_lookup(cache, "page-a", &len);
    ASSERT(got != NULL, "lookup should hit");
    ASSERT(len == sizeof(data), "lookup length should match");
    ASSERT(memcmp(got, data, sizeof(data)) == 0, "lookup data should match");

    /* Miss on unknown key. */
    ASSERT(gv_disk_page_cache_lookup(cache, "missing", &len) == NULL,
           "lookup of missing key should return NULL");

    GV_DiskPageCacheStats st;
    gv_disk_page_cache_get_stats(cache, &st);
    ASSERT(st.cache_hits == 1, "one hit recorded");
    ASSERT(st.cache_misses == 1, "one miss recorded");
    ASSERT(st.cached_entries == 1, "one entry cached");
    ASSERT(st.used_bytes == sizeof(data), "used bytes tracked");

    gv_disk_page_cache_destroy(cache);
    return 0;
}

static int test_zero_len_and_null_reject(void) {
    GV_DiskPageCache *cache = gv_disk_page_cache_create(1024);
    ASSERT(cache != NULL, "create should succeed");
    uint8_t data[8] = {0};

    ASSERT(gv_disk_page_cache_insert(cache, "k", data, 0) == -1,
           "zero-length insert must be rejected");
    ASSERT(gv_disk_page_cache_insert(cache, NULL, data, 8) == -1,
           "NULL key insert must be rejected");
    ASSERT(gv_disk_page_cache_insert(cache, "k", NULL, 8) == -1,
           "NULL data insert must be rejected");
    ASSERT(gv_disk_page_cache_insert(NULL, "k", data, 8) == -1,
           "NULL cache insert must be rejected");

    /* max_bytes == 0 cache rejects all inserts. */
    GV_DiskPageCache *zero = gv_disk_page_cache_create(0);
    ASSERT(zero != NULL, "create with 0 max_bytes should succeed");
    ASSERT(gv_disk_page_cache_insert(zero, "k", data, 8) == -1,
           "insert into zero-capacity cache must be rejected");
    gv_disk_page_cache_destroy(zero);

    gv_disk_page_cache_destroy(cache);
    return 0;
}

static int test_eviction_lru(void) {
    /* Capacity for exactly two 64-byte entries. */
    GV_DiskPageCache *cache = gv_disk_page_cache_create(128);
    ASSERT(cache != NULL, "create should succeed");

    uint8_t a[64], b[64], c[64];
    fill_buf(a, 64, 0x11);
    fill_buf(b, 64, 0x22);
    fill_buf(c, 64, 0x33);

    ASSERT(gv_disk_page_cache_insert(cache, "a", a, 64) == 0, "insert a");
    ASSERT(gv_disk_page_cache_insert(cache, "b", b, 64) == 0, "insert b");

    /* Touch "a" so "b" becomes the LRU victim. */
    size_t len = 0;
    ASSERT(gv_disk_page_cache_lookup(cache, "a", &len) != NULL, "a should be present");

    /* Inserting "c" pushes over the bound; LRU ("b") must be evicted. */
    ASSERT(gv_disk_page_cache_insert(cache, "c", c, 64) == 0, "insert c");

    GV_DiskPageCacheStats st;
    gv_disk_page_cache_get_stats(cache, &st);
    ASSERT(st.cached_entries == 2, "cache should hold exactly two entries");
    ASSERT(st.used_bytes <= st.max_bytes, "used bytes must respect the bound");

    ASSERT(gv_disk_page_cache_lookup(cache, "a", &len) != NULL, "a survived (recently used)");
    ASSERT(gv_disk_page_cache_lookup(cache, "c", &len) != NULL, "c present (just inserted)");
    ASSERT(gv_disk_page_cache_lookup(cache, "b", &len) == NULL, "b evicted (LRU)");

    gv_disk_page_cache_destroy(cache);
    return 0;
}

static int test_replace_same_key(void) {
    GV_DiskPageCache *cache = gv_disk_page_cache_create(1024);
    ASSERT(cache != NULL, "create should succeed");

    uint8_t small[16], big[128];
    fill_buf(small, sizeof(small), 0x01);
    fill_buf(big, sizeof(big), 0x02);

    ASSERT(gv_disk_page_cache_insert(cache, "k", small, sizeof(small)) == 0, "insert small");
    ASSERT(gv_disk_page_cache_insert(cache, "k", big, sizeof(big)) == 0, "replace with big");

    GV_DiskPageCacheStats st;
    gv_disk_page_cache_get_stats(cache, &st);
    ASSERT(st.cached_entries == 1, "replacing same key keeps one entry");
    ASSERT(st.used_bytes == sizeof(big), "used bytes reflect the larger value");

    size_t len = 0;
    const uint8_t *got = gv_disk_page_cache_lookup(cache, "k", &len);
    ASSERT(got != NULL && len == sizeof(big), "lookup returns replaced length");
    ASSERT(memcmp(got, big, sizeof(big)) == 0, "lookup returns replaced data");

    gv_disk_page_cache_destroy(cache);
    return 0;
}

static int test_remove_and_reinsert(void) {
    GV_DiskPageCache *cache = gv_disk_page_cache_create(256);
    ASSERT(cache != NULL, "create should succeed");

    uint8_t a[64], b[64];
    fill_buf(a, 64, 0xAA);
    fill_buf(b, 64, 0xBB);

    ASSERT(gv_disk_page_cache_insert(cache, "a", a, 64) == 0, "insert a");
    ASSERT(gv_disk_page_cache_insert(cache, "b", b, 64) == 0, "insert b");

    size_t len = 0;
    gv_disk_page_cache_remove(cache, "a");
    ASSERT(gv_disk_page_cache_lookup(cache, "a", &len) == NULL, "a removed");
    ASSERT(gv_disk_page_cache_lookup(cache, "b", &len) != NULL, "b still present");

    GV_DiskPageCacheStats st;
    gv_disk_page_cache_get_stats(cache, &st);
    ASSERT(st.cached_entries == 1, "one entry after removal");
    ASSERT(st.used_bytes == 64, "used bytes updated after removal");

    /* Reinsert previously removed key. */
    ASSERT(gv_disk_page_cache_insert(cache, "a", a, 64) == 0, "reinsert a");
    ASSERT(gv_disk_page_cache_lookup(cache, "a", &len) != NULL, "a present after reinsert");

    /* Removing a missing key is a no-op and safe. */
    gv_disk_page_cache_remove(cache, "does-not-exist");
    gv_disk_page_cache_remove(NULL, "a");

    gv_disk_page_cache_destroy(cache);
    return 0;
}

static int test_set_max_bytes_shrinks(void) {
    GV_DiskPageCache *cache = gv_disk_page_cache_create(256);
    ASSERT(cache != NULL, "create should succeed");

    uint8_t a[64], b[64], c[64];
    fill_buf(a, 64, 1);
    fill_buf(b, 64, 2);
    fill_buf(c, 64, 3);
    ASSERT(gv_disk_page_cache_insert(cache, "a", a, 64) == 0, "insert a");
    ASSERT(gv_disk_page_cache_insert(cache, "b", b, 64) == 0, "insert b");
    ASSERT(gv_disk_page_cache_insert(cache, "c", c, 64) == 0, "insert c");

    /* Shrinking the bound must evict LRU entries until within budget. */
    gv_disk_page_cache_set_max_bytes(cache, 64);

    GV_DiskPageCacheStats st;
    gv_disk_page_cache_get_stats(cache, &st);
    ASSERT(st.max_bytes == 64, "max_bytes updated");
    ASSERT(st.used_bytes <= 64, "used bytes respect shrunken bound");
    ASSERT(st.cached_entries == 1, "only one entry fits after shrink");

    gv_disk_page_cache_destroy(cache);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing disk page cache create/destroy...",   test_create_destroy},
        {"Testing disk page cache insert/lookup...",    test_insert_lookup},
        {"Testing disk page cache reject invalid...",   test_zero_len_and_null_reject},
        {"Testing disk page cache LRU eviction...",      test_eviction_lru},
        {"Testing disk page cache replace same key...", test_replace_same_key},
        {"Testing disk page cache remove/reinsert...",  test_remove_and_reinsert},
        {"Testing disk page cache set_max_bytes...",    test_set_max_bytes_shrinks},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
