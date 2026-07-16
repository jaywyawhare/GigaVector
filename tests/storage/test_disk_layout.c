#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/disk_layout.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_default_sector_size(void) {
    ASSERT(gv_disk_default_sector_size() == GV_DISK_SECTOR_SIZE_DEFAULT,
           "default sector size should equal GV_DISK_SECTOR_SIZE_DEFAULT");
    ASSERT(gv_disk_default_sector_size() == 4096u,
           "default sector size should be 4096");
    return 0;
}

static int test_normalize_zero(void) {
    /* 0 normalizes to the default sector size. */
    ASSERT(gv_disk_normalize_sector_size(0) == GV_DISK_SECTOR_SIZE_DEFAULT,
           "zero should normalize to default");
    return 0;
}

static int test_normalize_nonzero_passthrough(void) {
    /* Any non-zero value is returned unchanged. */
    ASSERT(gv_disk_normalize_sector_size(512) == 512,
           "512 should pass through unchanged");
    ASSERT(gv_disk_normalize_sector_size(4096) == 4096,
           "4096 should pass through unchanged");
    ASSERT(gv_disk_normalize_sector_size(1) == 1,
           "1 should pass through unchanged");
    return 0;
}

static int test_normalize_large(void) {
    size_t big = (size_t)1 << 20; /* 1 MiB */
    ASSERT(gv_disk_normalize_sector_size(big) == big,
           "large non-zero value should pass through unchanged");
    return 0;
}

static int test_deprecated_alias(void) {
    /* The deprecated alias must still resolve to the default. */
    ASSERT(GV_POSTING_DEFAULT_SECTOR_SIZE == GV_DISK_SECTOR_SIZE_DEFAULT,
           "deprecated alias should equal default");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing disk default sector size...",       test_default_sector_size},
        {"Testing normalize sector size zero...",     test_normalize_zero},
        {"Testing normalize sector size passthrough...", test_normalize_nonzero_passthrough},
        {"Testing normalize sector size large...",    test_normalize_large},
        {"Testing deprecated sector size alias...",   test_deprecated_alias},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
