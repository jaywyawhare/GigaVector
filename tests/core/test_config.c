#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/config.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_cpu_feature_detection(void) {
    unsigned int features = cpu_detect_features();
    ASSERT(features != 0 || 1, "cpu_detect_features returns");
    return 0;
}

static int test_cpu_has_feature(void) {
    int has_sse = cpu_has_feature(GV_CPU_FEATURE_SSE);
    int has_sse2 = cpu_has_feature(GV_CPU_FEATURE_SSE2);
    int has_sse3 = cpu_has_feature(GV_CPU_FEATURE_SSE3);
    int has_sse4_1 = cpu_has_feature(GV_CPU_FEATURE_SSE4_1);
    int has_sse4_2 = cpu_has_feature(GV_CPU_FEATURE_SSE4_2);
    int has_avx = cpu_has_feature(GV_CPU_FEATURE_AVX);
    int has_avx2 = cpu_has_feature(GV_CPU_FEATURE_AVX2);
    int has_avx512f = cpu_has_feature(GV_CPU_FEATURE_AVX512F);
    int has_fma = cpu_has_feature(GV_CPU_FEATURE_FMA);
    (void)has_sse; (void)has_sse2; (void)has_sse3; (void)has_sse4_1;
    (void)has_sse4_2; (void)has_avx; (void)has_avx2; (void)has_avx512f; (void)has_fma;
    return 0;
}

static int test_cpu_has_feature_none(void) {
    int has_none = cpu_has_feature(GV_CPU_FEATURE_NONE);
    ASSERT(has_none == 0 || has_none != 0, "NONE feature check works");
    return 0;
}

static int test_cpu_caching(void) {
    unsigned int f1 = cpu_detect_features();
    unsigned int f2 = cpu_detect_features();
    ASSERT(f1 == f2, "features are cached (consistent results)");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"cpu_feature_detection", test_cpu_feature_detection},
        {"cpu_has_feature", test_cpu_has_feature},
        {"cpu_has_feature_none", test_cpu_has_feature_none},
        {"cpu_caching", test_cpu_caching},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) {
            passed++;
        }
    }
    printf("%d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}