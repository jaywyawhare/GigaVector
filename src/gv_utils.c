#include <stddef.h>
#include <stdint.h>

#include "gigavector/gv_config.h"

#ifdef __x86_64__

#include <cpuid.h>

static unsigned int g_cpu_features = 0;
static int g_cpu_features_initialized = 0;

unsigned int gv_cpu_detect_features(void) {
    if (g_cpu_features_initialized) {
        return g_cpu_features;
    }

    unsigned int eax, ebx, ecx, edx;
    unsigned int features = GV_CPU_FEATURE_NONE;

    if (__get_cpuid_max(0, NULL) >= 1) {
        __cpuid(1, eax, ebx, ecx, edx);

        if (edx & bit_SSE) {
            features |= GV_CPU_FEATURE_SSE;
        }
        if (edx & bit_SSE2) {
            features |= GV_CPU_FEATURE_SSE2;
        }
        if (ecx & bit_SSE3) {
            features |= GV_CPU_FEATURE_SSE3;
        }
        if (ecx & bit_SSE4_1) {
            features |= GV_CPU_FEATURE_SSE4_1;
        }
        if (ecx & bit_SSE4_2) {
            features |= GV_CPU_FEATURE_SSE4_2;
        }
        if (ecx & bit_AVX) {
            features |= GV_CPU_FEATURE_AVX;
        }
        if (ecx & bit_FMA) {
            features |= GV_CPU_FEATURE_FMA;
        }
    }

    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        if (ebx & bit_AVX2) {
            features |= GV_CPU_FEATURE_AVX2;
        }
    }

    g_cpu_features = features;
    g_cpu_features_initialized = 1;
    return features;
}

#else

unsigned int gv_cpu_detect_features(void) {
    return GV_CPU_FEATURE_NONE;
}

#endif

int gv_cpu_has_feature(GV_CPUFeature feature) {
    unsigned int features = gv_cpu_detect_features();
    return (features & feature) != 0;
}

