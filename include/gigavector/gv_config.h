#ifndef GIGAVECTOR_GV_CONFIG_H
#define GIGAVECTOR_GV_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU feature flags for SIMD support.
 */
typedef enum {
    GV_CPU_FEATURE_NONE = 0,
    GV_CPU_FEATURE_SSE = 1 << 0,
    GV_CPU_FEATURE_SSE2 = 1 << 1,
    GV_CPU_FEATURE_SSE3 = 1 << 2,
    GV_CPU_FEATURE_SSE4_1 = 1 << 3,
    GV_CPU_FEATURE_SSE4_2 = 1 << 4,
    GV_CPU_FEATURE_AVX = 1 << 5,
    GV_CPU_FEATURE_AVX2 = 1 << 6,
    GV_CPU_FEATURE_FMA = 1 << 7,
    GV_CPU_FEATURE_AVX512F = 1 << 8
} GV_CPUFeature;

/**
 * @brief Detect available CPU features at runtime.
 *
 * @return Bitmask of available CPU features.
 */
unsigned int gv_cpu_detect_features(void);

/**
 * @brief Check if a specific CPU feature is available.
 *
 * @param feature Feature to check.
 * @return 1 if available, 0 otherwise.
 */
int gv_cpu_has_feature(GV_CPUFeature feature);

#ifdef __cplusplus
}
#endif

#endif

