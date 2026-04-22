/**
 * @file config.c
 * @brief Runtime CPU feature detection.
 */

#include "core/config.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#  define GV_ARCH_X86 1
#  ifdef _MSC_VER
#    include <intrin.h>
#  endif
#endif

static unsigned int s_features = (unsigned int)-1; /* sentinel: not yet detected */

#ifdef GV_ARCH_X86
static void cpuid(unsigned int leaf, unsigned int subleaf,
                  unsigned int *eax, unsigned int *ebx,
                  unsigned int *ecx, unsigned int *edx) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *eax = (unsigned int)regs[0];
    *ebx = (unsigned int)regs[1];
    *ecx = (unsigned int)regs[2];
    *edx = (unsigned int)regs[3];
#else
    __asm__ volatile (
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf)
    );
#endif
}
#endif /* GV_ARCH_X86 */

unsigned int cpu_detect_features(void) {
    if (s_features != (unsigned int)-1) {
        return s_features;
    }

    unsigned int features = GV_CPU_FEATURE_NONE;

#ifdef GV_ARCH_X86
    unsigned int eax, ebx, ecx, edx;

    /* Leaf 1: SSE, SSE2, SSE3, SSE4.1, SSE4.2, AVX, FMA */
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);

    if (edx & (1u << 25)) features |= GV_CPU_FEATURE_SSE;
    if (edx & (1u << 26)) features |= GV_CPU_FEATURE_SSE2;
    if (ecx & (1u <<  0)) features |= GV_CPU_FEATURE_SSE3;
    if (ecx & (1u << 19)) features |= GV_CPU_FEATURE_SSE4_1;
    if (ecx & (1u << 20)) features |= GV_CPU_FEATURE_SSE4_2;
    if (ecx & (1u << 28)) features |= GV_CPU_FEATURE_AVX;
    if (ecx & (1u << 12)) features |= GV_CPU_FEATURE_FMA;

    /* Leaf 7: AVX2, AVX-512F */
    cpuid(0, 0, &eax, &ebx, &ecx, &edx); /* get max leaf */
    if (eax >= 7) {
        cpuid(7, 0, &eax, &ebx, &ecx, &edx);
        if (ebx & (1u <<  5)) features |= GV_CPU_FEATURE_AVX2;
        if (ebx & (1u << 16)) features |= GV_CPU_FEATURE_AVX512F;
    }
#endif /* GV_ARCH_X86 */

    s_features = features;
    return features;
}

int cpu_has_feature(GV_CPUFeature feature) {
    return (cpu_detect_features() & (unsigned int)feature) != 0;
}
