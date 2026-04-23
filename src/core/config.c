/**
 * @file config.c
 * @brief Runtime CPU feature detection.
 */

#include <stdint.h>
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

    /*
     * AVX/AVX-512 require OS XSAVE support in addition to CPUID bits.
     * CPUID alone is not enough: in WSL2/Hyper-V the CPU may advertise AVX
     * while the hypervisor has not enabled the YMM/ZMM save-restore state in
     * XCR0.  Executing AVX instructions in that case raises SIGILL.
     * The safe sequence is: check OSXSAVE (ecx bit 27), then call xgetbv to
     * read XCR0, and only enable each feature if the OS has allocated the
     * required extended register state.
     */
    int osxsave = (ecx & (1u << 27)) != 0;
    uint64_t xcr0 = 0;
    if (osxsave) {
#ifdef _MSC_VER
        xcr0 = _xgetbv(0);
#else
        unsigned int xcr0_lo, xcr0_hi;
        __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
        xcr0 = ((uint64_t)xcr0_hi << 32) | (uint64_t)xcr0_lo;
#endif
    }
    /* XMM (bit 1) + YMM (bit 2) state must both be enabled for AVX/FMA/AVX2. */
    int avx_os = osxsave && ((xcr0 & 0x6u) == 0x6u);
    if ((ecx & (1u << 28)) && avx_os) features |= GV_CPU_FEATURE_AVX;
    if ((ecx & (1u << 12)) && avx_os) features |= GV_CPU_FEATURE_FMA;

    /* Leaf 7: AVX2, AVX-512F */
    cpuid(0, 0, &eax, &ebx, &ecx, &edx); /* get max leaf */
    if (eax >= 7) {
        cpuid(7, 0, &eax, &ebx, &ecx, &edx);
        if ((ebx & (1u <<  5)) && avx_os) features |= GV_CPU_FEATURE_AVX2;
        /* AVX-512F also needs opmask (bit 5) + ZMM_Hi256 (bit 6) + Hi16_ZMM (bit 7) in XCR0. */
        if ((ebx & (1u << 16)) && osxsave && ((xcr0 & 0xe6u) == 0xe6u))
            features |= GV_CPU_FEATURE_AVX512F;
    }
#endif /* GV_ARCH_X86 */

    s_features = features;
    return features;
}

int cpu_has_feature(GV_CPUFeature feature) {
    return (cpu_detect_features() & (unsigned int)feature) != 0;
}
