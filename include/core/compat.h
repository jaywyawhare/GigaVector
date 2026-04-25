#ifndef GV_COMPAT_H
#define GV_COMPAT_H

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <stdint.h>

static inline void usleep(unsigned long usec) {
    Sleep((DWORD)((usec + 999UL) / 1000UL));
}

static inline unsigned int sleep(unsigned int sec) {
    unsigned long long ms = (unsigned long long)sec * 1000ULL;
    Sleep((DWORD)(ms < 0xFFFFFFFFULL ? ms : 0xFFFFFFFFULL));
    return 0;
}

#ifndef getpid
#define getpid() ((int)GetCurrentProcessId())
#endif

/* strcasecmp / strncasecmp */
#ifndef strcasecmp
#define strcasecmp(a,b)    _stricmp((a),(b))
#endif
#ifndef strncasecmp
#define strncasecmp(a,b,n) _strnicmp((a),(b),(n))
#endif

/* strtok_r — MSVC only has strtok_s with the same signature */
#ifndef strtok_r
#define strtok_r(s,d,p) strtok_s((s),(d),(p))
#endif

/* __builtin_popcount — use MSVC intrinsic */
#ifndef __GNUC__
#include <intrin.h>
#define __builtin_popcount(x)  __popcnt(x)
#define __builtin_popcountl(x) __popcnt((unsigned)(x))
#define __builtin_prefetch(p,rw,loc) ((void)(p))
#endif

#ifndef _TIMEVAL_DEFINED
#define _TIMEVAL_DEFINED
struct timeval { long long tv_sec; long tv_usec; };
#endif

static inline int gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    if (!tv) return 0;
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | (uint64_t)ft.dwLowDateTime;
    t -= UINT64_C(116444736000000000);
    tv->tv_sec  = (long long)(t / UINT64_C(10000000));
    tv->tv_usec = (long)((t % UINT64_C(10000000)) / 10ULL);
    return 0;
}

#endif /* _WIN32 */

/* __attribute__((unused)) — silence MSVC which doesn't support GCC attributes */
#if !defined(__GNUC__) && !defined(__clang__)
#define __attribute__(x)
#endif

#endif /* GV_COMPAT_H */
