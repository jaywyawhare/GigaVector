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

#endif /* GV_COMPAT_H */
