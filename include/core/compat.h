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

/* ssize_t */
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#endif

/* S_ISDIR / S_ISREG — MSVC provides _S_IFMT/_S_IFDIR/_S_IFREG in <sys/stat.h> */
#include <sys/stat.h>
#ifndef S_ISDIR
#define S_ISDIR(m)  (((m) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISREG
#define S_ISREG(m)  (((m) & _S_IFMT) == _S_IFREG)
#endif

/* clock_gettime / struct timespec / CLOCK_* constants.
 * MinGW ships its own implementation; this shim is for MSVC only. */
#ifndef __GNUC__
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME  0
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
typedef int clockid_t;
#ifndef _TIMESPEC_DEFINED
#define _TIMESPEC_DEFINED
struct timespec { long long tv_sec; long tv_nsec; };
#endif
static inline int clock_gettime(clockid_t clk, struct timespec *ts) {
    if (!ts) return -1;
    if (clk == CLOCK_MONOTONIC) {
        LARGE_INTEGER freq, cnt;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&cnt);
        ts->tv_sec  = (long long)(cnt.QuadPart / freq.QuadPart);
        ts->tv_nsec = (long)(((cnt.QuadPart % freq.QuadPart) * 1000000000LL)
                             / freq.QuadPart);
    } else {
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | (uint64_t)ft.dwLowDateTime;
        t -= UINT64_C(116444736000000000);
        ts->tv_sec  = (long long)(t / UINT64_C(10000000));
        ts->tv_nsec = (long)((t % UINT64_C(10000000)) * 100ULL);
    }
    return 0;
}
#endif /* !__GNUC__ */

/* POSIX file-descriptor open flags — MSVC uses _O_* names in <io.h>.
 * MinGW provides the standard names via <fcntl.h>, so skip under GCC. */
#ifndef __GNUC__
#include <io.h>
#include <fcntl.h>
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif
#ifndef O_WRONLY
#define O_WRONLY _O_WRONLY
#endif
#ifndef O_RDWR
#define O_RDWR   _O_RDWR
#endif
#ifndef O_CREAT
#define O_CREAT  _O_CREAT
#endif
#ifndef O_TRUNC
#define O_TRUNC  _O_TRUNC
#endif
#ifndef O_APPEND
#define O_APPEND _O_APPEND
#endif
#ifndef O_BINARY
#define O_BINARY _O_BINARY
#endif
#define open(path, flags, ...) _open((path), (flags) | _O_BINARY, ##__VA_ARGS__)
#define close  _close
#define read   _read
#define write  _write
#endif /* !__GNUC__ */

#endif /* _WIN32 */

/* __attribute__((unused)) — silence MSVC which doesn't support GCC attributes */
#if !defined(__GNUC__) && !defined(__clang__)
#define __attribute__(x)
#endif

#endif /* GV_COMPAT_H */
