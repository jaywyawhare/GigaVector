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

#ifndef strcasecmp
#define strcasecmp(a,b)    _stricmp((a),(b))
#endif
#ifndef strncasecmp
#define strncasecmp(a,b,n) _strnicmp((a),(b),(n))
#endif

#ifndef strtok_r
#define strtok_r(s,d,p) strtok_s((s),(d),(p))
#endif

/* strcasestr is a GNU/BSD extension absent on MinGW/MSVC; provide a portable
 * case-insensitive substring search. POSIX platforms use the system version. */
#include <ctype.h>
#include <string.h>
static inline char *strcasestr(const char *haystack, const char *needle) {
    if (!*needle) return (char *)haystack;
    for (; *haystack; haystack++) {
        const char *h = haystack, *n = needle;
        while (*h && *n &&
               tolower((unsigned char)*h) == tolower((unsigned char)*n)) {
            h++;
            n++;
        }
        if (!*n) return (char *)haystack;
    }
    return NULL;
}

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

#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#endif

#include <sys/stat.h>
#ifndef S_ISDIR
#define S_ISDIR(m)  (((m) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISREG
#define S_ISREG(m)  (((m) & _S_IFMT) == _S_IFREG)
#endif

/* __GNUC__ guard: MinGW is also _WIN32 but ships clock_gettime natively */
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

/* __GNUC__ guard: MinGW provides O_* names via <fcntl.h> */
#ifndef __GNUC__
/* Prevent MSVC's <io.h> from also declaring the bare POSIX aliases
 * (read/write/close/open); we provide our own shims below and do not want a
 * clash with prototypes of identical names. */
#ifndef _CRT_DECLARE_NONSTDC_NAMES
#define _CRT_DECLARE_NONSTDC_NAMES 0
#endif
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
/*
 * POSIX I/O shims for MSVC.
 *
 * The read/write/close shims were previously OBJECT-LIKE macros
 * (#define read _read, etc.).  As bare object-like macros they leaked into
 * every translation unit and textually replaced ANY identifier spelled
 * read/write/close -- including struct member accesses like "x.read" and
 * local variables -- breaking otherwise-valid code.  They are now
 * "static inline" shim functions with the same names: a function name only
 * participates in ordinary-identifier lookup, so "x.read" still refers to the
 * struct member and is no longer clobbered.
 *
 * "open" stays a FUNCTION-LIKE macro: function-like macros only expand when
 * the name is immediately followed by '(', so "x.open" is never rewritten --
 * there is no leakage to fix there -- and keeping it a macro preserves the
 * forced _O_BINARY flag and the optional-mode variadic call form.
 *
 * All of this is scoped to MSVC only (inside the _WIN32 and !__GNUC__ guards);
 * POSIX/MinGW builds use the real system calls unchanged.
 */
#define open(path, flags, ...) _open((path), (flags) | _O_BINARY, ##__VA_ARGS__)
static inline int close(int fd) { return _close(fd); }
static inline int read(int fd, void *buf, unsigned int count) {
    return _read(fd, buf, count);
}
static inline int write(int fd, const void *buf, unsigned int count) {
    return _write(fd, buf, count);
}
#endif /* !__GNUC__ */

#endif /* _WIN32 */

#if !defined(__GNUC__) && !defined(__clang__)
#define __attribute__(x)
#endif

/* Atomically replace `dst` with `src`. POSIX rename() already replaces an
 * existing destination atomically; Win32 rename() fails when the target
 * exists, so route through MoveFileEx there. Use for temp-file publish. */
#include <stdio.h>
#include <errno.h>
static inline int gv_rename_replace(const char *src, const char *dst) {
#ifdef _WIN32
    if (MoveFileExA(src, dst,
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return 0;
    }
    errno = (int)GetLastError(); /* surface the Win32 error to callers' logs */
    return -1;
#else
    return rename(src, dst);
#endif
}

/* Portable reentrant PRNG (POSIX rand_r is unavailable on MinGW/MSVC). Returns
 * a value in [0, 0x7FFF]; use in place of rand_r across the codebase. */
static inline int gv_rand_r(unsigned int *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return (int)((*seed >> 16) & 0x7FFF);
}

#endif /* GV_COMPAT_H */
