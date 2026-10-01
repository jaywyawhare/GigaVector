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

/* MinGW/ucrt ships a real struct timeval AND gettimeofday in <sys/time.h>;
 * pulling it here makes gettimeofday available to every TU that includes
 * compat.h (several callers don't include <sys/time.h> themselves). MSVC has
 * neither, so there we define struct timeval and a gettimeofday shim instead.
 * Defining both only on MSVC avoids a struct-timeval redefinition on MinGW. */
#ifdef __GNUC__
#include <sys/time.h>
#else
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
#endif /* !__GNUC__ */

#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#endif

/* getline() is POSIX and absent on MinGW/MSVC even with _GNU_SOURCE. Provide a
 * portable shim: reads a line via fgetc, growing *lineptr with realloc. Returns
 * the length (excluding the NUL), or -1 at EOF with nothing read. ssize_t is
 * typedef'd just above for Windows. */
#include <stdio.h>
#include <stdlib.h>
static inline ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    if (!lineptr || !n || !stream) return -1;
    if (!*lineptr || *n == 0) {
        size_t cap = 128;
        char *buf = (char *)realloc(*lineptr, cap);
        if (!buf) return -1;
        *lineptr = buf;
        *n = cap;
    }
    size_t len = 0;
    int c;
    while ((c = fgetc(stream)) != EOF) {
        if (len + 1 >= *n) {
            size_t cap = *n * 2;
            char *buf = (char *)realloc(*lineptr, cap);
            if (!buf) return -1;
            *lineptr = buf;
            *n = cap;
        }
        (*lineptr)[len++] = (char)c;
        if (c == '\n') break;
    }
    if (len == 0 && c == EOF) return -1;
    (*lineptr)[len] = '\0';
    return (ssize_t)len;
}

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
 * POSIX I/O shims for MSVC (scoped to _WIN32 && !__GNUC__; POSIX/MinGW use real
 * syscalls). read/write/close are static inline functions, NOT object-like
 * macros: bare macros would textually clobber any identifier spelled read/write/
 * close (e.g. struct member "x.read"), whereas a function name only participates
 * in ordinary-identifier lookup. "open" stays a function-like macro (only expands
 * before '(', so "x.open" is safe) to force _O_BINARY and keep the variadic mode.
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

/* ftruncate() is POSIX and absent on BOTH MinGW and MSVC; both provide the
 * 64-bit _chsize_s() via <io.h>. MSVC already pulled <io.h> in above (under the
 * _CRT_DECLARE_NONSTDC_NAMES guard); MinGW needs it here. _chsize_s returns 0 on
 * success / errno on failure, matching ftruncate's != 0 error convention. */
#ifdef __GNUC__
#include <io.h>
#endif
static inline int ftruncate(int fd, long long length) {
    return _chsize_s(fd, length);
}

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

/* Portable logical CPU count, always >= 1. sysconf(_SC_NPROCESSORS_ONLN) is
 * POSIX-only; Windows has no equivalent and reports via GetSystemInfo(). */
#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
static inline long gv_get_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    long ncpu = (long)si.dwNumberOfProcessors;
#elif defined(__APPLE__)
    /* macOS hides _SC_NPROCESSORS_ONLN when a TU defines _POSIX_C_SOURCE, so
     * query the count through sysctl, which is always available. */
    int n = 0; size_t sz = sizeof(n);
    long ncpu = (sysctlbyname("hw.logicalcpu", &n, &sz, NULL, 0) == 0 && n > 0)
                    ? (long)n : 1;
#elif defined(_SC_NPROCESSORS_ONLN)
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
#else
    long ncpu = 1;
#endif
    return ncpu > 0 ? ncpu : 1;
}

/* Warn when a function's return value is ignored (GCC/Clang; no-op elsewhere). */
#if defined(__GNUC__) || defined(__clang__)
#define GV_NODISCARD __attribute__((warn_unused_result))
#else
#define GV_NODISCARD
#endif

#endif /* GV_COMPAT_H */
