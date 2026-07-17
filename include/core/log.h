#ifndef GIGAVECTOR_CORE_LOG_H
#define GIGAVECTOR_CORE_LOG_H

/**
 * @file log.h
 * @brief Pluggable failure-logging hook for the core database.
 *
 * The core database has historically had no failure logging: malloc/I/O/WAL/
 * corruption errors silently returned -1/NULL, making production debugging
 * impossible. This module provides a dependency-free, thread-safe logging seam
 * that hosts can wire into their own logging infrastructure.
 *
 * Design notes:
 *   - When no hook is installed, gv_log_emit() is a no-op. It NEVER prints to
 *     stderr by default, so test suites stay quiet.
 *   - A built-in stderr sink can be opted into at process start via the
 *     GV_LOG_STDERR environment variable (any value).
 *   - The default max level is GV_LOG_ERROR, overridable once at init via the
 *     GV_LOG_LEVEL environment variable (0-3).
 *   - Hook/level globals are guarded by a mutex so set/emit are safe under
 *     concurrency.
 */

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Severity levels, ordered most- to least-severe. */
enum {
    GV_LOG_ERROR = 0,
    GV_LOG_WARN = 1,
    GV_LOG_INFO = 2,
    GV_LOG_DEBUG = 3
};

/**
 * Logging hook. Receives the already-formatted message.
 *
 * @param level One of the GV_LOG_* level constants.
 * @param file  Source file the log originated from (__FILE__).
 * @param line  Source line the log originated from (__LINE__).
 * @param msg   Formatted, NUL-terminated message (bounded length).
 * @param ctx   Opaque context registered alongside the hook.
 */
typedef void (*GV_LogHook)(int level, const char *file, int line,
                           const char *msg, void *ctx);

/**
 * Install (or clear, with hook == NULL) the global logging hook. Thread-safe.
 * The ctx pointer is passed back verbatim to every hook invocation.
 */
void gv_log_set_hook(GV_LogHook hook, void *ctx);

/**
 * Set the maximum level that will be emitted. Messages with a numerically
 * larger level (less severe) than max_level are suppressed. Thread-safe.
 */
void gv_log_set_level(int max_level);

/**
 * Format (printf-style) and emit a log record. No-op when no hook is installed
 * or the level is suppressed. Formats into a bounded stack buffer via vsnprintf;
 * callers need not worry about buffer management. Thread-safe.
 */
void gv_log_emit(int level, const char *file, int line, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 4, 5)))
#endif
    ;

/** Convenience macros capturing the call site automatically. */
#define GV_LOG_ERROR(fmt, ...) \
    gv_log_emit(GV_LOG_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define GV_LOG_WARN(fmt, ...) \
    gv_log_emit(GV_LOG_WARN, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define GV_LOG_INFO(fmt, ...) \
    gv_log_emit(GV_LOG_INFO, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define GV_LOG_DEBUG(fmt, ...) \
    gv_log_emit(GV_LOG_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_CORE_LOG_H */
