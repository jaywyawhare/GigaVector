/**
 * @file log.c
 * @brief Pluggable failure-logging hook implementation.
 *
 * Dependency-free (stdio/stdarg/pthread/stdlib only). Hook and level globals are
 * guarded by a single mutex; a pthread_once handles one-time init that reads the
 * GV_LOG_LEVEL / GV_LOG_STDERR environment variables.
 */

#include "core/log.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* Guards g_hook, g_ctx, and g_max_level. */
static pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_log_once = PTHREAD_ONCE_INIT;

static GV_LogHook g_hook = NULL;
static void *g_ctx = NULL;
static int g_max_level = GV_LOG_ERROR;

/* Built-in stderr sink, installed only when GV_LOG_STDERR is set. */
static const char *level_name(int level) {
    switch (level) {
        case GV_LOG_ERROR: return "ERROR";
        case GV_LOG_WARN:  return "WARN";
        case GV_LOG_INFO:  return "INFO";
        case GV_LOG_DEBUG: return "DEBUG";
        default:           return "?";
    }
}

static void gv_log_stderr_sink(int level, const char *file, int line,
                               const char *msg, void *ctx) {
    (void)ctx;
    fprintf(stderr, "[gv][%s] %s:%d: %s\n", level_name(level),
            file ? file : "?", line, msg ? msg : "");
    fflush(stderr);
}

/* One-time initialization: honor GV_LOG_LEVEL and GV_LOG_STDERR. Runs under
 * pthread_once, so it executes exactly once regardless of concurrent callers.
 * We take the mutex here too so the visibility of the writes is consistent with
 * the rest of the accessors. */
static void gv_log_init_once(void) {
    int level = GV_LOG_ERROR;
    const char *lvl_env = getenv("GV_LOG_LEVEL");
    if (lvl_env != NULL && lvl_env[0] != '\0') {
        char *end = NULL;
        long v = strtol(lvl_env, &end, 10);
        if (end != lvl_env && v >= GV_LOG_ERROR && v <= GV_LOG_DEBUG) {
            level = (int)v;
        }
    }

    GV_LogHook hook = NULL;
    const char *stderr_env = getenv("GV_LOG_STDERR");
    if (stderr_env != NULL) {
        hook = gv_log_stderr_sink;
    }

    pthread_mutex_lock(&g_log_mutex);
    g_max_level = level;
    if (hook != NULL) {
        g_hook = hook;
        g_ctx = NULL;
    }
    pthread_mutex_unlock(&g_log_mutex);
}

void gv_log_set_hook(GV_LogHook hook, void *ctx) {
    pthread_once(&g_log_once, gv_log_init_once);
    pthread_mutex_lock(&g_log_mutex);
    g_hook = hook;
    g_ctx = ctx;
    pthread_mutex_unlock(&g_log_mutex);
}

void gv_log_set_level(int max_level) {
    pthread_once(&g_log_once, gv_log_init_once);
    if (max_level < GV_LOG_ERROR) {
        max_level = GV_LOG_ERROR;
    } else if (max_level > GV_LOG_DEBUG) {
        max_level = GV_LOG_DEBUG;
    }
    pthread_mutex_lock(&g_log_mutex);
    g_max_level = max_level;
    pthread_mutex_unlock(&g_log_mutex);
}

void gv_log_emit(int level, const char *file, int line, const char *fmt, ...) {
    pthread_once(&g_log_once, gv_log_init_once);

    /* Snapshot the hook/level under the lock, then release before formatting
     * and invoking the hook so we never hold the lock across user code. */
    pthread_mutex_lock(&g_log_mutex);
    GV_LogHook hook = g_hook;
    void *ctx = g_ctx;
    int max_level = g_max_level;
    pthread_mutex_unlock(&g_log_mutex);

    if (hook == NULL) {
        return; /* No sink installed: stay silent (tests must stay quiet). */
    }
    if (level > max_level) {
        return; /* Suppressed: less severe than the configured maximum. */
    }

    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt != NULL ? fmt : "", ap);
    va_end(ap);
    if (n < 0) {
        buf[0] = '\0'; /* Encoding error: emit an empty message rather than junk. */
    }

    hook(level, file, line, buf, ctx);
}
