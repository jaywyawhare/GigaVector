#ifndef GV_TEST_TMP_H
#define GV_TEST_TMP_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static inline const char *gv_test_tmp_root(void) {
    const char *root = getenv("TMPDIR");
    if (!root || !*root) root = getenv("TEMP");
    if (!root || !*root) root = getenv("TMP");
    if (!root || !*root) root = ".";
    return root;
}

static inline unsigned long gv_test_pid(void) {
#ifdef _WIN32
    return (unsigned long)_getpid();
#else
    return (unsigned long)getpid();
#endif
}

static inline int gv_test_make_temp_path(char *buf, size_t size,
                                         const char *stem, const char *suffix) {
    int n = snprintf(buf, size, "%s/%s_%lu%s",
                     gv_test_tmp_root(), stem, gv_test_pid(), suffix ? suffix : "");
    return (n < 0 || (size_t)n >= size) ? -1 : 0;
}

static inline int gv_test_mkdir(const char *path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) return 0;
#else
    if (mkdir(path, 0700) == 0 || errno == EEXIST) return 0;
#endif
    return -1;
}

#endif
