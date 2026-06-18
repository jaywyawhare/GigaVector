#ifndef GV_TEST_TMP_H
#define GV_TEST_TMP_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static inline const char *gv_test_tmp_root(void) {
    const char *root = getenv("TMPDIR");
    if (!root || !*root) root = getenv("TEMP");
    if (!root || !*root) root = getenv("TMP");
    if (!root || !*root) root = "/tmp";
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

static inline int gv_test_mkdtemp(char *buf, size_t size, const char *stem) {
    static unsigned seq;
    seq++;
    int n = snprintf(buf, size, "%s/%s_%lu_%u",
                     gv_test_tmp_root(), stem, gv_test_pid(), seq);
    if (n < 0 || (size_t)n >= size) return -1;
    return gv_test_mkdir(buf);
}

static inline int gv_test_mkstemp(char *buf, size_t size, const char *stem) {
    static unsigned seq;
    seq++;
    int n = snprintf(buf, size, "%s/%s_%lu_%u",
                     gv_test_tmp_root(), stem, gv_test_pid(), seq);
    if (n < 0 || (size_t)n >= size) return -1;
#ifdef _WIN32
    return _open(buf, _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    return open(buf, O_RDWR | O_CREAT | O_EXCL, 0600);
#endif
}

static inline void gv_test_remove_db(const char *path) {
    if (path == NULL || !*path) {
        return;
    }
    remove(path);
    char wal_path[512];
    int n = snprintf(wal_path, sizeof(wal_path), "%s.wal", path);
    if (n > 0 && (size_t)n < sizeof(wal_path)) {
        remove(wal_path);
    }
}

#endif
