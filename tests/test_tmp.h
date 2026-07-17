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

/*
 * Recursively remove a directory tree (or a single file). Portable: uses a
 * manual directory walk (readdir + unlink/rmdir) on POSIX and FindFirstFile on
 * Windows. Best-effort — ignores errors so it is safe to call in test teardown
 * even if the path is already gone. Returns 0 on success, -1 on failure.
 */
#ifdef _WIN32
#include <windows.h>
static inline int gv_test_rmrf(const char *path) {
    if (path == NULL || !*path) return 0;

    DWORD attr = GetFileAttributesA(path);
    if (attr == INVALID_FILE_ATTRIBUTES) return 0; /* already gone */
    if (!(attr & FILE_ATTRIBUTE_DIRECTORY)) {
        return DeleteFileA(path) ? 0 : -1;
    }

    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    int rc = 0;
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (strcmp(fd.cFileName, ".") == 0 || strcmp(fd.cFileName, "..") == 0)
                continue;
            char child[1024];
            snprintf(child, sizeof(child), "%s\\%s", path, fd.cFileName);
            if (gv_test_rmrf(child) != 0) rc = -1;
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    if (!RemoveDirectoryA(path)) rc = -1;
    return rc;
}
#else
#include <dirent.h>
static inline int gv_test_rmrf(const char *path) {
    if (path == NULL || !*path) return 0;

    struct stat st;
    if (lstat(path, &st) != 0) return 0; /* already gone */
    if (!S_ISDIR(st.st_mode)) {
        return unlink(path);
    }

    DIR *d = opendir(path);
    int rc = 0;
    if (d != NULL) {
        struct dirent *ent;
        while ((ent = readdir(d)) != NULL) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
                continue;
            char child[1024];
            int n = snprintf(child, sizeof(child), "%s/%s", path, ent->d_name);
            if (n < 0 || (size_t)n >= sizeof(child)) { rc = -1; continue; }
            if (gv_test_rmrf(child) != 0) rc = -1;
        }
        closedir(d);
    }
    if (rmdir(path) != 0) rc = -1;
    return rc;
}
#endif

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
