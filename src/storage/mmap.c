#ifndef _WIN32
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>

#include "storage/mmap.h"

#ifdef _WIN32

#include <windows.h>

struct GV_MMap {
    void  *addr;
    size_t size;
    HANDLE hFile;
    HANDLE hMapping;
};

GV_MMap *mmap_open_readonly(const char *path) {
    if (!path) return NULL;

    HANDLE hFile = CreateFileA(path, GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return NULL;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hFile, &sz) || sz.QuadPart == 0) {
        CloseHandle(hFile);
        return NULL;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        CloseHandle(hFile);
        return NULL;
    }

    void *addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return NULL;
    }

    GV_MMap *mm = (GV_MMap *)gv_alloc(sizeof(GV_MMap));
    if (!mm) {
        UnmapViewOfFile(addr);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return NULL;
    }

    mm->addr     = addr;
    mm->size     = (size_t)sz.QuadPart;
    mm->hFile    = hFile;
    mm->hMapping = hMapping;
    return mm;
}

void mmap_close(GV_MMap *mm) {
    if (!mm) return;
    if (mm->addr)    UnmapViewOfFile(mm->addr);
    if (mm->hMapping) CloseHandle(mm->hMapping);
    if (mm->hFile != INVALID_HANDLE_VALUE) CloseHandle(mm->hFile);
    gv_free(mm);
}

#else  /* POSIX */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct GV_MMap {
    void  *addr;
    size_t size;
    int    fd;
};

GV_MMap *mmap_open_readonly(const char *path) {
    if (path == NULL) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    if (st.st_size == 0)     { close(fd); return NULL; }

    void *addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) { close(fd); return NULL; }

    GV_MMap *mm = (GV_MMap *)gv_alloc(sizeof(GV_MMap));
    if (!mm) {
        munmap(addr, (size_t)st.st_size);
        close(fd);
        return NULL;
    }

    mm->addr = addr;
    mm->size = (size_t)st.st_size;
    mm->fd   = fd;
    return mm;
}

void mmap_close(GV_MMap *mm) {
    if (!mm) return;
    if (mm->addr && mm->size > 0) munmap(mm->addr, mm->size);
    if (mm->fd >= 0) close(mm->fd);
    gv_free(mm);
}

#endif /* _WIN32 */

void mmap_warmup(const GV_MMap *mm) {
    if (!mm || !mm->addr || mm->size == 0) return;
#ifndef _WIN32
    madvise(mm->addr, mm->size, MADV_WILLNEED);  /* advisory async readahead */
#endif
    /* Force residency now by faulting in one byte per 4 KiB page. */
    volatile const unsigned char *p = (const unsigned char *)mm->addr;
    unsigned char sink = 0;
    for (size_t off = 0; off < mm->size; off += 4096) sink ^= p[off];
    (void)sink;
}

int gv_file_warmup(const char *path) {
#ifdef _WIN32
    (void)path;   /* rely on the OS cache manager; no portable fadvise */
    return 0;
#else
    if (!path) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
#ifdef POSIX_FADV_WILLNEED
    posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);  /* kernel reads file into page cache */
#endif
    close(fd);
    return 0;
#endif
}

const void *mmap_data(const GV_MMap *mm) {
    return mm ? mm->addr : NULL;
}

size_t mmap_size(const GV_MMap *mm) {
    return mm ? mm->size : 0;
}
