#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gigavector/gv_mmap.h"

struct GV_MMap {
    void *addr;
    size_t size;
    int fd;
};

GV_MMap *gv_mmap_open_readonly(const char *path) {
    if (path == NULL) {
        return NULL;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }
    if (st.st_size == 0) {
        close(fd);
        return NULL;
    }

    void *addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    GV_MMap *mm = (GV_MMap *)malloc(sizeof(GV_MMap));
    if (!mm) {
        munmap(addr, (size_t)st.st_size);
        close(fd);
        return NULL;
    }

    mm->addr = addr;
    mm->size = (size_t)st.st_size;
    mm->fd = fd;
    return mm;
}

void gv_mmap_close(GV_MMap *mm) {
    if (!mm) {
        return;
    }
    if (mm->addr && mm->size > 0) {
        munmap(mm->addr, mm->size);
    }
    if (mm->fd >= 0) {
        close(mm->fd);
    }
    free(mm);
}

const void *gv_mmap_data(const GV_MMap *mm) {
    if (!mm) {
        return NULL;
    }
    return mm->addr;
}

size_t gv_mmap_size(const GV_MMap *mm) {
    if (!mm) {
        return 0;
    }
    return mm->size;
}


