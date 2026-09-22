/**
 * @file net_io.h
 * @brief Blocking full-read/full-write helpers shared by the RPC transports.
 */

#ifndef GIGAVECTOR_GV_NET_IO_H
#define GIGAVECTOR_GV_NET_IO_H

#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>

/** @brief Write @p len bytes, retrying short/interrupted writes. @return 0 or -1. */
static inline int gv_net_write_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) return -1;
        p += n; len -= (size_t)n;
    }
    return 0;
}

/** @brief Read exactly @p len bytes, retrying short/interrupted reads. @return 0 or -1. */
static inline int gv_net_read_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) return -1;
        p += n; len -= (size_t)n;
    }
    return 0;
}

#endif /* GIGAVECTOR_GV_NET_IO_H */
