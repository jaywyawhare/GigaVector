/**
 * @file fuzz_grpc_dispatch.c
 * @brief libFuzzer harness for in-memory gRPC message dispatch (no network).
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>

#include "api/grpc.h"
#include "storage/database.h"
#include "../test_tmp.h"

#define FUZZ_MAX_INPUT (64 * 1024)
#define FUZZ_MAX_FRAME (64 * 1024)
#define FUZZ_DIM 4

static GV_Database *g_db = NULL;
static GV_GrpcServer *g_server = NULL;
static char g_db_path[256];

/* The server/DB are a persistent singleton, set up once and reused across every
 * input for speed; they stay reachable for the whole run. libFuzzer exits via
 * _Exit() before any atexit teardown, so LSan sees them as leaked -- suppressed
 * in tests/fuzz/lsan.supp (leak:fuzz_setup_once), the same shutdown-ordering
 * artifact documented there for the TLS arena. */
static void fuzz_setup_once(void) {
    if (g_server) return;

    if (gv_test_make_temp_path(g_db_path, sizeof(g_db_path), "gv_fuzz_grpc", ".gv") != 0) {
        return;
    }
    remove(g_db_path);

    g_db = db_open(g_db_path, FUZZ_DIM, GV_INDEX_TYPE_FLAT);
    if (!g_db) return;

    GV_GrpcConfig cfg;
    grpc_config_init(&cfg);
    cfg.max_message_bytes = FUZZ_MAX_FRAME;
    g_server = grpc_create(g_db, &cfg);
}

static void fuzz_drain_fd(int fd) {
    uint8_t buf[4096];
    for (;;) {
        ssize_t n = recv(fd, buf, sizeof(buf), MSG_DONTWAIT);
        if (n <= 0) break;
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;

    fuzz_setup_once();
    if (!g_server) return 0;

    GV_GrpcMessage msg;
    if (grpc_decode_frame(data, size, FUZZ_MAX_FRAME, &msg) != GV_GRPC_OK) {
        return 0;
    }

    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        grpc_message_free(&msg);
        return 0;
    }

    (void)grpc_fuzz_dispatch_message(g_server, fds[1], &msg);
    fuzz_drain_fd(fds[0]);
    close(fds[0]);
    close(fds[1]);
    grpc_message_free(&msg);
    return 0;
}

#ifdef GV_FUZZ_STANDALONE
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input-file>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint8_t buf[FUZZ_MAX_INPUT];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    return LLVMFuzzerTestOneInput(buf, n);
}
#endif
