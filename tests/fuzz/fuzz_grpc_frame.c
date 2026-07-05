/**
 * @file fuzz_grpc_frame.c
 * @brief libFuzzer harness for gRPC-style wire frame parsing + payload decoders.
 */

#include <stddef.h>
#include "core/memory.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "api/grpc.h"

#define FUZZ_MAX_INPUT (256 * 1024)
#define FUZZ_MAX_FRAME (64 * 1024)

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;

    GV_GrpcMessage msg;
    if (grpc_decode_frame(data, size, FUZZ_MAX_FRAME, &msg) != GV_GRPC_OK) {
        return 0;
    }

    if (msg.msg_type == GV_MSG_SEARCH && msg.payload_len > 0) {
        float *query = NULL;
        size_t dimension = 0;
        size_t k = 0;
        int distance_type = 0;
        if (grpc_decode_search_request(msg.payload, msg.payload_len,
                                       &query, &dimension, &k, &distance_type) == 0) {
            gv_free(query);
        }
    }

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
