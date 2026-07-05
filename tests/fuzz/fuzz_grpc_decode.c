/**
 * @file fuzz_grpc_decode.c
 * @brief libFuzzer harness for gRPC-style search request decoding.
 *
 * Build: make fuzz-grpc-decode  (requires clang)
 * Run:   build/fuzz/fuzz_grpc_decode tests/fuzz/corpus/grpc -runs=10000
 */

#include <stddef.h>
#include "core/memory.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "api/grpc.h"

#define FUZZ_MAX_INPUT (64 * 1024)

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > FUZZ_MAX_INPUT) return 0;

    float *query = NULL;
    size_t dimension = 0;
    size_t k = 0;
    int distance_type = 0;

    if (grpc_decode_search_request(data, size, &query, &dimension, &k, &distance_type) == 0) {
        gv_free(query);
    }
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
