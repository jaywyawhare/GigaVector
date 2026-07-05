/**
 * @file test_grpc_frame_dst.c
 * @brief DST: encode search payload -> wire frame -> decode -> search oracle.
 */

#include <stdio.h>
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>

#include "api/grpc.h"
#include "storage/database.h"
#include "../test_tmp.h"
#include "dst_harness.h"

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1; \
        } \
    } while (0)

#define DIM 4

static void write_u32_be(uint8_t *buf, uint32_t val) {
    buf[0] = (uint8_t)(val >> 24);
    buf[1] = (uint8_t)(val >> 16);
    buf[2] = (uint8_t)(val >> 8);
    buf[3] = (uint8_t)(val);
}

static int test_grpc_frame_roundtrip_seeded(void) {
    char db_path[256];
    if (gv_test_make_temp_path(db_path, sizeof(db_path), "gv_dst_grpc", ".gv") != 0) return 0;
    remove(db_path);

    uint64_t seed = gv_dst_seed_from_env();
    size_t iters = gv_dst_iters_from_env(40);
    GV_DstRng rng = gv_dst_rng_seed(seed);

    fprintf(stderr, "DST grpc frame: seed=%llu iters=%zu\n",
            (unsigned long long)seed, iters);

    GV_Database *db = db_open(db_path, DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "db_open");

    GV_GrpcConfig cfg;
    grpc_config_init(&cfg);
    GV_GrpcServer *server = grpc_create(db, &cfg);
    ASSERT(server != NULL, "grpc_create");

    for (size_t i = 0; i < iters; i++) {
        float vec[DIM];
        for (size_t d = 0; d < DIM; d++) {
            vec[d] = gv_dst_rng_float(&rng);
        }
        ASSERT(db_add_vector(db, vec, DIM) == 0, "add_vector");

        uint8_t payload[256];
        size_t payload_len = 0;
        ASSERT(grpc_encode_search_request(vec, DIM, 5, 0, payload, sizeof(payload),
                                         &payload_len) == GV_GRPC_OK, "encode search");

        uint32_t frame_len = (uint32_t)(5 + payload_len);
        uint8_t frame[512];
        ASSERT(frame_len + 4 <= sizeof(frame), "frame fits");
        write_u32_be(frame, frame_len);
        frame[4] = GV_MSG_SEARCH;
        write_u32_be(frame + 5, (uint32_t)(i + 1));
        memcpy(frame + 9, payload, payload_len);

        GV_GrpcMessage msg;
        ASSERT(grpc_decode_frame(frame, 4 + frame_len, 65536, &msg) == GV_GRPC_OK,
               "decode frame");
        ASSERT(msg.msg_type == GV_MSG_SEARCH, "msg type");
        ASSERT(msg.payload_len == payload_len, "payload len");

        float *query = NULL;
        size_t dimension = 0;
        size_t k = 0;
        int distance_type = 0;
        ASSERT(grpc_decode_search_request(msg.payload, msg.payload_len,
                                          &query, &dimension, &k, &distance_type) == GV_GRPC_OK,
               "decode search payload");
        ASSERT(dimension == DIM, "dimension");
        ASSERT(k == 5, "k");

        GV_SearchResult results[8];
        int found = db_search(db, query, k, results, GV_DISTANCE_EUCLIDEAN);
        gv_free(query);
        grpc_message_free(&msg);
        ASSERT(found > 0, "search finds vector");
    }

    grpc_destroy(server);
    db_close(db);
    remove(db_path);
    return 0;
}

int main(void) {
    return test_grpc_frame_roundtrip_seeded();
}
