#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_grpc.h"
#include "gigavector/gv_database.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_grpc_config_init(void) {
    GV_GrpcConfig config;
    memset(&config, 0xFF, sizeof(config));
    gv_grpc_config_init(&config);

    ASSERT(config.port == 50051, "default port should be 50051");
    ASSERT(config.bind_address != NULL, "default bind_address should not be NULL");
    ASSERT(strcmp(config.bind_address, "0.0.0.0") == 0, "default bind_address should be 0.0.0.0");
    ASSERT(config.max_connections > 0, "default max_connections should be positive");
    ASSERT(config.max_message_bytes > 0, "default max_message_bytes should be positive");
    ASSERT(config.thread_pool_size > 0, "default thread_pool_size should be positive");
    ASSERT(config.enable_compression == 0, "default enable_compression should be 0");
    return 0;
}

static int test_grpc_config_init_idempotent(void) {
    GV_GrpcConfig c1, c2;
    memset(&c1, 0, sizeof(c1));
    memset(&c2, 0, sizeof(c2));
    gv_grpc_config_init(&c1);
    gv_grpc_config_init(&c2);
    ASSERT(c1.port == c2.port, "port should match on repeated init");
    ASSERT(c1.max_connections == c2.max_connections, "max_connections should match");
    ASSERT(c1.thread_pool_size == c2.thread_pool_size, "thread_pool_size should match");
    return 0;
}

static int test_grpc_create_destroy(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_GrpcConfig config;
    gv_grpc_config_init(&config);

    GV_GrpcServer *server = gv_grpc_create(db, &config);
    ASSERT(server != NULL, "gv_grpc_create should succeed");

    gv_grpc_destroy(server);
    gv_db_close(db);

    /* Destroy NULL should be safe */
    gv_grpc_destroy(NULL);
    return 0;
}

static int test_grpc_create_null_db(void) {
    GV_GrpcConfig config;
    gv_grpc_config_init(&config);

    GV_GrpcServer *server = gv_grpc_create(NULL, &config);
    ASSERT(server == NULL, "gv_grpc_create with NULL db should return NULL");
    return 0;
}

static int test_grpc_is_running_before_start(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_GrpcConfig config;
    gv_grpc_config_init(&config);

    GV_GrpcServer *server = gv_grpc_create(db, &config);
    ASSERT(server != NULL, "gv_grpc_create should succeed");

    int running = gv_grpc_is_running(server);
    ASSERT(running == 0, "server should not be running before start");

    /* NULL server */
    ASSERT(gv_grpc_is_running(NULL) == 0, "is_running(NULL) should return 0");

    gv_grpc_destroy(server);
    gv_db_close(db);
    return 0;
}

static int test_grpc_error_string_all_codes(void) {
    /* Test all defined error codes */
    const char *s;

    s = gv_grpc_error_string(GV_GRPC_OK);
    ASSERT(s != NULL, "error_string for OK should not be NULL");

    s = gv_grpc_error_string(GV_GRPC_ERROR_NULL);
    ASSERT(s != NULL, "error_string for ERROR_NULL should not be NULL");

    s = gv_grpc_error_string(GV_GRPC_ERROR_CONFIG);
    ASSERT(s != NULL, "error_string for ERROR_CONFIG should not be NULL");

    s = gv_grpc_error_string(GV_GRPC_ERROR_RUNNING);
    ASSERT(s != NULL, "error_string for ERROR_RUNNING should not be NULL");

    s = gv_grpc_error_string(GV_GRPC_ERROR_NOT_RUNNING);
    ASSERT(s != NULL, "error_string for ERROR_NOT_RUNNING should not be NULL");

    s = gv_grpc_error_string(GV_GRPC_ERROR_START);
    ASSERT(s != NULL, "error_string for ERROR_START should not be NULL");

    s = gv_grpc_error_string(GV_GRPC_ERROR_MEMORY);
    ASSERT(s != NULL, "error_string for ERROR_MEMORY should not be NULL");

    s = gv_grpc_error_string(GV_GRPC_ERROR_BIND);
    ASSERT(s != NULL, "error_string for ERROR_BIND should not be NULL");

    /* Unknown error code should still return a safe string */
    s = gv_grpc_error_string(-999);
    ASSERT(s != NULL, "error_string for unknown code should not be NULL");
    return 0;
}

static int test_grpc_error_strings_distinct(void) {
    /* Different error codes should have different messages */
    const char *ok = gv_grpc_error_string(GV_GRPC_OK);
    const char *null_err = gv_grpc_error_string(GV_GRPC_ERROR_NULL);
    ASSERT(strcmp(ok, null_err) != 0, "OK and ERROR_NULL should have different messages");

    const char *config_err = gv_grpc_error_string(GV_GRPC_ERROR_CONFIG);
    const char *bind_err = gv_grpc_error_string(GV_GRPC_ERROR_BIND);
    ASSERT(strcmp(config_err, bind_err) != 0, "ERROR_CONFIG and ERROR_BIND should differ");
    return 0;
}

static int test_grpc_encode_search_request(void) {
    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t buf[1024];
    size_t out_len = 0;

    int rc = gv_grpc_encode_search_request(query, 4, 10, 0, buf, sizeof(buf), &out_len);
    ASSERT(rc == 0, "encode_search_request should succeed");
    ASSERT(out_len > 0, "encoded length should be > 0");

    /* Decode and verify round-trip */
    float *decoded_query = NULL;
    size_t decoded_dim = 0, decoded_k = 0;
    int decoded_dist_type = -1;

    rc = gv_grpc_decode_search_request(buf, out_len,
                                        &decoded_query, &decoded_dim, &decoded_k, &decoded_dist_type);
    ASSERT(rc == 0, "decode_search_request should succeed");
    ASSERT(decoded_dim == 4, "decoded dimension should be 4");
    ASSERT(decoded_k == 10, "decoded k should be 10");
    ASSERT(decoded_dist_type == 0, "decoded distance_type should be 0");
    ASSERT(decoded_query != NULL, "decoded query should not be NULL");

    /* Verify vector values */
    for (size_t i = 0; i < 4; i++) {
        ASSERT(decoded_query[i] == query[i], "decoded query values should match");
    }

    free(decoded_query);
    return 0;
}

static int test_grpc_encode_search_request_large_k(void) {
    float query[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    uint8_t buf[2048];
    size_t out_len = 0;

    int rc = gv_grpc_encode_search_request(query, 8, 1000, 1, buf, sizeof(buf), &out_len);
    ASSERT(rc == 0, "encode with large k should succeed");
    ASSERT(out_len > 0, "encoded length should be > 0");

    float *decoded_query = NULL;
    size_t decoded_dim = 0, decoded_k = 0;
    int decoded_dist_type = -1;

    rc = gv_grpc_decode_search_request(buf, out_len,
                                        &decoded_query, &decoded_dim, &decoded_k, &decoded_dist_type);
    ASSERT(rc == 0, "decode should succeed");
    ASSERT(decoded_dim == 8, "decoded dimension should be 8");
    ASSERT(decoded_k == 1000, "decoded k should be 1000");
    ASSERT(decoded_dist_type == 1, "decoded distance_type should be 1");

    free(decoded_query);
    return 0;
}

static int test_grpc_encode_search_request_small_buf(void) {
    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t buf[4]; /* Too small: needs 12 + 4*4 = 28 bytes */
    size_t out_len = 0;

    int rc = gv_grpc_encode_search_request(query, 4, 10, 0, buf, sizeof(buf), &out_len);
    ASSERT(rc != 0, "encode with too-small buffer should fail");
    return 0;
}

static int test_grpc_decode_search_request_invalid(void) {
    float *query = NULL;
    size_t dim = 0, k = 0;
    int dist_type = 0;

    /* Too-short buffer (needs at least 12 bytes for the header) */
    uint8_t short_buf[4] = {0x00, 0x00, 0x00, 0x04};
    int rc = gv_grpc_decode_search_request(short_buf, sizeof(short_buf),
                                            &query, &dim, &k, &dist_type);
    ASSERT(rc != 0, "decode of too-short data should fail");

    /* Zero length */
    uint8_t dummy[1] = {0};
    rc = gv_grpc_decode_search_request(dummy, 0, &query, &dim, &k, &dist_type);
    ASSERT(rc != 0, "decode with zero length should fail");

    /* NULL buffer */
    rc = gv_grpc_decode_search_request(NULL, 10, &query, &dim, &k, &dist_type);
    ASSERT(rc != 0, "decode with NULL buffer should fail");

    /* Header claims large dimension but buffer is too small */
    uint8_t trunc_buf[12] = {0};
    trunc_buf[3] = 100; /* dimension = 100, but no float data follows */
    rc = gv_grpc_decode_search_request(trunc_buf, sizeof(trunc_buf),
                                        &query, &dim, &k, &dist_type);
    ASSERT(rc != 0, "decode with truncated payload should fail");
    return 0;
}

static int test_grpc_encode_add_request(void) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t buf[1024];
    size_t out_len = 0;

    int rc = gv_grpc_encode_add_request(data, 4, buf, sizeof(buf), &out_len);
    ASSERT(rc == 0, "encode_add_request should succeed");
    ASSERT(out_len > 0, "encoded length should be > 0");
    return 0;
}

static int test_grpc_encode_add_request_small_buf(void) {
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t buf[2]; /* Too small: needs 4 + 4*4 = 20 bytes */
    size_t out_len = 0;

    int rc = gv_grpc_encode_add_request(data, 4, buf, sizeof(buf), &out_len);
    ASSERT(rc != 0, "encode_add_request with too-small buffer should fail");
    return 0;
}

static int test_grpc_get_stats_initial(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_GrpcConfig config;
    gv_grpc_config_init(&config);

    GV_GrpcServer *server = gv_grpc_create(db, &config);
    ASSERT(server != NULL, "gv_grpc_create should succeed");

    GV_GrpcStats stats;
    memset(&stats, 0xFF, sizeof(stats));
    int rc = gv_grpc_get_stats(server, &stats);
    ASSERT(rc == 0, "get_stats should succeed");
    ASSERT(stats.total_requests == 0, "initial total_requests should be 0");
    ASSERT(stats.active_connections == 0, "initial active_connections should be 0");
    ASSERT(stats.bytes_sent == 0, "initial bytes_sent should be 0");
    ASSERT(stats.bytes_received == 0, "initial bytes_received should be 0");
    ASSERT(stats.errors == 0, "initial errors should be 0");

    gv_grpc_destroy(server);
    gv_db_close(db);
    return 0;
}

static int test_grpc_get_stats_null(void) {
    GV_GrpcStats stats;
    int rc = gv_grpc_get_stats(NULL, &stats);
    ASSERT(rc == -1, "get_stats(NULL) should return -1");
    return 0;
}

static int test_grpc_start_stop(void) {
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "create test database");

    GV_GrpcConfig config;
    gv_grpc_config_init(&config);
    /* Use a high port to avoid conflicts */
    config.port = 59999;

    GV_GrpcServer *server = gv_grpc_create(db, &config);
    ASSERT(server != NULL, "gv_grpc_create should succeed");

    /* Start the server */
    int rc = gv_grpc_start(server);
    ASSERT(rc == 0, "gv_grpc_start should succeed");

    int running = gv_grpc_is_running(server);
    ASSERT(running == 1, "server should be running after start");

    /* Starting again should fail (already running) */
    rc = gv_grpc_start(server);
    ASSERT(rc != 0, "starting already-running server should fail");

    /* Stop the server */
    rc = gv_grpc_stop(server);
    ASSERT(rc == 0, "gv_grpc_stop should succeed");

    running = gv_grpc_is_running(server);
    ASSERT(running == 0, "server should not be running after stop");

    /* Stopping again should fail (not running) */
    rc = gv_grpc_stop(server);
    ASSERT(rc != 0, "stopping already-stopped server should fail");

    gv_grpc_destroy(server);
    gv_db_close(db);
    return 0;
}

static int test_grpc_stop_null(void) {
    int rc = gv_grpc_stop(NULL);
    ASSERT(rc != 0, "gv_grpc_stop(NULL) should fail");

    rc = gv_grpc_start(NULL);
    ASSERT(rc != 0, "gv_grpc_start(NULL) should fail");
    return 0;
}

static int test_grpc_msg_type_values(void) {
    ASSERT(GV_MSG_ADD_VECTOR == 1, "GV_MSG_ADD_VECTOR should be 1");
    ASSERT(GV_MSG_SEARCH == 2, "GV_MSG_SEARCH should be 2");
    ASSERT(GV_MSG_DELETE == 3, "GV_MSG_DELETE should be 3");
    ASSERT(GV_MSG_UPDATE == 4, "GV_MSG_UPDATE should be 4");
    ASSERT(GV_MSG_GET == 5, "GV_MSG_GET should be 5");
    ASSERT(GV_MSG_BATCH_ADD == 6, "GV_MSG_BATCH_ADD should be 6");
    ASSERT(GV_MSG_BATCH_SEARCH == 7, "GV_MSG_BATCH_SEARCH should be 7");
    ASSERT(GV_MSG_STATS == 8, "GV_MSG_STATS should be 8");
    ASSERT(GV_MSG_HEALTH == 9, "GV_MSG_HEALTH should be 9");
    ASSERT(GV_MSG_SAVE == 10, "GV_MSG_SAVE should be 10");
    ASSERT(GV_MSG_RESPONSE == 128, "GV_MSG_RESPONSE should be 128");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing grpc_config_init...", test_grpc_config_init},
        {"Testing grpc_config_init_idempotent...", test_grpc_config_init_idempotent},
        {"Testing grpc_create_destroy...", test_grpc_create_destroy},
        {"Testing grpc_create_null_db...", test_grpc_create_null_db},
        {"Testing grpc_is_running_before_start...", test_grpc_is_running_before_start},
        {"Testing grpc_error_string_all_codes...", test_grpc_error_string_all_codes},
        {"Testing grpc_error_strings_distinct...", test_grpc_error_strings_distinct},
        {"Testing grpc_encode_search_request...", test_grpc_encode_search_request},
        {"Testing grpc_encode_search_large_k...", test_grpc_encode_search_request_large_k},
        {"Testing grpc_encode_search_small_buf...", test_grpc_encode_search_request_small_buf},
        {"Testing grpc_decode_search_invalid...", test_grpc_decode_search_request_invalid},
        {"Testing grpc_encode_add_request...", test_grpc_encode_add_request},
        {"Testing grpc_encode_add_small_buf...", test_grpc_encode_add_request_small_buf},
        {"Testing grpc_get_stats_initial...", test_grpc_get_stats_initial},
        {"Testing grpc_get_stats_null...", test_grpc_get_stats_null},
        {"Testing grpc_start_stop...", test_grpc_start_stop},
        {"Testing grpc_stop_null...", test_grpc_stop_null},
        {"Testing grpc_msg_type_values...", test_grpc_msg_type_values},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("  %s ", tests[i].name);
        if (tests[i].fn() == 0) {
            printf("OK\n");
            passed++;
        } else {
            printf("FAILED\n");
        }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
