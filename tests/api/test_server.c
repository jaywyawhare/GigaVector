/**
 * @file test_server.c
 * @brief Unit tests for HTTP REST server.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "storage/database.h"
#include "api/server.h"
#include "api/rest_handlers.h"
#include "features/json.h"

#define TEST_DIM 4

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_PASS() do { \
    tests_passed++; \
} while(0)

static void test_server_config_init(void) {
    GV_ServerConfig config;
    server_config_init(&config);

    TEST_ASSERT(config.port == 6969, "Default port should be 6969");
    TEST_ASSERT(config.thread_pool_size == 4, "Default thread pool size should be 4");
    TEST_ASSERT(config.max_connections == 100, "Default max connections should be 100");
    TEST_ASSERT(config.request_timeout_ms == 30000, "Default timeout should be 30000ms");
    TEST_ASSERT(config.max_request_body_bytes == 10485760, "Default max body should be 10MB");
    TEST_ASSERT(config.enable_cors == 0, "CORS should be disabled by default");
    TEST_ASSERT(config.enable_logging == 1, "Logging should be enabled by default");
    TEST_ASSERT(config.api_key == NULL, "API key should be NULL by default");

    TEST_PASS();
}

static void test_server_create_destroy(void) {
    GV_Database *db = db_open(NULL, TEST_DIM, GV_INDEX_TYPE_KDTREE);
    TEST_ASSERT(db != NULL, "Database should be created");

    GV_Server *server = server_create(db, NULL);
    TEST_ASSERT(server != NULL, "Server should be created with default config");

    TEST_ASSERT(server_is_running(server) == 0, "Server should not be running initially");

    server_destroy(server);
    db_close(db);

    TEST_PASS();
}

static void test_server_create_custom_config(void) {
    GV_Database *db = db_open(NULL, TEST_DIM, GV_INDEX_TYPE_KDTREE);
    TEST_ASSERT(db != NULL, "Database should be created");

    GV_ServerConfig config;
    server_config_init(&config);
    config.port = 9090;
    config.enable_cors = 1;
    config.api_key = "test-api-key";

    GV_Server *server = server_create(db, &config);
    TEST_ASSERT(server != NULL, "Server should be created with custom config");

    server_destroy(server);
    db_close(db);

    TEST_PASS();
}

static void test_server_error_string(void) {
    const char *str;

    str = server_error_string(GV_SERVER_OK);
    TEST_ASSERT(str != NULL && strlen(str) > 0, "Should return error string for OK");

    str = server_error_string(GV_SERVER_ERROR_NULL_POINTER);
    TEST_ASSERT(str != NULL && strlen(str) > 0, "Should return error string for NULL_POINTER");

    str = server_error_string(GV_SERVER_ERROR_START_FAILED);
    TEST_ASSERT(str != NULL && strlen(str) > 0, "Should return error string for START_FAILED");

    str = server_error_string(-999);
    TEST_ASSERT(str != NULL && strlen(str) > 0, "Should return error string for unknown error");

    TEST_PASS();
}

static void test_rest_response_json(void) {
    GV_JsonValue *data = json_object();
    json_object_set(data, "test", json_string("value"));

    GV_HttpResponse *response = rest_response_json(data);
    TEST_ASSERT(response != NULL, "Response should be created");
    TEST_ASSERT(response->status == GV_HTTP_200_OK, "Status should be 200 OK");
    TEST_ASSERT(response->body != NULL, "Body should not be NULL");
    TEST_ASSERT(strstr(response->body, "test") != NULL, "Body should contain 'test'");

    rest_response_free(response);

    TEST_PASS();
}

static void test_rest_response_error(void) {
    GV_HttpResponse *response = rest_response_error(
        GV_HTTP_400_BAD_REQUEST, "bad_request", "Invalid input");

    TEST_ASSERT(response != NULL, "Response should be created");
    TEST_ASSERT(response->status == GV_HTTP_400_BAD_REQUEST, "Status should be 400");
    TEST_ASSERT(strstr(response->body, "bad_request") != NULL, "Body should contain error code");
    TEST_ASSERT(strstr(response->body, "Invalid input") != NULL, "Body should contain message");

    rest_response_free(response);

    TEST_PASS();
}

static void test_rest_response_success(void) {
    GV_HttpResponse *response = rest_response_success("Operation completed");

    TEST_ASSERT(response != NULL, "Response should be created");
    TEST_ASSERT(response->status == GV_HTTP_200_OK, "Status should be 200 OK");
    TEST_ASSERT(strstr(response->body, "success") != NULL, "Body should contain 'success'");
    TEST_ASSERT(strstr(response->body, "true") != NULL, "Body should contain 'true'");

    rest_response_free(response);

    TEST_PASS();
}

static void test_parse_path_param(void) {
    char param[32];
    int result;

    result = rest_parse_path_param("/vectors/123", "/vectors/", param, sizeof(param));
    TEST_ASSERT(result == 0, "Should parse path param");
    TEST_ASSERT(strcmp(param, "123") == 0, "Param should be '123'");

    result = rest_parse_path_param("/vectors/456/more", "/vectors/", param, sizeof(param));
    TEST_ASSERT(result == 0, "Should parse path param with trailing path");
    TEST_ASSERT(strcmp(param, "456") == 0, "Param should be '456'");

    result = rest_parse_path_param("/vectors/789?query=1", "/vectors/", param, sizeof(param));
    TEST_ASSERT(result == 0, "Should parse path param with query string");
    TEST_ASSERT(strcmp(param, "789") == 0, "Param should be '789'");

    result = rest_parse_path_param("/other/123", "/vectors/", param, sizeof(param));
    TEST_ASSERT(result == -1, "Should fail for wrong prefix");

    TEST_PASS();
}

static void test_parse_query_param(void) {
    char value[64];
    int result;

    result = rest_parse_query_param("k=10&distance=cosine", "k", value, sizeof(value));
    TEST_ASSERT(result == 0, "Should parse query param 'k'");
    TEST_ASSERT(strcmp(value, "10") == 0, "Value should be '10'");

    result = rest_parse_query_param("k=10&distance=cosine", "distance", value, sizeof(value));
    TEST_ASSERT(result == 0, "Should parse query param 'distance'");
    TEST_ASSERT(strcmp(value, "cosine") == 0, "Value should be 'cosine'");

    result = rest_parse_query_param("k=10&distance=cosine", "missing", value, sizeof(value));
    TEST_ASSERT(result == -1, "Should fail for missing param");

    TEST_PASS();
}

static void test_handle_health(void) {
    GV_Database *db = db_open(NULL, TEST_DIM, GV_INDEX_TYPE_KDTREE);
    TEST_ASSERT(db != NULL, "Database should be created");

    GV_HandlerContext ctx = { .db = db, .config = NULL };
    GV_HttpRequest request = { .method = GV_HTTP_GET, .url = "/health" };

    GV_HttpResponse *response = rest_handle_health(&ctx, &request);
    TEST_ASSERT(response != NULL, "Response should be created");
    TEST_ASSERT(response->status == GV_HTTP_200_OK, "Status should be 200 OK");
    TEST_ASSERT(strstr(response->body, "status") != NULL, "Body should contain 'status'");

    rest_response_free(response);
    db_close(db);

    TEST_PASS();
}

static void test_handle_stats(void) {
    GV_Database *db = db_open(NULL, TEST_DIM, GV_INDEX_TYPE_KDTREE);
    TEST_ASSERT(db != NULL, "Database should be created");

    float vec1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float vec2[] = {5.0f, 6.0f, 7.0f, 8.0f};
    db_add_vector(db, vec1, TEST_DIM);
    db_add_vector(db, vec2, TEST_DIM);

    GV_HandlerContext ctx = { .db = db, .config = NULL };
    GV_HttpRequest request = { .method = GV_HTTP_GET, .url = "/stats" };

    GV_HttpResponse *response = rest_handle_stats(&ctx, &request);
    TEST_ASSERT(response != NULL, "Response should be created");
    TEST_ASSERT(response->status == GV_HTTP_200_OK, "Status should be 200 OK");
    TEST_ASSERT(strstr(response->body, "total_vectors") != NULL, "Body should contain 'total_vectors'");
    TEST_ASSERT(strstr(response->body, "2") != NULL, "Body should contain count '2'");

    rest_response_free(response);
    db_close(db);

    TEST_PASS();
}

static void test_router_health(void) {
    GV_Database *db = db_open(NULL, TEST_DIM, GV_INDEX_TYPE_KDTREE);
    TEST_ASSERT(db != NULL, "Database should be created");

    GV_HandlerContext ctx = { .db = db, .config = NULL };
    GV_HttpRequest request = { .method = GV_HTTP_GET, .url = "/health" };

    GV_HttpResponse *response = rest_route(&ctx, &request);
    TEST_ASSERT(response != NULL, "Response should be created");
    TEST_ASSERT(response->status == GV_HTTP_200_OK, "Status should be 200 OK");

    rest_response_free(response);
    db_close(db);

    TEST_PASS();
}

static void test_router_not_found(void) {
    GV_Database *db = db_open(NULL, TEST_DIM, GV_INDEX_TYPE_KDTREE);
    TEST_ASSERT(db != NULL, "Database should be created");

    GV_HandlerContext ctx = { .db = db, .config = NULL };
    GV_HttpRequest request = { .method = GV_HTTP_GET, .url = "/nonexistent" };

    GV_HttpResponse *response = rest_route(&ctx, &request);
    TEST_ASSERT(response != NULL, "Response should be created");
    TEST_ASSERT(response->status == GV_HTTP_404_NOT_FOUND, "Status should be 404 Not Found");

    rest_response_free(response);
    db_close(db);

    TEST_PASS();
}

int main(void) {
    test_server_config_init();
    test_server_create_destroy();
    test_server_create_custom_config();
    test_server_error_string();
    test_rest_response_json();
    test_rest_response_error();
    test_rest_response_success();
    test_parse_path_param();
    test_parse_query_param();
    test_handle_health();
    test_handle_stats();
    test_router_health();
    test_router_not_found();

    return tests_failed > 0 ? 1 : 0;
}
