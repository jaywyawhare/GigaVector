/**
 * @file test_rest_handlers.c
 * @brief Unit tests for REST API handlers (rest_handlers.h).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/database.h"
#include "api/server.h"
#include "api/rest_handlers.h"
#include "features/json.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

#define TEST_DB "tmp_test_rest.bin"
#define TEST_DIM 4

static GV_HandlerContext create_test_ctx(GV_Database *db, GV_ServerConfig *scfg) {
    server_config_init(scfg);
    GV_HandlerContext ctx;
    ctx.db = db;
    ctx.config = scfg;
    return ctx;
}

static int test_response_success(void) {
    GV_HttpResponse *resp = rest_response_success("Operation completed");
    ASSERT(resp != NULL, "response creation");
    ASSERT(resp->status == GV_HTTP_200_OK, "status should be 200");
    ASSERT(resp->body != NULL, "body should not be NULL");
    ASSERT(resp->body_length > 0, "body_length should be > 0");
    ASSERT(strstr(resp->body, "success") != NULL, "body should contain 'success'");
    ASSERT(strstr(resp->body, "true") != NULL, "body should contain 'true'");
    ASSERT(strstr(resp->body, "Operation completed") != NULL, "body should contain message");

    rest_response_free(resp);
    return 0;
}

static int test_response_success_messages(void) {
    const char *messages[] = {
        "Vector inserted",
        "Database saved",
        "",
        "Compaction completed successfully"
    };

    for (int i = 0; i < 4; i++) {
        GV_HttpResponse *resp = rest_response_success(messages[i]);
        ASSERT(resp != NULL, "response creation for each message");
        ASSERT(resp->status == GV_HTTP_200_OK, "status should be 200");
        rest_response_free(resp);
    }

    return 0;
}

static int test_response_error_codes(void) {
    struct {
        GV_HttpStatus status;
        const char *code;
        const char *message;
    } cases[] = {
        {GV_HTTP_400_BAD_REQUEST,     "bad_request",     "Invalid input data"},
        {GV_HTTP_401_UNAUTHORIZED,    "unauthorized",    "Missing API key"},
        {GV_HTTP_403_FORBIDDEN,       "forbidden",       "Insufficient permissions"},
        {GV_HTTP_404_NOT_FOUND,       "not_found",       "Resource not found"},
        {GV_HTTP_405_METHOD_NOT_ALLOWED, "method_not_allowed", "Use POST instead"},
        {GV_HTTP_500_INTERNAL_ERROR,  "internal_error",  "Unexpected server error"},
    };

    for (int i = 0; i < 6; i++) {
        GV_HttpResponse *resp = rest_response_error(
            cases[i].status, cases[i].code, cases[i].message);
        ASSERT(resp != NULL, "error response creation");
        ASSERT(resp->status == cases[i].status, "status code should match");
        ASSERT(resp->body != NULL, "body should not be NULL");
        ASSERT(strstr(resp->body, cases[i].code) != NULL, "body should contain error code");
        ASSERT(strstr(resp->body, cases[i].message) != NULL, "body should contain message");
        rest_response_free(resp);
    }

    return 0;
}

static int test_response_free_null(void) {
    rest_response_free(NULL);
    return 0;
}

static int test_parse_path_param(void) {
    char param[64];
    int ret;

    ret = rest_parse_path_param("/vectors/42", "/vectors/", param, sizeof(param));
    ASSERT(ret == 0, "parse /vectors/42");
    ASSERT(strcmp(param, "42") == 0, "param should be '42'");

    ret = rest_parse_path_param("/vectors/99/details", "/vectors/", param, sizeof(param));
    ASSERT(ret == 0, "parse with trailing path");
    ASSERT(strcmp(param, "99") == 0, "param should be '99'");

    ret = rest_parse_path_param("/vectors/7?format=json", "/vectors/", param, sizeof(param));
    ASSERT(ret == 0, "parse with query string");
    ASSERT(strcmp(param, "7") == 0, "param should be '7'");

    ret = rest_parse_path_param("/health", "/vectors/", param, sizeof(param));
    ASSERT(ret == -1, "wrong prefix should return -1");

    ret = rest_parse_path_param("/vectors/", "/vectors/", param, sizeof(param));
    (void)ret;

    return 0;
}

static int test_parse_path_param_edge(void) {
    char param[8];
    int ret;

    ret = rest_parse_path_param("/vectors/1234567", "/vectors/", param, sizeof(param));
    ASSERT(ret == 0, "parse large param");

    ret = rest_parse_path_param("/vectors/0", "/vectors/", param, sizeof(param));
    ASSERT(ret == 0, "parse zero param");
    ASSERT(strcmp(param, "0") == 0, "param should be '0'");

    return 0;
}

static int test_parse_query_param(void) {
    char value[64];
    int ret;

    ret = rest_parse_query_param("k=10", "k", value, sizeof(value));
    ASSERT(ret == 0, "parse single query param");
    ASSERT(strcmp(value, "10") == 0, "value should be '10'");

    ret = rest_parse_query_param("k=10&distance=cosine&format=json",
                                     "distance", value, sizeof(value));
    ASSERT(ret == 0, "parse middle query param");
    ASSERT(strcmp(value, "cosine") == 0, "value should be 'cosine'");

    ret = rest_parse_query_param("k=10&distance=cosine&format=json",
                                     "format", value, sizeof(value));
    ASSERT(ret == 0, "parse last query param");
    ASSERT(strcmp(value, "json") == 0, "value should be 'json'");

    ret = rest_parse_query_param("k=10&distance=cosine", "missing", value, sizeof(value));
    ASSERT(ret == -1, "missing param should return -1");

    return 0;
}

static int test_parse_query_param_edge(void) {
    char value[64];
    int ret;

    ret = rest_parse_query_param("", "k", value, sizeof(value));
    ASSERT(ret == -1, "empty query string should return -1");

    ret = rest_parse_query_param("k=&other=5", "k", value, sizeof(value));
    (void)ret;

    return 0;
}

static int test_handle_health(void) {
    remove(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_GET,
        .url = "/health",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = NULL,
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_handle_health(&ctx, &request);
    ASSERT(resp != NULL, "health response creation");
    ASSERT(resp->status == GV_HTTP_200_OK, "health status should be 200");
    ASSERT(resp->body != NULL, "health body should not be NULL");
    ASSERT(strstr(resp->body, "status") != NULL, "body should contain 'status'");

    rest_response_free(resp);
    db_close(db);
    remove(TEST_DB);
    return 0;
}

static int test_handle_stats(void) {
    remove(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[] = {0.0f, 0.0f, 1.0f, 0.0f};
    db_add_vector(db, v1, TEST_DIM);
    db_add_vector(db, v2, TEST_DIM);
    db_add_vector(db, v3, TEST_DIM);

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_GET,
        .url = "/stats",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = NULL,
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_handle_stats(&ctx, &request);
    ASSERT(resp != NULL, "stats response creation");
    ASSERT(resp->status == GV_HTTP_200_OK, "stats status should be 200");
    ASSERT(resp->body != NULL, "stats body should not be NULL");
    ASSERT(strstr(resp->body, "total_vectors") != NULL, "body should contain total_vectors");
    ASSERT(strstr(resp->body, "3") != NULL, "body should contain count 3");

    rest_response_free(resp);
    db_close(db);
    remove(TEST_DB);
    return 0;
}

static int test_handle_stats_empty(void) {
    remove(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_GET,
        .url = "/stats",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = NULL,
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_handle_stats(&ctx, &request);
    ASSERT(resp != NULL, "stats response for empty DB");
    ASSERT(resp->status == GV_HTTP_200_OK, "status should be 200");
    ASSERT(resp->body != NULL, "body should not be NULL");

    rest_response_free(resp);
    db_close(db);
    remove(TEST_DB);
    return 0;
}

static int test_route_get_health(void) {
    remove(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_GET,
        .url = "/health",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = NULL,
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_route(&ctx, &request);
    ASSERT(resp != NULL, "route health response");
    ASSERT(resp->status == GV_HTTP_200_OK, "route health status 200");

    rest_response_free(resp);
    db_close(db);
    remove(TEST_DB);
    return 0;
}

static int test_route_get_stats(void) {
    remove(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_GET,
        .url = "/stats",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = NULL,
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_route(&ctx, &request);
    ASSERT(resp != NULL, "route stats response");
    ASSERT(resp->status == GV_HTTP_200_OK, "route stats status 200");

    rest_response_free(resp);
    db_close(db);
    remove(TEST_DB);
    return 0;
}

static int test_route_not_found(void) {
    remove(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_GET,
        .url = "/nonexistent/path",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = NULL,
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_route(&ctx, &request);
    ASSERT(resp != NULL, "route 404 response");
    ASSERT(resp->status == GV_HTTP_404_NOT_FOUND, "unknown path should return 404");

    rest_response_free(resp);
    db_close(db);
    remove(TEST_DB);
    return 0;
}

static int test_route_method_mismatch(void) {
    remove(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_POST,
        .url = "/health",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = "application/json",
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_route(&ctx, &request);
    ASSERT(resp != NULL, "route method mismatch response");
    /* Should return 404 or 405 depending on router implementation */
    ASSERT(resp->status == GV_HTTP_404_NOT_FOUND ||
           resp->status == GV_HTTP_405_METHOD_NOT_ALLOWED,
           "POST /health should return 404 or 405");

    rest_response_free(resp);
    db_close(db);
    remove(TEST_DB);
    return 0;
}

int main(void) {
    int failed = 0;
    int passed = 0;

    remove(TEST_DB);

    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_response_success",          test_response_success},
        {"test_response_success_messages", test_response_success_messages},
        {"test_response_error_codes",      test_response_error_codes},
        {"test_response_free_null",        test_response_free_null},
        {"test_parse_path_param",          test_parse_path_param},
        {"test_parse_path_param_edge",     test_parse_path_param_edge},
        {"test_parse_query_param",         test_parse_query_param},
        {"test_parse_query_param_edge",    test_parse_query_param_edge},
        {"test_handle_health",             test_handle_health},
        {"test_handle_stats",              test_handle_stats},
        {"test_handle_stats_empty",        test_handle_stats_empty},
        {"test_route_get_health",          test_route_get_health},
        {"test_route_get_stats",           test_route_get_stats},
        {"test_route_not_found",           test_route_not_found},
        {"test_route_method_mismatch",     test_route_method_mismatch},
    };

    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < num_tests; i++) {
        int result = tests[i].fn();
        if (result == 0) {
            printf("  OK   %s\n", tests[i].name);
            passed++;
        } else {
            printf("  FAILED %s\n", tests[i].name);
            failed++;
        }
    }

    printf("\n%d/%d tests passed\n", passed, num_tests);
    remove(TEST_DB);
    return failed > 0 ? 1 : 0;
}
