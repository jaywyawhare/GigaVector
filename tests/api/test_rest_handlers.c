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
#include "api/txn_registry.h"
#include "features/json.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

#define TEST_DIM 4

static char test_db_path[512];
#define TEST_DB test_db_path

static int init_test_db_path(void) {
    return gv_test_make_temp_path(test_db_path, sizeof(test_db_path), "gv_test_rest", ".bin");
}

static GV_HandlerContext create_test_ctx(GV_Database *db, GV_ServerConfig *scfg) {
    server_config_init(scfg);
    GV_HandlerContext ctx;
    memset(&ctx, 0, sizeof(ctx));   /* kg/graph/txn_registry default to NULL */
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
    gv_test_remove_db(TEST_DB);
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
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_handle_stats(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[] = {0.0f, 0.0f, 1.0f, 0.0f};
    { int _r = db_add_vector(db, v1, TEST_DIM); (void)_r; }
    { int _r = db_add_vector(db, v2, TEST_DIM); (void)_r; }
    { int _r = db_add_vector(db, v3, TEST_DIM); (void)_r; }

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
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_handle_metrics(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    { int _r = db_add_vector(db, v1, TEST_DIM); (void)_r; }
    { int _r = db_add_vector(db, v2, TEST_DIM); (void)_r; }

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_GET,
        .url = "/metrics",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = NULL,
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_handle_metrics(&ctx, &request);
    ASSERT(resp != NULL, "metrics response creation");
    ASSERT(resp->status == GV_HTTP_200_OK, "metrics status should be 200");
    ASSERT(resp->body != NULL, "metrics body should not be NULL");
    ASSERT(resp->content_type != NULL && strstr(resp->content_type, "text/plain") != NULL,
           "metrics content-type is Prometheus text");
    /* Prometheus exposition: HELP/TYPE lines + sample values present. */
    ASSERT(strstr(resp->body, "# TYPE gigavector_vectors gauge") != NULL, "vectors TYPE line");
    ASSERT(strstr(resp->body, "gigavector_vectors 2") != NULL, "vectors gauge value");
    ASSERT(strstr(resp->body, "gigavector_dimension 4") != NULL, "dimension gauge value");
    ASSERT(strstr(resp->body, "gigavector_up 1") != NULL, "up gauge healthy");
    ASSERT(strstr(resp->body, "# TYPE gigavector_inserts_total counter") != NULL, "inserts TYPE line");
    ASSERT(resp->body_length == strlen(resp->body), "body_length matches body");

    rest_response_free(resp);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_handle_stats_empty(void) {
    gv_test_remove_db(TEST_DB);
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
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_route_get_health(void) {
    gv_test_remove_db(TEST_DB);
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
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_route_get_stats(void) {
    gv_test_remove_db(TEST_DB);
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
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_route_get_metrics(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_HttpRequest request = {
        .method = GV_HTTP_GET,
        .url = "/metrics",
        .query_string = NULL,
        .body = NULL,
        .body_length = 0,
        .content_type = NULL,
        .authorization = NULL
    };

    GV_HttpResponse *resp = rest_route(&ctx, &request);
    ASSERT(resp != NULL, "route metrics response");
    ASSERT(resp->status == GV_HTTP_200_OK, "route metrics status 200");
    ASSERT(strstr(resp->body, "gigavector_up") != NULL, "routed metrics body");

    rest_response_free(resp);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_request_is_mutation(void) {
    /* Reads */
    ASSERT(rest_request_is_mutation("/stats", GV_HTTP_GET) == 0, "GET is read");
    ASSERT(rest_request_is_mutation("/vectors/1", GV_HTTP_GET) == 0, "GET vector is read");
    ASSERT(rest_request_is_mutation("/search", GV_HTTP_POST) == 0, "POST /search is read");
    ASSERT(rest_request_is_mutation("/search/range", GV_HTTP_POST) == 0, "POST /search/range is read");
    ASSERT(rest_request_is_mutation("/search/batch", GV_HTTP_POST) == 0, "POST /search/batch is read");
    /* Mutations */
    ASSERT(rest_request_is_mutation("/vectors", GV_HTTP_POST) == 1, "POST /vectors is mutation");
    ASSERT(rest_request_is_mutation("/vectors/1", GV_HTTP_PUT) == 1, "PUT is mutation");
    ASSERT(rest_request_is_mutation("/vectors/1", GV_HTTP_DELETE) == 1, "DELETE is mutation");
    ASSERT(rest_request_is_mutation("/save", GV_HTTP_POST) == 1, "POST /save is mutation");
    ASSERT(rest_request_is_mutation("/compact", GV_HTTP_POST) == 1, "POST /compact is mutation");
    return 0;
}

static int test_read_only_mode(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");
    float v[] = {1.0f, 0.0f, 0.0f, 0.0f};
    { int _r = db_add_vector(db, v, TEST_DIM); (void)_r; }

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    scfg.read_only = 1;

    /* A read still works. */
    GV_HttpRequest read_req = {
        .method = GV_HTTP_GET, .url = "/stats", .query_string = NULL,
        .body = NULL, .body_length = 0, .content_type = NULL, .authorization = NULL
    };
    GV_HttpResponse *rr = rest_route(&ctx, &read_req);
    ASSERT(rr != NULL && rr->status == GV_HTTP_200_OK, "read allowed in read-only mode");
    rest_response_free(rr);

    /* A write is rejected with 403. */
    GV_HttpRequest write_req = {
        .method = GV_HTTP_POST, .url = "/vectors", .query_string = NULL,
        .body = "{\"data\":[1,0,0,0]}", .body_length = 18,
        .content_type = "application/json", .authorization = NULL
    };
    GV_HttpResponse *wr = rest_route(&ctx, &write_req);
    ASSERT(wr != NULL && wr->status == GV_HTTP_403_FORBIDDEN, "write rejected in read-only mode");
    ASSERT(strstr(wr->body, "read_only") != NULL, "403 body names read_only");
    rest_response_free(wr);

    /* POST /search is a read and still allowed. */
    GV_HttpRequest search_req = {
        .method = GV_HTTP_POST, .url = "/search", .query_string = NULL,
        .body = "{\"query\":[1,0,0,0],\"k\":1}", .body_length = 25,
        .content_type = "application/json", .authorization = NULL
    };
    GV_HttpResponse *sr = rest_route(&ctx, &search_req);
    ASSERT(sr != NULL && sr->status == GV_HTTP_200_OK, "search allowed in read-only mode");
    rest_response_free(sr);

    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

/* Extract the 32-hex txn_id value from a {"txn_id":"..."} JSON body. */
static int extract_txn_id(const char *body, char *out, size_t out_sz) {
    const char *p = strstr(body, "\"txn_id\"");
    if (!p) return -1;
    p = strchr(p, ':');
    if (!p) return -1;
    p = strchr(p, '"');
    if (!p) return -1;
    p++;                                  /* first char of value */
    const char *e = strchr(p, '"');
    if (!e || (size_t)(e - p) >= out_sz) return -1;
    memcpy(out, p, (size_t)(e - p));
    out[e - p] = '\0';
    return 0;
}

static int test_txn_endpoints(void) {
    gv_test_remove_db(TEST_DB);
    GV_Database *db = db_open(TEST_DB, TEST_DIM, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "database creation");

    GV_ServerConfig scfg;
    GV_HandlerContext ctx = create_test_ctx(db, &scfg);
    GV_TxnRegistry *reg = txn_registry_create();
    ASSERT(reg != NULL, "create registry");
    ctx.txn_registry = reg;

    /* BEGIN -> 200 + txn_id */
    GV_HttpRequest begin_req = {
        .method = GV_HTTP_POST, .url = "/txn/begin", .query_string = NULL,
        .body = NULL, .body_length = 0, .content_type = NULL, .authorization = NULL
    };
    GV_HttpResponse *br = rest_route(&ctx, &begin_req);
    ASSERT(br != NULL && br->status == GV_HTTP_200_OK, "begin -> 200");
    char tid[64] = {0};
    ASSERT(extract_txn_id(br->body, tid, sizeof(tid)) == 0 && strlen(tid) == 32, "begin returns a txn_id");
    rest_response_free(br);
    ASSERT(txn_registry_count(reg) == 1, "one open txn after begin");

    /* COMMIT with that id -> 200 */
    char cbody[128];
    snprintf(cbody, sizeof(cbody), "{\"txn_id\":\"%s\"}", tid);
    GV_HttpRequest commit_req = {
        .method = GV_HTTP_POST, .url = "/txn/commit", .query_string = NULL,
        .body = cbody, .body_length = strlen(cbody),
        .content_type = "application/json", .authorization = NULL
    };
    GV_HttpResponse *cr = rest_route(&ctx, &commit_req);
    ASSERT(cr != NULL && cr->status == GV_HTTP_200_OK, "commit -> 200");
    ASSERT(strstr(cr->body, "committed") != NULL, "commit body says committed");
    rest_response_free(cr);
    ASSERT(txn_registry_count(reg) == 0, "no open txns after commit");

    /* COMMIT unknown id -> 404 */
    const char *unknown = "{\"txn_id\":\"00000000000000000000000000000000\"}";
    GV_HttpRequest unk_req = {
        .method = GV_HTTP_POST, .url = "/txn/commit", .query_string = NULL,
        .body = unknown, .body_length = strlen(unknown),
        .content_type = "application/json", .authorization = NULL
    };
    GV_HttpResponse *ur = rest_route(&ctx, &unk_req);
    ASSERT(ur != NULL && ur->status == GV_HTTP_404_NOT_FOUND, "commit unknown -> 404");
    rest_response_free(ur);

    /* COMMIT with no body -> 400 */
    GV_HttpRequest nobody = {
        .method = GV_HTTP_POST, .url = "/txn/commit", .query_string = NULL,
        .body = NULL, .body_length = 0, .content_type = NULL, .authorization = NULL
    };
    GV_HttpResponse *nr = rest_route(&ctx, &nobody);
    ASSERT(nr != NULL && nr->status == GV_HTTP_400_BAD_REQUEST, "commit without txn_id -> 400");
    rest_response_free(nr);

    /* ROLLBACK a fresh txn -> 200 */
    GV_HttpResponse *br2 = rest_route(&ctx, &begin_req);
    char tid2[64] = {0};
    ASSERT(extract_txn_id(br2->body, tid2, sizeof(tid2)) == 0, "second begin txn_id");
    rest_response_free(br2);
    char rbody[128];
    snprintf(rbody, sizeof(rbody), "{\"txn_id\":\"%s\"}", tid2);
    GV_HttpRequest rb_req = {
        .method = GV_HTTP_POST, .url = "/txn/rollback", .query_string = NULL,
        .body = rbody, .body_length = strlen(rbody),
        .content_type = "application/json", .authorization = NULL
    };
    GV_HttpResponse *rbr = rest_route(&ctx, &rb_req);
    ASSERT(rbr != NULL && rbr->status == GV_HTTP_200_OK, "rollback -> 200");
    rest_response_free(rbr);
    ASSERT(txn_registry_count(reg) == 0, "no open txns after rollback");

    /* With no registry wired, txn endpoints report 503. */
    GV_HandlerContext ctx_no = create_test_ctx(db, &scfg);  /* txn_registry defaults to NULL */
    ctx_no.txn_registry = NULL;
    GV_HttpResponse *sr = rest_route(&ctx_no, &begin_req);
    ASSERT(sr != NULL && sr->status == GV_HTTP_503_SERVICE_UNAVAILABLE, "no registry -> 503");
    rest_response_free(sr);

    txn_registry_destroy(reg);
    db_close(db);
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_route_not_found(void) {
    gv_test_remove_db(TEST_DB);
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
    gv_test_remove_db(TEST_DB);
    return 0;
}

static int test_route_method_mismatch(void) {
    gv_test_remove_db(TEST_DB);
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
    gv_test_remove_db(TEST_DB);
    return 0;
}

int main(void) {
    if (init_test_db_path() != 0) {
        fprintf(stderr, "FAIL: temp db path\n");
        return 1;
    }
    int failed = 0;
    int passed = 0;

    gv_test_remove_db(TEST_DB);

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
        {"test_handle_metrics",            test_handle_metrics},
        {"test_route_get_health",          test_route_get_health},
        {"test_route_get_stats",           test_route_get_stats},
        {"test_route_get_metrics",         test_route_get_metrics},
        {"test_request_is_mutation",       test_request_is_mutation},
        {"test_read_only_mode",            test_read_only_mode},
        {"test_txn_endpoints",             test_txn_endpoints},
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
    gv_test_remove_db(TEST_DB);
    return failed > 0 ? 1 : 0;
}
