/**
 * @file fuzz_rest_decode.c
 * @brief libFuzzer harness for the REST request decode/routing path.
 *
 * rest_route() is the top-level dispatcher the HTTP server hands every parsed
 * request to. This builds a GV_HttpRequest from RAW fuzz bytes -- method, URL
 * (incl. query string + path params) and body all derived from the input -- and
 * drives rest_route() against a real database, exercising URL parsing, path/
 * query-param extraction and JSON body parsing on arbitrary input. The response
 * (owned) is freed each time.
 *
 * The DB + handler context are a persistent singleton (set up once, reused) for
 * speed; they stay reachable for the whole run. libFuzzer exits via _Exit()
 * before atexit teardown, so LSan sees them as leaked -- suppressed in
 * tests/fuzz/lsan.supp (leak:fuzz_setup_once), the same shutdown-ordering
 * artifact documented there and used by fuzz_grpc_dispatch.
 *
 * Build: make fuzz  (requires clang)
 * Run:   build/fuzz/fuzz_rest_decode tests/fuzz/corpus/rest -runs=10000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "api/rest_handlers.h"
#include "api/server.h"
#include "storage/database.h"
#include "../test_tmp.h"

#define FUZZ_MAX_INPUT (64 * 1024)
#define FUZZ_DIM 4

static GV_Database *g_db = NULL;
static GV_HandlerContext g_ctx;
static GV_ServerConfig g_cfg;
static int g_ready = 0;
static char g_db_path[256];

static void fuzz_setup_once(void) {
    if (g_ready) return;

    if (gv_test_make_temp_path(g_db_path, sizeof(g_db_path), "gv_fuzz_rest", ".gv") != 0) {
        return;
    }
    remove(g_db_path);

    g_db = db_open(g_db_path, FUZZ_DIM, GV_INDEX_TYPE_FLAT);
    if (!g_db) return;

    server_config_init(&g_cfg);

    memset(&g_ctx, 0, sizeof(g_ctx));
    g_ctx.db = g_db;
    g_ctx.config = &g_cfg;

    g_ready = 1;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;

    fuzz_setup_once();
    if (!g_ready) return 0;

    /* Byte 0 selects the HTTP method; the remainder splits on the first '\n'
     * into a NUL-terminated URL (path + optional ?query) and the body. */
    GV_HttpMethod methods[6] = {
        GV_HTTP_GET, GV_HTTP_POST, GV_HTTP_PUT,
        GV_HTTP_DELETE, GV_HTTP_OPTIONS, GV_HTTP_HEAD
    };

    const uint8_t *rest = data + 1;
    size_t rest_len = size - 1;

    const uint8_t *nl = (const uint8_t *)memchr(rest, '\n', rest_len);
    size_t url_len = nl ? (size_t)(nl - rest) : rest_len;
    size_t body_len = nl ? rest_len - url_len - 1 : 0;
    const uint8_t *body_src = nl ? nl + 1 : NULL;

    char *url = (char *)malloc(url_len + 1);
    if (!url) return 0;
    memcpy(url, rest, url_len);
    url[url_len] = '\0';

    char *body = NULL;
    if (body_len) {
        body = (char *)malloc(body_len + 1);
        if (!body) { free(url); return 0; }
        memcpy(body, body_src, body_len);
        body[body_len] = '\0';
    }

    /* Split a query string off the URL path at the first '?' so query-param
     * parsing is exercised the way the real server populates the request. */
    char *qmark = strchr(url, '?');
    const char *query_string = NULL;
    if (qmark) {
        *qmark = '\0';
        query_string = qmark + 1;
    }

    GV_HttpRequest req;
    memset(&req, 0, sizeof(req));
    req.method = methods[data[0] % 6];
    req.url = url;
    req.query_string = query_string;
    req.body = body;
    req.body_length = body_len;
    req.content_type = "application/json";
    req.authorization = NULL;

    GV_HttpResponse *resp = rest_route(&g_ctx, &req);
    if (resp) {
        rest_response_free(resp);
    }

    free(url);
    free(body);
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
