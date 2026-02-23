#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_tls.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_tls_config_init(void) {
    GV_TLSConfig config;
    memset(&config, 0xFF, sizeof(config));
    gv_tls_config_init(&config);

    ASSERT(config.cert_file == NULL, "default cert_file should be NULL");
    ASSERT(config.key_file == NULL, "default key_file should be NULL");
    ASSERT(config.ca_file == NULL, "default ca_file should be NULL");
    ASSERT(config.min_version == GV_TLS_1_2, "default min_version should be TLS 1.2");
    ASSERT(config.cipher_list == NULL, "default cipher_list should be NULL");
    ASSERT(config.verify_client == 0, "default verify_client should be 0");
    return 0;
}

static int test_tls_config_init_idempotent(void) {
    GV_TLSConfig c1, c2;
    memset(&c1, 0, sizeof(c1));
    memset(&c2, 0, sizeof(c2));
    gv_tls_config_init(&c1);
    gv_tls_config_init(&c2);
    ASSERT(c1.min_version == c2.min_version, "min_version should match on repeated init");
    ASSERT(c1.verify_client == c2.verify_client, "verify_client should match on repeated init");
    ASSERT(c1.cert_file == c2.cert_file, "cert_file should match on repeated init");
    return 0;
}

static int test_tls_is_available(void) {
    int avail = gv_tls_is_available();
    ASSERT(avail == 0 || avail == 1, "gv_tls_is_available should return 0 or 1");
    return 0;
}

static int test_tls_create_empty_config(void) {
    /* Creating TLS context with no cert/key files should fail gracefully */
    GV_TLSConfig config;
    gv_tls_config_init(&config);
    /* cert_file and key_file are NULL */

    GV_TLSContext *ctx = gv_tls_create(&config);
    ASSERT(ctx == NULL, "gv_tls_create with no cert/key should return NULL");
    return 0;
}

static int test_tls_create_nonexistent_files(void) {
    GV_TLSConfig config;
    gv_tls_config_init(&config);
    config.cert_file = "/tmp/nonexistent_cert_98765.pem";
    config.key_file = "/tmp/nonexistent_key_98765.pem";

    GV_TLSContext *ctx = gv_tls_create(&config);
    ASSERT(ctx == NULL, "gv_tls_create with nonexistent files should return NULL");
    return 0;
}

static int test_tls_create_null_config(void) {
    GV_TLSContext *ctx = gv_tls_create(NULL);
    ASSERT(ctx == NULL, "gv_tls_create(NULL) should return NULL");
    return 0;
}

static int test_tls_destroy_null(void) {
    /* Should not crash */
    gv_tls_destroy(NULL);
    return 0;
}

static int test_tls_version_string_null(void) {
    const char *ver = gv_tls_version_string(NULL);
    /* Should either return NULL or a safe default string, but not crash */
    (void)ver;
    return 0;
}

static int test_tls_cert_days_remaining_null(void) {
    int days = gv_tls_cert_days_remaining(NULL);
    ASSERT(days <= 0, "cert_days_remaining(NULL) should return <= 0 (error or no cert)");
    return 0;
}

static int test_tls_accept_null(void) {
    void *conn = NULL;
    int rc = gv_tls_accept(NULL, -1, &conn);
    ASSERT(rc == -1 || rc < 0, "tls_accept with NULL context should fail");
    ASSERT(conn == NULL, "conn should remain NULL on failure");
    return 0;
}

static int test_tls_read_null(void) {
    char buf[64];
    int rc = gv_tls_read(NULL, buf, sizeof(buf));
    ASSERT(rc <= 0, "tls_read(NULL) should return <= 0");
    return 0;
}

static int test_tls_write_null(void) {
    const char *data = "test";
    int rc = gv_tls_write(NULL, data, 4);
    ASSERT(rc <= 0, "tls_write(NULL) should return <= 0");
    return 0;
}

static int test_tls_close_conn_null(void) {
    /* Should not crash */
    gv_tls_close_conn(NULL);
    return 0;
}

static int test_tls_get_peer_cn_null(void) {
    char buf[256];
    memset(buf, 0, sizeof(buf));
    int rc = gv_tls_get_peer_cn(NULL, buf, sizeof(buf));
    ASSERT(rc == -1 || rc < 0, "get_peer_cn(NULL) should fail");
    return 0;
}

static int test_tls_version_enum_values(void) {
    ASSERT(GV_TLS_1_2 == 0, "GV_TLS_1_2 should be 0");
    ASSERT(GV_TLS_1_3 == 1, "GV_TLS_1_3 should be 1");
    return 0;
}

static int test_tls_config_tls13(void) {
    GV_TLSConfig config;
    gv_tls_config_init(&config);
    config.min_version = GV_TLS_1_3;
    ASSERT(config.min_version == GV_TLS_1_3, "should be able to set min_version to TLS 1.3");

    /* Creating without cert files should still fail */
    GV_TLSContext *ctx = gv_tls_create(&config);
    ASSERT(ctx == NULL, "gv_tls_create with TLS 1.3 but no cert should return NULL");
    return 0;
}

static int test_tls_config_mutual_tls(void) {
    GV_TLSConfig config;
    gv_tls_config_init(&config);
    config.verify_client = 1;
    config.ca_file = "/tmp/nonexistent_ca_98765.pem";
    ASSERT(config.verify_client == 1, "verify_client should be settable to 1");

    /* Without valid cert/key, creation should still fail */
    GV_TLSContext *ctx = gv_tls_create(&config);
    ASSERT(ctx == NULL, "gv_tls_create with mTLS but no cert should return NULL");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing tls_config_init...", test_tls_config_init},
        {"Testing tls_config_init_idempotent...", test_tls_config_init_idempotent},
        {"Testing tls_is_available...", test_tls_is_available},
        {"Testing tls_create_empty_config...", test_tls_create_empty_config},
        {"Testing tls_create_nonexistent_files...", test_tls_create_nonexistent_files},
        {"Testing tls_create_null_config...", test_tls_create_null_config},
        {"Testing tls_destroy_null...", test_tls_destroy_null},
        {"Testing tls_version_string_null...", test_tls_version_string_null},
        {"Testing tls_cert_days_remaining_null...", test_tls_cert_days_remaining_null},
        {"Testing tls_accept_null...", test_tls_accept_null},
        {"Testing tls_read_null...", test_tls_read_null},
        {"Testing tls_write_null...", test_tls_write_null},
        {"Testing tls_close_conn_null...", test_tls_close_conn_null},
        {"Testing tls_get_peer_cn_null...", test_tls_get_peer_cn_null},
        {"Testing tls_version_enum_values...", test_tls_version_enum_values},
        {"Testing tls_config_tls13...", test_tls_config_tls13},
        {"Testing tls_config_mutual_tls...", test_tls_config_mutual_tls},
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
