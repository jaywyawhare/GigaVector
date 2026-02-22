#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_auth.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ── Test: config init ─────────────────────────────────────────────────── */
static int test_config_init(void) {
    GV_AuthConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_auth_config_init(&cfg);
    ASSERT(cfg.type == GV_AUTH_NONE, "default auth type should be NONE");
    return 0;
}

/* ── Test: create/destroy with NULL config ─────────────────────────────── */
static int test_create_destroy_null(void) {
    GV_AuthManager *mgr = gv_auth_create(NULL);
    ASSERT(mgr != NULL, "auth manager creation with NULL config");
    gv_auth_destroy(mgr);
    /* double-free safety */
    gv_auth_destroy(NULL);
    return 0;
}

/* ── Test: create with API key config ──────────────────────────────────── */
static int test_create_api_key_config(void) {
    GV_AuthConfig cfg;
    gv_auth_config_init(&cfg);
    cfg.type = GV_AUTH_API_KEY;

    GV_AuthManager *mgr = gv_auth_create(&cfg);
    ASSERT(mgr != NULL, "auth manager creation with API key config");
    gv_auth_destroy(mgr);
    return 0;
}

/* ── Test: generate and verify API key ─────────────────────────────────── */
static int test_generate_verify_api_key(void) {
    GV_AuthConfig cfg;
    gv_auth_config_init(&cfg);
    cfg.type = GV_AUTH_API_KEY;

    GV_AuthManager *mgr = gv_auth_create(&cfg);
    ASSERT(mgr != NULL, "auth manager creation");

    char key_out[128] = {0};
    char key_id_out[64] = {0};
    int rc = gv_auth_generate_api_key(mgr, "test key", 0, key_out, key_id_out);
    ASSERT(rc == 0, "generate API key");
    ASSERT(strlen(key_out) > 0, "generated key should be non-empty");
    ASSERT(strlen(key_id_out) > 0, "generated key_id should be non-empty");

    /* Verify the generated key */
    GV_Identity identity;
    memset(&identity, 0, sizeof(identity));
    GV_AuthResult result = gv_auth_verify_api_key(mgr, key_out, &identity);
    ASSERT(result == GV_AUTH_SUCCESS, "verify generated API key should succeed");
    gv_auth_free_identity(&identity);

    gv_auth_destroy(mgr);
    return 0;
}

/* ── Test: revoke API key ──────────────────────────────────────────────── */
static int test_revoke_api_key(void) {
    GV_AuthConfig cfg;
    gv_auth_config_init(&cfg);
    cfg.type = GV_AUTH_API_KEY;

    GV_AuthManager *mgr = gv_auth_create(&cfg);
    ASSERT(mgr != NULL, "auth manager creation");

    char key_out[128] = {0};
    char key_id_out[64] = {0};
    int rc = gv_auth_generate_api_key(mgr, "revoke-test", 0, key_out, key_id_out);
    ASSERT(rc == 0, "generate API key for revocation");

    rc = gv_auth_revoke_api_key(mgr, key_id_out);
    ASSERT(rc == 0, "revoke API key");

    /* Verify revoked key is rejected */
    GV_Identity identity;
    memset(&identity, 0, sizeof(identity));
    GV_AuthResult result = gv_auth_verify_api_key(mgr, key_out, &identity);
    ASSERT(result != GV_AUTH_SUCCESS, "revoked key should fail verification");
    gv_auth_free_identity(&identity);

    gv_auth_destroy(mgr);
    return 0;
}

/* ── Test: list API keys ───────────────────────────────────────────────── */
static int test_list_api_keys(void) {
    GV_AuthConfig cfg;
    gv_auth_config_init(&cfg);
    cfg.type = GV_AUTH_API_KEY;

    GV_AuthManager *mgr = gv_auth_create(&cfg);
    ASSERT(mgr != NULL, "auth manager creation");

    char key_out[128], key_id_out[64];
    gv_auth_generate_api_key(mgr, "key-a", 0, key_out, key_id_out);
    gv_auth_generate_api_key(mgr, "key-b", 0, key_out, key_id_out);

    GV_APIKey *keys = NULL;
    size_t count = 0;
    int rc = gv_auth_list_api_keys(mgr, &keys, &count);
    ASSERT(rc == 0, "list API keys");
    ASSERT(count >= 2, "should have at least 2 keys");

    gv_auth_free_api_keys(keys, count);
    gv_auth_destroy(mgr);
    return 0;
}

/* ── Test: SHA-256 and hex conversion ──────────────────────────────────── */
static int test_sha256_and_hex(void) {
    const char *data = "hello";
    unsigned char hash[32];
    int rc = gv_auth_sha256(data, strlen(data), hash);
    ASSERT(rc == 0, "sha256 computation");

    char hex[65];
    gv_auth_to_hex(hash, 32, hex);
    ASSERT(strlen(hex) == 64, "hex output should be 64 characters");

    /* SHA-256("hello") is well-known */
    ASSERT(strncmp(hex, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", 64) == 0,
           "sha256 of 'hello' should match known value");

    return 0;
}

/* ── Test: auth result strings ─────────────────────────────────────────── */
static int test_auth_result_string(void) {
    const char *s;

    s = gv_auth_result_string(GV_AUTH_SUCCESS);
    ASSERT(s != NULL, "result string for SUCCESS should not be NULL");

    s = gv_auth_result_string(GV_AUTH_INVALID_KEY);
    ASSERT(s != NULL, "result string for INVALID_KEY should not be NULL");

    s = gv_auth_result_string(GV_AUTH_EXPIRED);
    ASSERT(s != NULL, "result string for EXPIRED should not be NULL");

    s = gv_auth_result_string(GV_AUTH_MISSING);
    ASSERT(s != NULL, "result string for MISSING should not be NULL");

    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config init...",              test_config_init},
        {"Testing create/destroy NULL...",      test_create_destroy_null},
        {"Testing create API key config...",    test_create_api_key_config},
        {"Testing generate/verify API key...",  test_generate_verify_api_key},
        {"Testing revoke API key...",           test_revoke_api_key},
        {"Testing list API keys...",            test_list_api_keys},
        {"Testing SHA-256 and hex...",          test_sha256_and_hex},
        {"Testing auth result strings...",      test_auth_result_string},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
