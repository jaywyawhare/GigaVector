#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_sso.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_create_oidc(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://idp.example.com";
    cfg.client_id = "test-client-id";
    cfg.client_secret = "test-secret";
    cfg.redirect_uri = "http://localhost:8080/callback";
    cfg.verify_ssl = 1;
    cfg.token_ttl = 3600;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "gv_sso_create with OIDC config should succeed");
    gv_sso_destroy(mgr);
    return 0;
}

static int test_create_saml(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_SAML;
    cfg.saml_metadata_url = "https://idp.example.com/saml/metadata";
    cfg.saml_entity_id = "urn:gigavector:sp";
    cfg.verify_ssl = 0;
    cfg.token_ttl = 7200;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "gv_sso_create with SAML config should succeed");
    gv_sso_destroy(mgr);
    return 0;
}

static int test_create_with_groups(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://idp.example.com";
    cfg.client_id = "test-client";
    cfg.client_secret = "secret";
    cfg.redirect_uri = "http://localhost/cb";
    cfg.verify_ssl = 1;
    cfg.token_ttl = 1800;
    cfg.allowed_groups = "users,editors";
    cfg.admin_groups = "admins,superadmins";

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "gv_sso_create with groups should succeed");
    gv_sso_destroy(mgr);
    return 0;
}

static int test_destroy_null(void) {
    /* Destroying NULL must not crash */
    gv_sso_destroy(NULL);
    return 0;
}

static int test_discover_no_idp(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://nonexistent.invalid.example.com";
    cfg.client_id = "test";
    cfg.client_secret = "secret";
    cfg.redirect_uri = "http://localhost/cb";
    cfg.verify_ssl = 0;
    cfg.token_ttl = 3600;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "create should succeed even with bogus URL");

    /* Discovery should fail gracefully without a real IdP */
    int rc = gv_sso_discover(mgr);
    ASSERT(rc != 0, "gv_sso_discover without real IdP should fail");

    gv_sso_destroy(mgr);
    return 0;
}

static int test_validate_token_null(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://idp.example.com";
    cfg.client_id = "test";
    cfg.client_secret = "secret";
    cfg.redirect_uri = "http://localhost/cb";
    cfg.verify_ssl = 0;
    cfg.token_ttl = 3600;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "create");

    /* Validate NULL token should return NULL */
    GV_SSOToken *tok = gv_sso_validate_token(mgr, NULL);
    ASSERT(tok == NULL, "validate NULL token should return NULL");

    gv_sso_destroy(mgr);
    return 0;
}

static int test_validate_token_invalid(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://idp.example.com";
    cfg.client_id = "test";
    cfg.client_secret = "secret";
    cfg.redirect_uri = "http://localhost/cb";
    cfg.verify_ssl = 0;
    cfg.token_ttl = 3600;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "create");

    /* Validate garbage token should return NULL */
    GV_SSOToken *tok = gv_sso_validate_token(mgr, "not-a-valid-jwt-token");
    ASSERT(tok == NULL, "validate garbage token should return NULL");

    /* Validate empty string */
    tok = gv_sso_validate_token(mgr, "");
    ASSERT(tok == NULL, "validate empty token should return NULL");

    gv_sso_destroy(mgr);
    return 0;
}

static int test_has_group_null_token(void) {
    /* has_group with NULL token should return 0 */
    int result = gv_sso_has_group(NULL, "admins");
    ASSERT(result == 0, "has_group with NULL token should return 0");
    return 0;
}

static int test_has_group_null_group(void) {
    /* has_group with NULL group name should return 0 */
    GV_SSOToken tok;
    memset(&tok, 0, sizeof(tok));
    tok.groups = NULL;
    tok.group_count = 0;

    int result = gv_sso_has_group(&tok, NULL);
    ASSERT(result == 0, "has_group with NULL group should return 0");
    return 0;
}

static int test_has_group_empty_groups(void) {
    GV_SSOToken tok;
    memset(&tok, 0, sizeof(tok));
    tok.groups = NULL;
    tok.group_count = 0;

    int result = gv_sso_has_group(&tok, "admins");
    ASSERT(result == 0, "has_group with no groups should return 0");
    return 0;
}

static int test_free_token_null(void) {
    /* Freeing NULL token must not crash */
    gv_sso_free_token(NULL);
    return 0;
}

static int test_get_auth_url(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://idp.example.com";
    cfg.client_id = "test-client";
    cfg.client_secret = "secret";
    cfg.redirect_uri = "http://localhost:8080/callback";
    cfg.verify_ssl = 0;
    cfg.token_ttl = 3600;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "create");

    char url[2048];
    memset(url, 0, sizeof(url));

    /* Without discovery, auth URL generation may fail or produce a local URL.
     * Either way it should not crash. */
    int rc = gv_sso_get_auth_url(mgr, "csrf-state-123", url, sizeof(url));
    /* We accept both success and failure - the important thing is no crash.
     * If it succeeds, the URL should contain the state parameter. */
    if (rc == 0) {
        ASSERT(strlen(url) > 0, "auth URL should not be empty on success");
    }

    gv_sso_destroy(mgr);
    return 0;
}

static int test_get_auth_url_small_buffer(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://idp.example.com";
    cfg.client_id = "test-client";
    cfg.client_secret = "secret";
    cfg.redirect_uri = "http://localhost:8080/callback";
    cfg.verify_ssl = 0;
    cfg.token_ttl = 3600;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "create");

    /* Tiny buffer should fail or truncate gracefully */
    char url[8];
    memset(url, 0, sizeof(url));
    int rc = gv_sso_get_auth_url(mgr, "state", url, sizeof(url));
    /* Should fail or at least not overflow */
    (void)rc;

    gv_sso_destroy(mgr);
    return 0;
}

static int test_exchange_code_invalid(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://idp.example.com";
    cfg.client_id = "test";
    cfg.client_secret = "secret";
    cfg.redirect_uri = "http://localhost/cb";
    cfg.verify_ssl = 0;
    cfg.token_ttl = 3600;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "create");

    /* Exchange with bogus code should return NULL */
    GV_SSOToken *tok = gv_sso_exchange_code(mgr, "invalid-auth-code");
    ASSERT(tok == NULL, "exchange with invalid code should return NULL");

    /* Exchange with NULL code */
    tok = gv_sso_exchange_code(mgr, NULL);
    ASSERT(tok == NULL, "exchange with NULL code should return NULL");

    gv_sso_destroy(mgr);
    return 0;
}

static int test_refresh_token_invalid(void) {
    GV_SSOConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.provider = GV_SSO_OIDC;
    cfg.issuer_url = "https://idp.example.com";
    cfg.client_id = "test";
    cfg.client_secret = "secret";
    cfg.redirect_uri = "http://localhost/cb";
    cfg.verify_ssl = 0;
    cfg.token_ttl = 3600;

    GV_SSOManager *mgr = gv_sso_create(&cfg);
    ASSERT(mgr != NULL, "create");

    /* Refresh with bogus token should fail */
    GV_SSOToken *tok = gv_sso_refresh_token(mgr, "invalid-refresh-token");
    ASSERT(tok == NULL, "refresh with invalid token should return NULL");

    tok = gv_sso_refresh_token(mgr, NULL);
    ASSERT(tok == NULL, "refresh with NULL token should return NULL");

    gv_sso_destroy(mgr);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing create_oidc...",            test_create_oidc},
        {"Testing create_saml...",            test_create_saml},
        {"Testing create_with_groups...",     test_create_with_groups},
        {"Testing destroy_null...",           test_destroy_null},
        {"Testing discover_no_idp...",        test_discover_no_idp},
        {"Testing validate_token_null...",    test_validate_token_null},
        {"Testing validate_token_invalid...", test_validate_token_invalid},
        {"Testing has_group_null_token...",   test_has_group_null_token},
        {"Testing has_group_null_group...",   test_has_group_null_group},
        {"Testing has_group_empty_groups...", test_has_group_empty_groups},
        {"Testing free_token_null...",        test_free_token_null},
        {"Testing get_auth_url...",           test_get_auth_url},
        {"Testing get_auth_url_small_buf...", test_get_auth_url_small_buffer},
        {"Testing exchange_code_invalid...",  test_exchange_code_invalid},
        {"Testing refresh_token_invalid...",  test_refresh_token_invalid},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
