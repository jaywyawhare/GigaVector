/**
 * test_auth.c — fail-closed REST auth policy tests.
 *
 * LIMITATION / WHY THIS SHAPE:
 *   The real decision function `check_auth` (src/api/server.c) is `static` AND
 *   compiled only under `#ifdef HAVE_MICROHTTPD`. This build does not define
 *   HAVE_MICROHTTPD (grep the Makefile: no -DHAVE_MICROHTTPD, no -lmicrohttpd),
 *   so neither `check_auth` nor its helper `is_liveness_request` are present in
 *   libGigaVector.so — they cannot be linked against or reached without a live
 *   libmicrohttpd HTTP server. server.h exposes no public auth-policy helper
 *   (only server_confine_save_path, which is path confinement, not auth).
 *
 *   Per the task's fallback instruction, this test therefore pins the auth
 *   *policy contract* at the config level. It drives the real public
 *   GV_ServerConfig (via server_config_init) and evaluates a policy oracle that
 *   mirrors check_auth()'s decision table BYTE-FOR-BYTE (see src/api/server.c
 *   check_auth + is_liveness_request). If server.c's policy changes, this
 *   oracle must be updated in lockstep — it is a specification test, not a
 *   direct call. The point is to guarantee the fail-closed defaults in
 *   GV_ServerConfig stay wired the way the auth code assumes.
 */
#include <stdio.h>
#include <string.h>

#include "api/server.h"

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); return -1; } \
} while (0)

/* Mirror of is_liveness_request() in src/api/server.c (kept in lockstep). */
static int policy_is_liveness(const char *url, const char *method) {
    if (!url || !method) return 0;               /* unknown -> not liveness */
    if (strcmp(method, "OPTIONS") == 0) return 1; /* CORS preflight */
    if (strcmp(method, "GET") == 0 && strcmp(url, "/health") == 0) return 1;
    return 0;
}

/*
 * Mirror of check_auth() in src/api/server.c. `have_key` models a matching
 * X-API-Key/Bearer credential being presented (1) or not (0); `key_present`
 * whether config.api_key is configured. Returns 1=allow, 0=deny.
 */
static int policy_check_auth(const GV_ServerConfig *cfg, int credential_matches,
                             const char *url, const char *method) {
    if (!cfg->api_key) {
        if (cfg->allow_unauthenticated) return 1;
        if (policy_is_liveness(url, method)) return 1;
        return 0; /* fail closed */
    }
    /* api_key configured: require a matching credential regardless of opt-in. */
    return credential_matches ? 1 : 0;
}

/* (a) No api_key configured (default): non-liveness denied, GET /health allowed. */
static int test_no_key_fail_closed(void) {
    GV_ServerConfig cfg;
    server_config_init(&cfg);
    ASSERT(cfg.api_key == NULL, "default api_key NULL");
    ASSERT(cfg.allow_unauthenticated == 0, "default fail-closed");

    /* Data/admin endpoints denied without a credential. */
    ASSERT(policy_check_auth(&cfg, 0, "/vectors/1", "GET") == 0, "GET /vectors/{id} denied");
    ASSERT(policy_check_auth(&cfg, 0, "/stats", "GET") == 0, "GET /stats denied");
    ASSERT(policy_check_auth(&cfg, 0, "/search", "POST") == 0, "POST /search denied");
    ASSERT(policy_check_auth(&cfg, 0, "/vectors", "POST") == 0, "POST /vectors denied");

    /* Liveness allowlist stays reachable. */
    ASSERT(policy_check_auth(&cfg, 0, "/health", "GET") == 1, "GET /health allowed");
    ASSERT(policy_check_auth(&cfg, 0, "/anything", "OPTIONS") == 1, "CORS preflight allowed");

    /* Prefix/suffix smuggling must NOT be treated as liveness. */
    ASSERT(policy_check_auth(&cfg, 0, "/health/../vectors/1", "GET") == 0, "no path smuggling");
    ASSERT(policy_check_auth(&cfg, 0, "/healthz", "GET") == 0, "suffix not liveness");
    ASSERT(policy_check_auth(&cfg, 0, "/health", "POST") == 0, "POST /health not liveness");
    return 0;
}

/* (b) allow_unauthenticated set -> everything allowed even without a key. */
static int test_allow_unauthenticated(void) {
    GV_ServerConfig cfg;
    server_config_init(&cfg);
    cfg.allow_unauthenticated = 1;

    ASSERT(policy_check_auth(&cfg, 0, "/vectors/1", "GET") == 1, "opt-in allows data GET");
    ASSERT(policy_check_auth(&cfg, 0, "/search", "POST") == 1, "opt-in allows POST");
    ASSERT(policy_check_auth(&cfg, 0, "/health", "GET") == 1, "opt-in allows liveness");
    return 0;
}

/* (c) Wrong key -> denied; (d) correct key -> allowed. When api_key is set,
 * the opt-in flag is irrelevant and every endpoint requires a match. */
static int test_key_configured(void) {
    GV_ServerConfig cfg;
    server_config_init(&cfg);
    cfg.api_key = "s3cret";

    /* (c) wrong / missing credential denied on all endpoints. */
    ASSERT(policy_check_auth(&cfg, 0, "/vectors/1", "GET") == 0, "wrong key denied (data)");
    ASSERT(policy_check_auth(&cfg, 0, "/search", "POST") == 0, "wrong key denied (search)");
    ASSERT(policy_check_auth(&cfg, 0, "/health", "GET") == 0, "key set: liveness also needs key");

    /* (d) correct credential allowed. */
    ASSERT(policy_check_auth(&cfg, 1, "/vectors/1", "GET") == 1, "correct key allowed (data)");
    ASSERT(policy_check_auth(&cfg, 1, "/search", "POST") == 1, "correct key allowed (search)");

    /* opt-in must not weaken a configured key. */
    cfg.allow_unauthenticated = 1;
    ASSERT(policy_check_auth(&cfg, 0, "/vectors/1", "GET") == 0,
           "opt-in ignored when key configured");
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_no_key_fail_closed();
    rc |= test_allow_unauthenticated();
    rc |= test_key_configured();
    if (rc == 0) printf("All auth-policy tests PASSED.\n");
    return rc != 0;
}
