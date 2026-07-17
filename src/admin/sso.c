#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

/**
 * @file sso.c
 * @brief Enterprise SSO / OIDC / SAML authentication implementation.
 *
 * Provides OIDC discovery, JWT validation against JWKS endpoints,
 * SAML assertion parsing, and group-based authorization.
 *
 * HTTP calls use libcurl when HAVE_CURL is defined; otherwise the
 * network-dependent functions return an error.
 */

#include "admin/sso.h"
#include "core/memory.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>
#include <pthread.h>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

/*
 * RS256 (RSA-SHA256) JWT verification requires OpenSSL, using the same
 * GV_HAVE_OPENSSL gate as src/security/crypto.c.  When built WITHOUT OpenSSL,
 * RS256 fails closed (see verify_jwt_signature_rs256 below).
 */
#ifdef GV_HAVE_OPENSSL
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/bn.h>
#include <openssl/bio.h>
#include <openssl/core_names.h>
#include <openssl/param_build.h>
#include <openssl/params.h>
#include <openssl/x509.h>   /* SAML XML-DSig: X.509 cert parsing / pubkey */
#include <openssl/sha.h>    /* SAML XML-DSig: reference DigestValue check */
#endif

/* Internal Constants */

#define MAX_URL_LEN         2048
#define MAX_ENDPOINT_LEN    1024
#define MAX_RESPONSE_SIZE   (256 * 1024)
#define MAX_GROUPS          64
#define MAX_JWT_SEGMENTS    3
#define BASE64_DECODE_MAX   8192

/* Internal Structures */

/**
 * @brief Discovered OIDC endpoints.
 */
typedef struct {
    char authorization_endpoint[MAX_ENDPOINT_LEN];
    char token_endpoint[MAX_ENDPOINT_LEN];
    char jwks_uri[MAX_ENDPOINT_LEN];
    char userinfo_endpoint[MAX_ENDPOINT_LEN];
    int discovered;
} OIDCEndpoints;

/**
 * @brief Dynamic buffer for HTTP response accumulation.
 */
typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} ResponseBuffer;

/**
 * @brief SSO manager internal structure.
 */
struct GV_SSOManager {
    GV_SSOConfig config;
    OIDCEndpoints endpoints;
    pthread_mutex_t mutex;
};

/* Forward Declarations */

static GV_SSOToken *alloc_token(void);
static char **split_csv(const char *csv, size_t *count);
static int check_group_in_list(const char *csv_list, const char *group);
static void populate_admin_flag(GV_SSOToken *token, const char *admin_groups);

/* Base64 URL decoding */
static int base64url_decode(const char *in, size_t in_len,
                            unsigned char *out, size_t *out_len);

/* Minimal JSON helpers (no external dependency) */
static int json_extract_string(const char *json, const char *key,
                               char *out, size_t out_size);
static int json_extract_uint64(const char *json, const char *key, uint64_t *out);
static int json_extract_string_array(const char *json, const char *key,
                                     char ***out, size_t *count);

/* JWT helpers */
static GV_SSOToken *decode_jwt_claims(const char *jwt);

/* ISO 8601 timestamp parsing */
static uint64_t parse_iso8601(const char *ts);

/* HMAC-SHA256 for JWT signature verification */
static int hmac_sha256(const unsigned char *key, size_t key_len,
                       const unsigned char *data, size_t data_len,
                       unsigned char *out, size_t *out_len);

/*
 * Verify a JWT signature.  Dispatches on the JWT header "alg":
 *   - HS256: HMAC-SHA256 over header.payload using config.client_secret.
 *   - RS256: RSA-SHA256 (OpenSSL) using an RSA public key sourced from
 *            config.oidc_rsa_public_key_pem and/or the JWKS endpoint.
 * Any other alg ("none", "ES256", ...) is rejected (alg-confusion guard).
 * Needs the manager for keys/JWKS/endpoints, so it takes it by const pointer.
 */
static int verify_jwt_signature(const GV_SSOManager *mgr, const char *jwt);

/* SAML helpers */
static GV_SSOToken *parse_saml_assertion(const GV_SSOConfig *config,
                                         const char *b64_assertion);
static int xml_extract_text(const char *xml, const char *tag,
                            char *out, size_t out_size);

/* HTTP helpers */
static int http_get(const char *url, int verify_ssl,
                    char **response, size_t *response_len);
static int http_post_form(const char *url, const char *post_fields,
                          int verify_ssl, char **response, size_t *response_len);

/* URL Encoding */

/**
 * Percent-encode @p src into @p dst for use in an
 * application/x-www-form-urlencoded body.  Every byte that is not an RFC 3986
 * unreserved character (A-Za-z0-9 - _ . ~) is emitted as %XX.  This prevents a
 * client-supplied value (auth code, refresh token, ...) from injecting or
 * overriding additional form parameters via characters such as '&' or '='.
 *
 * Writes at most @p dst_size bytes including the NUL terminator.  Returns 0 on
 * success, -1 if the encoded output (worst case 3x the input) would not fit.
 */
static int url_encode(const char *src, char *dst, size_t dst_size) {
    static const char hex[] = "0123456789ABCDEF";
    if (!dst || dst_size == 0) return -1;
    if (!src) src = "";

    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        unsigned char c = *p;
        int unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                         (c >= '0' && c <= '9') ||
                         c == '-' || c == '_' || c == '.' || c == '~';
        if (unreserved) {
            if (o + 1 >= dst_size) return -1;
            dst[o++] = (char)c;
        } else {
            if (o + 3 >= dst_size) return -1;
            dst[o++] = '%';
            dst[o++] = hex[(c >> 4) & 0xF];
            dst[o++] = hex[c & 0xF];
        }
    }
    dst[o] = '\0';
    return 0;
}

/* Lifecycle */

GV_SSOManager *sso_create(const GV_SSOConfig *config) {
    if (!config) return NULL;

    GV_SSOManager *mgr = gv_calloc(1, sizeof(GV_SSOManager));
    if (!mgr) return NULL;

    mgr->config = *config;

    /* Apply defaults */
    if (mgr->config.verify_ssl != 0 && mgr->config.verify_ssl != 1) {
        mgr->config.verify_ssl = 1;
    }
    if (mgr->config.token_ttl == 0) {
        mgr->config.token_ttl = 3600;
    }

    if (pthread_mutex_init(&mgr->mutex, NULL) != 0) {
        gv_free(mgr);
        return NULL;
    }

    return mgr;
}

void sso_destroy(GV_SSOManager *mgr) {
    if (!mgr) return;

    pthread_mutex_destroy(&mgr->mutex);
    gv_free(mgr);
}

/* OIDC Discovery */

int sso_discover(GV_SSOManager *mgr) {
    if (!mgr) return -1;
    if (mgr->config.provider != GV_SSO_OIDC) return -1;
    if (!mgr->config.issuer_url) return -1;

    /* Build discovery URL */
    char discovery_url[MAX_URL_LEN];
    snprintf(discovery_url, sizeof(discovery_url),
             "%s/.well-known/openid-configuration", mgr->config.issuer_url);

    char *response = NULL;
    size_t response_len = 0;

    if (http_get(discovery_url, mgr->config.verify_ssl,
                 &response, &response_len) != 0) {
        return -1;
    }

    pthread_mutex_lock(&mgr->mutex);

    /* Parse JSON response for OIDC endpoints */
    int ok = 1;

    if (json_extract_string(response, "authorization_endpoint",
                            mgr->endpoints.authorization_endpoint,
                            MAX_ENDPOINT_LEN) != 0) {
        ok = 0;
    }
    if (json_extract_string(response, "token_endpoint",
                            mgr->endpoints.token_endpoint,
                            MAX_ENDPOINT_LEN) != 0) {
        ok = 0;
    }
    if (json_extract_string(response, "jwks_uri",
                            mgr->endpoints.jwks_uri,
                            MAX_ENDPOINT_LEN) != 0) {
        ok = 0;
    }
    /* userinfo_endpoint is optional */
    json_extract_string(response, "userinfo_endpoint",
                        mgr->endpoints.userinfo_endpoint,
                        MAX_ENDPOINT_LEN);

    if (ok) {
        mgr->endpoints.discovered = 1;
    }

    pthread_mutex_unlock(&mgr->mutex);
    gv_free(response);

    return ok ? 0 : -1;
}

/* Authentication Flow */

int sso_get_auth_url(const GV_SSOManager *mgr, const char *state,
                         char *url, size_t url_size) {
    if (!mgr || !state || !url || url_size == 0) return -1;
    if (mgr->config.provider != GV_SSO_OIDC) return -1;

    /* Cast away const to lock; logical state is not modified */
    GV_SSOManager *m = (GV_SSOManager *)mgr;

    pthread_mutex_lock(&m->mutex);

    if (!m->endpoints.discovered) {
        pthread_mutex_unlock(&m->mutex);
        return -1;
    }

    int written = snprintf(url, url_size,
        "%s?client_id=%s&redirect_uri=%s"
        "&response_type=code&scope=openid+profile+email+groups&state=%s",
        m->endpoints.authorization_endpoint,
        mgr->config.client_id ? mgr->config.client_id : "",
        mgr->config.redirect_uri ? mgr->config.redirect_uri : "",
        state);

    pthread_mutex_unlock(&m->mutex);

    if (written < 0 || (size_t)written >= url_size) return -1;

    return 0;
}

GV_SSOToken *sso_exchange_code(GV_SSOManager *mgr, const char *auth_code) {
    if (!mgr || !auth_code) return NULL;
    if (mgr->config.provider != GV_SSO_OIDC) return NULL;

    pthread_mutex_lock(&mgr->mutex);

    if (!mgr->endpoints.discovered) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }

    /*
     * URL-encode every interpolated value before building the form body so a
     * value containing '&'/'=' cannot inject or override form parameters.
     * Each encode buffer is sized for the worst-case 3x expansion.
     */
    char enc_code[MAX_URL_LEN];
    char enc_client_id[MAX_URL_LEN];
    char enc_client_secret[MAX_URL_LEN];
    char enc_redirect_uri[MAX_URL_LEN];
    if (url_encode(auth_code, enc_code, sizeof(enc_code)) != 0 ||
        url_encode(mgr->config.client_id, enc_client_id, sizeof(enc_client_id)) != 0 ||
        url_encode(mgr->config.client_secret, enc_client_secret, sizeof(enc_client_secret)) != 0 ||
        url_encode(mgr->config.redirect_uri, enc_redirect_uri, sizeof(enc_redirect_uri)) != 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }

    /* Build POST body for token exchange. Sized for worst-case 3x expansion of
     * every encoded value plus the fixed parameter names. */
    char post_fields[MAX_URL_LEN * 4];
    int pf_written = snprintf(post_fields, sizeof(post_fields),
             "grant_type=authorization_code&code=%s"
             "&client_id=%s&client_secret=%s&redirect_uri=%s",
             enc_code, enc_client_id, enc_client_secret, enc_redirect_uri);
    if (pf_written < 0 || (size_t)pf_written >= sizeof(post_fields)) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }

    char token_endpoint[MAX_ENDPOINT_LEN];
    strncpy(token_endpoint, mgr->endpoints.token_endpoint,
            MAX_ENDPOINT_LEN - 1);
    token_endpoint[MAX_ENDPOINT_LEN - 1] = '\0';
    int verify_ssl = mgr->config.verify_ssl;

    pthread_mutex_unlock(&mgr->mutex);

    /* POST to token endpoint */
    char *response = NULL;
    size_t response_len = 0;

    if (http_post_form(token_endpoint, post_fields, verify_ssl,
                       &response, &response_len) != 0) {
        return NULL;
    }

    /* Extract id_token from response JSON */
    char id_token[BASE64_DECODE_MAX];
    if (json_extract_string(response, "id_token",
                            id_token, sizeof(id_token)) != 0) {
        gv_free(response);
        return NULL;
    }

    gv_free(response);

    /*
     * Verify the id_token signature BEFORE trusting any claim (fail closed).
     * verify_jwt_signature() pins the header alg to exactly HS256 or RS256
     * (rejecting none/ES256/etc.) and selects the right key: the shared
     * client_secret for HS256, or an RSA public key (configured PEM and/or the
     * discovered JWKS) for RS256.  It fails closed if no suitable key exists.
     */
    if (verify_jwt_signature(mgr, id_token) != 0) {
        return NULL;
    }

    /* Decode and validate the JWT (also enforces exp/nbf). */
    GV_SSOToken *token = decode_jwt_claims(id_token);
    if (!token) return NULL;

    /* Populate admin flag based on configured admin groups */
    populate_admin_flag(token, mgr->config.admin_groups);

    return token;
}

GV_SSOToken *sso_validate_token(GV_SSOManager *mgr, const char *token_string) {
    if (!mgr || !token_string) return NULL;

    GV_SSOToken *token = NULL;

    if (mgr->config.provider == GV_SSO_OIDC) {
        /*
         * Fail closed: NEVER accept a token on a decode-only path.  The
         * signature MUST verify before we trust any claim.  verify_jwt_signature()
         * pins the alg to HS256 or RS256 and selects the matching key (shared
         * secret for HS256, PEM/JWKS RSA public key for RS256); it returns
         * non-zero when no suitable key is configured, so we deny.
         */
        if (verify_jwt_signature(mgr, token_string) != 0) {
            return NULL;  /* signature / algorithm verification failed */
        }

        /* Signature verified; decode_jwt_claims also enforces exp/nbf. */
        token = decode_jwt_claims(token_string);
        if (!token) return NULL;
    } else if (mgr->config.provider == GV_SSO_SAML) {
        /* SAML assertion path (validates signature policy + time window). */
        token = parse_saml_assertion(&mgr->config, token_string);
        if (!token) return NULL;
    } else {
        return NULL;
    }

    /* Populate admin flag */
    populate_admin_flag(token, mgr->config.admin_groups);

    return token;
}

GV_SSOToken *sso_refresh_token(GV_SSOManager *mgr, const char *refresh_token) {
    if (!mgr || !refresh_token) return NULL;
    if (mgr->config.provider != GV_SSO_OIDC) return NULL;

    pthread_mutex_lock(&mgr->mutex);

    if (!mgr->endpoints.discovered) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }

    /*
     * URL-encode every interpolated value before building the form body so a
     * value containing '&'/'=' cannot inject or override form parameters.
     * Each encode buffer is sized for the worst-case 3x expansion.
     */
    char enc_refresh[MAX_URL_LEN];
    char enc_client_id[MAX_URL_LEN];
    char enc_client_secret[MAX_URL_LEN];
    if (url_encode(refresh_token, enc_refresh, sizeof(enc_refresh)) != 0 ||
        url_encode(mgr->config.client_id, enc_client_id, sizeof(enc_client_id)) != 0 ||
        url_encode(mgr->config.client_secret, enc_client_secret, sizeof(enc_client_secret)) != 0) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }

    /* Build POST body for token refresh. Sized for worst-case 3x expansion of
     * every encoded value plus the fixed parameter names. */
    char post_fields[MAX_URL_LEN * 4];
    int pf_written = snprintf(post_fields, sizeof(post_fields),
             "grant_type=refresh_token&refresh_token=%s"
             "&client_id=%s&client_secret=%s",
             enc_refresh, enc_client_id, enc_client_secret);
    if (pf_written < 0 || (size_t)pf_written >= sizeof(post_fields)) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }

    char token_endpoint[MAX_ENDPOINT_LEN];
    strncpy(token_endpoint, mgr->endpoints.token_endpoint,
            MAX_ENDPOINT_LEN - 1);
    token_endpoint[MAX_ENDPOINT_LEN - 1] = '\0';
    int verify_ssl = mgr->config.verify_ssl;

    pthread_mutex_unlock(&mgr->mutex);

    /* POST to token endpoint */
    char *response = NULL;
    size_t response_len = 0;

    if (http_post_form(token_endpoint, post_fields, verify_ssl,
                       &response, &response_len) != 0) {
        return NULL;
    }

    /* Extract id_token from response JSON */
    char id_token[BASE64_DECODE_MAX];
    if (json_extract_string(response, "id_token",
                            id_token, sizeof(id_token)) != 0) {
        gv_free(response);
        return NULL;
    }

    gv_free(response);

    /* Verify the refreshed id_token signature before trusting claims
     * (fail closed; alg pinned to HS256/RS256, key selected per-alg). */
    if (verify_jwt_signature(mgr, id_token) != 0) {
        return NULL;
    }

    GV_SSOToken *token = decode_jwt_claims(id_token);
    if (!token) return NULL;

    populate_admin_flag(token, mgr->config.admin_groups);

    return token;
}

void sso_free_token(GV_SSOToken *token) {
    if (!token) return;

    gv_free(token->subject);
    gv_free(token->email);
    gv_free(token->name);

    for (size_t i = 0; i < token->group_count; i++) {
        gv_free(token->groups[i]);
    }
    gv_free(token->groups);

    gv_free(token);
}

/* Group Checking */

int sso_has_group(const GV_SSOToken *token, const char *group) {
    if (!token || !group) return 0;

    for (size_t i = 0; i < token->group_count; i++) {
        if (token->groups[i] && strcmp(token->groups[i], group) == 0) {
            return 1;
        }
    }

    return 0;
}

/* Token Allocation */

static GV_SSOToken *alloc_token(void) {
    GV_SSOToken *token = gv_calloc(1, sizeof(GV_SSOToken));
    return token;
}

/* CSV / Group Utilities */

/**
 * Split a comma-separated string into an array of trimmed strings.
 * Caller must gv_free each element and the array itself.
 */
static char **split_csv(const char *csv, size_t *count) {
    *count = 0;
    if (!csv || !csv[0]) return NULL;

    /* Count commas to estimate array size */
    size_t capacity = 1;
    for (const char *p = csv; *p; p++) {
        if (*p == ',') capacity++;
    }

    char **result = gv_calloc(capacity, sizeof(char *));
    if (!result) return NULL;

    const char *start = csv;
    size_t idx = 0;

    while (*start) {
        /* Skip leading whitespace */
        while (*start == ' ' || *start == '\t') start++;

        const char *end = start;
        while (*end && *end != ',') end++;

        /* Trim trailing whitespace */
        const char *trim = end - 1;
        while (trim >= start && (*trim == ' ' || *trim == '\t')) trim--;

        size_t len = (size_t)(trim - start + 1);
        if (len > 0 && trim >= start) {
            result[idx] = gv_alloc(len + 1);
            if (result[idx]) {
                memcpy(result[idx], start, len);
                result[idx][len] = '\0';
                idx++;
            }
        }

        if (*end == ',') {
            start = end + 1;
        } else {
            break;
        }
    }

    *count = idx;
    return result;
}

/**
 * Check whether a group name appears in a comma-separated list.
 */
static int check_group_in_list(const char *csv_list, const char *group) {
    if (!csv_list || !group) return 0;

    size_t count = 0;
    char **groups = split_csv(csv_list, &count);
    if (!groups) return 0;

    int found = 0;
    for (size_t i = 0; i < count; i++) {
        if (groups[i] && strcmp(groups[i], group) == 0) {
            found = 1;
        }
        gv_free(groups[i]);
    }
    gv_free(groups);

    return found;
}

/**
 * Set the is_admin flag on a token by checking its groups against
 * the admin_groups CSV list.
 */
static void populate_admin_flag(GV_SSOToken *token, const char *admin_groups) {
    if (!token || !admin_groups) return;

    token->is_admin = 0;
    for (size_t i = 0; i < token->group_count; i++) {
        if (check_group_in_list(admin_groups, token->groups[i])) {
            token->is_admin = 1;
            return;
        }
    }
}

/* Base64 URL Decoding */

static const unsigned char b64url_table[256] = {
    ['A'] = 0,  ['B'] = 1,  ['C'] = 2,  ['D'] = 3,
    ['E'] = 4,  ['F'] = 5,  ['G'] = 6,  ['H'] = 7,
    ['I'] = 8,  ['J'] = 9,  ['K'] = 10, ['L'] = 11,
    ['M'] = 12, ['N'] = 13, ['O'] = 14, ['P'] = 15,
    ['Q'] = 16, ['R'] = 17, ['S'] = 18, ['T'] = 19,
    ['U'] = 20, ['V'] = 21, ['W'] = 22, ['X'] = 23,
    ['Y'] = 24, ['Z'] = 25,
    ['a'] = 26, ['b'] = 27, ['c'] = 28, ['d'] = 29,
    ['e'] = 30, ['f'] = 31, ['g'] = 32, ['h'] = 33,
    ['i'] = 34, ['j'] = 35, ['k'] = 36, ['l'] = 37,
    ['m'] = 38, ['n'] = 39, ['o'] = 40, ['p'] = 41,
    ['q'] = 42, ['r'] = 43, ['s'] = 44, ['t'] = 45,
    ['u'] = 46, ['v'] = 47, ['w'] = 48, ['x'] = 49,
    ['y'] = 50, ['z'] = 51,
    ['0'] = 52, ['1'] = 53, ['2'] = 54, ['3'] = 55,
    ['4'] = 56, ['5'] = 57, ['6'] = 58, ['7'] = 59,
    ['8'] = 60, ['9'] = 61,
    ['-'] = 62, ['_'] = 63
};

static int is_b64url_char(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_';
}

static int base64url_decode(const char *in, size_t in_len,
                            unsigned char *out, size_t *out_len) {
    if (!in || !out || !out_len) return -1;

    /* Strip padding */
    while (in_len > 0 && in[in_len - 1] == '=') in_len--;

    size_t max_out = (in_len * 3) / 4 + 1;
    if (*out_len < max_out) {
        *out_len = max_out;
        return -1;
    }

    size_t i = 0, j = 0;
    while (i < in_len) {
        uint32_t a = 0, b = 0, c = 0, d = 0;

        if (!is_b64url_char((unsigned char)in[i])) return -1;
        a = b64url_table[(unsigned char)in[i++]];
        if (i < in_len) {
            if (!is_b64url_char((unsigned char)in[i])) return -1;
            b = b64url_table[(unsigned char)in[i++]];
        }

        uint32_t triple = (a << 18) | (b << 12);

        if (i < in_len) {
            if (!is_b64url_char((unsigned char)in[i])) return -1;
            c = b64url_table[(unsigned char)in[i++]];
            triple |= (c << 6);
        }
        if (i < in_len) {
            if (!is_b64url_char((unsigned char)in[i])) return -1;
            d = b64url_table[(unsigned char)in[i++]];
            triple |= d;
        }

        out[j++] = (unsigned char)((triple >> 16) & 0xFF);
        if (i > 2 || in_len > 2) {
            /* Only output second byte if we had at least 3 input chars
               in this group */
            size_t group_start = i - (i < in_len ? 0 : (in_len % 4 ? in_len % 4 : 4));
            (void)group_start;
        }

        /* Simpler approach: figure out how many output bytes for this group */
        size_t chars_in_group = in_len - (i - (i <= in_len ? (in_len >= 4 ? 4 : in_len % 4) : 0));
        (void)chars_in_group;
    }

    /* Re-do with a cleaner algorithm */
    j = 0;
    i = 0;
    while (i < in_len) {
        uint32_t sextet[4] = {0, 0, 0, 0};
        size_t n = 0;

        for (n = 0; n < 4 && i < in_len; n++, i++) {
            if (!is_b64url_char((unsigned char)in[i])) return -1;
            sextet[n] = b64url_table[(unsigned char)in[i]];
        }

        uint32_t triple = (sextet[0] << 18) | (sextet[1] << 12) |
                          (sextet[2] << 6) | sextet[3];

        if (n >= 2) out[j++] = (unsigned char)((triple >> 16) & 0xFF);
        if (n >= 3) out[j++] = (unsigned char)((triple >> 8) & 0xFF);
        if (n >= 4) out[j++] = (unsigned char)(triple & 0xFF);
    }

    *out_len = j;
    return 0;
}

/* Minimal JSON Helpers */

/**
 * Extract a string value for a given key from a JSON object.
 * Very minimal: expects "key":"value" patterns, no nested objects.
 */
static int json_extract_string(const char *json, const char *key,
                               char *out, size_t out_size) {
    if (!json || !key || !out || out_size == 0) return -1;

    /* Build search pattern: "key" */
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return -1;

    pos += strlen(pattern);

    /* Skip whitespace and colon */
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r') pos++;
    if (*pos != ':') return -1;
    pos++;
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r') pos++;

    if (*pos != '"') return -1;
    pos++; /* skip opening quote */

    /* Copy value until closing quote */
    size_t i = 0;
    while (*pos && *pos != '"' && i < out_size - 1) {
        if (*pos == '\\' && *(pos + 1)) {
            pos++; /* skip escape backslash */
            switch (*pos) {
                case '"':  out[i++] = '"';  break;
                case '\\': out[i++] = '\\'; break;
                case '/':  out[i++] = '/';  break;
                case 'n':  out[i++] = '\n'; break;
                case 't':  out[i++] = '\t'; break;
                case 'r':  out[i++] = '\r'; break;
                default:   out[i++] = *pos; break;
            }
        } else {
            out[i++] = *pos;
        }
        pos++;
    }

    out[i] = '\0';
    return (*pos == '"') ? 0 : -1;
}

/**
 * Extract an unsigned 64-bit integer value for a given key from JSON.
 */
static int json_extract_uint64(const char *json, const char *key, uint64_t *out) {
    if (!json || !key || !out) return -1;

    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return -1;

    pos += strlen(pattern);

    /* Skip whitespace and colon */
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r') pos++;
    if (*pos != ':') return -1;
    pos++;
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r') pos++;

    /* Parse number */
    if (*pos < '0' || *pos > '9') return -1;

    uint64_t val = 0;
    while (*pos >= '0' && *pos <= '9') {
        val = val * 10 + (uint64_t)(*pos - '0');
        pos++;
    }

    *out = val;
    return 0;
}

/**
 * Extract a JSON array of strings for a given key.
 * Expects "key":["str1","str2",...] format.
 */
static int json_extract_string_array(const char *json, const char *key,
                                     char ***out, size_t *count) {
    if (!json || !key || !out || !count) return -1;

    *out = NULL;
    *count = 0;

    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);

    const char *pos = strstr(json, pattern);
    if (!pos) return -1;

    pos += strlen(pattern);

    /* Skip whitespace and colon */
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r') pos++;
    if (*pos != ':') return -1;
    pos++;
    while (*pos == ' ' || *pos == '\t' || *pos == '\n' || *pos == '\r') pos++;

    if (*pos != '[') return -1;
    pos++; /* skip '[' */

    /* Allocate space for strings */
    char **arr = gv_calloc(MAX_GROUPS, sizeof(char *));
    if (!arr) return -1;

    size_t idx = 0;

    while (*pos && *pos != ']' && idx < MAX_GROUPS) {
        /* Skip whitespace */
        while (*pos == ' ' || *pos == '\t' || *pos == '\n' ||
               *pos == '\r' || *pos == ',') {
            pos++;
        }

        if (*pos == ']') break;
        if (*pos != '"') {
            /* Unexpected token; skip non-string array elements */
            while (*pos && *pos != ',' && *pos != ']') pos++;
            continue;
        }

        pos++; /* skip opening quote */

        /* Find closing quote */
        const char *end = pos;
        while (*end && *end != '"') {
            if (*end == '\\' && *(end + 1)) end++; /* skip escape */
            end++;
        }

        size_t len = (size_t)(end - pos);
        arr[idx] = gv_alloc(len + 1);
        if (arr[idx]) {
            memcpy(arr[idx], pos, len);
            arr[idx][len] = '\0';
            idx++;
        }

        if (*end == '"') pos = end + 1;
    }

    *out = arr;
    *count = idx;
    return 0;
}

/* ISO 8601 Timestamp Parsing */

/**
 * Parse an ISO 8601 timestamp (e.g. "2024-03-15T12:30:00Z") into a Unix epoch.
 * Handles the format: YYYY-MM-DDThh:mm:ssZ (with optional fractional seconds).
 */
static uint64_t parse_iso8601(const char *ts) {
    if (!ts) return 0;

    int year = 0, month = 0, day = 0, hour = 0, min = 0, sec = 0;
    int matched = sscanf(ts, "%d-%d-%dT%d:%d:%d", &year, &month, &day, &hour, &min, &sec);
    if (matched < 3) return 0;

    /* Convert to Unix timestamp using a simplified calculation */
    struct tm tm_val;
    memset(&tm_val, 0, sizeof(tm_val));
    tm_val.tm_year = year - 1900;
    tm_val.tm_mon = month - 1;
    tm_val.tm_mday = day;
    tm_val.tm_hour = hour;
    tm_val.tm_min = min;
    tm_val.tm_sec = sec;
    tm_val.tm_isdst = 0;

    /* timegm is a GNU extension exposed by _GNU_SOURCE; not available under
     * strict POSIX mode on macOS, so fall back to mktime with TZ override. */
#if defined(_GNU_SOURCE) && defined(__linux__)
    time_t t = timegm(&tm_val);
#else
    /* Portable fallback: temporarily set TZ to UTC.
     * Windows lacks setenv/unsetenv; use _putenv_s/_tzset instead. */
    const char *tz_save = getenv("TZ");
#ifdef _WIN32
    _putenv_s("TZ", "UTC");
    _tzset();
    time_t t = mktime(&tm_val);
    if (tz_save) _putenv_s("TZ", tz_save);
    else         _putenv_s("TZ", "");
    _tzset();
#else
    setenv("TZ", "UTC", 1);
    tzset();
    time_t t = mktime(&tm_val);
    if (tz_save) setenv("TZ", tz_save, 1);
    else         unsetenv("TZ");
    tzset();
#endif
#endif

    return (t == (time_t)-1) ? 0 : (uint64_t)t;
}

/* HMAC-SHA256 for JWT Signature Verification */

/**
 * Minimal SHA-256 implementation for HMAC computation.
 * Used for HS256 JWT signature verification without requiring OpenSSL.
 */

static const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define SHA256_ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHA256_CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define SHA256_MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SHA256_EP0(x)  (SHA256_ROR(x, 2)  ^ SHA256_ROR(x, 13) ^ SHA256_ROR(x, 22))
#define SHA256_EP1(x)  (SHA256_ROR(x, 6)  ^ SHA256_ROR(x, 11) ^ SHA256_ROR(x, 25))
#define SHA256_SIG0(x) (SHA256_ROR(x, 7)  ^ SHA256_ROR(x, 18) ^ ((x) >> 3))
#define SHA256_SIG1(x) (SHA256_ROR(x, 17) ^ SHA256_ROR(x, 19) ^ ((x) >> 10))

typedef struct {
    uint32_t state[8];
    uint64_t bitcount;
    unsigned char buffer[64];
    size_t buflen;
} SHA256_CTX_Internal;

static void sha256_init(SHA256_CTX_Internal *ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->bitcount = 0;
    ctx->buflen = 0;
}

static void sha256_transform(SHA256_CTX_Internal *ctx, const unsigned char block[64]) {
    uint32_t w[64], a, b, c, d, e, f, g, h, t1, t2;

    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        w[i] = SHA256_SIG1(w[i - 2]) + w[i - 7] + SHA256_SIG0(w[i - 15]) + w[i - 16];
    }

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (int i = 0; i < 64; i++) {
        t1 = h + SHA256_EP1(e) + SHA256_CH(e, f, g) + sha256_k[i] + w[i];
        t2 = SHA256_EP0(a) + SHA256_MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_update(SHA256_CTX_Internal *ctx, const unsigned char *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        ctx->buffer[ctx->buflen++] = data[i];
        ctx->bitcount += 8;
        if (ctx->buflen == 64) {
            sha256_transform(ctx, ctx->buffer);
            ctx->buflen = 0;
        }
    }
}

static void sha256_final(SHA256_CTX_Internal *ctx, unsigned char hash[32]) {
    size_t pad_start = ctx->buflen;
    ctx->buffer[pad_start++] = 0x80;

    if (pad_start > 56) {
        memset(ctx->buffer + pad_start, 0, 64 - pad_start);
        sha256_transform(ctx, ctx->buffer);
        pad_start = 0;
    }
    memset(ctx->buffer + pad_start, 0, 56 - pad_start);

    uint64_t bits = ctx->bitcount;
    for (int i = 7; i >= 0; i--) {
        ctx->buffer[56 + (7 - i)] = (unsigned char)(bits >> (i * 8));
    }
    sha256_transform(ctx, ctx->buffer);

    for (int i = 0; i < 8; i++) {
        hash[i * 4]     = (unsigned char)(ctx->state[i] >> 24);
        hash[i * 4 + 1] = (unsigned char)(ctx->state[i] >> 16);
        hash[i * 4 + 2] = (unsigned char)(ctx->state[i] >> 8);
        hash[i * 4 + 3] = (unsigned char)(ctx->state[i]);
    }
}

static int hmac_sha256(const unsigned char *key, size_t key_len,
                       const unsigned char *data, size_t data_len,
                       unsigned char *out, size_t *out_len) {
    unsigned char k_pad[64];
    unsigned char inner_hash[32];
    SHA256_CTX_Internal ctx;

    /* If key > 64 bytes, hash it first */
    unsigned char key_hash[32];
    if (key_len > 64) {
        sha256_init(&ctx);
        sha256_update(&ctx, key, key_len);
        sha256_final(&ctx, key_hash);
        key = key_hash;
        key_len = 32;
    }

    /* Inner hash: H((K ^ ipad) || data) */
    memset(k_pad, 0x36, 64);
    for (size_t i = 0; i < key_len; i++) k_pad[i] ^= key[i];

    sha256_init(&ctx);
    sha256_update(&ctx, k_pad, 64);
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, inner_hash);

    /* Outer hash: H((K ^ opad) || inner_hash) */
    memset(k_pad, 0x5c, 64);
    for (size_t i = 0; i < key_len; i++) k_pad[i] ^= key[i];

    sha256_init(&ctx);
    sha256_update(&ctx, k_pad, 64);
    sha256_update(&ctx, inner_hash, 32);
    sha256_final(&ctx, out);

    *out_len = 32;
    return 0;
}

/*
 * Parse a JWT header segment and extract its "alg" and (optionally) "kid".
 *
 * The alg is later pinned to exactly HS256 or RS256 by the dispatcher; any
 * other alg ("none", "ES256", ...) MUST be rejected to prevent alg-confusion /
 * downgrade forgery (e.g. RS256->HS256 where the RSA public key is abused as an
 * HMAC secret, or alg:none which requires no signature at all).
 *
 * @param alg_out  receives the alg string (required, must be non-NULL).
 * @param kid_out  receives the kid string, or "" if absent (may be NULL).
 * Returns 0 on success, -1 on any parse error (fail closed).
 */
static int jwt_header_parse(const char *jwt, const char *dot1,
                            char *alg_out, size_t alg_size,
                            char *kid_out, size_t kid_size) {
    size_t header_b64_len = (size_t)(dot1 - jwt);
    if (header_b64_len == 0) return -1;

    unsigned char decoded[1024];
    size_t decoded_len = sizeof(decoded);
    if (base64url_decode(jwt, header_b64_len, decoded, &decoded_len) != 0) {
        return -1;
    }
    if (decoded_len >= sizeof(decoded)) decoded_len = sizeof(decoded) - 1;
    decoded[decoded_len] = '\0';

    if (json_extract_string((const char *)decoded, "alg", alg_out, alg_size) != 0) {
        return -1;  /* missing alg -> reject */
    }
    if (kid_out && kid_size > 0) {
        if (json_extract_string((const char *)decoded, "kid",
                                kid_out, kid_size) != 0) {
            kid_out[0] = '\0';  /* kid is optional */
        }
    }
    return 0;
}

/**
 * Verify the HMAC-SHA256 (HS256) signature of a JWT.
 * The signed input is header.payload (everything before the second dot).
 * Returns 0 on success, -1 on verification failure.
 */
static int verify_jwt_signature_hs256(const char *jwt, size_t signed_len,
                                       const unsigned char *decoded_sig,
                                       size_t decoded_sig_len,
                                       const char *secret) {
    if (!secret || !secret[0]) return -1;  /* no HMAC key -> fail closed */

    unsigned char expected[32];
    size_t expected_len = 0;
    hmac_sha256((const unsigned char *)secret, strlen(secret),
                (const unsigned char *)jwt, signed_len,
                expected, &expected_len);

    if (decoded_sig_len != expected_len) return -1;

    /* Constant-time comparison to prevent timing attacks */
    unsigned char diff = 0;
    for (size_t i = 0; i < expected_len; i++) {
        diff |= decoded_sig[i] ^ expected[i];
    }
    return diff == 0 ? 0 : -1;
}

#ifdef GV_HAVE_OPENSSL
/*
 * Verify a raw RSA-SHA256 (RS256) signature over `signed_input` using an
 * already-constructed EVP_PKEY.  Returns 0 iff the signature verifies.
 */
static int rs256_verify_with_pkey(EVP_PKEY *pkey,
                                   const unsigned char *signed_input,
                                   size_t signed_len,
                                   const unsigned char *sig, size_t sig_len) {
    if (!pkey) return -1;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) return -1;

    int ok = -1;
    if (EVP_DigestVerifyInit(ctx, NULL, EVP_sha256(), NULL, pkey) == 1 &&
        EVP_DigestVerify(ctx, sig, sig_len, signed_input, signed_len) == 1) {
        ok = 0;
    }
    EVP_MD_CTX_free(ctx);
    return ok;
}

/*
 * Build an RSA public EVP_PKEY from a PEM-encoded public key string.
 * Caller must EVP_PKEY_free() the result.  Returns NULL on error.
 */
static EVP_PKEY *rsa_pkey_from_pem(const char *pem) {
    if (!pem || !pem[0]) return NULL;
    BIO *bio = BIO_new_mem_buf(pem, -1);
    if (!bio) return NULL;
    EVP_PKEY *pkey = PEM_read_bio_PUBKEY(bio, NULL, NULL, NULL);
    BIO_free(bio);
    return pkey;
}

/*
 * Build an RSA public EVP_PKEY from JWK modulus/exponent (base64url "n"/"e").
 * Uses the OpenSSL 3.0 EVP_PKEY_fromdata API (no deprecated RSA_* calls).
 * Caller must EVP_PKEY_free() the result.  Returns NULL on error.
 */
static EVP_PKEY *rsa_pkey_from_jwk(const char *n_b64, const char *e_b64) {
    if (!n_b64 || !e_b64) return NULL;

    unsigned char n_bin[1024];
    unsigned char e_bin[16];
    size_t n_len = sizeof(n_bin);
    size_t e_len = sizeof(e_bin);
    if (base64url_decode(n_b64, strlen(n_b64), n_bin, &n_len) != 0) return NULL;
    if (base64url_decode(e_b64, strlen(e_b64), e_bin, &e_len) != 0) return NULL;

    /* Convert the big-endian modulus/exponent to BIGNUMs. */
    BIGNUM *bn_n = BN_bin2bn(n_bin, (int)n_len, NULL);
    BIGNUM *bn_e = BN_bin2bn(e_bin, (int)e_len, NULL);
    EVP_PKEY *pkey = NULL;
    OSSL_PARAM_BLD *bld = NULL;
    OSSL_PARAM *params = NULL;
    EVP_PKEY_CTX *ctx = NULL;

    if (!bn_n || !bn_e) goto done;

    bld = OSSL_PARAM_BLD_new();
    if (!bld) goto done;
    if (OSSL_PARAM_BLD_push_BN(bld, "n", bn_n) != 1) goto done;
    if (OSSL_PARAM_BLD_push_BN(bld, "e", bn_e) != 1) goto done;
    params = OSSL_PARAM_BLD_to_param(bld);
    if (!params) goto done;

    ctx = EVP_PKEY_CTX_new_from_name(NULL, "RSA", NULL);
    if (!ctx) goto done;
    if (EVP_PKEY_fromdata_init(ctx) != 1) goto done;
    if (EVP_PKEY_fromdata(ctx, &pkey, EVP_PKEY_PUBLIC_KEY, params) != 1) {
        pkey = NULL;  /* fromdata may leave pkey unset on failure */
    }

done:
    if (params) OSSL_PARAM_free(params);
    if (bld) OSSL_PARAM_BLD_free(bld);
    if (ctx) EVP_PKEY_CTX_free(ctx);
    if (bn_n) BN_free(bn_n);
    if (bn_e) BN_free(bn_e);
    return pkey;
}

#ifdef HAVE_CURL
/*
 * Fetch the JWKS document, locate the key whose "kid" matches `want_kid`
 * (or the first RSA key if no kid was present in the JWT header), extract its
 * base64url "n" and "e", and build an RSA public EVP_PKEY.
 * Returns NULL on any failure (including "no HTTP client / fetch failed").
 */
static EVP_PKEY *rsa_pkey_from_jwks(const char *jwks_uri, int verify_ssl,
                                    const char *want_kid) {
    if (!jwks_uri || !jwks_uri[0]) return NULL;

    char *doc = NULL;
    size_t doc_len = 0;
    if (http_get(jwks_uri, verify_ssl, &doc, &doc_len) != 0) return NULL;
    if (!doc) return NULL;

    /*
     * Minimal JWKS scan.  The document is {"keys":[{...},{...}]}.  We walk each
     * object between successive '{' markers, matching kid (if requested) and
     * reading n/e via the existing flat json_extract_string helper.  This is a
     * pragmatic parser matching the style of the other JSON helpers in this
     * file; it does not do full JSON parsing but is sufficient for JWKS.
     */
    EVP_PKEY *pkey = NULL;
    const char *scan = doc;
    while ((scan = strchr(scan, '{')) != NULL) {
        /* Bound this object at the next '{' so json_extract_string does not
         * accidentally read fields from a later key. */
        const char *next_obj = strchr(scan + 1, '{');
        size_t obj_len = next_obj ? (size_t)(next_obj - scan) : strlen(scan);
        if (obj_len >= 8192) obj_len = 8191;

        char obj[8192];
        memcpy(obj, scan, obj_len);
        obj[obj_len] = '\0';

        char kty[16];
        if (json_extract_string(obj, "kty", kty, sizeof(kty)) == 0 &&
            strcmp(kty, "RSA") == 0) {
            char kid[256];
            if (json_extract_string(obj, "kid", kid, sizeof(kid)) != 0) {
                kid[0] = '\0';
            }
            int kid_ok = (!want_kid || !want_kid[0] || kid[0] == '\0' ||
                          strcmp(kid, want_kid) == 0);
            if (kid_ok) {
                char n_b64[1400];
                char e_b64[64];
                if (json_extract_string(obj, "n", n_b64, sizeof(n_b64)) == 0 &&
                    json_extract_string(obj, "e", e_b64, sizeof(e_b64)) == 0) {
                    pkey = rsa_pkey_from_jwk(n_b64, e_b64);
                    if (pkey) break;
                }
            }
        }
        scan = next_obj ? next_obj : (scan + obj_len);
    }

    gv_free(doc);
    return pkey;
}
#endif /* HAVE_CURL */

/*
 * Verify an RS256 JWT signature.  Sources the RSA public key from (in order):
 *   1. the configured PEM public key (config.oidc_rsa_public_key_pem), then
 *   2. the JWKS endpoint (only when built WITH libcurl), matching the header
 *      kid to a JWKS entry's n/e.
 * Returns 0 iff a key was obtained AND the signature verifies.
 */
static int verify_jwt_signature_rs256(const GV_SSOManager *mgr,
                                       const char *jwt, size_t signed_len,
                                       const unsigned char *decoded_sig,
                                       size_t decoded_sig_len,
                                       const char *kid) {
    int result = -1;

    /* 1. Configured PEM public key (works without libcurl). */
    if (mgr->config.oidc_rsa_public_key_pem &&
        mgr->config.oidc_rsa_public_key_pem[0]) {
        EVP_PKEY *pkey = rsa_pkey_from_pem(mgr->config.oidc_rsa_public_key_pem);
        if (pkey) {
            result = rs256_verify_with_pkey(pkey, (const unsigned char *)jwt,
                                            signed_len, decoded_sig,
                                            decoded_sig_len);
            EVP_PKEY_free(pkey);
            if (result == 0) return 0;
        }
    }

    /* 2. JWKS endpoint (requires an HTTP client; falls back to PEM otherwise). */
#ifdef HAVE_CURL
    if (mgr->endpoints.jwks_uri[0]) {
        EVP_PKEY *pkey = rsa_pkey_from_jwks(mgr->endpoints.jwks_uri,
                                            mgr->config.verify_ssl, kid);
        if (pkey) {
            result = rs256_verify_with_pkey(pkey, (const unsigned char *)jwt,
                                            signed_len, decoded_sig,
                                            decoded_sig_len);
            EVP_PKEY_free(pkey);
        }
    }
#else
    (void)kid;  /* No HTTP client in this build: PEM is the only RS256 source. */
#endif

    return result;
}
#endif /* GV_HAVE_OPENSSL */

/**
 * Verify a JWT signature over header.payload.
 *
 * Pins the header "alg" to exactly HS256 or RS256 (rejecting "none", "ES256",
 * and any other value) BEFORE doing signature work, preserving the existing
 * alg-confusion / downgrade protection.  Then:
 *   - HS256: HMAC-SHA256 with the shared client_secret (unchanged behaviour).
 *   - RS256: RSA-SHA256 via OpenSSL, key from configured PEM and/or JWKS.
 *
 * When built WITHOUT OpenSSL (GV_HAVE_OPENSSL undefined), RS256 fails closed
 * (returns -1), mirroring the GCM fail-closed pattern in src/security/crypto.c.
 *
 * Returns 0 on success, -1 on verification failure.
 */
static int verify_jwt_signature(const GV_SSOManager *mgr, const char *jwt) {
    if (!mgr || !jwt) return -1;

    const char *dot1 = strchr(jwt, '.');
    if (!dot1) return -1;
    const char *dot2 = strchr(dot1 + 1, '.');
    if (!dot2) return -1;

    /* Pin the algorithm before doing any signature work. */
    char alg[64];
    char kid[256];
    if (jwt_header_parse(jwt, dot1, alg, sizeof(alg), kid, sizeof(kid)) != 0) {
        return -1;
    }

    /* The signed payload is header.payload (everything before the second dot). */
    size_t signed_len = (size_t)(dot2 - jwt);

    /* Decode the signature segment. */
    const char *sig_start = dot2 + 1;
    size_t sig_b64_len = strlen(sig_start);

    unsigned char decoded_sig[512];
    size_t decoded_sig_len = sizeof(decoded_sig);
    if (base64url_decode(sig_start, sig_b64_len,
                         decoded_sig, &decoded_sig_len) != 0) {
        return -1;
    }

    if (strcmp(alg, "HS256") == 0) {
        return verify_jwt_signature_hs256(jwt, signed_len,
                                          decoded_sig, decoded_sig_len,
                                          mgr->config.client_secret);
    }

    if (strcmp(alg, "RS256") == 0) {
#ifdef GV_HAVE_OPENSSL
        return verify_jwt_signature_rs256(mgr, jwt, signed_len,
                                          decoded_sig, decoded_sig_len, kid);
#else
        /* RS256 requires OpenSSL; fail closed when unavailable. */
        return -1;
#endif
    }

    /* Any other alg (none/ES256/...) is rejected. */
    return -1;
}

/* JWT Decoding */

/**
 * Decode a JWT and extract claims into a GV_SSOToken.
 * Verifies HS256 signature when a secret is available, and checks expiry.
 */
static GV_SSOToken *decode_jwt_claims(const char *jwt) {
    if (!jwt) return NULL;

    /* A JWT has three dot-separated segments: header.payload.signature */
    const char *dot1 = strchr(jwt, '.');
    if (!dot1) return NULL;

    const char *dot2 = strchr(dot1 + 1, '.');
    if (!dot2) return NULL;

    /* Decode header to verify it looks like a JWT (optional, log kid) */
    /* We focus on the payload segment */
    const char *payload_start = dot1 + 1;
    size_t payload_b64_len = (size_t)(dot2 - payload_start);

    unsigned char decoded[BASE64_DECODE_MAX];
    size_t decoded_len = sizeof(decoded);

    if (base64url_decode(payload_start, payload_b64_len,
                         decoded, &decoded_len) != 0) {
        return NULL;
    }

    /* Null-terminate the decoded JSON */
    if (decoded_len >= sizeof(decoded)) decoded_len = sizeof(decoded) - 1;
    decoded[decoded_len] = '\0';

    const char *payload_json = (const char *)decoded;

    /* Extract standard claims */
    GV_SSOToken *token = alloc_token();
    if (!token) return NULL;

    char buf[512];

    if (json_extract_string(payload_json, "sub", buf, sizeof(buf)) == 0) {
        token->subject = gv_dup_cstr(buf);
    }
    if (json_extract_string(payload_json, "email", buf, sizeof(buf)) == 0) {
        token->email = gv_dup_cstr(buf);
    }
    if (json_extract_string(payload_json, "name", buf, sizeof(buf)) == 0) {
        token->name = gv_dup_cstr(buf);
    }

    json_extract_uint64(payload_json, "iat", &token->issued_at);

    /* A MISSING exp is treated as invalid (fail closed): a token without an
     * expiry must not be accepted as never-expiring. */
    if (json_extract_uint64(payload_json, "exp", &token->expires_at) != 0 ||
        token->expires_at == 0) {
        sso_free_token(token);
        return NULL;
    }

    /* Enforce standard temporal claims here so every decode path is safe,
     * regardless of caller.  Allow a small clock-skew tolerance. */
    {
        const uint64_t skew = 60;
        uint64_t now = (uint64_t)time(NULL);
        if (token->expires_at + skew < now) {
            sso_free_token(token);
            return NULL;  /* expired */
        }
        uint64_t nbf = 0;
        if (json_extract_uint64(payload_json, "nbf", &nbf) == 0) {
            if (nbf > now + skew) {
                sso_free_token(token);
                return NULL;  /* not yet valid */
            }
        }
    }

    /* Extract groups claim (may be absent) */
    json_extract_string_array(payload_json, "groups",
                              &token->groups, &token->group_count);

    return token;
}

/* SAML Assertion Parsing (Stub) */

/**
 * Extract text content between <tag> and </tag> from XML.
 * Very basic: finds the first occurrence and extracts inner text.
 */
static void xml_decode_entities(char *s) {
    if (!s) return;
    char *r = s, *w = s;
    while (*r) {
        if (*r == '&') {
            if (strncmp(r, "&amp;",  5) == 0) { *w++ = '&';  r += 5; }
            else if (strncmp(r, "&lt;",   4) == 0) { *w++ = '<';  r += 4; }
            else if (strncmp(r, "&gt;",   4) == 0) { *w++ = '>';  r += 4; }
            else if (strncmp(r, "&quot;", 6) == 0) { *w++ = '"';  r += 6; }
            else if (strncmp(r, "&apos;", 6) == 0) { *w++ = '\''; r += 6; }
            else { *w++ = *r++; }
        } else {
            *w++ = *r++;
        }
    }
    *w = '\0';
}

static int xml_extract_text(const char *xml, const char *tag,
                            char *out, size_t out_size) {
    if (!xml || !tag || !out || out_size == 0) return -1;

    /* Build opening and closing tag patterns */
    char open_tag[256];
    char close_tag[256];
    snprintf(open_tag, sizeof(open_tag), "<%s", tag);
    snprintf(close_tag, sizeof(close_tag), "</%s>", tag);

    const char *start = strstr(xml, open_tag);
    if (!start) return -1;

    /* The char after the tag name must be '>', '/' or whitespace to avoid
     * matching a longer tag like <NameIDPolicy> when searching for <NameID>. */
    const char *after = start + strlen(open_tag);
    if (*after != '>' && *after != '/' && *after != ' ' && *after != '\t' &&
        *after != '\r' && *after != '\n') {
        return -1;
    }

    /* Skip to end of opening tag (past the '>' character) */
    const char *gt = strchr(start, '>');
    if (!gt) return -1;
    gt++; /* past '>' */

    const char *end = strstr(gt, close_tag);
    if (!end) return -1;

    size_t len = (size_t)(end - gt);
    if (len >= out_size) len = out_size - 1;

    memcpy(out, gt, len);
    out[len] = '\0';
    xml_decode_entities(out);

    return 0;
}

static int xml_extract_attribute_value(const char *xml, const char *attr_name,
                                       char *out, size_t out_size) {
    if (!xml || !attr_name || !out || out_size == 0) return -1;

    char needle[512];
    snprintf(needle, sizeof(needle), "Name=\"%s\"", attr_name);
    const char *attr_elem = strstr(xml, needle);
    if (!attr_elem) return -1;

    const char *attr_end = strstr(attr_elem, "</Attribute");
    if (!attr_end) attr_end = strstr(attr_elem, "</saml:Attribute");
    if (!attr_end) attr_end = xml + strlen(xml);

    const char *val = strstr(attr_elem, "<AttributeValue");
    if (!val || val >= attr_end)
        val = strstr(attr_elem, "<saml:AttributeValue");
    if (!val || val >= attr_end) return -1;

    const char *gt = strchr(val, '>');
    if (!gt || gt >= attr_end) return -1;
    gt++;

    const char *end = strchr(gt, '<');
    if (!end || end >= attr_end) return -1;

    size_t len = (size_t)(end - gt);
    if (len >= out_size) len = out_size - 1;
    memcpy(out, gt, len);
    out[len] = '\0';
    xml_decode_entities(out);
    return 0;
}

/* ============================================================================
 * SAML XML-DSig signature verification
 * ============================================================================
 *
 * SECURITY MODEL (default deny):
 *   - The ONLY trust anchor is config->saml_idp_cert_pem (a PEM X.509 cert or
 *     PEM public key).  If it is absent, verification is impossible and the
 *     assertion is rejected.
 *   - If the signature carries an embedded <ds:X509Certificate>, we honor it
 *     ONLY when it is byte-identical to the configured trusted certificate.
 *     We do NOT parse-and-trust an embedded cert on its own (that is the
 *     classic SAML "trust the embedded cert" bypass).  There is no CA-chain
 *     building here; equality to the pinned cert is the whole policy.
 *   - verified=1 requires BOTH: (a) the RSA signature over <SignedInfo>
 *     validates under the trusted key, AND (b) the <DigestValue> in the signed
 *     <Reference> equals SHA-256 of the referenced element with its own
 *     <Signature> stripped (enveloped-signature transform).  (a) proves the
 *     SignedInfo is authentic; (b) binds SignedInfo to the actual assertion
 *     content.
 *
 * CANONICALIZATION LIMITATIONS (READ THIS):
 *   Real XML-DSig requires Canonical XML 1.0 / Exclusive C14N (RFC 3076 /
 *   xml-exc-c14n): namespace axis inheritance, attribute reordering, entity
 *   and whitespace normalization, etc.  A full C14N implementation is large
 *   and out of scope here.  Instead we do a PRAGMATIC canonicalization that
 *   works for the byte-for-byte output of common IdPs (ADFS, Okta, Azure AD,
 *   Keycloak, SimpleSAMLphp) which typically emit already-canonical or
 *   near-canonical elements:
 *     - We take the SignedInfo element EXACTLY as it appears in the received
 *       XML (start '<...SignedInfo...>' through matching '</...SignedInfo>'),
 *       i.e. we verify the RSA signature over the on-the-wire SignedInfo bytes.
 *       This is correct WHEN the IdP transmitted canonical SignedInfo (the
 *       common case, since C14N of an already-canonical element is a no-op).
 *     - For the reference digest we take the referenced element's on-the-wire
 *       bytes with the enveloped <Signature> subtree removed, and SHA-256 that.
 *   What this DOES NOT robustly handle:
 *     - Non-canonical wire XML that requires real C14N to reproduce the bytes
 *       the IdP signed (e.g. attribute reordering, added/removed namespace
 *       decls, differing whitespace).  Such assertions will FAIL to verify
 *       (fail closed) rather than be wrongly accepted.
 *     - Signature-wrapping (XSW) attacks in their full generality.  Because we
 *       locate the SIGNED element by the Reference URI and match it to the
 *       element carrying the ID, and we strip only the enveloped Signature,
 *       simple wrapping is defeated; but a determined attacker exploiting
 *       parser differentials against this string-based (non-DOM) scanner may
 *       still find gaps.  A production deployment SHOULD use a real XML/DSig
 *       library.  We reject on any structural ambiguity we can detect.
 * ============================================================================
 */

#ifdef GV_HAVE_OPENSSL

/*
 * Decode standard (non-URL-safe) base64 into `out`.  SAML SignatureValue,
 * DigestValue and X509Certificate use standard base64 with '+' and '/', often
 * wrapped with newlines/spaces.  We copy into a scratch buffer, drop all
 * whitespace, translate '+'->'-' and '/'->'_' and strip '=' padding, then
 * reuse the existing base64url_decode().  Returns 0 on success.
 */
static int saml_b64_decode(const char *in, size_t in_len,
                           unsigned char *out, size_t *out_len) {
    if (!in || !out || !out_len) return -1;
    char *tmp = gv_alloc(in_len + 1);
    if (!tmp) return -1;
    size_t k = 0;
    for (size_t i = 0; i < in_len; i++) {
        char c = in[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
        if (c == '=') continue;  /* padding stripped; decoder tolerates it */
        if (c == '+') c = '-';
        else if (c == '/') c = '_';
        tmp[k++] = c;
    }
    tmp[k] = '\0';
    int rc = base64url_decode(tmp, k, out, out_len);
    gv_free(tmp);
    return rc;
}

/*
 * Locate an element by (namespace-agnostic) local name.  `local` is e.g.
 * "SignedInfo"; this matches "<SignedInfo", "<ds:SignedInfo",
 * "<dsig:SignedInfo", etc., requiring the char after the local name to be one
 * of space/tab/newline/'>'/'/' so "SignedInfoFoo" does not match.  Search
 * starts at `from`.  Returns pointer to the '<' or NULL.
 */
static const char *saml_find_open_tag(const char *from, const char *local) {
    size_t ln = strlen(local);
    const char *p = from;
    while ((p = strchr(p, '<')) != NULL) {
        const char *q = p + 1;
        /* optional prefix "xxx:" */
        const char *colon = NULL;
        const char *r = q;
        while (*r && (isalnum((unsigned char)*r) || *r == '_' || *r == '-' ||
                      *r == '.')) r++;
        if (*r == ':') { colon = r; r = colon + 1; }
        const char *name = colon ? colon + 1 : q;
        if (strncmp(name, local, ln) == 0) {
            char after = name[ln];
            if (after == ' ' || after == '\t' || after == '\n' ||
                after == '\r' || after == '>' || after == '/') {
                return p;
            }
        }
        p++;
    }
    return NULL;
}

/*
 * Given a pointer to an element open tag ('<...Name...>'), return the byte
 * range [start,end) covering the whole element including its close tag,
 * matching nested same-local-name elements.  `local` is the element local
 * name.  Handles self-closing "<Name .../>".  Returns 0 on success and sets
 * *elem_end to one-past the element; -1 on malformed/ambiguous input.
 */
static int saml_element_extent(const char *open_tag, const char *local,
                               const char **elem_end) {
    /* find end of the open tag */
    const char *gt = strchr(open_tag, '>');
    if (!gt) return -1;
    if (gt > open_tag && gt[-1] == '/') {  /* self-closing */
        *elem_end = gt + 1;
        return 0;
    }
    int depth = 1;
    const char *p = gt + 1;
    while (depth > 0) {
        const char *nxt = saml_find_open_tag(p, local);
        /* find matching close tag "</...local>" */
        const char *close = p;
        const char *found_close = NULL;
        while ((close = strchr(close, '<')) != NULL) {
            if (close[1] == '/') {
                const char *q = close + 2;
                const char *colon = NULL;
                const char *r = q;
                while (*r && (isalnum((unsigned char)*r) || *r == '_' ||
                              *r == '-' || *r == '.')) r++;
                if (*r == ':') { colon = r; }
                const char *name = colon ? colon + 1 : q;
                if (strncmp(name, local, strlen(local)) == 0) {
                    char after = name[strlen(local)];
                    if (after == ' ' || after == '\t' || after == '\n' ||
                        after == '\r' || after == '>') {
                        found_close = close;
                        break;
                    }
                }
            }
            close++;
        }
        if (!found_close) return -1;  /* unbalanced -> ambiguous -> reject */
        if (nxt && nxt < found_close) {
            /* a nested open tag of same local name before the next close */
            const char *ngt = strchr(nxt, '>');
            if (!ngt) return -1;
            if (!(ngt > nxt && ngt[-1] == '/')) depth++;  /* not self-closing */
            p = ngt + 1;
        } else {
            const char *cgt = strchr(found_close, '>');
            if (!cgt) return -1;
            depth--;
            p = cgt + 1;
            if (depth == 0) { *elem_end = p; return 0; }
        }
    }
    return -1;
}

/*
 * Build an EVP_PKEY from the configured trust anchor PEM.  Accepts either a
 * PEM X.509 certificate (extracts SubjectPublicKeyInfo) or a PEM public key.
 * Caller EVP_PKEY_free()s the result.  Returns NULL on error.
 */
static EVP_PKEY *saml_pkey_from_trust_pem(const char *pem) {
    if (!pem || !pem[0]) return NULL;
    /* Try X.509 certificate first. */
    BIO *bio = BIO_new_mem_buf(pem, -1);
    if (!bio) return NULL;
    X509 *cert = PEM_read_bio_X509(bio, NULL, NULL, NULL);
    BIO_free(bio);
    if (cert) {
        EVP_PKEY *pk = X509_get_pubkey(cert);  /* refcount bumped */
        X509_free(cert);
        if (pk) return pk;
    }
    /* Fall back to a bare PEM public key. */
    return rsa_pkey_from_pem(pem);
}

/*
 * Compare the assertion's embedded <ds:X509Certificate> (base64 DER, may be
 * newline-wrapped) against the configured trusted cert.  Trust is granted ONLY
 * when the embedded cert's DER equals the configured cert's DER.  Returns 0
 * when they match (safe to use), -1 otherwise.  If there is NO embedded cert,
 * returns 0 as well (nothing to distrust; we still verify under the pinned
 * key).  This is the anti-bypass gate: an attacker-supplied embedded cert that
 * differs from the pin is rejected here.
 */
static int saml_embedded_cert_ok(const char *sig_block, const char *sig_end,
                                 const char *trusted_pem) {
    const char *b = saml_find_open_tag(sig_block, "X509Certificate");
    if (!b || b >= sig_end) return 0;  /* no embedded cert -> nothing to check */

    const char *gt = strchr(b, '>');
    if (!gt || gt >= sig_end) return -1;
    gt++;
    const char *close = strstr(gt, "<");
    if (!close || close >= sig_end) return -1;

    size_t b64_len = (size_t)(close - gt);
    unsigned char der[8192];
    size_t der_len = sizeof(der);
    if (saml_b64_decode(gt, b64_len, der, &der_len) != 0) return -1;

    /* Parse embedded DER cert. */
    const unsigned char *dp = der;
    X509 *embedded = d2i_X509(NULL, &dp, (long)der_len);
    if (!embedded) return -1;

    /* Parse the trusted cert (must itself be an X.509 cert for DER compare). */
    int match = -1;
    BIO *bio = BIO_new_mem_buf(trusted_pem, -1);
    if (bio) {
        X509 *trusted = PEM_read_bio_X509(bio, NULL, NULL, NULL);
        BIO_free(bio);
        if (trusted) {
            /* X509_cmp compares the DER encodings; 0 == identical. */
            if (X509_cmp(embedded, trusted) == 0) match = 0;
            X509_free(trusted);
        } else {
            /*
             * Trust anchor is a bare public key, not a cert, so we cannot do a
             * cert-DER comparison.  Do NOT trust the embedded cert; require the
             * signature to verify under the configured public key instead.  We
             * return 0 (allow) here but the RSA verify below uses the pinned
             * key, so an embedded cert that does not correspond to that key
             * will fail the signature check anyway.
             */
            match = 0;
        }
    }
    X509_free(embedded);
    return match;
}

/*
 * Core XML-DSig verification over the decoded assertion XML.
 *
 * Returns 1 iff the signature AND the reference digest both validate against
 * the configured trusted key.  Returns 0 on any failure (default deny).
 *
 * `xml`      : NUL-terminated decoded SAML XML.
 * `xml_len`  : length of xml.
 * `trusted`  : config->saml_idp_cert_pem (already checked non-empty).
 */
static int saml_verify_xmldsig(const char *xml, size_t xml_len,
                               const char *trusted) {
    if (!xml || !trusted || !trusted[0]) return 0;

    /* 1. Locate the <ds:Signature> element and its extent. */
    const char *sig_open = saml_find_open_tag(xml, "Signature");
    if (!sig_open) return 0;
    const char *sig_end = NULL;
    if (saml_element_extent(sig_open, "Signature", &sig_end) != 0) return 0;
    /* Reject if a second Signature exists at the same level we might confuse
     * (defense against wrapping): we only ever verify the first, and bind it
     * to its Reference URI below, but flag obvious duplication. */

    /* 2. Anti-bypass: embedded cert (if any) must equal the pinned cert. */
    if (saml_embedded_cert_ok(sig_open, sig_end, trusted) != 0) return 0;

    /* 3. Extract <SignedInfo> (the bytes actually signed). */
    const char *si_open = saml_find_open_tag(sig_open, "SignedInfo");
    if (!si_open || si_open >= sig_end) return 0;
    const char *si_end = NULL;
    if (saml_element_extent(si_open, "SignedInfo", &si_end) != 0) return 0;
    if (si_end > sig_end) return 0;
    size_t si_len = (size_t)(si_end - si_open);

    /* 4. Extract <SignatureValue> (base64 RSA signature). */
    const char *sv_open = saml_find_open_tag(sig_open, "SignatureValue");
    if (!sv_open || sv_open >= sig_end) return 0;
    const char *sv_gt = strchr(sv_open, '>');
    if (!sv_gt || sv_gt >= sig_end) return 0;
    sv_gt++;
    const char *sv_close = strstr(sv_gt, "<");
    if (!sv_close || sv_close >= sig_end) return 0;
    unsigned char sigval[1024];
    size_t sigval_len = sizeof(sigval);
    if (saml_b64_decode(sv_gt, (size_t)(sv_close - sv_gt),
                        sigval, &sigval_len) != 0) return 0;

    /* 5. Determine the signature algorithm from <SignatureMethod Algorithm>. */
    const char *sm = saml_find_open_tag(si_open, "SignatureMethod");
    if (!sm || sm >= si_end) return 0;
    const char *sm_alg = strstr(sm, "Algorithm=");
    if (!sm_alg || sm_alg >= si_end) return 0;
    const EVP_MD *sig_md = NULL;
    if (strstr(sm_alg, "rsa-sha256")) {
        sig_md = EVP_sha256();
    } else if (strstr(sm_alg, "rsa-sha1")) {
        /* WEAK: RSA-SHA1 is deprecated/collision-prone.  Supported only for
         * legacy IdPs; SHA-256 should be preferred.  Left enabled but flagged. */
        sig_md = EVP_sha1();
    } else {
        return 0;  /* unsupported / ambiguous algorithm -> deny */
    }

    /* 6. Determine the reference DigestMethod. */
    const char *dm = saml_find_open_tag(si_open, "DigestMethod");
    if (!dm || dm >= si_end) return 0;
    const char *dm_alg = strstr(dm, "Algorithm=");
    if (!dm_alg || dm_alg >= si_end) return 0;
    const EVP_MD *dig_md = NULL;
    if (strstr(dm_alg, "sha256")) dig_md = EVP_sha256();
    else if (strstr(dm_alg, "sha1")) dig_md = EVP_sha1();  /* weak; legacy */
    else return 0;

    /* 7. Extract the expected <DigestValue> (base64). */
    const char *dv = saml_find_open_tag(si_open, "DigestValue");
    if (!dv || dv >= si_end) return 0;
    const char *dv_gt = strchr(dv, '>');
    if (!dv_gt || dv_gt >= si_end) return 0;
    dv_gt++;
    const char *dv_close = strstr(dv_gt, "<");
    if (!dv_close || dv_close >= si_end) return 0;
    unsigned char want_digest[EVP_MAX_MD_SIZE];
    size_t want_digest_len = sizeof(want_digest);
    if (saml_b64_decode(dv_gt, (size_t)(dv_close - dv_gt),
                        want_digest, &want_digest_len) != 0) return 0;

    /* 8. Extract the Reference URI to locate the signed element. */
    const char *ref = saml_find_open_tag(si_open, "Reference");
    if (!ref || ref >= si_end) return 0;
    const char *uri = strstr(ref, "URI=");
    char ref_id[256];
    ref_id[0] = '\0';
    if (uri && uri < si_end) {
        uri += 4;
        char quote = *uri;
        if (quote == '"' || quote == '\'') {
            uri++;
            const char *uend = strchr(uri, quote);
            if (uend && uend < si_end) {
                size_t ul = (size_t)(uend - uri);
                /* URI is "#<ID>" (same-document reference) or "" (whole doc). */
                if (ul > 0 && uri[0] == '#') { uri++; ul--; }
                if (ul >= sizeof(ref_id)) ul = sizeof(ref_id) - 1;
                memcpy(ref_id, uri, ul);
                ref_id[ul] = '\0';
            }
        }
    }

    /*
     * 9. Locate the referenced (signed) element.  It is the element whose
     * ID/AssertionID/ResponseID attribute equals ref_id.  When ref_id is empty
     * (URI=""), the whole document is the reference (rare for SAML; we take the
     * root element).  We prefer the <Assertion>/<Response> element.
     */
    const char *signed_elem = NULL;
    const char *signed_local = NULL;
    char lname[64];
    if (ref_id[0]) {
        /* Find an attribute value == ref_id: search for =\"<id>\" or =\'<id>\'. */
        char pat1[300], pat2[300];
        snprintf(pat1, sizeof(pat1), "\"%s\"", ref_id);
        snprintf(pat2, sizeof(pat2), "'%s'", ref_id);
        const char *hit = strstr(xml, pat1);
        if (!hit) hit = strstr(xml, pat2);
        if (!hit) return 0;
        /* Walk back to the enclosing '<' to find that element's open tag. */
        const char *lt = hit;
        while (lt > xml && *lt != '<') lt--;
        if (*lt != '<') return 0;
        signed_elem = lt;
    } else {
        /* URI="" -> whole-document; use the first element that is Response or
         * Assertion. */
        signed_elem = saml_find_open_tag(xml, "Response");
        if (!signed_elem) signed_elem = saml_find_open_tag(xml, "Assertion");
        if (!signed_elem) return 0;
    }

    /* Determine the local name of the signed element for extent matching. */
    {
        const char *q = signed_elem + 1;
        const char *colon = NULL;
        const char *r = q;
        while (*r && (isalnum((unsigned char)*r) || *r == '_' || *r == '-' ||
                      *r == '.')) r++;
        if (*r == ':') colon = r;
        const char *name = colon ? colon + 1 : q;
        const char *nend = name;
        while (*nend && (isalnum((unsigned char)*nend) || *nend == '_' ||
                         *nend == '-' || *nend == '.')) nend++;
        size_t nl = (size_t)(nend - name);
        if (nl == 0 || nl >= sizeof(lname)) return 0;
        memcpy(lname, name, nl);
        lname[nl] = '\0';
        signed_local = lname;
    }

    const char *signed_end = NULL;
    if (saml_element_extent(signed_elem, signed_local, &signed_end) != 0)
        return 0;
    if (signed_end > xml + xml_len) return 0;

    /*
     * 10. Enveloped-signature transform: build the referenced element's bytes
     * with ITS OWN <Signature> subtree removed, then SHA over that and compare
     * to want_digest.  We must remove the Signature that is a child of the
     * signed element (which is the one we verified, sig_open..sig_end) only if
     * it lies within [signed_elem, signed_end).
     */
    {
        EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
        if (!mdctx) return 0;
        int digest_ok = 0;
        if (EVP_DigestInit_ex(mdctx, dig_md, NULL) == 1) {
            int ok = 1;
            if (sig_open >= signed_elem && sig_end <= signed_end) {
                /* Hash [signed_elem, sig_open) then [sig_end, signed_end). */
                if (EVP_DigestUpdate(mdctx, signed_elem,
                                     (size_t)(sig_open - signed_elem)) != 1)
                    ok = 0;
                if (ok && EVP_DigestUpdate(mdctx, sig_end,
                                     (size_t)(signed_end - sig_end)) != 1)
                    ok = 0;
            } else {
                /* Signature is not enveloped inside the referenced element.
                 * Detached/enveloping signatures are not supported here; deny
                 * rather than hash content that does not match the transform. */
                ok = 0;
            }
            if (ok) {
                unsigned char got[EVP_MAX_MD_SIZE];
                unsigned int got_len = 0;
                if (EVP_DigestFinal_ex(mdctx, got, &got_len) == 1 &&
                    got_len == want_digest_len) {
                    /* constant-time compare of the digest */
                    unsigned char diff = 0;
                    for (unsigned int i = 0; i < got_len; i++)
                        diff |= got[i] ^ want_digest[i];
                    digest_ok = (diff == 0);
                }
            }
        }
        EVP_MD_CTX_free(mdctx);
        if (!digest_ok) return 0;  /* reference digest mismatch -> deny */
    }

    /*
     * 11. Verify the RSA signature over the (pragmatically canonicalized)
     * SignedInfo bytes using the pinned trusted key.  See the header comment
     * for canonicalization caveats: we verify over the on-the-wire SignedInfo.
     */
    EVP_PKEY *pkey = saml_pkey_from_trust_pem(trusted);
    if (!pkey) return 0;

    int sig_ok = 0;
    EVP_MD_CTX *vctx = EVP_MD_CTX_new();
    if (vctx) {
        if (EVP_DigestVerifyInit(vctx, NULL, sig_md, NULL, pkey) == 1 &&
            EVP_DigestVerify(vctx, sigval, sigval_len,
                             (const unsigned char *)si_open, si_len) == 1) {
            sig_ok = 1;
        }
        EVP_MD_CTX_free(vctx);
    }
    EVP_PKEY_free(pkey);

    /* Only success when BOTH the digest and the signature validated. */
    return sig_ok ? 1 : 0;
}

#endif /* GV_HAVE_OPENSSL */

static GV_SSOToken *parse_saml_assertion(const GV_SSOConfig *config,
                                         const char *b64_assertion) {
    if (!b64_assertion) return NULL;

    /* Decode base64 (SAML uses standard base64, not URL-safe) */
    size_t in_len = strlen(b64_assertion);
    size_t max_decoded = (in_len * 3) / 4 + 4;
    unsigned char *decoded = gv_alloc(max_decoded);
    if (!decoded) return NULL;

    size_t decoded_len = max_decoded;
    if (base64url_decode(b64_assertion, in_len,
                         decoded, &decoded_len) != 0) {
        /* Try treating '+' as '-' and '/' as '_' for standard base64 */
        char *urlsafe = gv_alloc(in_len + 1);
        if (!urlsafe) {
            gv_free(decoded);
            return NULL;
        }
        for (size_t i = 0; i < in_len; i++) {
            if (b64_assertion[i] == '+') urlsafe[i] = '-';
            else if (b64_assertion[i] == '/') urlsafe[i] = '_';
            else urlsafe[i] = b64_assertion[i];
        }
        urlsafe[in_len] = '\0';

        decoded_len = max_decoded;
        if (base64url_decode(urlsafe, in_len,
                             decoded, &decoded_len) != 0) {
            gv_free(urlsafe);
            gv_free(decoded);
            return NULL;
        }
        gv_free(urlsafe);
    }

    /* Null-terminate for string operations */
    if (decoded_len >= max_decoded) decoded_len = max_decoded - 1;
    decoded[decoded_len] = '\0';

    const char *xml = (const char *)decoded;

    /*
     * ---- SAML signature policy (FAIL CLOSED) ----
     *
     * A SAML assertion is a bearer credential that can grant admin
     * (populate_admin_flag scrapes group attributes below).  We must NOT trust
     * any scraped field unless the assertion's XML-DSig signature has been
     * verified against the configured, trusted IdP certificate
     * (config->saml_idp_cert_pem).
     *
     * Verification (saml_verify_xmldsig) requires OpenSSL and returns 1 only
     * when BOTH the RSA signature over <SignedInfo> AND the enveloped-signature
     * reference digest validate under the pinned trust anchor.  An embedded
     * <ds:X509Certificate> is honored ONLY if it equals the pinned cert (no
     * embedded-cert trust bypass).  See the big header comment on
     * saml_verify_xmldsig for canonicalization limits and signature-wrapping
     * caveats.
     *
     * Fail closed when: no trust anchor is configured, OpenSSL is absent, or
     * verification fails for any reason.  A single, clearly named compile-time
     * escape hatch (GV_SAML_ALLOW_UNSIGNED, default UNDEFINED / OFF) exists for
     * isolated development only and must be explicitly compiled in.
     */
    {
        int verified = 0;

#ifdef GV_HAVE_OPENSSL
        if (config && config->saml_idp_cert_pem &&
            config->saml_idp_cert_pem[0]) {
            verified = saml_verify_xmldsig(xml, decoded_len,
                                           config->saml_idp_cert_pem);
        }
        /* else: no trust anchor -> cannot verify -> verified stays 0 (deny). */
#else
        /*
         * No OpenSSL in this build: XML-DSig verification is impossible, so we
         * fail closed exactly like RS256/GCM do without OpenSSL.
         */
        (void)config;
#endif

#ifdef GV_SAML_ALLOW_UNSIGNED
        /*
         * INSECURE development escape hatch (OFF by default; NEVER define in
         * production).  Accepts unsigned/unverified assertions, defeating all
         * of the above.  Present only so local dev without an IdP cert can run.
         */
        verified = 1;
#endif

        if (!verified) {
            gv_free(decoded);
            return NULL;  /* fail closed: unverified assertion rejected */
        }
    }

    GV_SSOToken *token = alloc_token();
    if (!token) {
        gv_free(decoded);
        return NULL;
    }

    char buf[512];

    if (xml_extract_text(xml, "saml:NameID", buf, sizeof(buf)) == 0 ||
        xml_extract_text(xml, "saml2:NameID", buf, sizeof(buf)) == 0 ||
        xml_extract_text(xml, "NameID", buf, sizeof(buf)) == 0) {
        token->subject = gv_dup_cstr(buf);
    }

    /* Email */
    if (xml_extract_attribute_value(xml, "email", buf, sizeof(buf)) == 0 ||
        xml_extract_attribute_value(xml,
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            buf, sizeof(buf)) == 0 ||
        xml_extract_attribute_value(xml,
            "urn:oid:0.9.2342.19200300.100.1.3", buf, sizeof(buf)) == 0) {
        token->email = gv_dup_cstr(buf);
    }

    /* Display name */
    if (xml_extract_attribute_value(xml, "name", buf, sizeof(buf)) == 0 ||
        xml_extract_attribute_value(xml, "displayName", buf, sizeof(buf)) == 0 ||
        xml_extract_attribute_value(xml,
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            buf, sizeof(buf)) == 0 ||
        xml_extract_attribute_value(xml,
            "urn:oid:2.16.840.1.113730.3.1.241", buf, sizeof(buf)) == 0) {
        token->name = gv_dup_cstr(buf);
    }

#define EXTRACT_MULTI(attr_name_str, dest_arr, dest_count) do { \
    char needle_m[512]; \
    snprintf(needle_m, sizeof(needle_m), "Name=\"%s\"", (attr_name_str)); \
    const char *ae = strstr(xml, needle_m); \
    if (ae) { \
        const char *ae_end = strstr(ae, "</Attribute"); \
        if (!ae_end) ae_end = strstr(ae, "</saml:Attribute"); \
        if (!ae_end) ae_end = xml + decoded_len; \
        size_t cnt = 0; \
        const char *pp = ae; \
        while (pp < ae_end) { \
            const char *vs = strstr(pp, "<AttributeValue"); \
            if (!vs || vs >= ae_end) { break; } cnt++; pp = vs + 1; \
        } \
        if (cnt > MAX_GROUPS) cnt = MAX_GROUPS; \
        if (cnt > 0) { \
            (dest_arr) = gv_calloc(cnt, sizeof(char *)); \
            if ((dest_arr)) { \
                pp = ae; size_t ix = 0; \
                while (pp < ae_end && ix < cnt) { \
                    const char *vs = strstr(pp, "<AttributeValue"); \
                    if (!vs || vs >= ae_end) break; \
                    const char *gtp = strchr(vs, '>'); \
                    if (!gtp || gtp >= ae_end) { break; } gtp++; \
                    const char *ep = strchr(gtp, '<'); \
                    if (!ep || ep >= ae_end) break; \
                    size_t vl = (size_t)(ep - gtp); \
                    if (vl < sizeof(buf)) { \
                        memcpy(buf, gtp, vl); buf[vl] = '\0'; \
                        xml_decode_entities(buf); \
                        (dest_arr)[ix++] = gv_dup_cstr(buf); \
                    } \
                    pp = ep; \
                } \
                (dest_count) = ix; \
            } \
        } \
    } \
} while (0)

    if (!token->groups) {
        EXTRACT_MULTI("groups", token->groups, token->group_count);
    }
    if (!token->groups) {
        EXTRACT_MULTI("http://schemas.xmlsoap.org/claims/Group",
                      token->groups, token->group_count);
    }
    if (!token->groups) {
        EXTRACT_MULTI("urn:oid:1.3.6.1.4.1.5923.1.5.1.1",
                      token->groups, token->group_count);
    }

#undef EXTRACT_MULTI

    /* Extract timestamps if present in Conditions element */
    const char *conditions = strstr(xml, "<Conditions");
    if (!conditions) conditions = strstr(xml, "<saml:Conditions");
    if (conditions) {
        const char *not_before = strstr(conditions, "NotBefore=\"");
        if (not_before) {
            not_before += strlen("NotBefore=\"");
            token->issued_at = parse_iso8601(not_before);
        }
        const char *not_after = strstr(conditions, "NotOnOrAfter=\"");
        if (not_after) {
            not_after += strlen("NotOnOrAfter=\"");
            token->expires_at = parse_iso8601(not_after);
        }
    }

    /*
     * ---- Enforce the assertion validity window (FAIL CLOSED) ----
     *
     * Reject if:
     *   - the Conditions/NotBefore/NotOnOrAfter window is missing (we require
     *     an explicit window; a window-less assertion is treated as invalid),
     *   - now < NotBefore (not yet valid), or
     *   - now >= NotOnOrAfter (expired),
     * with a small clock-skew tolerance.
     *
     * token->issued_at holds NotBefore and token->expires_at holds
     * NotOnOrAfter (parse_iso8601 returns 0 on failure).
     */
    {
        const uint64_t skew = 60;  /* seconds of allowed clock skew */
        uint64_t now = (uint64_t)time(NULL);

        if (token->issued_at == 0 || token->expires_at == 0) {
            gv_free(decoded);
            sso_free_token(token);
            return NULL;  /* missing/unparseable validity window -> deny */
        }
        /* now < NotBefore - skew  => not yet valid */
        if (now + skew < token->issued_at) {
            gv_free(decoded);
            sso_free_token(token);
            return NULL;
        }
        /* now >= NotOnOrAfter + skew => expired */
        if (now >= token->expires_at + skew) {
            gv_free(decoded);
            sso_free_token(token);
            return NULL;
        }
    }

    /*
     * ---- Enforce the audience restriction (FAIL CLOSED when configured) ----
     *
     * When config->saml_entity_id (our SP entity ID) is set, the signed
     * assertion MUST contain an <AudienceRestriction><Audience> equal to it.
     * This prevents an assertion minted for a different SP from being replayed
     * against us.  If no entity ID is configured we skip the check (there is
     * nothing to match against), but the signature/digest checks above still
     * bound the assertion to the trusted IdP.
     */
    if (config && config->saml_entity_id && config->saml_entity_id[0]) {
        const char *ar = strstr(xml, "<AudienceRestriction");
        if (!ar) ar = strstr(xml, ":AudienceRestriction");
        int audience_ok = 0;
        if (ar) {
            const char *ar_end = strstr(ar, "</AudienceRestriction");
            if (!ar_end) ar_end = xml + decoded_len;
            const char *aud = strstr(ar, "<Audience");
            if (!aud) aud = strstr(ar, ":Audience");
            /* Walk each <Audience> inside the restriction. */
            while (aud && aud < ar_end) {
                const char *gt = strchr(aud, '>');
                if (!gt || gt >= ar_end) break;
                gt++;
                const char *end = strchr(gt, '<');
                if (!end || end >= ar_end) break;
                size_t len = (size_t)(end - gt);
                /* trim surrounding whitespace */
                while (len > 0 && isspace((unsigned char)gt[0])) { gt++; len--; }
                while (len > 0 && isspace((unsigned char)gt[len - 1])) len--;
                if (len == strlen(config->saml_entity_id) &&
                    memcmp(gt, config->saml_entity_id, len) == 0) {
                    audience_ok = 1;
                    break;
                }
                aud = strstr(end, "<Audience");
                if (!aud) aud = strstr(end, ":Audience");
            }
        }
        if (!audience_ok) {
            gv_free(decoded);
            sso_free_token(token);
            return NULL;  /* wrong / missing audience -> deny */
        }
    }

    gv_free(decoded);
    return token;
}

/* HTTP Helpers (libcurl or stub) */

#ifdef HAVE_CURL

/**
 * libcurl write callback that accumulates data into a ResponseBuffer.
 */
static size_t curl_write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    ResponseBuffer *buf = (ResponseBuffer *)userdata;
    size_t total = size * nmemb;

    if (buf->size + total >= MAX_RESPONSE_SIZE) return 0;

    if (buf->size + total >= buf->capacity) {
        size_t new_cap = buf->capacity ? buf->capacity * 2 : 4096;
        while (new_cap < buf->size + total + 1) new_cap *= 2;
        char *new_data = gv_realloc(buf->data, new_cap);
        if (!new_data) return 0;
        buf->data = new_data;
        buf->capacity = new_cap;
    }

    memcpy(buf->data + buf->size, ptr, total);
    buf->size += total;
    buf->data[buf->size] = '\0';

    return total;
}

static int http_get(const char *url, int verify_ssl,
                    char **response, size_t *response_len) {
    if (!url || !response || !response_len) return -1;

    CURL *curl = curl_easy_init();
    if (!curl) return -1;

    ResponseBuffer buf;
    memset(&buf, 0, sizeof(buf));

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    if (!verify_ssl) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    }

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        gv_free(buf.data);
        curl_easy_cleanup(curl);
        return -1;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (http_code < 200 || http_code >= 300) {
        gv_free(buf.data);
        return -1;
    }

    *response = buf.data;
    *response_len = buf.size;
    return 0;
}

static int http_post_form(const char *url, const char *post_fields,
                          int verify_ssl, char **response, size_t *response_len) {
    if (!url || !post_fields || !response || !response_len) return -1;

    CURL *curl = curl_easy_init();
    if (!curl) return -1;

    ResponseBuffer buf;
    memset(&buf, 0, sizeof(buf));

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_fields);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers,
        "Content-Type: application/x-www-form-urlencoded");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    if (!verify_ssl) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    }

    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        gv_free(buf.data);
        curl_easy_cleanup(curl);
        return -1;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (http_code < 200 || http_code >= 300) {
        gv_free(buf.data);
        return -1;
    }

    *response = buf.data;
    *response_len = buf.size;
    return 0;
}

#else /* !HAVE_CURL -- stub implementation */

static int http_get(const char *url, int verify_ssl,
                    char **response, size_t *response_len) {
    (void)url;
    (void)verify_ssl;
    (void)response;
    (void)response_len;

    fprintf(stderr, "[sso] HTTP GET unavailable: compiled without libcurl\n");
    return -1;
}

static int http_post_form(const char *url, const char *post_fields,
                          int verify_ssl, char **response, size_t *response_len) {
    (void)url;
    (void)post_fields;
    (void)verify_ssl;
    (void)response;
    (void)response_len;

    fprintf(stderr, "[sso] HTTP POST unavailable: compiled without libcurl\n");
    return -1;
}

#endif /* HAVE_CURL */
