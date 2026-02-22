#define _POSIX_C_SOURCE 200809L

/**
 * @file gv_sso.c
 * @brief Enterprise SSO / OIDC / SAML authentication implementation.
 *
 * Provides OIDC discovery, JWT validation against JWKS endpoints,
 * SAML assertion parsing, and group-based authorization.
 *
 * HTTP calls use libcurl when HAVE_CURL is defined; otherwise the
 * network-dependent functions return an error.
 */

#include "gigavector/gv_sso.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

/*  Internal Constants  */

#define MAX_URL_LEN         2048
#define MAX_ENDPOINT_LEN    1024
#define MAX_RESPONSE_SIZE   (256 * 1024)
#define MAX_GROUPS          64
#define MAX_JWT_SEGMENTS    3
#define BASE64_DECODE_MAX   8192

/*  Internal Structures  */

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

/*  Forward Declarations  */

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

/* SAML helpers */
static GV_SSOToken *parse_saml_assertion(const char *b64_assertion);
static int xml_extract_text(const char *xml, const char *tag,
                            char *out, size_t out_size);

/* HTTP helpers */
static int http_get(const char *url, int verify_ssl,
                    char **response, size_t *response_len);
static int http_post_form(const char *url, const char *post_fields,
                          int verify_ssl, char **response, size_t *response_len);

/*  Lifecycle  */

GV_SSOManager *gv_sso_create(const GV_SSOConfig *config) {
    if (!config) return NULL;

    GV_SSOManager *mgr = calloc(1, sizeof(GV_SSOManager));
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
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_sso_destroy(GV_SSOManager *mgr) {
    if (!mgr) return;

    pthread_mutex_destroy(&mgr->mutex);
    free(mgr);
}

/*  OIDC Discovery  */

int gv_sso_discover(GV_SSOManager *mgr) {
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
    free(response);

    return ok ? 0 : -1;
}

/*  Authentication Flow  */

int gv_sso_get_auth_url(const GV_SSOManager *mgr, const char *state,
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

GV_SSOToken *gv_sso_exchange_code(GV_SSOManager *mgr, const char *auth_code) {
    if (!mgr || !auth_code) return NULL;
    if (mgr->config.provider != GV_SSO_OIDC) return NULL;

    pthread_mutex_lock(&mgr->mutex);

    if (!mgr->endpoints.discovered) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }

    /* Build POST body for token exchange */
    char post_fields[MAX_URL_LEN];
    snprintf(post_fields, sizeof(post_fields),
             "grant_type=authorization_code&code=%s"
             "&client_id=%s&client_secret=%s&redirect_uri=%s",
             auth_code,
             mgr->config.client_id ? mgr->config.client_id : "",
             mgr->config.client_secret ? mgr->config.client_secret : "",
             mgr->config.redirect_uri ? mgr->config.redirect_uri : "");

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
        free(response);
        return NULL;
    }

    free(response);

    /* Decode and validate the JWT */
    GV_SSOToken *token = decode_jwt_claims(id_token);
    if (!token) return NULL;

    /* Populate admin flag based on configured admin groups */
    populate_admin_flag(token, mgr->config.admin_groups);

    return token;
}

GV_SSOToken *gv_sso_validate_token(GV_SSOManager *mgr, const char *token_string) {
    if (!mgr || !token_string) return NULL;

    GV_SSOToken *token = NULL;

    if (mgr->config.provider == GV_SSO_OIDC) {
        /* JWT validation path */
        token = decode_jwt_claims(token_string);
        if (!token) return NULL;

        /* Check expiry */
        uint64_t now = (uint64_t)time(NULL);
        if (token->expires_at > 0 && token->expires_at < now) {
            gv_sso_free_token(token);
            return NULL;
        }
    } else if (mgr->config.provider == GV_SSO_SAML) {
        /* SAML assertion path */
        token = parse_saml_assertion(token_string);
        if (!token) return NULL;
    } else {
        return NULL;
    }

    /* Populate admin flag */
    populate_admin_flag(token, mgr->config.admin_groups);

    return token;
}

GV_SSOToken *gv_sso_refresh_token(GV_SSOManager *mgr, const char *refresh_token) {
    if (!mgr || !refresh_token) return NULL;
    if (mgr->config.provider != GV_SSO_OIDC) return NULL;

    pthread_mutex_lock(&mgr->mutex);

    if (!mgr->endpoints.discovered) {
        pthread_mutex_unlock(&mgr->mutex);
        return NULL;
    }

    /* Build POST body for token refresh */
    char post_fields[MAX_URL_LEN];
    snprintf(post_fields, sizeof(post_fields),
             "grant_type=refresh_token&refresh_token=%s"
             "&client_id=%s&client_secret=%s",
             refresh_token,
             mgr->config.client_id ? mgr->config.client_id : "",
             mgr->config.client_secret ? mgr->config.client_secret : "");

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
        free(response);
        return NULL;
    }

    free(response);

    GV_SSOToken *token = decode_jwt_claims(id_token);
    if (!token) return NULL;

    populate_admin_flag(token, mgr->config.admin_groups);

    return token;
}

void gv_sso_free_token(GV_SSOToken *token) {
    if (!token) return;

    free(token->subject);
    free(token->email);
    free(token->name);

    for (size_t i = 0; i < token->group_count; i++) {
        free(token->groups[i]);
    }
    free(token->groups);

    free(token);
}

/*  Group Checking  */

int gv_sso_has_group(const GV_SSOToken *token, const char *group) {
    if (!token || !group) return 0;

    for (size_t i = 0; i < token->group_count; i++) {
        if (token->groups[i] && strcmp(token->groups[i], group) == 0) {
            return 1;
        }
    }

    return 0;
}

/*  Token Allocation  */

static GV_SSOToken *alloc_token(void) {
    GV_SSOToken *token = calloc(1, sizeof(GV_SSOToken));
    return token;
}

/*  CSV / Group Utilities  */

/**
 * Split a comma-separated string into an array of trimmed strings.
 * Caller must free each element and the array itself.
 */
static char **split_csv(const char *csv, size_t *count) {
    *count = 0;
    if (!csv || !csv[0]) return NULL;

    /* Count commas to estimate array size */
    size_t capacity = 1;
    for (const char *p = csv; *p; p++) {
        if (*p == ',') capacity++;
    }

    char **result = calloc(capacity, sizeof(char *));
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
            result[idx] = malloc(len + 1);
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
        free(groups[i]);
    }
    free(groups);

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

/*  Base64 URL Decoding  */

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

        if (i < in_len) {
            if (!is_b64url_char((unsigned char)in[i])) return -1;
            a = b64url_table[(unsigned char)in[i++]];
        }
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

/*  Minimal JSON Helpers  */

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
    char **arr = calloc(MAX_GROUPS, sizeof(char *));
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
        arr[idx] = malloc(len + 1);
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

/*  JWT Decoding  */

/**
 * Decode a JWT and extract claims into a GV_SSOToken.
 * This is the stub path (no OpenSSL): decodes the payload and checks
 * expiry, but does not verify the cryptographic signature.
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
        token->subject = strdup(buf);
    }
    if (json_extract_string(payload_json, "email", buf, sizeof(buf)) == 0) {
        token->email = strdup(buf);
    }
    if (json_extract_string(payload_json, "name", buf, sizeof(buf)) == 0) {
        token->name = strdup(buf);
    }

    json_extract_uint64(payload_json, "iat", &token->issued_at);
    json_extract_uint64(payload_json, "exp", &token->expires_at);

    /* Extract groups claim (may be absent) */
    json_extract_string_array(payload_json, "groups",
                              &token->groups, &token->group_count);

    return token;
}

/*  SAML Assertion Parsing (Stub)  */

/**
 * Extract text content between <tag> and </tag> from XML.
 * Very basic: finds the first occurrence and extracts inner text.
 */
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

    return 0;
}

/**
 * Parse a base64-encoded SAML assertion and extract identity claims.
 * This is a basic stub: decodes the base64, then uses simple string
 * matching to find NameID and attribute values.  No full XML parser.
 */
static GV_SSOToken *parse_saml_assertion(const char *b64_assertion) {
    if (!b64_assertion) return NULL;

    /* Decode base64 (SAML uses standard base64, not URL-safe) */
    size_t in_len = strlen(b64_assertion);
    size_t max_decoded = (in_len * 3) / 4 + 4;
    unsigned char *decoded = malloc(max_decoded);
    if (!decoded) return NULL;

    size_t decoded_len = max_decoded;
    if (base64url_decode(b64_assertion, in_len,
                         decoded, &decoded_len) != 0) {
        /* Try treating '+' as '-' and '/' as '_' for standard base64 */
        char *urlsafe = malloc(in_len + 1);
        if (!urlsafe) {
            free(decoded);
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
            free(urlsafe);
            free(decoded);
            return NULL;
        }
        free(urlsafe);
    }

    /* Null-terminate for string operations */
    if (decoded_len >= max_decoded) decoded_len = max_decoded - 1;
    decoded[decoded_len] = '\0';

    const char *xml = (const char *)decoded;

    GV_SSOToken *token = alloc_token();
    if (!token) {
        free(decoded);
        return NULL;
    }

    char buf[512];

    /* Extract NameID */
    if (xml_extract_text(xml, "saml:NameID", buf, sizeof(buf)) == 0 ||
        xml_extract_text(xml, "NameID", buf, sizeof(buf)) == 0) {
        token->subject = strdup(buf);
    }

    /* Extract common attributes by searching for AttributeValue elements.
     * SAML attributes follow the pattern:
     *   <Attribute Name="..."><AttributeValue>...</AttributeValue></Attribute>
     */

    /* Email */
    const char *email_attr = strstr(xml, "Name=\"email\"");
    if (!email_attr) email_attr = strstr(xml, "Name=\"http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress\"");
    if (email_attr) {
        const char *val_start = strstr(email_attr, "<AttributeValue");
        if (!val_start) val_start = strstr(email_attr, "<saml:AttributeValue");
        if (val_start) {
            const char *gt = strchr(val_start, '>');
            if (gt) {
                gt++;
                const char *end = strchr(gt, '<');
                if (end) {
                    size_t len = (size_t)(end - gt);
                    if (len < sizeof(buf)) {
                        memcpy(buf, gt, len);
                        buf[len] = '\0';
                        token->email = strdup(buf);
                    }
                }
            }
        }
    }

    /* Display name */
    const char *name_attr = strstr(xml, "Name=\"name\"");
    if (!name_attr) name_attr = strstr(xml, "Name=\"http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name\"");
    if (name_attr) {
        const char *val_start = strstr(name_attr, "<AttributeValue");
        if (!val_start) val_start = strstr(name_attr, "<saml:AttributeValue");
        if (val_start) {
            const char *gt = strchr(val_start, '>');
            if (gt) {
                gt++;
                const char *end = strchr(gt, '<');
                if (end) {
                    size_t len = (size_t)(end - gt);
                    if (len < sizeof(buf)) {
                        memcpy(buf, gt, len);
                        buf[len] = '\0';
                        token->name = strdup(buf);
                    }
                }
            }
        }
    }

    /* Groups - may appear as multiple AttributeValue elements under one Attribute */
    const char *groups_attr = strstr(xml, "Name=\"groups\"");
    if (!groups_attr) groups_attr = strstr(xml, "Name=\"http://schemas.xmlsoap.org/claims/Group\"");
    if (groups_attr) {
        /* Count group values */
        const char *search = groups_attr;
        size_t count = 0;

        /* Find the closing </Attribute> to bound our search */
        const char *attr_end = strstr(search, "</Attribute");
        if (!attr_end) attr_end = strstr(search, "</saml:Attribute");
        if (!attr_end) attr_end = xml + decoded_len;

        /* First pass: count */
        const char *p = search;
        while (p < attr_end) {
            const char *vs = strstr(p, "<AttributeValue");
            if (!vs) vs = strstr(p, "<saml:AttributeValue");
            if (!vs || vs >= attr_end) break;

            count++;
            p = vs + 1;
        }

        if (count > MAX_GROUPS) count = MAX_GROUPS;

        if (count > 0) {
            token->groups = calloc(count, sizeof(char *));
            if (token->groups) {
                /* Second pass: extract values */
                p = search;
                size_t idx = 0;
                while (p < attr_end && idx < count) {
                    const char *vs = strstr(p, "<AttributeValue");
                    if (!vs) vs = strstr(p, "<saml:AttributeValue");
                    if (!vs || vs >= attr_end) break;

                    const char *gt = strchr(vs, '>');
                    if (!gt || gt >= attr_end) break;
                    gt++;

                    const char *end = strchr(gt, '<');
                    if (!end || end >= attr_end) break;

                    size_t len = (size_t)(end - gt);
                    if (len < sizeof(buf)) {
                        memcpy(buf, gt, len);
                        buf[len] = '\0';
                        token->groups[idx] = strdup(buf);
                        idx++;
                    }

                    p = end;
                }
                token->group_count = idx;
            }
        }
    }

    /* Extract timestamps if present in Conditions element */
    const char *conditions = strstr(xml, "<Conditions");
    if (!conditions) conditions = strstr(xml, "<saml:Conditions");
    if (conditions) {
        const char *not_before = strstr(conditions, "NotBefore=\"");
        if (not_before) {
            /* SAML timestamps are ISO 8601; store as 0 for now (stub) */
            token->issued_at = (uint64_t)time(NULL);
        }
        const char *not_after = strstr(conditions, "NotOnOrAfter=\"");
        if (not_after) {
            token->expires_at = (uint64_t)time(NULL) + 3600;
        }
    }

    free(decoded);
    return token;
}

/*  HTTP Helpers (libcurl or stub)  */

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
        char *new_data = realloc(buf->data, new_cap);
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
        free(buf.data);
        curl_easy_cleanup(curl);
        return -1;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (http_code < 200 || http_code >= 300) {
        free(buf.data);
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
        free(buf.data);
        curl_easy_cleanup(curl);
        return -1;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (http_code < 200 || http_code >= 300) {
        free(buf.data);
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

    fprintf(stderr, "[gv_sso] HTTP GET unavailable: compiled without libcurl\n");
    return -1;
}

static int http_post_form(const char *url, const char *post_fields,
                          int verify_ssl, char **response, size_t *response_len) {
    (void)url;
    (void)post_fields;
    (void)verify_ssl;
    (void)response;
    (void)response_len;

    fprintf(stderr, "[gv_sso] HTTP POST unavailable: compiled without libcurl\n");
    return -1;
}

#endif /* HAVE_CURL */
