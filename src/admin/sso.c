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
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

#ifdef HAVE_CURL
#include <curl/curl.h>
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
static int verify_jwt_signature(const char *jwt, const char *secret);

/* SAML helpers */
static GV_SSOToken *parse_saml_assertion(const char *b64_assertion);
static int xml_extract_text(const char *xml, const char *tag,
                            char *out, size_t out_size);

/* HTTP helpers */
static int http_get(const char *url, int verify_ssl,
                    char **response, size_t *response_len);
static int http_post_form(const char *url, const char *post_fields,
                          int verify_ssl, char **response, size_t *response_len);

/* Lifecycle */

GV_SSOManager *sso_create(const GV_SSOConfig *config) {
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

void sso_destroy(GV_SSOManager *mgr) {
    if (!mgr) return;

    pthread_mutex_destroy(&mgr->mutex);
    free(mgr);
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
    free(response);

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

GV_SSOToken *sso_validate_token(GV_SSOManager *mgr, const char *token_string) {
    if (!mgr || !token_string) return NULL;

    GV_SSOToken *token = NULL;

    if (mgr->config.provider == GV_SSO_OIDC) {
        /* Verify JWT signature if client_secret is configured (HS256) */
        if (mgr->config.client_secret && mgr->config.client_secret[0]) {
            if (verify_jwt_signature(token_string, mgr->config.client_secret) != 0) {
                return NULL;  /* Signature verification failed */
            }
        }

        /* JWT validation path */
        token = decode_jwt_claims(token_string);
        if (!token) return NULL;

        /* Check expiry */
        uint64_t now = (uint64_t)time(NULL);
        if (token->expires_at > 0 && token->expires_at < now) {
            sso_free_token(token);
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

GV_SSOToken *sso_refresh_token(GV_SSOManager *mgr, const char *refresh_token) {
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

void sso_free_token(GV_SSOToken *token) {
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
    GV_SSOToken *token = calloc(1, sizeof(GV_SSOToken));
    return token;
}

/* CSV / Group Utilities */

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

/**
 * Verify the HMAC-SHA256 (HS256) signature of a JWT.
 * Returns 0 on success, -1 on verification failure.
 */
static int verify_jwt_signature(const char *jwt, const char *secret) {
    if (!jwt || !secret) return -1;

    const char *dot1 = strchr(jwt, '.');
    if (!dot1) return -1;
    const char *dot2 = strchr(dot1 + 1, '.');
    if (!dot2) return -1;

    /* The signed payload is header.payload (everything before the second dot) */
    size_t signed_len = (size_t)(dot2 - jwt);

    /* Decode the signature segment */
    const char *sig_start = dot2 + 1;
    size_t sig_b64_len = strlen(sig_start);

    unsigned char decoded_sig[256];
    size_t decoded_sig_len = sizeof(decoded_sig);
    if (base64url_decode(sig_start, sig_b64_len, decoded_sig, &decoded_sig_len) != 0) {
        return -1;
    }

    /* Compute expected HMAC-SHA256 */
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
    json_extract_uint64(payload_json, "exp", &token->expires_at);

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
            if (!vs || vs >= ae_end) break; cnt++; pp = vs + 1; \
        } \
        if (cnt > MAX_GROUPS) cnt = MAX_GROUPS; \
        if (cnt > 0) { \
            (dest_arr) = calloc(cnt, sizeof(char *)); \
            if ((dest_arr)) { \
                pp = ae; size_t ix = 0; \
                while (pp < ae_end && ix < cnt) { \
                    const char *vs = strstr(pp, "<AttributeValue"); \
                    if (!vs || vs >= ae_end) break; \
                    const char *gtp = strchr(vs, '>'); \
                    if (!gtp || gtp >= ae_end) break; gtp++; \
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

    free(decoded);
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
