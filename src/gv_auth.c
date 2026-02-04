/**
 * @file gv_auth.c
 * @brief Authentication implementation.
 */

#include "gigavector/gv_auth.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

#define MAX_API_KEYS 256
#define KEY_ID_LEN 16
#define KEY_LEN 32
#define HASH_LEN 32

/**
 * @brief API key entry.
 */
typedef struct {
    char key_id[KEY_ID_LEN * 2 + 1];
    char key_hash[HASH_LEN * 2 + 1];
    char *description;
    uint64_t created_at;
    uint64_t expires_at;
    int enabled;
} APIKeyEntry;

/**
 * @brief Auth manager internal structure.
 */
struct GV_AuthManager {
    GV_AuthConfig config;
    APIKeyEntry keys[MAX_API_KEYS];
    size_t key_count;
    pthread_rwlock_t rwlock;
};

/* ============================================================================
 * Simple SHA-256 Implementation (for portability)
 * ============================================================================ */

/* Minimal SHA-256 - in production, use OpenSSL or similar */

#define SHA256_BLOCK_SIZE 32

typedef struct {
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
} SHA256_CTX;

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

static const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha256_init(SHA256_CTX *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

static void sha256_transform(SHA256_CTX *ctx, const uint8_t data[]) {
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j + 1] << 16) |
               ((uint32_t)data[j + 2] << 8) | ((uint32_t)data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_update(SHA256_CTX *ctx, const uint8_t data[], size_t len) {
    uint32_t i;
    for (i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

static void sha256_final(SHA256_CTX *ctx, uint8_t hash[]) {
    uint32_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = (uint8_t)(ctx->bitlen);
    ctx->data[62] = (uint8_t)(ctx->bitlen >> 8);
    ctx->data[61] = (uint8_t)(ctx->bitlen >> 16);
    ctx->data[60] = (uint8_t)(ctx->bitlen >> 24);
    ctx->data[59] = (uint8_t)(ctx->bitlen >> 32);
    ctx->data[58] = (uint8_t)(ctx->bitlen >> 40);
    ctx->data[57] = (uint8_t)(ctx->bitlen >> 48);
    ctx->data[56] = (uint8_t)(ctx->bitlen >> 56);
    sha256_transform(ctx, ctx->data);

    for (i = 0; i < 4; ++i) {
        hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
    }
}

/* ============================================================================
 * Hashing Utilities
 * ============================================================================ */

int gv_auth_sha256(const void *data, size_t len, unsigned char *hash_out) {
    if (!data || !hash_out) return -1;

    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)data, len);
    sha256_final(&ctx, hash_out);

    return 0;
}

void gv_auth_to_hex(const unsigned char *hash, size_t hash_len, char *hex_out) {
    static const char hex_chars[] = "0123456789abcdef";
    for (size_t i = 0; i < hash_len; i++) {
        hex_out[i * 2] = hex_chars[(hash[i] >> 4) & 0x0f];
        hex_out[i * 2 + 1] = hex_chars[hash[i] & 0x0f];
    }
    hex_out[hash_len * 2] = '\0';
}

/* ============================================================================
 * Random Generation
 * ============================================================================ */

static void generate_random_bytes(unsigned char *buf, size_t len) {
    FILE *fp = fopen("/dev/urandom", "rb");
    if (fp) {
        size_t read = fread(buf, 1, len, fp);
        fclose(fp);
        if (read == len) return;
    }
    /* Fallback to weak random */
    for (size_t i = 0; i < len; i++) {
        buf[i] = (unsigned char)(rand() & 0xff);
    }
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_AuthConfig DEFAULT_CONFIG = {
    .type = GV_AUTH_NONE,
    .jwt = {
        .secret = NULL,
        .secret_len = 0,
        .issuer = NULL,
        .audience = NULL,
        .clock_skew_seconds = 60
    }
};

void gv_auth_config_init(GV_AuthConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_AuthManager *gv_auth_create(const GV_AuthConfig *config) {
    GV_AuthManager *auth = calloc(1, sizeof(GV_AuthManager));
    if (!auth) return NULL;

    auth->config = config ? *config : DEFAULT_CONFIG;

    if (pthread_rwlock_init(&auth->rwlock, NULL) != 0) {
        free(auth);
        return NULL;
    }

    return auth;
}

void gv_auth_destroy(GV_AuthManager *auth) {
    if (!auth) return;

    for (size_t i = 0; i < auth->key_count; i++) {
        free(auth->keys[i].description);
    }

    pthread_rwlock_destroy(&auth->rwlock);
    free(auth);
}

/* ============================================================================
 * API Key Management
 * ============================================================================ */

int gv_auth_generate_api_key(GV_AuthManager *auth, const char *description,
                              uint64_t expires_at, char *key_out, char *key_id_out) {
    if (!auth || !key_out || !key_id_out) return -1;

    pthread_rwlock_wrlock(&auth->rwlock);

    if (auth->key_count >= MAX_API_KEYS) {
        pthread_rwlock_unlock(&auth->rwlock);
        return -1;
    }

    /* Generate random key ID and key */
    unsigned char key_id_bytes[KEY_ID_LEN];
    unsigned char key_bytes[KEY_LEN];
    generate_random_bytes(key_id_bytes, KEY_ID_LEN);
    generate_random_bytes(key_bytes, KEY_LEN);

    /* Convert to hex */
    gv_auth_to_hex(key_id_bytes, KEY_ID_LEN, key_id_out);
    gv_auth_to_hex(key_bytes, KEY_LEN, key_out);

    /* Hash the key for storage */
    unsigned char hash[HASH_LEN];
    gv_auth_sha256(key_bytes, KEY_LEN, hash);

    /* Store the key entry */
    APIKeyEntry *entry = &auth->keys[auth->key_count];
    strncpy(entry->key_id, key_id_out, sizeof(entry->key_id) - 1);
    gv_auth_to_hex(hash, HASH_LEN, entry->key_hash);
    entry->description = description ? strdup(description) : NULL;
    entry->created_at = (uint64_t)time(NULL);
    entry->expires_at = expires_at;
    entry->enabled = 1;

    auth->key_count++;

    pthread_rwlock_unlock(&auth->rwlock);
    return 0;
}

int gv_auth_add_api_key(GV_AuthManager *auth, const char *key_id,
                         const char *key_hash, const char *description,
                         uint64_t expires_at) {
    if (!auth || !key_id || !key_hash) return -1;

    pthread_rwlock_wrlock(&auth->rwlock);

    if (auth->key_count >= MAX_API_KEYS) {
        pthread_rwlock_unlock(&auth->rwlock);
        return -1;
    }

    APIKeyEntry *entry = &auth->keys[auth->key_count];
    strncpy(entry->key_id, key_id, sizeof(entry->key_id) - 1);
    strncpy(entry->key_hash, key_hash, sizeof(entry->key_hash) - 1);
    entry->description = description ? strdup(description) : NULL;
    entry->created_at = (uint64_t)time(NULL);
    entry->expires_at = expires_at;
    entry->enabled = 1;

    auth->key_count++;

    pthread_rwlock_unlock(&auth->rwlock);
    return 0;
}

int gv_auth_revoke_api_key(GV_AuthManager *auth, const char *key_id) {
    if (!auth || !key_id) return -1;

    pthread_rwlock_wrlock(&auth->rwlock);

    for (size_t i = 0; i < auth->key_count; i++) {
        if (strcmp(auth->keys[i].key_id, key_id) == 0) {
            auth->keys[i].enabled = 0;
            pthread_rwlock_unlock(&auth->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&auth->rwlock);
    return -1;
}

int gv_auth_list_api_keys(GV_AuthManager *auth, GV_APIKey **keys, size_t *count) {
    if (!auth || !keys || !count) return -1;

    pthread_rwlock_rdlock(&auth->rwlock);

    *count = auth->key_count;
    if (*count == 0) {
        *keys = NULL;
        pthread_rwlock_unlock(&auth->rwlock);
        return 0;
    }

    *keys = malloc(*count * sizeof(GV_APIKey));
    if (!*keys) {
        pthread_rwlock_unlock(&auth->rwlock);
        return -1;
    }

    for (size_t i = 0; i < *count; i++) {
        (*keys)[i].key_id = strdup(auth->keys[i].key_id);
        (*keys)[i].key_hash = strdup(auth->keys[i].key_hash);
        (*keys)[i].description = auth->keys[i].description ? strdup(auth->keys[i].description) : NULL;
        (*keys)[i].created_at = auth->keys[i].created_at;
        (*keys)[i].expires_at = auth->keys[i].expires_at;
        (*keys)[i].enabled = auth->keys[i].enabled;
    }

    pthread_rwlock_unlock(&auth->rwlock);
    return 0;
}

void gv_auth_free_api_keys(GV_APIKey *keys, size_t count) {
    if (!keys) return;
    for (size_t i = 0; i < count; i++) {
        free(keys[i].key_id);
        free(keys[i].key_hash);
        free(keys[i].description);
    }
    free(keys);
}

/* ============================================================================
 * Authentication
 * ============================================================================ */

GV_AuthResult gv_auth_verify_api_key(GV_AuthManager *auth, const char *api_key,
                                      GV_Identity *identity) {
    if (!auth || !api_key) return GV_AUTH_MISSING;

    /* Convert key hex string to bytes */
    size_t key_len = strlen(api_key);
    if (key_len != KEY_LEN * 2) return GV_AUTH_INVALID_FORMAT;

    unsigned char key_bytes[KEY_LEN];
    for (size_t i = 0; i < KEY_LEN; i++) {
        unsigned int byte;
        if (sscanf(api_key + i * 2, "%2x", &byte) != 1) {
            return GV_AUTH_INVALID_FORMAT;
        }
        key_bytes[i] = (unsigned char)byte;
    }

    /* Hash the key */
    unsigned char hash[HASH_LEN];
    gv_auth_sha256(key_bytes, KEY_LEN, hash);
    char hash_hex[HASH_LEN * 2 + 1];
    gv_auth_to_hex(hash, HASH_LEN, hash_hex);

    pthread_rwlock_rdlock(&auth->rwlock);

    uint64_t now = (uint64_t)time(NULL);

    for (size_t i = 0; i < auth->key_count; i++) {
        if (strcmp(auth->keys[i].key_hash, hash_hex) == 0) {
            if (!auth->keys[i].enabled) {
                pthread_rwlock_unlock(&auth->rwlock);
                return GV_AUTH_INVALID_KEY;
            }
            if (auth->keys[i].expires_at > 0 && auth->keys[i].expires_at < now) {
                pthread_rwlock_unlock(&auth->rwlock);
                return GV_AUTH_EXPIRED;
            }

            if (identity) {
                memset(identity, 0, sizeof(*identity));
                identity->key_id = strdup(auth->keys[i].key_id);
                identity->subject = strdup(auth->keys[i].key_id);
                identity->auth_time = now;
                identity->expires_at = auth->keys[i].expires_at;
            }

            pthread_rwlock_unlock(&auth->rwlock);
            return GV_AUTH_SUCCESS;
        }
    }

    pthread_rwlock_unlock(&auth->rwlock);
    return GV_AUTH_INVALID_KEY;
}

GV_AuthResult gv_auth_verify_jwt(GV_AuthManager *auth, const char *token,
                                  GV_Identity *identity) {
    if (!auth || !token) return GV_AUTH_MISSING;

    /* Minimal JWT validation (header.payload.signature) */
    const char *dot1 = strchr(token, '.');
    if (!dot1) return GV_AUTH_INVALID_FORMAT;

    const char *dot2 = strchr(dot1 + 1, '.');
    if (!dot2) return GV_AUTH_INVALID_FORMAT;

    /* For now, just check format - full JWT validation requires more code */
    /* In production, use a proper JWT library */

    if (auth->config.jwt.secret == NULL) {
        return GV_AUTH_INVALID_SIGNATURE;
    }

    /* Placeholder: accept any well-formed JWT when secret is set */
    if (identity) {
        memset(identity, 0, sizeof(*identity));
        identity->subject = strdup("jwt-user");
        identity->auth_time = (uint64_t)time(NULL);
    }

    return GV_AUTH_SUCCESS;
}

GV_AuthResult gv_auth_authenticate(GV_AuthManager *auth, const char *credential,
                                    GV_Identity *identity) {
    if (!auth) return GV_AUTH_MISSING;
    if (!credential || strlen(credential) == 0) return GV_AUTH_MISSING;

    if (auth->config.type == GV_AUTH_NONE) {
        if (identity) {
            memset(identity, 0, sizeof(*identity));
            identity->subject = strdup("anonymous");
            identity->auth_time = (uint64_t)time(NULL);
        }
        return GV_AUTH_SUCCESS;
    }

    /* Detect type: JWT has dots, API key is hex */
    if (strchr(credential, '.') != NULL) {
        return gv_auth_verify_jwt(auth, credential, identity);
    } else {
        return gv_auth_verify_api_key(auth, credential, identity);
    }
}

void gv_auth_free_identity(GV_Identity *identity) {
    if (!identity) return;
    free(identity->subject);
    free(identity->key_id);
    memset(identity, 0, sizeof(*identity));
}

/* ============================================================================
 * JWT Generation
 * ============================================================================ */

/* Base64url encoding */
static const char base64url_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

static size_t base64url_encode(const void *data, size_t len, char *out) {
    const unsigned char *in = (const unsigned char *)data;
    size_t out_len = 0;

    for (size_t i = 0; i < len; i += 3) {
        unsigned int n = ((unsigned int)in[i]) << 16;
        if (i + 1 < len) n |= ((unsigned int)in[i + 1]) << 8;
        if (i + 2 < len) n |= in[i + 2];

        out[out_len++] = base64url_chars[(n >> 18) & 0x3f];
        out[out_len++] = base64url_chars[(n >> 12) & 0x3f];
        if (i + 1 < len) out[out_len++] = base64url_chars[(n >> 6) & 0x3f];
        if (i + 2 < len) out[out_len++] = base64url_chars[n & 0x3f];
    }
    out[out_len] = '\0';
    return out_len;
}

int gv_auth_generate_jwt(GV_AuthManager *auth, const char *subject,
                          uint64_t expires_in, char *token_out, size_t token_size) {
    if (!auth || !subject || !token_out || token_size < 256) return -1;
    if (!auth->config.jwt.secret) return -1;

    /* Build header */
    const char *header = "{\"alg\":\"HS256\",\"typ\":\"JWT\"}";
    char header_b64[128];
    base64url_encode(header, strlen(header), header_b64);

    /* Build payload */
    uint64_t now = (uint64_t)time(NULL);
    uint64_t exp = now + expires_in;

    char payload[512];
    snprintf(payload, sizeof(payload),
             "{\"sub\":\"%s\",\"iat\":%llu,\"exp\":%llu}",
             subject, (unsigned long long)now, (unsigned long long)exp);

    char payload_b64[512];
    base64url_encode(payload, strlen(payload), payload_b64);

    /* Build signature input */
    char sig_input[1024];
    snprintf(sig_input, sizeof(sig_input), "%s.%s", header_b64, payload_b64);

    /* HMAC-SHA256 - simplified (XOR secret with inner/outer pad) */
    unsigned char key_ipad[64], key_opad[64];
    memset(key_ipad, 0x36, 64);
    memset(key_opad, 0x5c, 64);

    for (size_t i = 0; i < auth->config.jwt.secret_len && i < 64; i++) {
        key_ipad[i] ^= ((unsigned char *)auth->config.jwt.secret)[i];
        key_opad[i] ^= ((unsigned char *)auth->config.jwt.secret)[i];
    }

    unsigned char inner_hash[32];
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, key_ipad, 64);
    sha256_update(&ctx, (uint8_t *)sig_input, strlen(sig_input));
    sha256_final(&ctx, inner_hash);

    unsigned char signature[32];
    sha256_init(&ctx);
    sha256_update(&ctx, key_opad, 64);
    sha256_update(&ctx, inner_hash, 32);
    sha256_final(&ctx, signature);

    char sig_b64[64];
    base64url_encode(signature, 32, sig_b64);

    /* Combine */
    snprintf(token_out, token_size, "%s.%s.%s", header_b64, payload_b64, sig_b64);

    return 0;
}

/* ============================================================================
 * Utility
 * ============================================================================ */

const char *gv_auth_result_string(GV_AuthResult result) {
    switch (result) {
        case GV_AUTH_SUCCESS: return "Authentication successful";
        case GV_AUTH_INVALID_KEY: return "Invalid API key";
        case GV_AUTH_EXPIRED: return "Token expired";
        case GV_AUTH_INVALID_SIGNATURE: return "Invalid signature";
        case GV_AUTH_INVALID_FORMAT: return "Malformed credential";
        case GV_AUTH_MISSING: return "No credentials provided";
        default: return "Unknown error";
    }
}
