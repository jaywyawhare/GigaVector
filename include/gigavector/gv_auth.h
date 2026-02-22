#ifndef GIGAVECTOR_GV_AUTH_H
#define GIGAVECTOR_GV_AUTH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_auth.h
 * @brief Authentication for GigaVector.
 *
 * Provides API key and JWT-based authentication.
 */

/**
 * @brief Authentication type.
 */
typedef enum {
    GV_AUTH_NONE = 0,               /**< No authentication required. */
    GV_AUTH_API_KEY = 1,            /**< API key authentication. */
    GV_AUTH_JWT = 2                 /**< JWT bearer token. */
} GV_AuthType;

/**
 * @brief Authentication result.
 */
typedef enum {
    GV_AUTH_SUCCESS = 0,            /**< Authentication successful. */
    GV_AUTH_INVALID_KEY = 1,        /**< Invalid API key. */
    GV_AUTH_EXPIRED = 2,            /**< Token expired. */
    GV_AUTH_INVALID_SIGNATURE = 3,  /**< Invalid JWT signature. */
    GV_AUTH_INVALID_FORMAT = 4,     /**< Malformed credential. */
    GV_AUTH_MISSING = 5             /**< No credentials provided. */
} GV_AuthResult;

/**
 * @brief API key configuration.
 */
typedef struct {
    char *key_id;                   /**< Key identifier. */
    char *key_hash;                 /**< SHA-256 hash of key. */
    char *description;              /**< Human-readable description. */
    uint64_t created_at;            /**< Creation timestamp. */
    uint64_t expires_at;            /**< Expiration (0 = never). */
    int enabled;                    /**< Whether key is active. */
} GV_APIKey;

/**
 * @brief JWT configuration.
 */
typedef struct {
    const char *secret;             /**< HMAC secret (HS256). */
    size_t secret_len;              /**< Secret length. */
    const char *issuer;             /**< Expected issuer (iss claim). */
    const char *audience;           /**< Expected audience (aud claim). */
    uint64_t clock_skew_seconds;    /**< Allowed clock skew (default: 60). */
} GV_JWTConfig;

/**
 * @brief Authentication configuration.
 */
typedef struct {
    GV_AuthType type;               /**< Authentication type. */
    GV_JWTConfig jwt;               /**< JWT configuration (if type == JWT). */
} GV_AuthConfig;

/**
 * @brief Authenticated identity.
 */
typedef struct {
    char *subject;                  /**< Subject (user/service ID). */
    char *key_id;                   /**< API key ID (if API key auth). */
    uint64_t auth_time;             /**< When authentication occurred. */
    uint64_t expires_at;            /**< When auth expires (0 = session). */
    void *claims;                   /**< Additional JWT claims (opaque). */
} GV_Identity;

/**
 * @brief Opaque auth manager handle.
 */
typedef struct GV_AuthManager GV_AuthManager;

/* Configuration */

/**
 * @brief Initialize authentication configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void gv_auth_config_init(GV_AuthConfig *config);

/* Auth Manager Lifecycle */

/**
 * @brief Create an authentication manager.
 *
 * @param config Authentication configuration (NULL for no auth).
 * @return Auth manager instance, or NULL on error.
 */
GV_AuthManager *gv_auth_create(const GV_AuthConfig *config);

/**
 * @brief Destroy an authentication manager.
 *
 * @param auth Auth manager instance (safe to call with NULL).
 */
void gv_auth_destroy(GV_AuthManager *auth);

/* API Key Management */

/**
 * @brief Generate a new API key.
 *
 * @param auth Auth manager.
 * @param description Human-readable description.
 * @param expires_at Expiration timestamp (0 = never).
 * @param key_out Output buffer for generated key (at least 64 bytes).
 * @param key_id_out Output buffer for key ID (at least 32 bytes).
 * @return 0 on success, -1 on error.
 */
int gv_auth_generate_api_key(GV_AuthManager *auth, const char *description,
                              uint64_t expires_at, char *key_out, char *key_id_out);

/**
 * @brief Add an existing API key.
 *
 * @param auth Auth manager.
 * @param key_id Key identifier.
 * @param key_hash SHA-256 hash of key (hex string).
 * @param description Human-readable description.
 * @param expires_at Expiration timestamp (0 = never).
 * @return 0 on success, -1 on error.
 */
int gv_auth_add_api_key(GV_AuthManager *auth, const char *key_id,
                         const char *key_hash, const char *description,
                         uint64_t expires_at);

/**
 * @brief Revoke an API key.
 *
 * @param auth Auth manager.
 * @param key_id Key identifier to revoke.
 * @return 0 on success, -1 on error.
 */
int gv_auth_revoke_api_key(GV_AuthManager *auth, const char *key_id);

/**
 * @brief List all API keys.
 *
 * @param auth Auth manager.
 * @param keys Output array of keys.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int gv_auth_list_api_keys(GV_AuthManager *auth, GV_APIKey **keys, size_t *count);

/**
 * @brief Free API key list.
 *
 * @param keys Keys array to free.
 * @param count Number of keys.
 */
void gv_auth_free_api_keys(GV_APIKey *keys, size_t count);

/* Authentication */

/**
 * @brief Authenticate with an API key.
 *
 * @param auth Auth manager.
 * @param api_key API key string.
 * @param identity Output identity on success.
 * @return Authentication result.
 */
GV_AuthResult gv_auth_verify_api_key(GV_AuthManager *auth, const char *api_key,
                                      GV_Identity *identity);

/**
 * @brief Authenticate with a JWT.
 *
 * @param auth Auth manager.
 * @param token JWT token string.
 * @param identity Output identity on success.
 * @return Authentication result.
 */
GV_AuthResult gv_auth_verify_jwt(GV_AuthManager *auth, const char *token,
                                  GV_Identity *identity);

/**
 * @brief Authenticate with any credential.
 *
 * Auto-detects credential type and validates.
 *
 * @param auth Auth manager.
 * @param credential Credential string (API key or JWT).
 * @param identity Output identity on success.
 * @return Authentication result.
 */
GV_AuthResult gv_auth_authenticate(GV_AuthManager *auth, const char *credential,
                                    GV_Identity *identity);

/**
 * @brief Free identity resources.
 *
 * @param identity Identity to free.
 */
void gv_auth_free_identity(GV_Identity *identity);

/* JWT Utilities */

/**
 * @brief Generate a JWT token.
 *
 * @param auth Auth manager.
 * @param subject Subject (user/service ID).
 * @param expires_in Seconds until expiration.
 * @param token_out Output buffer for token (must be large enough).
 * @param token_size Size of output buffer.
 * @return 0 on success, -1 on error.
 */
int gv_auth_generate_jwt(GV_AuthManager *auth, const char *subject,
                          uint64_t expires_in, char *token_out, size_t token_size);

/**
 * @brief Get auth result description.
 *
 * @param result Auth result.
 * @return Human-readable description.
 */
const char *gv_auth_result_string(GV_AuthResult result);

/* Hashing Utilities */

/**
 * @brief Compute SHA-256 hash.
 *
 * @param data Input data.
 * @param len Data length.
 * @param hash_out Output buffer (32 bytes).
 * @return 0 on success, -1 on error.
 */
int gv_auth_sha256(const void *data, size_t len, unsigned char *hash_out);

/**
 * @brief Convert hash to hex string.
 *
 * @param hash Hash bytes.
 * @param hash_len Hash length.
 * @param hex_out Output buffer (2 * hash_len + 1).
 */
void gv_auth_to_hex(const unsigned char *hash, size_t hash_len, char *hex_out);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_AUTH_H */
