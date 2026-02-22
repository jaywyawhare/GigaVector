#ifndef GIGAVECTOR_GV_SSO_H
#define GIGAVECTOR_GV_SSO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_sso.h
 * @brief Enterprise SSO / OIDC / SAML authentication for GigaVector.
 *
 * Provides single sign-on support with OIDC discovery, JWT validation
 * against OIDC JWKS endpoints, and SAML assertion parsing.
 */

/**
 * @brief SSO provider type.
 */
typedef enum {
    GV_SSO_OIDC = 0,               /**< OpenID Connect provider. */
    GV_SSO_SAML = 1                /**< SAML 2.0 provider. */
} GV_SSOProvider;

/**
 * @brief SSO configuration.
 */
typedef struct {
    int provider;                   /**< GV_SSOProvider value. */
    const char *issuer_url;         /**< OIDC issuer URL. */
    const char *client_id;          /**< OIDC client identifier. */
    const char *client_secret;      /**< OIDC client secret. */
    const char *redirect_uri;       /**< OAuth2 redirect URI. */
    const char *saml_metadata_url;  /**< SAML IdP metadata URL. */
    const char *saml_entity_id;     /**< SAML service provider entity ID. */
    int verify_ssl;                 /**< Verify TLS certificates (default: 1). */
    size_t token_ttl;               /**< Token time-to-live in seconds (default: 3600). */
    const char *allowed_groups;     /**< Comma-separated list of allowed groups. */
    const char *admin_groups;       /**< Comma-separated list of admin groups. */
} GV_SSOConfig;

/**
 * @brief Opaque SSO manager handle.
 */
typedef struct GV_SSOManager GV_SSOManager;

/**
 * @brief Authenticated SSO token with user claims.
 */
typedef struct {
    char *subject;                  /**< Subject identifier (sub claim). */
    char *email;                    /**< User email address. */
    char *name;                     /**< User display name. */
    char **groups;                  /**< Array of group memberships. */
    size_t group_count;             /**< Number of groups. */
    uint64_t issued_at;             /**< Token issue timestamp (iat). */
    uint64_t expires_at;            /**< Token expiration timestamp (exp). */
    int is_admin;                   /**< Whether user is in an admin group. */
} GV_SSOToken;

/* Lifecycle */

/**
 * @brief Create an SSO manager.
 *
 * @param config SSO configuration.
 * @return SSO manager instance, or NULL on error.
 */
GV_SSOManager *gv_sso_create(const GV_SSOConfig *config);

/**
 * @brief Destroy an SSO manager.
 *
 * @param mgr SSO manager instance (safe to call with NULL).
 */
void gv_sso_destroy(GV_SSOManager *mgr);

/* OIDC Discovery */

/**
 * @brief Perform OIDC discovery.
 *
 * Fetches {issuer_url}/.well-known/openid-configuration and extracts
 * authorization_endpoint, token_endpoint, jwks_uri, and userinfo_endpoint.
 *
 * @param mgr SSO manager.
 * @return 0 on success, -1 on error.
 */
int gv_sso_discover(GV_SSOManager *mgr);

/* Authentication Flow */

/**
 * @brief Build the authorization URL for OIDC login redirect.
 *
 * Constructs {auth_endpoint}?client_id=...&redirect_uri=...&response_type=code
 * &scope=openid+profile+email+groups&state=...
 *
 * @param mgr SSO manager.
 * @param state Opaque state parameter for CSRF protection.
 * @param url Output buffer for the authorization URL.
 * @param url_size Size of output buffer.
 * @return 0 on success, -1 on error.
 */
int gv_sso_get_auth_url(const GV_SSOManager *mgr, const char *state,
                         char *url, size_t url_size);

/**
 * @brief Exchange an authorization code for tokens.
 *
 * POSTs to the token_endpoint with the authorization code and returns
 * the validated identity from the id_token.
 *
 * @param mgr SSO manager.
 * @param auth_code Authorization code from the IdP callback.
 * @return SSO token with claims, or NULL on error.  Caller must free
 *         with gv_sso_free_token().
 */
GV_SSOToken *gv_sso_exchange_code(GV_SSOManager *mgr, const char *auth_code);

/**
 * @brief Validate an existing token string (JWT or SAML assertion).
 *
 * For OIDC: decodes the JWT, verifies signature against JWKS, and
 * extracts claims.  For SAML: decodes and parses the XML assertion.
 *
 * @param mgr SSO manager.
 * @param token_string Raw token / assertion string.
 * @return SSO token with claims, or NULL on error.  Caller must free
 *         with gv_sso_free_token().
 */
GV_SSOToken *gv_sso_validate_token(GV_SSOManager *mgr, const char *token_string);

/**
 * @brief Refresh an OIDC access token using a refresh token.
 *
 * @param mgr SSO manager.
 * @param refresh_token The refresh token string.
 * @return New SSO token with claims, or NULL on error.  Caller must free
 *         with gv_sso_free_token().
 */
GV_SSOToken *gv_sso_refresh_token(GV_SSOManager *mgr, const char *refresh_token);

/**
 * @brief Free an SSO token.
 *
 * @param token Token to free (safe to call with NULL).
 */
void gv_sso_free_token(GV_SSOToken *token);

/* Group Checking */

/**
 * @brief Check whether a token has a specific group membership.
 *
 * @param token SSO token.
 * @param group Group name to check.
 * @return 1 if the token has the group, 0 otherwise.
 */
int gv_sso_has_group(const GV_SSOToken *token, const char *group);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_SSO_H */
