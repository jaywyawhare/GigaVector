#ifndef GIGAVECTOR_GV_TLS_H
#define GIGAVECTOR_GV_TLS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_TLS_1_2 = 0,
    GV_TLS_1_3 = 1
} GV_TLSVersion;

typedef struct {
    const char *cert_file;        /* Path to PEM certificate file */
    const char *key_file;         /* Path to PEM private key file */
    const char *ca_file;          /* Optional CA bundle for client verification */
    GV_TLSVersion min_version;    /* Minimum TLS version (default: TLS 1.2) */
    const char *cipher_list;      /* Optional cipher suite list (NULL for defaults) */
    int verify_client;            /* Enable mutual TLS (default: 0) */
} GV_TLSConfig;

typedef struct GV_TLSContext GV_TLSContext;

/**
 * @brief Initialize a configuration structure with default values.
 *
 * @param config Configuration to apply/output.
 */
void tls_config_init(GV_TLSConfig *config);
GV_TLSContext *tls_create(const GV_TLSConfig *config);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param ctx ctx.
 */
void tls_destroy(GV_TLSContext *ctx);

/**
 * @brief Query a boolean condition.
 *
 * @return 1 if true, 0 if false, -1 on error.
 */
int tls_is_available(void);
const char *tls_version_string(const GV_TLSContext *ctx);

/**
 * @brief Perform the operation.
 *
 * @param ctx ctx.
 * @param client_fd client_fd.
 * @param tls_conn tls_conn.
 * @return 0 on success, -1 on error.
 */
int tls_accept(GV_TLSContext *ctx, int client_fd, void **tls_conn);
/**
 * @brief Perform the operation.
 *
 * @param tls_conn tls_conn.
 * @param buf buf.
 * @param len len.
 * @return 0 on success, -1 on error.
 */
int tls_read(void *tls_conn, void *buf, size_t len);
/**
 * @brief Perform the operation.
 *
 * @param tls_conn tls_conn.
 * @param buf buf.
 * @param len len.
 * @return 0 on success, -1 on error.
 */
int tls_write(void *tls_conn, const void *buf, size_t len);
/**
 * @brief Perform the operation.
 *
 * @param tls_conn tls_conn.
 */
void tls_close_conn(void *tls_conn);

/**
 * @brief Get a value.
 *
 * @param tls_conn tls_conn.
 * @param buf buf.
 * @param buf_size buf_size.
 * @return 0 on success, -1 on error.
 */
int tls_get_peer_cn(void *tls_conn, char *buf, size_t buf_size);
/**
 * @brief Perform the operation.
 *
 * @param ctx ctx.
 * @return 0 on success, -1 on error.
 */
int tls_cert_days_remaining(const GV_TLSContext *ctx);

#ifdef __cplusplus
}
#endif
#endif
