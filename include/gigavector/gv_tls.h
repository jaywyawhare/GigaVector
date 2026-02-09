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

void gv_tls_config_init(GV_TLSConfig *config);
GV_TLSContext *gv_tls_create(const GV_TLSConfig *config);
void gv_tls_destroy(GV_TLSContext *ctx);

int gv_tls_is_available(void);
const char *gv_tls_version_string(const GV_TLSContext *ctx);

/* Connection-level operations */
int gv_tls_accept(GV_TLSContext *ctx, int client_fd, void **tls_conn);
int gv_tls_read(void *tls_conn, void *buf, size_t len);
int gv_tls_write(void *tls_conn, const void *buf, size_t len);
void gv_tls_close_conn(void *tls_conn);

/* Certificate info */
int gv_tls_get_peer_cn(void *tls_conn, char *buf, size_t buf_size);
int gv_tls_cert_days_remaining(const GV_TLSContext *ctx);

#ifdef __cplusplus
}
#endif
#endif
