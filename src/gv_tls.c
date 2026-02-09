/**
 * @file gv_tls.c
 * @brief TLS/HTTPS transport layer for the GigaVector HTTP server.
 *
 * This module wraps OpenSSL to provide TLS support.  When compiled without
 * GV_HAVE_OPENSSL every public function degrades gracefully: the availability
 * check returns 0 and context creation fails with a clear error, so the rest
 * of the server can still operate in plain-HTTP mode.
 */

#include "gigavector/gv_tls.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * OpenSSL Implementation
 * ============================================================================ */

#ifdef GV_HAVE_OPENSSL

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/pem.h>

/* ---- internal context --------------------------------------------------- */

struct GV_TLSContext {
    SSL_CTX    *ssl_ctx;
    char       *cert_path;   /* kept for cert_days_remaining */
};

/* ---- helpers ------------------------------------------------------------ */

static void tls_log_errors(const char *prefix) {
    unsigned long err;
    while ((err = ERR_get_error()) != 0) {
        char buf[256];
        ERR_error_string_n(err, buf, sizeof(buf));
        fprintf(stderr, "gv_tls: %s: %s\n", prefix, buf);
    }
}

/* ---- public API --------------------------------------------------------- */

int gv_tls_is_available(void) {
    return 1;
}

void gv_tls_config_init(GV_TLSConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->min_version   = GV_TLS_1_2;
    config->cipher_list   = NULL;
    config->verify_client = 0;
}

GV_TLSContext *gv_tls_create(const GV_TLSConfig *config) {
    if (!config || !config->cert_file || !config->key_file) {
        fprintf(stderr, "gv_tls: cert_file and key_file are required\n");
        return NULL;
    }

    /* One-time library init (safe to call more than once in OpenSSL >= 1.1) */
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();

    const SSL_METHOD *method = TLS_server_method();
    if (!method) {
        tls_log_errors("TLS_server_method");
        return NULL;
    }

    SSL_CTX *ssl_ctx = SSL_CTX_new(method);
    if (!ssl_ctx) {
        tls_log_errors("SSL_CTX_new");
        return NULL;
    }

    /* --- minimum protocol version ---------------------------------------- */
    long min_ver = (config->min_version == GV_TLS_1_3)
                       ? TLS1_3_VERSION
                       : TLS1_2_VERSION;
    SSL_CTX_set_min_proto_version(ssl_ctx, min_ver);

    /* --- cipher suites --------------------------------------------------- */
    if (config->cipher_list) {
        if (SSL_CTX_set_cipher_list(ssl_ctx, config->cipher_list) != 1) {
            tls_log_errors("SSL_CTX_set_cipher_list");
            SSL_CTX_free(ssl_ctx);
            return NULL;
        }
    }

    /* --- certificate & key ----------------------------------------------- */
    if (SSL_CTX_use_certificate_chain_file(ssl_ctx, config->cert_file) != 1) {
        tls_log_errors("SSL_CTX_use_certificate_chain_file");
        SSL_CTX_free(ssl_ctx);
        return NULL;
    }

    if (SSL_CTX_use_PrivateKey_file(ssl_ctx, config->key_file,
                                     SSL_FILETYPE_PEM) != 1) {
        tls_log_errors("SSL_CTX_use_PrivateKey_file");
        SSL_CTX_free(ssl_ctx);
        return NULL;
    }

    if (SSL_CTX_check_private_key(ssl_ctx) != 1) {
        tls_log_errors("SSL_CTX_check_private_key");
        SSL_CTX_free(ssl_ctx);
        return NULL;
    }

    /* --- mutual TLS (optional) ------------------------------------------- */
    if (config->verify_client) {
        SSL_CTX_set_verify(ssl_ctx,
                           SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT,
                           NULL);
        if (config->ca_file) {
            if (SSL_CTX_load_verify_locations(ssl_ctx, config->ca_file,
                                               NULL) != 1) {
                tls_log_errors("SSL_CTX_load_verify_locations");
                SSL_CTX_free(ssl_ctx);
                return NULL;
            }
        }
    }

    /* --- build wrapper --------------------------------------------------- */
    GV_TLSContext *ctx = calloc(1, sizeof(GV_TLSContext));
    if (!ctx) {
        SSL_CTX_free(ssl_ctx);
        return NULL;
    }
    ctx->ssl_ctx   = ssl_ctx;
    ctx->cert_path = strdup(config->cert_file);

    return ctx;
}

void gv_tls_destroy(GV_TLSContext *ctx) {
    if (!ctx) return;
    if (ctx->ssl_ctx)   SSL_CTX_free(ctx->ssl_ctx);
    free(ctx->cert_path);
    free(ctx);
}

const char *gv_tls_version_string(const GV_TLSContext *ctx) {
    if (!ctx || !ctx->ssl_ctx) return "unknown";
    /* Create a temporary SSL object to query the negotiated protocol.
     * Before any connection is made we report the highest version the
     * context supports. */
    SSL *tmp = SSL_new(ctx->ssl_ctx);
    if (!tmp) return "unknown";
    const char *ver = SSL_get_version(tmp);
    SSL_free(tmp);
    return ver;
}

/* ---- connection-level operations ---------------------------------------- */

int gv_tls_accept(GV_TLSContext *ctx, int client_fd, void **tls_conn) {
    if (!ctx || !tls_conn || client_fd < 0) return -1;

    SSL *ssl = SSL_new(ctx->ssl_ctx);
    if (!ssl) {
        tls_log_errors("SSL_new");
        return -1;
    }

    if (SSL_set_fd(ssl, client_fd) != 1) {
        tls_log_errors("SSL_set_fd");
        SSL_free(ssl);
        return -1;
    }

    int ret = SSL_accept(ssl);
    if (ret != 1) {
        int err = SSL_get_error(ssl, ret);
        fprintf(stderr, "gv_tls: SSL_accept failed (ssl_err=%d)\n", err);
        tls_log_errors("SSL_accept");
        SSL_free(ssl);
        return -1;
    }

    *tls_conn = ssl;
    return 0;
}

int gv_tls_read(void *tls_conn, void *buf, size_t len) {
    if (!tls_conn || !buf || len == 0) return -1;
    int n = SSL_read((SSL *)tls_conn, buf, (int)len);
    if (n <= 0) {
        int err = SSL_get_error((SSL *)tls_conn, n);
        if (err == SSL_ERROR_ZERO_RETURN) return 0;  /* clean shutdown */
        return -1;
    }
    return n;
}

int gv_tls_write(void *tls_conn, const void *buf, size_t len) {
    if (!tls_conn || !buf || len == 0) return -1;
    int n = SSL_write((SSL *)tls_conn, buf, (int)len);
    if (n <= 0) return -1;
    return n;
}

void gv_tls_close_conn(void *tls_conn) {
    if (!tls_conn) return;
    SSL *ssl = (SSL *)tls_conn;
    SSL_shutdown(ssl);
    SSL_free(ssl);
}

/* ---- certificate info --------------------------------------------------- */

int gv_tls_get_peer_cn(void *tls_conn, char *buf, size_t buf_size) {
    if (!tls_conn || !buf || buf_size == 0) return -1;

    SSL *ssl = (SSL *)tls_conn;
    X509 *cert = SSL_get_peer_certificate(ssl);
    if (!cert) return -1;

    X509_NAME *subject = X509_get_subject_name(cert);
    if (!subject) {
        X509_free(cert);
        return -1;
    }

    int idx = X509_NAME_get_text_by_NID(subject, NID_commonName,
                                         buf, (int)buf_size);
    X509_free(cert);
    return (idx < 0) ? -1 : 0;
}

int gv_tls_cert_days_remaining(const GV_TLSContext *ctx) {
    if (!ctx || !ctx->cert_path) return -1;

    FILE *fp = fopen(ctx->cert_path, "r");
    if (!fp) return -1;

    X509 *cert = PEM_read_X509(fp, NULL, NULL, NULL);
    fclose(fp);
    if (!cert) return -1;

    const ASN1_TIME *not_after = X509_get0_notAfter(cert);
    if (!not_after) {
        X509_free(cert);
        return -1;
    }

    int day_diff = 0, sec_diff = 0;
    if (!ASN1_TIME_diff(&day_diff, &sec_diff, NULL, not_after)) {
        X509_free(cert);
        return -1;
    }

    X509_free(cert);

    /* If sec_diff is negative the cert is already expired by a partial day,
     * but day_diff may still be >= 0.  Be conservative: count only full
     * positive days. */
    if (day_diff < 0) return 0;
    return day_diff;
}

/* ============================================================================
 * Stub Implementation (no OpenSSL)
 * ============================================================================ */

#else /* !GV_HAVE_OPENSSL */

struct GV_TLSContext {
    int dummy;  /* unused – structure must not be zero-sized */
};

int gv_tls_is_available(void) {
    return 0;
}

void gv_tls_config_init(GV_TLSConfig *config) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->min_version   = GV_TLS_1_2;
    config->cipher_list   = NULL;
    config->verify_client = 0;
}

GV_TLSContext *gv_tls_create(const GV_TLSConfig *config) {
    (void)config;
    fprintf(stderr,
            "gv_tls: TLS support not available "
            "(compile with -DGV_HAVE_OPENSSL and link against libssl)\n");
    return NULL;
}

void gv_tls_destroy(GV_TLSContext *ctx) {
    (void)ctx;
}

const char *gv_tls_version_string(const GV_TLSContext *ctx) {
    (void)ctx;
    return "none (no TLS support)";
}

int gv_tls_accept(GV_TLSContext *ctx, int client_fd, void **tls_conn) {
    (void)ctx;
    (void)client_fd;
    (void)tls_conn;
    return -1;
}

int gv_tls_read(void *tls_conn, void *buf, size_t len) {
    (void)tls_conn;
    (void)buf;
    (void)len;
    return -1;
}

int gv_tls_write(void *tls_conn, const void *buf, size_t len) {
    (void)tls_conn;
    (void)buf;
    (void)len;
    return -1;
}

void gv_tls_close_conn(void *tls_conn) {
    (void)tls_conn;
}

int gv_tls_get_peer_cn(void *tls_conn, char *buf, size_t buf_size) {
    (void)tls_conn;
    (void)buf;
    (void)buf_size;
    return -1;
}

int gv_tls_cert_days_remaining(const GV_TLSContext *ctx) {
    (void)ctx;
    return -1;
}

#endif /* GV_HAVE_OPENSSL */
