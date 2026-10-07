/**
 * @file gvserver.c
 * @brief Long-running GigaVector HTTP server daemon (container entrypoint).
 *
 * Opens a persistent database and serves the REST API until SIGINT/SIGTERM,
 * then shuts down gracefully. All configuration comes from the environment so
 * the same binary works unchanged in Docker/Kubernetes.
 *
 *   GV_PORT           listen port                 (default 8080)
 *   GV_BIND           bind address                (default 0.0.0.0)
 *   GV_DATA_DIR       data / save confinement dir (default /data)
 *   GV_DB_PATH        database file               (default $GV_DATA_DIR/gigavector.db)
 *   GV_DIMENSION      vector dimension            (default 128)
 *   GV_INDEX          kdtree|hnsw|ivfpq|flat      (default hnsw)
 *   GV_API_KEY        API key (enables auth)      (default none)
 *   GV_ALLOW_UNAUTH   "1" to allow unauthenticated mutations (default 0)
 *   GV_THREADS        worker threads              (default 4)
 *   GV_READ_ONLY      "1" to reject writes/admin (read replica) (default 0)
 *   GV_TLS_CERT       path to PEM certificate chain (enables HTTPS with GV_TLS_KEY)
 *   GV_TLS_KEY        path to PEM private key
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

#include "storage/database.h"
#include "api/server.h"

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static const char *env_or(const char *key, const char *fallback) {
    const char *v = getenv(key);
    return (v && *v) ? v : fallback;
}

static GV_IndexType parse_index(const char *s) {
    if (strcmp(s, "kdtree") == 0) return GV_INDEX_TYPE_KDTREE;
    if (strcmp(s, "ivfpq") == 0)  return GV_INDEX_TYPE_IVFPQ;
    if (strcmp(s, "flat") == 0)   return GV_INDEX_TYPE_FLAT;
    return GV_INDEX_TYPE_HNSW;
}

/* Read an entire file into a malloc'd NUL-terminated buffer (caller frees),
 * or NULL on error. Used to load TLS cert/key PEM files. */
static char *read_file_all(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

int main(void) {
    const char *data_dir = env_or("GV_DATA_DIR", "/data");
    const char *index_s  = env_or("GV_INDEX", "hnsw");
    size_t dim = (size_t)strtoul(env_or("GV_DIMENSION", "128"), NULL, 10);

    char db_path[1024];
    const char *explicit_db = getenv("GV_DB_PATH");
    if (explicit_db && *explicit_db) {
        snprintf(db_path, sizeof(db_path), "%s", explicit_db);
    } else {
        snprintf(db_path, sizeof(db_path), "%s/gigavector.db", data_dir);
    }

    GV_Database *db = db_open(db_path, dim, parse_index(index_s));
    if (!db) {
        fprintf(stderr, "gvserver: failed to open database at %s (dim=%zu, index=%s)\n",
                db_path, dim, index_s);
        return EXIT_FAILURE;
    }

    GV_ServerConfig cfg;
    server_config_init(&cfg);
    cfg.port             = (uint16_t)strtoul(env_or("GV_PORT", "8080"), NULL, 10);
    cfg.bind_address     = env_or("GV_BIND", "0.0.0.0");
    cfg.thread_pool_size = (size_t)strtoul(env_or("GV_THREADS", "4"), NULL, 10);
    cfg.data_dir         = data_dir;
    const char *api_key  = getenv("GV_API_KEY");
    if (api_key && *api_key) cfg.api_key = api_key;
    cfg.allow_unauthenticated = strcmp(env_or("GV_ALLOW_UNAUTH", "0"), "1") == 0;
    cfg.read_only = strcmp(env_or("GV_READ_ONLY", "0"), "1") == 0;

    /* Optional HTTPS: both cert and key must be provided. */
    char *tls_cert = NULL, *tls_key = NULL;
    const char *cert_path = getenv("GV_TLS_CERT");
    const char *key_path  = getenv("GV_TLS_KEY");
    if (cert_path && *cert_path && key_path && *key_path) {
        tls_cert = read_file_all(cert_path);
        tls_key  = read_file_all(key_path);
        if (!tls_cert || !tls_key) {
            fprintf(stderr, "gvserver: failed to read TLS cert/key (%s, %s)\n", cert_path, key_path);
            free(tls_cert); free(tls_key);
            db_close(db);
            return EXIT_FAILURE;
        }
        cfg.tls_cert_pem = tls_cert;
        cfg.tls_key_pem  = tls_key;
    }

    GV_Server *server = server_create(db, &cfg);
    if (!server) {
        fprintf(stderr, "gvserver: failed to create server\n");
        free(tls_cert); free(tls_key);
        db_close(db);
        return EXIT_FAILURE;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    if (server_start(server) != GV_SERVER_OK) {
        fprintf(stderr, "gvserver: failed to start on %s:%u\n", cfg.bind_address, cfg.port);
        server_destroy(server);
        free(tls_cert); free(tls_key);
        db_close(db);
        return EXIT_FAILURE;
    }

    printf("gvserver: listening on %s://%s:%u (db=%s dim=%zu index=%s auth=%s%s)\n",
           cfg.tls_cert_pem ? "https" : "http",
           cfg.bind_address, cfg.port, db_path, dim, index_s,
           cfg.api_key ? "on" : (cfg.allow_unauthenticated ? "off(insecure)" : "read-only"),
           cfg.read_only ? " read-only" : "");
    fflush(stdout);

    /* pause() is POSIX-only; on Windows (MinGW) poll the stop flag instead. */
#ifdef _WIN32
    while (!g_stop) Sleep(100);
#else
    while (!g_stop) pause();
#endif

    printf("gvserver: shutting down\n");
    server_stop(server);
    server_destroy(server);
    free(tls_cert); free(tls_key);
    db_close(db);
    return EXIT_SUCCESS;
}
