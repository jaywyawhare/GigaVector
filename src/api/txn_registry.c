/**
 * @file txn_registry.c
 * @brief Thread-safe registry mapping opaque tokens to live GV_DBTxn handles.
 */
#include "api/txn_registry.h"
#include "core/memory.h"
#include "security/crypto.h"

#include <string.h>
#include <pthread.h>

typedef struct {
    char      token[GV_TXN_TOKEN_SIZE];
    GV_DBTxn *txn;
} TxnEntry;

struct GV_TxnRegistry {
    TxnEntry       *entries;
    size_t          count;
    size_t          cap;
    pthread_mutex_t mutex;
};

GV_TxnRegistry *txn_registry_create(void) {
    GV_TxnRegistry *reg = (GV_TxnRegistry *)gv_calloc(1, sizeof(*reg));
    if (!reg) return NULL;
    if (pthread_mutex_init(&reg->mutex, NULL) != 0) {
        gv_free(reg);
        return NULL;
    }
    return reg;
}

void txn_registry_destroy(GV_TxnRegistry *reg) {
    if (!reg) return;
    /* Roll back anything the client left open. */
    for (size_t i = 0; i < reg->count; i++) {
        if (reg->entries[i].txn) db_rollback(reg->entries[i].txn);
    }
    gv_free(reg->entries);
    pthread_mutex_destroy(&reg->mutex);
    gv_free(reg);
}

/* Lower-case hex encode. */
static void to_hex(const unsigned char *in, size_t n, char *out) {
    static const char *h = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i * 2]     = h[(in[i] >> 4) & 0xF];
        out[i * 2 + 1] = h[in[i] & 0xF];
    }
    out[n * 2] = '\0';
}

/* Find the index of a token, or -1. Caller holds the mutex. */
static long find_locked(GV_TxnRegistry *reg, const char *token) {
    for (size_t i = 0; i < reg->count; i++) {
        if (strcmp(reg->entries[i].token, token) == 0) return (long)i;
    }
    return -1;
}

/* Remove entry at index i (swap with last). Caller holds the mutex. */
static void remove_at_locked(GV_TxnRegistry *reg, size_t i) {
    reg->entries[i] = reg->entries[reg->count - 1];
    reg->count--;
}

int txn_registry_begin(GV_TxnRegistry *reg, GV_Database *db,
                       char *token_out, size_t token_size) {
    if (!reg || !db || !token_out || token_size < GV_TXN_TOKEN_SIZE) return -1;

    GV_DBTxn *txn = db_begin(db);
    if (!txn) return -1;

    unsigned char raw[16];
    if (gv_secure_random_bytes(raw, sizeof(raw)) != 0) {
        db_rollback(txn);
        return -1;
    }
    char token[GV_TXN_TOKEN_SIZE];
    to_hex(raw, sizeof(raw), token);

    pthread_mutex_lock(&reg->mutex);
    if (reg->count == reg->cap) {
        size_t nc = reg->cap ? reg->cap * 2 : 8;
        TxnEntry *ne = (TxnEntry *)gv_realloc(reg->entries, nc * sizeof(TxnEntry));
        if (!ne) {
            pthread_mutex_unlock(&reg->mutex);
            db_rollback(txn);
            return -1;
        }
        reg->entries = ne;
        reg->cap = nc;
    }
    memcpy(reg->entries[reg->count].token, token, GV_TXN_TOKEN_SIZE);
    reg->entries[reg->count].txn = txn;
    reg->count++;
    pthread_mutex_unlock(&reg->mutex);

    memcpy(token_out, token, GV_TXN_TOKEN_SIZE);
    return 0;
}

GV_DBTxn *txn_registry_get(GV_TxnRegistry *reg, const char *token) {
    if (!reg || !token) return NULL;
    pthread_mutex_lock(&reg->mutex);
    long i = find_locked(reg, token);
    GV_DBTxn *txn = (i >= 0) ? reg->entries[i].txn : NULL;
    pthread_mutex_unlock(&reg->mutex);
    return txn;
}

int txn_registry_commit(GV_TxnRegistry *reg, const char *token) {
    if (!reg || !token) return -1;
    pthread_mutex_lock(&reg->mutex);
    long i = find_locked(reg, token);
    if (i < 0) { pthread_mutex_unlock(&reg->mutex); return -2; }  /* unknown token */
    GV_DBTxn *txn = reg->entries[i].txn;
    remove_at_locked(reg, (size_t)i);
    pthread_mutex_unlock(&reg->mutex);
    /* db_commit frees the txn (and consumes it) regardless of outcome. */
    return db_commit(txn);
}

int txn_registry_rollback(GV_TxnRegistry *reg, const char *token) {
    if (!reg || !token) return -1;
    pthread_mutex_lock(&reg->mutex);
    long i = find_locked(reg, token);
    if (i < 0) { pthread_mutex_unlock(&reg->mutex); return -2; }  /* unknown token */
    GV_DBTxn *txn = reg->entries[i].txn;
    remove_at_locked(reg, (size_t)i);
    pthread_mutex_unlock(&reg->mutex);
    return db_rollback(txn);
}

size_t txn_registry_count(GV_TxnRegistry *reg) {
    if (!reg) return 0;
    pthread_mutex_lock(&reg->mutex);
    size_t n = reg->count;
    pthread_mutex_unlock(&reg->mutex);
    return n;
}
