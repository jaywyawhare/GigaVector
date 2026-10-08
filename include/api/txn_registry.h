/**
 * @file txn_registry.h
 * @brief Server-side registry of in-flight client transactions.
 *
 * Maps an opaque token to a live GV_DBTxn so a stateful transaction can span
 * multiple REST requests (BEGIN -> stage -> COMMIT/ROLLBACK). Thread-safe.
 */
#ifndef GIGAVECTOR_GV_TXN_REGISTRY_H
#define GIGAVECTOR_GV_TXN_REGISTRY_H

#include <stddef.h>

#include "storage/database.h"
#include "storage/transaction.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_TxnRegistry GV_TxnRegistry;

/** Token buffer size: 32 hex chars + NUL. */
#define GV_TXN_TOKEN_SIZE 33

/**
 * @brief Create an empty transaction registry.
 * @return Registry, or NULL on allocation failure.
 */
GV_TxnRegistry *txn_registry_create(void);

/**
 * @brief Destroy the registry, rolling back any still-open transactions.
 */
void txn_registry_destroy(GV_TxnRegistry *reg);

/**
 * @brief Begin a transaction on @p db and register it under a fresh token.
 *
 * @param reg Registry.
 * @param db Database.
 * @param token_out Buffer receiving the NUL-terminated token (>= GV_TXN_TOKEN_SIZE).
 * @param token_size Size of @p token_out.
 * @return 0 on success, -1 on error.
 */
int txn_registry_begin(GV_TxnRegistry *reg, GV_Database *db,
                       char *token_out, size_t token_size);

/**
 * @brief Look up the live transaction for @p token (for staging inserts/reads).
 * @return The transaction, or NULL if the token is unknown.
 */
GV_DBTxn *txn_registry_get(GV_TxnRegistry *reg, const char *token);

/**
 * @brief Commit and remove the transaction identified by @p token.
 * @return GV_TXN_OK (0), GV_TXN_CONFLICT (1), -1 (commit error), or -2 (unknown token).
 */
int txn_registry_commit(GV_TxnRegistry *reg, const char *token);

/**
 * @brief Roll back and remove the transaction identified by @p token.
 * @return 0 on success, -2 if the token is unknown, -1 on error.
 */
int txn_registry_rollback(GV_TxnRegistry *reg, const char *token);

/**
 * @brief Number of currently open (registered) transactions.
 */
size_t txn_registry_count(GV_TxnRegistry *reg);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_TXN_REGISTRY_H */
