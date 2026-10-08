/* Server-side transaction registry: token lifecycle, staging, commit/rollback. */
#include <stdio.h>
#include <string.h>

#include "api/txn_registry.h"
#include "storage/database.h"
#include "storage/transaction.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

#define D 4u

int main(void) {
    GV_Database *db = db_open(NULL, D, GV_INDEX_TYPE_FLAT);
    ASSERT(db != NULL, "open db");

    GV_TxnRegistry *reg = txn_registry_create();
    ASSERT(reg != NULL, "create registry");
    ASSERT(txn_registry_count(reg) == 0, "registry starts empty");

    /* Begin -> token maps to a live txn we can stage into. */
    char tok1[GV_TXN_TOKEN_SIZE];
    ASSERT(txn_registry_begin(reg, db, tok1, sizeof(tok1)) == 0, "begin txn 1");
    ASSERT(strlen(tok1) == 32, "token is 32 hex chars");
    ASSERT(txn_registry_count(reg) == 1, "one open txn");

    GV_DBTxn *t1 = txn_registry_get(reg, tok1);
    ASSERT(t1 != NULL, "get returns the live txn");
    float v[D] = {1, 0, 0, 0};
    ASSERT(db_txn_add_vector(t1, v, D) == 0, "stage an insert into the txn");

    /* Unknown token lookups/ops. */
    ASSERT(txn_registry_get(reg, "deadbeef") == NULL, "unknown token -> NULL");
    ASSERT(txn_registry_commit(reg, "deadbeef") == -2, "commit unknown -> -2");
    ASSERT(txn_registry_rollback(reg, "deadbeef") == -2, "rollback unknown -> -2");

    /* Commit removes the entry and makes the data visible. */
    ASSERT(txn_registry_commit(reg, tok1) == GV_TXN_OK, "commit txn 1");
    ASSERT(txn_registry_count(reg) == 0, "entry removed after commit");
    ASSERT(txn_registry_get(reg, tok1) == NULL, "token invalid after commit");
    {
        GV_SearchResult r[2];
        int n = db_search(db, v, 2, r, GV_DISTANCE_EUCLIDEAN);
        ASSERT(n == 1, "committed insert is searchable");
        gv_search_results_free(r, (size_t)(n > 0 ? n : 0));
    }

    /* Rollback discards staged work. */
    char tok2[GV_TXN_TOKEN_SIZE];
    ASSERT(txn_registry_begin(reg, db, tok2, sizeof(tok2)) == 0, "begin txn 2");
    GV_DBTxn *t2 = txn_registry_get(reg, tok2);
    float v2[D] = {0, 1, 0, 0};
    ASSERT(db_txn_add_vector(t2, v2, D) == 0, "stage into txn 2");
    ASSERT(txn_registry_rollback(reg, tok2) == 0, "rollback txn 2");
    ASSERT(txn_registry_count(reg) == 0, "entry removed after rollback");
    {
        GV_SearchResult r[4];
        int n = db_search(db, v2, 4, r, GV_DISTANCE_EUCLIDEAN);
        /* only the committed vector exists; rolled-back insert is absent */
        ASSERT(n == 1, "rolled-back insert not visible");
        gv_search_results_free(r, (size_t)(n > 0 ? n : 0));
    }

    /* Tokens are distinct across begins. */
    char ta[GV_TXN_TOKEN_SIZE], tb[GV_TXN_TOKEN_SIZE];
    ASSERT(txn_registry_begin(reg, db, ta, sizeof(ta)) == 0, "begin a");
    ASSERT(txn_registry_begin(reg, db, tb, sizeof(tb)) == 0, "begin b");
    ASSERT(strcmp(ta, tb) != 0, "distinct tokens");
    ASSERT(txn_registry_count(reg) == 2, "two open txns");

    /* Destroy rolls back the two still-open transactions (no leak/crash). */
    txn_registry_destroy(reg);

    db_close(db);
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL TXN-REGISTRY TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
