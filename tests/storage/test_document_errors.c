/**
 * test_document_errors.c — error/edge cases for the document ingest pipeline
 * (src/storage/document_ingest.c):
 *   - NULL config / NULL db / NULL text are rejected.
 *   - No embedder configured (embed == NULL and default_embedder == NULL) fails.
 *   - An embedder callback that reports an error (-1) aborts the ingest.
 *   - An empty document produces zero chunks / zero vectors (rc 0).
 *   - Deleting a non-existent document deletes nothing (returns 0).
 */
#include <stdio.h>
#include <string.h>
#include "storage/database.h"
#include "storage/document_ingest.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                         else { printf("ok: %s\n", (m)); } } while (0)

#define DIM 4

/* Well-behaved dim=4 embedder. */
static int embed_ok(void *ctx, const char **texts, size_t n, size_t dim, float *out) {
    (void)ctx;
    for (size_t j = 0; j < n; j++) {
        size_t len = texts[j] ? strlen(texts[j]) : 0;
        for (size_t d = 0; d < dim; d++) out[j * dim + d] = (float)((len + d) % 5);
    }
    return 0;
}

/* Embedder that always reports failure (simulates a backend error / a caller
 * that cannot satisfy the requested dimension). */
static int embed_err(void *ctx, const char **texts, size_t n, size_t dim, float *out) {
    (void)ctx; (void)texts; (void)n; (void)dim; (void)out;
    return -1;
}

int main(void) {
    GV_Database *db = db_open(NULL, DIM, GV_INDEX_TYPE_FLAT);
    CHECK(db != NULL, "db opened");

    GV_IngestConfig ok_cfg;
    memset(&ok_cfg, 0, sizeof(ok_cfg));
    ok_cfg.embed = embed_ok;
    ok_cfg.chunk_tokens = 4;
    ok_cfg.overlap_tokens = 1;

    GV_IngestStats st;

    /* --- NULL argument rejection --- */
    CHECK(gv_ingest_document(NULL, NULL, NULL, "text", "d", &ok_cfg, &st) == -1,
          "NULL db rejected");
    CHECK(gv_ingest_document(db, NULL, NULL, NULL, "d", &ok_cfg, &st) == -1,
          "NULL text rejected");
    CHECK(gv_ingest_document(db, NULL, NULL, "text", "d", NULL, &st) == -1,
          "NULL config rejected");

    /* --- No embedder configured (embed == NULL, default_embedder == NULL) --- */
    {
        GV_IngestConfig no_embed;
        memset(&no_embed, 0, sizeof(no_embed));
        no_embed.chunk_tokens = 4;
        memset(&st, 0, sizeof(st));
        int rc = gv_ingest_document(db, NULL, NULL, "hello world", "dNULLemb",
                                    &no_embed, &st);
        CHECK(rc == -1, "no embedder -> ingest fails");
        CHECK(st.vectors == 0, "no vectors stored when embedder missing");
    }

    /* --- Embedder reports an error (e.g. wrong/unavailable dimension) --- */
    {
        GV_IngestConfig err_cfg = ok_cfg;
        err_cfg.embed = embed_err;
        memset(&st, 0, sizeof(st));
        int rc = gv_ingest_document(db, NULL, NULL, "hello world", "dERR",
                                    &err_cfg, &st);
        CHECK(rc == -1, "embedder error aborts ingest");
    }

    /* --- Empty document: zero chunks, zero vectors, rc 0 --- */
    {
        memset(&st, 0, sizeof(st));
        int rc = gv_ingest_document(db, NULL, NULL, "", "dEMPTY", &ok_cfg, &st);
        CHECK(rc == 0, "empty document ingests cleanly");
        CHECK(st.chunks == 0 && st.vectors == 0, "empty document yields no chunks");
    }
    {
        memset(&st, 0, sizeof(st));
        int rc = gv_ingest_document(db, NULL, NULL, "   \n\t  ", "dWS", &ok_cfg, &st);
        CHECK(rc == 0, "whitespace-only document ingests cleanly");
        CHECK(st.chunks == 0, "whitespace-only document yields no chunks");
    }

    /* --- A good ingest, then delete of a non-existent doc --- */
    {
        memset(&st, 0, sizeof(st));
        int rc = gv_ingest_document(db, NULL, NULL, "alpha beta gamma delta",
                                    "docGood", &ok_cfg, &st);
        CHECK(rc == 0 && st.vectors >= 1, "good document ingested");

        int deleted = db_delete_by_doc(db, "no_such_doc");
        CHECK(deleted == 0, "delete of non-existent doc removes nothing");

        int deleted2 = db_delete_by_doc(db, "docGood");
        CHECK(deleted2 == (int)st.vectors, "delete of existing doc removes its chunks");
    }

    db_close(db);
    printf(failures ? "\nSOME DOCUMENT-ERROR TESTS FAILED (%d)\n"
                    : "\nALL DOCUMENT-ERROR TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
