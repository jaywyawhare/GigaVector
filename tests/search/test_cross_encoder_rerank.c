#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "search/cross_encoder_rerank.h"
#include "core/memory.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_create_destroy(void) {
    /* NULL model path => fallback mode, should still succeed */
    GV_CrossEncoder *ce = cross_encoder_create(NULL, 512);
    ASSERT(ce != NULL, "cross_encoder_create(NULL) should return non-NULL for fallback mode");
    cross_encoder_destroy(ce);

    /* Destroying NULL is safe */
    cross_encoder_destroy(NULL);

    /* Non-existent model path without ONNX => fallback */
    GV_CrossEncoder *ce2 = cross_encoder_create("/nonexistent/model.onnx", 256);
    ASSERT(ce2 != NULL, "cross_encoder_create with missing model should fall back");
    cross_encoder_destroy(ce2);

    return 0;
}

static int test_batch_rerank_fallback(void) {
    GV_CrossEncoder *ce = cross_encoder_create(NULL, 512);
    ASSERT(ce != NULL, "create should succeed");

    const char *docs[] = {
        "the quick brown fox jumps over the lazy dog",
        "machine learning models for natural language processing",
        "brown fox spotted in the park near the dog",
        "completely unrelated topic about cooking recipes"
    };
    size_t doc_count = 4;
    float scores[4];

    int rc = cross_encoder_rerank_batch(ce, "brown fox dog", docs, doc_count, scores);
    ASSERT(rc == 0, "batch rerank should succeed");

    /* Docs 0 and 2 contain "brown", "fox", "dog" => should score highest */
    ASSERT(scores[0] > 0.0f, "doc 0 should have positive score");
    ASSERT(scores[2] > 0.0f, "doc 2 should have positive score");
    ASSERT(scores[3] >= 0.0f, "doc 4 should have non-negative score");

    /* Doc 2 has all three query terms; doc 0 has all three too.
     * Both should score higher than docs 1 and 3. */
    ASSERT(scores[0] > scores[1], "doc 0 should score above doc 1");
    ASSERT(scores[2] > scores[1], "doc 2 should score above doc 1");
    ASSERT(scores[2] > scores[3], "doc 2 should score above doc 3");

    cross_encoder_destroy(ce);
    return 0;
}

/* ── Rerank via callback API ───────────────────────────────────────────── */

static const char *RERANK_DOCS[] = {
    "the quick brown fox jumps over the lazy dog",
    "machine learning models for natural language processing",
    "brown fox spotted in the park near the dog",
    "completely unrelated topic about cooking recipes"
};

/* The getter receives the index into the results array (see header contract)
 * and must return the text for the result at that position — i.e. the doc whose
 * id the result holds. RERANK_DOCS is indexed by doc id. */
static const char *rerank_getter(size_t idx, void *user_data) {
    const GV_SearchResult *results = (const GV_SearchResult *)user_data;
    uint64_t id = results[idx].id;
    if (id < 4) return RERANK_DOCS[id];
    return "";
}

static int test_rerank_callback(void) {
    GV_CrossEncoder *ce = cross_encoder_create(NULL, 512);
    ASSERT(ce != NULL, "create should succeed");

    /* Start with results in a non-relevant order */
    GV_SearchResult results[4];
    memset(results, 0, sizeof(results));

    /* Deliberately put the unrelated doc first */
    results[0].id = 3;  /* cooking recipes */
    results[0].distance = 0.3f;
    results[1].id = 1;  /* machine learning */
    results[1].distance = 0.2f;
    results[2].id = 0;  /* quick brown fox */
    results[2].distance = 0.1f;
    results[3].id = 2;  /* brown fox in park */
    results[3].distance = 0.0f;

    int rc = cross_encoder_rerank(ce, "brown fox dog", results, 4,
                                   rerank_getter, results);
    ASSERT(rc == 0, "rerank should succeed");

    /* After reranking, results containing "brown fox dog" should be first.
     * ids 0 and 2 contain those terms and should rank above ids 1 and 3. */
    ASSERT(results[0].id == 2 || results[0].id == 0,
           "top result should be doc 0 or 2 (contain query terms)");
    ASSERT(results[1].id == 2 || results[1].id == 0,
           "second result should be doc 0 or 2");

    /* The two non-relevant docs (ids 1 and 3 — no query-term overlap, so they
     * tie at score 0) rank last, in either order. */
    ASSERT((results[2].id == 1 || results[2].id == 3) &&
           (results[3].id == 1 || results[3].id == 3),
           "non-relevant docs (1, 3) should rank last");

    cross_encoder_destroy(ce);
    return 0;
}

static int test_rerank_empty(void) {
    GV_CrossEncoder *ce = cross_encoder_create(NULL, 512);
    ASSERT(ce != NULL, "create should succeed");

    /* Reranking 0 results should return 0, not error */
    int rc = cross_encoder_rerank_batch(ce, "query", NULL, 0, NULL);
    ASSERT(rc == 0, "rerank with 0 docs should return 0");

    cross_encoder_destroy(ce);
    return 0;
}

static int test_rerank_null_args(void) {
    GV_CrossEncoder *ce = cross_encoder_create(NULL, 512);

    ASSERT(cross_encoder_rerank(NULL, "q", NULL, 0, NULL, NULL) == -1,
           "NULL ce should return -1");
    ASSERT(cross_encoder_rerank_batch(NULL, "q", NULL, 0, NULL) == -1,
           "NULL ce batch should return -1");
    ASSERT(cross_encoder_rerank_batch(ce, NULL, NULL, 0, NULL) == -1,
           "NULL query should return -1");

    cross_encoder_destroy(ce);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing cross-encoder create/destroy...",             test_create_destroy},
        {"Testing cross-encoder batch rerank (fallback)...",    test_batch_rerank_fallback},
        {"Testing cross-encoder rerank via callback...",        test_rerank_callback},
        {"Testing cross-encoder rerank empty input...",         test_rerank_empty},
        {"Testing cross-encoder rerank null args...",           test_rerank_null_args},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("%s ", tests[i].name);
        if (tests[i].fn() == 0) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
        }
    }
    return passed == n ? 0 : 1;
}
