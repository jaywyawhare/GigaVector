#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "multimodal/splade.h"
#include "multimodal/learned_sparse.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_defaults(void) {
    GV_SpladeConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    splade_config_init(&cfg);
    ASSERT(cfg.vocab_size == 30522, "default vocab_size");
    ASSERT(cfg.top_k == 128, "default top_k");
    ASSERT(cfg.onnx_model_path == NULL, "default no model");
    return 0;
}

static int test_encode_fallback(void) {
    GV_SpladeConfig cfg;
    splade_config_init(&cfg);
    GV_SpladeEncoder *enc = splade_create(&cfg);
    ASSERT(enc != NULL, "create encoder");
    ASSERT(splade_is_neural(enc) == 0, "fallback encoder (no model)");

    GV_LSSparseEntry *e = NULL;
    size_t n = 0;
    ASSERT(splade_encode(enc, "deep neural networks", &e, &n) == 0, "encode ok");
    ASSERT(n == 3, "three distinct terms -> three entries");
    for (size_t i = 0; i < n; i++) ASSERT(e[i].weight > 0.0f, "positive weights");
    splade_free_entries(e);

    /* Repeated term saturates (log(1+tf)) and still yields one entry. */
    ASSERT(splade_encode(enc, "alpha alpha alpha", &e, &n) == 0, "encode repeat");
    ASSERT(n == 1, "one unique term");
    ASSERT(e[0].weight > log1pf(1.0f) - 1e-6f, "tf=3 weight exceeds tf=1 weight");
    splade_free_entries(e);

    splade_destroy(enc);
    return 0;
}

static int test_top_k_truncation(void) {
    GV_SpladeConfig cfg;
    splade_config_init(&cfg);
    cfg.top_k = 3;
    GV_SpladeEncoder *enc = splade_create(&cfg);
    ASSERT(enc != NULL, "create");
    GV_LSSparseEntry *e = NULL;
    size_t n = 0;
    ASSERT(splade_encode(enc, "a b c d e f g", &e, &n) == 0, "encode 7 terms");
    ASSERT(n == 3, "truncated to top_k=3");
    splade_free_entries(e);
    splade_destroy(enc);
    return 0;
}

static int test_end_to_end_search(void) {
    GV_SpladeConfig cfg;
    splade_config_init(&cfg);
    GV_SpladeEncoder *enc = splade_create(&cfg);
    ASSERT(enc != NULL, "create encoder");

    GV_LearnedSparseConfig icfg;
    ls_config_init(&icfg);
    icfg.vocab_size = cfg.vocab_size; /* token ids are hashed into this space */
    GV_LearnedSparseIndex *idx = ls_create(&icfg);
    ASSERT(idx != NULL, "create index");

    int d0 = splade_index_add(enc, idx, "machine learning models for language");
    int d1 = splade_index_add(enc, idx, "deep neural networks and transformers");
    int d2 = splade_index_add(enc, idx, "cooking recipes with fresh vegetables");
    ASSERT(d0 >= 0 && d1 >= 0 && d2 >= 0, "all docs inserted");

    GV_LearnedSparseResult res[3];
    int n = splade_index_search(enc, idx, "deep neural networks learning", 3, res);
    ASSERT(n >= 1, "at least one hit");
    /* The networks/neural/deep doc must rank first. */
    ASSERT(res[0].doc_index == (size_t)d1, "neural-networks doc ranks first");
    /* The unrelated cooking doc must not appear. */
    for (int i = 0; i < n; i++) ASSERT(res[i].doc_index != (size_t)d2, "cooking doc excluded");

    ls_destroy(idx);
    splade_destroy(enc);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_config_defaults();
    rc |= test_encode_fallback();
    rc |= test_top_k_truncation();
    rc |= test_end_to_end_search();
    if (rc == 0) printf("All SPLADE tests passed\n");
    else printf("SPLADE tests FAILED\n");
    return rc ? 1 : 0;
}
