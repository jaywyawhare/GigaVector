/*
 * SPLADE learned-sparse text encoder.
 *
 * Bridges raw text to the (token_id, weight) entries consumed by the
 * learned-sparse index. See include/multimodal/splade.h for the two back-ends
 * (neural ONNX SPLADE, and a deterministic log(1+tf) fallback).
 */

#include "multimodal/splade.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "core/utils.h"
#include "multimodal/tokenizer.h"

#ifdef GV_HAVE_ONNX
#include "multimodal/onnx.h"
#endif

#define GV_SPLADE_DEFAULT_VOCAB 30522u
#define GV_SPLADE_DEFAULT_TOPK  128u

struct GV_SpladeEncoder {
    GV_SpladeConfig cfg;
    GV_Tokenizer *tok;
#ifdef GV_HAVE_ONNX
    GV_ONNXModel *model;
#endif
    int neural;
};

void splade_config_init(GV_SpladeConfig *config) {
    if (!config) return;
    config->vocab_size = GV_SPLADE_DEFAULT_VOCAB;
    config->top_k = GV_SPLADE_DEFAULT_TOPK;
    config->onnx_model_path = NULL;
}

GV_SpladeEncoder *splade_create(const GV_SpladeConfig *config) {
    GV_SpladeEncoder *enc = (GV_SpladeEncoder *)gv_calloc(1, sizeof(*enc));
    if (!enc) return NULL;
    if (config) {
        enc->cfg = *config;
    } else {
        splade_config_init(&enc->cfg);
    }
    if (enc->cfg.vocab_size == 0) enc->cfg.vocab_size = GV_SPLADE_DEFAULT_VOCAB;
    if (enc->cfg.top_k == 0) enc->cfg.top_k = GV_SPLADE_DEFAULT_TOPK;

    GV_TokenizerConfig tcfg;
    tokenizer_config_init(&tcfg);
    tcfg.type = GV_TOKENIZER_SIMPLE;
    tcfg.lowercase = 1;
    enc->tok = tokenizer_create(&tcfg);
    if (!enc->tok) { gv_free(enc); return NULL; }

#ifdef GV_HAVE_ONNX
    if (enc->cfg.onnx_model_path) {
        GV_ONNXConfig ocfg = {0};
        ocfg.model_path = enc->cfg.onnx_model_path;
        ocfg.num_threads = 4;
        ocfg.optimization_level = 2;
        enc->model = onnx_load(&ocfg);
        enc->neural = enc->model ? 1 : 0;
    }
#endif
    return enc;
}

void splade_destroy(GV_SpladeEncoder *enc) {
    if (!enc) return;
#ifdef GV_HAVE_ONNX
    if (enc->model) onnx_destroy(enc->model);
#endif
    tokenizer_destroy(enc->tok);
    gv_free(enc);
}

int splade_is_neural(const GV_SpladeEncoder *enc) {
    return enc ? enc->neural : 0;
}

/* Sort entries by descending weight (then ascending token for stability). */
static int entry_cmp_desc(const void *a, const void *b) {
    const GV_LSSparseEntry *x = (const GV_LSSparseEntry *)a;
    const GV_LSSparseEntry *y = (const GV_LSSparseEntry *)b;
    if (x->weight < y->weight) return 1;
    if (x->weight > y->weight) return -1;
    if (x->token_id < y->token_id) return -1;
    if (x->token_id > y->token_id) return 1;
    return 0;
}

/* Keep only the top_k highest-weighted entries (entries is sorted in place). */
static void truncate_top_k(GV_LSSparseEntry *entries, size_t *count, size_t top_k) {
    if (*count <= top_k) return;
    qsort(entries, *count, sizeof(*entries), entry_cmp_desc);
    *count = top_k;
}

/* ---- deterministic fallback: log(1 + tf) over hashed vocabulary slots ---- */

typedef struct { uint32_t token; float tf; int used; } FallbackSlot;

static int encode_fallback(GV_SpladeEncoder *enc, const char *text,
                           GV_LSSparseEntry **out, size_t *out_count) {
    GV_TokenList toks;
    memset(&toks, 0, sizeof(toks));
    if (tokenizer_tokenize(enc->tok, text, 0, &toks) != 0) return -1;

    /* Open-addressing map token_id -> accumulated term frequency. */
    size_t cap = 1;
    while (cap < toks.count * 2 + 8) cap <<= 1;
    FallbackSlot *map = (FallbackSlot *)gv_calloc(cap, sizeof(*map));
    if (!map) { token_list_free(&toks); return -1; }

    size_t unique = 0;
    for (size_t i = 0; i < toks.count; i++) {
        uint32_t id = (uint32_t)(hash_str(toks.tokens[i].text) % enc->cfg.vocab_size);
        size_t h = id & (cap - 1);
        while (map[h].used && map[h].token != id) h = (h + 1) & (cap - 1);
        if (!map[h].used) { map[h].used = 1; map[h].token = id; unique++; }
        map[h].tf += 1.0f;
    }
    token_list_free(&toks);

    GV_LSSparseEntry *entries = (GV_LSSparseEntry *)gv_alloc((unique ? unique : 1) * sizeof(*entries));
    if (!entries) { gv_free(map); return -1; }
    size_t n = 0;
    for (size_t h = 0; h < cap; h++) {
        if (!map[h].used) continue;
        entries[n].token_id = map[h].token;
        entries[n].weight = log1pf(map[h].tf); /* SPLADE-like saturation */
        n++;
    }
    gv_free(map);

    truncate_top_k(entries, &n, enc->cfg.top_k);
    *out = entries;
    *out_count = n;
    return 0;
}

/* ---- neural path: SPLADE term weights from an ONNX model ---- */

#ifdef GV_HAVE_ONNX
static int encode_neural(GV_SpladeEncoder *enc, const char *text,
                         GV_LSSparseEntry **out, size_t *out_count) {
    size_t vocab = enc->cfg.vocab_size;
    float *weights = (float *)gv_calloc(vocab, sizeof(float));
    if (!weights) return -1;

    /* A SPLADE MLM export maps a text to one vocab-sized term-weight vector
     * (the log(1+ReLU(.)) max-pool is baked into the graph). */
    const char *texts[1] = { text };
    if (onnx_embed(enc->model, texts, 1, weights, vocab) != 0) {
        gv_free(weights);
        return -1;
    }

    GV_LSSparseEntry *entries = (GV_LSSparseEntry *)gv_alloc(vocab * sizeof(*entries));
    if (!entries) { gv_free(weights); return -1; }
    size_t n = 0;
    for (size_t j = 0; j < vocab; j++) {
        float w = weights[j];
        if (w > 0.0f) { /* ReLU: keep positive term weights only */
            entries[n].token_id = (uint32_t)j;
            entries[n].weight = w;
            n++;
        }
    }
    gv_free(weights);

    truncate_top_k(entries, &n, enc->cfg.top_k);
    *out = entries;
    *out_count = n;
    return 0;
}
#endif

int splade_encode(GV_SpladeEncoder *enc, const char *text,
                  GV_LSSparseEntry **out, size_t *out_count) {
    if (!enc || !text || !out || !out_count) return -1;
    *out = NULL;
    *out_count = 0;
#ifdef GV_HAVE_ONNX
    if (enc->neural) return encode_neural(enc, text, out, out_count);
#endif
    return encode_fallback(enc, text, out, out_count);
}

void splade_free_entries(GV_LSSparseEntry *entries) {
    gv_free(entries);
}

int splade_index_add(GV_SpladeEncoder *enc, GV_LearnedSparseIndex *idx,
                     const char *text) {
    if (!enc || !idx || !text) return -1;
    GV_LSSparseEntry *entries = NULL;
    size_t count = 0;
    if (splade_encode(enc, text, &entries, &count) != 0) return -1;
    int doc_id = ls_insert(idx, entries, count);
    splade_free_entries(entries);
    return doc_id;
}

int splade_index_search(GV_SpladeEncoder *enc, const GV_LearnedSparseIndex *idx,
                        const char *query_text, size_t k,
                        GV_LearnedSparseResult *results) {
    if (!enc || !idx || !query_text || !results) return -1;
    GV_LSSparseEntry *entries = NULL;
    size_t count = 0;
    if (splade_encode(enc, query_text, &entries, &count) != 0) return -1;
    int n = ls_search(idx, entries, count, k, results);
    splade_free_entries(entries);
    return n;
}
