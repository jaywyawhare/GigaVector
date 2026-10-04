/**
 * @file cross_encoder_rerank.c
 * @brief Cross-encoder reranking with ONNX and fallback implementations.
 *
 * When an ONNX model is loaded, delegates to onnx_rerank() for true
 * neural cross-encoder scoring. When no model is available (NULL path or
 * ONNX Runtime not compiled in), falls back to a fast TF-IDF-like term
 * overlap score that approximates relevance.
 */

#include "search/cross_encoder_rerank.h"
#include "multimodal/onnx.h"
#include "core/memory.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define CE_MAX_TOKEN_LEN 128
#define CE_MAX_TERMS     512

struct GV_CrossEncoder {
    GV_ONNXModel *model;
    size_t        max_seq_len;
    int           using_fallback;
};

/* ── Simple term bag for fallback scoring ──────────────────────────────── */

typedef struct {
    char  words[CE_MAX_TERMS][CE_MAX_TOKEN_LEN];
    float tf[CE_MAX_TERMS];
    int   count;
} TermBag;

static void termbag_init(TermBag *bag) {
    bag->count = 0;
}

static void termbag_add(TermBag *bag, const char *word) {
    if (bag->count >= CE_MAX_TERMS) return;

    for (int i = 0; i < bag->count; i++) {
        if (strcmp(bag->words[i], word) == 0) {
            bag->tf[i] += 1.0f;
            return;
        }
    }

    size_t len = strlen(word);
    if (len >= CE_MAX_TOKEN_LEN) len = CE_MAX_TOKEN_LEN - 1;
    memcpy(bag->words[bag->count], word, len);
    bag->words[bag->count][len] = '\0';
    bag->tf[bag->count] = 1.0f;
    bag->count++;
}

static float termbag_tf(const TermBag *bag, const char *word) {
    for (int i = 0; i < bag->count; i++) {
        if (strcmp(bag->words[i], word) == 0) return bag->tf[i];
    }
    return 0.0f;
}

static void tokenize_to_bag(const char *text, TermBag *bag) {
    termbag_init(bag);
    if (!text) return;

    const char *p = text;
    while (*p) {
        while (*p && !isalpha((unsigned char)*p)) p++;
        if (!*p) break;

        char token[CE_MAX_TOKEN_LEN];
        size_t tlen = 0;
        while (*p && isalpha((unsigned char)*p) &&
               tlen < CE_MAX_TOKEN_LEN - 1) {
            token[tlen++] = (char)tolower((unsigned char)*p);
            p++;
        }
        token[tlen] = '\0';
        if (tlen >= 3) {
            termbag_add(bag, token);
        }
    }
}

/**
 * @brief TF-IDF overlap score of a document against the query.
 *
 * Per query term: document_tf * idf[term], where idf is the real inverse
 * document frequency computed across the reranked batch (see the batch loop).
 * Normalised by query length.
 */
static float fallback_score(const TermBag *query, const TermBag *doc, const float *idf) {
    if (query->count == 0) return 0.0f;

    float score = 0.0f;
    for (int i = 0; i < query->count; i++) {
        float doc_tf = termbag_tf(doc, query->words[i]);
        if (doc_tf > 0.0f) score += doc_tf * idf[i];
    }

    return score / (float)query->count;
}

/* ── Public API ────────────────────────────────────────────────────────── */

GV_CrossEncoder *cross_encoder_create(const char *model_path, size_t max_seq_len) {
    GV_CrossEncoder *ce = (GV_CrossEncoder *)gv_calloc(1, sizeof(GV_CrossEncoder));
    if (!ce) return NULL;

    ce->max_seq_len = max_seq_len > 0 ? max_seq_len : 512;

    if (model_path && onnx_available()) {
        GV_ONNXConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.model_path    = model_path;
        cfg.num_threads   = 4;
        cfg.optimization_level = 2;

        ce->model = onnx_load(&cfg);
        if (ce->model) {
            ce->using_fallback = 0;
            return ce;
        }
    }

    /* Fallback mode: no ONNX model available */
    ce->model = NULL;
    ce->using_fallback = 1;
    return ce;
}

void cross_encoder_destroy(GV_CrossEncoder *ce) {
    if (!ce) return;
    if (ce->model) onnx_destroy(ce->model);
    gv_free(ce);
}

int cross_encoder_rerank_batch(GV_CrossEncoder *ce, const char *query,
                               const char **document_texts, size_t doc_count,
                               float *out_scores) {
    if (!ce || !query) return -1;
    if (doc_count == 0) return 0;
    if (!document_texts || !out_scores) return -1;

    if (!ce->using_fallback && ce->model) {
        return onnx_rerank(ce->model, query, document_texts, doc_count, out_scores);
    }

    /* Fallback: TF-IDF overlap scoring with a real IDF computed over this batch
     * (the only corpus the reranker has). Memory stays constant in doc_count:
     * pass 1 counts document frequency per query term with one reused bag; pass
     * 2 re-tokenises each document to score it. */
    TermBag qbag;
    tokenize_to_bag(query, &qbag);

    size_t df[CE_MAX_TERMS];
    for (int t = 0; t < qbag.count; t++) df[t] = 0;
    for (size_t i = 0; i < doc_count; i++) {
        TermBag dbag;
        tokenize_to_bag(document_texts[i], &dbag);
        for (int t = 0; t < qbag.count; t++)
            if (termbag_tf(&dbag, qbag.words[t]) > 0.0f) df[t]++;
    }

    float idf[CE_MAX_TERMS];
    for (int t = 0; t < qbag.count; t++) /* smoothed IDF: rarer-in-batch weighs more, always >= 0 */
        idf[t] = logf(((float)doc_count + 1.0f) / ((float)df[t] + 1.0f)) + 1.0f;

    for (size_t i = 0; i < doc_count; i++) {
        TermBag dbag;
        tokenize_to_bag(document_texts[i], &dbag);
        out_scores[i] = fallback_score(&qbag, &dbag, idf);
    }

    return 0;
}

/* Comparison struct for sorting results with their scores. */
typedef struct {
    size_t index;
    float  score;
} CEScoreEntry;

static int compare_scores_desc(const void *a, const void *b) {
    const CEScoreEntry *ea = (const CEScoreEntry *)a;
    const CEScoreEntry *eb = (const CEScoreEntry *)b;
    if (eb->score > ea->score) return  1;
    if (eb->score < ea->score) return -1;
    /* Stable tiebreak: preserve original result order for equal scores. */
    if (ea->index < eb->index) return -1;
    if (ea->index > eb->index) return  1;
    return 0;
}

int cross_encoder_rerank(GV_CrossEncoder *ce, const char *query,
                         GV_SearchResult *results, size_t count,
                         GV_CrossEncoderTextGetter text_getter, void *user_data) {
    if (!ce || !query || !results || !text_getter) return -1;
    if (count == 0) return 0;

    /* Extract texts via callback */
    const char **texts = (const char **)gv_calloc(count, sizeof(char *));
    CEScoreEntry *entries = (CEScoreEntry *)gv_calloc(count, sizeof(CEScoreEntry));
    if (!texts || !entries) {
        gv_free(texts);
        gv_free(entries);
        return -1;
    }

    for (size_t i = 0; i < count; i++) {
        texts[i] = text_getter(i, user_data);
        if (!texts[i]) texts[i] = "";
        entries[i].index = i;
    }

    /* Score all documents */
    float *scores = (float *)gv_calloc(count, sizeof(float));
    if (!scores) {
        gv_free(texts);
        gv_free(entries);
        return -1;
    }

    int rc = cross_encoder_rerank_batch(ce, query, texts, count, scores);
    if (rc != 0) {
        gv_free(texts);
        gv_free(entries);
        gv_free(scores);
        return -1;
    }

    /* Record scores and build sortable entries */
    for (size_t i = 0; i < count; i++) {
        entries[i].score = scores[i];
    }

    /* Sort entries by descending score */
    qsort(entries, count, sizeof(CEScoreEntry), compare_scores_desc);

    /* Reorder results in-place and update distance to reflect new score.
     * We use a scratch buffer to avoid overwriting results we haven't read yet. */
    GV_SearchResult *tmp = (GV_SearchResult *)gv_calloc(count, sizeof(GV_SearchResult));
    if (!tmp) {
        gv_free(texts);
        gv_free(entries);
        gv_free(scores);
        return -1;
    }

    for (size_t i = 0; i < count; i++) {
        tmp[i] = results[entries[i].index];
        /* Store reranker score in distance field (convention: higher = better here,
         * unlike raw distance where lower = better; caller should interpret). */
        tmp[i].distance = entries[i].score;
    }

    memcpy(results, tmp, count * sizeof(GV_SearchResult));

    gv_free(tmp);
    gv_free(texts);
    gv_free(entries);
    gv_free(scores);

    return 0;
}
