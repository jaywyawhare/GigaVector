/**
 * @file chunker.c
 * @brief Word-based document chunker with overlap and deterministic chunk_ids.
 */
#include "storage/chunker.h"
#include "core/memory.h"
#include "specialized/point_id.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define GV_CHUNK_DEFAULT_TOKENS  512
#define GV_CHUNK_DEFAULT_OVERLAP 128

/* A word span within the source text. */
typedef struct { size_t start; size_t end; } WordSpan;

static int is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

/* Tokenize into word spans (whitespace-delimited). Returns count, fills *out. */
static size_t tokenize(const char *text, WordSpan **out) {
    size_t cap = 64, n = 0;
    WordSpan *w = (WordSpan *)gv_alloc(cap * sizeof(WordSpan));
    if (!w) return 0;
    size_t i = 0;
    while (text[i]) {
        while (text[i] && is_space(text[i])) i++;
        if (!text[i]) break;
        size_t s = i;
        while (text[i] && !is_space(text[i])) i++;
        if (n == cap) {
            cap *= 2;
            WordSpan *nw = (WordSpan *)gv_realloc(w, cap * sizeof(WordSpan));
            if (!nw) { gv_free(w); return 0; }
            w = nw;
        }
        w[n].start = s;
        w[n].end = i;
        n++;
    }
    *out = w;
    return n;
}

void gv_chunks_free(GV_Chunk *chunks, size_t count) {
    if (!chunks) return;
    for (size_t i = 0; i < count; i++) gv_free(chunks[i].text);
    gv_free(chunks);
}

int gv_chunk_document(const char *text, size_t chunk_tokens, size_t overlap_tokens,
                      const char *doc_id, GV_Chunk **out_chunks, size_t *out_count) {
    if (!text || !out_chunks || !out_count) return -1;
    if (chunk_tokens == 0)   chunk_tokens = GV_CHUNK_DEFAULT_TOKENS;
    if (overlap_tokens >= chunk_tokens) overlap_tokens = chunk_tokens / 4;

    /* Resolve doc_id (generate a uuid-derived one if not supplied). */
    char doc[32];
    if (doc_id && doc_id[0]) {
        snprintf(doc, sizeof(doc), "%s", doc_id);
    } else {
        char uuid[40];
        if (point_id_generate_uuid(uuid, sizeof(uuid)) != 0) {
            snprintf(doc, sizeof(doc), "doc");
        } else {
            /* strip dashes, take up to 16 hex chars */
            size_t j = 0;
            for (size_t i = 0; uuid[i] && j < 16; i++) {
                if (uuid[i] != '-') doc[j++] = uuid[i];
            }
            doc[j] = '\0';
        }
    }

    WordSpan *words = NULL;
    size_t nwords = tokenize(text, &words);
    if (nwords == 0) { gv_free(words); *out_chunks = NULL; *out_count = 0; return 0; }

    size_t stride = chunk_tokens - overlap_tokens;
    if (stride == 0) stride = chunk_tokens;

    size_t cap = 8, nc = 0;
    GV_Chunk *chunks = (GV_Chunk *)gv_alloc(cap * sizeof(GV_Chunk));
    if (!chunks) { gv_free(words); return -1; }

    for (size_t start_w = 0; start_w < nwords; start_w += stride) {
        size_t end_w = start_w + chunk_tokens;
        if (end_w > nwords) end_w = nwords;

        size_t cs = words[start_w].start;
        size_t ce = words[end_w - 1].end;
        size_t len = ce - cs;

        char *ctext = (char *)gv_alloc(len + 1);
        if (!ctext) { gv_chunks_free(chunks, nc); gv_free(words); return -1; }
        memcpy(ctext, text + cs, len);
        ctext[len] = '\0';

        if (nc == cap) {
            cap *= 2;
            GV_Chunk *nchunks = (GV_Chunk *)gv_realloc(chunks, cap * sizeof(GV_Chunk));
            if (!nchunks) { gv_free(ctext); gv_chunks_free(chunks, nc); gv_free(words); return -1; }
            chunks = nchunks;
        }

        GV_Chunk *ck = &chunks[nc];
        snprintf(ck->doc_id, sizeof(ck->doc_id), "%s", doc);
        snprintf(ck->chunk_id, sizeof(ck->chunk_id), "%s:%04zu", doc, nc);
        ck->chunk_index = nc;
        ck->text = ctext;
        ck->char_start = cs;
        ck->char_end = ce;
        nc++;

        if (end_w == nwords) break; /* last chunk reached */
    }

    gv_free(words);
    *out_chunks = chunks;
    *out_count = nc;
    return 0;
}
