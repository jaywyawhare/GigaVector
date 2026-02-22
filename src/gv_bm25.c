/**
 * @file gv_bm25.c
 * @brief BM25 full-text search implementation.
 */

#include "gigavector/gv_bm25.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

/*  Internal Constants  */

#define TERM_HASH_BUCKETS 4096
#define DOC_HASH_BUCKETS 1024
#define INITIAL_POSTING_CAPACITY 16

/*  Internal Structures  */

/**
 * @brief Posting entry (term occurrence in a document).
 */
typedef struct {
    size_t doc_id;
    size_t term_freq;
} GV_Posting;

/**
 * @brief Posting list for a term.
 */
typedef struct GV_PostingList {
    char *term;
    GV_Posting *postings;
    size_t count;
    size_t capacity;
    struct GV_PostingList *next;
} GV_PostingList;

/**
 * @brief Document info entry.
 */
typedef struct GV_DocInfo {
    size_t doc_id;
    size_t doc_length;              /* Total terms in document */
    struct GV_DocInfo *next;
} GV_DocInfo;

/**
 * @brief BM25 index structure.
 */
struct GV_BM25Index {
    GV_BM25Config config;
    GV_Tokenizer *tokenizer;

    /* Inverted index (term -> posting list) */
    GV_PostingList *term_buckets[TERM_HASH_BUCKETS];
    size_t total_terms;

    /* Document info */
    GV_DocInfo *doc_buckets[DOC_HASH_BUCKETS];
    size_t total_documents;
    size_t total_doc_length;        /* Sum of all document lengths */

    pthread_rwlock_t rwlock;
};

/*  Hash Functions  */

static size_t hash_string(const char *str) {
    size_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static size_t hash_size(size_t val) {
    return val;
}

/*  Configuration  */

static const GV_BM25Config DEFAULT_CONFIG = {
    .k1 = 1.2,
    .b = 0.75,
    .tokenizer = {
        .type = GV_TOKENIZER_SIMPLE,
        .lowercase = 1,
        .remove_stopwords = 0,
        .min_token_length = 1,
        .max_token_length = 256
    }
};

void gv_bm25_config_init(GV_BM25Config *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/*  Lifecycle  */

GV_BM25Index *gv_bm25_create(const GV_BM25Config *config) {
    GV_BM25Index *index = calloc(1, sizeof(GV_BM25Index));
    if (!index) return NULL;

    index->config = config ? *config : DEFAULT_CONFIG;

    index->tokenizer = gv_tokenizer_create(&index->config.tokenizer);
    if (!index->tokenizer) {
        free(index);
        return NULL;
    }

    if (pthread_rwlock_init(&index->rwlock, NULL) != 0) {
        gv_tokenizer_destroy(index->tokenizer);
        free(index);
        return NULL;
    }

    return index;
}

void gv_bm25_destroy(GV_BM25Index *index) {
    if (!index) return;

    /* Free posting lists */
    for (size_t i = 0; i < TERM_HASH_BUCKETS; i++) {
        GV_PostingList *pl = index->term_buckets[i];
        while (pl) {
            GV_PostingList *next = pl->next;
            free(pl->term);
            free(pl->postings);
            free(pl);
            pl = next;
        }
    }

    /* Free document info */
    for (size_t i = 0; i < DOC_HASH_BUCKETS; i++) {
        GV_DocInfo *di = index->doc_buckets[i];
        while (di) {
            GV_DocInfo *next = di->next;
            free(di);
            di = next;
        }
    }

    pthread_rwlock_destroy(&index->rwlock);
    gv_tokenizer_destroy(index->tokenizer);
    free(index);
}

/*  Internal Helpers  */

static GV_PostingList *find_posting_list(GV_BM25Index *index, const char *term) {
    size_t bucket = hash_string(term) % TERM_HASH_BUCKETS;
    GV_PostingList *pl = index->term_buckets[bucket];
    while (pl) {
        if (strcmp(pl->term, term) == 0) {
            return pl;
        }
        pl = pl->next;
    }
    return NULL;
}

static GV_PostingList *get_or_create_posting_list(GV_BM25Index *index, const char *term) {
    size_t bucket = hash_string(term) % TERM_HASH_BUCKETS;

    /* Check existing */
    GV_PostingList *pl = index->term_buckets[bucket];
    while (pl) {
        if (strcmp(pl->term, term) == 0) {
            return pl;
        }
        pl = pl->next;
    }

    /* Create new */
    pl = calloc(1, sizeof(GV_PostingList));
    if (!pl) return NULL;

    pl->term = strdup(term);
    if (!pl->term) {
        free(pl);
        return NULL;
    }

    pl->postings = malloc(INITIAL_POSTING_CAPACITY * sizeof(GV_Posting));
    if (!pl->postings) {
        free(pl->term);
        free(pl);
        return NULL;
    }
    pl->capacity = INITIAL_POSTING_CAPACITY;

    /* Insert at head */
    pl->next = index->term_buckets[bucket];
    index->term_buckets[bucket] = pl;
    index->total_terms++;

    return pl;
}

static GV_DocInfo *find_doc_info(GV_BM25Index *index, size_t doc_id) {
    size_t bucket = hash_size(doc_id) % DOC_HASH_BUCKETS;
    GV_DocInfo *di = index->doc_buckets[bucket];
    while (di) {
        if (di->doc_id == doc_id) {
            return di;
        }
        di = di->next;
    }
    return NULL;
}

static GV_DocInfo *get_or_create_doc_info(GV_BM25Index *index, size_t doc_id) {
    size_t bucket = hash_size(doc_id) % DOC_HASH_BUCKETS;

    /* Check existing */
    GV_DocInfo *di = index->doc_buckets[bucket];
    while (di) {
        if (di->doc_id == doc_id) {
            return di;
        }
        di = di->next;
    }

    /* Create new */
    di = calloc(1, sizeof(GV_DocInfo));
    if (!di) return NULL;

    di->doc_id = doc_id;
    di->next = index->doc_buckets[bucket];
    index->doc_buckets[bucket] = di;
    index->total_documents++;

    return di;
}

static int add_posting(GV_PostingList *pl, size_t doc_id, size_t term_freq) {
    /* Check if document already in list */
    for (size_t i = 0; i < pl->count; i++) {
        if (pl->postings[i].doc_id == doc_id) {
            pl->postings[i].term_freq += term_freq;
            return 0;
        }
    }

    /* Grow if needed */
    if (pl->count >= pl->capacity) {
        size_t new_capacity = pl->capacity * 2;
        GV_Posting *new_postings = realloc(pl->postings, new_capacity * sizeof(GV_Posting));
        if (!new_postings) return -1;
        pl->postings = new_postings;
        pl->capacity = new_capacity;
    }

    pl->postings[pl->count].doc_id = doc_id;
    pl->postings[pl->count].term_freq = term_freq;
    pl->count++;

    return 0;
}

static void remove_doc_from_posting_list(GV_PostingList *pl, size_t doc_id) {
    for (size_t i = 0; i < pl->count; i++) {
        if (pl->postings[i].doc_id == doc_id) {
            /* Shift remaining */
            for (size_t j = i; j < pl->count - 1; j++) {
                pl->postings[j] = pl->postings[j + 1];
            }
            pl->count--;
            return;
        }
    }
}

/*  Indexing Operations  */

int gv_bm25_add_document(GV_BM25Index *index, size_t doc_id, const char *text) {
    if (!index || !text) return -1;

    GV_TokenList tokens;
    if (gv_tokenizer_tokenize(index->tokenizer, text, 0, &tokens) != 0) {
        return -1;
    }

    /* Get unique terms and their frequencies */
    char **unique_terms;
    size_t unique_count;
    if (gv_token_list_unique(&tokens, &unique_terms, &unique_count) != 0) {
        gv_token_list_free(&tokens);
        return -1;
    }

    pthread_rwlock_wrlock(&index->rwlock);

    /* Get or create document info */
    GV_DocInfo *di = get_or_create_doc_info(index, doc_id);
    if (!di) {
        pthread_rwlock_unlock(&index->rwlock);
        gv_unique_tokens_free(unique_terms, unique_count);
        gv_token_list_free(&tokens);
        return -1;
    }

    /* Update document length */
    index->total_doc_length -= di->doc_length;
    di->doc_length = tokens.count;
    index->total_doc_length += di->doc_length;

    /* Add each unique term to posting list */
    for (size_t i = 0; i < unique_count; i++) {
        /* Count term frequency */
        size_t tf = 0;
        for (size_t j = 0; j < tokens.count; j++) {
            if (strcmp(tokens.tokens[j].text, unique_terms[i]) == 0) {
                tf++;
            }
        }

        GV_PostingList *pl = get_or_create_posting_list(index, unique_terms[i]);
        if (pl) {
            add_posting(pl, doc_id, tf);
        }
    }

    pthread_rwlock_unlock(&index->rwlock);

    gv_unique_tokens_free(unique_terms, unique_count);
    gv_token_list_free(&tokens);

    return 0;
}

int gv_bm25_add_document_terms(GV_BM25Index *index, size_t doc_id,
                                const char **terms, size_t term_count) {
    if (!index || !terms || term_count == 0) return -1;

    pthread_rwlock_wrlock(&index->rwlock);

    GV_DocInfo *di = get_or_create_doc_info(index, doc_id);
    if (!di) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    index->total_doc_length -= di->doc_length;
    di->doc_length = term_count;
    index->total_doc_length += di->doc_length;

    for (size_t i = 0; i < term_count; i++) {
        GV_PostingList *pl = get_or_create_posting_list(index, terms[i]);
        if (pl) {
            add_posting(pl, doc_id, 1);
        }
    }

    pthread_rwlock_unlock(&index->rwlock);
    return 0;
}

int gv_bm25_remove_document(GV_BM25Index *index, size_t doc_id) {
    if (!index) return -1;

    pthread_rwlock_wrlock(&index->rwlock);

    /* Find and remove document info */
    size_t bucket = hash_size(doc_id) % DOC_HASH_BUCKETS;
    GV_DocInfo **pp = &index->doc_buckets[bucket];
    GV_DocInfo *di = NULL;

    while (*pp) {
        if ((*pp)->doc_id == doc_id) {
            di = *pp;
            *pp = (*pp)->next;
            break;
        }
        pp = &(*pp)->next;
    }

    if (!di) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    index->total_doc_length -= di->doc_length;
    index->total_documents--;
    free(di);

    /* Remove from all posting lists */
    for (size_t i = 0; i < TERM_HASH_BUCKETS; i++) {
        GV_PostingList *pl = index->term_buckets[i];
        while (pl) {
            remove_doc_from_posting_list(pl, doc_id);
            pl = pl->next;
        }
    }

    pthread_rwlock_unlock(&index->rwlock);
    return 0;
}

int gv_bm25_update_document(GV_BM25Index *index, size_t doc_id, const char *text) {
    gv_bm25_remove_document(index, doc_id);
    return gv_bm25_add_document(index, doc_id, text);
}

/*  BM25 Scoring  */

static double compute_idf(GV_BM25Index *index, size_t doc_freq) {
    double N = (double)index->total_documents;
    double df = (double)doc_freq;
    return log((N - df + 0.5) / (df + 0.5) + 1.0);
}

static double compute_bm25_term_score(GV_BM25Index *index, size_t term_freq,
                                       size_t doc_length, size_t doc_freq) {
    double k1 = index->config.k1;
    double b = index->config.b;
    double avgdl = (double)index->total_doc_length / (double)index->total_documents;

    double tf = (double)term_freq;
    double dl = (double)doc_length;
    double idf = compute_idf(index, doc_freq);

    double tf_component = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avgdl)));

    return idf * tf_component;
}

/*  Search Operations  */

typedef struct {
    size_t doc_id;
    double score;
} DocScore;

static int compare_doc_scores(const void *a, const void *b) {
    const DocScore *da = (const DocScore *)a;
    const DocScore *db = (const DocScore *)b;
    if (db->score > da->score) return 1;
    if (db->score < da->score) return -1;
    return 0;
}

int gv_bm25_search(GV_BM25Index *index, const char *query, size_t k,
                   GV_BM25Result *results) {
    if (!index || !query || !results || k == 0) return -1;

    GV_TokenList tokens;
    if (gv_tokenizer_tokenize(index->tokenizer, query, 0, &tokens) != 0) {
        return -1;
    }

    char **unique_terms;
    size_t unique_count;
    if (gv_token_list_unique(&tokens, &unique_terms, &unique_count) != 0) {
        gv_token_list_free(&tokens);
        return -1;
    }

    int result = gv_bm25_search_terms(index, (const char **)unique_terms, unique_count,
                                       k, results);

    gv_unique_tokens_free(unique_terms, unique_count);
    gv_token_list_free(&tokens);

    return result;
}

int gv_bm25_search_terms(GV_BM25Index *index, const char **terms, size_t term_count,
                          size_t k, GV_BM25Result *results) {
    if (!index || !terms || term_count == 0 || !results || k == 0) return -1;

    pthread_rwlock_rdlock(&index->rwlock);

    if (index->total_documents == 0) {
        pthread_rwlock_unlock(&index->rwlock);
        return 0;
    }

    /* Accumulate scores for all documents */
    DocScore *scores = calloc(index->total_documents, sizeof(DocScore));
    if (!scores) {
        pthread_rwlock_unlock(&index->rwlock);
        return -1;
    }

    /* Initialize document IDs */
    size_t score_count = 0;
    for (size_t i = 0; i < DOC_HASH_BUCKETS; i++) {
        GV_DocInfo *di = index->doc_buckets[i];
        while (di) {
            scores[score_count].doc_id = di->doc_id;
            scores[score_count].score = 0.0;
            score_count++;
            di = di->next;
        }
    }

    /* Score each query term */
    for (size_t t = 0; t < term_count; t++) {
        GV_PostingList *pl = find_posting_list(index, terms[t]);
        if (!pl) continue;

        size_t doc_freq = pl->count;

        for (size_t p = 0; p < pl->count; p++) {
            size_t doc_id = pl->postings[p].doc_id;
            size_t term_freq = pl->postings[p].term_freq;

            /* Find document info */
            GV_DocInfo *di = find_doc_info(index, doc_id);
            if (!di) continue;

            double term_score = compute_bm25_term_score(index, term_freq,
                                                         di->doc_length, doc_freq);

            /* Add to document score */
            for (size_t s = 0; s < score_count; s++) {
                if (scores[s].doc_id == doc_id) {
                    scores[s].score += term_score;
                    break;
                }
            }
        }
    }

    /* Sort by score */
    qsort(scores, score_count, sizeof(DocScore), compare_doc_scores);

    /* Copy top k results */
    size_t result_count = score_count < k ? score_count : k;
    size_t actual_count = 0;
    for (size_t i = 0; i < result_count; i++) {
        if (scores[i].score > 0.0) {
            results[actual_count].doc_id = scores[i].doc_id;
            results[actual_count].score = scores[i].score;
            actual_count++;
        }
    }

    free(scores);
    pthread_rwlock_unlock(&index->rwlock);

    return (int)actual_count;
}

int gv_bm25_score_document(GV_BM25Index *index, size_t doc_id, const char *query,
                            double *score) {
    if (!index || !query || !score) return -1;

    GV_TokenList tokens;
    if (gv_tokenizer_tokenize(index->tokenizer, query, 0, &tokens) != 0) {
        return -1;
    }

    char **unique_terms;
    size_t unique_count;
    if (gv_token_list_unique(&tokens, &unique_terms, &unique_count) != 0) {
        gv_token_list_free(&tokens);
        return -1;
    }

    pthread_rwlock_rdlock(&index->rwlock);

    GV_DocInfo *di = find_doc_info(index, doc_id);
    if (!di) {
        pthread_rwlock_unlock(&index->rwlock);
        gv_unique_tokens_free(unique_terms, unique_count);
        gv_token_list_free(&tokens);
        *score = 0.0;
        return 0;
    }

    *score = 0.0;
    for (size_t t = 0; t < unique_count; t++) {
        GV_PostingList *pl = find_posting_list(index, unique_terms[t]);
        if (!pl) continue;

        /* Find term frequency in this document */
        for (size_t p = 0; p < pl->count; p++) {
            if (pl->postings[p].doc_id == doc_id) {
                *score += compute_bm25_term_score(index, pl->postings[p].term_freq,
                                                   di->doc_length, pl->count);
                break;
            }
        }
    }

    pthread_rwlock_unlock(&index->rwlock);
    gv_unique_tokens_free(unique_terms, unique_count);
    gv_token_list_free(&tokens);

    return 0;
}

/*  Index Information  */

int gv_bm25_get_stats(const GV_BM25Index *index, GV_BM25Stats *stats) {
    if (!index || !stats) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    stats->total_documents = index->total_documents;
    stats->total_terms = index->total_terms;

    /* Count total postings */
    stats->total_postings = 0;
    for (size_t i = 0; i < TERM_HASH_BUCKETS; i++) {
        GV_PostingList *pl = index->term_buckets[i];
        while (pl) {
            stats->total_postings += pl->count;
            pl = pl->next;
        }
    }

    stats->avg_document_length = index->total_documents > 0 ?
        (double)index->total_doc_length / (double)index->total_documents : 0.0;

    /* Estimate memory usage */
    stats->memory_bytes = sizeof(GV_BM25Index);
    for (size_t i = 0; i < TERM_HASH_BUCKETS; i++) {
        GV_PostingList *pl = index->term_buckets[i];
        while (pl) {
            stats->memory_bytes += sizeof(GV_PostingList);
            stats->memory_bytes += strlen(pl->term) + 1;
            stats->memory_bytes += pl->capacity * sizeof(GV_Posting);
            pl = pl->next;
        }
    }
    stats->memory_bytes += index->total_documents * sizeof(GV_DocInfo);

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    return 0;
}

size_t gv_bm25_get_doc_freq(const GV_BM25Index *index, const char *term) {
    if (!index || !term) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);
    GV_PostingList *pl = find_posting_list((GV_BM25Index *)index, term);
    size_t freq = pl ? pl->count : 0;
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);

    return freq;
}

int gv_bm25_has_document(const GV_BM25Index *index, size_t doc_id) {
    if (!index) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);
    int exists = find_doc_info((GV_BM25Index *)index, doc_id) != NULL ? 1 : 0;
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);

    return exists;
}

/*  Persistence  */

int gv_bm25_save(const GV_BM25Index *index, const char *filepath) {
    if (!index || !filepath) return -1;

    FILE *fp = fopen(filepath, "wb");
    if (!fp) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->rwlock);

    /* Write header */
    const char magic[] = "GV_BM25";
    fwrite(magic, 1, 7, fp);

    uint32_t version = 1;
    fwrite(&version, sizeof(version), 1, fp);

    /* Write config */
    fwrite(&index->config.k1, sizeof(index->config.k1), 1, fp);
    fwrite(&index->config.b, sizeof(index->config.b), 1, fp);

    /* Write statistics */
    fwrite(&index->total_documents, sizeof(index->total_documents), 1, fp);
    fwrite(&index->total_terms, sizeof(index->total_terms), 1, fp);
    fwrite(&index->total_doc_length, sizeof(index->total_doc_length), 1, fp);

    /* Write document info */
    for (size_t i = 0; i < DOC_HASH_BUCKETS; i++) {
        GV_DocInfo *di = index->doc_buckets[i];
        while (di) {
            fwrite(&di->doc_id, sizeof(di->doc_id), 1, fp);
            fwrite(&di->doc_length, sizeof(di->doc_length), 1, fp);
            di = di->next;
        }
    }

    /* Write sentinel for end of doc info */
    size_t sentinel = (size_t)-1;
    fwrite(&sentinel, sizeof(sentinel), 1, fp);

    /* Write posting lists */
    for (size_t i = 0; i < TERM_HASH_BUCKETS; i++) {
        GV_PostingList *pl = index->term_buckets[i];
        while (pl) {
            size_t term_len = strlen(pl->term);
            fwrite(&term_len, sizeof(term_len), 1, fp);
            fwrite(pl->term, 1, term_len, fp);
            fwrite(&pl->count, sizeof(pl->count), 1, fp);
            fwrite(pl->postings, sizeof(GV_Posting), pl->count, fp);
            pl = pl->next;
        }
    }

    /* Write sentinel for end of posting lists */
    size_t zero = 0;
    fwrite(&zero, sizeof(zero), 1, fp);

    pthread_rwlock_unlock((pthread_rwlock_t *)&index->rwlock);
    fclose(fp);

    return 0;
}

GV_BM25Index *gv_bm25_load(const char *filepath) {
    if (!filepath) return NULL;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) return NULL;

    /* Read and verify header */
    char magic[7];
    if (fread(magic, 1, 7, fp) != 7 || memcmp(magic, "GV_BM25", 7) != 0) {
        fclose(fp);
        return NULL;
    }

    uint32_t version;
    if (fread(&version, sizeof(version), 1, fp) != 1 || version != 1) {
        fclose(fp);
        return NULL;
    }

    /* Read config */
    GV_BM25Config config;
    gv_bm25_config_init(&config);
    fread(&config.k1, sizeof(config.k1), 1, fp);
    fread(&config.b, sizeof(config.b), 1, fp);

    GV_BM25Index *index = gv_bm25_create(&config);
    if (!index) {
        fclose(fp);
        return NULL;
    }

    /* Read statistics */
    fread(&index->total_documents, sizeof(index->total_documents), 1, fp);
    fread(&index->total_terms, sizeof(index->total_terms), 1, fp);
    fread(&index->total_doc_length, sizeof(index->total_doc_length), 1, fp);

    /* Reset counters (will be rebuilt) */
    size_t expected_docs = index->total_documents;
    size_t expected_terms = index->total_terms;
    index->total_documents = 0;
    index->total_terms = 0;

    /* Read document info */
    while (1) {
        size_t doc_id;
        if (fread(&doc_id, sizeof(doc_id), 1, fp) != 1) break;
        if (doc_id == (size_t)-1) break;

        size_t doc_length;
        fread(&doc_length, sizeof(doc_length), 1, fp);

        GV_DocInfo *di = get_or_create_doc_info(index, doc_id);
        if (di) {
            di->doc_length = doc_length;
        }
    }

    /* Read posting lists */
    while (1) {
        size_t term_len;
        if (fread(&term_len, sizeof(term_len), 1, fp) != 1) break;
        if (term_len == 0) break;

        char *term = malloc(term_len + 1);
        if (!term) break;
        fread(term, 1, term_len, fp);
        term[term_len] = '\0';

        size_t posting_count;
        fread(&posting_count, sizeof(posting_count), 1, fp);

        GV_PostingList *pl = get_or_create_posting_list(index, term);
        if (pl && posting_count > 0) {
            if (posting_count > pl->capacity) {
                GV_Posting *new_postings = realloc(pl->postings, posting_count * sizeof(GV_Posting));
                if (new_postings) {
                    pl->postings = new_postings;
                    pl->capacity = posting_count;
                }
            }
            fread(pl->postings, sizeof(GV_Posting), posting_count, fp);
            pl->count = posting_count;
        }

        free(term);
    }

    fclose(fp);

    /* Verify */
    if (index->total_documents != expected_docs) {
        /* Warning: document count mismatch */
    }
    (void)expected_terms;  /* Suppress unused warning */

    return index;
}
