/**
 * @file gv_fulltext.c
 * @brief Full-text search implementation with Porter stemming, phrase matching,
 *        multilingual tokenization, and BlockMax WAND optimization.
 */

#include "gigavector/gv_fulltext.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <pthread.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define FT_TERM_HASH_BUCKETS  4096
#define FT_DOC_HASH_BUCKETS   1024
#define FT_INITIAL_POSTING_CAP 16
#define FT_INITIAL_POS_CAP     8
#define FT_MAX_WORD_LEN       256
#define FT_BM25_K1            1.2f
#define FT_BM25_B             0.75f

/* ============================================================================
 * Stopword Lists
 * ============================================================================ */

static const char *STOPWORDS_EN[] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
    "into", "is", "it", "its", "no", "not", "of", "on", "or", "our",
    "own", "she", "so", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "up", "us", "was", "we",
    "were", "what", "when", "which", "who", "will", "with", "would",
    "you", "your", NULL
};

static const char *STOPWORDS_DE[] = {
    "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin",
    "bis", "bist", "da", "damit", "dann", "das", "dass", "dein",
    "dem", "den", "der", "des", "die", "dies", "doch", "du", "durch",
    "ein", "eine", "einem", "einen", "einer", "er", "es", "etwas",
    "fur", "hat", "ich", "ihm", "ihn", "ihr", "im", "in", "ist",
    "ja", "kann", "kein", "mein", "mit", "nach", "nicht", "noch",
    "nun", "nur", "ob", "oder", "ohne", "sein", "sich", "sie", "sind",
    "so", "und", "uns", "von", "was", "wer", "wie", "wir", "zu",
    "zum", "zur", NULL
};

static const char *STOPWORDS_FR[] = {
    "a", "au", "aux", "avec", "c", "ce", "ces", "dans", "de", "des",
    "du", "elle", "en", "est", "et", "eu", "il", "je", "la", "le",
    "les", "leur", "lui", "ma", "mais", "me", "mon", "ne", "ni",
    "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu",
    "que", "qui", "sa", "se", "ses", "si", "son", "sur", "ta", "te",
    "tes", "ton", "tu", "un", "une", "vos", "votre", "vous", "y", NULL
};

static const char *STOPWORDS_ES[] = {
    "a", "al", "algo", "con", "de", "del", "el", "ella", "en", "es",
    "esa", "ese", "esta", "este", "hay", "la", "las", "lo", "los",
    "me", "mi", "muy", "no", "nos", "o", "para", "pero", "por", "que",
    "se", "si", "sin", "su", "sus", "te", "tu", "tus", "un", "una",
    "usted", "y", "yo", NULL
};

static const char *STOPWORDS_IT[] = {
    "a", "ai", "al", "alle", "allo", "anche", "che", "chi", "ci",
    "con", "cui", "da", "dal", "dei", "del", "della", "delle", "di",
    "e", "gli", "ha", "ho", "i", "il", "in", "io", "la", "le", "lei",
    "li", "lo", "lui", "ma", "me", "mi", "mia", "mio", "ne", "no",
    "noi", "non", "o", "per", "poi", "se", "si", "su", "suo", "tra",
    "tu", "tuo", "un", "una", "uno", "vi", "voi", NULL
};

static const char *STOPWORDS_PT[] = {
    "a", "ao", "aos", "as", "com", "da", "das", "de", "do", "dos",
    "e", "ela", "ele", "em", "era", "essa", "esse", "esta", "este",
    "eu", "foi", "ja", "lhe", "me", "meu", "na", "nas", "nao", "no",
    "nos", "o", "os", "ou", "para", "pela", "pelo", "por", "que", "se",
    "sem", "ser", "seu", "sua", "te", "teu", "tu", "um", "uma", "vai",
    "vou", NULL
};

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Position list for a term within a single document.
 */
typedef struct {
    size_t *positions;              /* Ordered array of token positions. */
    size_t count;
    size_t capacity;
} FT_PositionList;

/**
 * @brief Single posting entry (term occurrence in one document).
 */
typedef struct {
    size_t doc_id;
    size_t term_freq;
    FT_PositionList pos;            /* Positions (populated when phrase match enabled). */
} FT_Posting;

/**
 * @brief Block-level upper-bound metadata for BlockMax WAND.
 */
typedef struct {
    float *block_maxes;             /* Max BM25 contribution per block. */
    size_t block_count;
} FT_BlockMax;

/**
 * @brief Posting list for a single term across all documents.
 */
typedef struct FT_PostingList {
    char *term;
    FT_Posting *postings;
    size_t count;
    size_t capacity;
    FT_BlockMax bmax;               /* BlockMax metadata (lazily built). */
    struct FT_PostingList *next;    /* Hash chain. */
} FT_PostingList;

/**
 * @brief Document metadata entry.
 */
typedef struct FT_DocInfo {
    size_t doc_id;
    size_t doc_length;              /* Total tokens in document. */
    struct FT_DocInfo *next;        /* Hash chain. */
} FT_DocInfo;

/**
 * @brief Full-text index structure.
 */
struct GV_FTIndex {
    GV_FTConfig config;

    /* Inverted index: term -> posting list */
    FT_PostingList *term_buckets[FT_TERM_HASH_BUCKETS];
    size_t total_terms;

    /* Document metadata */
    FT_DocInfo *doc_buckets[FT_DOC_HASH_BUCKETS];
    size_t total_documents;
    size_t total_doc_length;        /* Sum of all document lengths. */

    pthread_rwlock_t rwlock;
};

/* ============================================================================
 * Hash Functions
 * ============================================================================ */

static size_t ft_hash_string(const char *str) {
    size_t hash = 5381;
    int c;
    while ((c = (unsigned char)*str++)) {
        hash = ((hash << 5) + hash) + (size_t)c;
    }
    return hash;
}

static size_t ft_hash_size(size_t val) {
    return val;
}

/* ============================================================================
 * Stopword Helpers
 * ============================================================================ */

static const char **ft_stopwords_for_lang(GV_FTLanguage lang) {
    switch (lang) {
        case GV_LANG_ENGLISH: return STOPWORDS_EN;
        case GV_LANG_GERMAN:  return STOPWORDS_DE;
        case GV_LANG_FRENCH:  return STOPWORDS_FR;
        case GV_LANG_SPANISH: return STOPWORDS_ES;
        case GV_LANG_ITALIAN: return STOPWORDS_IT;
        case GV_LANG_PORTUGUESE: return STOPWORDS_PT;
        case GV_LANG_AUTO:    return STOPWORDS_EN;
        default:              return STOPWORDS_EN;
    }
}

static int ft_is_stopword(const char *word, GV_FTLanguage lang) {
    const char **list = ft_stopwords_for_lang(lang);
    for (size_t i = 0; list[i]; i++) {
        if (strcmp(word, list[i]) == 0) return 1;
    }
    return 0;
}

/* ============================================================================
 * Porter Stemmer (English)
 * ============================================================================ */

/**
 * Measure: the number of consonant-vowel sequences in word[0..k].
 */
static int porter_measure(const char *w, int k) {
    int n = 0;
    int i = 0;
    /* Skip initial consonants */
    while (i <= k) {
        char c = (char)tolower((unsigned char)w[i]);
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u')
            break;
        i++;
    }
    while (i <= k) {
        /* Skip vowels */
        while (i <= k) {
            char c = (char)tolower((unsigned char)w[i]);
            if (!(c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'))
                break;
            i++;
        }
        if (i > k) break;
        n++;
        /* Skip consonants */
        while (i <= k) {
            char c = (char)tolower((unsigned char)w[i]);
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u')
                break;
            i++;
        }
    }
    return n;
}

static int porter_is_vowel(char c) {
    c = (char)tolower((unsigned char)c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

static int porter_has_vowel(const char *w, int k) {
    for (int i = 0; i <= k; i++) {
        if (porter_is_vowel(w[i])) return 1;
    }
    return 0;
}

static int porter_ends_double_consonant(const char *w, int k) {
    if (k < 1) return 0;
    if (w[k] != w[k - 1]) return 0;
    return !porter_is_vowel(w[k]);
}

static int porter_cvc(const char *w, int k) {
    if (k < 2) return 0;
    if (porter_is_vowel(w[k])) return 0;
    if (!porter_is_vowel(w[k - 1])) return 0;
    if (porter_is_vowel(w[k - 2])) return 0;
    char c = (char)tolower((unsigned char)w[k]);
    if (c == 'w' || c == 'x' || c == 'y') return 0;
    return 1;
}

static int porter_ends_with(const char *w, int k, const char *suffix) {
    int slen = (int)strlen(suffix);
    if (slen > k + 1) return 0;
    return memcmp(w + k + 1 - slen, suffix, (size_t)slen) == 0;
}

static void porter_set_end(char *w, int *k, const char *suffix, const char *replacement) {
    int slen = (int)strlen(suffix);
    int rlen = (int)strlen(replacement);
    int base = *k + 1 - slen;
    memcpy(w + base, replacement, (size_t)rlen);
    *k = base + rlen - 1;
    w[*k + 1] = '\0';
}

/**
 * Porter stemmer step 1a: plurals.
 */
static void porter_step1a(char *w, int *k) {
    if (porter_ends_with(w, *k, "sses")) {
        *k -= 2; w[*k + 1] = '\0';
    } else if (porter_ends_with(w, *k, "ies")) {
        porter_set_end(w, k, "ies", "i");
    } else if (!porter_ends_with(w, *k, "ss") && porter_ends_with(w, *k, "s")) {
        (*k)--; w[*k + 1] = '\0';
    }
}

/**
 * Porter stemmer step 1b: -ed, -ing.
 */
static void porter_step1b(char *w, int *k) {
    if (porter_ends_with(w, *k, "eed")) {
        if (porter_measure(w, *k - 3) > 0) {
            (*k)--; w[*k + 1] = '\0';
        }
        return;
    }
    int changed = 0;
    if (porter_ends_with(w, *k, "ed") && porter_has_vowel(w, *k - 2)) {
        *k -= 2; w[*k + 1] = '\0'; changed = 1;
    } else if (porter_ends_with(w, *k, "ing") && porter_has_vowel(w, *k - 3)) {
        *k -= 3; w[*k + 1] = '\0'; changed = 1;
    }
    if (changed) {
        if (porter_ends_with(w, *k, "at") || porter_ends_with(w, *k, "bl") ||
            porter_ends_with(w, *k, "iz")) {
            (*k)++; w[*k] = 'e'; w[*k + 1] = '\0';
        } else if (porter_ends_double_consonant(w, *k) &&
                   w[*k] != 'l' && w[*k] != 's' && w[*k] != 'z') {
            w[*k] = '\0'; (*k)--;
        } else if (porter_measure(w, *k) == 1 && porter_cvc(w, *k)) {
            (*k)++; w[*k] = 'e'; w[*k + 1] = '\0';
        }
    }
}

/**
 * Porter stemmer step 1c: y -> i after vowel.
 */
static void porter_step1c(char *w, int *k) {
    if (*k > 0 && (w[*k] == 'y' || w[*k] == 'Y') && porter_has_vowel(w, *k - 1)) {
        w[*k] = 'i';
    }
}

/**
 * Porter stemmer step 2: double suffixes mapped to single.
 */
static void porter_step2(char *w, int *k) {
    if (*k < 1) return;
    struct { const char *from; const char *to; } rules[] = {
        {"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"},
        {"anci", "ance"}, {"izer", "ize"}, {"abli", "able"},
        {"alli", "al"}, {"entli", "ent"}, {"eli", "e"},
        {"ousli", "ous"}, {"ization", "ize"}, {"ation", "ate"},
        {"ator", "ate"}, {"alism", "al"}, {"iveness", "ive"},
        {"fulness", "ful"}, {"ousness", "ous"}, {"aliti", "al"},
        {"iviti", "ive"}, {"biliti", "ble"}, {NULL, NULL}
    };
    for (int i = 0; rules[i].from; i++) {
        if (porter_ends_with(w, *k, rules[i].from)) {
            int base = *k + 1 - (int)strlen(rules[i].from);
            if (porter_measure(w, base - 1) > 0) {
                porter_set_end(w, k, rules[i].from, rules[i].to);
            }
            return;
        }
    }
}

/**
 * Porter stemmer step 3: -icate, -ative, -alize, etc.
 */
static void porter_step3(char *w, int *k) {
    if (*k < 1) return;
    struct { const char *from; const char *to; } rules[] = {
        {"icate", "ic"}, {"ative", ""}, {"alize", "al"},
        {"iciti", "ic"}, {"ical", "ic"}, {"ful", ""},
        {"ness", ""}, {NULL, NULL}
    };
    for (int i = 0; rules[i].from; i++) {
        if (porter_ends_with(w, *k, rules[i].from)) {
            int base = *k + 1 - (int)strlen(rules[i].from);
            if (porter_measure(w, base - 1) > 0) {
                porter_set_end(w, k, rules[i].from, rules[i].to);
            }
            return;
        }
    }
}

/**
 * Porter stemmer step 4: remove -ance, -ence, -er, -ic, etc.
 */
static void porter_step4(char *w, int *k) {
    if (*k < 1) return;
    const char *suffixes[] = {
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
        "ous", "ive", "ize", NULL
    };
    for (int i = 0; suffixes[i]; i++) {
        if (porter_ends_with(w, *k, suffixes[i])) {
            int base = *k + 1 - (int)strlen(suffixes[i]);
            if (strcmp(suffixes[i], "ion") == 0) {
                /* Special case: -ion requires s or t before it */
                if (base >= 1 && (w[base - 1] == 's' || w[base - 1] == 't')) {
                    if (porter_measure(w, base - 1) > 1) {
                        *k = base - 1;
                        w[*k + 1] = '\0';
                    }
                }
            } else {
                if (porter_measure(w, base - 1) > 1) {
                    *k = base - 1;
                    w[*k + 1] = '\0';
                }
            }
            return;
        }
    }
}

/**
 * Porter stemmer step 5a: remove trailing e.
 */
static void porter_step5a(char *w, int *k) {
    if (w[*k] == 'e') {
        int m = porter_measure(w, *k - 1);
        if (m > 1 || (m == 1 && !porter_cvc(w, *k - 1))) {
            (*k)--; w[*k + 1] = '\0';
        }
    }
}

/**
 * Porter stemmer step 5b: ll -> l.
 */
static void porter_step5b(char *w, int *k) {
    if (*k > 0 && w[*k] == 'l' && w[*k - 1] == 'l' && porter_measure(w, *k) > 1) {
        (*k)--; w[*k + 1] = '\0';
    }
}

/**
 * Full Porter stemmer for a single English word.
 */
static void porter_stem_english(char *w, int len) {
    if (len <= 2) return;
    int k = len - 1;
    porter_step1a(w, &k);
    porter_step1b(w, &k);
    porter_step1c(w, &k);
    porter_step2(w, &k);
    porter_step3(w, &k);
    porter_step4(w, &k);
    porter_step5a(w, &k);
    porter_step5b(w, &k);
}

/* ============================================================================
 * Simplified Suffix Stripping (Non-English)
 * ============================================================================ */

static void stem_strip_suffix(char *w, int *len, const char *suffix) {
    int slen = (int)strlen(suffix);
    if (*len > slen && memcmp(w + *len - slen, suffix, (size_t)slen) == 0) {
        *len -= slen;
        w[*len] = '\0';
    }
}

static void stem_german(char *w, int len) {
    if (len <= 3) return;
    /* Order matters: longer suffixes first */
    const char *suffixes[] = {
        "ungen", "keit", "heit", "lich", "isch", "ung", "ige", "ig",
        "en", "er", "em", "es", "st", "nd", NULL
    };
    for (int i = 0; suffixes[i]; i++) {
        int before = len;
        stem_strip_suffix(w, &len, suffixes[i]);
        if (len != before) return;
    }
}

static void stem_french(char *w, int len) {
    if (len <= 3) return;
    const char *suffixes[] = {
        "issement", "ement", "ation", "tion", "ment", "ence", "ance",
        "eux", "eur", "euse", "ier", "iere", "ique", "iste",
        "ite", "ite", "if", "ive", "es", "er", "ee", "ie", "e", NULL
    };
    for (int i = 0; suffixes[i]; i++) {
        int before = len;
        stem_strip_suffix(w, &len, suffixes[i]);
        if (len != before) return;
    }
}

static void stem_spanish(char *w, int len) {
    if (len <= 3) return;
    /* UTF-8 multi-byte suffixes are checked as byte sequences */
    const char *suffixes[] = {
        "amiento", "imiento", "mente", "mente", "cion", "ci\xc3\xb3n",
        "idad", "ando", "endo", "ador", "edor", "idor", "anza", "encia",
        "ible", "able", "ista", "oso", "osa", "ivo", "iva",
        "ar", "er", "ir", "es", "as", "os", NULL
    };
    for (int i = 0; suffixes[i]; i++) {
        int before = len;
        stem_strip_suffix(w, &len, suffixes[i]);
        if (len != before) return;
    }
}

static void stem_italian(char *w, int len) {
    if (len <= 3) return;
    const char *suffixes[] = {
        "amento", "imento", "mente", "zione", "atore", "atrice",
        "abile", "ibile", "ismo", "ista", "anza", "enza",
        "oso", "osa", "ivo", "iva", "ita",
        "are", "ere", "ire", "ato", "ito", "uto",
        "ai", "ei", "ci", NULL
    };
    for (int i = 0; suffixes[i]; i++) {
        int before = len;
        stem_strip_suffix(w, &len, suffixes[i]);
        if (len != before) return;
    }
}

static void stem_portuguese(char *w, int len) {
    if (len <= 3) return;
    const char *suffixes[] = {
        "amento", "imento", "mente", "c\xc3\xa3o", "cao", "idade",
        "ador", "edor", "idor", "avel", "ivel", "ismo", "ista",
        "oso", "osa", "ivo", "iva",
        "ar", "er", "ir", "as", "es", "os", NULL
    };
    for (int i = 0; suffixes[i]; i++) {
        int before = len;
        stem_strip_suffix(w, &len, suffixes[i]);
        if (len != before) return;
    }
}

/* ============================================================================
 * Public Stemming API
 * ============================================================================ */

int gv_ft_stem(const char *word, GV_FTLanguage lang, char *output, size_t output_size) {
    if (!word || !output || output_size == 0) return -1;

    size_t wlen = strlen(word);
    if (wlen == 0) {
        output[0] = '\0';
        return 0;
    }
    if (wlen >= output_size || wlen >= FT_MAX_WORD_LEN) return -1;

    /* Copy and lowercase */
    char buf[FT_MAX_WORD_LEN];
    for (size_t i = 0; i < wlen; i++) {
        buf[i] = (char)tolower((unsigned char)word[i]);
    }
    buf[wlen] = '\0';
    int len = (int)wlen;

    switch (lang) {
        case GV_LANG_ENGLISH:
        case GV_LANG_AUTO:
            porter_stem_english(buf, len);
            break;
        case GV_LANG_GERMAN:
            stem_german(buf, len);
            break;
        case GV_LANG_FRENCH:
            stem_french(buf, len);
            break;
        case GV_LANG_SPANISH:
            stem_spanish(buf, len);
            break;
        case GV_LANG_ITALIAN:
            stem_italian(buf, len);
            break;
        case GV_LANG_PORTUGUESE:
            stem_portuguese(buf, len);
            break;
        default:
            break;
    }

    size_t rlen = strlen(buf);
    if (rlen >= output_size) return -1;
    memcpy(output, buf, rlen + 1);
    return 0;
}

/* ============================================================================
 * Tokenizer
 * ============================================================================ */

/**
 * Internal token produced during tokenization.
 */
typedef struct {
    char text[FT_MAX_WORD_LEN];
    size_t position;                /* Ordinal position in document. */
} FT_Token;

typedef struct {
    FT_Token *tokens;
    size_t count;
    size_t capacity;
} FT_TokenList;

static void ft_token_list_init(FT_TokenList *tl) {
    memset(tl, 0, sizeof(*tl));
}

static void ft_token_list_free(FT_TokenList *tl) {
    free(tl->tokens);
    memset(tl, 0, sizeof(*tl));
}

static int ft_token_list_push(FT_TokenList *tl, const char *text, size_t pos) {
    if (tl->count >= tl->capacity) {
        size_t new_cap = tl->capacity == 0 ? 32 : tl->capacity * 2;
        FT_Token *buf = realloc(tl->tokens, new_cap * sizeof(FT_Token));
        if (!buf) return -1;
        tl->tokens = buf;
        tl->capacity = new_cap;
    }
    FT_Token *t = &tl->tokens[tl->count++];
    size_t len = strlen(text);
    if (len >= FT_MAX_WORD_LEN) len = FT_MAX_WORD_LEN - 1;
    memcpy(t->text, text, len);
    t->text[len] = '\0';
    t->position = pos;
    return 0;
}

/**
 * Tokenize text: split on non-alphanumeric, lowercase, remove stopwords, stem.
 */
static int ft_tokenize(const char *text, GV_FTLanguage lang,
                        int do_stem, FT_TokenList *out) {
    if (!text || !out) return -1;
    ft_token_list_init(out);

    size_t tlen = strlen(text);
    size_t i = 0;
    size_t position = 0;
    char word[FT_MAX_WORD_LEN];

    while (i < tlen) {
        /* Skip non-alphanumeric */
        while (i < tlen && !isalnum((unsigned char)text[i])) i++;
        if (i >= tlen) break;

        /* Extract word */
        size_t start = i;
        while (i < tlen && isalnum((unsigned char)text[i])) i++;
        size_t wlen = i - start;
        if (wlen == 0) continue;
        if (wlen >= FT_MAX_WORD_LEN) wlen = FT_MAX_WORD_LEN - 1;

        /* Lowercase */
        for (size_t j = 0; j < wlen; j++) {
            word[j] = (char)tolower((unsigned char)text[start + j]);
        }
        word[wlen] = '\0';

        /* Skip stopwords */
        if (ft_is_stopword(word, lang)) {
            position++;
            continue;
        }

        /* Stem */
        if (do_stem) {
            char stemmed[FT_MAX_WORD_LEN];
            if (gv_ft_stem(word, lang, stemmed, sizeof(stemmed)) == 0) {
                if (ft_token_list_push(out, stemmed, position) != 0) {
                    ft_token_list_free(out);
                    return -1;
                }
            } else {
                if (ft_token_list_push(out, word, position) != 0) {
                    ft_token_list_free(out);
                    return -1;
                }
            }
        } else {
            if (ft_token_list_push(out, word, position) != 0) {
                ft_token_list_free(out);
                return -1;
            }
        }
        position++;
    }
    return 0;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_FTConfig DEFAULT_FT_CONFIG = {
    .language           = GV_LANG_ENGLISH,
    .enable_stemming    = 1,
    .enable_phrase_match = 1,
    .use_blockmax_wand  = 1,
    .block_size         = 128
};

void gv_ft_config_init(GV_FTConfig *config) {
    if (!config) return;
    *config = DEFAULT_FT_CONFIG;
}

/* ============================================================================
 * Posting List Helpers
 * ============================================================================ */

static FT_PostingList *ft_find_posting_list(const GV_FTIndex *idx, const char *term) {
    size_t bucket = ft_hash_string(term) % FT_TERM_HASH_BUCKETS;
    FT_PostingList *pl = idx->term_buckets[bucket];
    while (pl) {
        if (strcmp(pl->term, term) == 0) return pl;
        pl = pl->next;
    }
    return NULL;
}

static FT_PostingList *ft_get_or_create_posting_list(GV_FTIndex *idx, const char *term) {
    size_t bucket = ft_hash_string(term) % FT_TERM_HASH_BUCKETS;
    FT_PostingList *pl = idx->term_buckets[bucket];
    while (pl) {
        if (strcmp(pl->term, term) == 0) return pl;
        pl = pl->next;
    }

    pl = calloc(1, sizeof(FT_PostingList));
    if (!pl) return NULL;
    pl->term = strdup(term);
    if (!pl->term) { free(pl); return NULL; }

    pl->postings = malloc(FT_INITIAL_POSTING_CAP * sizeof(FT_Posting));
    if (!pl->postings) { free(pl->term); free(pl); return NULL; }
    pl->capacity = FT_INITIAL_POSTING_CAP;

    pl->next = idx->term_buckets[bucket];
    idx->term_buckets[bucket] = pl;
    idx->total_terms++;
    return pl;
}

/* ============================================================================
 * Document Info Helpers
 * ============================================================================ */

static FT_DocInfo *ft_find_doc_info(const GV_FTIndex *idx, size_t doc_id) {
    size_t bucket = ft_hash_size(doc_id) % FT_DOC_HASH_BUCKETS;
    FT_DocInfo *di = idx->doc_buckets[bucket];
    while (di) {
        if (di->doc_id == doc_id) return di;
        di = di->next;
    }
    return NULL;
}

static FT_DocInfo *ft_get_or_create_doc_info(GV_FTIndex *idx, size_t doc_id) {
    size_t bucket = ft_hash_size(doc_id) % FT_DOC_HASH_BUCKETS;
    FT_DocInfo *di = idx->doc_buckets[bucket];
    while (di) {
        if (di->doc_id == doc_id) return di;
        di = di->next;
    }

    di = calloc(1, sizeof(FT_DocInfo));
    if (!di) return NULL;
    di->doc_id = doc_id;
    di->next = idx->doc_buckets[bucket];
    idx->doc_buckets[bucket] = di;
    idx->total_documents++;
    return di;
}

/* ============================================================================
 * Posting Insertion with Positions
 * ============================================================================ */

static int ft_add_posting(FT_PostingList *pl, size_t doc_id, size_t position,
                           int store_positions) {
    /* Find existing posting for this doc */
    for (size_t i = 0; i < pl->count; i++) {
        if (pl->postings[i].doc_id == doc_id) {
            pl->postings[i].term_freq++;
            if (store_positions) {
                FT_PositionList *pos = &pl->postings[i].pos;
                if (pos->count >= pos->capacity) {
                    size_t new_cap = pos->capacity == 0 ? FT_INITIAL_POS_CAP : pos->capacity * 2;
                    size_t *buf = realloc(pos->positions, new_cap * sizeof(size_t));
                    if (!buf) return -1;
                    pos->positions = buf;
                    pos->capacity = new_cap;
                }
                pos->positions[pos->count++] = position;
            }
            return 0;
        }
    }

    /* New posting */
    if (pl->count >= pl->capacity) {
        size_t new_cap = pl->capacity * 2;
        FT_Posting *buf = realloc(pl->postings, new_cap * sizeof(FT_Posting));
        if (!buf) return -1;
        pl->postings = buf;
        pl->capacity = new_cap;
    }

    FT_Posting *p = &pl->postings[pl->count];
    memset(p, 0, sizeof(*p));
    p->doc_id = doc_id;
    p->term_freq = 1;

    if (store_positions) {
        p->pos.positions = malloc(FT_INITIAL_POS_CAP * sizeof(size_t));
        if (!p->pos.positions) return -1;
        p->pos.capacity = FT_INITIAL_POS_CAP;
        p->pos.positions[0] = position;
        p->pos.count = 1;
    }

    pl->count++;
    return 0;
}

static void ft_remove_doc_from_posting_list(FT_PostingList *pl, size_t doc_id) {
    for (size_t i = 0; i < pl->count; i++) {
        if (pl->postings[i].doc_id == doc_id) {
            free(pl->postings[i].pos.positions);
            for (size_t j = i; j < pl->count - 1; j++) {
                pl->postings[j] = pl->postings[j + 1];
            }
            pl->count--;
            return;
        }
    }
}

/* ============================================================================
 * BlockMax WAND Precomputation
 * ============================================================================ */

static void ft_invalidate_block_maxes(FT_PostingList *pl) {
    free(pl->bmax.block_maxes);
    pl->bmax.block_maxes = NULL;
    pl->bmax.block_count = 0;
}

/**
 * Precompute per-block maximum BM25 contribution for a posting list.
 */
static void ft_build_block_maxes(FT_PostingList *pl, size_t block_size,
                                  size_t total_docs, size_t total_doc_length,
                                  const GV_FTIndex *idx) {
    if (!pl || pl->count == 0 || block_size == 0) return;

    /* Free old */
    free(pl->bmax.block_maxes);
    size_t nblocks = (pl->count + block_size - 1) / block_size;
    pl->bmax.block_maxes = malloc(nblocks * sizeof(float));
    if (!pl->bmax.block_maxes) { pl->bmax.block_count = 0; return; }
    pl->bmax.block_count = nblocks;

    double avgdl = total_docs > 0 ? (double)total_doc_length / (double)total_docs : 1.0;
    double idf_val = log(((double)total_docs - (double)pl->count + 0.5) /
                          ((double)pl->count + 0.5) + 1.0);

    for (size_t b = 0; b < nblocks; b++) {
        float bmax = 0.0f;
        size_t start = b * block_size;
        size_t end = start + block_size;
        if (end > pl->count) end = pl->count;

        for (size_t i = start; i < end; i++) {
            size_t tf = pl->postings[i].term_freq;
            size_t doc_id = pl->postings[i].doc_id;
            FT_DocInfo *di = ft_find_doc_info(idx, doc_id);
            double dl = di ? (double)di->doc_length : avgdl;
            double tf_d = (double)tf;
            double tf_comp = (tf_d * (FT_BM25_K1 + 1.0)) /
                             (tf_d + FT_BM25_K1 * (1.0 - FT_BM25_B + FT_BM25_B * (dl / avgdl)));
            float score = (float)(idf_val * tf_comp);
            if (score > bmax) bmax = score;
        }
        pl->bmax.block_maxes[b] = bmax;
    }
}

/* ============================================================================
 * BM25 Scoring
 * ============================================================================ */

static float ft_compute_idf(size_t total_docs, size_t doc_freq) {
    double N = (double)total_docs;
    double df = (double)doc_freq;
    return (float)log((N - df + 0.5) / (df + 0.5) + 1.0);
}

static float ft_compute_bm25_term(size_t term_freq, size_t doc_length,
                                    size_t total_docs, size_t total_doc_length,
                                    size_t doc_freq) {
    double avgdl = total_docs > 0 ? (double)total_doc_length / (double)total_docs : 1.0;
    double tf = (double)term_freq;
    double dl = (double)doc_length;
    double idf = (double)ft_compute_idf(total_docs, doc_freq);
    double tf_comp = (tf * (FT_BM25_K1 + 1.0)) /
                     (tf + FT_BM25_K1 * (1.0 - FT_BM25_B + FT_BM25_B * (dl / avgdl)));
    return (float)(idf * tf_comp);
}

/* ============================================================================
 * Index Lifecycle
 * ============================================================================ */

GV_FTIndex *gv_ft_create(const GV_FTConfig *config) {
    GV_FTIndex *idx = calloc(1, sizeof(GV_FTIndex));
    if (!idx) return NULL;

    idx->config = config ? *config : DEFAULT_FT_CONFIG;

    if (pthread_rwlock_init(&idx->rwlock, NULL) != 0) {
        free(idx);
        return NULL;
    }

    return idx;
}

void gv_ft_destroy(GV_FTIndex *idx) {
    if (!idx) return;

    /* Free posting lists */
    for (size_t i = 0; i < FT_TERM_HASH_BUCKETS; i++) {
        FT_PostingList *pl = idx->term_buckets[i];
        while (pl) {
            FT_PostingList *next = pl->next;
            for (size_t j = 0; j < pl->count; j++) {
                free(pl->postings[j].pos.positions);
            }
            free(pl->postings);
            free(pl->bmax.block_maxes);
            free(pl->term);
            free(pl);
            pl = next;
        }
    }

    /* Free document info */
    for (size_t i = 0; i < FT_DOC_HASH_BUCKETS; i++) {
        FT_DocInfo *di = idx->doc_buckets[i];
        while (di) {
            FT_DocInfo *next = di->next;
            free(di);
            di = next;
        }
    }

    pthread_rwlock_destroy(&idx->rwlock);
    free(idx);
}

/* ============================================================================
 * Indexing Operations
 * ============================================================================ */

int gv_ft_add_document(GV_FTIndex *idx, size_t doc_id, const char *text) {
    if (!idx || !text) return -1;

    FT_TokenList tokens;
    if (ft_tokenize(text, idx->config.language, idx->config.enable_stemming, &tokens) != 0) {
        return -1;
    }

    pthread_rwlock_wrlock(&idx->rwlock);

    FT_DocInfo *di = ft_get_or_create_doc_info(idx, doc_id);
    if (!di) {
        pthread_rwlock_unlock(&idx->rwlock);
        ft_token_list_free(&tokens);
        return -1;
    }

    /* Update document length */
    idx->total_doc_length -= di->doc_length;
    di->doc_length = tokens.count;
    idx->total_doc_length += di->doc_length;

    /* Insert each token into the inverted index */
    int store_pos = idx->config.enable_phrase_match;
    for (size_t i = 0; i < tokens.count; i++) {
        FT_PostingList *pl = ft_get_or_create_posting_list(idx, tokens.tokens[i].text);
        if (pl) {
            ft_add_posting(pl, doc_id, tokens.tokens[i].position, store_pos);
            ft_invalidate_block_maxes(pl);
        }
    }

    pthread_rwlock_unlock(&idx->rwlock);
    ft_token_list_free(&tokens);
    return 0;
}

int gv_ft_remove_document(GV_FTIndex *idx, size_t doc_id) {
    if (!idx) return -1;

    pthread_rwlock_wrlock(&idx->rwlock);

    /* Remove document info */
    size_t bucket = ft_hash_size(doc_id) % FT_DOC_HASH_BUCKETS;
    FT_DocInfo **pp = &idx->doc_buckets[bucket];
    FT_DocInfo *di = NULL;
    while (*pp) {
        if ((*pp)->doc_id == doc_id) {
            di = *pp;
            *pp = (*pp)->next;
            break;
        }
        pp = &(*pp)->next;
    }

    if (!di) {
        pthread_rwlock_unlock(&idx->rwlock);
        return -1;
    }

    idx->total_doc_length -= di->doc_length;
    idx->total_documents--;
    free(di);

    /* Remove from all posting lists */
    for (size_t i = 0; i < FT_TERM_HASH_BUCKETS; i++) {
        FT_PostingList *pl = idx->term_buckets[i];
        while (pl) {
            ft_remove_doc_from_posting_list(pl, doc_id);
            ft_invalidate_block_maxes(pl);
            pl = pl->next;
        }
    }

    pthread_rwlock_unlock(&idx->rwlock);
    return 0;
}

/* ============================================================================
 * Priority Queue for BlockMax WAND
 * ============================================================================ */

/**
 * @brief Min-heap entry for top-k result collection.
 */
typedef struct {
    size_t doc_id;
    float score;
} FT_HeapEntry;

typedef struct {
    FT_HeapEntry *entries;
    size_t count;
    size_t capacity;
} FT_MinHeap;

static void ft_heap_init(FT_MinHeap *h, size_t capacity) {
    h->entries = malloc(capacity * sizeof(FT_HeapEntry));
    h->count = 0;
    h->capacity = capacity;
}

static void ft_heap_free(FT_MinHeap *h) {
    free(h->entries);
    h->count = 0;
    h->capacity = 0;
}

static void ft_heap_sift_up(FT_MinHeap *h, size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (h->entries[i].score < h->entries[parent].score) {
            FT_HeapEntry tmp = h->entries[i];
            h->entries[i] = h->entries[parent];
            h->entries[parent] = tmp;
            i = parent;
        } else {
            break;
        }
    }
}

static void ft_heap_sift_down(FT_MinHeap *h, size_t i) {
    while (1) {
        size_t smallest = i;
        size_t left = 2 * i + 1;
        size_t right = 2 * i + 2;
        if (left < h->count && h->entries[left].score < h->entries[smallest].score)
            smallest = left;
        if (right < h->count && h->entries[right].score < h->entries[smallest].score)
            smallest = right;
        if (smallest == i) break;
        FT_HeapEntry tmp = h->entries[i];
        h->entries[i] = h->entries[smallest];
        h->entries[smallest] = tmp;
        i = smallest;
    }
}

static void ft_heap_push(FT_MinHeap *h, size_t doc_id, float score) {
    if (h->count < h->capacity) {
        h->entries[h->count].doc_id = doc_id;
        h->entries[h->count].score = score;
        ft_heap_sift_up(h, h->count);
        h->count++;
    } else if (score > h->entries[0].score) {
        h->entries[0].doc_id = doc_id;
        h->entries[0].score = score;
        ft_heap_sift_down(h, 0);
    }
}

static float ft_heap_threshold(const FT_MinHeap *h) {
    if (h->count < h->capacity) return 0.0f;
    return h->entries[0].score;
}

/* ============================================================================
 * BlockMax WAND Search
 * ============================================================================ */

/**
 * @brief Cursor over a posting list for WAND evaluation.
 */
typedef struct {
    FT_PostingList *pl;
    size_t cursor;                  /* Current position in postings array. */
    float idf;                      /* Precomputed IDF for this term. */
} FT_TermCursor;

/**
 * Advance cursor to the first posting with doc_id >= target.
 */
static void ft_cursor_advance_to(FT_TermCursor *tc, size_t target) {
    while (tc->cursor < tc->pl->count && tc->pl->postings[tc->cursor].doc_id < target) {
        tc->cursor++;
    }
}

static int ft_cursor_exhausted(const FT_TermCursor *tc) {
    return tc->cursor >= tc->pl->count;
}

static size_t ft_cursor_doc_id(const FT_TermCursor *tc) {
    return tc->pl->postings[tc->cursor].doc_id;
}

/**
 * Get the BlockMax upper bound for the block containing the current cursor.
 */
static float ft_cursor_block_max(const FT_TermCursor *tc, size_t block_size) {
    if (!tc->pl->bmax.block_maxes || tc->pl->bmax.block_count == 0) {
        /* Fallback: use IDF * (k1+1) as a safe upper bound */
        return tc->idf * (float)(FT_BM25_K1 + 1.0);
    }
    size_t block_idx = tc->cursor / block_size;
    if (block_idx >= tc->pl->bmax.block_count) {
        block_idx = tc->pl->bmax.block_count - 1;
    }
    return tc->pl->bmax.block_maxes[block_idx];
}

/**
 * Skip cursor past the current block if the block max cannot beat threshold.
 */
static void ft_cursor_skip_block(FT_TermCursor *tc, size_t block_size) {
    size_t block_idx = tc->cursor / block_size;
    size_t next_block_start = (block_idx + 1) * block_size;
    if (next_block_start < tc->pl->count) {
        tc->cursor = next_block_start;
    } else {
        tc->cursor = tc->pl->count;
    }
}

/**
 * Compare cursors by current doc_id (for qsort).
 */
static int ft_cursor_cmp(const void *a, const void *b) {
    const FT_TermCursor *ca = (const FT_TermCursor *)a;
    const FT_TermCursor *cb = (const FT_TermCursor *)b;
    if (ft_cursor_exhausted(ca) && ft_cursor_exhausted(cb)) return 0;
    if (ft_cursor_exhausted(ca)) return 1;
    if (ft_cursor_exhausted(cb)) return -1;
    size_t da = ft_cursor_doc_id(ca);
    size_t db = ft_cursor_doc_id(cb);
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/**
 * Perform BlockMax WAND search across all query term cursors.
 */
static int ft_search_blockmax_wand(const GV_FTIndex *idx, FT_TermCursor *cursors,
                                    size_t ncursors, size_t limit,
                                    FT_MinHeap *heap) {
    size_t block_size = idx->config.block_size;

    /* Ensure block maxes are computed for each posting list */
    for (size_t i = 0; i < ncursors; i++) {
        if (cursors[i].pl->bmax.block_maxes == NULL) {
            ft_build_block_maxes(cursors[i].pl, block_size,
                                  idx->total_documents, idx->total_doc_length, idx);
        }
    }

    while (1) {
        /* Sort cursors by current doc_id (non-exhausted first) */
        qsort(cursors, ncursors, sizeof(FT_TermCursor), ft_cursor_cmp);

        /* Find the first non-exhausted cursor */
        size_t first_active = 0;
        while (first_active < ncursors && ft_cursor_exhausted(&cursors[first_active])) {
            first_active++;
        }
        if (first_active >= ncursors) break;

        /* Compute upper-bound sum of block maxes for all active cursors */
        float threshold = ft_heap_threshold(heap);
        float upper_bound = 0.0f;
        for (size_t i = first_active; i < ncursors; i++) {
            if (!ft_cursor_exhausted(&cursors[i])) {
                upper_bound += ft_cursor_block_max(&cursors[i], block_size);
            }
        }

        /* If the combined upper bound cannot beat the threshold, try to skip */
        if (upper_bound <= threshold && heap->count >= limit) {
            /* Advance all cursors: skip blocks that cannot contribute */
            int any_advanced = 0;
            for (size_t i = first_active; i < ncursors; i++) {
                if (!ft_cursor_exhausted(&cursors[i])) {
                    size_t old_cursor = cursors[i].cursor;
                    ft_cursor_skip_block(&cursors[i], block_size);
                    if (cursors[i].cursor != old_cursor) any_advanced = 1;
                }
            }
            if (!any_advanced) break;
            continue;
        }

        /* Pivot: find the smallest doc_id that all cursors can reach */
        size_t pivot_doc = ft_cursor_doc_id(&cursors[first_active]);

        /* Accumulate WAND-style: find prefix whose upper bound exceeds threshold */
        float prefix_ub = 0.0f;
        size_t pivot_idx = first_active;
        for (size_t i = first_active; i < ncursors; i++) {
            if (ft_cursor_exhausted(&cursors[i])) break;
            prefix_ub += ft_cursor_block_max(&cursors[i], block_size);
            pivot_idx = i;
            if (prefix_ub > threshold) break;
        }

        if (prefix_ub <= threshold && heap->count >= limit) {
            /* Cannot beat threshold; advance first cursor */
            if (!ft_cursor_exhausted(&cursors[first_active])) {
                cursors[first_active].cursor++;
            }
            continue;
        }

        /* Get pivot document */
        if (!ft_cursor_exhausted(&cursors[pivot_idx])) {
            pivot_doc = ft_cursor_doc_id(&cursors[pivot_idx]);
        }

        /* Advance all cursors to pivot_doc */
        for (size_t i = first_active; i < ncursors; i++) {
            if (!ft_cursor_exhausted(&cursors[i])) {
                ft_cursor_advance_to(&cursors[i], pivot_doc);
            }
        }

        /* Score the pivot document: sum BM25 contributions from all matching cursors */
        float doc_score = 0.0f;
        int any_match = 0;
        FT_DocInfo *di = ft_find_doc_info(idx, pivot_doc);

        for (size_t i = first_active; i < ncursors; i++) {
            if (ft_cursor_exhausted(&cursors[i])) continue;
            if (ft_cursor_doc_id(&cursors[i]) == pivot_doc) {
                FT_Posting *p = &cursors[i].pl->postings[cursors[i].cursor];
                size_t dl = di ? di->doc_length : 1;
                doc_score += ft_compute_bm25_term(p->term_freq, dl,
                                                    idx->total_documents,
                                                    idx->total_doc_length,
                                                    cursors[i].pl->count);
                any_match = 1;
            }
        }

        if (any_match) {
            ft_heap_push(heap, pivot_doc, doc_score);
        }

        /* Advance cursors that were at pivot_doc */
        for (size_t i = first_active; i < ncursors; i++) {
            if (!ft_cursor_exhausted(&cursors[i]) &&
                ft_cursor_doc_id(&cursors[i]) == pivot_doc) {
                cursors[i].cursor++;
            }
        }
    }

    return 0;
}

/* ============================================================================
 * Naive Search (fallback when BlockMax WAND disabled)
 * ============================================================================ */

typedef struct {
    size_t doc_id;
    float score;
} FT_DocScore;

static int ft_docscore_cmp_desc(const void *a, const void *b) {
    const FT_DocScore *da = (const FT_DocScore *)a;
    const FT_DocScore *db = (const FT_DocScore *)b;
    if (db->score > da->score) return 1;
    if (db->score < da->score) return -1;
    return 0;
}

static int ft_search_naive(const GV_FTIndex *idx, const FT_TokenList *query_tokens,
                            size_t limit, FT_MinHeap *heap) {
    /* Collect unique query terms */
    const char *unique_terms[256];
    size_t unique_count = 0;
    for (size_t i = 0; i < query_tokens->count && unique_count < 256; i++) {
        int found = 0;
        for (size_t j = 0; j < unique_count; j++) {
            if (strcmp(unique_terms[j], query_tokens->tokens[i].text) == 0) {
                found = 1; break;
            }
        }
        if (!found) unique_terms[unique_count++] = query_tokens->tokens[i].text;
    }

    /* Accumulate scores */
    size_t score_cap = 256;
    FT_DocScore *scores = calloc(score_cap, sizeof(FT_DocScore));
    if (!scores) return -1;
    size_t score_count = 0;

    for (size_t t = 0; t < unique_count; t++) {
        FT_PostingList *pl = ft_find_posting_list(idx, unique_terms[t]);
        if (!pl) continue;

        for (size_t p = 0; p < pl->count; p++) {
            size_t doc_id = pl->postings[p].doc_id;
            size_t tf = pl->postings[p].term_freq;
            FT_DocInfo *di = ft_find_doc_info(idx, doc_id);
            if (!di) continue;

            float s = ft_compute_bm25_term(tf, di->doc_length,
                                             idx->total_documents,
                                             idx->total_doc_length,
                                             pl->count);

            /* Find or insert doc in scores array */
            int found = 0;
            for (size_t k = 0; k < score_count; k++) {
                if (scores[k].doc_id == doc_id) {
                    scores[k].score += s;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                if (score_count >= score_cap) {
                    score_cap *= 2;
                    FT_DocScore *tmp = realloc(scores, score_cap * sizeof(FT_DocScore));
                    if (!tmp) { free(scores); return -1; }
                    scores = tmp;
                }
                scores[score_count].doc_id = doc_id;
                scores[score_count].score = s;
                score_count++;
            }
        }
    }

    /* Push to heap */
    for (size_t i = 0; i < score_count; i++) {
        if (scores[i].score > 0.0f) {
            ft_heap_push(heap, scores[i].doc_id, scores[i].score);
        }
    }

    free(scores);
    return 0;
}

/* ============================================================================
 * Search Operations
 * ============================================================================ */

int gv_ft_search(const GV_FTIndex *idx, const char *query, size_t limit,
                 GV_FTResult *results) {
    if (!idx || !query || !results || limit == 0) return -1;

    FT_TokenList tokens;
    if (ft_tokenize(query, idx->config.language, idx->config.enable_stemming, &tokens) != 0) {
        return -1;
    }
    if (tokens.count == 0) {
        ft_token_list_free(&tokens);
        return 0;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    if (idx->total_documents == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        ft_token_list_free(&tokens);
        return 0;
    }

    FT_MinHeap heap;
    ft_heap_init(&heap, limit);
    if (!heap.entries) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        ft_token_list_free(&tokens);
        return -1;
    }

    if (idx->config.use_blockmax_wand) {
        /* Collect unique query terms and build cursors */
        const char *unique_terms[256];
        size_t unique_count = 0;
        for (size_t i = 0; i < tokens.count && unique_count < 256; i++) {
            int found = 0;
            for (size_t j = 0; j < unique_count; j++) {
                if (strcmp(unique_terms[j], tokens.tokens[i].text) == 0) {
                    found = 1; break;
                }
            }
            if (!found) unique_terms[unique_count++] = tokens.tokens[i].text;
        }

        /* Build cursors for terms that exist in index */
        FT_TermCursor *cursors = calloc(unique_count, sizeof(FT_TermCursor));
        size_t ncursors = 0;
        if (cursors) {
            for (size_t i = 0; i < unique_count; i++) {
                FT_PostingList *pl = ft_find_posting_list(idx, unique_terms[i]);
                if (pl && pl->count > 0) {
                    cursors[ncursors].pl = pl;
                    cursors[ncursors].cursor = 0;
                    cursors[ncursors].idf = ft_compute_idf(idx->total_documents, pl->count);
                    ncursors++;
                }
            }

            if (ncursors > 0) {
                ft_search_blockmax_wand(idx, cursors, ncursors, limit, &heap);
            }
            free(cursors);
        }
    } else {
        ft_search_naive(idx, &tokens, limit, &heap);
    }

    /* Extract results from heap (sorted by score descending) */
    int count = (int)heap.count;

    /* Heap is a min-heap; extract into results in descending order */
    FT_HeapEntry *sorted = malloc(heap.count * sizeof(FT_HeapEntry));
    if (sorted) {
        memcpy(sorted, heap.entries, heap.count * sizeof(FT_HeapEntry));
        /* Sort descending */
        for (size_t i = 0; i < (size_t)count; i++) {
            for (size_t j = i + 1; j < (size_t)count; j++) {
                if (sorted[j].score > sorted[i].score) {
                    FT_HeapEntry tmp = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = tmp;
                }
            }
        }
        for (int i = 0; i < count; i++) {
            results[i].doc_id = sorted[i].doc_id;
            results[i].score = sorted[i].score;
            results[i].match_positions = NULL;
            results[i].match_count = 0;
        }
        free(sorted);
    } else {
        /* Fallback: unsorted */
        for (int i = 0; i < count; i++) {
            results[i].doc_id = heap.entries[i].doc_id;
            results[i].score = heap.entries[i].score;
            results[i].match_positions = NULL;
            results[i].match_count = 0;
        }
    }

    ft_heap_free(&heap);
    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    ft_token_list_free(&tokens);
    return count;
}

/* ============================================================================
 * Phrase Search
 * ============================================================================ */

/**
 * Check if a document contains all phrase terms at consecutive positions.
 * Returns 1 if phrase match found, 0 otherwise.
 * If match_positions_out is non-NULL, stores the starting positions.
 */
static int ft_check_phrase(const GV_FTIndex *idx, size_t doc_id,
                            const char **terms, size_t nterms,
                            size_t **match_positions_out, size_t *match_count_out) {
    if (nterms == 0) return 0;

    /* Find posting for each term in this document */
    FT_Posting *postings[256];
    if (nterms > 256) return 0;

    for (size_t i = 0; i < nterms; i++) {
        FT_PostingList *pl = ft_find_posting_list(idx, terms[i]);
        if (!pl) return 0;
        postings[i] = NULL;
        for (size_t j = 0; j < pl->count; j++) {
            if (pl->postings[j].doc_id == doc_id) {
                postings[i] = &pl->postings[j];
                break;
            }
        }
        if (!postings[i] || postings[i]->pos.count == 0) return 0;
    }

    /* For each starting position of the first term, check consecutive positions */
    size_t *match_starts = NULL;
    size_t match_cap = 0;
    size_t match_cnt = 0;

    for (size_t p = 0; p < postings[0]->pos.count; p++) {
        size_t start_pos = postings[0]->pos.positions[p];
        int matched = 1;

        for (size_t t = 1; t < nterms; t++) {
            size_t target = start_pos + t;
            int found = 0;
            for (size_t q = 0; q < postings[t]->pos.count; q++) {
                if (postings[t]->pos.positions[q] == target) {
                    found = 1;
                    break;
                }
                if (postings[t]->pos.positions[q] > target) break;
            }
            if (!found) { matched = 0; break; }
        }

        if (matched) {
            if (match_positions_out) {
                if (match_cnt >= match_cap) {
                    size_t new_cap = match_cap == 0 ? 8 : match_cap * 2;
                    size_t *tmp = realloc(match_starts, new_cap * sizeof(size_t));
                    if (!tmp) { free(match_starts); return 0; }
                    match_starts = tmp;
                    match_cap = new_cap;
                }
                match_starts[match_cnt] = start_pos;
            }
            match_cnt++;
        }
    }

    if (match_cnt > 0) {
        if (match_positions_out) {
            *match_positions_out = match_starts;
            *match_count_out = match_cnt;
        }
        return 1;
    }

    free(match_starts);
    return 0;
}

int gv_ft_search_phrase(const GV_FTIndex *idx, const char *phrase, size_t limit,
                        GV_FTResult *results) {
    if (!idx || !phrase || !results || limit == 0) return -1;
    if (!idx->config.enable_phrase_match) return -1;

    FT_TokenList tokens;
    if (ft_tokenize(phrase, idx->config.language, idx->config.enable_stemming, &tokens) != 0) {
        return -1;
    }
    if (tokens.count == 0) {
        ft_token_list_free(&tokens);
        return 0;
    }

    /* Collect ordered unique terms (preserving phrase order) */
    const char *terms[256];
    size_t nterms = 0;
    for (size_t i = 0; i < tokens.count && nterms < 256; i++) {
        terms[nterms++] = tokens.tokens[i].text;
    }

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    if (idx->total_documents == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        ft_token_list_free(&tokens);
        return 0;
    }

    /* Find candidate documents: those containing the rarest phrase term */
    FT_PostingList *rarest = NULL;
    size_t rarest_count = (size_t)-1;
    for (size_t i = 0; i < nterms; i++) {
        FT_PostingList *pl = ft_find_posting_list(idx, terms[i]);
        if (!pl || pl->count == 0) {
            /* Term not in index; no results possible */
            pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
            ft_token_list_free(&tokens);
            return 0;
        }
        if (pl->count < rarest_count) {
            rarest = pl;
            rarest_count = pl->count;
        }
    }

    /* Check each candidate for phrase match */
    size_t result_count = 0;
    FT_DocScore *candidates = malloc(rarest_count * sizeof(FT_DocScore));
    if (!candidates) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
        ft_token_list_free(&tokens);
        return -1;
    }

    size_t cand_count = 0;
    for (size_t i = 0; i < rarest->count; i++) {
        size_t doc_id = rarest->postings[i].doc_id;
        size_t *match_pos = NULL;
        size_t match_cnt = 0;

        if (ft_check_phrase(idx, doc_id, terms, nterms, &match_pos, &match_cnt)) {
            /* Compute BM25 score for the phrase terms */
            float score = 0.0f;
            FT_DocInfo *di = ft_find_doc_info(idx, doc_id);
            for (size_t t = 0; t < nterms; t++) {
                FT_PostingList *pl = ft_find_posting_list(idx, terms[t]);
                if (!pl || !di) continue;
                for (size_t j = 0; j < pl->count; j++) {
                    if (pl->postings[j].doc_id == doc_id) {
                        score += ft_compute_bm25_term(pl->postings[j].term_freq,
                                                       di->doc_length,
                                                       idx->total_documents,
                                                       idx->total_doc_length,
                                                       pl->count);
                        break;
                    }
                }
            }

            if (cand_count < rarest_count) {
                candidates[cand_count].doc_id = doc_id;
                candidates[cand_count].score = score;
                cand_count++;
            }

            /* Store result if within limit (we will sort afterwards) */
            if (result_count < limit) {
                results[result_count].doc_id = doc_id;
                results[result_count].score = score;
                results[result_count].match_positions = match_pos;
                results[result_count].match_count = match_cnt;
                result_count++;
            } else {
                free(match_pos);
            }
        }
    }

    /* Sort results by score descending */
    if (result_count > 1) {
        /* Simple insertion sort for small result sets */
        for (size_t i = 1; i < result_count; i++) {
            GV_FTResult key = results[i];
            size_t j = i;
            while (j > 0 && results[j - 1].score < key.score) {
                results[j] = results[j - 1];
                j--;
            }
            results[j] = key;
        }
    }

    free(candidates);
    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    ft_token_list_free(&tokens);
    return (int)result_count;
}

/* ============================================================================
 * Result Cleanup
 * ============================================================================ */

void gv_ft_free_results(GV_FTResult *results, size_t count) {
    if (!results) return;
    for (size_t i = 0; i < count; i++) {
        free(results[i].match_positions);
        results[i].match_positions = NULL;
        results[i].match_count = 0;
    }
}

/* ============================================================================
 * Index Information
 * ============================================================================ */

size_t gv_ft_doc_count(const GV_FTIndex *idx) {
    if (!idx) return 0;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);
    size_t count = idx->total_documents;
    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    return count;
}

/* ============================================================================
 * Persistence
 * ============================================================================ */

int gv_ft_save(const GV_FTIndex *idx, const char *path) {
    if (!idx || !path) return -1;

    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&idx->rwlock);

    /* Magic header */
    const char magic[] = "GV_FT01";
    fwrite(magic, 1, 7, fp);

    /* Config */
    uint32_t lang = (uint32_t)idx->config.language;
    uint32_t flags = 0;
    if (idx->config.enable_stemming)    flags |= 0x01;
    if (idx->config.enable_phrase_match) flags |= 0x02;
    if (idx->config.use_blockmax_wand)  flags |= 0x04;
    fwrite(&lang, sizeof(lang), 1, fp);
    fwrite(&flags, sizeof(flags), 1, fp);
    uint64_t bs = (uint64_t)idx->config.block_size;
    fwrite(&bs, sizeof(bs), 1, fp);

    /* Stats */
    uint64_t td = (uint64_t)idx->total_documents;
    uint64_t tt = (uint64_t)idx->total_terms;
    uint64_t tl = (uint64_t)idx->total_doc_length;
    fwrite(&td, sizeof(td), 1, fp);
    fwrite(&tt, sizeof(tt), 1, fp);
    fwrite(&tl, sizeof(tl), 1, fp);

    /* Document info */
    for (size_t i = 0; i < FT_DOC_HASH_BUCKETS; i++) {
        FT_DocInfo *di = idx->doc_buckets[i];
        while (di) {
            uint64_t did = (uint64_t)di->doc_id;
            uint64_t dl = (uint64_t)di->doc_length;
            fwrite(&did, sizeof(did), 1, fp);
            fwrite(&dl, sizeof(dl), 1, fp);
            di = di->next;
        }
    }
    /* Sentinel */
    uint64_t sentinel = UINT64_MAX;
    fwrite(&sentinel, sizeof(sentinel), 1, fp);

    /* Posting lists */
    for (size_t i = 0; i < FT_TERM_HASH_BUCKETS; i++) {
        FT_PostingList *pl = idx->term_buckets[i];
        while (pl) {
            uint32_t term_len = (uint32_t)strlen(pl->term);
            fwrite(&term_len, sizeof(term_len), 1, fp);
            fwrite(pl->term, 1, term_len, fp);

            uint64_t pcount = (uint64_t)pl->count;
            fwrite(&pcount, sizeof(pcount), 1, fp);

            for (size_t j = 0; j < pl->count; j++) {
                FT_Posting *p = &pl->postings[j];
                uint64_t did = (uint64_t)p->doc_id;
                uint64_t tf = (uint64_t)p->term_freq;
                fwrite(&did, sizeof(did), 1, fp);
                fwrite(&tf, sizeof(tf), 1, fp);

                /* Positions */
                uint64_t pcnt = (uint64_t)p->pos.count;
                fwrite(&pcnt, sizeof(pcnt), 1, fp);
                for (size_t k = 0; k < p->pos.count; k++) {
                    uint64_t pos = (uint64_t)p->pos.positions[k];
                    fwrite(&pos, sizeof(pos), 1, fp);
                }
            }
            pl = pl->next;
        }
    }
    /* Sentinel for end of posting lists */
    uint32_t zero = 0;
    fwrite(&zero, sizeof(zero), 1, fp);

    pthread_rwlock_unlock((pthread_rwlock_t *)&idx->rwlock);
    fclose(fp);
    return 0;
}

GV_FTIndex *gv_ft_load(const char *path) {
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    /* Verify magic */
    char magic[7];
    if (fread(magic, 1, 7, fp) != 7 || memcmp(magic, "GV_FT01", 7) != 0) {
        fclose(fp);
        return NULL;
    }

    /* Read config */
    uint32_t lang, flags;
    uint64_t bs;
    if (fread(&lang, sizeof(lang), 1, fp) != 1 ||
        fread(&flags, sizeof(flags), 1, fp) != 1 ||
        fread(&bs, sizeof(bs), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    GV_FTConfig config;
    gv_ft_config_init(&config);
    config.language = (GV_FTLanguage)lang;
    config.enable_stemming = (flags & 0x01) ? 1 : 0;
    config.enable_phrase_match = (flags & 0x02) ? 1 : 0;
    config.use_blockmax_wand = (flags & 0x04) ? 1 : 0;
    config.block_size = (size_t)bs;

    GV_FTIndex *idx = gv_ft_create(&config);
    if (!idx) { fclose(fp); return NULL; }

    /* Read stats */
    uint64_t td, tt, tl;
    if (fread(&td, sizeof(td), 1, fp) != 1 ||
        fread(&tt, sizeof(tt), 1, fp) != 1 ||
        fread(&tl, sizeof(tl), 1, fp) != 1) {
        gv_ft_destroy(idx);
        fclose(fp);
        return NULL;
    }

    /* Read document info */
    while (1) {
        uint64_t did;
        if (fread(&did, sizeof(did), 1, fp) != 1) break;
        if (did == UINT64_MAX) break;

        uint64_t dl;
        if (fread(&dl, sizeof(dl), 1, fp) != 1) break;

        FT_DocInfo *di = ft_get_or_create_doc_info(idx, (size_t)did);
        if (di) {
            di->doc_length = (size_t)dl;
        }
    }
    idx->total_doc_length = (size_t)tl;

    /* Read posting lists */
    while (1) {
        uint32_t term_len;
        if (fread(&term_len, sizeof(term_len), 1, fp) != 1) break;
        if (term_len == 0) break;

        char *term = malloc((size_t)term_len + 1);
        if (!term) break;
        if (fread(term, 1, term_len, fp) != term_len) { free(term); break; }
        term[term_len] = '\0';

        uint64_t pcount;
        if (fread(&pcount, sizeof(pcount), 1, fp) != 1) { free(term); break; }

        FT_PostingList *pl = ft_get_or_create_posting_list(idx, term);
        free(term);
        if (!pl) break;

        /* Ensure capacity */
        if ((size_t)pcount > pl->capacity) {
            FT_Posting *buf = realloc(pl->postings, (size_t)pcount * sizeof(FT_Posting));
            if (!buf) break;
            pl->postings = buf;
            pl->capacity = (size_t)pcount;
        }

        for (uint64_t j = 0; j < pcount; j++) {
            uint64_t did, tf, pcnt;
            if (fread(&did, sizeof(did), 1, fp) != 1) goto done;
            if (fread(&tf, sizeof(tf), 1, fp) != 1) goto done;
            if (fread(&pcnt, sizeof(pcnt), 1, fp) != 1) goto done;

            FT_Posting *p = &pl->postings[pl->count];
            memset(p, 0, sizeof(*p));
            p->doc_id = (size_t)did;
            p->term_freq = (size_t)tf;

            if (pcnt > 0) {
                p->pos.positions = malloc((size_t)pcnt * sizeof(size_t));
                if (!p->pos.positions) goto done;
                p->pos.capacity = (size_t)pcnt;
                for (uint64_t k = 0; k < pcnt; k++) {
                    uint64_t pos;
                    if (fread(&pos, sizeof(pos), 1, fp) != 1) goto done;
                    p->pos.positions[p->pos.count++] = (size_t)pos;
                }
            }
            pl->count++;
        }
    }

done:
    fclose(fp);
    return idx;
}
