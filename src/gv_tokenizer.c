/**
 * @file gv_tokenizer.c
 * @brief Text tokenization implementation.
 */

#include "gigavector/gv_tokenizer.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

struct GV_Tokenizer {
    GV_TokenizerConfig config;
};

/* ============================================================================
 * Stopwords List
 * ============================================================================ */

static const char *STOPWORDS[] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with", "you", "your",
    NULL
};

#define STOPWORDS_COUNT (sizeof(STOPWORDS) / sizeof(STOPWORDS[0]) - 1)

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_TokenizerConfig DEFAULT_CONFIG = {
    .type = GV_TOKENIZER_SIMPLE,
    .lowercase = 1,
    .remove_stopwords = 0,
    .min_token_length = 1,
    .max_token_length = 256
};

void gv_tokenizer_config_init(GV_TokenizerConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_Tokenizer *gv_tokenizer_create(const GV_TokenizerConfig *config) {
    GV_Tokenizer *tokenizer = calloc(1, sizeof(GV_Tokenizer));
    if (!tokenizer) return NULL;

    tokenizer->config = config ? *config : DEFAULT_CONFIG;

    return tokenizer;
}

void gv_tokenizer_destroy(GV_Tokenizer *tokenizer) {
    free(tokenizer);
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

static int is_token_char_whitespace(char c) {
    return !isspace((unsigned char)c);
}

static int is_token_char_simple(char c) {
    return isalnum((unsigned char)c);
}

static int token_list_grow(GV_TokenList *list) {
    size_t new_capacity = list->capacity == 0 ? 16 : list->capacity * 2;
    GV_Token *new_tokens = realloc(list->tokens, new_capacity * sizeof(GV_Token));
    if (!new_tokens) return -1;
    list->tokens = new_tokens;
    list->capacity = new_capacity;
    return 0;
}

static int add_token(GV_TokenList *list, const char *start, size_t len,
                     size_t position, size_t offset_start, size_t offset_end,
                     const GV_TokenizerConfig *config) {
    /* Check length constraints */
    if (len < config->min_token_length || len > config->max_token_length) {
        return 0;  /* Skip but not an error */
    }

    /* Grow if needed */
    if (list->count >= list->capacity) {
        if (token_list_grow(list) != 0) return -1;
    }

    /* Allocate token text */
    char *text = malloc(len + 1);
    if (!text) return -1;

    /* Copy and optionally lowercase */
    if (config->lowercase) {
        for (size_t i = 0; i < len; i++) {
            text[i] = (char)tolower((unsigned char)start[i]);
        }
    } else {
        memcpy(text, start, len);
    }
    text[len] = '\0';

    /* Check stopwords */
    if (config->remove_stopwords && gv_is_stopword(text)) {
        free(text);
        return 0;  /* Skip stopword */
    }

    /* Add to list */
    GV_Token *token = &list->tokens[list->count++];
    token->text = text;
    token->position = position;
    token->offset_start = offset_start;
    token->offset_end = offset_end;

    return 0;
}

/* ============================================================================
 * Tokenization
 * ============================================================================ */

int gv_tokenizer_tokenize(GV_Tokenizer *tokenizer, const char *text,
                          size_t text_len, GV_TokenList *result) {
    if (!tokenizer || !text || !result) return -1;

    memset(result, 0, sizeof(*result));

    if (text_len == 0) {
        text_len = strlen(text);
    }

    if (text_len == 0) return 0;

    int (*is_token_char)(char);
    switch (tokenizer->config.type) {
        case GV_TOKENIZER_WHITESPACE:
            is_token_char = is_token_char_whitespace;
            break;
        case GV_TOKENIZER_SIMPLE:
        case GV_TOKENIZER_STANDARD:
        default:
            is_token_char = is_token_char_simple;
            break;
    }

    size_t position = 0;
    size_t i = 0;

    while (i < text_len) {
        /* Skip non-token characters */
        while (i < text_len && !is_token_char(text[i])) {
            i++;
        }

        if (i >= text_len) break;

        /* Start of token */
        size_t start = i;

        /* Find end of token */
        while (i < text_len && is_token_char(text[i])) {
            i++;
        }

        size_t len = i - start;
        if (len > 0) {
            if (add_token(result, text + start, len, position,
                          start, i, &tokenizer->config) != 0) {
                gv_token_list_free(result);
                return -1;
            }
            position++;
        }
    }

    return 0;
}

void gv_token_list_free(GV_TokenList *list) {
    if (!list) return;

    for (size_t i = 0; i < list->count; i++) {
        free(list->tokens[i].text);
    }
    free(list->tokens);
    memset(list, 0, sizeof(*list));
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

int gv_tokenize_simple(const char *text, GV_TokenList *result) {
    GV_Tokenizer *tokenizer = gv_tokenizer_create(NULL);
    if (!tokenizer) return -1;

    int ret = gv_tokenizer_tokenize(tokenizer, text, 0, result);
    gv_tokenizer_destroy(tokenizer);

    return ret;
}

int gv_token_list_unique(const GV_TokenList *list, char ***unique_tokens, size_t *unique_count) {
    if (!list || !unique_tokens || !unique_count) return -1;

    if (list->count == 0) {
        *unique_tokens = NULL;
        *unique_count = 0;
        return 0;
    }

    /* Allocate maximum possible size */
    char **tokens = malloc(list->count * sizeof(char *));
    if (!tokens) return -1;

    size_t count = 0;

    for (size_t i = 0; i < list->count; i++) {
        /* Check if already in list */
        int found = 0;
        for (size_t j = 0; j < count; j++) {
            if (strcmp(tokens[j], list->tokens[i].text) == 0) {
                found = 1;
                break;
            }
        }

        if (!found) {
            tokens[count] = strdup(list->tokens[i].text);
            if (!tokens[count]) {
                gv_unique_tokens_free(tokens, count);
                return -1;
            }
            count++;
        }
    }

    /* Shrink to actual size */
    if (count < list->count) {
        char **shrunk = realloc(tokens, count * sizeof(char *));
        if (shrunk) tokens = shrunk;
    }

    *unique_tokens = tokens;
    *unique_count = count;
    return 0;
}

void gv_unique_tokens_free(char **tokens, size_t count) {
    if (!tokens) return;
    for (size_t i = 0; i < count; i++) {
        free(tokens[i]);
    }
    free(tokens);
}

int gv_is_stopword(const char *word) {
    if (!word) return 0;

    for (size_t i = 0; i < STOPWORDS_COUNT; i++) {
        if (strcasecmp(word, STOPWORDS[i]) == 0) {
            return 1;
        }
    }
    return 0;
}
