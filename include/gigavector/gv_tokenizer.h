#ifndef GIGAVECTOR_GV_TOKENIZER_H
#define GIGAVECTOR_GV_TOKENIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_tokenizer.h
 * @brief Text tokenization for full-text search.
 *
 * Provides tokenization utilities for BM25 and hybrid search.
 */

/**
 * @brief Tokenizer type enumeration.
 */
typedef enum {
    GV_TOKENIZER_WHITESPACE = 0,    /**< Split on whitespace. */
    GV_TOKENIZER_SIMPLE = 1,        /**< Split on non-alphanumeric, lowercase. */
    GV_TOKENIZER_STANDARD = 2       /**< Standard tokenizer with stopwords. */
} GV_TokenizerType;

/**
 * @brief Tokenizer configuration.
 */
typedef struct {
    GV_TokenizerType type;          /**< Tokenizer type (default: SIMPLE). */
    int lowercase;                  /**< Convert to lowercase (default: 1). */
    int remove_stopwords;           /**< Remove common stopwords (default: 0). */
    size_t min_token_length;        /**< Minimum token length (default: 1). */
    size_t max_token_length;        /**< Maximum token length (default: 256). */
} GV_TokenizerConfig;

/**
 * @brief Token structure.
 */
typedef struct {
    char *text;                     /**< Token text (null-terminated). */
    size_t position;                /**< Position in original text. */
    size_t offset_start;            /**< Start offset in original text. */
    size_t offset_end;              /**< End offset in original text. */
} GV_Token;

/**
 * @brief Tokenization result.
 */
typedef struct {
    GV_Token *tokens;               /**< Array of tokens. */
    size_t count;                   /**< Number of tokens. */
    size_t capacity;                /**< Allocated capacity. */
} GV_TokenList;

/**
 * @brief Opaque tokenizer handle.
 */
typedef struct GV_Tokenizer GV_Tokenizer;

/* Configuration */

/**
 * @brief Initialize tokenizer configuration with defaults.
 *
 * @param config Configuration to initialize.
 */
void gv_tokenizer_config_init(GV_TokenizerConfig *config);

/* Tokenizer Lifecycle */

/**
 * @brief Create a tokenizer.
 *
 * @param config Tokenizer configuration (NULL for defaults).
 * @return Tokenizer instance, or NULL on error.
 */
GV_Tokenizer *gv_tokenizer_create(const GV_TokenizerConfig *config);

/**
 * @brief Destroy a tokenizer.
 *
 * @param tokenizer Tokenizer instance (safe to call with NULL).
 */
void gv_tokenizer_destroy(GV_Tokenizer *tokenizer);

/* Tokenization */

/**
 * @brief Tokenize text.
 *
 * @param tokenizer Tokenizer instance.
 * @param text Text to tokenize.
 * @param text_len Length of text (or 0 for null-terminated).
 * @param result Output token list.
 * @return 0 on success, -1 on error.
 */
int gv_tokenizer_tokenize(GV_Tokenizer *tokenizer, const char *text,
                          size_t text_len, GV_TokenList *result);

/**
 * @brief Free token list resources.
 *
 * @param list Token list to free.
 */
void gv_token_list_free(GV_TokenList *list);

/* Utility Functions */

/**
 * @brief Tokenize text with default settings (simple tokenizer).
 *
 * Convenience function for quick tokenization.
 *
 * @param text Text to tokenize.
 * @param result Output token list.
 * @return 0 on success, -1 on error.
 */
int gv_tokenize_simple(const char *text, GV_TokenList *result);

/**
 * @brief Get unique tokens from a token list.
 *
 * @param list Input token list.
 * @param unique_tokens Output array of unique token strings.
 * @param unique_count Output count of unique tokens.
 * @return 0 on success, -1 on error.
 */
int gv_token_list_unique(const GV_TokenList *list, char ***unique_tokens, size_t *unique_count);

/**
 * @brief Free unique tokens array.
 *
 * @param tokens Array to free.
 * @param count Number of tokens.
 */
void gv_unique_tokens_free(char **tokens, size_t count);

/**
 * @brief Check if a word is a stopword.
 *
 * @param word Word to check.
 * @return 1 if stopword, 0 otherwise.
 */
int gv_is_stopword(const char *word);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_TOKENIZER_H */
