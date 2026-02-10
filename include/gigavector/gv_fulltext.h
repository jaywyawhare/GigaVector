#ifndef GIGAVECTOR_GV_FULLTEXT_H
#define GIGAVECTOR_GV_FULLTEXT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_fulltext.h
 * @brief Full-text search enhancements with Porter stemming, phrase matching,
 *        multilingual tokenization, and BlockMax WAND optimization.
 *
 * Provides an advanced full-text search index built on BM25 scoring with
 * language-aware stemming, exact phrase matching via positional posting lists,
 * and BlockMax WAND query evaluation for efficient top-k retrieval.
 */

/* ============================================================================
 * Language Enumeration
 * ============================================================================ */

/**
 * @brief Supported languages for stemming and stopword removal.
 */
typedef enum {
    GV_LANG_ENGLISH    = 0,         /**< English (Porter stemmer). */
    GV_LANG_GERMAN     = 1,         /**< German (suffix stripping). */
    GV_LANG_FRENCH     = 2,         /**< French (suffix stripping). */
    GV_LANG_SPANISH    = 3,         /**< Spanish (suffix stripping). */
    GV_LANG_ITALIAN    = 4,         /**< Italian (suffix stripping). */
    GV_LANG_PORTUGUESE = 5,         /**< Portuguese (suffix stripping). */
    GV_LANG_AUTO       = 6          /**< Auto-detect language. */
} GV_FTLanguage;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Full-text index configuration.
 */
typedef struct {
    GV_FTLanguage language;         /**< Language for stemming/stopwords (default: ENGLISH). */
    int enable_stemming;            /**< Enable stemming during indexing and search (default: 1). */
    int enable_phrase_match;        /**< Enable positional indexing for phrase queries (default: 1). */
    int use_blockmax_wand;          /**< Use BlockMax WAND optimization for scoring (default: 1). */
    size_t block_size;              /**< Posting list block size for BlockMax WAND (default: 128). */
} GV_FTConfig;

/* ============================================================================
 * Opaque Index Handle
 * ============================================================================ */

/**
 * @brief Opaque full-text index handle.
 */
typedef struct GV_FTIndex GV_FTIndex;

/* ============================================================================
 * Result Structure
 * ============================================================================ */

/**
 * @brief Full-text search result entry.
 */
typedef struct {
    size_t doc_id;                  /**< Document ID. */
    float score;                    /**< BM25 relevance score. */
    size_t *match_positions;        /**< Array of term match positions in the document. */
    size_t match_count;             /**< Number of match positions. */
} GV_FTResult;

/* ============================================================================
 * Configuration Initialization
 * ============================================================================ */

/**
 * @brief Initialize full-text configuration with defaults.
 *
 * Default values:
 * - language: GV_LANG_ENGLISH
 * - enable_stemming: 1
 * - enable_phrase_match: 1
 * - use_blockmax_wand: 1
 * - block_size: 128
 *
 * @param config Configuration to initialize.
 */
void gv_ft_config_init(GV_FTConfig *config);

/* ============================================================================
 * Index Lifecycle
 * ============================================================================ */

/**
 * @brief Create a full-text index.
 *
 * @param config Full-text configuration (NULL for defaults).
 * @return Full-text index instance, or NULL on error.
 */
GV_FTIndex *gv_ft_create(const GV_FTConfig *config);

/**
 * @brief Destroy a full-text index and free all resources.
 *
 * @param idx Full-text index instance (safe to call with NULL).
 */
void gv_ft_destroy(GV_FTIndex *idx);

/* ============================================================================
 * Indexing Operations
 * ============================================================================ */

/**
 * @brief Add a document to the full-text index.
 *
 * The text is tokenized, lowercased, stopword-filtered, and stemmed
 * according to the configured language. Positional information is stored
 * when phrase matching is enabled.
 *
 * @param idx Full-text index.
 * @param doc_id Document ID.
 * @param text Document text content.
 * @return 0 on success, -1 on error.
 */
int gv_ft_add_document(GV_FTIndex *idx, size_t doc_id, const char *text);

/**
 * @brief Remove a document from the full-text index.
 *
 * @param idx Full-text index.
 * @param doc_id Document ID to remove.
 * @return 0 on success, -1 on error (not found).
 */
int gv_ft_remove_document(GV_FTIndex *idx, size_t doc_id);

/* ============================================================================
 * Search Operations
 * ============================================================================ */

/**
 * @brief Search the index with a text query using BM25 scoring.
 *
 * When BlockMax WAND is enabled, the query is evaluated using block-level
 * upper-bound pruning for efficient top-k retrieval.
 *
 * @param idx Full-text index.
 * @param query Query text (tokenized and stemmed like documents).
 * @param limit Maximum number of results to return.
 * @param results Output results array (must be pre-allocated with limit elements).
 * @return Number of results found (>= 0), or -1 on error.
 */
int gv_ft_search(const GV_FTIndex *idx, const char *query, size_t limit,
                 GV_FTResult *results);

/**
 * @brief Search for an exact phrase in the index.
 *
 * Finds documents containing all phrase terms at consecutive positions.
 * Results are ranked by BM25 score. Requires enable_phrase_match in config.
 *
 * @param idx Full-text index.
 * @param phrase Phrase to search for.
 * @param limit Maximum number of results to return.
 * @param results Output results array (must be pre-allocated with limit elements).
 * @return Number of results found (>= 0), or -1 on error.
 */
int gv_ft_search_phrase(const GV_FTIndex *idx, const char *phrase, size_t limit,
                        GV_FTResult *results);

/* ============================================================================
 * Stemming
 * ============================================================================ */

/**
 * @brief Stem a single word using the specified language rules.
 *
 * For English, applies the classic Porter stemming algorithm.
 * For other languages, applies simplified suffix stripping rules.
 *
 * @param word Input word to stem.
 * @param lang Language to use for stemming.
 * @param output Output buffer for the stemmed word.
 * @param output_size Size of the output buffer.
 * @return 0 on success, -1 on error (buffer too small or invalid input).
 */
int gv_ft_stem(const char *word, GV_FTLanguage lang, char *output, size_t output_size);

/* ============================================================================
 * Result Cleanup
 * ============================================================================ */

/**
 * @brief Free match_positions arrays in search results.
 *
 * Frees the dynamically allocated match_positions arrays within each result.
 * Does not free the results array itself (caller-managed).
 *
 * @param results Results array to clean up.
 * @param count Number of results.
 */
void gv_ft_free_results(GV_FTResult *results, size_t count);

/* ============================================================================
 * Index Information
 * ============================================================================ */

/**
 * @brief Get the number of documents in the index.
 *
 * @param idx Full-text index.
 * @return Number of indexed documents.
 */
size_t gv_ft_doc_count(const GV_FTIndex *idx);

/* ============================================================================
 * Persistence
 * ============================================================================ */

/**
 * @brief Save the full-text index to a file.
 *
 * @param idx Full-text index.
 * @param path Output file path.
 * @return 0 on success, -1 on error.
 */
int gv_ft_save(const GV_FTIndex *idx, const char *path);

/**
 * @brief Load a full-text index from a file.
 *
 * @param path Input file path.
 * @return Full-text index instance, or NULL on error.
 */
GV_FTIndex *gv_ft_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_FULLTEXT_H */
