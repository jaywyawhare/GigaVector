#ifndef GIGAVECTOR_GV_IMPORTANCE_H
#define GIGAVECTOR_GV_IMPORTANCE_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_importance.h
 * @brief State-of-the-art importance scoring for memory systems.
 *
 * This module implements a multi-factor importance scoring algorithm inspired by:
 * - Cognitive science research on memory consolidation (Ebbinghaus forgetting curve)
 * - Spaced repetition systems (SM-2 algorithm concepts)
 * - Modern ML-based ranking (BM25, learning-to-rank principles)
 * - mem0 and supermemory approaches (semantic similarity + recency)
 *
 * The final importance score combines:
 * 1. Content-based features (informativeness, specificity, entity density)
 * 2. Temporal factors (recency, decay, periodic access patterns)
 * 3. Access patterns (retrieval frequency, retrieval recency)
 * 4. Contextual signals (emotional salience, personal relevance markers)
 * 5. Structural features (relationships to other memories)
 */

/* ============================================================================
 * Configuration and Weight Structures
 * ============================================================================ */

/**
 * @brief Weights for different scoring components.
 *
 * All weights should sum to 1.0 for normalized output.
 * Default weights are calibrated based on cognitive science research.
 */
typedef struct {
    double content_weight;          /**< Weight for content-based score (default: 0.30) */
    double temporal_weight;         /**< Weight for temporal score (default: 0.25) */
    double access_weight;           /**< Weight for access pattern score (default: 0.20) */
    double salience_weight;         /**< Weight for emotional/personal salience (default: 0.15) */
    double structural_weight;       /**< Weight for relationship/graph score (default: 0.10) */
} GV_ImportanceWeights;

/**
 * @brief Temporal decay configuration.
 *
 * Based on Ebbinghaus forgetting curve: R = e^(-t/S)
 * where R is retention, t is time, S is stability.
 */
typedef struct {
    double half_life_hours;         /**< Time for score to decay to 50% (default: 168 = 1 week) */
    double min_decay_factor;        /**< Minimum decay multiplier (default: 0.1) */
    double recency_boost_hours;     /**< Recent memories get boosted within this window (default: 24) */
    double recency_boost_factor;    /**< Boost factor for recent memories (default: 1.5) */
} GV_TemporalDecayConfig;

/**
 * @brief Access pattern tracking configuration.
 *
 * Inspired by spaced repetition: memories accessed more frequently
 * and at optimal intervals are considered more important.
 */
typedef struct {
    double retrieval_boost_base;    /**< Base boost per retrieval (default: 0.05) */
    double retrieval_boost_decay;   /**< Decay factor for old retrievals (default: 0.95) */
    double optimal_interval_hours;  /**< Optimal retrieval interval for max boost (default: 48) */
    double interval_tolerance;      /**< Tolerance for interval matching (default: 0.5) */
    size_t max_tracked_accesses;    /**< Maximum access events to track (default: 100) */
} GV_AccessPatternConfig;

/**
 * @brief Content analysis configuration.
 */
typedef struct {
    double min_word_count;          /**< Minimum words for full score (default: 5) */
    double optimal_word_count;      /**< Optimal word count for content (default: 20) */
    double max_word_count;          /**< Words beyond this don't add value (default: 100) */
    int enable_entity_detection;    /**< Enable named entity bonus (default: 1) */
    int enable_specificity_scoring; /**< Enable specificity analysis (default: 1) */
} GV_ContentAnalysisConfig;

/**
 * @brief Main importance scoring configuration.
 */
typedef struct {
    GV_ImportanceWeights weights;
    GV_TemporalDecayConfig temporal;
    GV_AccessPatternConfig access;
    GV_ContentAnalysisConfig content;
    int enable_adaptive_weights;    /**< Dynamically adjust weights based on patterns (default: 0) */
    double base_score;              /**< Starting score for new memories (default: 0.5) */
} GV_ImportanceConfig;

/* ============================================================================
 * Access Event Tracking
 * ============================================================================ */

/**
 * @brief Single access event for a memory.
 */
typedef struct {
    time_t timestamp;               /**< When the access occurred */
    double relevance_at_access;     /**< Relevance score when accessed (0.0-1.0) */
    int access_type;                /**< 0=search result, 1=direct lookup, 2=related fetch */
} GV_AccessEvent;

/**
 * @brief Access history for a memory.
 */
typedef struct {
    GV_AccessEvent *events;         /**< Array of access events */
    size_t event_count;             /**< Number of events recorded */
    size_t event_capacity;          /**< Allocated capacity */
    time_t last_access;             /**< Most recent access timestamp */
    uint32_t total_accesses;        /**< Total lifetime access count */
    double avg_relevance;           /**< Running average relevance when accessed */
} GV_AccessHistory;

/* ============================================================================
 * Scoring Context and Results
 * ============================================================================ */

/**
 * @brief Input context for importance scoring.
 *
 * Provides all information needed to compute a comprehensive importance score.
 */
typedef struct {
    /* Content features */
    const char *content;            /**< Memory content text */
    size_t content_length;          /**< Length in bytes */

    /* Temporal features */
    time_t creation_time;           /**< When memory was created */
    time_t last_modified;           /**< Last modification time */
    time_t current_time;            /**< Current time (for decay calculation) */

    /* Access pattern features */
    const GV_AccessHistory *access_history; /**< Access history, can be NULL */

    /* Structural features */
    size_t relationship_count;      /**< Number of related memories */
    size_t incoming_links;          /**< Memories that reference this one */
    size_t outgoing_links;          /**< Memories this one references */

    /* Optional: embedding for semantic analysis */
    const float *embedding;         /**< Vector embedding, can be NULL */
    size_t embedding_dim;           /**< Embedding dimension */

    /* Optional: query context for relevance boosting */
    const char *query_context;      /**< Current query if in search context */
    double semantic_similarity;     /**< Pre-computed similarity to query (0.0-1.0) */
} GV_ImportanceContext;

/**
 * @brief Detailed breakdown of importance score components.
 */
typedef struct {
    double final_score;             /**< Final combined score (0.0-1.0) */

    /* Component scores (each 0.0-1.0) */
    double content_score;           /**< Content-based score */
    double temporal_score;          /**< Temporal/recency score */
    double access_score;            /**< Access pattern score */
    double salience_score;          /**< Salience/emotional score */
    double structural_score;        /**< Relationship/graph score */

    /* Sub-component details */
    double informativeness;         /**< Content informativeness */
    double specificity;             /**< Content specificity */
    double entity_density;          /**< Named entity density */
    double decay_factor;            /**< Applied temporal decay */
    double retrieval_boost;         /**< Boost from retrievals */
    double recency_bonus;           /**< Bonus for recent memory */

    /* Confidence and metadata */
    double confidence;              /**< Confidence in score (0.0-1.0) */
    int factors_used;               /**< Bitmask of factors that contributed */
} GV_ImportanceResult;

/* Factor bitmask values */
#define GV_FACTOR_CONTENT     (1 << 0)
#define GV_FACTOR_TEMPORAL    (1 << 1)
#define GV_FACTOR_ACCESS      (1 << 2)
#define GV_FACTOR_SALIENCE    (1 << 3)
#define GV_FACTOR_STRUCTURAL  (1 << 4)
#define GV_FACTOR_QUERY       (1 << 5)

/* ============================================================================
 * Core API Functions
 * ============================================================================ */

/**
 * @brief Create default importance configuration.
 *
 * @return Default configuration with research-backed weights.
 */
GV_ImportanceConfig gv_importance_config_default(void);

/**
 * @brief Calculate importance score with full context.
 *
 * This is the main scoring function that combines all factors.
 *
 * @param config Scoring configuration; NULL uses defaults.
 * @param context Scoring context with all available features.
 * @param result Output: detailed scoring result.
 * @return 0 on success, -1 on error.
 */
int gv_importance_calculate(const GV_ImportanceConfig *config,
                            const GV_ImportanceContext *context,
                            GV_ImportanceResult *result);

/**
 * @brief Quick importance score from content only.
 *
 * Simplified scoring when only content is available.
 * Uses content analysis + default temporal assumptions.
 *
 * @param content Memory content text.
 * @param len Content length.
 * @return Importance score (0.0-1.0).
 */
double gv_importance_score_content(const char *content, size_t len);

/**
 * @brief Score extracted facts (optimized for short LLM-extracted content).
 *
 * Unlike gv_importance_score_content(), this function is optimized for
 * short extracted facts like "Name is John" or "Works at Google".
 * It does NOT penalize short content since LLM extraction already
 * filtered for important facts.
 *
 * Scoring factors:
 * - Specificity (numbers, proper nouns, concrete details)
 * - Entity density (named entities, structured data)
 * - Information density (unique words / total words)
 *
 * @param content Extracted fact content.
 * @param len Content length.
 * @return Importance score (0.0-1.0).
 */
double gv_importance_score_extracted(const char *content, size_t len);

/**
 * @brief Calculate temporal decay factor.
 *
 * Computes decay based on Ebbinghaus forgetting curve.
 *
 * @param config Temporal decay configuration; NULL uses defaults.
 * @param age_seconds Age of memory in seconds.
 * @return Decay factor (0.0-1.0).
 */
double gv_importance_temporal_decay(const GV_TemporalDecayConfig *config,
                                     double age_seconds);

/**
 * @brief Calculate access pattern score.
 *
 * Scores based on retrieval frequency and patterns.
 *
 * @param config Access pattern configuration; NULL uses defaults.
 * @param history Access history for the memory.
 * @param current_time Current timestamp.
 * @return Access pattern score (0.0-1.0).
 */
double gv_importance_access_score(const GV_AccessPatternConfig *config,
                                   const GV_AccessHistory *history,
                                   time_t current_time);

/**
 * @brief Update importance score after memory access.
 *
 * Should be called when a memory is retrieved to update its importance.
 *
 * @param history Access history to update (will be modified).
 * @param timestamp When the access occurred.
 * @param relevance Relevance score at access time (0.0-1.0).
 * @param access_type Type of access (0=search, 1=direct, 2=related).
 * @return 0 on success, -1 on error.
 */
int gv_importance_record_access(GV_AccessHistory *history,
                                 time_t timestamp,
                                 double relevance,
                                 int access_type);

/* ============================================================================
 * Content Analysis Functions
 * ============================================================================ */

/**
 * @brief Calculate content informativeness score.
 *
 * Measures information density using:
 * - Lexical diversity (unique words / total words)
 * - Average word length (proxy for vocabulary sophistication)
 * - Punctuation patterns (indicates structure)
 *
 * @param content Text content.
 * @param len Content length.
 * @return Informativeness score (0.0-1.0).
 */
double gv_importance_informativeness(const char *content, size_t len);

/**
 * @brief Calculate content specificity score.
 *
 * Detects specific vs. generic content using:
 * - Presence of numbers, dates, proper nouns
 * - Quantifiers and specific details
 * - Absence of vague language patterns
 *
 * @param content Text content.
 * @param len Content length.
 * @return Specificity score (0.0-1.0).
 */
double gv_importance_specificity(const char *content, size_t len);

/**
 * @brief Calculate salience indicators.
 *
 * Detects emotional and personal relevance markers:
 * - First person pronouns (I, my, me)
 * - Emotional keywords
 * - Preference indicators (like, love, hate, prefer)
 * - Important markers (important, remember, always, never)
 *
 * @param content Text content.
 * @param len Content length.
 * @return Salience score (0.0-1.0).
 */
double gv_importance_salience(const char *content, size_t len);

/**
 * @brief Detect named entities in content.
 *
 * Simple pattern-based entity detection:
 * - Capitalized words (potential proper nouns)
 * - Email patterns
 * - URL patterns
 * - Number patterns (dates, amounts, etc.)
 *
 * @param content Text content.
 * @param len Content length.
 * @return Entity density score (0.0-1.0).
 */
double gv_importance_entity_density(const char *content, size_t len);

/* ============================================================================
 * Access History Management
 * ============================================================================ */

/**
 * @brief Initialize access history structure.
 *
 * @param history History structure to initialize.
 * @param initial_capacity Initial event array capacity.
 * @return 0 on success, -1 on error.
 */
int gv_access_history_init(GV_AccessHistory *history, size_t initial_capacity);

/**
 * @brief Free access history resources.
 *
 * @param history History to free; safe to call with NULL.
 */
void gv_access_history_free(GV_AccessHistory *history);

/**
 * @brief Serialize access history to JSON string.
 *
 * @param history History to serialize.
 * @return JSON string (caller must free) or NULL on error.
 */
char *gv_access_history_serialize(const GV_AccessHistory *history);

/**
 * @brief Deserialize access history from JSON string.
 *
 * @param json JSON string to parse.
 * @param history Output history structure.
 * @return 0 on success, -1 on error.
 */
int gv_access_history_deserialize(const char *json, GV_AccessHistory *history);

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

/**
 * @brief Calculate importance scores for multiple memories.
 *
 * Efficient batch processing with shared configuration.
 *
 * @param config Scoring configuration; NULL uses defaults.
 * @param contexts Array of scoring contexts.
 * @param results Array of results (must be pre-allocated).
 * @param count Number of memories to score.
 * @return Number of successfully scored memories.
 */
int gv_importance_calculate_batch(const GV_ImportanceConfig *config,
                                   const GV_ImportanceContext *contexts,
                                   GV_ImportanceResult *results,
                                   size_t count);

/**
 * @brief Re-rank memories by importance.
 *
 * Takes pre-computed similarity scores and re-ranks by combined importance.
 *
 * @param config Scoring configuration; NULL uses defaults.
 * @param contexts Array of scoring contexts (with semantic_similarity set).
 * @param results Array of results.
 * @param indices Output: sorted indices (most important first).
 * @param count Number of memories.
 * @param similarity_weight Weight for similarity vs importance (0.0-1.0).
 * @return 0 on success, -1 on error.
 */
int gv_importance_rerank(const GV_ImportanceConfig *config,
                          const GV_ImportanceContext *contexts,
                          GV_ImportanceResult *results,
                          size_t *indices,
                          size_t count,
                          double similarity_weight);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_IMPORTANCE_H */
