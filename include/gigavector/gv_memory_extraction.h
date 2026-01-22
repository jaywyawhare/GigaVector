#ifndef GIGAVECTOR_GV_MEMORY_EXTRACTION_H
#define GIGAVECTOR_GV_MEMORY_EXTRACTION_H

#include <stddef.h>

#include "gv_memory_layer.h"
#include "gv_llm.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Extracted memory candidate structure.
 */
typedef struct {
    char *content;                  /**< Extracted memory content. */
    double importance_score;        /**< Importance score (0.0-1.0). */
    GV_MemoryType memory_type;       /**< Detected memory type. */
    char *extraction_context;       /**< Context where memory was extracted. */
} GV_MemoryCandidate;

/**
 * @brief Extract memory candidates from conversation text using LLM.
 *
 * This function uses LLM to extract factual memories from conversations,
 * similar to Mem0's approach. Falls back to heuristics if LLM unavailable.
 *
 * @param llm LLM handle; NULL to use heuristics only.
 * @param conversation Conversation text; must be non-NULL.
 * @param conversation_id Conversation identifier; can be NULL.
 * @param is_agent_memory 1 if extracting agent memories, 0 for user memories.
 * @param custom_prompt Custom extraction prompt; NULL for default.
 * @param candidates Output array; must be pre-allocated.
 * @param max_candidates Maximum number of candidates to extract.
 * @param actual_count Output: actual number of candidates found.
 * @return 0 on success, -1 on error.
 */
int gv_memory_extract_candidates_from_conversation_llm(GV_LLM *llm,
                                                        const char *conversation,
                                                        const char *conversation_id,
                                                        int is_agent_memory,
                                                        const char *custom_prompt,
                                                        GV_MemoryCandidate *candidates,
                                                        size_t max_candidates,
                                                        size_t *actual_count);

/**
 * @brief Extract memory candidates from conversation text (heuristics fallback).
 *
 * This function analyzes conversation text and identifies potential memories
 * based on importance scoring and content analysis (no LLM).
 *
 * @param conversation Conversation text; must be non-NULL.
 * @param conversation_id Conversation identifier; can be NULL.
 * @param threshold Minimum importance threshold (0.0-1.0).
 * @param candidates Output array; must be pre-allocated.
 * @param max_candidates Maximum number of candidates to extract.
 * @param actual_count Output: actual number of candidates found.
 * @return 0 on success, -1 on error.
 */
int gv_memory_extract_candidates_from_conversation(const char *conversation,
                                                    const char *conversation_id,
                                                    double threshold,
                                                    GV_MemoryCandidate *candidates,
                                                    size_t max_candidates,
                                                    size_t *actual_count);

/**
 * @brief Extract memory candidates from plain text.
 *
 * @param text Document text; must be non-NULL.
 * @param source Source identifier; can be NULL.
 * @param threshold Minimum importance threshold (0.0-1.0).
 * @param candidates Output array; must be pre-allocated.
 * @param max_candidates Maximum number of candidates to extract.
 * @param actual_count Output: actual number of candidates found.
 * @return 0 on success, -1 on error.
 */
int gv_memory_extract_candidates_from_text(const char *text,
                                            const char *source,
                                            double threshold,
                                            GV_MemoryCandidate *candidates,
                                            size_t max_candidates,
                                            size_t *actual_count);

/**
 * @brief Score a memory candidate for importance.
 *
 * @param candidate Memory candidate to score; must be non-NULL.
 * @return Importance score (0.0-1.0).
 */
double gv_memory_score_candidate(const GV_MemoryCandidate *candidate);

/**
 * @brief Detect memory type from content.
 *
 * @param content Memory content; must be non-NULL.
 * @return Detected memory type.
 */
GV_MemoryType gv_memory_detect_type(const char *content);

/**
 * @brief Free memory candidate structure.
 *
 * @param candidate Candidate to free; safe to call with NULL.
 */
void gv_memory_candidate_free(GV_MemoryCandidate *candidate);

/**
 * @brief Free array of memory candidates.
 *
 * @param candidates Array of candidates; can be NULL.
 * @param count Number of candidates in array.
 */
void gv_memory_candidates_free(GV_MemoryCandidate *candidates, size_t count);

#ifdef __cplusplus
}
#endif

#endif

