#ifndef GIGAVECTOR_GV_MEMORY_CONSOLIDATION_H
#define GIGAVECTOR_GV_MEMORY_CONSOLIDATION_H

#include <stddef.h>

#include "gv_memory_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Memory pair for consolidation.
 */
typedef struct {
    char *memory_id_1;              /**< First memory ID. */
    char *memory_id_2;              /**< Second memory ID. */
    float similarity;               /**< Similarity score (0.0-1.0). */
} GV_MemoryPair;

/**
 * @brief Find similar memories for consolidation.
 *
 * Uses vector similarity search to find memories that are similar enough
 * to be consolidated.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param threshold Similarity threshold (0.0-1.0).
 * @param pairs Output array of memory pairs; must be pre-allocated.
 * @param max_pairs Maximum number of pairs to find.
 * @param actual_count Output: actual number of pairs found.
 * @return 0 on success, -1 on error.
 */
int gv_memory_find_similar(GV_MemoryLayer *layer, double threshold,
                           GV_MemoryPair *pairs, size_t max_pairs,
                           size_t *actual_count);

/**
 * @brief Consolidate two memories using specified strategy.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id_1 First memory ID; must be non-NULL.
 * @param memory_id_2 Second memory ID; must be non-NULL.
 * @param strategy Consolidation strategy to use.
 * @return Consolidated memory ID (caller must free) or NULL on failure.
 */
char *gv_memory_consolidate_pair(GV_MemoryLayer *layer,
                                  const char *memory_id_1,
                                  const char *memory_id_2,
                                  GV_ConsolidationStrategy strategy);

/**
 * @brief Merge two memories into one.
 *
 * Combines content and metadata from both memories into a single memory.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id_1 First memory ID; must be non-NULL.
 * @param memory_id_2 Second memory ID; must be non-NULL.
 * @return Merged memory ID (caller must free) or NULL on failure.
 */
char *gv_memory_merge(GV_MemoryLayer *layer, const char *memory_id_1,
                      const char *memory_id_2);

/**
 * @brief Update existing memory with new information.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param existing_memory_id Existing memory ID; must be non-NULL.
 * @param new_memory_id New memory ID to merge in; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_memory_update_from_new(GV_MemoryLayer *layer,
                               const char *existing_memory_id,
                               const char *new_memory_id);

/**
 * @brief Create relationship link between two memories.
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id_1 First memory ID; must be non-NULL.
 * @param memory_id_2 Second memory ID; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_memory_link(GV_MemoryLayer *layer, const char *memory_id_1,
                    const char *memory_id_2);

/**
 * @brief Archive a memory (mark as redundant but keep for history).
 *
 * @param layer Memory layer; must be non-NULL.
 * @param memory_id Memory ID to archive; must be non-NULL.
 * @return 0 on success, -1 on error.
 */
int gv_memory_archive(GV_MemoryLayer *layer, const char *memory_id);

/**
 * @brief Free memory pair structure.
 *
 * @param pair Pair to free; safe to call with NULL.
 */
void gv_memory_pair_free(GV_MemoryPair *pair);

/**
 * @brief Free array of memory pairs.
 *
 * @param pairs Array of pairs; can be NULL.
 * @param count Number of pairs in array.
 */
void gv_memory_pairs_free(GV_MemoryPair *pairs, size_t count);

#ifdef __cplusplus
}
#endif

#endif

