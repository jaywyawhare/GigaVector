#ifndef GIGAVECTOR_RABITQ_H
#define GIGAVECTOR_RABITQ_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "core/types.h"
#include "search/distance.h"
#include "storage/soa_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * RaBitQ index: 1-bit randomised binary quantisation with exact-rerank.
 *
 * Vectors are L2-normalised, randomly sign-flipped, Walsh-Hadamard
 * transformed (O(d log d)), then binarised.  Search uses XNOR+popcount
 * for an O(n·d/64) first-pass, then reranks the top candidates with
 * exact cosine on the stored float vectors.
 */
typedef struct {
    uint64_t seed;          /**< RNG seed for random sign vector (default: 42). */
    size_t   rerank_factor; /**< Rerank top (rerank_factor * k) candidates; 0 = use 4. */
} GV_RaBitQConfig;

void  *rabitq_create(size_t dimension, const GV_RaBitQConfig *config,
                     GV_SoAStorage *soa_storage);

int    rabitq_insert(void *index, GV_Vector *vector);

int    rabitq_search(void *index, const GV_Vector *query, size_t k,
                     GV_SearchResult *results, GV_DistanceType distance_type,
                     const char *filter_key, const char *filter_value);

int    rabitq_range_search(void *index, const GV_Vector *query, float radius,
                           GV_SearchResult *results, size_t max_results,
                           GV_DistanceType distance_type,
                           const char *filter_key, const char *filter_value);

int    rabitq_delete(void *index, size_t vector_index);
int    rabitq_update(void *index, size_t vector_index,
                     const float *new_data, size_t dimension);

int    rabitq_save(const void *index, FILE *out, uint32_t version);
int    rabitq_load(void **index_ptr, FILE *in, size_t dimension,
                   uint32_t version);

size_t rabitq_count(const void *index);
void   rabitq_destroy(void *index);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_RABITQ_H */
