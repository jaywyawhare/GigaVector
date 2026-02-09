#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "gigavector/gv_dedup.h"

/* ------------------------------------------------------------------ */
/*  Linked-list node for hash bucket chaining                         */
/* ------------------------------------------------------------------ */
typedef struct GV_DedupBucketNode {
    size_t index;                    /**< Index into the flat vectors array. */
    struct GV_DedupBucketNode *next; /**< Next node in the chain. */
} GV_DedupBucketNode;

/* ------------------------------------------------------------------ */
/*  One LSH hash table: array of bucket heads                         */
/* ------------------------------------------------------------------ */
typedef struct {
    GV_DedupBucketNode **buckets;  /**< Array of bucket head pointers. */
    size_t num_buckets;            /**< Number of buckets (1 << hash_bits). */
} GV_DedupHashTable;

/* ------------------------------------------------------------------ */
/*  Internal (full) definition of the opaque GV_DedupIndex            */
/* ------------------------------------------------------------------ */
struct GV_DedupIndex {
    size_t dimension;              /**< Dimensionality of stored vectors. */
    GV_DedupConfig config;         /**< Copy of the caller's configuration. */

    /* Vector storage (flat, contiguous array) */
    float *vectors;                /**< Flat array: count * dimension floats. */
    size_t count;                  /**< Number of vectors currently stored. */
    size_t capacity;               /**< Number of vectors that can be stored without realloc. */

    /* LSH hyperplanes – single flat allocation:
     *   hyperplanes[t * hash_bits * dimension + b * dimension + d]
     * gives the d-th component of the b-th hyperplane for table t. */
    float *hyperplanes;

    /* Per-table hash structures */
    GV_DedupHashTable *tables;     /**< Array of num_hash_tables tables. */
};

/* ------------------------------------------------------------------ */
/*  PRNG helpers (xorshift64)                                         */
/* ------------------------------------------------------------------ */
static uint64_t dedup_xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float dedup_uniform_random(uint64_t *state) {
    return (float)dedup_xorshift64(state) / (float)UINT64_MAX;
}

static float dedup_gaussian_random(uint64_t *state) {
    float u1 = dedup_uniform_random(state);
    float u2 = dedup_uniform_random(state);
    if (u1 < 1e-9f) u1 = 1e-9f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

/* ------------------------------------------------------------------ */
/*  Hyperplane generation                                             */
/* ------------------------------------------------------------------ */
static void dedup_generate_hyperplanes(float *hyperplanes,
                                       size_t num_tables,
                                       size_t hash_bits,
                                       size_t dimension,
                                       uint64_t seed) {
    uint64_t state = seed;
    size_t total = num_tables * hash_bits * dimension;
    for (size_t i = 0; i < total; ++i) {
        hyperplanes[i] = dedup_gaussian_random(&state);
    }
}

/* ------------------------------------------------------------------ */
/*  Hash a single vector for one table                                */
/* ------------------------------------------------------------------ */
static uint32_t dedup_hash_vector(const float *data,
                                  size_t dimension,
                                  const float *hyperplanes,
                                  size_t table_idx,
                                  size_t hash_bits) {
    uint32_t hash = 0;
    size_t base = table_idx * hash_bits * dimension;

    for (size_t b = 0; b < hash_bits; ++b) {
        const float *plane = hyperplanes + base + b * dimension;
        float dot = 0.0f;
        for (size_t d = 0; d < dimension; ++d) {
            dot += data[d] * plane[d];
        }
        if (dot >= 0.0f) {
            hash |= (1U << b);
        }
    }
    return hash;
}

/* ------------------------------------------------------------------ */
/*  Bucket helpers                                                    */
/* ------------------------------------------------------------------ */
static int dedup_bucket_add(GV_DedupHashTable *table, uint32_t bucket_idx, size_t vec_index) {
    GV_DedupBucketNode *node = (GV_DedupBucketNode *)malloc(sizeof(GV_DedupBucketNode));
    if (node == NULL) {
        return -1;
    }
    node->index = vec_index;
    node->next = table->buckets[bucket_idx];
    table->buckets[bucket_idx] = node;
    return 0;
}

static void dedup_bucket_free_chain(GV_DedupBucketNode *head) {
    while (head != NULL) {
        GV_DedupBucketNode *next = head->next;
        free(head);
        head = next;
    }
}

/* ------------------------------------------------------------------ */
/*  L2 squared distance                                               */
/* ------------------------------------------------------------------ */
static float dedup_l2_distance_sq(const float *a, const float *b, size_t dimension) {
    float sum = 0.0f;
    for (size_t d = 0; d < dimension; ++d) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/*  Insert a vector index into all hash tables                        */
/* ------------------------------------------------------------------ */
static int dedup_insert_into_tables(GV_DedupIndex *dedup, size_t vec_index) {
    const float *data = dedup->vectors + vec_index * dedup->dimension;
    for (size_t t = 0; t < dedup->config.num_hash_tables; ++t) {
        uint32_t h = dedup_hash_vector(data, dedup->dimension, dedup->hyperplanes,
                                       t, dedup->config.hash_bits);
        uint32_t bucket_idx = h % dedup->tables[t].num_buckets;
        if (dedup_bucket_add(&dedup->tables[t], bucket_idx, vec_index) != 0) {
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

GV_DedupIndex *gv_dedup_create(size_t dimension, const GV_DedupConfig *config) {
    if (dimension == 0) {
        return NULL;
    }

    GV_DedupIndex *dedup = (GV_DedupIndex *)calloc(1, sizeof(GV_DedupIndex));
    if (dedup == NULL) {
        return NULL;
    }

    dedup->dimension = dimension;

    /* Apply defaults for missing config values */
    if (config != NULL) {
        dedup->config = *config;
    }
    if (dedup->config.epsilon <= 0.0f) {
        dedup->config.epsilon = 1e-5f;
    }
    if (dedup->config.num_hash_tables == 0) {
        dedup->config.num_hash_tables = 8;
    }
    if (dedup->config.hash_bits == 0) {
        dedup->config.hash_bits = 12;
    }
    if (dedup->config.seed == 0) {
        dedup->config.seed = 42;
    }

    /* Clamp hash_bits to avoid absurd bucket counts (max 24 -> 16M buckets) */
    if (dedup->config.hash_bits > 24) {
        dedup->config.hash_bits = 24;
    }

    size_t num_buckets = (size_t)1 << dedup->config.hash_bits;

    /* Allocate initial vector storage */
    dedup->capacity = 256;
    dedup->count = 0;
    dedup->vectors = (float *)malloc(dedup->capacity * dimension * sizeof(float));
    if (dedup->vectors == NULL) {
        free(dedup);
        return NULL;
    }

    /* Allocate hyperplanes (flat array) */
    size_t hp_count = dedup->config.num_hash_tables * dedup->config.hash_bits * dimension;
    dedup->hyperplanes = (float *)malloc(hp_count * sizeof(float));
    if (dedup->hyperplanes == NULL) {
        free(dedup->vectors);
        free(dedup);
        return NULL;
    }
    dedup_generate_hyperplanes(dedup->hyperplanes,
                               dedup->config.num_hash_tables,
                               dedup->config.hash_bits,
                               dimension,
                               dedup->config.seed);

    /* Allocate hash tables */
    dedup->tables = (GV_DedupHashTable *)calloc(dedup->config.num_hash_tables,
                                                 sizeof(GV_DedupHashTable));
    if (dedup->tables == NULL) {
        free(dedup->hyperplanes);
        free(dedup->vectors);
        free(dedup);
        return NULL;
    }

    for (size_t t = 0; t < dedup->config.num_hash_tables; ++t) {
        dedup->tables[t].num_buckets = num_buckets;
        dedup->tables[t].buckets = (GV_DedupBucketNode **)calloc(num_buckets,
                                                                   sizeof(GV_DedupBucketNode *));
        if (dedup->tables[t].buckets == NULL) {
            /* Clean up previously allocated tables */
            for (size_t i = 0; i < t; ++i) {
                free(dedup->tables[i].buckets);
            }
            free(dedup->tables);
            free(dedup->hyperplanes);
            free(dedup->vectors);
            free(dedup);
            return NULL;
        }
    }

    return dedup;
}

void gv_dedup_destroy(GV_DedupIndex *dedup) {
    if (dedup == NULL) {
        return;
    }

    if (dedup->tables != NULL) {
        for (size_t t = 0; t < dedup->config.num_hash_tables; ++t) {
            if (dedup->tables[t].buckets != NULL) {
                for (size_t b = 0; b < dedup->tables[t].num_buckets; ++b) {
                    dedup_bucket_free_chain(dedup->tables[t].buckets[b]);
                }
                free(dedup->tables[t].buckets);
            }
        }
        free(dedup->tables);
    }

    free(dedup->hyperplanes);
    free(dedup->vectors);
    free(dedup);
}

int gv_dedup_check(GV_DedupIndex *dedup, const float *data, size_t dimension) {
    if (dedup == NULL || data == NULL) {
        return -1;
    }
    if (dimension != dedup->dimension) {
        return -1;
    }
    if (dedup->count == 0) {
        return -1;
    }

    float eps_sq = dedup->config.epsilon * dedup->config.epsilon;

    /* Collect candidate set from all hash tables (de-duplicate with a seen array) */
    int *seen = (int *)calloc(dedup->count, sizeof(int));
    if (seen == NULL) {
        return -1;
    }

    for (size_t t = 0; t < dedup->config.num_hash_tables; ++t) {
        uint32_t h = dedup_hash_vector(data, dimension, dedup->hyperplanes,
                                       t, dedup->config.hash_bits);
        uint32_t bucket_idx = h % dedup->tables[t].num_buckets;

        GV_DedupBucketNode *node = dedup->tables[t].buckets[bucket_idx];
        while (node != NULL) {
            size_t idx = node->index;
            if (idx < dedup->count && !seen[idx]) {
                seen[idx] = 1;
                const float *existing = dedup->vectors + idx * dedup->dimension;
                float dist_sq = dedup_l2_distance_sq(data, existing, dimension);
                if (dist_sq <= eps_sq) {
                    free(seen);
                    return (int)idx;
                }
            }
            node = node->next;
        }
    }

    free(seen);
    return -1;
}

int gv_dedup_insert(GV_DedupIndex *dedup, const float *data, size_t dimension) {
    if (dedup == NULL || data == NULL) {
        return -1;
    }
    if (dimension != dedup->dimension) {
        return -1;
    }

    /* Check for an existing duplicate first */
    int dup = gv_dedup_check(dedup, data, dimension);
    if (dup >= 0) {
        return 1; /* duplicate found */
    }
    /* dup == -1 could mean "unique" or "error on empty index"; either way we insert. */

    /* Grow storage if necessary */
    if (dedup->count >= dedup->capacity) {
        size_t new_capacity = dedup->capacity * 2;
        float *new_vectors = (float *)realloc(dedup->vectors,
                                               new_capacity * dedup->dimension * sizeof(float));
        if (new_vectors == NULL) {
            return -1;
        }
        dedup->vectors = new_vectors;
        dedup->capacity = new_capacity;
    }

    /* Copy vector data into flat storage */
    size_t new_index = dedup->count;
    memcpy(dedup->vectors + new_index * dedup->dimension,
           data,
           dedup->dimension * sizeof(float));
    dedup->count++;

    /* Insert into all hash tables */
    if (dedup_insert_into_tables(dedup, new_index) != 0) {
        /* Roll back the vector append on hash-table failure */
        dedup->count--;
        return -1;
    }

    return 0; /* successfully inserted */
}

int gv_dedup_scan(GV_DedupIndex *dedup, GV_DedupResult *results, size_t max_results) {
    if (dedup == NULL || results == NULL || max_results == 0) {
        return -1;
    }
    if (dedup->count < 2) {
        return 0;
    }

    float eps_sq = dedup->config.epsilon * dedup->config.epsilon;
    size_t result_count = 0;

    /*
     * For each vector i we use LSH to find candidates j > i that are
     * within epsilon.  A global "pair seen" matrix would be expensive,
     * so we use a simple per-i seen array for de-duplicating candidates
     * across tables, and we only report pairs where j > i to avoid
     * reporting (i,j) and (j,i) both.
     */
    for (size_t i = 0; i < dedup->count && result_count < max_results; ++i) {
        const float *vec_i = dedup->vectors + i * dedup->dimension;

        /* Allocate a per-vector seen flags array */
        int *seen = (int *)calloc(dedup->count, sizeof(int));
        if (seen == NULL) {
            return (result_count > 0) ? (int)result_count : -1;
        }

        for (size_t t = 0; t < dedup->config.num_hash_tables; ++t) {
            uint32_t h = dedup_hash_vector(vec_i, dedup->dimension, dedup->hyperplanes,
                                           t, dedup->config.hash_bits);
            uint32_t bucket_idx = h % dedup->tables[t].num_buckets;

            GV_DedupBucketNode *node = dedup->tables[t].buckets[bucket_idx];
            while (node != NULL) {
                size_t j = node->index;
                if (j > i && j < dedup->count && !seen[j]) {
                    seen[j] = 1;
                    const float *vec_j = dedup->vectors + j * dedup->dimension;
                    float dist_sq = dedup_l2_distance_sq(vec_i, vec_j, dedup->dimension);
                    if (dist_sq <= eps_sq) {
                        results[result_count].original_index = i;
                        results[result_count].duplicate_index = j;
                        results[result_count].distance = sqrtf(dist_sq);
                        result_count++;
                        if (result_count >= max_results) {
                            free(seen);
                            return (int)result_count;
                        }
                    }
                }
                node = node->next;
            }
        }

        free(seen);
    }

    return (int)result_count;
}

size_t gv_dedup_count(const GV_DedupIndex *dedup) {
    if (dedup == NULL) {
        return 0;
    }
    return dedup->count;
}

void gv_dedup_clear(GV_DedupIndex *dedup) {
    if (dedup == NULL) {
        return;
    }

    /* Free all bucket chains */
    for (size_t t = 0; t < dedup->config.num_hash_tables; ++t) {
        for (size_t b = 0; b < dedup->tables[t].num_buckets; ++b) {
            dedup_bucket_free_chain(dedup->tables[t].buckets[b]);
            dedup->tables[t].buckets[b] = NULL;
        }
    }

    /* Reset vector count (keep the allocated buffer and hyperplanes) */
    dedup->count = 0;
}
