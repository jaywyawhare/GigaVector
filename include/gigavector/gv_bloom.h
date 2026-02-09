#ifndef GIGAVECTOR_GV_BLOOM_H
#define GIGAVECTOR_GV_BLOOM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque Bloom filter structure for skip-index membership testing.
 *
 * A Bloom filter is a space-efficient probabilistic data structure that
 * supports membership queries with no false negatives: if an item was added,
 * gv_bloom_check will always return 1. False positives are possible at a
 * configurable rate.
 */
typedef struct GV_BloomFilter GV_BloomFilter;

/**
 * @brief Create a new Bloom filter.
 *
 * Computes optimal bit-array size and hash count from the expected number
 * of items and the desired false-positive rate.
 *
 * @param expected_items  Expected number of distinct items to insert.
 * @param fp_rate         Target false-positive probability (e.g. 0.01 for 1%).
 * @return Pointer to a new GV_BloomFilter, or NULL on failure.
 */
GV_BloomFilter *gv_bloom_create(size_t expected_items, double fp_rate);

/**
 * @brief Destroy a Bloom filter and free all associated memory.
 *
 * @param bf  Bloom filter to destroy.  NULL is safely ignored.
 */
void gv_bloom_destroy(GV_BloomFilter *bf);

/**
 * @brief Add an arbitrary data item to the Bloom filter.
 *
 * @param bf   Bloom filter.
 * @param data Pointer to the item bytes.
 * @param len  Length of the item in bytes.
 * @return 0 on success, -1 on error (NULL arguments).
 */
int gv_bloom_add(GV_BloomFilter *bf, const void *data, size_t len);

/**
 * @brief Convenience wrapper to add a NUL-terminated string.
 *
 * @param bf   Bloom filter.
 * @param str  String to add (must not be NULL).
 * @return 0 on success, -1 on error.
 */
int gv_bloom_add_string(GV_BloomFilter *bf, const char *str);

/**
 * @brief Test whether an item may be present in the Bloom filter.
 *
 * @param bf   Bloom filter (const).
 * @param data Pointer to the item bytes.
 * @param len  Length of the item in bytes.
 * @return 1 if the item is possibly present, 0 if definitely absent,
 *         -1 on error (NULL arguments).
 */
int gv_bloom_check(const GV_BloomFilter *bf, const void *data, size_t len);

/**
 * @brief Convenience wrapper to check a NUL-terminated string.
 *
 * @param bf   Bloom filter (const).
 * @param str  String to check (must not be NULL).
 * @return 1 if possibly present, 0 if definitely absent, -1 on error.
 */
int gv_bloom_check_string(const GV_BloomFilter *bf, const char *str);

/**
 * @brief Return the number of items that have been added.
 *
 * @param bf  Bloom filter (const).
 * @return Item count, or 0 if bf is NULL.
 */
size_t gv_bloom_count(const GV_BloomFilter *bf);

/**
 * @brief Estimate the current false-positive rate.
 *
 * Uses the formula: (1 - e^(-k*n/m))^k
 *
 * @param bf  Bloom filter (const).
 * @return Estimated false-positive probability, or 0.0 if bf is NULL.
 */
double gv_bloom_fp_rate(const GV_BloomFilter *bf);

/**
 * @brief Reset the Bloom filter, clearing all bits and the item count.
 *
 * @param bf  Bloom filter.  NULL is safely ignored.
 */
void gv_bloom_clear(GV_BloomFilter *bf);

/**
 * @brief Serialize a Bloom filter to an open file.
 *
 * Format: num_bits | num_hashes | count | target_fp_rate | bit_array
 *
 * @param bf   Bloom filter (const).
 * @param out  Writable FILE pointer.
 * @return 0 on success, -1 on error.
 */
int gv_bloom_save(const GV_BloomFilter *bf, FILE *out);

/**
 * @brief Deserialize a Bloom filter from an open file.
 *
 * On success, *bf_ptr is set to a newly allocated filter that must be
 * freed with gv_bloom_destroy().
 *
 * @param bf_ptr  Output pointer for the loaded filter.
 * @param in      Readable FILE pointer.
 * @return 0 on success, -1 on error.
 */
int gv_bloom_load(GV_BloomFilter **bf_ptr, FILE *in);

/**
 * @brief Merge two Bloom filters by OR-ing their bit arrays.
 *
 * Both filters must have the same num_bits and num_hashes.  The returned
 * filter's count is the sum of the two input counts (an upper bound).
 *
 * @param a  First Bloom filter (const).
 * @param b  Second Bloom filter (const).
 * @return A new merged GV_BloomFilter, or NULL on error / incompatibility.
 */
GV_BloomFilter *gv_bloom_merge(const GV_BloomFilter *a, const GV_BloomFilter *b);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_BLOOM_H */
