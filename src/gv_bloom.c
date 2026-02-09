#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_bloom.h"

/* ------------------------------------------------------------------ */
/*  Internal Bloom filter structure                                    */
/* ------------------------------------------------------------------ */

struct GV_BloomFilter {
    uint8_t *bits;           /* Bit array (packed, 1 bit per position). */
    size_t   num_bits;       /* Total number of bits (m).               */
    size_t   num_hashes;     /* Number of hash functions (k).           */
    size_t   count;          /* Number of items inserted so far.        */
    double   target_fp_rate; /* Desired false-positive probability.     */
};

/* ------------------------------------------------------------------ */
/*  FNV-1a 64-bit hash                                                 */
/* ------------------------------------------------------------------ */

#define FNV_OFFSET_BASIS UINT64_C(14695981039346656037)
#define FNV_PRIME        UINT64_C(1099511628211)

/**
 * @brief FNV-1a 64-bit hash of an arbitrary byte buffer.
 */
static uint64_t fnv1a_64(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)p[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

/* ------------------------------------------------------------------ */
/*  Double-hashing helpers                                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Derive two independent 32-bit hashes from a single FNV-1a
 *        64-bit digest by splitting the upper and lower halves.
 */
static void bloom_hash_pair(const void *data, size_t len,
                            uint32_t *h1_out, uint32_t *h2_out)
{
    uint64_t h = fnv1a_64(data, len);
    *h1_out = (uint32_t)(h & 0xFFFFFFFF);
    *h2_out = (uint32_t)(h >> 32);
}

/**
 * @brief Compute the i-th hash position using double hashing:
 *        h_i(x) = (h1 + i * h2) mod m
 */
static size_t bloom_hash_i(uint32_t h1, uint32_t h2, size_t i, size_t m)
{
    return ((size_t)h1 + i * (size_t)h2) % m;
}

/* ------------------------------------------------------------------ */
/*  Bit manipulation helpers                                           */
/* ------------------------------------------------------------------ */

static void bit_set(uint8_t *bits, size_t pos)
{
    bits[pos / 8] |= (uint8_t)(1U << (pos % 8));
}

static int bit_get(const uint8_t *bits, size_t pos)
{
    return (bits[pos / 8] >> (pos % 8)) & 1;
}

/* ------------------------------------------------------------------ */
/*  Optimal sizing helpers                                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute optimal number of bits (m).
 *        m = -(n * ln(p)) / (ln(2)^2)
 */
static size_t bloom_optimal_bits(size_t n, double p)
{
    if (n == 0) {
        n = 1;
    }
    if (p <= 0.0) {
        p = 1e-15;
    }
    if (p >= 1.0) {
        p = 0.999;
    }
    double m = -((double)n * log(p)) / (log(2.0) * log(2.0));
    if (m < 8.0) {
        m = 8.0;
    }
    return (size_t)ceil(m);
}

/**
 * @brief Compute optimal number of hash functions (k).
 *        k = (m / n) * ln(2)
 */
static size_t bloom_optimal_hashes(size_t m, size_t n)
{
    if (n == 0) {
        n = 1;
    }
    double k = ((double)m / (double)n) * log(2.0);
    if (k < 1.0) {
        k = 1.0;
    }
    return (size_t)round(k);
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

GV_BloomFilter *gv_bloom_create(size_t expected_items, double fp_rate)
{
    GV_BloomFilter *bf = (GV_BloomFilter *)calloc(1, sizeof(GV_BloomFilter));
    if (bf == NULL) {
        return NULL;
    }

    bf->target_fp_rate = fp_rate;
    bf->num_bits       = bloom_optimal_bits(expected_items, fp_rate);
    bf->num_hashes     = bloom_optimal_hashes(bf->num_bits, expected_items);
    bf->count          = 0;

    /* Allocate the byte array (ceil(num_bits / 8)). */
    size_t byte_count = (bf->num_bits + 7) / 8;
    bf->bits = (uint8_t *)calloc(byte_count, 1);
    if (bf->bits == NULL) {
        free(bf);
        return NULL;
    }

    return bf;
}

void gv_bloom_destroy(GV_BloomFilter *bf)
{
    if (bf == NULL) {
        return;
    }
    free(bf->bits);
    free(bf);
}

int gv_bloom_add(GV_BloomFilter *bf, const void *data, size_t len)
{
    if (bf == NULL || data == NULL) {
        return -1;
    }

    uint32_t h1, h2;
    bloom_hash_pair(data, len, &h1, &h2);

    for (size_t i = 0; i < bf->num_hashes; i++) {
        size_t pos = bloom_hash_i(h1, h2, i, bf->num_bits);
        bit_set(bf->bits, pos);
    }

    bf->count++;
    return 0;
}

int gv_bloom_add_string(GV_BloomFilter *bf, const char *str)
{
    if (bf == NULL || str == NULL) {
        return -1;
    }
    return gv_bloom_add(bf, str, strlen(str));
}

int gv_bloom_check(const GV_BloomFilter *bf, const void *data, size_t len)
{
    if (bf == NULL || data == NULL) {
        return -1;
    }

    uint32_t h1, h2;
    bloom_hash_pair(data, len, &h1, &h2);

    for (size_t i = 0; i < bf->num_hashes; i++) {
        size_t pos = bloom_hash_i(h1, h2, i, bf->num_bits);
        if (!bit_get(bf->bits, pos)) {
            return 0; /* Definitely absent. */
        }
    }

    return 1; /* Possibly present. */
}

int gv_bloom_check_string(const GV_BloomFilter *bf, const char *str)
{
    if (bf == NULL || str == NULL) {
        return -1;
    }
    return gv_bloom_check(bf, str, strlen(str));
}

size_t gv_bloom_count(const GV_BloomFilter *bf)
{
    if (bf == NULL) {
        return 0;
    }
    return bf->count;
}

double gv_bloom_fp_rate(const GV_BloomFilter *bf)
{
    if (bf == NULL) {
        return 0.0;
    }
    if (bf->count == 0) {
        return 0.0;
    }
    /* (1 - e^(-k * n / m))^k */
    double k = (double)bf->num_hashes;
    double n = (double)bf->count;
    double m = (double)bf->num_bits;
    return pow(1.0 - exp(-k * n / m), k);
}

void gv_bloom_clear(GV_BloomFilter *bf)
{
    if (bf == NULL) {
        return;
    }
    size_t byte_count = (bf->num_bits + 7) / 8;
    memset(bf->bits, 0, byte_count);
    bf->count = 0;
}

/* ------------------------------------------------------------------ */
/*  Serialization helpers                                              */
/* ------------------------------------------------------------------ */

static int write_size(FILE *out, size_t v)
{
    return (fwrite(&v, sizeof(size_t), 1, out) == 1) ? 0 : -1;
}

static int write_double(FILE *out, double v)
{
    return (fwrite(&v, sizeof(double), 1, out) == 1) ? 0 : -1;
}

static int read_size(FILE *in, size_t *v)
{
    return (v != NULL && fread(v, sizeof(size_t), 1, in) == 1) ? 0 : -1;
}

static int read_double(FILE *in, double *v)
{
    return (v != NULL && fread(v, sizeof(double), 1, in) == 1) ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/*  Save / Load                                                        */
/* ------------------------------------------------------------------ */

int gv_bloom_save(const GV_BloomFilter *bf, FILE *out)
{
    if (bf == NULL || out == NULL) {
        return -1;
    }

    if (write_size(out, bf->num_bits) != 0) {
        return -1;
    }
    if (write_size(out, bf->num_hashes) != 0) {
        return -1;
    }
    if (write_size(out, bf->count) != 0) {
        return -1;
    }
    if (write_double(out, bf->target_fp_rate) != 0) {
        return -1;
    }

    size_t byte_count = (bf->num_bits + 7) / 8;
    if (fwrite(bf->bits, 1, byte_count, out) != byte_count) {
        return -1;
    }

    return 0;
}

int gv_bloom_load(GV_BloomFilter **bf_ptr, FILE *in)
{
    if (bf_ptr == NULL || in == NULL) {
        return -1;
    }

    *bf_ptr = NULL;

    size_t num_bits    = 0;
    size_t num_hashes  = 0;
    size_t count       = 0;
    double fp_rate     = 0.0;

    if (read_size(in, &num_bits) != 0) {
        return -1;
    }
    if (read_size(in, &num_hashes) != 0) {
        return -1;
    }
    if (read_size(in, &count) != 0) {
        return -1;
    }
    if (read_double(in, &fp_rate) != 0) {
        return -1;
    }

    if (num_bits == 0 || num_hashes == 0) {
        return -1;
    }

    GV_BloomFilter *bf = (GV_BloomFilter *)calloc(1, sizeof(GV_BloomFilter));
    if (bf == NULL) {
        return -1;
    }

    bf->num_bits       = num_bits;
    bf->num_hashes     = num_hashes;
    bf->count          = count;
    bf->target_fp_rate = fp_rate;

    size_t byte_count = (num_bits + 7) / 8;
    bf->bits = (uint8_t *)malloc(byte_count);
    if (bf->bits == NULL) {
        free(bf);
        return -1;
    }

    if (fread(bf->bits, 1, byte_count, in) != byte_count) {
        free(bf->bits);
        free(bf);
        return -1;
    }

    *bf_ptr = bf;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Merge                                                              */
/* ------------------------------------------------------------------ */

GV_BloomFilter *gv_bloom_merge(const GV_BloomFilter *a, const GV_BloomFilter *b)
{
    if (a == NULL || b == NULL) {
        return NULL;
    }
    if (a->num_bits != b->num_bits || a->num_hashes != b->num_hashes) {
        return NULL;
    }

    GV_BloomFilter *merged = (GV_BloomFilter *)calloc(1, sizeof(GV_BloomFilter));
    if (merged == NULL) {
        return NULL;
    }

    merged->num_bits       = a->num_bits;
    merged->num_hashes     = a->num_hashes;
    merged->count          = a->count + b->count;
    merged->target_fp_rate = a->target_fp_rate;

    size_t byte_count = (a->num_bits + 7) / 8;
    merged->bits = (uint8_t *)malloc(byte_count);
    if (merged->bits == NULL) {
        free(merged);
        return NULL;
    }

    for (size_t i = 0; i < byte_count; i++) {
        merged->bits[i] = a->bits[i] | b->bits[i];
    }

    return merged;
}
