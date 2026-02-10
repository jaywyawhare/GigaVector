/**
 * @file gv_muvera.c
 * @brief MUVERA encoding: converts variable-length multi-vector embeddings
 *        into fixed-size single vectors.
 *
 * Algorithm overview (MUVERA -- MUlti-Vector Encoding via Random Accumulation):
 *   1. Generate `num_projections` random sign vectors (+1/-1) of length
 *      token_dimension, seeded deterministically via xoshiro256**.
 *   2. Generate a random projection matrix to reduce token_dimension down to
 *      a reduced dimension (token_dimension / (num_buckets * some factor))
 *      so that the concatenated output stays within output_dimension.
 *   3. For each projection i, compute the dot product of the sign vector with
 *      each token.  Hash the token to bucket 0 (dot <= 0) or bucket 1 (dot > 0).
 *   4. Average the (reduced) tokens in each bucket.
 *   5. Concatenate all bucket averages across all projections.
 *   6. Optionally L2-normalize the output.
 *
 * Thread safety: the encoder is immutable after creation.  All encode
 * functions operate only on const state plus caller-owned buffers.
 */

#include "gigavector/gv_muvera.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define GV_MUVERA_MAGIC       "GV_MUVR"
#define GV_MUVERA_MAGIC_LEN   7
#define GV_MUVERA_VERSION     1
#define GV_MUVERA_NUM_BUCKETS 2   /* Binary hashing: bucket 0 and bucket 1. */

/* ============================================================================
 * xoshiro256** PRNG
 * ============================================================================ */

/**
 * @brief xoshiro256** state: 4 x uint64_t.
 */
typedef struct {
    uint64_t s[4];
} GV_Xoshiro256;

static inline uint64_t gv_xoshiro_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/**
 * @brief Seed the xoshiro256** generator from a single 64-bit seed using
 *        SplitMix64 to fill the state array.
 */
static void gv_xoshiro_seed(GV_Xoshiro256 *rng, uint64_t seed) {
    /* SplitMix64 to expand single seed into 4 state words. */
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        rng->s[i] = z;
    }
}

/**
 * @brief Generate the next uint64_t from xoshiro256**.
 */
static uint64_t gv_xoshiro_next(GV_Xoshiro256 *rng) {
    const uint64_t result = gv_xoshiro_rotl(rng->s[1] * 5, 7) * 9;
    const uint64_t t = rng->s[1] << 17;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;
    rng->s[3] = gv_xoshiro_rotl(rng->s[3], 45);

    return result;
}

/**
 * @brief Generate a random sign: +1.0f or -1.0f.
 */
static float gv_xoshiro_random_sign(GV_Xoshiro256 *rng) {
    return (gv_xoshiro_next(rng) & 1) ? 1.0f : -1.0f;
}

/**
 * @brief Generate a uniform float in (-1, 1) for random projection matrices.
 */
static float gv_xoshiro_uniform(GV_Xoshiro256 *rng) {
    uint64_t v = gv_xoshiro_next(rng);
    /* Map to (-1.0, 1.0). */
    return ((float)(v >> 40) / (float)(1ULL << 24)) - 1.0f;
}

/* ============================================================================
 * Internal Encoder Structure
 * ============================================================================ */

struct GV_MuveraEncoder {
    GV_MuveraConfig config;

    size_t reduced_dim;   /**< Reduced per-token dimension after random projection. */

    /**
     * Sign vectors for hashing: num_projections x token_dimension floats.
     * Each element is +1.0f or -1.0f.
     * Layout: sign_vectors[i * token_dimension + d]
     */
    float *sign_vectors;

    /**
     * Random projection matrix to reduce token_dimension -> reduced_dim.
     * Layout: proj_matrix[r * token_dimension + d]
     * where r in [0, reduced_dim), d in [0, token_dimension).
     * Scaled by 1/sqrt(reduced_dim) for approximate norm preservation.
     */
    float *proj_matrix;
};

/* ============================================================================
 * Dot-product helper (scalar)
 * ============================================================================ */

static float gv_muvera_dot(const float *a, const float *b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* ============================================================================
 * Internal: project a token from token_dimension -> reduced_dim
 * ============================================================================ */

static void gv_muvera_project_token(const GV_MuveraEncoder *enc,
                                    const float *token,
                                    float *reduced) {
    size_t td = enc->config.token_dimension;
    size_t rd = enc->reduced_dim;

    for (size_t r = 0; r < rd; r++) {
        reduced[r] = gv_muvera_dot(enc->proj_matrix + r * td, token, td);
    }
}

/* ============================================================================
 * Internal: encode a single token set
 * ============================================================================ */

static int gv_muvera_encode_single(const GV_MuveraEncoder *enc,
                                   const float *tokens, size_t num_tokens,
                                   float *output) {
    size_t td  = enc->config.token_dimension;
    size_t np  = enc->config.num_projections;
    size_t rd  = enc->reduced_dim;
    size_t od  = enc->config.output_dimension;

    /* Zero the output. */
    memset(output, 0, od * sizeof(float));

    if (num_tokens == 0) {
        return 0;
    }

    /* Pre-project all tokens into reduced space.
     * reduced_tokens: num_tokens x reduced_dim */
    float *reduced_tokens = (float *)malloc(num_tokens * rd * sizeof(float));
    if (!reduced_tokens) return -1;

    for (size_t t = 0; t < num_tokens; t++) {
        gv_muvera_project_token(enc, tokens + t * td, reduced_tokens + t * rd);
    }

    /* For each projection, hash tokens into buckets, accumulate, and average. */
    /* Output layout: for projection i, bucket b:
     *   output[(i * NUM_BUCKETS + b) * reduced_dim ... + reduced_dim - 1]
     */
    size_t *bucket_counts = (size_t *)calloc(np * GV_MUVERA_NUM_BUCKETS, sizeof(size_t));
    if (!bucket_counts) {
        free(reduced_tokens);
        return -1;
    }

    for (size_t i = 0; i < np; i++) {
        const float *sign_vec = enc->sign_vectors + i * td;

        for (size_t t = 0; t < num_tokens; t++) {
            /* Hash: dot(sign_vector, token) > 0 => bucket 1, else bucket 0. */
            float dp = gv_muvera_dot(sign_vec, tokens + t * td, td);
            size_t bucket = (dp > 0.0f) ? 1 : 0;

            /* Accumulate reduced token into the appropriate output slot. */
            size_t out_offset = (i * GV_MUVERA_NUM_BUCKETS + bucket) * rd;
            const float *rtok = reduced_tokens + t * rd;
            for (size_t d = 0; d < rd; d++) {
                output[out_offset + d] += rtok[d];
            }
            bucket_counts[i * GV_MUVERA_NUM_BUCKETS + bucket]++;
        }
    }

    /* Divide by bucket counts to get means. */
    for (size_t i = 0; i < np; i++) {
        for (size_t b = 0; b < GV_MUVERA_NUM_BUCKETS; b++) {
            size_t cnt = bucket_counts[i * GV_MUVERA_NUM_BUCKETS + b];
            if (cnt > 0) {
                size_t out_offset = (i * GV_MUVERA_NUM_BUCKETS + b) * rd;
                float inv = 1.0f / (float)cnt;
                for (size_t d = 0; d < rd; d++) {
                    output[out_offset + d] *= inv;
                }
            }
        }
    }

    free(bucket_counts);
    free(reduced_tokens);

    /* Optionally L2-normalize the output. */
    if (enc->config.normalize) {
        float norm_sq = 0.0f;
        for (size_t d = 0; d < od; d++) {
            norm_sq += output[d] * output[d];
        }
        if (norm_sq > 0.0f) {
            float inv_norm = 1.0f / sqrtf(norm_sq);
            for (size_t d = 0; d < od; d++) {
                output[d] *= inv_norm;
            }
        }
    }

    return 0;
}

/* ============================================================================
 * Serialization helpers
 * ============================================================================ */

static int gv_muvera_write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int gv_muvera_read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int gv_muvera_write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, f) == 1 ? 0 : -1;
}

static int gv_muvera_read_u64(FILE *f, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

static const GV_MuveraConfig DEFAULT_CONFIG = {
    .token_dimension  = 128,
    .num_projections  = 64,
    .output_dimension = 0,    /* 0 = auto-compute */
    .seed             = 42,
    .normalize        = 1
};

void gv_muvera_config_init(GV_MuveraConfig *config) {
    if (!config) return;
    *config = DEFAULT_CONFIG;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

GV_MuveraEncoder *gv_muvera_create(const GV_MuveraConfig *config) {
    GV_MuveraConfig cfg = config ? *config : DEFAULT_CONFIG;

    /* Apply defaults for zero fields. */
    if (cfg.token_dimension == 0) cfg.token_dimension = 128;
    if (cfg.num_projections == 0) cfg.num_projections = 64;
    if (cfg.seed == 0)            cfg.seed = 42;

    /* Compute reduced dimension.
     * output_dimension = num_projections * NUM_BUCKETS * reduced_dim
     * If output_dimension is specified, derive reduced_dim from it.
     * Otherwise, use the default: output_dimension = num_projections * token_dimension / 4,
     * which gives reduced_dim = token_dimension / (4 * NUM_BUCKETS / 1)
     *   = token_dimension / (4 * 2 / (num_projections / num_projections))
     * Actually: output_dim = np * 2 * rd  =>  rd = output_dim / (np * 2)
     */
    size_t np = cfg.num_projections;
    size_t td = cfg.token_dimension;

    if (cfg.output_dimension == 0) {
        cfg.output_dimension = np * td / 4;
    }

    size_t od = cfg.output_dimension;
    size_t reduced_dim = od / (np * GV_MUVERA_NUM_BUCKETS);
    if (reduced_dim == 0) reduced_dim = 1;

    /* Adjust output_dimension to be exactly representable. */
    cfg.output_dimension = np * GV_MUVERA_NUM_BUCKETS * reduced_dim;

    GV_MuveraEncoder *enc = (GV_MuveraEncoder *)calloc(1, sizeof(GV_MuveraEncoder));
    if (!enc) return NULL;

    enc->config      = cfg;
    enc->reduced_dim = reduced_dim;

    /* Allocate sign vectors: num_projections x token_dimension. */
    enc->sign_vectors = (float *)malloc(np * td * sizeof(float));
    if (!enc->sign_vectors) {
        free(enc);
        return NULL;
    }

    /* Allocate projection matrix: reduced_dim x token_dimension. */
    enc->proj_matrix = (float *)malloc(reduced_dim * td * sizeof(float));
    if (!enc->proj_matrix) {
        free(enc->sign_vectors);
        free(enc);
        return NULL;
    }

    /* Initialize PRNG and generate random data. */
    GV_Xoshiro256 rng;
    gv_xoshiro_seed(&rng, cfg.seed);

    /* Generate sign vectors (+1/-1). */
    for (size_t i = 0; i < np * td; i++) {
        enc->sign_vectors[i] = gv_xoshiro_random_sign(&rng);
    }

    /* Generate random projection matrix, scaled by 1/sqrt(reduced_dim). */
    float scale = 1.0f / sqrtf((float)reduced_dim);
    for (size_t i = 0; i < reduced_dim * td; i++) {
        enc->proj_matrix[i] = gv_xoshiro_uniform(&rng) * scale;
    }

    return enc;
}

void gv_muvera_destroy(GV_MuveraEncoder *enc) {
    if (!enc) return;

    free(enc->sign_vectors);
    free(enc->proj_matrix);
    free(enc);
}

/* ============================================================================
 * Encode
 * ============================================================================ */

int gv_muvera_encode(const GV_MuveraEncoder *enc,
                     const float *tokens, size_t num_tokens,
                     float *output) {
    if (!enc || !output) return -1;
    if (num_tokens > 0 && !tokens) return -1;

    return gv_muvera_encode_single(enc, tokens, num_tokens, output);
}

size_t gv_muvera_output_dimension(const GV_MuveraEncoder *enc) {
    if (!enc) return 0;
    return enc->config.output_dimension;
}

int gv_muvera_encode_batch(const GV_MuveraEncoder *enc,
                           const float **token_sets,
                           const size_t *token_counts,
                           size_t batch_size,
                           float *outputs) {
    if (!enc || !token_sets || !token_counts || !outputs) return -1;
    if (batch_size == 0) return 0;

    size_t od = enc->config.output_dimension;

    for (size_t i = 0; i < batch_size; i++) {
        if (token_counts[i] > 0 && !token_sets[i]) return -1;

        int rc = gv_muvera_encode_single(enc, token_sets[i], token_counts[i],
                                         outputs + i * od);
        if (rc != 0) return -1;
    }

    return 0;
}

/* ============================================================================
 * Save
 * ============================================================================ */

int gv_muvera_save(const GV_MuveraEncoder *enc, const char *path) {
    if (!enc || !path) return -1;

    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    /* Magic + version. */
    if (fwrite(GV_MUVERA_MAGIC, 1, GV_MUVERA_MAGIC_LEN, fp) != GV_MUVERA_MAGIC_LEN) goto fail;
    if (gv_muvera_write_u32(fp, GV_MUVERA_VERSION) != 0) goto fail;

    /* Configuration. */
    if (gv_muvera_write_u64(fp, (uint64_t)enc->config.token_dimension) != 0) goto fail;
    if (gv_muvera_write_u64(fp, (uint64_t)enc->config.num_projections) != 0) goto fail;
    if (gv_muvera_write_u64(fp, (uint64_t)enc->config.output_dimension) != 0) goto fail;
    if (gv_muvera_write_u64(fp, enc->config.seed) != 0) goto fail;
    if (gv_muvera_write_u32(fp, (uint32_t)enc->config.normalize) != 0) goto fail;

    /* Reduced dimension. */
    if (gv_muvera_write_u64(fp, (uint64_t)enc->reduced_dim) != 0) goto fail;

    /* Sign vectors. */
    {
        size_t count = enc->config.num_projections * enc->config.token_dimension;
        if (fwrite(enc->sign_vectors, sizeof(float), count, fp) != count) goto fail;
    }

    /* Projection matrix. */
    {
        size_t count = enc->reduced_dim * enc->config.token_dimension;
        if (fwrite(enc->proj_matrix, sizeof(float), count, fp) != count) goto fail;
    }

    fclose(fp);
    return 0;

fail:
    fclose(fp);
    return -1;
}

/* ============================================================================
 * Load
 * ============================================================================ */

GV_MuveraEncoder *gv_muvera_load(const char *path) {
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    /* Verify magic. */
    char magic[GV_MUVERA_MAGIC_LEN];
    if (fread(magic, 1, GV_MUVERA_MAGIC_LEN, fp) != GV_MUVERA_MAGIC_LEN ||
        memcmp(magic, GV_MUVERA_MAGIC, GV_MUVERA_MAGIC_LEN) != 0) {
        fclose(fp);
        return NULL;
    }

    /* Verify version. */
    uint32_t version = 0;
    if (gv_muvera_read_u32(fp, &version) != 0 || version != GV_MUVERA_VERSION) {
        fclose(fp);
        return NULL;
    }

    /* Read configuration. */
    uint64_t td = 0, np = 0, od = 0, seed = 0;
    uint32_t norm = 0;

    if (gv_muvera_read_u64(fp, &td) != 0)   { fclose(fp); return NULL; }
    if (gv_muvera_read_u64(fp, &np) != 0)    { fclose(fp); return NULL; }
    if (gv_muvera_read_u64(fp, &od) != 0)    { fclose(fp); return NULL; }
    if (gv_muvera_read_u64(fp, &seed) != 0)  { fclose(fp); return NULL; }
    if (gv_muvera_read_u32(fp, &norm) != 0)  { fclose(fp); return NULL; }

    /* Read reduced dimension. */
    uint64_t rd = 0;
    if (gv_muvera_read_u64(fp, &rd) != 0) { fclose(fp); return NULL; }

    /* Sanity checks. */
    if (td == 0 || np == 0 || od == 0 || rd == 0) {
        fclose(fp);
        return NULL;
    }

    /* Allocate encoder. */
    GV_MuveraEncoder *enc = (GV_MuveraEncoder *)calloc(1, sizeof(GV_MuveraEncoder));
    if (!enc) { fclose(fp); return NULL; }

    enc->config.token_dimension  = (size_t)td;
    enc->config.num_projections  = (size_t)np;
    enc->config.output_dimension = (size_t)od;
    enc->config.seed             = seed;
    enc->config.normalize        = (int)norm;
    enc->reduced_dim             = (size_t)rd;

    /* Allocate and read sign vectors. */
    {
        size_t count = (size_t)(np * td);
        enc->sign_vectors = (float *)malloc(count * sizeof(float));
        if (!enc->sign_vectors) {
            free(enc);
            fclose(fp);
            return NULL;
        }
        if (fread(enc->sign_vectors, sizeof(float), count, fp) != count) {
            free(enc->sign_vectors);
            free(enc);
            fclose(fp);
            return NULL;
        }
    }

    /* Allocate and read projection matrix. */
    {
        size_t count = (size_t)(rd * td);
        enc->proj_matrix = (float *)malloc(count * sizeof(float));
        if (!enc->proj_matrix) {
            free(enc->sign_vectors);
            free(enc);
            fclose(fp);
            return NULL;
        }
        if (fread(enc->proj_matrix, sizeof(float), count, fp) != count) {
            free(enc->proj_matrix);
            free(enc->sign_vectors);
            free(enc);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return enc;
}
