#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "gigavector/gv_quantization.h"

#ifdef __POPCNT__
#include <popcntintrin.h>
#endif

/* ------------------------------------------------------------------ */
/*  File format constants                                              */
/* ------------------------------------------------------------------ */
#define GV_QUANT_MAGIC_0 'G'
#define GV_QUANT_MAGIC_1 'V'
#define GV_QUANT_MAGIC_2 'Q'
#define GV_QUANT_MAGIC_3 'T'
#define GV_QUANT_FILE_VERSION 1

/* ------------------------------------------------------------------ */
/*  Internal codebook structure                                        */
/* ------------------------------------------------------------------ */
struct GV_QuantCodebook {
    GV_QuantType type;
    GV_QuantMode mode;
    size_t       dimension;

    /* Per-dimension statistics used for scalar quantization.
     * Asymmetric mode: min_vals / max_vals  (range mapping).
     * Symmetric mode:  mean_vals / std_vals (zero-centred mapping).       */
    float *min_vals;
    float *max_vals;
    float *mean_vals;
    float *std_vals;

    /* Ternary threshold (fraction of standard deviation). */
    float ternary_threshold;

    /* RaBitQ rotation matrix (dimension * dimension floats), stored
     * row-major.  NULL when use_rabitq == 0. */
    float *rotation;
    int    use_rabitq;
    uint64_t rabitq_seed;
};

/* ------------------------------------------------------------------ */
/*  Small I/O helpers (matching gv_codebook.c style)                   */
/* ------------------------------------------------------------------ */
static int write_u8(FILE *f, uint8_t v) {
    return fwrite(&v, sizeof(uint8_t), 1, f) == 1 ? 0 : -1;
}

static int read_u8(FILE *f, uint8_t *v) {
    return (v && fread(v, sizeof(uint8_t), 1, f) == 1) ? 0 : -1;
}

static int write_u32(FILE *f, uint32_t v) {
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int read_u32(FILE *f, uint32_t *v) {
    return (v && fread(v, sizeof(uint32_t), 1, f) == 1) ? 0 : -1;
}

static int write_u64(FILE *f, uint64_t v) {
    return fwrite(&v, sizeof(uint64_t), 1, f) == 1 ? 0 : -1;
}

static int read_u64(FILE *f, uint64_t *v) {
    return (v && fread(v, sizeof(uint64_t), 1, f) == 1) ? 0 : -1;
}

static int write_f32(FILE *f, float v) {
    return fwrite(&v, sizeof(float), 1, f) == 1 ? 0 : -1;
}

static int read_f32(FILE *f, float *v) {
    return (v && fread(v, sizeof(float), 1, f) == 1) ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/*  Popcount helper                                                    */
/* ------------------------------------------------------------------ */
static size_t gv_popcount64(uint64_t x) {
#ifdef __POPCNT__
    return (size_t)_mm_popcnt_u64(x);
#else
    size_t count = 0;
    while (x) {
        count += (x & 1);
        x >>= 1;
    }
    return count;
#endif
}

/* ------------------------------------------------------------------ */
/*  Simple xorshift64 PRNG (deterministic, no global state)            */
/* ------------------------------------------------------------------ */
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* ------------------------------------------------------------------ */
/*  RaBitQ: generate a random orthogonal matrix via Householder        */
/*  reflections.  Produces a dimension x dimension rotation matrix.    */
/* ------------------------------------------------------------------ */
static float *rabitq_generate_rotation(size_t dimension, uint64_t seed) {
    float *R = (float *)malloc(dimension * dimension * sizeof(float));
    if (!R) return NULL;

    /* Start with identity. */
    memset(R, 0, dimension * dimension * sizeof(float));
    for (size_t i = 0; i < dimension; i++) {
        R[i * dimension + i] = 1.0f;
    }

    uint64_t rng = seed ? seed : 0x123456789ABCULL;

    float *v = (float *)malloc(dimension * sizeof(float));
    float *tmp_row = (float *)malloc(dimension * sizeof(float));
    if (!v || !tmp_row) {
        free(v);
        free(tmp_row);
        free(R);
        return NULL;
    }

    /* Apply (dimension - 1) random Householder reflections.
     * Each reflection H_k = I - 2 v v^T is applied to the current matrix
     * from the left:  R <- H_k R.  The resulting product is a uniformly
     * random orthogonal matrix (Stewart's algorithm). */
    for (size_t k = 0; k < dimension - 1; k++) {
        /* Generate a random vector of length (dimension - k) and normalise. */
        float norm_sq = 0.0f;
        for (size_t i = k; i < dimension; i++) {
            /* Map PRNG output to a float in roughly [-1, 1]. */
            uint64_t r = xorshift64(&rng);
            float f = (float)((int64_t)(r & 0x7FFFFFFFULL) - 0x3FFFFFFFLL)
                      / (float)0x3FFFFFFFLL;
            v[i] = f;
            norm_sq += f * f;
        }
        for (size_t i = 0; i < k; i++) {
            v[i] = 0.0f;
        }

        if (norm_sq < 1e-12f) continue;

        float inv_norm = 1.0f / sqrtf(norm_sq);
        for (size_t i = k; i < dimension; i++) {
            v[i] *= inv_norm;
        }

        /* Apply H = I - 2 v v^T to R from the left.
         * For each column j of R:  R[:,j] -= 2 v (v^T R[:,j])  */
        for (size_t j = 0; j < dimension; j++) {
            float dot = 0.0f;
            for (size_t i = k; i < dimension; i++) {
                dot += v[i] * R[i * dimension + j];
            }
            dot *= 2.0f;
            for (size_t i = k; i < dimension; i++) {
                R[i * dimension + j] -= dot * v[i];
            }
        }
    }

    free(v);
    free(tmp_row);
    return R;
}

/* ------------------------------------------------------------------ */
/*  Helper: apply rotation matrix to a vector (out = R * in).          */
/* ------------------------------------------------------------------ */
static void apply_rotation(const float *R, const float *in, float *out,
                           size_t dimension) {
    for (size_t i = 0; i < dimension; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < dimension; j++) {
            sum += R[i * dimension + j] * in[j];
        }
        out[i] = sum;
    }
}

/* ------------------------------------------------------------------ */
/*  Helper: apply transpose of rotation matrix (out = R^T * in).       */
/* ------------------------------------------------------------------ */
static void apply_rotation_transpose(const float *R, const float *in,
                                     float *out, size_t dimension) {
    for (size_t i = 0; i < dimension; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < dimension; j++) {
            sum += R[j * dimension + i] * in[j];
        }
        out[i] = sum;
    }
}

/* ------------------------------------------------------------------ */
/*  Bits-per-value for each quantization type                          */
/* ------------------------------------------------------------------ */
static size_t bits_per_value(GV_QuantType type) {
    switch (type) {
        case GV_QUANT_BINARY:  return 1;
        case GV_QUANT_TERNARY: return 2; /* stored as 2 bits */
        case GV_QUANT_2BIT:    return 2;
        case GV_QUANT_4BIT:    return 4;
        case GV_QUANT_8BIT:    return 8;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Number of quantization levels for scalar types                     */
/* ------------------------------------------------------------------ */
static size_t quant_levels(GV_QuantType type) {
    switch (type) {
        case GV_QUANT_BINARY:  return 2;
        case GV_QUANT_TERNARY: return 3;
        case GV_QUANT_2BIT:    return 4;
        case GV_QUANT_4BIT:    return 16;
        case GV_QUANT_8BIT:    return 256;
    }
    return 0;
}

/* ================================================================== */
/*  Public API                                                         */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/*  Config init                                                        */
/* ------------------------------------------------------------------ */

void gv_quant_config_init(GV_QuantConfig *config) {
    if (!config) return;
    config->type        = GV_QUANT_8BIT;
    config->mode        = GV_QUANT_SYMMETRIC;
    config->use_rabitq  = 0;
    config->rabitq_seed = 0;
}

/* ------------------------------------------------------------------ */
/*  Code size                                                          */
/* ------------------------------------------------------------------ */

size_t gv_quant_code_size(const GV_QuantCodebook *cb, size_t dimension) {
    if (!cb || dimension == 0) return 0;
    size_t bpv = bits_per_value(cb->type);
    if (bpv == 0) return 0;
    return (dimension * bpv + 7) / 8;
}

/* ------------------------------------------------------------------ */
/*  Memory ratio                                                       */
/* ------------------------------------------------------------------ */

float gv_quant_memory_ratio(const GV_QuantCodebook *cb, size_t dimension) {
    if (!cb || dimension == 0) return 0.0f;
    size_t code_bytes = gv_quant_code_size(cb, dimension);
    if (code_bytes == 0) return 0.0f;
    return (float)(dimension * sizeof(float)) / (float)code_bytes;
}

/* ------------------------------------------------------------------ */
/*  Training                                                           */
/* ------------------------------------------------------------------ */

GV_QuantCodebook *gv_quant_train(const float *vectors, size_t count,
                                 size_t dimension,
                                 const GV_QuantConfig *config) {
    if (!vectors || count == 0 || dimension == 0 || !config) return NULL;

    GV_QuantCodebook *cb = (GV_QuantCodebook *)calloc(1, sizeof(GV_QuantCodebook));
    if (!cb) return NULL;

    cb->type      = config->type;
    cb->mode      = config->mode;
    cb->dimension = dimension;
    cb->ternary_threshold = 0.5f; /* default: half a std-dev */

    /* ---- Per-dimension statistics -------------------------------- */
    cb->min_vals  = (float *)malloc(dimension * sizeof(float));
    cb->max_vals  = (float *)malloc(dimension * sizeof(float));
    cb->mean_vals = (float *)malloc(dimension * sizeof(float));
    cb->std_vals  = (float *)malloc(dimension * sizeof(float));
    if (!cb->min_vals || !cb->max_vals || !cb->mean_vals || !cb->std_vals) {
        gv_quant_codebook_destroy(cb);
        return NULL;
    }

    /* Initialise accumulators. */
    for (size_t d = 0; d < dimension; d++) {
        cb->min_vals[d]  = FLT_MAX;
        cb->max_vals[d]  = -FLT_MAX;
        cb->mean_vals[d] = 0.0f;
    }

    /* First pass: min, max, mean. */
    for (size_t i = 0; i < count; i++) {
        const float *v = &vectors[i * dimension];
        for (size_t d = 0; d < dimension; d++) {
            float val = v[d];
            if (val < cb->min_vals[d]) cb->min_vals[d] = val;
            if (val > cb->max_vals[d]) cb->max_vals[d] = val;
            cb->mean_vals[d] += val;
        }
    }

    float inv_count = 1.0f / (float)count;
    for (size_t d = 0; d < dimension; d++) {
        cb->mean_vals[d] *= inv_count;
    }

    /* Second pass: standard deviation. */
    for (size_t d = 0; d < dimension; d++) {
        cb->std_vals[d] = 0.0f;
    }
    for (size_t i = 0; i < count; i++) {
        const float *v = &vectors[i * dimension];
        for (size_t d = 0; d < dimension; d++) {
            float diff = v[d] - cb->mean_vals[d];
            cb->std_vals[d] += diff * diff;
        }
    }
    for (size_t d = 0; d < dimension; d++) {
        cb->std_vals[d] = sqrtf(cb->std_vals[d] * inv_count);
        if (cb->std_vals[d] < 1e-9f) {
            cb->std_vals[d] = 1e-9f;
        }
    }

    /* ---- RaBitQ rotation (binary mode only) ---------------------- */
    cb->use_rabitq  = 0;
    cb->rabitq_seed = 0;
    cb->rotation    = NULL;

    if (config->type == GV_QUANT_BINARY && config->use_rabitq) {
        cb->use_rabitq  = 1;
        cb->rabitq_seed = config->rabitq_seed;
        cb->rotation = rabitq_generate_rotation(dimension, config->rabitq_seed);
        if (!cb->rotation) {
            gv_quant_codebook_destroy(cb);
            return NULL;
        }
    }

    return cb;
}

/* ------------------------------------------------------------------ */
/*  Encoding helpers                                                   */
/* ------------------------------------------------------------------ */

/**
 * Quantise a single scalar value to an integer code [0, levels-1]
 * using the per-dimension min/max (asymmetric) or mean/std (symmetric).
 */
static uint8_t scalar_quantize_value(float val, float lo, float hi,
                                     size_t levels) {
    if (hi <= lo) return 0;
    float norm = (val - lo) / (hi - lo);
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;
    size_t q = (size_t)(norm * (float)(levels - 1) + 0.5f);
    if (q >= levels) q = levels - 1;
    return (uint8_t)q;
}

/**
 * Dequantise an integer code back to a float value.
 */
static float scalar_dequantize_value(uint8_t code, float lo, float hi,
                                     size_t levels) {
    if (levels <= 1) return lo;
    return lo + (float)code / (float)(levels - 1) * (hi - lo);
}

/**
 * Get effective lo/hi for a given dimension from the codebook.
 */
static void get_quant_range(const GV_QuantCodebook *cb, size_t d,
                            float *lo, float *hi) {
    if (cb->mode == GV_QUANT_ASYMMETRIC) {
        *lo = cb->min_vals[d];
        *hi = cb->max_vals[d];
    } else {
        /* Symmetric: centre on mean, span +/- 3 standard deviations. */
        float half = 3.0f * cb->std_vals[d];
        *lo = cb->mean_vals[d] - half;
        *hi = cb->mean_vals[d] + half;
    }
}

/* ------------------------------------------------------------------ */
/*  Encode                                                             */
/* ------------------------------------------------------------------ */

int gv_quant_encode(const GV_QuantCodebook *cb, const float *vector,
                    size_t dimension, uint8_t *codes) {
    if (!cb || !vector || !codes) return -1;
    if (dimension != cb->dimension) return -1;

    size_t code_bytes = gv_quant_code_size(cb, dimension);
    memset(codes, 0, code_bytes);

    /* Optional scratch buffer for rotated vector (RaBitQ). */
    float *rotated = NULL;
    const float *src = vector;

    if (cb->use_rabitq && cb->rotation) {
        rotated = (float *)malloc(dimension * sizeof(float));
        if (!rotated) return -1;
        apply_rotation(cb->rotation, vector, rotated, dimension);
        src = rotated;
    }

    switch (cb->type) {

    case GV_QUANT_BINARY: {
        /* 1-bit: sign of each component, packed MSB-first. */
        for (size_t d = 0; d < dimension; d++) {
            if (src[d] >= 0.0f) {
                codes[d / 8] |= (uint8_t)(1U << (7 - (d % 8)));
            }
        }
        break;
    }

    case GV_QUANT_TERNARY: {
        /* 1.5-bit stored as 2 bits per value.
         * Encoding:  00 = -1,  01 = 0,  10 = +1.
         * Values within [-threshold*std, +threshold*std] map to 0. */
        for (size_t d = 0; d < dimension; d++) {
            float thresh = cb->ternary_threshold * cb->std_vals[d];
            uint8_t code;
            if (vector[d] > thresh) {
                code = 2; /* +1 */
            } else if (vector[d] < -thresh) {
                code = 0; /* -1 */
            } else {
                code = 1; /*  0 */
            }
            size_t bit_pos = d * 2;
            size_t byte_idx = bit_pos / 8;
            size_t shift = 6 - (bit_pos % 8);
            codes[byte_idx] |= (uint8_t)(code << shift);
        }
        break;
    }

    case GV_QUANT_2BIT: {
        /* 4 levels, 4 values per byte, packed MSB-first (2 bits each). */
        size_t levels = quant_levels(GV_QUANT_2BIT);
        for (size_t d = 0; d < dimension; d++) {
            float lo, hi;
            get_quant_range(cb, d, &lo, &hi);
            uint8_t q = scalar_quantize_value(vector[d], lo, hi, levels);
            size_t bit_pos = d * 2;
            size_t byte_idx = bit_pos / 8;
            size_t shift = 6 - (bit_pos % 8);
            codes[byte_idx] |= (uint8_t)(q << shift);
        }
        break;
    }

    case GV_QUANT_4BIT: {
        /* 16 levels, 2 values per byte, high nibble first. */
        size_t levels = quant_levels(GV_QUANT_4BIT);
        for (size_t d = 0; d < dimension; d++) {
            float lo, hi;
            get_quant_range(cb, d, &lo, &hi);
            uint8_t q = scalar_quantize_value(vector[d], lo, hi, levels);
            size_t byte_idx = d / 2;
            if (d % 2 == 0) {
                codes[byte_idx] |= (uint8_t)(q << 4);
            } else {
                codes[byte_idx] |= q;
            }
        }
        break;
    }

    case GV_QUANT_8BIT: {
        /* 256 levels, one byte per dimension. */
        size_t levels = quant_levels(GV_QUANT_8BIT);
        for (size_t d = 0; d < dimension; d++) {
            float lo, hi;
            get_quant_range(cb, d, &lo, &hi);
            codes[d] = scalar_quantize_value(vector[d], lo, hi, levels);
        }
        break;
    }
    }

    free(rotated);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Decode                                                             */
/* ------------------------------------------------------------------ */

int gv_quant_decode(const GV_QuantCodebook *cb, const uint8_t *codes,
                    size_t dimension, float *output) {
    if (!cb || !codes || !output) return -1;
    if (dimension != cb->dimension) return -1;

    switch (cb->type) {

    case GV_QUANT_BINARY: {
        /* Decode to +1 / -1. */
        for (size_t d = 0; d < dimension; d++) {
            int bit = (codes[d / 8] >> (7 - (d % 8))) & 1;
            output[d] = bit ? 1.0f : -1.0f;
        }
        /* If RaBitQ was used, apply the inverse (transpose) rotation. */
        if (cb->use_rabitq && cb->rotation) {
            float *tmp = (float *)malloc(dimension * sizeof(float));
            if (!tmp) return -1;
            memcpy(tmp, output, dimension * sizeof(float));
            apply_rotation_transpose(cb->rotation, tmp, output, dimension);
            free(tmp);
        }
        break;
    }

    case GV_QUANT_TERNARY: {
        for (size_t d = 0; d < dimension; d++) {
            size_t bit_pos = d * 2;
            size_t byte_idx = bit_pos / 8;
            size_t shift = 6 - (bit_pos % 8);
            uint8_t code = (codes[byte_idx] >> shift) & 0x03;
            if (code == 2) {
                output[d] = cb->std_vals[d];  /* +1 scaled by std */
            } else if (code == 0) {
                output[d] = -cb->std_vals[d]; /* -1 scaled by std */
            } else {
                output[d] = 0.0f;
            }
        }
        break;
    }

    case GV_QUANT_2BIT: {
        size_t levels = quant_levels(GV_QUANT_2BIT);
        for (size_t d = 0; d < dimension; d++) {
            size_t bit_pos = d * 2;
            size_t byte_idx = bit_pos / 8;
            size_t shift = 6 - (bit_pos % 8);
            uint8_t q = (codes[byte_idx] >> shift) & 0x03;
            float lo, hi;
            get_quant_range(cb, d, &lo, &hi);
            output[d] = scalar_dequantize_value(q, lo, hi, levels);
        }
        break;
    }

    case GV_QUANT_4BIT: {
        size_t levels = quant_levels(GV_QUANT_4BIT);
        for (size_t d = 0; d < dimension; d++) {
            size_t byte_idx = d / 2;
            uint8_t q;
            if (d % 2 == 0) {
                q = (codes[byte_idx] >> 4) & 0x0F;
            } else {
                q = codes[byte_idx] & 0x0F;
            }
            float lo, hi;
            get_quant_range(cb, d, &lo, &hi);
            output[d] = scalar_dequantize_value(q, lo, hi, levels);
        }
        break;
    }

    case GV_QUANT_8BIT: {
        size_t levels = quant_levels(GV_QUANT_8BIT);
        for (size_t d = 0; d < dimension; d++) {
            float lo, hi;
            get_quant_range(cb, d, &lo, &hi);
            output[d] = scalar_dequantize_value(codes[d], lo, hi, levels);
        }
        break;
    }
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Asymmetric distance (float query vs quantized codes)               */
/* ------------------------------------------------------------------ */

float gv_quant_distance(const GV_QuantCodebook *cb, const float *query,
                        size_t dimension, const uint8_t *codes) {
    if (!cb || !query || !codes) return -1.0f;
    if (dimension != cb->dimension) return -1.0f;

    /* For binary/RaBitQ: encode the query, then compute Hamming. */
    if (cb->type == GV_QUANT_BINARY) {
        size_t code_bytes = gv_quant_code_size(cb, dimension);
        uint8_t *qcodes = (uint8_t *)calloc(code_bytes, sizeof(uint8_t));
        if (!qcodes) return -1.0f;

        if (gv_quant_encode(cb, query, dimension, qcodes) != 0) {
            free(qcodes);
            return -1.0f;
        }

        size_t hamming = 0;
        size_t full_u64 = code_bytes / 8;
        for (size_t i = 0; i < full_u64; i++) {
            uint64_t a = ((const uint64_t *)qcodes)[i];
            uint64_t b = ((const uint64_t *)codes)[i];
            hamming += gv_popcount64(a ^ b);
        }
        for (size_t i = full_u64 * 8; i < code_bytes; i++) {
            hamming += gv_popcount64((uint64_t)(qcodes[i] ^ codes[i]));
        }

        /* Mask out padding bits in the last byte. */
        size_t tail_bits = dimension % 8;
        if (tail_bits > 0 && code_bytes > 0) {
            uint8_t mask = (uint8_t)(0xFF << (8 - tail_bits));
            uint8_t last_xor = (qcodes[code_bytes - 1] ^ codes[code_bytes - 1]);
            size_t masked_pop = gv_popcount64((uint64_t)(last_xor & mask));
            size_t full_pop   = gv_popcount64((uint64_t)last_xor);
            hamming = hamming - full_pop + masked_pop;
        }

        free(qcodes);
        return (float)hamming;
    }

    /* For scalar types: use a lookup-table approach.  For each dimension
     * pre-compute the squared difference between the query component and
     * every possible reconstructed level, then simply look up by code.  */
    size_t levels = quant_levels(cb->type);
    size_t bpv = bits_per_value(cb->type);

    /* Build per-dimension distance table. */
    float *dist_table = (float *)malloc(dimension * levels * sizeof(float));
    if (!dist_table) return -1.0f;

    for (size_t d = 0; d < dimension; d++) {
        float lo, hi;
        get_quant_range(cb, d, &lo, &hi);
        for (size_t l = 0; l < levels; l++) {
            float recon;
            if (cb->type == GV_QUANT_TERNARY) {
                if (l == 2)      recon =  cb->std_vals[d];
                else if (l == 0) recon = -cb->std_vals[d];
                else             recon =  0.0f;
            } else {
                recon = scalar_dequantize_value((uint8_t)l, lo, hi, levels);
            }
            float diff = query[d] - recon;
            dist_table[d * levels + l] = diff * diff;
        }
    }

    /* Accumulate distance by extracting each code. */
    float dist_sq = 0.0f;

    switch (cb->type) {

    case GV_QUANT_TERNARY: /* fall through */
    case GV_QUANT_2BIT: {
        for (size_t d = 0; d < dimension; d++) {
            size_t bit_pos = d * 2;
            size_t byte_idx = bit_pos / 8;
            size_t shift = 6 - (bit_pos % 8);
            uint8_t q = (codes[byte_idx] >> shift) & 0x03;
            dist_sq += dist_table[d * levels + q];
        }
        break;
    }

    case GV_QUANT_4BIT: {
        for (size_t d = 0; d < dimension; d++) {
            size_t byte_idx = d / 2;
            uint8_t q;
            if (d % 2 == 0) {
                q = (codes[byte_idx] >> 4) & 0x0F;
            } else {
                q = codes[byte_idx] & 0x0F;
            }
            dist_sq += dist_table[d * levels + q];
        }
        break;
    }

    case GV_QUANT_8BIT: {
        for (size_t d = 0; d < dimension; d++) {
            dist_sq += dist_table[d * levels + codes[d]];
        }
        break;
    }

    default:
        break;
    }

    (void)bpv;
    free(dist_table);
    return dist_sq;
}

/* ------------------------------------------------------------------ */
/*  Symmetric distance (both quantized)                                */
/* ------------------------------------------------------------------ */

float gv_quant_distance_qq(const GV_QuantCodebook *cb,
                           const uint8_t *codes_a, const uint8_t *codes_b,
                           size_t dimension) {
    if (!cb || !codes_a || !codes_b) return -1.0f;
    if (dimension != cb->dimension) return -1.0f;

    /* Binary / RaBitQ: Hamming distance. */
    if (cb->type == GV_QUANT_BINARY) {
        size_t code_bytes = gv_quant_code_size(cb, dimension);
        size_t hamming = 0;
        size_t full_u64 = code_bytes / 8;

        for (size_t i = 0; i < full_u64; i++) {
            uint64_t a = ((const uint64_t *)codes_a)[i];
            uint64_t b = ((const uint64_t *)codes_b)[i];
            hamming += gv_popcount64(a ^ b);
        }
        for (size_t i = full_u64 * 8; i < code_bytes; i++) {
            hamming += gv_popcount64((uint64_t)(codes_a[i] ^ codes_b[i]));
        }

        size_t tail_bits = dimension % 8;
        if (tail_bits > 0 && code_bytes > 0) {
            uint8_t mask = (uint8_t)(0xFF << (8 - tail_bits));
            uint8_t last_xor = (codes_a[code_bytes - 1] ^ codes_b[code_bytes - 1]);
            size_t masked_pop = gv_popcount64((uint64_t)(last_xor & mask));
            size_t full_pop   = gv_popcount64((uint64_t)last_xor);
            hamming = hamming - full_pop + masked_pop;
        }
        return (float)hamming;
    }

    /* Scalar types: decode both, compute squared Euclidean. */
    float *buf_a = (float *)malloc(dimension * sizeof(float));
    float *buf_b = (float *)malloc(dimension * sizeof(float));
    if (!buf_a || !buf_b) {
        free(buf_a);
        free(buf_b);
        return -1.0f;
    }

    if (gv_quant_decode(cb, codes_a, dimension, buf_a) != 0 ||
        gv_quant_decode(cb, codes_b, dimension, buf_b) != 0) {
        free(buf_a);
        free(buf_b);
        return -1.0f;
    }

    float dist_sq = 0.0f;
    for (size_t d = 0; d < dimension; d++) {
        float diff = buf_a[d] - buf_b[d];
        dist_sq += diff * diff;
    }

    free(buf_a);
    free(buf_b);
    return dist_sq;
}

/* ------------------------------------------------------------------ */
/*  Serialisation                                                      */
/* ------------------------------------------------------------------ */

int gv_quant_codebook_save(const GV_QuantCodebook *cb, const char *path) {
    if (!cb || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Magic */
    if (write_u8(f, GV_QUANT_MAGIC_0) != 0) goto fail;
    if (write_u8(f, GV_QUANT_MAGIC_1) != 0) goto fail;
    if (write_u8(f, GV_QUANT_MAGIC_2) != 0) goto fail;
    if (write_u8(f, GV_QUANT_MAGIC_3) != 0) goto fail;

    /* Version */
    if (write_u32(f, GV_QUANT_FILE_VERSION) != 0) goto fail;

    /* Header */
    if (write_u32(f, (uint32_t)cb->type)      != 0) goto fail;
    if (write_u32(f, (uint32_t)cb->mode)       != 0) goto fail;
    if (write_u32(f, (uint32_t)cb->dimension)  != 0) goto fail;
    if (write_u32(f, (uint32_t)cb->use_rabitq) != 0) goto fail;
    if (write_u64(f, cb->rabitq_seed)          != 0) goto fail;
    if (write_f32(f, cb->ternary_threshold)    != 0) goto fail;

    /* Per-dimension arrays */
    size_t dim = cb->dimension;
    if (fwrite(cb->min_vals,  sizeof(float), dim, f) != dim) goto fail;
    if (fwrite(cb->max_vals,  sizeof(float), dim, f) != dim) goto fail;
    if (fwrite(cb->mean_vals, sizeof(float), dim, f) != dim) goto fail;
    if (fwrite(cb->std_vals,  sizeof(float), dim, f) != dim) goto fail;

    /* Rotation matrix (if present) */
    if (cb->use_rabitq && cb->rotation) {
        size_t n = dim * dim;
        if (fwrite(cb->rotation, sizeof(float), n, f) != n) goto fail;
    }

    if (fclose(f) != 0) return -1;
    return 0;

fail:
    fclose(f);
    return -1;
}

GV_QuantCodebook *gv_quant_codebook_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    /* Magic */
    uint8_t mag[4];
    if (read_u8(f, &mag[0]) != 0) goto fail;
    if (read_u8(f, &mag[1]) != 0) goto fail;
    if (read_u8(f, &mag[2]) != 0) goto fail;
    if (read_u8(f, &mag[3]) != 0) goto fail;
    if (mag[0] != GV_QUANT_MAGIC_0 || mag[1] != GV_QUANT_MAGIC_1 ||
        mag[2] != GV_QUANT_MAGIC_2 || mag[3] != GV_QUANT_MAGIC_3)
        goto fail;

    /* Version */
    uint32_t version = 0;
    if (read_u32(f, &version) != 0) goto fail;
    if (version != GV_QUANT_FILE_VERSION) goto fail;

    /* Header */
    uint32_t type_u32 = 0, mode_u32 = 0, dim_u32 = 0, rabitq_u32 = 0;
    uint64_t seed_u64 = 0;
    float    thresh_f = 0.0f;

    if (read_u32(f, &type_u32)   != 0) goto fail;
    if (read_u32(f, &mode_u32)   != 0) goto fail;
    if (read_u32(f, &dim_u32)    != 0) goto fail;
    if (read_u32(f, &rabitq_u32) != 0) goto fail;
    if (read_u64(f, &seed_u64)   != 0) goto fail;
    if (read_f32(f, &thresh_f)   != 0) goto fail;

    size_t dim = (size_t)dim_u32;
    if (dim == 0) goto fail;

    GV_QuantCodebook *cb = (GV_QuantCodebook *)calloc(1, sizeof(GV_QuantCodebook));
    if (!cb) goto fail;

    cb->type               = (GV_QuantType)type_u32;
    cb->mode               = (GV_QuantMode)mode_u32;
    cb->dimension          = dim;
    cb->use_rabitq         = (int)rabitq_u32;
    cb->rabitq_seed        = seed_u64;
    cb->ternary_threshold  = thresh_f;

    cb->min_vals  = (float *)malloc(dim * sizeof(float));
    cb->max_vals  = (float *)malloc(dim * sizeof(float));
    cb->mean_vals = (float *)malloc(dim * sizeof(float));
    cb->std_vals  = (float *)malloc(dim * sizeof(float));
    if (!cb->min_vals || !cb->max_vals || !cb->mean_vals || !cb->std_vals) {
        gv_quant_codebook_destroy(cb);
        goto fail;
    }

    if (fread(cb->min_vals,  sizeof(float), dim, f) != dim) { gv_quant_codebook_destroy(cb); goto fail; }
    if (fread(cb->max_vals,  sizeof(float), dim, f) != dim) { gv_quant_codebook_destroy(cb); goto fail; }
    if (fread(cb->mean_vals, sizeof(float), dim, f) != dim) { gv_quant_codebook_destroy(cb); goto fail; }
    if (fread(cb->std_vals,  sizeof(float), dim, f) != dim) { gv_quant_codebook_destroy(cb); goto fail; }

    /* Rotation matrix */
    cb->rotation = NULL;
    if (cb->use_rabitq) {
        size_t n = dim * dim;
        cb->rotation = (float *)malloc(n * sizeof(float));
        if (!cb->rotation) { gv_quant_codebook_destroy(cb); goto fail; }
        if (fread(cb->rotation, sizeof(float), n, f) != n) {
            gv_quant_codebook_destroy(cb);
            goto fail;
        }
    }

    fclose(f);
    return cb;

fail:
    fclose(f);
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Destroy                                                            */
/* ------------------------------------------------------------------ */

void gv_quant_codebook_destroy(GV_QuantCodebook *cb) {
    if (!cb) return;
    free(cb->min_vals);
    free(cb->max_vals);
    free(cb->mean_vals);
    free(cb->std_vals);
    free(cb->rotation);
    free(cb);
}
