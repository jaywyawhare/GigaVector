#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "gigavector/gv_codebook.h"

/* ------------------------------------------------------------------ */
/*  File format constants                                              */
/* ------------------------------------------------------------------ */
#define GV_CODEBOOK_MAGIC_0 'G'
#define GV_CODEBOOK_MAGIC_1 'V'
#define GV_CODEBOOK_MAGIC_2 'C'
#define GV_CODEBOOK_MAGIC_3 'B'
#define GV_CODEBOOK_VERSION  1

/* ------------------------------------------------------------------ */
/*  Small I/O helpers                                                  */
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

/* ------------------------------------------------------------------ */
/*  Squared Euclidean distance between two sub-vectors                 */
/* ------------------------------------------------------------------ */
static float subvec_dist_sq(const float *a, const float *b, size_t len) {
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/*  Simple xorshift32 PRNG (deterministic, no global state)            */
/* ------------------------------------------------------------------ */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* ------------------------------------------------------------------ */
/*  K-means for a single subspace                                      */
/* ------------------------------------------------------------------ */

/**
 * Train one sub-quantizer codebook in-place.
 *
 * @param codebook   Output: ksub * dsub floats (overwritten).
 * @param subvecs    Row-major sub-vectors extracted from training data
 *                   (count * dsub floats).
 * @param count      Number of training sub-vectors.
 * @param dsub       Sub-vector dimensionality.
 * @param ksub       Number of centroids.
 * @param iters      K-means iterations.
 * @param rng_state  Pointer to PRNG state (modified in place).
 */
static void kmeans_subspace(float *codebook, const float *subvecs,
                            size_t count, size_t dsub, size_t ksub,
                            size_t iters, uint32_t *rng_state) {
    if (count == 0 || ksub == 0 || dsub == 0) return;

    /* -------------------------------------------------------------- */
    /*  Initialise centroids by picking random training vectors        */
    /* -------------------------------------------------------------- */
    size_t init_k = ksub < count ? ksub : count;

    /* Fisher-Yates partial shuffle to pick init_k unique indices. */
    size_t *perm = (size_t *)malloc(count * sizeof(size_t));
    if (!perm) return;
    for (size_t i = 0; i < count; i++) perm[i] = i;

    for (size_t i = 0; i < init_k; i++) {
        size_t j = i + (xorshift32(rng_state) % (uint32_t)(count - i));
        size_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
        memcpy(&codebook[i * dsub], &subvecs[perm[i] * dsub],
               dsub * sizeof(float));
    }
    free(perm);

    /* If fewer vectors than centroids, zero-fill the rest. */
    for (size_t k = init_k; k < ksub; k++) {
        memset(&codebook[k * dsub], 0, dsub * sizeof(float));
    }

    /* -------------------------------------------------------------- */
    /*  Allocate working buffers for Lloyd iterations                   */
    /* -------------------------------------------------------------- */
    uint32_t *assignments = (uint32_t *)malloc(count * sizeof(uint32_t));
    float    *accum       = (float *)malloc(ksub * dsub * sizeof(float));
    uint32_t *counts      = (uint32_t *)malloc(ksub * sizeof(uint32_t));
    if (!assignments || !accum || !counts) {
        free(assignments);
        free(accum);
        free(counts);
        return;
    }

    /* -------------------------------------------------------------- */
    /*  Lloyd iterations                                               */
    /* -------------------------------------------------------------- */
    for (size_t it = 0; it < iters; it++) {

        /* ---- Assignment step ------------------------------------ */
        for (size_t i = 0; i < count; i++) {
            float best_d = FLT_MAX;
            uint32_t best_k = 0;
            const float *vec = &subvecs[i * dsub];
            for (size_t k = 0; k < ksub; k++) {
                float d = subvec_dist_sq(vec, &codebook[k * dsub], dsub);
                if (d < best_d) {
                    best_d = d;
                    best_k = (uint32_t)k;
                }
            }
            assignments[i] = best_k;
        }

        /* ---- Update step ---------------------------------------- */
        memset(accum,  0, ksub * dsub * sizeof(float));
        memset(counts, 0, ksub * sizeof(uint32_t));

        for (size_t i = 0; i < count; i++) {
            uint32_t k = assignments[i];
            counts[k]++;
            const float *vec = &subvecs[i * dsub];
            float *dst = &accum[k * dsub];
            for (size_t d = 0; d < dsub; d++) {
                dst[d] += vec[d];
            }
        }

        for (size_t k = 0; k < ksub; k++) {
            if (counts[k] > 0) {
                float inv = 1.0f / (float)counts[k];
                for (size_t d = 0; d < dsub; d++) {
                    codebook[k * dsub + d] = accum[k * dsub + d] * inv;
                }
            } else {
                /* Empty cluster: reinitialise from a random training vector. */
                size_t rand_idx = xorshift32(rng_state) % (uint32_t)count;
                memcpy(&codebook[k * dsub], &subvecs[rand_idx * dsub],
                       dsub * sizeof(float));
            }
        }
    }

    free(assignments);
    free(accum);
    free(counts);
}

/* ================================================================== */
/*  Public API                                                         */
/* ================================================================== */

GV_Codebook *gv_codebook_create(size_t dimension, size_t m, uint8_t nbits) {
    if (dimension == 0 || m == 0 || nbits == 0 || nbits > 8) return NULL;
    if (dimension % m != 0) return NULL;

    GV_Codebook *cb = (GV_Codebook *)calloc(1, sizeof(GV_Codebook));
    if (!cb) return NULL;

    cb->dimension = dimension;
    cb->m         = m;
    cb->nbits     = nbits;
    cb->ksub      = (size_t)1 << nbits;
    cb->dsub      = dimension / m;
    cb->trained   = 0;

    size_t total_floats = cb->m * cb->ksub * cb->dsub;
    cb->centroids = (float *)calloc(total_floats, sizeof(float));
    if (!cb->centroids) {
        free(cb);
        return NULL;
    }

    return cb;
}

void gv_codebook_destroy(GV_Codebook *cb) {
    if (!cb) return;
    free(cb->centroids);
    free(cb);
}

/* ------------------------------------------------------------------ */
/*  Training                                                           */
/* ------------------------------------------------------------------ */

int gv_codebook_train(GV_Codebook *cb, const float *data, size_t count,
                      size_t train_iters) {
    if (!cb || !data || count == 0 || train_iters == 0) return -1;

    /* Temporary buffer for extracted sub-vectors (count * dsub). */
    float *subvecs = (float *)malloc(count * cb->dsub * sizeof(float));
    if (!subvecs) return -1;

    /* Seed the PRNG with something that varies across runs. */
    uint32_t rng_state = (uint32_t)(count * 2654435761u + cb->m * 40503u + 1);

    for (size_t mi = 0; mi < cb->m; mi++) {
        /* Extract the mi-th sub-vector from every training vector. */
        for (size_t i = 0; i < count; i++) {
            memcpy(&subvecs[i * cb->dsub],
                   &data[i * cb->dimension + mi * cb->dsub],
                   cb->dsub * sizeof(float));
        }

        float *sub_codebook = &cb->centroids[mi * cb->ksub * cb->dsub];
        kmeans_subspace(sub_codebook, subvecs, count, cb->dsub, cb->ksub,
                        train_iters, &rng_state);
    }

    free(subvecs);
    cb->trained = 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Encode                                                             */
/* ------------------------------------------------------------------ */

int gv_codebook_encode(const GV_Codebook *cb, const float *vector,
                       uint8_t *codes) {
    if (!cb || !vector || !codes) return -1;
    if (!cb->trained) return -1;

    for (size_t mi = 0; mi < cb->m; mi++) {
        const float *subvec = &vector[mi * cb->dsub];
        const float *sub_codebook = &cb->centroids[mi * cb->ksub * cb->dsub];

        float   best_d = FLT_MAX;
        uint8_t best_c = 0;

        for (size_t k = 0; k < cb->ksub; k++) {
            float d = subvec_dist_sq(subvec, &sub_codebook[k * cb->dsub],
                                     cb->dsub);
            if (d < best_d) {
                best_d = d;
                best_c = (uint8_t)k;
            }
        }
        codes[mi] = best_c;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Decode                                                             */
/* ------------------------------------------------------------------ */

int gv_codebook_decode(const GV_Codebook *cb, const uint8_t *codes,
                       float *output) {
    if (!cb || !codes || !output) return -1;
    if (!cb->trained) return -1;

    for (size_t mi = 0; mi < cb->m; mi++) {
        const float *centroid =
            &cb->centroids[mi * cb->ksub * cb->dsub + codes[mi] * cb->dsub];
        memcpy(&output[mi * cb->dsub], centroid, cb->dsub * sizeof(float));
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Asymmetric Distance Computation (ADC)                              */
/* ------------------------------------------------------------------ */

float gv_codebook_distance_adc(const GV_Codebook *cb, const float *query,
                               const uint8_t *codes) {
    if (!cb || !query || !codes) return -1.0f;
    if (!cb->trained) return -1.0f;

    /*
     * Build a distance look-up table: for every subspace mi and every
     * centroid k, store the squared distance from the query sub-vector
     * to that centroid.  Then accumulate the entry corresponding to
     * each code.
     */
    float *table = (float *)malloc(cb->m * cb->ksub * sizeof(float));
    if (!table) return -1.0f;

    for (size_t mi = 0; mi < cb->m; mi++) {
        const float *q_sub = &query[mi * cb->dsub];
        const float *sub_codebook = &cb->centroids[mi * cb->ksub * cb->dsub];
        for (size_t k = 0; k < cb->ksub; k++) {
            table[mi * cb->ksub + k] =
                subvec_dist_sq(q_sub, &sub_codebook[k * cb->dsub], cb->dsub);
        }
    }

    float dist_sq = 0.0f;
    for (size_t mi = 0; mi < cb->m; mi++) {
        dist_sq += table[mi * cb->ksub + codes[mi]];
    }

    free(table);
    return sqrtf(dist_sq);
}

/* ------------------------------------------------------------------ */
/*  Serialisation: FILE* variants                                      */
/* ------------------------------------------------------------------ */

int gv_codebook_save_fp(const GV_Codebook *cb, FILE *out) {
    if (!cb || !out) return -1;

    /* Magic bytes "GVCB" */
    if (write_u8(out, GV_CODEBOOK_MAGIC_0) != 0) return -1;
    if (write_u8(out, GV_CODEBOOK_MAGIC_1) != 0) return -1;
    if (write_u8(out, GV_CODEBOOK_MAGIC_2) != 0) return -1;
    if (write_u8(out, GV_CODEBOOK_MAGIC_3) != 0) return -1;

    /* Version */
    if (write_u32(out, GV_CODEBOOK_VERSION) != 0) return -1;

    /* Header fields */
    if (write_u32(out, (uint32_t)cb->dimension) != 0) return -1;
    if (write_u32(out, (uint32_t)cb->m)         != 0) return -1;
    if (write_u8(out, cb->nbits)                 != 0) return -1;
    if (write_u32(out, (uint32_t)cb->trained)    != 0) return -1;

    /* Centroid data */
    size_t n_floats = cb->m * cb->ksub * cb->dsub;
    if (fwrite(cb->centroids, sizeof(float), n_floats, out) != n_floats)
        return -1;

    return 0;
}

GV_Codebook *gv_codebook_load_fp(FILE *in) {
    if (!in) return NULL;

    /* Read and validate magic */
    uint8_t mag[4];
    if (read_u8(in, &mag[0]) != 0) return NULL;
    if (read_u8(in, &mag[1]) != 0) return NULL;
    if (read_u8(in, &mag[2]) != 0) return NULL;
    if (read_u8(in, &mag[3]) != 0) return NULL;

    if (mag[0] != GV_CODEBOOK_MAGIC_0 || mag[1] != GV_CODEBOOK_MAGIC_1 ||
        mag[2] != GV_CODEBOOK_MAGIC_2 || mag[3] != GV_CODEBOOK_MAGIC_3)
        return NULL;

    /* Version */
    uint32_t version = 0;
    if (read_u32(in, &version) != 0) return NULL;
    if (version != GV_CODEBOOK_VERSION) return NULL;

    /* Header fields */
    uint32_t dimension = 0, m = 0, trained = 0;
    uint8_t  nbits = 0;

    if (read_u32(in, &dimension) != 0) return NULL;
    if (read_u32(in, &m)         != 0) return NULL;
    if (read_u8(in, &nbits)      != 0) return NULL;
    if (read_u32(in, &trained)   != 0) return NULL;

    /* Allocate the codebook via the normal constructor. */
    GV_Codebook *cb = gv_codebook_create((size_t)dimension, (size_t)m, nbits);
    if (!cb) return NULL;

    /* Read centroid data */
    size_t n_floats = cb->m * cb->ksub * cb->dsub;
    if (fread(cb->centroids, sizeof(float), n_floats, in) != n_floats) {
        gv_codebook_destroy(cb);
        return NULL;
    }

    cb->trained = (int)trained;
    return cb;
}

/* ------------------------------------------------------------------ */
/*  Serialisation: path-based convenience wrappers                     */
/* ------------------------------------------------------------------ */

int gv_codebook_save(const GV_Codebook *cb, const char *filepath) {
    if (!cb || !filepath) return -1;

    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    int rc = gv_codebook_save_fp(cb, f);
    if (fclose(f) != 0) rc = -1;
    return rc;
}

GV_Codebook *gv_codebook_load(const char *filepath) {
    if (!filepath) return NULL;

    FILE *f = fopen(filepath, "rb");
    if (!f) return NULL;

    GV_Codebook *cb = gv_codebook_load_fp(f);
    fclose(f);
    return cb;
}

/* ------------------------------------------------------------------ */
/*  Deep copy                                                          */
/* ------------------------------------------------------------------ */

GV_Codebook *gv_codebook_copy(const GV_Codebook *cb) {
    if (!cb) return NULL;

    GV_Codebook *copy = gv_codebook_create(cb->dimension, cb->m, cb->nbits);
    if (!copy) return NULL;

    size_t n_floats = cb->m * cb->ksub * cb->dsub;
    memcpy(copy->centroids, cb->centroids, n_floats * sizeof(float));
    copy->trained = cb->trained;

    return copy;
}
