#include <stdlib.h>
#include "core/memory.h"
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "storage/turboquant.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct GV_TurboQuantCode {
    size_t dim;
    uint8_t bits;
    size_t projections;
    float *radii;
    uint16_t *angle_indices;
    int8_t *qjl_signs;
    float residual_norm;
};

struct GV_TurboQuantizer {
    size_t dim;
    uint8_t bits;
    uint8_t polar_bits;
    size_t projections;
    uint64_t seed;
    int use_qjl;
    int use_fhwt;
    float *rotation;
    float *fhwt_signs;
    float *qjl_matrix;
};

static uint64_t tq_rng(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * 0x2545F4914F6CDD1DULL;
}

static float tq_uniform01(uint64_t *state) {
    return (float)(tq_rng(state) >> 11) * (1.0f / 9007199254740992.0f);
}

static float tq_gaussian(uint64_t *state) {
    float u1 = tq_uniform01(state);
    if (u1 < 1e-10f) {
        u1 = 1e-10f;
    }
    float u2 = tq_uniform01(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static int tq_is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

static void tq_fwht(float *values, size_t n) {
    size_t step = 1;
    while (step < n) {
        size_t block = step * 2;
        for (size_t start = 0; start < n; start += block) {
            for (size_t offset = 0; offset < step; offset++) {
                float a = values[start + offset];
                float b = values[start + offset + step];
                values[start + offset] = a + b;
                values[start + offset + step] = a - b;
            }
        }
        step = block;
    }
    float scale = 1.0f / sqrtf((float)n);
    for (size_t i = 0; i < n; i++) {
        values[i] *= scale;
    }
}

static int tq_build_fhwt_signs(size_t dim, uint64_t seed, float **out_signs) {
    float *signs = (float *)gv_alloc(dim * sizeof(float));
    if (!signs) {
        return -1;
    }
    uint64_t rng = seed ^ 0xA11CE55ED5A5EED5ULL;
    for (size_t i = 0; i < dim; i++) {
        signs[i] = (tq_rng(&rng) & 1) ? 1.0f : -1.0f;
    }
    *out_signs = signs;
    return 0;
}

static int tq_build_rotation_qr(size_t dim, uint64_t seed, float **out_matrix) {
    float *matrix = (float *)gv_calloc(dim * dim, sizeof(float));
    float *col = (float *)gv_alloc(dim * sizeof(float));
    float *tmp = (float *)gv_alloc(dim * sizeof(float));
    if (!matrix || !col || !tmp) {
        gv_free(matrix);
        gv_free(col);
        gv_free(tmp);
        return -1;
    }

    uint64_t rng = seed;
    for (size_t j = 0; j < dim; j++) {
        for (size_t i = 0; i < dim; i++) {
            col[i] = tq_gaussian(&rng);
        }
        for (size_t k = 0; k < j; k++) {
            float dot = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                dot += matrix[i * dim + k] * col[i];
            }
            for (size_t i = 0; i < dim; i++) {
                col[i] -= dot * matrix[i * dim + k];
            }
        }
        float norm = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            norm += col[i] * col[i];
        }
        if (norm <= 0.0f) {
            gv_free(matrix);
            gv_free(col);
            gv_free(tmp);
            return -1;
        }
        norm = 1.0f / sqrtf(norm);
        for (size_t i = 0; i < dim; i++) {
            matrix[i * dim + j] = col[i] * norm;
        }
    }

    gv_free(col);
    gv_free(tmp);
    *out_matrix = matrix;
    return 0;
}

static void tq_rotate_apply(const GV_TurboQuantizer *q, const float *input, float *output) {
    size_t d = q->dim;
    if (q->use_fhwt) {
        for (size_t i = 0; i < d; i++) {
            output[i] = input[i] * q->fhwt_signs[i];
        }
        tq_fwht(output, d);
        return;
    }
    for (size_t i = 0; i < d; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < d; j++) {
            sum += q->rotation[i * d + j] * input[j];
        }
        output[i] = sum;
    }
}

static void tq_rotate_inverse(const GV_TurboQuantizer *q, const float *input, float *output) {
    size_t d = q->dim;
    if (q->use_fhwt) {
        memcpy(output, input, d * sizeof(float));
        tq_fwht(output, d);
        for (size_t i = 0; i < d; i++) {
            output[i] *= q->fhwt_signs[i];
        }
        return;
    }
    for (size_t j = 0; j < d; j++) {
        float sum = 0.0f;
        for (size_t i = 0; i < d; i++) {
            sum += q->rotation[i * d + j] * input[i];
        }
        output[j] = sum;
    }
}

static int tq_build_qjl_matrix(size_t dim, size_t projections, uint64_t seed, float **out_matrix) {
    if (projections == 0) {
        *out_matrix = NULL;
        return 0;
    }
    float *matrix = (float *)gv_alloc(projections * dim * sizeof(float));
    if (!matrix) {
        return -1;
    }
    uint64_t rng = seed ^ 0xDEADBEEF12345678ULL;
    for (size_t i = 0; i < projections * dim; i++) {
        matrix[i] = tq_gaussian(&rng);
    }
    *out_matrix = matrix;
    return 0;
}

static void tq_encode_pair(float a, float b, uint8_t bits, float *radius_out, uint16_t *index_out) {
    float r = sqrtf(a * a + b * b);
    float theta = atan2f(b, a);
    uint32_t levels = 1u << bits;
    float normalized = (theta + (float)M_PI) / (2.0f * (float)M_PI);
    uint32_t idx = (uint32_t)floorf(normalized * (float)levels);
    if (idx >= levels) {
        idx = levels - 1;
    }
    *radius_out = r;
    *index_out = (uint16_t)idx;
}

static float tq_dequantize_angle(uint16_t index, uint8_t bits) {
    uint32_t levels = 1u << bits;
    float idx = (float)index;
    return (idx / (float)levels) * (2.0f * (float)M_PI) - (float)M_PI;
}

static float tq_polar_inner_product(const GV_TurboQuantCode *code, const float *rotated_query) {
    size_t pairs = code->dim / 2;
    float estimate = 0.0f;
    for (size_t i = 0; i < pairs; i++) {
        float theta = tq_dequantize_angle(code->angle_indices[i], code->bits);
        float r = code->radii[i];
        float qa = rotated_query[2 * i];
        float qb = rotated_query[2 * i + 1];
        estimate += r * (qa * cosf(theta) + qb * sinf(theta));
    }
    return estimate;
}

static float tq_code_norm_sq(const GV_TurboQuantCode *code) {
    float sum = 0.0f;
    size_t pairs = code->dim / 2;
    for (size_t i = 0; i < pairs; i++) {
        float r = code->radii[i];
        sum += r * r;
    }
    return sum;
}

static float tq_qjl_inner_product(const GV_TurboQuantizer *q, const GV_TurboQuantCode *code,
                                  const float *qjl_projected) {
    if (code->projections == 0 || code->qjl_signs == NULL || qjl_projected == NULL) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < code->projections; i++) {
        sum += (float)code->qjl_signs[i] * qjl_projected[i];
    }
    float scale = sqrtf((float)M_PI / 2.0f) * code->residual_norm / (float)code->projections;
    return scale * sum;
}

GV_TurboQuantizer *turboquant_create(size_t dimension, const GV_TurboQuantConfig *config) {
    if (dimension == 0 || dimension % 2 != 0) {
        return NULL;
    }

    GV_TurboQuantConfig cfg = {
        .bits = 8,
        .projections = dimension / 4,
        .seed = 42,
        .use_qjl = 1,
        .rotation = GV_TURBOQUANT_ROTATION_AUTO
    };
    if (config) {
        cfg = *config;
    }
    if (cfg.bits < 1 || cfg.bits > 16) {
        return NULL;
    }
    if (cfg.use_qjl) {
        if (cfg.bits < 2) {
            return NULL;
        }
        if (cfg.projections == 0) {
            return NULL;
        }
    } else if (cfg.projections > 0) {
        cfg.projections = 0;
    }

    GV_TurboQuantizer *q = (GV_TurboQuantizer *)gv_calloc(1, sizeof(GV_TurboQuantizer));
    if (!q) {
        return NULL;
    }

    q->dim = dimension;
    q->bits = cfg.bits;
    q->polar_bits = cfg.use_qjl ? (uint8_t)(cfg.bits - 1) : cfg.bits;
    q->projections = cfg.use_qjl ? cfg.projections : 0;
    q->seed = cfg.seed;
    q->use_qjl = cfg.use_qjl ? 1 : 0;

    int want_fhwt = 0;
    if (cfg.rotation == GV_TURBOQUANT_ROTATION_FHWT ||
        (cfg.rotation == GV_TURBOQUANT_ROTATION_AUTO && tq_is_power_of_two(dimension))) {
        want_fhwt = 1;
    }
    q->use_fhwt = want_fhwt;

    if (q->use_fhwt) {
        if (!tq_is_power_of_two(dimension)) {
            turboquant_destroy(q);
            return NULL;
        }
        if (tq_build_fhwt_signs(dimension, cfg.seed, &q->fhwt_signs) != 0) {
            turboquant_destroy(q);
            return NULL;
        }
    } else {
        if (tq_build_rotation_qr(dimension, cfg.seed, &q->rotation) != 0) {
            turboquant_destroy(q);
            return NULL;
        }
    }

    if (q->use_qjl) {
        uint64_t qjl_seed = cfg.seed ^ 0xCAFEBABE00000001ULL;
        if (tq_build_qjl_matrix(dimension, q->projections, qjl_seed, &q->qjl_matrix) != 0) {
            turboquant_destroy(q);
            return NULL;
        }
    }

    return q;
}

void turboquant_destroy(GV_TurboQuantizer *quantizer) {
    if (!quantizer) {
        return;
    }
    gv_free(quantizer->rotation);
    gv_free(quantizer->fhwt_signs);
    gv_free(quantizer->qjl_matrix);
    gv_free(quantizer);
}

size_t turboquant_dimension(const GV_TurboQuantizer *quantizer) {
    return quantizer ? quantizer->dim : 0;
}

GV_TurboQuantCode *turboquant_encode(const GV_TurboQuantizer *quantizer, const float *vector) {
    if (!quantizer || !vector) {
        return NULL;
    }

    size_t dim = quantizer->dim;
    size_t pairs = dim / 2;
    float *rotated = (float *)gv_alloc(dim * sizeof(float));
    if (!rotated) {
        return NULL;
    }
    tq_rotate_apply(quantizer, vector, rotated);

    GV_TurboQuantCode *code = (GV_TurboQuantCode *)gv_calloc(1, sizeof(GV_TurboQuantCode));
    if (!code) {
        gv_free(rotated);
        return NULL;
    }
    code->dim = dim;
    code->bits = quantizer->polar_bits;
    code->projections = quantizer->projections;
    code->radii = (float *)gv_alloc(pairs * sizeof(float));
    code->angle_indices = (uint16_t *)gv_alloc(pairs * sizeof(uint16_t));
    if (!code->radii || !code->angle_indices) {
        turboquant_code_destroy(code);
        gv_free(rotated);
        return NULL;
    }

    for (size_t i = 0; i < pairs; i++) {
        tq_encode_pair(rotated[2 * i], rotated[2 * i + 1], quantizer->polar_bits,
                       &code->radii[i], &code->angle_indices[i]);
    }

    if (quantizer->use_qjl && quantizer->projections > 0) {
        float *recon_rotated = (float *)gv_alloc(dim * sizeof(float));
        float *residual = (float *)gv_alloc(dim * sizeof(float));
        code->qjl_signs = (int8_t *)gv_alloc(quantizer->projections * sizeof(int8_t));
        if (!recon_rotated || !residual || !code->qjl_signs) {
            gv_free(recon_rotated);
            gv_free(residual);
            turboquant_code_destroy(code);
            gv_free(rotated);
            return NULL;
        }

        for (size_t i = 0; i < pairs; i++) {
            float theta = tq_dequantize_angle(code->angle_indices[i], quantizer->polar_bits);
            float r = code->radii[i];
            recon_rotated[2 * i] = r * cosf(theta);
            recon_rotated[2 * i + 1] = r * sinf(theta);
        }

        float *recon = (float *)gv_alloc(dim * sizeof(float));
        if (!recon) {
            gv_free(recon_rotated);
            gv_free(residual);
            turboquant_code_destroy(code);
            gv_free(rotated);
            return NULL;
        }
        tq_rotate_inverse(quantizer, recon_rotated, recon);
        for (size_t i = 0; i < dim; i++) {
            residual[i] = vector[i] - recon[i];
        }
        code->residual_norm = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            code->residual_norm += residual[i] * residual[i];
        }
        code->residual_norm = sqrtf(code->residual_norm);

        for (size_t p = 0; p < quantizer->projections; p++) {
            const float *row = quantizer->qjl_matrix + p * dim;
            float dot = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                dot += row[i] * residual[i];
            }
            code->qjl_signs[p] = (dot >= 0.0f) ? 1 : -1;
        }

        gv_free(recon);
        gv_free(recon_rotated);
        gv_free(residual);
    }

    gv_free(rotated);
    return code;
}

void turboquant_code_destroy(GV_TurboQuantCode *code) {
    if (!code) {
        return;
    }
    gv_free(code->radii);
    gv_free(code->angle_indices);
    gv_free(code->qjl_signs);
    gv_free(code);
}

int turboquant_prepare_query(const GV_TurboQuantizer *quantizer, const float *query,
                             GV_TurboQuantQuery *out) {
    if (!quantizer || !query || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->projections = quantizer->projections;
    out->rotated_query = (float *)gv_alloc(quantizer->dim * sizeof(float));
    if (!out->rotated_query) {
        return -1;
    }
    tq_rotate_apply(quantizer, query, out->rotated_query);

    if (quantizer->use_qjl && quantizer->projections > 0) {
        out->qjl_projected = (float *)gv_alloc(quantizer->projections * sizeof(float));
        if (!out->qjl_projected) {
            turboquant_query_destroy(out);
            return -1;
        }
        for (size_t p = 0; p < quantizer->projections; p++) {
            const float *row = quantizer->qjl_matrix + p * quantizer->dim;
            float dot = 0.0f;
            for (size_t i = 0; i < quantizer->dim; i++) {
                dot += row[i] * query[i];
            }
            out->qjl_projected[p] = dot;
        }
    }
    return 0;
}

void turboquant_query_destroy(GV_TurboQuantQuery *query) {
    if (!query) {
        return;
    }
    gv_free(query->rotated_query);
    gv_free(query->qjl_projected);
    query->rotated_query = NULL;
    query->qjl_projected = NULL;
    query->projections = 0;
}

float turboquant_inner_product(const GV_TurboQuantizer *quantizer, const GV_TurboQuantCode *code,
                               const GV_TurboQuantQuery *query) {
    if (!quantizer || !code || !query || !query->rotated_query) {
        return -1.0f;
    }
    float ip = tq_polar_inner_product(code, query->rotated_query);
    ip += tq_qjl_inner_product(quantizer, code, query->qjl_projected);
    return ip;
}

float turboquant_l2_squared(const GV_TurboQuantizer *quantizer, const GV_TurboQuantCode *code,
                            const GV_TurboQuantQuery *query) {
    if (!quantizer || !code || !query || !query->rotated_query) {
        return -1.0f;
    }
    float query_norm_sq = 0.0f;
    for (size_t i = 0; i < quantizer->dim; i++) {
        float v = query->rotated_query[i];
        query_norm_sq += v * v;
    }
    float ip = turboquant_inner_product(quantizer, code, query);
    float dist = query_norm_sq + tq_code_norm_sq(code) - 2.0f * ip;
    return dist < 0.0f ? 0.0f : dist;
}

float turboquant_distance_prepared(const GV_TurboQuantizer *quantizer,
                                   const GV_TurboQuantCode *code,
                                   const GV_TurboQuantQuery *query,
                                   GV_DistanceType distance_type) {
    if (!quantizer || !code || !query) {
        return -1.0f;
    }
    switch (distance_type) {
    case GV_DISTANCE_EUCLIDEAN:
        return turboquant_l2_squared(quantizer, code, query);
    case GV_DISTANCE_COSINE: {
        float ip = turboquant_inner_product(quantizer, code, query);
        float query_norm_sq = 0.0f;
        for (size_t i = 0; i < quantizer->dim; i++) {
            float v = query->rotated_query[i];
            query_norm_sq += v * v;
        }
        float code_norm_sq = tq_code_norm_sq(code);
        float denom = sqrtf(query_norm_sq * code_norm_sq);
        if (denom <= 0.0f) {
            return 1.0f;
        }
        float cos_sim = ip / denom;
        if (cos_sim > 1.0f) {
            cos_sim = 1.0f;
        }
        if (cos_sim < -1.0f) {
            cos_sim = -1.0f;
        }
        return 1.0f - cos_sim;
    }
    case GV_DISTANCE_DOT_PRODUCT:
        return -turboquant_inner_product(quantizer, code, query);
    case GV_DISTANCE_MANHATTAN:
    case GV_DISTANCE_HAMMING:
    default:
        return -1.0f;
    }
}

float turboquant_distance(const GV_TurboQuantizer *quantizer, const GV_TurboQuantCode *code,
                          const float *query, GV_DistanceType distance_type) {
    GV_TurboQuantQuery prepared;
    if (turboquant_prepare_query(quantizer, query, &prepared) != 0) {
        return -1.0f;
    }
    float dist = turboquant_distance_prepared(quantizer, code, &prepared, distance_type);
    turboquant_query_destroy(&prepared);
    return dist;
}
