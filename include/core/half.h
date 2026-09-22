#ifndef GIGAVECTOR_GV_HALF_H
#define GIGAVECTOR_GV_HALF_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* IEEE-754 binary16 (half precision) codec — halves vector RAM at ~3 decimal
 * digits of precision (near-lossless for normalized embeddings). Pure integer
 * bit-twiddling; no hardware fp16 required. */

static inline uint16_t gv_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) {                 /* Inf / NaN */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);        /* overflow -> Inf */
    if (exp <= 0) {                                            /* subnormal / underflow */
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half += 1u;            /* round to nearest even-ish */
        return (uint16_t)(sign | half);
    }
    uint16_t half = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if (mant & 0x1000u) half += 1u;                            /* round to nearest */
    return half;
}

static inline float gv_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }                         /* zero */
        else {                                                 /* subnormal -> normalize */
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);               /* Inf / NaN */
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &out, sizeof(f));
    return f;
}

/* Symmetric per-vector int8 codec — quarters vector RAM. The scale is
 * max(|x_i|)/127; decode is v = q * scale. Store one float scale per vector. */

static inline float gv_i8_encode(const float *src, int8_t *dst, size_t dim) {
    float amax = 0.0f;
    for (size_t i = 0; i < dim; i++) { float a = src[i] < 0 ? -src[i] : src[i]; if (a > amax) amax = a; }
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    float inv = 1.0f / scale;
    for (size_t i = 0; i < dim; i++) {
        float q = src[i] * inv;
        int r = (int)(q < 0 ? q - 0.5f : q + 0.5f);
        if (r > 127) r = 127; else if (r < -127) r = -127;
        dst[i] = (int8_t)r;
    }
    return scale;
}

static inline void gv_i8_decode(const int8_t *src, float scale, float *dst, size_t dim) {
    for (size_t i = 0; i < dim; i++) dst[i] = (float)src[i] * scale;
}

#ifdef __cplusplus
}
#endif

#endif
