#include "features/cypher_vector.h"
#include "core/memory.h"
#include "core/utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

float cypher_vector_distance(const float *stored, size_t stored_dim,
                             const float *query, size_t query_dim,
                             GV_CypherVecDistType type) {
    if (!stored || !query || stored_dim == 0 || query_dim == 0) return -1.0f;
    size_t dim = stored_dim < query_dim ? stored_dim : query_dim;
    switch (type) {
        case GV_VECDIST_L2: {
            float sum = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                float d = stored[i] - query[i];
                sum += d * d;
            }
            return sqrtf(sum);
        }
        case GV_VECDIST_COSINE: {
            float dot = 0.0f, na = 0.0f, nb = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                dot += stored[i] * query[i];
                na  += stored[i] * stored[i];
                nb  += query[i] * query[i];
            }
            float denom = sqrtf(na) * sqrtf(nb);
            if (denom < 1e-12f) return 1.0f;
            float sim = dot / denom;
            if (sim > 1.0f) sim = 1.0f;
            if (sim < -1.0f) sim = -1.0f;
            return 1.0f - sim;
        }
        case GV_VECDIST_DOT: {
            float dot = 0.0f;
            for (size_t i = 0; i < dim; i++) dot += stored[i] * query[i];
            return -dot;
        }
        case GV_VECDIST_HAMMING: {
            int dist = 0;
            for (size_t i = 0; i < dim; i++) {
                uint8_t ab = stored[i] > 0.0f ? 1 : 0;
                uint8_t qb = query[i]  > 0.0f ? 1 : 0;
                dist += (ab != qb) ? 1 : 0;
            }
            return (float)dist;
        }
    }
    return -1.0f;
}

GV_CypherVecDistType cypher_parse_vecdist_type(const char *s) {
    if (!s) return GV_VECDIST_L2;
    if (strcasecmp(s, "l2") == 0 || strcasecmp(s, "euclidean") == 0)
        return GV_VECDIST_L2;
    if (strcasecmp(s, "cosine") == 0)
        return GV_VECDIST_COSINE;
    if (strcasecmp(s, "dot") == 0 || strcasecmp(s, "dotproduct") == 0)
        return GV_VECDIST_DOT;
    if (strcasecmp(s, "hamming") == 0)
        return GV_VECDIST_HAMMING;
    return GV_VECDIST_L2;
}

/* Parse a comma-separated float vector string into a float array.
 * Returns the number of floats parsed, or 0 on error.
 * Caller must free *out with gv_free(). */
static size_t parse_float_vector(const char *s, float **out) {
    *out = NULL;
    if (!s || !*s) return 0;
    size_t cap = 64, n = 0;
    float *v = (float *)gv_alloc(cap * sizeof(float));
    if (!v) return 0;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end;
        float val = strtof(p, &end);
        if (end == p) break;
        if (n == cap) {
            cap *= 2;
            float *nv = (float *)gv_realloc(v, cap * sizeof(float));
            if (!nv) { gv_free(v); *out = NULL; return 0; }
            v = nv;
        }
        v[n++] = val;
        p = end;
    }
    if (n == 0) { gv_free(v); *out = NULL; return 0; }
    *out = v;
    return n;
}

/* Internal function called by cypher.c OPD_FUNC handler for vector_distance.
 * `stored_str` is the comma-separated stored vector property value.
 * `query_str` is the comma-separated query vector (from $param or literal).
 * `metric` is the distance metric string (NULL or empty for L2 default).
 * Returns a heap-allocated string representation of the distance float. */
char *cypher_eval_vector_distance(const char *stored_str, const char *query_str,
                                  const char *metric) {
    if (!stored_str || !query_str) return gv_dup_cstr("");
    float *stored_vec = NULL, *query_vec = NULL;
    size_t sd = parse_float_vector(stored_str, &stored_vec);
    size_t qd = parse_float_vector(query_str, &query_vec);
    if (sd == 0 || qd == 0) {
        gv_free(stored_vec); gv_free(query_vec);
        return gv_dup_cstr("");
    }
    GV_CypherVecDistType type = cypher_parse_vecdist_type(metric);
    float dist = cypher_vector_distance(stored_vec, sd, query_vec, qd, type);
    gv_free(stored_vec); gv_free(query_vec);
    if (dist < 0.0f) return gv_dup_cstr("");
    /* Use %g for compact representation (matches cypher.c fmt_num style) */
    char buf[48];
    if (dist == (float)(long long)dist && dist < 1e15f && dist > -1e15f)
        snprintf(buf, sizeof(buf), "%lld", (long long)dist);
    else
        snprintf(buf, sizeof(buf), "%g", dist);
    return gv_dup_cstr(buf);
}

/* Registration: for now this is a no-op because vector_distance functions
 * are hardwired into cypher.c's OPD_FUNC evaluation. This function exists
 * so callers can "opt in" and check the return value. Future versions could
 * use a pluggable function registry. */
int cypher_register_vector_functions(struct GV_CypherEngine *ctx, size_t dimension) {
    (void)ctx; (void)dimension;
    return 0;
}
