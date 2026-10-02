/* OPQ rotation: orthonormality, PQ distortion reduction, and save/load. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "index/opq.h"
#include "index/pq.h"
#include "core/types.h"
#include "schema/vector.h"
#include "../test_tmp.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

/* Anisotropic + correlated data: variance concentrated in a few latent directions. */
static float *make_data(size_t N, size_t D) {
    float *data = (float *)malloc(N * D * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        float lat[4]; for (int a = 0; a < 4; a++) lat[a] = ((float)rand() / RAND_MAX - 0.5f);
        for (size_t d = 0; d < D; d++) {
            float v = 0; for (int a = 0; a < 4; a++) v += lat[a] * sinf((float)(d * 7 + a * 13));
            v += ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
            data[i * D + d] = v;
        }
    }
    return data;
}

int main(void) {
    srand(11);
    const size_t N = 4000, D = 32, M = 8;
    float *data = make_data(N, D);

    /* rotation is orthonormal -> distance preserving */
    GV_OPQ *o = opq_train(D, M, data, N);
    ASSERT(o != NULL, "opq_train");
    float x[32], y[32];
    for (size_t d = 0; d < D; d++) x[d] = data[7 * D + d];
    opq_rotate(o, x, y);
    double nx = 0, ny = 0; for (size_t d = 0; d < D; d++) { nx += x[d] * x[d]; ny += y[d] * y[d]; }
    ASSERT(fabs(ny / nx - 1.0) < 1e-3, "rotation preserves norm (orthonormal)");
    opq_free(o);

    /* OPQ lowers PQ reconstruction error vs plain PQ */
    GV_PQConfig base = {.m = M, .nbits = 8, .train_iters = 15, .use_opq = 0};
    GV_PQConfig withopq = base; withopq.use_opq = 1;
    void *pb = pq_create(D, &base), *po = pq_create(D, &withopq);
    pq_train(pb, data, N); pq_train(po, data, N);
    for (size_t i = 0; i < N; i++) {
        pq_insert(pb, vector_create_from_data(D, data + i * D));
        pq_insert(po, vector_create_from_data(D, data + i * D));
    }
    double eb = pq_avg_quantization_error(pb), eo = pq_avg_quantization_error(po);
    printf("PQ reconstruction error: base=%.5f OPQ=%.5f\n", eb, eo);
    ASSERT(eo < eb, "OPQ reduces PQ distortion");

    /* save/load round-trips the rotation (reconstruction error unchanged) */
    char opq_path[512];
    gv_test_make_temp_path(opq_path, sizeof(opq_path), "gv_opq", ".bin");
    FILE *f = fopen(opq_path, "wb"); ASSERT(pq_save(po, f, 1) == 0, "save"); fclose(f);
    void *pl = NULL; FILE *g = fopen(opq_path, "rb");
    ASSERT(pq_load(&pl, g, D, 1) == 0 && pl, "load"); fclose(g);
    double el = pq_avg_quantization_error(pl);
    ASSERT(fabs(el - eo) < 1e-4, "OPQ survives save/load");

    pq_destroy(pb); pq_destroy(po); if (pl) pq_destroy(pl);
    free(data); remove(opq_path);
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL OPQ TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
