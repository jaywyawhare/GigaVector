/* Low-precision (fp16/int8) compact vector store: codec fidelity, memory, search. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "storage/compact_store.h"
#include "core/half.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

int main(void) {
    /* fp16 codec fidelity over a typical embedding range */
    float maxerr = 0.0f;
    srand(7);
    for (int i = 0; i < 20000; i++) {
        float f = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        float g = gv_f16_to_f32(gv_f32_to_f16(f));
        float e = fabsf(f - g) / (fabsf(f) + 1e-6f);
        if (e > maxerr) maxerr = e;
    }
    ASSERT(maxerr < 0.001f, "fp16 round-trip near-lossless (<0.1% rel err)");

    const size_t N = 2000, D = 48, K = 10;
    float *v = (float *)malloc(N * D * sizeof(float));
    for (size_t i = 0; i < N * D; i++) v[i] = (float)rand() / RAND_MAX;

    GV_CompactStore *f16 = compact_store_create(D, GV_COMPACT_F16);
    GV_CompactStore *i8 = compact_store_create(D, GV_COMPACT_I8);
    ASSERT(f16 && i8, "create stores");
    for (size_t i = 0; i < N; i++) { compact_store_add(f16, v + i * D); compact_store_add(i8, v + i * D); }
    ASSERT(compact_store_count(f16) == N, "count");

    size_t f32b = N * D * sizeof(float);
    ASSERT(compact_store_memory_bytes(f16) * 2 == f32b, "fp16 = half the RAM of float32");
    ASSERT(compact_store_memory_bytes(i8) < f32b / 3, "int8 < a third the RAM of float32");

    /* nearest neighbour of a stored vector must be itself */
    size_t ids[16]; float ds[16];
    int n = compact_store_search(f16, v, K, GV_DISTANCE_EUCLIDEAN, ids, ds);
    ASSERT(n == (int)K && ids[0] == 0, "fp16 search returns self as nearest");
    n = compact_store_search(i8, v, K, GV_DISTANCE_EUCLIDEAN, ids, ds);
    ASSERT(n == (int)K && ids[0] == 0, "int8 search returns self as nearest");

    /* decode round-trips a stored vector within tolerance */
    float out[48];
    ASSERT(compact_store_get(f16, 5, out) == 0, "decode ok");
    float re = 0; for (size_t d = 0; d < D; d++) re += fabsf(out[d] - v[5 * D + d]);
    ASSERT(re / D < 0.01f, "fp16 decode close to original");

    compact_store_free(f16); compact_store_free(i8); free(v);
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL COMPACT-STORE TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
