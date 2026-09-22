/* Distance metrics — focus on Jaccard/Tanimoto (binary + continuous). */
#include <stdio.h>
#include <math.h>
#include "search/distance.h"
#include "core/types.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static GV_Vector V(float *d, size_t n) { GV_Vector v; v.data = d; v.dimension = n; v.metadata = NULL; return v; }

int main(void) {
    float a[] = {1, 1, 1, 0}, b[] = {0, 1, 1, 1};
    GV_Vector va = V(a, 4), vb = V(b, 4);

    /* binary Jaccard: |A∩B|=2, |A∪B|=4 -> 0.5 */
    ASSERT(fabsf(distance(&va, &vb, GV_DISTANCE_JACCARD) - 0.5f) < 1e-5f, "binary jaccard = 0.5");
    ASSERT(fabsf(distance(&va, &va, GV_DISTANCE_JACCARD)) < 1e-6f, "identical -> 0");
    ASSERT(fabsf(distance_scalar(&va, &vb, GV_DISTANCE_JACCARD) - 0.5f) < 1e-5f, "scalar path matches");

    float z[] = {0, 0, 0, 0}; GV_Vector vz = V(z, 4);
    ASSERT(fabsf(distance(&vz, &vz, GV_DISTANCE_JACCARD)) < 1e-6f, "both-zero (empty sets) -> 0");

    float c[] = {0, 0, 0, 1}, e[] = {1, 0, 0, 0}; GV_Vector vc = V(c, 4), ve = V(e, 4);
    ASSERT(fabsf(distance(&vc, &ve, GV_DISTANCE_JACCARD) - 1.0f) < 1e-5f, "disjoint -> 1");

    /* continuous Tanimoto: dot/(|a|^2+|b|^2-dot) with real values */
    float p[] = {1.0f, 2.0f}, q[] = {2.0f, 1.0f}; GV_Vector vp = V(p, 2), vq = V(q, 2);
    /* dot=4, |p|^2=5, |q|^2=5 -> 4/(10-4)=0.6667 -> dist 0.3333 */
    ASSERT(fabsf(distance(&vp, &vq, GV_DISTANCE_JACCARD) - (1.0f - 4.0f / 6.0f)) < 1e-5f, "continuous tanimoto");

    /* invalid args */
    ASSERT(distance_jaccard(NULL, &vb) < 0.0f, "null -> negative");

    /* sanity: other metrics still route */
    ASSERT(distance(&va, &vb, GV_DISTANCE_EUCLIDEAN) >= 0.0f, "euclidean routes");
    ASSERT(distance(&va, &vb, GV_DISTANCE_HAMMING) == 2.0f, "hamming = 2");

    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL DISTANCE TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
