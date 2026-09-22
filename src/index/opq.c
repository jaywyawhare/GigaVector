#include "index/opq.h"
#include "core/memory.h"

#include <math.h>
#include <string.h>

struct GV_OPQ {
    size_t dim;
    float *R;   /* dim x dim, row-major: rotated[i] = dot(R + i*dim, in) */
};

size_t opq_dimension(const GV_OPQ *o) { return o ? o->dim : 0; }
const float *opq_matrix(const GV_OPQ *o) { return o ? o->R : NULL; }

void opq_free(GV_OPQ *o) {
    if (!o) return;
    gv_free(o->R);
    gv_free(o);
}

void opq_rotate(const GV_OPQ *o, const float *in, float *out) {
    if (!o || !in || !out) return;
    size_t d = o->dim;
    for (size_t i = 0; i < d; i++) {
        const float *row = o->R + i * d;
        float acc = 0.0f;
        for (size_t j = 0; j < d; j++) acc += row[j] * in[j];
        out[i] = acc;
    }
}

GV_OPQ *opq_from_matrix(size_t dim, const float *matrix) {
    if (dim == 0 || !matrix) return NULL;
    GV_OPQ *o = (GV_OPQ *)gv_calloc(1, sizeof(*o));
    if (!o) return NULL;
    o->dim = dim;
    o->R = (float *)gv_alloc(dim * dim * sizeof(float));
    if (!o->R) { gv_free(o); return NULL; }
    memcpy(o->R, matrix, dim * dim * sizeof(float));
    return o;
}

/* Cyclic Jacobi eigen-decomposition of a symmetric dim×dim matrix A (in place).
 * On return A's diagonal holds eigenvalues and V holds eigenvectors as columns. */
static void jacobi_eigen(double *A, double *V, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) V[i * n + j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (size_t p = 0; p < n; p++)
            for (size_t q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
        if (off < 1e-12) break;

        for (size_t p = 0; p < n; p++) {
            for (size_t q = p + 1; q < n; q++) {
                double apq = A[p * n + q];
                if (fabs(apq) < 1e-15) continue;
                double app = A[p * n + p], aqq = A[q * n + q];
                double phi = 0.5 * atan2(2.0 * apq, aqq - app);
                double c = cos(phi), s = sin(phi);
                for (size_t i = 0; i < n; i++) {
                    double aip = A[i * n + p], aiq = A[i * n + q];
                    A[i * n + p] = c * aip - s * aiq;
                    A[i * n + q] = s * aip + c * aiq;
                }
                for (size_t i = 0; i < n; i++) {
                    double api = A[p * n + i], aqi = A[q * n + i];
                    A[p * n + i] = c * api - s * aqi;
                    A[q * n + i] = s * api + c * aqi;
                }
                for (size_t i = 0; i < n; i++) {
                    double vip = V[i * n + p], viq = V[i * n + q];
                    V[i * n + p] = c * vip - s * viq;
                    V[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
}

GV_OPQ *opq_train(size_t dim, size_t m, const float *data, size_t count) {
    if (dim == 0 || count == 0 || !data) return NULL;
    if (m == 0) m = 1;

    double *cov = (double *)gv_calloc(dim * dim, sizeof(double));
    double *mean = (double *)gv_calloc(dim, sizeof(double));
    double *V = (double *)gv_alloc(dim * dim * sizeof(double));
    GV_OPQ *o = (GV_OPQ *)gv_calloc(1, sizeof(*o));
    if (!cov || !mean || !V || !o) { gv_free(cov); gv_free(mean); gv_free(V); gv_free(o); return NULL; }
    o->dim = dim;
    o->R = (float *)gv_alloc(dim * dim * sizeof(float));
    if (!o->R) { gv_free(cov); gv_free(mean); gv_free(V); gv_free(o); return NULL; }

    for (size_t i = 0; i < count; i++)
        for (size_t d = 0; d < dim; d++) mean[d] += data[i * dim + d];
    for (size_t d = 0; d < dim; d++) mean[d] /= (double)count;

    /* Centered covariance (symmetric). */
    for (size_t i = 0; i < count; i++) {
        const float *x = data + i * dim;
        for (size_t a = 0; a < dim; a++) {
            double xa = x[a] - mean[a];
            for (size_t b = a; b < dim; b++) cov[a * dim + b] += xa * (x[b] - mean[b]);
        }
    }
    for (size_t a = 0; a < dim; a++)
        for (size_t b = a; b < dim; b++) {
            cov[a * dim + b] /= (double)count;
            cov[b * dim + a] = cov[a * dim + b];
        }

    jacobi_eigen(cov, V, dim);   /* eigenvalues -> cov diagonal; eigenvectors -> V columns */

    /* Sort axes by eigenvalue descending. */
    size_t *order = (size_t *)gv_alloc(dim * sizeof(size_t));
    if (!order) { gv_free(cov); gv_free(mean); gv_free(V); gv_free(o->R); gv_free(o); return NULL; }
    for (size_t i = 0; i < dim; i++) order[i] = i;
    for (size_t i = 0; i + 1 < dim; i++)
        for (size_t j = i + 1; j < dim; j++)
            if (cov[order[j] * dim + order[j]] > cov[order[i] * dim + order[i]]) {
                size_t t = order[i]; order[i] = order[j]; order[j] = t;
            }

    /* Balanced allocation: greedily assign each axis (high variance first) to the
     * sub-quantizer with the smallest running log-variance sum, filling each to
     * exactly dim/m axes. This is the parametric-OPQ variance-balancing step. */
    size_t per = dim / m;
    if (per == 0) per = dim;                 /* m > dim: degenerate, one bucket */
    size_t nbucket = per ? dim / per : 1;
    if (nbucket == 0) nbucket = 1;
    double *blog = (double *)gv_calloc(nbucket, sizeof(double));
    size_t *bcnt = (size_t *)gv_calloc(nbucket, sizeof(size_t));
    size_t *axis_of_slot = (size_t *)gv_alloc(dim * sizeof(size_t));
    if (!blog || !bcnt || !axis_of_slot) {
        gv_free(blog); gv_free(bcnt); gv_free(axis_of_slot);
        gv_free(order); gv_free(cov); gv_free(mean); gv_free(V);
        gv_free(o->R); gv_free(o);
        return NULL;
    }
    for (size_t i = 0; i < dim; i++) {
        size_t ax = order[i];
        double var = cov[ax * dim + ax];
        double lv = log(var > 1e-12 ? var : 1e-12);
        /* pick the least-loaded non-full bucket */
        long best = -1;
        for (size_t b = 0; b < nbucket; b++) {
            if (bcnt[b] >= per) continue;
            if (best < 0 || blog[b] < blog[best]) best = (long)b;
        }
        if (best < 0) best = 0;
        size_t slot = (size_t)best * per + bcnt[best];
        if (slot < dim) axis_of_slot[slot] = ax;
        blog[best] += lv;
        bcnt[best]++;
    }

    /* R row i = the eigenvector for the axis placed in slot i (V column axis). */
    for (size_t i = 0; i < dim; i++) {
        size_t ax = axis_of_slot[i];
        for (size_t j = 0; j < dim; j++) o->R[i * dim + j] = (float)V[j * dim + ax];
    }

    gv_free(cov); gv_free(mean); gv_free(V);
    gv_free(order); gv_free(blog); gv_free(bcnt); gv_free(axis_of_slot);
    return o;
}
