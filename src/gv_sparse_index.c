#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_sparse_index.h"
#include "gigavector/gv_utils.h"

typedef struct GV_SparsePosting {
    size_t vector_id;
    float value;
    struct GV_SparsePosting *next;
} GV_SparsePosting;

struct GV_SparseIndex {
    size_t dimension;
    size_t count;
    size_t capacity;
    GV_SparseVector **vectors;
    GV_SparsePosting **postings; /* array of length dimension */
    double *df;                  /* document frequency per dimension */
    double *doc_len;             /* document length per vector (sum of values) */
    double avg_doc_len;
    int use_bm25;                /* non-zero to use BM25 scoring */
};

GV_SparseIndex *gv_sparse_index_create(size_t dimension) {
    if (dimension == 0) {
        return NULL;
    }
    GV_SparseIndex *idx = (GV_SparseIndex *)calloc(1, sizeof(GV_SparseIndex));
    if (!idx) return NULL;
    idx->dimension = dimension;
    idx->capacity = 1024;
    idx->vectors = (GV_SparseVector **)calloc(idx->capacity, sizeof(GV_SparseVector *));
    if (!idx->vectors) {
        free(idx);
        return NULL;
    }
    idx->postings = (GV_SparsePosting **)calloc(dimension, sizeof(GV_SparsePosting *));
    if (!idx->postings) {
        free(idx->vectors);
        free(idx);
        return NULL;
    }
    idx->df = (double *)calloc(dimension, sizeof(double));
    if (!idx->df) {
        free(idx->postings);
        free(idx->vectors);
        free(idx);
        return NULL;
    }
    idx->doc_len = (double *)calloc(idx->capacity, sizeof(double));
    if (!idx->doc_len) {
        free(idx->df);
        free(idx->postings);
        free(idx->vectors);
        free(idx);
        return NULL;
    }
    idx->avg_doc_len = 0.0;
    idx->use_bm25 = 1; /* enable BM25 weighting by default */
    return idx;
}

static void gv_sparse_postings_free(GV_SparsePosting **postings, size_t dimension) {
    if (!postings) return;
    for (size_t i = 0; i < dimension; ++i) {
        GV_SparsePosting *p = postings[i];
        while (p) {
            GV_SparsePosting *n = p->next;
            free(p);
            p = n;
        }
    }
}

void gv_sparse_index_destroy(GV_SparseIndex *index) {
    if (!index) return;
    gv_sparse_postings_free(index->postings, index->dimension);
    free(index->postings);
    free(index->df);
    free(index->doc_len);
    if (index->vectors) {
        for (size_t i = 0; i < index->count; ++i) {
            gv_sparse_vector_destroy(index->vectors[i]);
        }
    }
    free(index->vectors);
    free(index);
}

int gv_sparse_index_add(GV_SparseIndex *index, GV_SparseVector *vector) {
    if (!index || !vector) {
        return -1;
    }
    if (vector->dimension != index->dimension) {
        return -1;
    }
    if (index->count == index->capacity) {
        size_t newcap = index->capacity * 2;
        GV_SparseVector **tmp_vec =
            (GV_SparseVector **)realloc(index->vectors, newcap * sizeof(GV_SparseVector *));
        if (!tmp_vec) {
            return -1;
        }
        double *tmp_len = (double *)realloc(index->doc_len, newcap * sizeof(double));
        if (!tmp_len) {
            /* keep old arrays intact */
            index->vectors = tmp_vec; /* already reallocated; but failure is unlikely */
            return -1;
        }
        /* zero-init new doc_len region */
        memset(tmp_len + index->capacity, 0, (newcap - index->capacity) * sizeof(double));
        index->vectors = tmp_vec;
        index->doc_len = tmp_len;
        index->capacity = newcap;
    }

    size_t vid = index->count;
    index->vectors[vid] = vector;

    /* compute document length (sum of values) and update postings/df */
    double dl = 0.0;
    for (size_t i = 0; i < vector->nnz; ++i) {
        uint32_t dim = vector->entries[i].index;
        float val = vector->entries[i].value;
        if (dim >= index->dimension) {
            continue;
        }
        dl += (double)val;

        GV_SparsePosting *p = (GV_SparsePosting *)malloc(sizeof(GV_SparsePosting));
        if (!p) {
            return -1;
        }
        p->vector_id = vid;
        p->value = val;
        p->next = index->postings[dim];
        index->postings[dim] = p;

        /* update document frequency for this dimension */
        index->df[dim] += 1.0;
    }
    index->doc_len[vid] = dl > 0.0 ? dl : 0.0;

    /* update average document length incrementally */
    index->count++;
    if (index->count == 1) {
        index->avg_doc_len = index->doc_len[vid];
    } else {
        index->avg_doc_len =
            ((index->avg_doc_len * (double)(index->count - 1)) + index->doc_len[vid]) /
            (double)index->count;
    }

    return 0;
}

int gv_sparse_index_search(const GV_SparseIndex *index, const GV_SparseVector *query,
                           size_t k, GV_SearchResult *results, GV_DistanceType distance_type) {
    if (!index || !query || !results || k == 0) {
        return -1;
    }
    if (query->dimension != index->dimension) {
        return -1;
    }
    if (index->count == 0) {
        return 0;
    }

    if (k > index->count) {
        k = index->count;
    }

    float *scores = (float *)calloc(index->count, sizeof(float));
    int *touched = (int *)calloc(index->count, sizeof(int));
    if (!scores || !touched) {
        free(scores);
        free(touched);
        return -1;
    }

    for (size_t i = 0; i < query->nnz; ++i) {
        uint32_t dim = query->entries[i].index;
        float qv = query->entries[i].value;
        if (dim >= index->dimension) continue;
        GV_SparsePosting *p = index->postings[dim];

        if (index->use_bm25) {
            /* BM25-style scoring: treat value as term frequency */
            double df = index->df[dim];
            double N = (double)index->count;
            if (df <= 0.0 || N <= 0.0) {
                continue;
            }
            /* BM25+ style idf (always positive) */
            double idf = log((N - df + 0.5) / (df + 0.5) + 1.0);
            const double k1 = 1.5;
            const double b = 0.75;
            double avgdl = index->avg_doc_len > 0.0 ? index->avg_doc_len : 1.0;

            while (p) {
                size_t vid = p->vector_id;
                double tf = (double)p->value;
                double dl = index->doc_len[vid] > 0.0 ? index->doc_len[vid] : avgdl;
                double denom = tf + k1 * (1.0 - b + b * dl / avgdl);
                if (denom > 0.0) {
                    double w = idf * (tf * (k1 + 1.0) / denom);
                    scores[vid] += (float)(qv * w);
                    touched[vid] = 1;
                }
                p = p->next;
            }
        } else {
            /* plain dot-product scoring */
            while (p) {
                scores[p->vector_id] += qv * p->value;
                touched[p->vector_id] = 1;
                p = p->next;
            }
        }
    }

    for (size_t i = 0; i < k; ++i) {
        results[i].vector = NULL;
        results[i].sparse_vector = NULL;
        results[i].is_sparse = 1;
        results[i].distance = -FLT_MAX;
    }

    size_t filled = 0;
    for (size_t vid = 0; vid < index->count; ++vid) {
        if (!touched[vid]) continue;
        float score = scores[vid];
        float dist;
        if (distance_type == GV_DISTANCE_DOT_PRODUCT || distance_type == GV_DISTANCE_COSINE) {
            dist = -score; /* higher dot → smaller distance */
        } else {
            continue; /* unsupported */
        }

        if (filled < k) {
            results[filled].sparse_vector = index->vectors[vid];
            results[filled].vector = NULL;
            results[filled].is_sparse = 1;
            results[filled].distance = dist;
            filled++;
            for (size_t j = filled; j > 0 && j > 1; --j) {
                if (results[j - 1].distance < results[j - 2].distance) {
                    GV_SearchResult tmp = results[j - 1];
                    results[j - 1] = results[j - 2];
                    results[j - 2] = tmp;
                } else {
                    break;
                }
            }
        } else if (dist < results[k - 1].distance) {
            results[k - 1].sparse_vector = index->vectors[vid];
            results[k - 1].vector = NULL;
            results[k - 1].is_sparse = 1;
            results[k - 1].distance = dist;
            for (size_t j = k; j > 0 && j > 1; --j) {
                if (results[j - 1].distance < results[j - 2].distance) {
                    GV_SearchResult tmp = results[j - 1];
                    results[j - 1] = results[j - 2];
                    results[j - 2] = tmp;
                } else {
                    break;
                }
            }
        }
    }

    free(scores);
    free(touched);
    return (int)filled;
}



