#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_sparse_index.h"
#include "gigavector/gv_metadata.h"

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
    int *deleted;                /* Deletion flags: 1 if deleted, 0 if active */
};

static int gv_sparse_write_uint32(FILE *out, uint32_t value) {
    return (fwrite(&value, sizeof(uint32_t), 1, out) == 1) ? 0 : -1;
}

static int gv_sparse_write_size(FILE *out, size_t value) {
    return (fwrite(&value, sizeof(size_t), 1, out) == 1) ? 0 : -1;
}

static int gv_sparse_write_floats(FILE *out, const float *data, size_t count) {
    return (data != NULL && fwrite(data, sizeof(float), count, out) == count) ? 0 : -1;
}

static int gv_sparse_write_string(FILE *out, const char *str, uint32_t len) {
    if (gv_sparse_write_uint32(out, len) != 0) {
        return -1;
    }
    if (len == 0) {
        return 0;
    }
    return (fwrite(str, 1, len, out) == len) ? 0 : -1;
}

static int gv_sparse_write_metadata(FILE *out, const GV_Metadata *meta_head) {
    uint32_t count = 0;
    const GV_Metadata *cursor = meta_head;
    while (cursor != NULL) {
        count++;
        cursor = cursor->next;
    }

    if (gv_sparse_write_uint32(out, count) != 0) {
        return -1;
    }

    cursor = meta_head;
    while (cursor != NULL) {
        size_t key_len = strlen(cursor->key);
        size_t val_len = strlen(cursor->value);
        if (key_len > UINT32_MAX || val_len > UINT32_MAX) {
            return -1;
        }
        if (gv_sparse_write_string(out, cursor->key, (uint32_t)key_len) != 0) {
            return -1;
        }
        if (gv_sparse_write_string(out, cursor->value, (uint32_t)val_len) != 0) {
            return -1;
        }
        cursor = cursor->next;
    }
    return 0;
}

static int gv_sparse_read_uint32(FILE *in, uint32_t *value) {
    return (value != NULL && fread(value, sizeof(uint32_t), 1, in) == 1) ? 0 : -1;
}

static int gv_sparse_read_size(FILE *in, size_t *value) {
    return (value != NULL && fread(value, sizeof(size_t), 1, in) == 1) ? 0 : -1;
}

static int gv_sparse_read_floats(FILE *in, float *data, size_t count) {
    return (data != NULL && fread(data, sizeof(float), count, in) == count) ? 0 : -1;
}

static int gv_sparse_read_string(FILE *in, char **out_str, uint32_t len) {
    if (out_str == NULL) {
        return -1;
    }
    *out_str = NULL;
    if (len == 0) {
        *out_str = (char *)malloc(1);
        if (*out_str == NULL) {
            return -1;
        }
        (*out_str)[0] = '\0';
        return 0;
    }

    char *buf = (char *)malloc(len + 1);
    if (buf == NULL) {
        return -1;
    }
    if (fread(buf, 1, len, in) != len) {
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    *out_str = buf;
    return 0;
}

static int gv_sparse_read_metadata(FILE *in, GV_SparseVector *sv) {
    if (sv == NULL) {
        return -1;
    }

    uint32_t count = 0;
    if (gv_sparse_read_uint32(in, &count) != 0) {
        return -1;
    }

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t key_len = 0;
        uint32_t val_len = 0;
        char *key = NULL;
        char *value = NULL;

        if (gv_sparse_read_uint32(in, &key_len) != 0) {
            return -1;
        }
        if (gv_sparse_read_string(in, &key, key_len) != 0) {
            free(key);
            return -1;
        }

        if (gv_sparse_read_uint32(in, &val_len) != 0) {
            free(key);
            return -1;
        }
        if (gv_sparse_read_string(in, &value, val_len) != 0) {
            free(key);
            return -1;
        }

        /* attach metadata onto the sparse vector; treat it as GV_Vector for API */
        if (gv_vector_set_metadata((GV_Vector *)sv, key, value) != 0) {
            free(key);
            free(value);
            return -1;
        }

        free(key);
        free(value);
    }

    return 0;
}

GV_SparseIndex *gv_sparse_index_create(size_t dimension) {
    if (dimension == 0) {
        return NULL;
    }
    GV_SparseIndex *idx = (GV_SparseIndex *)calloc(1, sizeof(GV_SparseIndex));
    if (!idx) return NULL;
    idx->dimension = dimension;
    idx->capacity = 1024;
    idx->vectors = (GV_SparseVector **)calloc(idx->capacity, sizeof(GV_SparseVector *));
    idx->deleted = (int *)calloc(idx->capacity, sizeof(int));
    if (!idx->vectors || !idx->deleted) {
        free(idx->vectors);
        free(idx->deleted);
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
    free(index->deleted);
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
        int *tmp_deleted = (int *)realloc(index->deleted, newcap * sizeof(int));
        if (!tmp_len || !tmp_deleted) {
            /* keep old arrays intact */
            if (tmp_vec) free(tmp_vec);
            if (tmp_len) free(tmp_len);
            if (tmp_deleted) free(tmp_deleted);
            return -1;
        }
        /* zero-init new doc_len and deleted regions */
        memset(tmp_len + index->capacity, 0, (newcap - index->capacity) * sizeof(double));
        memset(tmp_deleted + index->capacity, 0, (newcap - index->capacity) * sizeof(int));
        index->vectors = tmp_vec;
        index->doc_len = tmp_len;
        index->deleted = tmp_deleted;
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
    index->deleted[vid] = 0;

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
                if (index->deleted[vid] != 0) {
                    p = p->next;
                    continue;
                }
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
                size_t vid = p->vector_id;
                if (index->deleted[vid] == 0) {
                    scores[vid] += qv * p->value;
                    touched[vid] = 1;
                }
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
        if (!touched[vid] || index->deleted[vid] != 0) continue;
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

int gv_sparse_index_save(const GV_SparseIndex *index, FILE *out, uint32_t version) {
    (void)version;
    if (index == NULL || out == NULL) {
        return -1;
    }

    for (size_t vid = 0; vid < index->count; ++vid) {
        GV_SparseVector *sv = index->vectors[vid];
        if (sv == NULL) {
            return -1;
        }

        if (gv_sparse_write_size(out, sv->nnz) != 0) {
            return -1;
        }
        for (size_t i = 0; i < sv->nnz; ++i) {
            uint32_t idx = sv->entries[i].index;
            float val = sv->entries[i].value;
            if (gv_sparse_write_uint32(out, idx) != 0) {
                return -1;
            }
            if (gv_sparse_write_floats(out, &val, 1) != 0) {
                return -1;
            }
        }

        if (gv_sparse_write_metadata(out, sv->metadata) != 0) {
            return -1;
        }
    }
    return 0;
}

int gv_sparse_index_load(GV_SparseIndex **index_out, FILE *in,
                         size_t dimension, size_t count, uint32_t version) {
    (void)version;
    if (index_out == NULL || in == NULL || dimension == 0) {
        return -1;
    }

    GV_SparseIndex *idx = gv_sparse_index_create(dimension);
    if (!idx) {
        return -1;
    }

    for (size_t v = 0; v < count; ++v) {
        size_t nnz = 0;
        if (gv_sparse_read_size(in, &nnz) != 0) {
            gv_sparse_index_destroy(idx);
            return -1;
        }

        uint32_t *indices = NULL;
        float *values = NULL;
        if (nnz > 0) {
            indices = (uint32_t *)malloc(nnz * sizeof(uint32_t));
            values = (float *)malloc(nnz * sizeof(float));
            if (!indices || !values) {
                free(indices);
                free(values);
                gv_sparse_index_destroy(idx);
                return -1;
            }
            for (size_t i = 0; i < nnz; ++i) {
                if (gv_sparse_read_uint32(in, &indices[i]) != 0) {
                    free(indices);
                    free(values);
                    gv_sparse_index_destroy(idx);
                    return -1;
                }
                if (gv_sparse_read_floats(in, &values[i], 1) != 0) {
                    free(indices);
                    free(values);
                    gv_sparse_index_destroy(idx);
                    return -1;
                }
            }
        }

        GV_SparseVector *sv = gv_sparse_vector_create(dimension, indices, values, nnz);
        free(indices);
        free(values);
        if (!sv) {
            gv_sparse_index_destroy(idx);
            return -1;
        }

        if (gv_sparse_read_metadata(in, sv) != 0) {
            gv_sparse_vector_destroy(sv);
            gv_sparse_index_destroy(idx);
            return -1;
        }

        if (gv_sparse_index_add(idx, sv) != 0) {
            gv_sparse_vector_destroy(sv);
            gv_sparse_index_destroy(idx);
            return -1;
        }
    }

    *index_out = idx;
    return 0;
}




int gv_sparse_index_delete(GV_SparseIndex *index, size_t vector_index) {
    if (index == NULL || vector_index >= index->count) {
        return -1;
    }

    if (index->deleted[vector_index] != 0) {
        return -1;
    }

    index->deleted[vector_index] = 1;
    return 0;
}
