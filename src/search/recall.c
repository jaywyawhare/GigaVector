#include "search/recall.h"
#include "search/distance.h"
#include "storage/soa_storage.h"
#include "core/memory.h"

/* Insert (d, id) into a k-sized ascending-by-distance top-k buffer. */
static void topk_push(float *ds, size_t *ids, size_t *n, size_t k, float d, size_t id) {
    if (*n < k) {
        size_t p = *n;
        while (p > 0 && ds[p - 1] > d) { ds[p] = ds[p - 1]; ids[p] = ids[p - 1]; p--; }
        ds[p] = d; ids[p] = id; (*n)++;
    } else if (k > 0 && d < ds[k - 1]) {
        size_t p = k - 1;
        while (p > 0 && ds[p - 1] > d) { ds[p] = ds[p - 1]; ids[p] = ids[p - 1]; p--; }
        ds[p] = d; ids[p] = id;
    }
}

int db_evaluate_recall(const GV_Database *db, const float *queries, size_t nq,
                       size_t dim, size_t k, GV_DistanceType metric,
                       GV_RecallReport *out) {
    if (!db || !queries || !out || nq == 0 || k == 0 || dim == 0) return -1;
    if (!db->soa_storage) return -1;
    size_t N = soa_storage_count(db->soa_storage);
    if (N == 0) return -1;
    if (k > N) k = N;

    float *gtd = (float *)gv_alloc(k * sizeof(float));
    size_t *gtid = (size_t *)gv_alloc(k * sizeof(size_t));
    GV_SearchResult *ann = (GV_SearchResult *)gv_alloc(k * sizeof(GV_SearchResult));
    if (!gtd || !gtid || !ann) { gv_free(gtd); gv_free(gtid); gv_free(ann); return -1; }

    double acc = 0.0;
    for (size_t q = 0; q < nq; q++) {
        const float *qd = queries + q * dim;
        GV_Vector qv; qv.dimension = dim; qv.data = (float *)qd; qv.metadata = NULL;

        /* Exact ground truth: brute-force top-k over stored vectors. Done before
         * db_search, which resets the TLS arena - our heap buffers are unaffected. */
        size_t gn = 0;
        for (size_t i = 0; i < N; i++) {
            if (soa_storage_is_deleted(db->soa_storage, i)) continue;
            GV_Vector view;
            if (soa_storage_get_vector_view(db->soa_storage, i, &view) != 0) continue;
            float d = distance(&view, &qv, metric);
            if (d < 0.0f) continue;
            topk_push(gtd, gtid, &gn, k, d, i);
        }

        int an = db_search(db, qd, k, ann, metric);

        size_t hit = 0;
        for (size_t i = 0; i < gn; i++)
            for (int j = 0; j < an; j++)
                if (ann[j].id == gtid[i]) { hit++; break; }
        if (an > 0) gv_search_results_free(ann, (size_t)an);

        acc += gn ? (double)hit / (double)gn : 0.0;
    }

    gv_free(gtd); gv_free(gtid); gv_free(ann);
    out->recall = acc / (double)nq;
    out->queries = nq;
    out->k = k;
    return 0;
}
