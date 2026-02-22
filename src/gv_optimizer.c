#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "gigavector/gv_optimizer.h"

/*  Internal structure  */
struct GV_QueryOptimizer {
    GV_CollectionStats stats;

    /* Running averages updated via exponential moving average */
    double avg_latency_us;
    double avg_recall;
    size_t sample_count;

    /* Heuristic thresholds ------------------------------------------------- */
    size_t exact_scan_threshold;       /* Use exact scan below this count      */
    double selective_filter_threshold;  /* Selectivity below this -> oversample */
    double ema_alpha;                  /* Exponential moving average alpha      */
    size_t ef_search_cap;              /* Hard cap on ef_search                */
    size_t nprobe_cap;                 /* Hard cap on nprobe                   */
};

/*  Helpers  */

static double log2_safe(double x)
{
    if (x <= 1.0)
        return 1.0;
    return log(x) / log(2.0);
}

static size_t size_max(size_t a, size_t b)
{
    return a > b ? a : b;
}

static size_t size_min(size_t a, size_t b)
{
    return a < b ? a : b;
}

/*  ef_search recommendation (used internally and in public API)  */
static size_t compute_ef_search(const GV_QueryOptimizer *opt, size_t k)
{
    /* Base ef = max(k * 2, 50) */
    size_t ef = size_max(k * 2, 50);

    /* High-dimensional boost */
    if (opt->stats.dimension > 256) {
        ef = (size_t)((double)ef * 1.5);
    }

    /* Large collection clamp */
    if (opt->stats.total_vectors > 100000) {
        ef = size_min(ef, 200);
    }

    /* Hard cap */
    ef = size_min(ef, opt->ef_search_cap);

    return ef;
}

/*  nprobe recommendation (used internally and in public API)  */
static size_t compute_nprobe(const GV_QueryOptimizer *opt)
{
    size_t n = opt->stats.total_vectors;
    size_t nprobe;

    if (n < 10000) {
        nprobe = size_max(4, n / 1000);
    } else if (n < 100000) {
        nprobe = size_max(8, n / 5000);
    } else {
        nprobe = size_max(16, n / 20000);
    }

    /* Hard cap */
    nprobe = size_min(nprobe, opt->nprobe_cap);

    return nprobe;
}

/*  Cost estimation helpers  */
static double estimate_exact_scan_cost(const GV_CollectionStats *st)
{
    return (double)st->total_vectors * (double)st->dimension;
}

static double estimate_index_cost(const GV_CollectionStats *st, size_t k, size_t ef)
{
    double log_n = log2_safe((double)st->total_vectors);
    return (double)k * (double)ef * (double)st->dimension * log_n;
}

/*  Public API  */

GV_QueryOptimizer *gv_optimizer_create(void)
{
    GV_QueryOptimizer *opt = (GV_QueryOptimizer *)calloc(1, sizeof(GV_QueryOptimizer));
    if (!opt)
        return NULL;

    /* Sensible defaults */
    opt->exact_scan_threshold      = 1000;
    opt->selective_filter_threshold = 0.01;
    opt->ema_alpha                 = 0.1;
    opt->ef_search_cap             = 500;
    opt->nprobe_cap                = 128;

    return opt;
}

void gv_optimizer_destroy(GV_QueryOptimizer *opt)
{
    free(opt);
}

void gv_optimizer_update_stats(GV_QueryOptimizer *opt, const GV_CollectionStats *stats)
{
    if (!opt || !stats)
        return;
    memcpy(&opt->stats, stats, sizeof(GV_CollectionStats));
}

int gv_optimizer_plan(const GV_QueryOptimizer *opt, size_t k,
                      int has_filter, double filter_selectivity,
                      GV_QueryPlan *plan)
{
    if (!opt || !plan)
        return -1;

    memset(plan, 0, sizeof(GV_QueryPlan));

    /* ---- Rule 1: small collection -> exact scan ---- */
    if (opt->stats.total_vectors <= opt->exact_scan_threshold) {
        plan->strategy         = GV_PLAN_EXACT_SCAN;
        plan->estimated_recall = 1.0;
        plan->estimated_cost   = estimate_exact_scan_cost(&opt->stats);
        plan->ef_search        = 0;
        plan->nprobe           = 0;
        plan->rerank_top       = 0;
        plan->use_metadata_index = 0;
        plan->oversample_k    = 0;

        snprintf(plan->explanation, sizeof(plan->explanation),
                 "Exact scan: collection has %zu vectors (<= %zu threshold)",
                 opt->stats.total_vectors, opt->exact_scan_threshold);
        return 0;
    }

    /* ---- Rule 2: very selective filter -> oversample + post-filter ---- */
    if (has_filter && filter_selectivity > 0.0 &&
        filter_selectivity < opt->selective_filter_threshold) {

        size_t oversample = (size_t)((double)k / filter_selectivity * 1.5);
        /* Clamp oversample so it never exceeds total vectors */
        oversample = size_min(oversample, opt->stats.total_vectors);

        plan->strategy           = GV_PLAN_OVERSAMPLE_FILTER;
        plan->oversample_k       = oversample;
        plan->ef_search          = compute_ef_search(opt, oversample);
        plan->nprobe             = compute_nprobe(opt);
        plan->rerank_top         = oversample;
        plan->use_metadata_index = 1;
        plan->estimated_recall   = 0.95; /* typically high with oversample */

        /* Cost: index search for oversample_k, then linear filter */
        plan->estimated_cost = estimate_index_cost(&opt->stats, oversample,
                                                   plan->ef_search) +
                               (double)oversample * (double)opt->stats.dimension;

        snprintf(plan->explanation, sizeof(plan->explanation),
                 "Oversample+filter: selectivity %.4f < %.4f, "
                 "fetching %zu candidates for k=%zu",
                 filter_selectivity, opt->selective_filter_threshold,
                 oversample, k);
        return 0;
    }

    /* ---- Rule 3: default index search ---- */
    {
        size_t ef     = compute_ef_search(opt, k);
        size_t nprobe = compute_nprobe(opt);

        plan->strategy           = GV_PLAN_INDEX_SEARCH;
        plan->ef_search          = ef;
        plan->nprobe             = nprobe;
        plan->rerank_top         = k;
        plan->use_metadata_index = has_filter ? 1 : 0;
        plan->oversample_k      = 0;

        plan->estimated_cost   = estimate_index_cost(&opt->stats, k, ef);

        /* Recall estimate: higher ef -> better recall, rough heuristic */
        {
            double ef_ratio = (double)ef / (double)size_max(k, 1);
            plan->estimated_recall = 1.0 - 1.0 / (1.0 + ef_ratio);
            if (plan->estimated_recall > 1.0)
                plan->estimated_recall = 1.0;
        }

        snprintf(plan->explanation, sizeof(plan->explanation),
                 "Index search: %zu vectors, ef_search=%zu, nprobe=%zu, "
                 "est. recall=%.2f%s",
                 opt->stats.total_vectors, ef, nprobe,
                 plan->estimated_recall,
                 has_filter ? " (with metadata pre-filter)" : "");
    }

    return 0;
}

void gv_optimizer_record_result(GV_QueryOptimizer *opt, const GV_QueryPlan *plan,
                                 uint64_t actual_latency_us, double actual_recall)
{
    if (!opt || !plan)
        return;

    (void)plan; /* plan info can be used for per-strategy tracking in future */

    double alpha = opt->ema_alpha;

    if (opt->sample_count == 0) {
        /* First sample: initialise directly */
        opt->avg_latency_us = (double)actual_latency_us;
        opt->avg_recall     = actual_recall;
    } else {
        /* Exponential moving average */
        opt->avg_latency_us = alpha * (double)actual_latency_us +
                              (1.0 - alpha) * opt->avg_latency_us;
        opt->avg_recall     = alpha * actual_recall +
                              (1.0 - alpha) * opt->avg_recall;
    }

    opt->sample_count++;
}

size_t gv_optimizer_recommend_ef_search(const GV_QueryOptimizer *opt, size_t k)
{
    if (!opt)
        return size_max(k * 2, 50);
    return compute_ef_search(opt, k);
}

size_t gv_optimizer_recommend_nprobe(const GV_QueryOptimizer *opt, size_t k)
{
    (void)k; /* nprobe is independent of k in current heuristic */
    if (!opt)
        return 8;
    return compute_nprobe(opt);
}
