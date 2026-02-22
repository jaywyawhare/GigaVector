#ifndef GIGAVECTOR_GV_RANKING_H
#define GIGAVECTOR_GV_RANKING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_ranking.h
 * @brief Custom ranking expressions for combining vector similarity with
 *        business-logic signals.
 *
 * Allows users to build configurable ranking formulas that blend the raw
 * vector distance score (_score) with arbitrary per-document signals such
 * as timestamps, popularity counts, prices, or geo-distances.
 *
 * Example expression:
 * @code
 *   "0.7 * _score + 0.3 * decay_exp(timestamp, 1700000000, 86400)"
 * @endcode
 */

/* Enumerations */

/**
 * @brief Ranking operation types used in the expression tree.
 */
typedef enum {
    GV_RANK_ADD          = 0,   /**< Binary addition:  left + right. */
    GV_RANK_MUL          = 1,   /**< Binary multiply:  left * right. */
    GV_RANK_MAX          = 2,   /**< Binary maximum:   max(left, right). */
    GV_RANK_MIN          = 3,   /**< Binary minimum:   min(left, right). */
    GV_RANK_POW          = 4,   /**< Binary power:     pow(left, right). */
    GV_RANK_LOG          = 5,   /**< Unary log:        log(child). */
    GV_RANK_NEG          = 6,   /**< Unary negation:   -child. */
    GV_RANK_CLAMP        = 7,   /**< Ternary clamp:    clamp(child, lo, hi). */
    GV_RANK_LINEAR       = 8,   /**< Linear transform: a * child + b. */
    GV_RANK_DECAY_EXP    = 9,   /**< Exponential decay: exp(-|val-origin|/scale). */
    GV_RANK_DECAY_GAUSS  = 10,  /**< Gaussian decay:    exp(-0.5*((val-origin)/scale)^2). */
    GV_RANK_DECAY_LINEAR = 11   /**< Linear decay:      max(0, 1 - |val-origin|/scale). */
} GV_RankOp;

/* Data Structures */

/**
 * @brief A named signal value supplied per-document at scoring time.
 *
 * Callers build an array of these to pass business-logic data (timestamp,
 * popularity, price, geo-distance, ...) into the ranking expression.
 */
typedef struct {
    const char *name;   /**< Signal name (e.g. "timestamp", "popularity"). */
    double      value;  /**< Numeric value for this document. */
} GV_RankSignal;

/**
 * @brief A single node in the ranking expression tree.
 *
 * Internal recursive structure representing parsed expressions.  Leaf nodes
 * reference either a named signal or a numeric constant; interior nodes
 * combine their children with the operation stored in @c op.
 */
typedef struct GV_RankNode GV_RankNode;
struct GV_RankNode {
    GV_RankOp op;   /**< Operation this node performs. */

    /** Operands -- interpretation depends on @c op. */
    union {
        /** Leaf: named signal (including the built-in "_score"). */
        const char *signal_name;

        /** Leaf: numeric constant. */
        double constant;

        /** Interior: child pointers (up to 3 for clamp). */
        struct {
            GV_RankNode *left;      /**< First child / value operand. */
            GV_RankNode *right;     /**< Second child (NULL for unary ops). */
            GV_RankNode *third;     /**< Third child (only for clamp hi). */
        } children;
    } operand;

    /** Parameters used by linear transform and decay functions. */
    double scale;           /**< Decay scale or linear coefficient 'a'. */
    double offset;          /**< Linear offset 'b'. */
    double decay_origin;    /**< Origin value for decay functions. */
    double decay_scale;     /**< Scale (half-life width) for decay functions. */
};

/**
 * @brief Opaque compiled ranking expression.
 *
 * Created by gv_rank_expr_parse() or gv_rank_expr_create_weighted() and
 * freed with gv_rank_expr_destroy().
 */
typedef struct GV_RankExpr GV_RankExpr;

/**
 * @brief Result entry produced by gv_rank_search().
 */
typedef struct {
    size_t index;           /**< Vector index in the database. */
    float  final_score;     /**< Score after applying the ranking expression. */
    float  vector_score;    /**< Raw vector distance / similarity score. */
} GV_RankedResult;

/* Expression Construction */

/**
 * @brief Parse a human-readable ranking expression string.
 *
 * Supported grammar (recursive descent):
 * @code
 *   expr   := term (('+' | '-') term)*
 *   term   := factor (('*' | '/') factor)*
 *   factor := NUMBER | IDENT | func '(' args ')' | '(' expr ')' | '-' factor
 *   func   := "decay_exp" | "decay_gauss" | "decay_linear"
 *           | "log" | "pow" | "clamp" | "max" | "min" | "linear"
 * @endcode
 *
 * The built-in variable @c _score resolves to the vector similarity score
 * at evaluation time.
 *
 * @param expression  Null-terminated expression string.
 * @return Compiled expression, or NULL on parse error.  Free with
 *         gv_rank_expr_destroy().
 */
GV_RankExpr *gv_rank_expr_parse(const char *expression);

/**
 * @brief Create a simple weighted-sum ranking expression.
 *
 * Equivalent to: w0*signal0 + w1*signal1 + ...
 *
 * @param n             Number of signals.
 * @param signal_names  Array of @p n signal name strings (copied internally).
 * @param weights       Array of @p n weights.
 * @return Compiled expression, or NULL on error.  Free with
 *         gv_rank_expr_destroy().
 */
GV_RankExpr *gv_rank_expr_create_weighted(size_t n, const char **signal_names,
                                          const double *weights);

/* Expression Evaluation */

/**
 * @brief Evaluate a ranking expression for one document.
 *
 * @param expr          Compiled ranking expression.
 * @param vector_score  Raw vector similarity / distance score (bound to
 *                      the @c _score variable).
 * @param signals       Array of per-document signal values.
 * @param signal_count  Number of elements in @p signals.
 * @return Computed ranking score, or 0.0 on error.
 */
double gv_rank_expr_eval(const GV_RankExpr *expr, float vector_score,
                         const GV_RankSignal *signals, size_t signal_count);

/* Expression Lifecycle */

/**
 * @brief Destroy a ranking expression and free all associated memory.
 *
 * Safe to call with NULL.
 *
 * @param expr  Expression to destroy.
 */
void gv_rank_expr_destroy(GV_RankExpr *expr);

/* Ranked Search */

/**
 * @brief Perform a vector search and re-rank results with a custom expression.
 *
 * 1. Oversample: retrieve @p oversample candidates via gv_db_search().
 * 2. Score each candidate by evaluating @p expr with the raw vector distance
 *    and the per-vector signals.
 * 3. Sort by final_score descending and return the top @p k results.
 *
 * @param db                  Database handle (cast to GV_Database* internally).
 * @param query               Query vector.
 * @param dimension           Query vector dimension.
 * @param k                   Number of results to return.
 * @param oversample          Number of candidates to fetch before re-ranking
 *                            (should be >= k; pass 0 to default to k * 4).
 * @param distance_type       Distance metric (GV_DistanceType cast to int).
 * @param expr                Compiled ranking expression.
 * @param per_vector_signals  Signal array laid out contiguously: signals for
 *                            vector i start at offset i * signal_stride.
 * @param signal_stride       Number of GV_RankSignal entries per vector.
 * @param results             Output array of at least @p k elements.
 * @return Number of results written (0 to k), or -1 on error.
 */
int gv_rank_search(const void *db, const float *query, size_t dimension,
                   size_t k, size_t oversample, int distance_type,
                   const GV_RankExpr *expr,
                   const GV_RankSignal *per_vector_signals,
                   size_t signal_stride, GV_RankedResult *results);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_RANKING_H */
