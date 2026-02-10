#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_ranking.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ---------- test_parse_simple_expression ---------- */
static int test_parse_simple_expression(void) {
    GV_RankExpr *expr = gv_rank_expr_parse("_score");
    ASSERT(expr != NULL, "parsing '_score' should succeed");

    /* Evaluating with vector_score=0.9 should return ~0.9 */
    double result = gv_rank_expr_eval(expr, 0.9f, NULL, 0);
    ASSERT(fabs(result - 0.9) < 0.01, "_score should evaluate to the vector_score");

    gv_rank_expr_destroy(expr);
    return 0;
}

/* ---------- test_parse_weighted_expression ---------- */
static int test_parse_weighted_expression(void) {
    GV_RankExpr *expr = gv_rank_expr_parse("0.7 * _score + 0.3 * popularity");
    ASSERT(expr != NULL, "parsing weighted expression should succeed");

    GV_RankSignal signals[] = {
        {"popularity", 100.0}
    };
    double result = gv_rank_expr_eval(expr, 0.5f, signals, 1);
    /* 0.7 * 0.5 + 0.3 * 100.0 = 0.35 + 30.0 = 30.35 */
    ASSERT(fabs(result - 30.35) < 0.1, "weighted expression should compute correctly");

    gv_rank_expr_destroy(expr);
    return 0;
}

/* ---------- test_parse_invalid_expression ---------- */
static int test_parse_invalid_expression(void) {
    GV_RankExpr *expr = gv_rank_expr_parse("((( invalid +++");
    /* Should return NULL on parse error */
    ASSERT(expr == NULL, "parsing invalid expression should return NULL");

    /* Destroy NULL should be safe */
    gv_rank_expr_destroy(NULL);
    return 0;
}

/* ---------- test_create_weighted ---------- */
static int test_create_weighted(void) {
    const char *names[] = {"_score", "freshness"};
    double weights[] = {0.6, 0.4};
    GV_RankExpr *expr = gv_rank_expr_create_weighted(2, names, weights);
    ASSERT(expr != NULL, "create_weighted should succeed");

    GV_RankSignal signals[] = {
        {"_score", 0.0},      /* will be overridden by vector_score */
        {"freshness", 0.8},
    };
    double result = gv_rank_expr_eval(expr, 1.0f, signals, 2);
    /* 0.6 * 1.0 + 0.4 * 0.8 = 0.6 + 0.32 = 0.92 */
    ASSERT(fabs(result - 0.92) < 0.1, "weighted sum should be approximately correct");

    gv_rank_expr_destroy(expr);
    return 0;
}

/* ---------- test_eval_with_math_ops ---------- */
static int test_eval_with_math_ops(void) {
    GV_RankExpr *expr = gv_rank_expr_parse("max(_score, 0.5)");
    ASSERT(expr != NULL, "parsing max expression should succeed");

    double r1 = gv_rank_expr_eval(expr, 0.3f, NULL, 0);
    ASSERT(fabs(r1 - 0.5) < 0.01, "max(0.3, 0.5) should be 0.5");

    double r2 = gv_rank_expr_eval(expr, 0.8f, NULL, 0);
    ASSERT(fabs(r2 - 0.8) < 0.01, "max(0.8, 0.5) should be 0.8");

    gv_rank_expr_destroy(expr);
    return 0;
}

/* ---------- test_eval_multiple_signals ---------- */
static int test_eval_multiple_signals(void) {
    GV_RankExpr *expr = gv_rank_expr_parse("_score + price + rating");
    ASSERT(expr != NULL, "parsing expression with multiple signals should succeed");

    GV_RankSignal signals[] = {
        {"price", 10.0},
        {"rating", 4.5},
    };
    double result = gv_rank_expr_eval(expr, 0.5f, signals, 2);
    /* 0.5 + 10.0 + 4.5 = 15.0 */
    ASSERT(fabs(result - 15.0) < 0.1, "sum of signals should be correct");

    gv_rank_expr_destroy(expr);
    return 0;
}

/* ---------- test_destroy_null ---------- */
static int test_destroy_null(void) {
    /* Should not crash */
    gv_rank_expr_destroy(NULL);
    return 0;
}

/* ---------- test_parse_constant_expression ---------- */
static int test_parse_constant_expression(void) {
    GV_RankExpr *expr = gv_rank_expr_parse("42.0");
    ASSERT(expr != NULL, "parsing constant should succeed");

    double result = gv_rank_expr_eval(expr, 0.0f, NULL, 0);
    ASSERT(fabs(result - 42.0) < 0.01, "constant should evaluate to 42.0");

    gv_rank_expr_destroy(expr);
    return 0;
}

/* ========== test runner ========== */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing ranking parse simple...",         test_parse_simple_expression},
        {"Testing ranking parse weighted...",       test_parse_weighted_expression},
        {"Testing ranking parse invalid...",        test_parse_invalid_expression},
        {"Testing ranking create weighted...",      test_create_weighted},
        {"Testing ranking eval math ops...",        test_eval_with_math_ops},
        {"Testing ranking eval multiple signals..", test_eval_multiple_signals},
        {"Testing ranking destroy null...",         test_destroy_null},
        {"Testing ranking parse constant...",       test_parse_constant_expression},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        printf("%s", tests[i].name);
        if (tests[i].fn() == 0) { printf(" [OK]\n"); passed++; }
        else { printf(" [FAIL]\n"); }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
