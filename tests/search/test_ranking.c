#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "search/ranking.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_parse_simple_expression(void) {
    GV_RankExpr *expr = rank_expr_parse("_score");
    ASSERT(expr != NULL, "parsing '_score' should succeed");

    double result = rank_expr_eval(expr, 0.9f, NULL, 0);
    ASSERT(fabs(result - 0.9) < 0.01, "_score should evaluate to the vector_score");

    rank_expr_destroy(expr);
    return 0;
}

static int test_parse_weighted_expression(void) {
    GV_RankExpr *expr = rank_expr_parse("0.7 * _score + 0.3 * popularity");
    ASSERT(expr != NULL, "parsing weighted expression should succeed");

    GV_RankSignal signals[] = {
        {"popularity", 100.0}
    };
    double result = rank_expr_eval(expr, 0.5f, signals, 1);
    /* 0.7 * 0.5 + 0.3 * 100.0 = 0.35 + 30.0 = 30.35 */
    ASSERT(fabs(result - 30.35) < 0.1, "weighted expression should compute correctly");

    rank_expr_destroy(expr);
    return 0;
}

static int test_parse_invalid_expression(void) {
    GV_RankExpr *expr = rank_expr_parse("((( invalid +++");
    ASSERT(expr == NULL, "parsing invalid expression should return NULL");

    rank_expr_destroy(NULL);
    return 0;
}

static int test_create_weighted(void) {
    const char *names[] = {"_score", "freshness"};
    double weights[] = {0.6, 0.4};
    GV_RankExpr *expr = rank_expr_create_weighted(2, names, weights);
    ASSERT(expr != NULL, "create_weighted should succeed");

    GV_RankSignal signals[] = {
        {"_score", 0.0},      /* will be overridden by vector_score */
        {"freshness", 0.8},
    };
    double result = rank_expr_eval(expr, 1.0f, signals, 2);
    /* 0.6 * 1.0 + 0.4 * 0.8 = 0.6 + 0.32 = 0.92 */
    ASSERT(fabs(result - 0.92) < 0.1, "weighted sum should be approximately correct");

    rank_expr_destroy(expr);
    return 0;
}

static int test_eval_with_math_ops(void) {
    GV_RankExpr *expr = rank_expr_parse("max(_score, 0.5)");
    ASSERT(expr != NULL, "parsing max expression should succeed");

    double r1 = rank_expr_eval(expr, 0.3f, NULL, 0);
    ASSERT(fabs(r1 - 0.5) < 0.01, "max(0.3, 0.5) should be 0.5");

    double r2 = rank_expr_eval(expr, 0.8f, NULL, 0);
    ASSERT(fabs(r2 - 0.8) < 0.01, "max(0.8, 0.5) should be 0.8");

    rank_expr_destroy(expr);
    return 0;
}

static int test_eval_multiple_signals(void) {
    GV_RankExpr *expr = rank_expr_parse("_score + price + rating");
    ASSERT(expr != NULL, "parsing expression with multiple signals should succeed");

    GV_RankSignal signals[] = {
        {"price", 10.0},
        {"rating", 4.5},
    };
    double result = rank_expr_eval(expr, 0.5f, signals, 2);
    /* 0.5 + 10.0 + 4.5 = 15.0 */
    ASSERT(fabs(result - 15.0) < 0.1, "sum of signals should be correct");

    rank_expr_destroy(expr);
    return 0;
}

static int test_destroy_null(void) {
    rank_expr_destroy(NULL);
    return 0;
}

static int test_parse_constant_expression(void) {
    GV_RankExpr *expr = rank_expr_parse("42.0");
    ASSERT(expr != NULL, "parsing constant should succeed");

    double result = rank_expr_eval(expr, 0.0f, NULL, 0);
    ASSERT(fabs(result - 42.0) < 0.01, "constant should evaluate to 42.0");

    rank_expr_destroy(expr);
    return 0;
}

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
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
