/**
 * @file gv_ranking.c
 * @brief Custom ranking expressions implementation.
 *
 * Provides a recursive-descent parser for ranking expression strings, an
 * expression-tree evaluator, decay scoring functions, and the top-level
 * gv_rank_search() that oversamples with gv_db_search and re-ranks.
 */

#include "gigavector/gv_ranking.h"
#include "gigavector/gv_database.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <float.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

/** Default oversample multiplier when caller passes oversample == 0. */
#define GV_RANK_DEFAULT_OVERSAMPLE_FACTOR 4

/** Maximum identifier / signal name length in parsed expressions. */
#define GV_RANK_MAX_IDENT 128

/* ============================================================================
 * Opaque Expression Type
 * ============================================================================ */

struct GV_RankExpr {
    GV_RankNode *root;  /**< Root of the expression tree. */
};

/* ============================================================================
 * Node Helpers (Internal)
 * ============================================================================ */

/** Sentinel op value used for leaf nodes that hold a signal reference. */
#define GV_RANK_OP_SIGNAL  ((GV_RankOp)100)

/** Sentinel op value used for leaf nodes that hold a numeric constant. */
#define GV_RANK_OP_CONST   ((GV_RankOp)101)

/** Sentinel for binary subtraction (parsed as ADD with negated right). */
#define GV_RANK_OP_SUB     ((GV_RankOp)102)

/** Sentinel for binary division (parsed as MUL with reciprocal right). */
#define GV_RANK_OP_DIV     ((GV_RankOp)103)

static GV_RankNode *node_alloc(void) {
    GV_RankNode *n = calloc(1, sizeof(GV_RankNode));
    return n;
}

static GV_RankNode *node_const(double value) {
    GV_RankNode *n = node_alloc();
    if (!n) return NULL;
    n->op = GV_RANK_OP_CONST;
    n->operand.constant = value;
    return n;
}

static GV_RankNode *node_signal(const char *name) {
    GV_RankNode *n = node_alloc();
    if (!n) return NULL;
    n->op = GV_RANK_OP_SIGNAL;
    n->operand.signal_name = strdup(name);
    if (!n->operand.signal_name) {
        free(n);
        return NULL;
    }
    return n;
}

static GV_RankNode *node_binary(GV_RankOp op, GV_RankNode *left, GV_RankNode *right) {
    GV_RankNode *n = node_alloc();
    if (!n) return NULL;
    n->op = op;
    n->operand.children.left = left;
    n->operand.children.right = right;
    return n;
}

static GV_RankNode *node_unary(GV_RankOp op, GV_RankNode *child) {
    GV_RankNode *n = node_alloc();
    if (!n) return NULL;
    n->op = op;
    n->operand.children.left = child;
    return n;
}

static void node_free(GV_RankNode *n) {
    if (!n) return;

    if (n->op == GV_RANK_OP_SIGNAL) {
        free((void *)n->operand.signal_name);
    } else if (n->op != GV_RANK_OP_CONST) {
        /* Interior node -- free children recursively. */
        node_free(n->operand.children.left);
        node_free(n->operand.children.right);
        node_free(n->operand.children.third);
    }
    free(n);
}

/* ============================================================================
 * Lexer (Internal)
 * ============================================================================ */

typedef enum {
    TOK_NUM,        /**< Numeric literal. */
    TOK_IDENT,      /**< Identifier / signal name. */
    TOK_PLUS,       /**< '+' */
    TOK_MINUS,      /**< '-' */
    TOK_STAR,       /**< '*' */
    TOK_SLASH,      /**< '/' */
    TOK_LPAREN,     /**< '(' */
    TOK_RPAREN,     /**< ')' */
    TOK_COMMA,      /**< ',' */
    TOK_EOF,        /**< End of input. */
    TOK_ERROR       /**< Lexer error. */
} TokenType;

typedef struct {
    TokenType type;
    double    num_val;
    char      ident[GV_RANK_MAX_IDENT];
} Token;

typedef struct {
    const char *src;
    size_t      pos;
    Token       cur;
    int         has_error;
} Parser;

static void skip_ws(Parser *p) {
    while (p->src[p->pos] && isspace((unsigned char)p->src[p->pos])) {
        p->pos++;
    }
}

static void next_token(Parser *p) {
    skip_ws(p);

    char c = p->src[p->pos];
    if (c == '\0') {
        p->cur.type = TOK_EOF;
        return;
    }

    /* Single-character tokens. */
    switch (c) {
        case '+': p->cur.type = TOK_PLUS;   p->pos++; return;
        case '-': p->cur.type = TOK_MINUS;  p->pos++; return;
        case '*': p->cur.type = TOK_STAR;   p->pos++; return;
        case '/': p->cur.type = TOK_SLASH;  p->pos++; return;
        case '(': p->cur.type = TOK_LPAREN; p->pos++; return;
        case ')': p->cur.type = TOK_RPAREN; p->pos++; return;
        case ',': p->cur.type = TOK_COMMA;  p->pos++; return;
        default: break;
    }

    /* Numeric literal. */
    if (isdigit((unsigned char)c) || c == '.') {
        char *end = NULL;
        double val = strtod(p->src + p->pos, &end);
        if (end == p->src + p->pos) {
            p->cur.type = TOK_ERROR;
            p->has_error = 1;
            return;
        }
        p->cur.type = TOK_NUM;
        p->cur.num_val = val;
        p->pos = (size_t)(end - p->src);
        return;
    }

    /* Identifier: [a-zA-Z_][a-zA-Z0-9_]* */
    if (isalpha((unsigned char)c) || c == '_') {
        size_t start = p->pos;
        while (isalnum((unsigned char)p->src[p->pos]) || p->src[p->pos] == '_') {
            p->pos++;
        }
        size_t len = p->pos - start;
        if (len >= GV_RANK_MAX_IDENT) len = GV_RANK_MAX_IDENT - 1;
        memcpy(p->cur.ident, p->src + start, len);
        p->cur.ident[len] = '\0';
        p->cur.type = TOK_IDENT;
        return;
    }

    p->cur.type = TOK_ERROR;
    p->has_error = 1;
}

/* ============================================================================
 * Recursive-Descent Parser (Internal)
 * ============================================================================ */

/* Forward declarations for mutual recursion. */
static GV_RankNode *parse_expr(Parser *p);
static GV_RankNode *parse_term(Parser *p);
static GV_RankNode *parse_factor(Parser *p);

/**
 * @brief Parse a function call: func '(' arg (',' arg)* ')'
 *
 * Supported functions:
 *   decay_exp(signal, origin, scale)
 *   decay_gauss(signal, origin, scale)
 *   decay_linear(signal, origin, scale)
 *   log(expr)
 *   pow(base, exponent)
 *   clamp(expr, lo, hi)
 *   max(a, b)
 *   min(a, b)
 *   linear(expr, a, b)  -- computes a*expr + b
 */
static GV_RankNode *parse_func(Parser *p, const char *name) {
    /* Consume '(' */
    if (p->cur.type != TOK_LPAREN) {
        p->has_error = 1;
        return NULL;
    }
    next_token(p);

    /* ---------- decay functions ---------- */
    if (strcmp(name, "decay_exp") == 0 ||
        strcmp(name, "decay_gauss") == 0 ||
        strcmp(name, "decay_linear") == 0) {

        GV_RankOp op;
        if (strcmp(name, "decay_exp") == 0)        op = GV_RANK_DECAY_EXP;
        else if (strcmp(name, "decay_gauss") == 0)  op = GV_RANK_DECAY_GAUSS;
        else                                        op = GV_RANK_DECAY_LINEAR;

        /* First arg: value expression (typically a signal). */
        GV_RankNode *val = parse_expr(p);
        if (!val || p->has_error) { node_free(val); return NULL; }

        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(val); return NULL; }
        next_token(p);

        /* Second arg: origin. */
        GV_RankNode *origin_node = parse_expr(p);
        if (!origin_node || p->has_error) { node_free(val); node_free(origin_node); return NULL; }

        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(val); node_free(origin_node); return NULL; }
        next_token(p);

        /* Third arg: scale. */
        GV_RankNode *scale_node = parse_expr(p);
        if (!scale_node || p->has_error) { node_free(val); node_free(origin_node); node_free(scale_node); return NULL; }

        if (p->cur.type != TOK_RPAREN) { p->has_error = 1; node_free(val); node_free(origin_node); node_free(scale_node); return NULL; }
        next_token(p);

        /* Build decay node.  Store origin and scale as constants when possible. */
        GV_RankNode *n = node_alloc();
        if (!n) { node_free(val); node_free(origin_node); node_free(scale_node); return NULL; }
        n->op = op;
        n->operand.children.left  = val;
        n->operand.children.right = origin_node;
        n->operand.children.third = scale_node;

        /* Cache numeric constants into the parameter fields for fast eval. */
        if (origin_node->op == GV_RANK_OP_CONST) n->decay_origin = origin_node->operand.constant;
        if (scale_node->op == GV_RANK_OP_CONST)  n->decay_scale  = scale_node->operand.constant;

        return n;
    }

    /* ---------- log(expr) ---------- */
    if (strcmp(name, "log") == 0) {
        GV_RankNode *child = parse_expr(p);
        if (!child || p->has_error) { node_free(child); return NULL; }
        if (p->cur.type != TOK_RPAREN) { p->has_error = 1; node_free(child); return NULL; }
        next_token(p);
        return node_unary(GV_RANK_LOG, child);
    }

    /* ---------- pow(base, exp) ---------- */
    if (strcmp(name, "pow") == 0) {
        GV_RankNode *base = parse_expr(p);
        if (!base || p->has_error) { node_free(base); return NULL; }
        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(base); return NULL; }
        next_token(p);
        GV_RankNode *exponent = parse_expr(p);
        if (!exponent || p->has_error) { node_free(base); node_free(exponent); return NULL; }
        if (p->cur.type != TOK_RPAREN) { p->has_error = 1; node_free(base); node_free(exponent); return NULL; }
        next_token(p);
        return node_binary(GV_RANK_POW, base, exponent);
    }

    /* ---------- clamp(expr, lo, hi) ---------- */
    if (strcmp(name, "clamp") == 0) {
        GV_RankNode *child = parse_expr(p);
        if (!child || p->has_error) { node_free(child); return NULL; }
        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(child); return NULL; }
        next_token(p);
        GV_RankNode *lo = parse_expr(p);
        if (!lo || p->has_error) { node_free(child); node_free(lo); return NULL; }
        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(child); node_free(lo); return NULL; }
        next_token(p);
        GV_RankNode *hi = parse_expr(p);
        if (!hi || p->has_error) { node_free(child); node_free(lo); node_free(hi); return NULL; }
        if (p->cur.type != TOK_RPAREN) { p->has_error = 1; node_free(child); node_free(lo); node_free(hi); return NULL; }
        next_token(p);

        GV_RankNode *n = node_alloc();
        if (!n) { node_free(child); node_free(lo); node_free(hi); return NULL; }
        n->op = GV_RANK_CLAMP;
        n->operand.children.left  = child;
        n->operand.children.right = lo;
        n->operand.children.third = hi;
        return n;
    }

    /* ---------- max(a, b) ---------- */
    if (strcmp(name, "max") == 0) {
        GV_RankNode *a = parse_expr(p);
        if (!a || p->has_error) { node_free(a); return NULL; }
        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(a); return NULL; }
        next_token(p);
        GV_RankNode *b = parse_expr(p);
        if (!b || p->has_error) { node_free(a); node_free(b); return NULL; }
        if (p->cur.type != TOK_RPAREN) { p->has_error = 1; node_free(a); node_free(b); return NULL; }
        next_token(p);
        return node_binary(GV_RANK_MAX, a, b);
    }

    /* ---------- min(a, b) ---------- */
    if (strcmp(name, "min") == 0) {
        GV_RankNode *a = parse_expr(p);
        if (!a || p->has_error) { node_free(a); return NULL; }
        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(a); return NULL; }
        next_token(p);
        GV_RankNode *b = parse_expr(p);
        if (!b || p->has_error) { node_free(a); node_free(b); return NULL; }
        if (p->cur.type != TOK_RPAREN) { p->has_error = 1; node_free(a); node_free(b); return NULL; }
        next_token(p);
        return node_binary(GV_RANK_MIN, a, b);
    }

    /* ---------- linear(expr, a, b) -> a*expr + b ---------- */
    if (strcmp(name, "linear") == 0) {
        GV_RankNode *child = parse_expr(p);
        if (!child || p->has_error) { node_free(child); return NULL; }
        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(child); return NULL; }
        next_token(p);
        GV_RankNode *a_node = parse_expr(p);
        if (!a_node || p->has_error) { node_free(child); node_free(a_node); return NULL; }
        if (p->cur.type != TOK_COMMA) { p->has_error = 1; node_free(child); node_free(a_node); return NULL; }
        next_token(p);
        GV_RankNode *b_node = parse_expr(p);
        if (!b_node || p->has_error) { node_free(child); node_free(a_node); node_free(b_node); return NULL; }
        if (p->cur.type != TOK_RPAREN) { p->has_error = 1; node_free(child); node_free(a_node); node_free(b_node); return NULL; }
        next_token(p);

        GV_RankNode *n = node_alloc();
        if (!n) { node_free(child); node_free(a_node); node_free(b_node); return NULL; }
        n->op = GV_RANK_LINEAR;
        n->operand.children.left  = child;
        n->operand.children.right = a_node;
        n->operand.children.third = b_node;

        /* Cache coefficients when they are constants. */
        if (a_node->op == GV_RANK_OP_CONST) n->scale  = a_node->operand.constant;
        if (b_node->op == GV_RANK_OP_CONST) n->offset = b_node->operand.constant;

        return n;
    }

    /* Unknown function. */
    p->has_error = 1;
    return NULL;
}

/**
 * @brief Parse a factor (highest precedence).
 *
 * factor := NUMBER | IDENT | func '(' args ')' | '(' expr ')' | '-' factor
 */
static GV_RankNode *parse_factor(Parser *p) {
    if (p->has_error) return NULL;

    /* Unary minus. */
    if (p->cur.type == TOK_MINUS) {
        next_token(p);
        GV_RankNode *child = parse_factor(p);
        if (!child) return NULL;
        return node_unary(GV_RANK_NEG, child);
    }

    /* Parenthesized sub-expression. */
    if (p->cur.type == TOK_LPAREN) {
        next_token(p);
        GV_RankNode *inner = parse_expr(p);
        if (!inner || p->has_error) { node_free(inner); return NULL; }
        if (p->cur.type != TOK_RPAREN) { p->has_error = 1; node_free(inner); return NULL; }
        next_token(p);
        return inner;
    }

    /* Numeric literal. */
    if (p->cur.type == TOK_NUM) {
        GV_RankNode *n = node_const(p->cur.num_val);
        next_token(p);
        return n;
    }

    /* Identifier: either a function call or a signal reference. */
    if (p->cur.type == TOK_IDENT) {
        char name[GV_RANK_MAX_IDENT];
        memcpy(name, p->cur.ident, GV_RANK_MAX_IDENT);
        next_token(p);

        /* Function call? */
        if (p->cur.type == TOK_LPAREN) {
            return parse_func(p, name);
        }

        /* Signal reference (includes the built-in _score). */
        return node_signal(name);
    }

    /* Unexpected token. */
    p->has_error = 1;
    return NULL;
}

/**
 * @brief Parse a term: factor (('*' | '/') factor)*
 */
static GV_RankNode *parse_term(Parser *p) {
    GV_RankNode *left = parse_factor(p);
    if (!left || p->has_error) return left;

    while (p->cur.type == TOK_STAR || p->cur.type == TOK_SLASH) {
        TokenType op_tok = p->cur.type;
        next_token(p);

        GV_RankNode *right = parse_factor(p);
        if (!right || p->has_error) {
            node_free(left);
            node_free(right);
            return NULL;
        }

        if (op_tok == TOK_STAR) {
            left = node_binary(GV_RANK_MUL, left, right);
        } else {
            /* Division: represent as MUL(left, POW(right, -1)). */
            GV_RankNode *neg_one = node_const(-1.0);
            GV_RankNode *recip = node_binary(GV_RANK_POW, right, neg_one);
            if (!recip) { node_free(left); return NULL; }
            left = node_binary(GV_RANK_MUL, left, recip);
        }
        if (!left) return NULL;
    }

    return left;
}

/**
 * @brief Parse an expression: term (('+' | '-') term)*
 */
static GV_RankNode *parse_expr(Parser *p) {
    GV_RankNode *left = parse_term(p);
    if (!left || p->has_error) return left;

    while (p->cur.type == TOK_PLUS || p->cur.type == TOK_MINUS) {
        TokenType op_tok = p->cur.type;
        next_token(p);

        GV_RankNode *right = parse_term(p);
        if (!right || p->has_error) {
            node_free(left);
            node_free(right);
            return NULL;
        }

        if (op_tok == TOK_PLUS) {
            left = node_binary(GV_RANK_ADD, left, right);
        } else {
            /* Subtraction: ADD(left, NEG(right)). */
            GV_RankNode *neg = node_unary(GV_RANK_NEG, right);
            if (!neg) { node_free(left); return NULL; }
            left = node_binary(GV_RANK_ADD, left, neg);
        }
        if (!left) return NULL;
    }

    return left;
}

/* ============================================================================
 * Expression Construction (Public API)
 * ============================================================================ */

GV_RankExpr *gv_rank_expr_parse(const char *expression) {
    if (!expression) return NULL;

    Parser parser;
    memset(&parser, 0, sizeof(parser));
    parser.src = expression;
    parser.pos = 0;
    parser.has_error = 0;

    /* Prime the lexer. */
    next_token(&parser);

    GV_RankNode *root = parse_expr(&parser);

    if (parser.has_error || !root) {
        node_free(root);
        return NULL;
    }

    /* Ensure we consumed the entire input. */
    if (parser.cur.type != TOK_EOF) {
        node_free(root);
        return NULL;
    }

    GV_RankExpr *expr = calloc(1, sizeof(GV_RankExpr));
    if (!expr) {
        node_free(root);
        return NULL;
    }
    expr->root = root;
    return expr;
}

GV_RankExpr *gv_rank_expr_create_weighted(size_t n, const char **signal_names,
                                          const double *weights) {
    if (n == 0 || !signal_names || !weights) return NULL;

    /* Build: w0*s0 + w1*s1 + ... */
    GV_RankNode *sum = NULL;

    for (size_t i = 0; i < n; i++) {
        if (!signal_names[i]) {
            node_free(sum);
            return NULL;
        }

        GV_RankNode *w = node_const(weights[i]);
        GV_RankNode *s = node_signal(signal_names[i]);
        if (!w || !s) {
            node_free(w);
            node_free(s);
            node_free(sum);
            return NULL;
        }

        GV_RankNode *term = node_binary(GV_RANK_MUL, w, s);
        if (!term) {
            node_free(sum);
            return NULL;
        }

        if (!sum) {
            sum = term;
        } else {
            sum = node_binary(GV_RANK_ADD, sum, term);
            if (!sum) return NULL;
        }
    }

    GV_RankExpr *expr = calloc(1, sizeof(GV_RankExpr));
    if (!expr) {
        node_free(sum);
        return NULL;
    }
    expr->root = sum;
    return expr;
}

/* ============================================================================
 * Expression Lifecycle
 * ============================================================================ */

void gv_rank_expr_destroy(GV_RankExpr *expr) {
    if (!expr) return;
    node_free(expr->root);
    free(expr);
}

/* ============================================================================
 * Signal Lookup (Internal)
 * ============================================================================ */

/**
 * @brief Look up a signal value by name from the signal array.
 *
 * @return The signal value, or 0.0 if not found.
 */
static double lookup_signal(const char *name, float vector_score,
                            const GV_RankSignal *signals, size_t signal_count) {
    /* Built-in: _score */
    if (strcmp(name, "_score") == 0) {
        return (double)vector_score;
    }

    if (signals) {
        for (size_t i = 0; i < signal_count; i++) {
            if (signals[i].name && strcmp(signals[i].name, name) == 0) {
                return signals[i].value;
            }
        }
    }

    return 0.0;
}

/* ============================================================================
 * Decay Functions (Internal)
 * ============================================================================ */

/**
 * @brief Exponential decay: exp(-|val - origin| / scale).
 */
static double decay_exp(double val, double origin, double scale) {
    if (scale <= 0.0) return 0.0;
    return exp(-fabs(val - origin) / scale);
}

/**
 * @brief Gaussian decay: exp(-0.5 * ((val - origin) / scale)^2).
 */
static double decay_gauss(double val, double origin, double scale) {
    if (scale <= 0.0) return 0.0;
    double z = (val - origin) / scale;
    return exp(-0.5 * z * z);
}

/**
 * @brief Linear decay: max(0, 1 - |val - origin| / scale).
 */
static double decay_linear(double val, double origin, double scale) {
    if (scale <= 0.0) return 0.0;
    double d = fabs(val - origin) / scale;
    return d >= 1.0 ? 0.0 : 1.0 - d;
}

/* ============================================================================
 * Expression Tree Evaluation (Internal)
 * ============================================================================ */

static double eval_node(const GV_RankNode *n, float vector_score,
                        const GV_RankSignal *signals, size_t signal_count) {
    if (!n) return 0.0;

    switch ((int)n->op) {

    /* ---- Leaf: constant ---- */
    case GV_RANK_OP_CONST:
        return n->operand.constant;

    /* ---- Leaf: signal reference ---- */
    case GV_RANK_OP_SIGNAL:
        return lookup_signal(n->operand.signal_name, vector_score,
                             signals, signal_count);

    /* ---- Binary arithmetic ---- */
    case GV_RANK_ADD: {
        double l = eval_node(n->operand.children.left, vector_score, signals, signal_count);
        double r = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        return l + r;
    }
    case GV_RANK_MUL: {
        double l = eval_node(n->operand.children.left, vector_score, signals, signal_count);
        double r = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        return l * r;
    }
    case GV_RANK_MAX: {
        double l = eval_node(n->operand.children.left, vector_score, signals, signal_count);
        double r = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        return l > r ? l : r;
    }
    case GV_RANK_MIN: {
        double l = eval_node(n->operand.children.left, vector_score, signals, signal_count);
        double r = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        return l < r ? l : r;
    }
    case GV_RANK_POW: {
        double l = eval_node(n->operand.children.left, vector_score, signals, signal_count);
        double r = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        return pow(l, r);
    }

    /* ---- Unary ---- */
    case GV_RANK_LOG: {
        double v = eval_node(n->operand.children.left, vector_score, signals, signal_count);
        return v > 0.0 ? log(v) : 0.0;
    }
    case GV_RANK_NEG: {
        double v = eval_node(n->operand.children.left, vector_score, signals, signal_count);
        return -v;
    }

    /* ---- Clamp(child, lo, hi) ---- */
    case GV_RANK_CLAMP: {
        double v  = eval_node(n->operand.children.left,  vector_score, signals, signal_count);
        double lo = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        double hi = eval_node(n->operand.children.third, vector_score, signals, signal_count);
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    /* ---- Linear: a * child + b ---- */
    case GV_RANK_LINEAR: {
        double v = eval_node(n->operand.children.left,  vector_score, signals, signal_count);
        double a = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        double b = eval_node(n->operand.children.third, vector_score, signals, signal_count);
        return a * v + b;
    }

    /* ---- Decay functions ---- */
    case GV_RANK_DECAY_EXP: {
        double val    = eval_node(n->operand.children.left,  vector_score, signals, signal_count);
        double origin = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        double sc     = eval_node(n->operand.children.third, vector_score, signals, signal_count);
        return decay_exp(val, origin, sc);
    }
    case GV_RANK_DECAY_GAUSS: {
        double val    = eval_node(n->operand.children.left,  vector_score, signals, signal_count);
        double origin = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        double sc     = eval_node(n->operand.children.third, vector_score, signals, signal_count);
        return decay_gauss(val, origin, sc);
    }
    case GV_RANK_DECAY_LINEAR: {
        double val    = eval_node(n->operand.children.left,  vector_score, signals, signal_count);
        double origin = eval_node(n->operand.children.right, vector_score, signals, signal_count);
        double sc     = eval_node(n->operand.children.third, vector_score, signals, signal_count);
        return decay_linear(val, origin, sc);
    }

    default:
        break;
    }

    return 0.0;
}

/* ============================================================================
 * Expression Evaluation (Public API)
 * ============================================================================ */

double gv_rank_expr_eval(const GV_RankExpr *expr, float vector_score,
                         const GV_RankSignal *signals, size_t signal_count) {
    if (!expr || !expr->root) return 0.0;
    return eval_node(expr->root, vector_score, signals, signal_count);
}

/* ============================================================================
 * Ranked Search
 * ============================================================================ */

/** Internal candidate used during re-ranking. */
typedef struct {
    size_t index;
    float  vector_score;
    double final_score;
} RankCandidate;

static int compare_ranked_desc(const void *a, const void *b) {
    const RankCandidate *ca = (const RankCandidate *)a;
    const RankCandidate *cb = (const RankCandidate *)b;
    if (cb->final_score > ca->final_score) return 1;
    if (cb->final_score < ca->final_score) return -1;
    return 0;
}

int gv_rank_search(const void *db, const float *query, size_t dimension,
                   size_t k, size_t oversample, int distance_type,
                   const GV_RankExpr *expr,
                   const GV_RankSignal *per_vector_signals,
                   size_t signal_stride, GV_RankedResult *results) {
    if (!db || !query || k == 0 || !results || !expr) return -1;

    const GV_Database *database = (const GV_Database *)db;
    (void)dimension; /* dimension is carried by the database internally. */

    /* Determine oversample count. */
    size_t fetch_k = oversample > 0 ? oversample : k * GV_RANK_DEFAULT_OVERSAMPLE_FACTOR;
    if (fetch_k < k) fetch_k = k;

    /* Allocate search results buffer. */
    GV_SearchResult *search_results = malloc(fetch_k * sizeof(GV_SearchResult));
    if (!search_results) return -1;
    memset(search_results, 0, fetch_k * sizeof(GV_SearchResult));

    /* Step 1: Oversample with gv_db_search. */
    int found = gv_db_search(database, query, fetch_k, search_results,
                             (GV_DistanceType)distance_type);
    if (found < 0) {
        free(search_results);
        return -1;
    }
    if (found == 0) {
        free(search_results);
        return 0;
    }

    /* Step 2: Score each candidate. */
    RankCandidate *candidates = malloc((size_t)found * sizeof(RankCandidate));
    if (!candidates) {
        free(search_results);
        return -1;
    }

    for (int i = 0; i < found; i++) {
        /*
         * Convert distance to a similarity score for the expression.
         * GigaVector stores distances where lower = more similar, so we
         * convert: similarity = 1 / (1 + distance).
         */
        float dist = search_results[i].distance;
        float similarity = 1.0f / (1.0f + dist);

        candidates[i].index = (size_t)i;
        candidates[i].vector_score = similarity;

        /* Locate per-vector signals for this candidate. */
        const GV_RankSignal *signals = NULL;
        size_t sig_count = 0;
        if (per_vector_signals && signal_stride > 0) {
            signals = per_vector_signals + (size_t)i * signal_stride;
            sig_count = signal_stride;
        }

        candidates[i].final_score = gv_rank_expr_eval(expr, similarity,
                                                       signals, sig_count);
    }

    free(search_results);

    /* Step 3: Sort by final_score descending. */
    qsort(candidates, (size_t)found, sizeof(RankCandidate), compare_ranked_desc);

    /* Step 4: Copy top-k into output. */
    size_t result_count = (size_t)found < k ? (size_t)found : k;
    for (size_t i = 0; i < result_count; i++) {
        results[i].index        = candidates[i].index;
        results[i].final_score  = (float)candidates[i].final_score;
        results[i].vector_score = candidates[i].vector_score;
    }

    free(candidates);

    return (int)result_count;
}
