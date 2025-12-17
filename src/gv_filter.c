#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_filter.h"
#include "gigavector/gv_metadata.h"

typedef enum {
    GV_FILTER_NODE_COMPARISON,
    GV_FILTER_NODE_AND,
    GV_FILTER_NODE_OR,
    GV_FILTER_NODE_NOT
} GV_FilterNodeType;

typedef enum {
    GV_FILTER_OP_EQ,
    GV_FILTER_OP_NE,
    GV_FILTER_OP_LT,
    GV_FILTER_OP_LE,
    GV_FILTER_OP_GT,
    GV_FILTER_OP_GE,
    GV_FILTER_OP_CONTAINS,
    GV_FILTER_OP_PREFIX
} GV_FilterOp;

typedef struct GV_FilterNode {
    GV_FilterNodeType type;
    struct GV_FilterNode *left;
    struct GV_FilterNode *right;
    struct GV_FilterNode *child; /* for NOT */
    char *key;
    char *value;
    double numeric_value;
    int is_numeric;
    GV_FilterOp op;
} GV_FilterNode;

struct GV_Filter {
    GV_FilterNode *root;
};

typedef struct {
    const char *input;
    size_t pos;
} GV_FilterLexer;

typedef enum {
    TOK_EOF,
    TOK_IDENT,
    TOK_NUMBER,
    TOK_STRING,
    TOK_AND,
    TOK_OR,
    TOK_NOT,
    TOK_CONTAINS,
    TOK_PREFIX,
    TOK_EQ,
    TOK_NE,
    TOK_LT,
    TOK_LE,
    TOK_GT,
    TOK_GE,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_ERROR
} GV_FilterTokenType;

typedef struct {
    GV_FilterTokenType type;
    char *text;
} GV_FilterToken;

static void gv_filter_free_token(GV_FilterToken *tok) {
    if (tok && tok->text) {
        free(tok->text);
        tok->text = NULL;
    }
}

static void gv_filter_lexer_skip_ws(GV_FilterLexer *lx) {
    while (lx->input[lx->pos] != '\0' &&
           (lx->input[lx->pos] == ' ' || lx->input[lx->pos] == '\t' ||
            lx->input[lx->pos] == '\n' || lx->input[lx->pos] == '\r')) {
        lx->pos++;
    }
}

static int gv_filter_match_kw(const char *s, size_t len, const char *kw) {
    size_t kwlen = strlen(kw);
    if (len != kwlen) return 0;
    for (size_t i = 0; i < len; ++i) {
        if ((char)tolower((unsigned char)s[i]) != (char)tolower((unsigned char)kw[i])) {
            return 0;
        }
    }
    return 1;
}

static GV_FilterToken gv_filter_lexer_next(GV_FilterLexer *lx) {
    GV_FilterToken tok;
    tok.type = TOK_EOF;
    tok.text = NULL;

    gv_filter_lexer_skip_ws(lx);
    char c = lx->input[lx->pos];
    if (c == '\0') {
        tok.type = TOK_EOF;
        return tok;
    }

    if (isalpha((unsigned char)c) || c == '_') {
        size_t start = lx->pos;
        lx->pos++;
        while (isalnum((unsigned char)lx->input[lx->pos]) || lx->input[lx->pos] == '_' || lx->input[lx->pos] == '.') {
            lx->pos++;
        }
        size_t len = lx->pos - start;
        char *text = (char *)malloc(len + 1);
        if (!text) {
            tok.type = TOK_ERROR;
            return tok;
        }
        memcpy(text, lx->input + start, len);
        text[len] = '\0';

        if (gv_filter_match_kw(text, len, "AND")) {
            free(text);
            tok.type = TOK_AND;
        } else if (gv_filter_match_kw(text, len, "OR")) {
            free(text);
            tok.type = TOK_OR;
        } else if (gv_filter_match_kw(text, len, "NOT")) {
            free(text);
            tok.type = TOK_NOT;
        } else if (gv_filter_match_kw(text, len, "CONTAINS")) {
            free(text);
            tok.type = TOK_CONTAINS;
        } else if (gv_filter_match_kw(text, len, "PREFIX")) {
            free(text);
            tok.type = TOK_PREFIX;
        } else {
            tok.type = TOK_IDENT;
            tok.text = text;
        }
        return tok;
    }

    if (c == '"' || c == '\'') {
        char quote = c;
        lx->pos++;
        size_t start = lx->pos;
        while (lx->input[lx->pos] != '\0' && lx->input[lx->pos] != quote) {
            if (lx->input[lx->pos] == '\\' && lx->input[lx->pos + 1] != '\0') {
                lx->pos += 2;
            } else {
                lx->pos++;
            }
        }
        if (lx->input[lx->pos] != quote) {
            tok.type = TOK_ERROR;
            return tok;
        }
        size_t len = lx->pos - start;
        char *text = (char *)malloc(len + 1);
        if (!text) {
            tok.type = TOK_ERROR;
            return tok;
        }
        memcpy(text, lx->input + start, len);
        text[len] = '\0';
        lx->pos++; /* consume closing quote */

        tok.type = TOK_STRING;
        tok.text = text;
        return tok;
    }

    if (c == '-' || isdigit((unsigned char)c)) {
        size_t start = lx->pos;
        lx->pos++;
        while (isdigit((unsigned char)lx->input[lx->pos]) || lx->input[lx->pos] == '.') {
            lx->pos++;
        }
        size_t len = lx->pos - start;
        char *text = (char *)malloc(len + 1);
        if (!text) {
            tok.type = TOK_ERROR;
            return tok;
        }
        memcpy(text, lx->input + start, len);
        text[len] = '\0';
        tok.type = TOK_NUMBER;
        tok.text = text;
        return tok;
    }

    if (c == '=' && lx->input[lx->pos + 1] == '=') {
        lx->pos += 2;
        tok.type = TOK_EQ;
        return tok;
    }
    if (c == '!' && lx->input[lx->pos + 1] == '=') {
        lx->pos += 2;
        tok.type = TOK_NE;
        return tok;
    }
    if (c == '<') {
        if (lx->input[lx->pos + 1] == '=') {
            lx->pos += 2;
            tok.type = TOK_LE;
        } else {
            lx->pos++;
            tok.type = TOK_LT;
        }
        return tok;
    }
    if (c == '>') {
        if (lx->input[lx->pos + 1] == '=') {
            lx->pos += 2;
            tok.type = TOK_GE;
        } else {
            lx->pos++;
            tok.type = TOK_GT;
        }
        return tok;
    }
    if (c == '(') {
        lx->pos++;
        tok.type = TOK_LPAREN;
        return tok;
    }
    if (c == ')') {
        lx->pos++;
        tok.type = TOK_RPAREN;
        return tok;
    }

    lx->pos++;
    tok.type = TOK_ERROR;
    return tok;
}

typedef struct {
    GV_FilterLexer lexer;
    GV_FilterToken current;
} GV_FilterParser;

static void gv_filter_parser_advance(GV_FilterParser *p) {
    gv_filter_free_token(&p->current);
    p->current = gv_filter_lexer_next(&p->lexer);
}

static int gv_filter_expect(GV_FilterParser *p, GV_FilterTokenType type) {
    if (p->current.type != type) {
        return 0;
    }
    return 1;
}

static GV_FilterNode *gv_filter_node_new(GV_FilterNodeType type) {
    GV_FilterNode *n = (GV_FilterNode *)calloc(1, sizeof(GV_FilterNode));
    if (!n) return NULL;
    n->type = type;
    return n;
}

static void gv_filter_node_free(GV_FilterNode *node) {
    if (!node) return;
    gv_filter_node_free(node->left);
    gv_filter_node_free(node->right);
    gv_filter_node_free(node->child);
    free(node->key);
    free(node->value);
    free(node);
}

static GV_FilterNode *gv_filter_parse_expr(GV_FilterParser *p);

static GV_FilterNode *gv_filter_parse_primary(GV_FilterParser *p) {
    if (p->current.type == TOK_LPAREN) {
        gv_filter_parser_advance(p);
        GV_FilterNode *node = gv_filter_parse_expr(p);
        if (!node) return NULL;
        if (!gv_filter_expect(p, TOK_RPAREN)) {
            gv_filter_node_free(node);
            return NULL;
        }
        gv_filter_parser_advance(p);
        return node;
    }

    if (p->current.type != TOK_IDENT) {
        return NULL;
    }
    char *key = p->current.text;
    p->current.text = NULL;
    gv_filter_parser_advance(p);

    GV_FilterOp op;
    if (p->current.type == TOK_EQ) {
        op = GV_FILTER_OP_EQ;
    } else if (p->current.type == TOK_NE) {
        op = GV_FILTER_OP_NE;
    } else if (p->current.type == TOK_LT) {
        op = GV_FILTER_OP_LT;
    } else if (p->current.type == TOK_LE) {
        op = GV_FILTER_OP_LE;
    } else if (p->current.type == TOK_GT) {
        op = GV_FILTER_OP_GT;
    } else if (p->current.type == TOK_GE) {
        op = GV_FILTER_OP_GE;
    } else if (p->current.type == TOK_CONTAINS) {
        op = GV_FILTER_OP_CONTAINS;
    } else if (p->current.type == TOK_PREFIX) {
        op = GV_FILTER_OP_PREFIX;
    } else {
        free(key);
        return NULL;
    }
    GV_FilterTokenType optype = p->current.type;
    gv_filter_parser_advance(p);

    if (optype == TOK_CONTAINS || optype == TOK_PREFIX) {
        if (p->current.type != TOK_STRING && p->current.type != TOK_IDENT) {
            free(key);
            return NULL;
        }
        char *val = p->current.text;
        p->current.text = NULL;
        gv_filter_parser_advance(p);

        GV_FilterNode *node = gv_filter_node_new(GV_FILTER_NODE_COMPARISON);
        if (!node) {
            free(key);
            free(val);
            return NULL;
        }
        node->key = key;
        node->value = val;
        node->is_numeric = 0;
        node->op = op;
        return node;
    }

    if (p->current.type != TOK_STRING && p->current.type != TOK_NUMBER && p->current.type != TOK_IDENT) {
        free(key);
        return NULL;
    }
    char *val = p->current.text;
    p->current.text = NULL;
    GV_FilterTokenType vtype = p->current.type;
    gv_filter_parser_advance(p);

    GV_FilterNode *node = gv_filter_node_new(GV_FILTER_NODE_COMPARISON);
    if (!node) {
        free(key);
        free(val);
        return NULL;
    }
    node->key = key;
    node->value = val;
    node->op = op;
    if (vtype == TOK_NUMBER) {
        char *endptr = NULL;
        node->numeric_value = strtod(node->value, endptr ? &endptr : NULL);
        node->is_numeric = 1;
    } else {
        node->is_numeric = 0;
    }
    return node;
}

static GV_FilterNode *gv_filter_parse_not(GV_FilterParser *p) {
    if (p->current.type == TOK_NOT) {
        gv_filter_parser_advance(p);
        GV_FilterNode *child = gv_filter_parse_not(p);
        if (!child) return NULL;
        GV_FilterNode *node = gv_filter_node_new(GV_FILTER_NODE_NOT);
        if (!node) {
            gv_filter_node_free(child);
            return NULL;
        }
        node->child = child;
        return node;
    }
    return gv_filter_parse_primary(p);
}

static GV_FilterNode *gv_filter_parse_and(GV_FilterParser *p) {
    GV_FilterNode *left = gv_filter_parse_not(p);
    if (!left) return NULL;
    while (p->current.type == TOK_AND) {
        gv_filter_parser_advance(p);
        GV_FilterNode *right = gv_filter_parse_not(p);
        if (!right) {
            gv_filter_node_free(left);
            return NULL;
        }
        GV_FilterNode *node = gv_filter_node_new(GV_FILTER_NODE_AND);
        if (!node) {
            gv_filter_node_free(left);
            gv_filter_node_free(right);
            return NULL;
        }
        node->left = left;
        node->right = right;
        left = node;
    }
    return left;
}

static GV_FilterNode *gv_filter_parse_expr(GV_FilterParser *p) {
    GV_FilterNode *left = gv_filter_parse_and(p);
    if (!left) return NULL;
    while (p->current.type == TOK_OR) {
        gv_filter_parser_advance(p);
        GV_FilterNode *right = gv_filter_parse_and(p);
        if (!right) {
            gv_filter_node_free(left);
            return NULL;
        }
        GV_FilterNode *node = gv_filter_node_new(GV_FILTER_NODE_OR);
        if (!node) {
            gv_filter_node_free(left);
            gv_filter_node_free(right);
            return NULL;
        }
        node->left = left;
        node->right = right;
        left = node;
    }
    return left;
}

GV_Filter *gv_filter_parse(const char *expr) {
    if (expr == NULL) {
        return NULL;
    }
    GV_FilterParser parser;
    parser.lexer.input = expr;
    parser.lexer.pos = 0;
    parser.current.type = TOK_EOF;
    parser.current.text = NULL;
    gv_filter_parser_advance(&parser);

    GV_FilterNode *root = gv_filter_parse_expr(&parser);
    if (!root || parser.current.type != TOK_EOF) {
        gv_filter_node_free(root);
        gv_filter_free_token(&parser.current);
        return NULL;
    }
    gv_filter_free_token(&parser.current);

    GV_Filter *filter = (GV_Filter *)malloc(sizeof(GV_Filter));
    if (!filter) {
        gv_filter_node_free(root);
        return NULL;
    }
    filter->root = root;
    return filter;
}

static int gv_filter_eval_node(const GV_FilterNode *node, const GV_Vector *vector) {
    if (!node || !vector) {
        return -1;
    }
    switch (node->type) {
    case GV_FILTER_NODE_AND: {
        int l = gv_filter_eval_node(node->left, vector);
        if (l <= 0) return l;
        int r = gv_filter_eval_node(node->right, vector);
        if (r < 0) return r;
        return l && r;
    }
    case GV_FILTER_NODE_OR: {
        int l = gv_filter_eval_node(node->left, vector);
        if (l < 0) return l;
        if (l == 1) return 1;
        int r = gv_filter_eval_node(node->right, vector);
        if (r < 0) return r;
        return r ? 1 : 0;
    }
    case GV_FILTER_NODE_NOT: {
        int v = gv_filter_eval_node(node->child, vector);
        if (v < 0) return v;
        return v ? 0 : 1;
    }
    case GV_FILTER_NODE_COMPARISON: {
        const char *meta_val = gv_vector_get_metadata(vector, node->key);
        if (!meta_val) {
            return 0;
        }
        if (node->op == GV_FILTER_OP_CONTAINS) {
            if (!node->value) return 0;
            return strstr(meta_val, node->value) != NULL;
        }
        if (node->op == GV_FILTER_OP_PREFIX) {
            if (!node->value) return 0;
            size_t want = strlen(node->value);
            return strncmp(meta_val, node->value, want) == 0;
        }
        if (node->is_numeric) {
            char *endptr = NULL;
            double v = strtod(meta_val, &endptr);
            if (endptr == meta_val) {
                return 0;
            }
            double ref = node->numeric_value;
            switch (node->op) {
            case GV_FILTER_OP_EQ: return v == ref;
            case GV_FILTER_OP_NE: return v != ref;
            case GV_FILTER_OP_LT: return v < ref;
            case GV_FILTER_OP_LE: return v <= ref;
            case GV_FILTER_OP_GT: return v > ref;
            case GV_FILTER_OP_GE: return v >= ref;
            default: return 0;
            }
        } else {
            if (!node->value) return 0;
            int cmp = strcmp(meta_val, node->value);
            switch (node->op) {
            case GV_FILTER_OP_EQ: return cmp == 0;
            case GV_FILTER_OP_NE: return cmp != 0;
            default:
                return 0;
            }
        }
    }
    default:
        return -1;
    }
}

int gv_filter_eval(const GV_Filter *filter, const GV_Vector *vector) {
    if (!filter || !filter->root || !vector) {
        return -1;
    }
    return gv_filter_eval_node(filter->root, vector);
}

void gv_filter_destroy(GV_Filter *filter) {
    if (!filter) return;
    gv_filter_node_free(filter->root);
    free(filter);
}


