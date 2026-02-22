#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_sql.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_distance.h"
#include "gigavector/gv_filter.h"
#include "gigavector/gv_metadata.h"
#include "gigavector/gv_soa_storage.h"

/*  Constants  */

#define GV_SQL_ERROR_SIZE      512
#define GV_SQL_MAX_QUERY_DIMS  16384
#define GV_SQL_MAX_TOKENS      4096

/*  Token types  */

typedef enum {
    GV_SQL_TOK_EOF = 0,
    GV_SQL_TOK_IDENT,
    GV_SQL_TOK_STRING,
    GV_SQL_TOK_NUMBER,
    GV_SQL_TOK_STAR,
    GV_SQL_TOK_COMMA,
    GV_SQL_TOK_LPAREN,
    GV_SQL_TOK_RPAREN,
    GV_SQL_TOK_LBRACKET,
    GV_SQL_TOK_RBRACKET,
    GV_SQL_TOK_EQ,
    GV_SQL_TOK_NE,
    GV_SQL_TOK_LT,
    GV_SQL_TOK_LE,
    GV_SQL_TOK_GT,
    GV_SQL_TOK_GE,
    /* Keywords */
    GV_SQL_TOK_SELECT,
    GV_SQL_TOK_FROM,
    GV_SQL_TOK_WHERE,
    GV_SQL_TOK_AND,
    GV_SQL_TOK_OR,
    GV_SQL_TOK_NOT,
    GV_SQL_TOK_LIKE,
    GV_SQL_TOK_LIMIT,
    GV_SQL_TOK_ORDER,
    GV_SQL_TOK_BY,
    GV_SQL_TOK_ASC,
    GV_SQL_TOK_DESC,
    GV_SQL_TOK_ANN,
    GV_SQL_TOK_DELETE,
    GV_SQL_TOK_UPDATE,
    GV_SQL_TOK_SET,
    GV_SQL_TOK_COUNT,
    GV_SQL_TOK_ERROR
} GV_SQLTokenType;

typedef struct {
    GV_SQLTokenType type;
    char *text;
    double num_value;
} GV_SQLToken;

/*  Tokeniser  */

typedef struct {
    const char *input;
    size_t pos;
    char error[GV_SQL_ERROR_SIZE];
} GV_SQLLexer;

static void gv_sql_lexer_init(GV_SQLLexer *lx, const char *input)
{
    lx->input = input;
    lx->pos = 0;
    lx->error[0] = '\0';
}

static void gv_sql_token_free(GV_SQLToken *tok)
{
    if (tok && tok->text) {
        free(tok->text);
        tok->text = NULL;
    }
}

static void gv_sql_lexer_skip_ws(GV_SQLLexer *lx)
{
    while (lx->input[lx->pos] != '\0' &&
           (lx->input[lx->pos] == ' '  || lx->input[lx->pos] == '\t' ||
            lx->input[lx->pos] == '\n' || lx->input[lx->pos] == '\r')) {
        lx->pos++;
    }
}

static int gv_sql_kw_match(const char *s, size_t len, const char *kw)
{
    size_t kwlen = strlen(kw);
    if (len != kwlen) return 0;
    for (size_t i = 0; i < len; i++) {
        if ((char)toupper((unsigned char)s[i]) != (char)toupper((unsigned char)kw[i]))
            return 0;
    }
    return 1;
}

static char *gv_sql_strndup(const char *s, size_t n)
{
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static GV_SQLToken gv_sql_lexer_next(GV_SQLLexer *lx)
{
    GV_SQLToken tok;
    memset(&tok, 0, sizeof(tok));

    gv_sql_lexer_skip_ws(lx);
    char c = lx->input[lx->pos];

    if (c == '\0') {
        tok.type = GV_SQL_TOK_EOF;
        return tok;
    }

    /* Single-character tokens */
    if (c == '*') { lx->pos++; tok.type = GV_SQL_TOK_STAR;     return tok; }
    if (c == ',') { lx->pos++; tok.type = GV_SQL_TOK_COMMA;    return tok; }
    if (c == '(') { lx->pos++; tok.type = GV_SQL_TOK_LPAREN;   return tok; }
    if (c == ')') { lx->pos++; tok.type = GV_SQL_TOK_RPAREN;   return tok; }
    if (c == '[') { lx->pos++; tok.type = GV_SQL_TOK_LBRACKET; return tok; }
    if (c == ']') { lx->pos++; tok.type = GV_SQL_TOK_RBRACKET; return tok; }

    /* Two-character operators */
    if (c == '!' && lx->input[lx->pos + 1] == '=') {
        lx->pos += 2; tok.type = GV_SQL_TOK_NE; return tok;
    }
    if (c == '<') {
        if (lx->input[lx->pos + 1] == '=') {
            lx->pos += 2; tok.type = GV_SQL_TOK_LE;
        } else {
            lx->pos++;    tok.type = GV_SQL_TOK_LT;
        }
        return tok;
    }
    if (c == '>') {
        if (lx->input[lx->pos + 1] == '=') {
            lx->pos += 2; tok.type = GV_SQL_TOK_GE;
        } else {
            lx->pos++;    tok.type = GV_SQL_TOK_GT;
        }
        return tok;
    }
    if (c == '=') {
        lx->pos++;
        tok.type = GV_SQL_TOK_EQ;
        return tok;
    }

    /* Quoted string (single or double) */
    if (c == '\'' || c == '"') {
        char quote = c;
        lx->pos++;
        size_t start = lx->pos;
        while (lx->input[lx->pos] != '\0' && lx->input[lx->pos] != quote) {
            if (lx->input[lx->pos] == '\\' && lx->input[lx->pos + 1] != '\0')
                lx->pos += 2;
            else
                lx->pos++;
        }
        if (lx->input[lx->pos] != quote) {
            snprintf(lx->error, GV_SQL_ERROR_SIZE, "Unterminated string literal");
            tok.type = GV_SQL_TOK_ERROR;
            return tok;
        }
        size_t len = lx->pos - start;
        tok.text = gv_sql_strndup(lx->input + start, len);
        lx->pos++; /* consume closing quote */
        tok.type = tok.text ? GV_SQL_TOK_STRING : GV_SQL_TOK_ERROR;
        return tok;
    }

    /* Number (including negative and decimal) */
    if (isdigit((unsigned char)c) || (c == '-' && isdigit((unsigned char)lx->input[lx->pos + 1]))) {
        size_t start = lx->pos;
        if (c == '-') lx->pos++;
        while (isdigit((unsigned char)lx->input[lx->pos])) lx->pos++;
        if (lx->input[lx->pos] == '.') {
            lx->pos++;
            while (isdigit((unsigned char)lx->input[lx->pos])) lx->pos++;
        }
        /* Scientific notation */
        if (lx->input[lx->pos] == 'e' || lx->input[lx->pos] == 'E') {
            lx->pos++;
            if (lx->input[lx->pos] == '+' || lx->input[lx->pos] == '-') lx->pos++;
            while (isdigit((unsigned char)lx->input[lx->pos])) lx->pos++;
        }
        size_t len = lx->pos - start;
        tok.text = gv_sql_strndup(lx->input + start, len);
        if (!tok.text) { tok.type = GV_SQL_TOK_ERROR; return tok; }
        tok.num_value = strtod(tok.text, NULL);
        tok.type = GV_SQL_TOK_NUMBER;
        return tok;
    }

    /* Identifier or keyword */
    if (isalpha((unsigned char)c) || c == '_') {
        size_t start = lx->pos;
        lx->pos++;
        while (isalnum((unsigned char)lx->input[lx->pos]) ||
               lx->input[lx->pos] == '_' || lx->input[lx->pos] == '.') {
            lx->pos++;
        }
        size_t len = lx->pos - start;

        /* Check keywords */
        if (gv_sql_kw_match(lx->input + start, len, "SELECT"))  { tok.type = GV_SQL_TOK_SELECT;  return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "FROM"))    { tok.type = GV_SQL_TOK_FROM;    return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "WHERE"))   { tok.type = GV_SQL_TOK_WHERE;   return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "AND"))     { tok.type = GV_SQL_TOK_AND;     return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "OR"))      { tok.type = GV_SQL_TOK_OR;      return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "NOT"))     { tok.type = GV_SQL_TOK_NOT;     return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "LIKE"))    { tok.type = GV_SQL_TOK_LIKE;    return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "LIMIT"))   { tok.type = GV_SQL_TOK_LIMIT;   return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "ORDER"))   { tok.type = GV_SQL_TOK_ORDER;   return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "BY"))      { tok.type = GV_SQL_TOK_BY;      return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "ASC"))     { tok.type = GV_SQL_TOK_ASC;     return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "DESC"))    { tok.type = GV_SQL_TOK_DESC;    return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "ANN"))     { tok.type = GV_SQL_TOK_ANN;     return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "DELETE"))  { tok.type = GV_SQL_TOK_DELETE;  return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "UPDATE"))  { tok.type = GV_SQL_TOK_UPDATE;  return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "SET"))     { tok.type = GV_SQL_TOK_SET;     return tok; }
        if (gv_sql_kw_match(lx->input + start, len, "COUNT"))   { tok.type = GV_SQL_TOK_COUNT;   return tok; }

        tok.text = gv_sql_strndup(lx->input + start, len);
        tok.type = tok.text ? GV_SQL_TOK_IDENT : GV_SQL_TOK_ERROR;
        return tok;
    }

    snprintf(lx->error, GV_SQL_ERROR_SIZE, "Unexpected character '%c' at position %zu", c, lx->pos);
    lx->pos++;
    tok.type = GV_SQL_TOK_ERROR;
    return tok;
}

/*  Token buffer (pre-tokenise the full query)  */

typedef struct {
    GV_SQLToken *tokens;
    size_t count;
    size_t pos;
    char error[GV_SQL_ERROR_SIZE];
} GV_SQLTokenBuf;

static int gv_sql_tokenize(GV_SQLTokenBuf *buf, const char *query)
{
    GV_SQLLexer lx;
    gv_sql_lexer_init(&lx, query);

    buf->tokens = (GV_SQLToken *)calloc(GV_SQL_MAX_TOKENS, sizeof(GV_SQLToken));
    if (!buf->tokens) {
        snprintf(buf->error, GV_SQL_ERROR_SIZE, "Out of memory during tokenization");
        return -1;
    }
    buf->count = 0;
    buf->pos = 0;
    buf->error[0] = '\0';

    for (;;) {
        if (buf->count >= GV_SQL_MAX_TOKENS) {
            snprintf(buf->error, GV_SQL_ERROR_SIZE, "Query exceeds maximum token count");
            return -1;
        }
        GV_SQLToken tok = gv_sql_lexer_next(&lx);
        if (tok.type == GV_SQL_TOK_ERROR) {
            snprintf(buf->error, GV_SQL_ERROR_SIZE, "Tokenization error: %s", lx.error);
            gv_sql_token_free(&tok);
            return -1;
        }
        buf->tokens[buf->count++] = tok;
        if (tok.type == GV_SQL_TOK_EOF)
            break;
    }
    return 0;
}

static void gv_sql_tokenbuf_free(GV_SQLTokenBuf *buf)
{
    if (!buf || !buf->tokens) return;
    for (size_t i = 0; i < buf->count; i++)
        gv_sql_token_free(&buf->tokens[i]);
    free(buf->tokens);
    buf->tokens = NULL;
    buf->count = 0;
}

static GV_SQLToken *gv_sql_peek(GV_SQLTokenBuf *buf)
{
    if (buf->pos < buf->count)
        return &buf->tokens[buf->pos];
    return NULL;
}

static GV_SQLToken *gv_sql_advance(GV_SQLTokenBuf *buf)
{
    if (buf->pos < buf->count)
        return &buf->tokens[buf->pos++];
    return NULL;
}

static int gv_sql_expect(GV_SQLTokenBuf *buf, GV_SQLTokenType type)
{
    GV_SQLToken *tok = gv_sql_peek(buf);
    if (!tok || tok->type != type) return 0;
    buf->pos++;
    return 1;
}

/*  Parsed AST types  */

typedef enum {
    GV_SQL_STMT_SELECT,
    GV_SQL_STMT_DELETE,
    GV_SQL_STMT_UPDATE
} GV_SQLStmtType;

typedef enum {
    GV_SQL_WHERE_CMP,
    GV_SQL_WHERE_AND,
    GV_SQL_WHERE_OR,
    GV_SQL_WHERE_NOT
} GV_SQLWhereType;

typedef enum {
    GV_SQL_CMP_EQ,
    GV_SQL_CMP_NE,
    GV_SQL_CMP_LT,
    GV_SQL_CMP_LE,
    GV_SQL_CMP_GT,
    GV_SQL_CMP_GE,
    GV_SQL_CMP_LIKE
} GV_SQLCmpOp;

typedef struct GV_SQLWhere {
    GV_SQLWhereType type;
    /* For CMP */
    char *field;
    GV_SQLCmpOp op;
    char *value;
    double num_value;
    int is_numeric;
    /* For AND / OR */
    struct GV_SQLWhere *left;
    struct GV_SQLWhere *right;
    /* For NOT */
    struct GV_SQLWhere *child;
} GV_SQLWhere;

typedef struct {
    float *query_vector;
    size_t query_dim;
    size_t k;
    GV_DistanceType metric;
} GV_SQLAnn;

typedef struct {
    char *field;
    char *value;
} GV_SQLSetClause;

typedef struct {
    GV_SQLStmtType type;
    char *table;

    /* SELECT-specific */
    int is_count;        /**< 1 if SELECT COUNT(*) */
    int has_ann;
    GV_SQLAnn ann;

    /* WHERE clause */
    GV_SQLWhere *where;

    /* ORDER BY */
    char *order_field;
    int order_desc;      /**< 1 if DESC, 0 if ASC */

    /* LIMIT */
    size_t limit;
    int has_limit;

    /* UPDATE SET clauses */
    GV_SQLSetClause *set_clauses;
    size_t set_count;
} GV_SQLStmt;

/*  AST cleanup  */

static void gv_sql_where_free(GV_SQLWhere *w)
{
    if (!w) return;
    gv_sql_where_free(w->left);
    gv_sql_where_free(w->right);
    gv_sql_where_free(w->child);
    free(w->field);
    free(w->value);
    free(w);
}

static void gv_sql_stmt_free(GV_SQLStmt *s)
{
    if (!s) return;
    free(s->table);
    gv_sql_where_free(s->where);
    free(s->order_field);
    if (s->ann.query_vector) free(s->ann.query_vector);
    if (s->set_clauses) {
        for (size_t i = 0; i < s->set_count; i++) {
            free(s->set_clauses[i].field);
            free(s->set_clauses[i].value);
        }
        free(s->set_clauses);
    }
    free(s);
}

/*  Recursive descent parser  */

static char *gv_sql_strdup(const char *s)
{
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *p = (char *)malloc(len);
    if (!p) return NULL;
    memcpy(p, s, len);
    return p;
}

/* Forward declarations */
static GV_SQLWhere *gv_sql_parse_where_expr(GV_SQLTokenBuf *buf);

/* parse_ann: ANN(query=[...], k=N, metric=cosine|euclidean|dot) */
static int gv_sql_parse_ann(GV_SQLTokenBuf *buf, GV_SQLAnn *ann)
{
    /* Already consumed ANN token; expect '(' */
    if (!gv_sql_expect(buf, GV_SQL_TOK_LPAREN)) return -1;

    ann->query_vector = NULL;
    ann->query_dim = 0;
    ann->k = 10;
    ann->metric = GV_DISTANCE_COSINE;

    float tmp_vec[GV_SQL_MAX_QUERY_DIMS];
    size_t tmp_count = 0;

    /* Parse comma-separated key=value pairs inside parens */
    while (gv_sql_peek(buf) && gv_sql_peek(buf)->type != GV_SQL_TOK_RPAREN) {
        GV_SQLToken *key = gv_sql_peek(buf);
        if (!key || key->type != GV_SQL_TOK_IDENT) return -1;
        char *kname = key->text;
        gv_sql_advance(buf);

        if (!gv_sql_expect(buf, GV_SQL_TOK_EQ)) return -1;

        if (gv_sql_kw_match(kname, strlen(kname), "query")) {
            /* Expect '[' number, number, ... ']' */
            if (!gv_sql_expect(buf, GV_SQL_TOK_LBRACKET)) return -1;
            tmp_count = 0;
            while (gv_sql_peek(buf) && gv_sql_peek(buf)->type != GV_SQL_TOK_RBRACKET) {
                GV_SQLToken *num = gv_sql_peek(buf);
                if (!num || num->type != GV_SQL_TOK_NUMBER) return -1;
                if (tmp_count >= GV_SQL_MAX_QUERY_DIMS) return -1;
                tmp_vec[tmp_count++] = (float)num->num_value;
                gv_sql_advance(buf);
                /* Optional comma */
                if (gv_sql_peek(buf) && gv_sql_peek(buf)->type == GV_SQL_TOK_COMMA)
                    gv_sql_advance(buf);
            }
            if (!gv_sql_expect(buf, GV_SQL_TOK_RBRACKET)) return -1;
        } else if (gv_sql_kw_match(kname, strlen(kname), "k")) {
            GV_SQLToken *num = gv_sql_peek(buf);
            if (!num || num->type != GV_SQL_TOK_NUMBER) return -1;
            ann->k = (size_t)num->num_value;
            gv_sql_advance(buf);
        } else if (gv_sql_kw_match(kname, strlen(kname), "metric")) {
            GV_SQLToken *val = gv_sql_peek(buf);
            if (!val || val->type != GV_SQL_TOK_IDENT) return -1;
            if (gv_sql_kw_match(val->text, strlen(val->text), "cosine"))
                ann->metric = GV_DISTANCE_COSINE;
            else if (gv_sql_kw_match(val->text, strlen(val->text), "euclidean"))
                ann->metric = GV_DISTANCE_EUCLIDEAN;
            else if (gv_sql_kw_match(val->text, strlen(val->text), "dot"))
                ann->metric = GV_DISTANCE_DOT_PRODUCT;
            else
                return -1;
            gv_sql_advance(buf);
        } else {
            return -1;
        }

        /* Optional comma between params */
        if (gv_sql_peek(buf) && gv_sql_peek(buf)->type == GV_SQL_TOK_COMMA)
            gv_sql_advance(buf);
    }

    if (!gv_sql_expect(buf, GV_SQL_TOK_RPAREN)) return -1;

    if (tmp_count == 0 || ann->k == 0) return -1;

    ann->query_vector = (float *)malloc(tmp_count * sizeof(float));
    if (!ann->query_vector) return -1;
    memcpy(ann->query_vector, tmp_vec, tmp_count * sizeof(float));
    ann->query_dim = tmp_count;

    return 0;
}

/* parse_where primary: field CMP value | NOT expr | ( expr ) */
static GV_SQLWhere *gv_sql_parse_where_primary(GV_SQLTokenBuf *buf)
{
    GV_SQLToken *tok = gv_sql_peek(buf);
    if (!tok) return NULL;

    /* NOT */
    if (tok->type == GV_SQL_TOK_NOT) {
        gv_sql_advance(buf);
        GV_SQLWhere *child = gv_sql_parse_where_primary(buf);
        if (!child) return NULL;
        GV_SQLWhere *node = (GV_SQLWhere *)calloc(1, sizeof(GV_SQLWhere));
        if (!node) { gv_sql_where_free(child); return NULL; }
        node->type = GV_SQL_WHERE_NOT;
        node->child = child;
        return node;
    }

    /* Parenthesized expression */
    if (tok->type == GV_SQL_TOK_LPAREN) {
        gv_sql_advance(buf);
        GV_SQLWhere *expr = gv_sql_parse_where_expr(buf);
        if (!expr) return NULL;
        if (!gv_sql_expect(buf, GV_SQL_TOK_RPAREN)) {
            gv_sql_where_free(expr);
            return NULL;
        }
        return expr;
    }

    /* field CMP value */
    if (tok->type != GV_SQL_TOK_IDENT) return NULL;
    char *field = gv_sql_strdup(tok->text);
    if (!field) return NULL;
    gv_sql_advance(buf);

    tok = gv_sql_peek(buf);
    if (!tok) { free(field); return NULL; }

    GV_SQLCmpOp op;
    switch (tok->type) {
    case GV_SQL_TOK_EQ:   op = GV_SQL_CMP_EQ;   break;
    case GV_SQL_TOK_NE:   op = GV_SQL_CMP_NE;   break;
    case GV_SQL_TOK_LT:   op = GV_SQL_CMP_LT;   break;
    case GV_SQL_TOK_LE:   op = GV_SQL_CMP_LE;   break;
    case GV_SQL_TOK_GT:   op = GV_SQL_CMP_GT;   break;
    case GV_SQL_TOK_GE:   op = GV_SQL_CMP_GE;   break;
    case GV_SQL_TOK_LIKE: op = GV_SQL_CMP_LIKE;  break;
    default:
        free(field);
        return NULL;
    }
    gv_sql_advance(buf);

    tok = gv_sql_peek(buf);
    if (!tok || (tok->type != GV_SQL_TOK_STRING &&
                 tok->type != GV_SQL_TOK_NUMBER &&
                 tok->type != GV_SQL_TOK_IDENT)) {
        free(field);
        return NULL;
    }

    GV_SQLWhere *node = (GV_SQLWhere *)calloc(1, sizeof(GV_SQLWhere));
    if (!node) { free(field); return NULL; }
    node->type = GV_SQL_WHERE_CMP;
    node->field = field;
    node->op = op;

    if (tok->type == GV_SQL_TOK_NUMBER) {
        node->value = gv_sql_strdup(tok->text);
        node->num_value = tok->num_value;
        node->is_numeric = 1;
    } else {
        node->value = gv_sql_strdup(tok->text ? tok->text : "");
        node->is_numeric = 0;
    }
    gv_sql_advance(buf);

    if (!node->value) {
        gv_sql_where_free(node);
        return NULL;
    }
    return node;
}

/* parse_where AND: primary (AND primary)* */
static GV_SQLWhere *gv_sql_parse_where_and(GV_SQLTokenBuf *buf)
{
    GV_SQLWhere *left = gv_sql_parse_where_primary(buf);
    if (!left) return NULL;

    while (gv_sql_peek(buf) && gv_sql_peek(buf)->type == GV_SQL_TOK_AND) {
        gv_sql_advance(buf);
        GV_SQLWhere *right = gv_sql_parse_where_primary(buf);
        if (!right) { gv_sql_where_free(left); return NULL; }
        GV_SQLWhere *node = (GV_SQLWhere *)calloc(1, sizeof(GV_SQLWhere));
        if (!node) { gv_sql_where_free(left); gv_sql_where_free(right); return NULL; }
        node->type = GV_SQL_WHERE_AND;
        node->left = left;
        node->right = right;
        left = node;
    }
    return left;
}

/* parse_where OR: and_expr (OR and_expr)* */
static GV_SQLWhere *gv_sql_parse_where_expr(GV_SQLTokenBuf *buf)
{
    GV_SQLWhere *left = gv_sql_parse_where_and(buf);
    if (!left) return NULL;

    while (gv_sql_peek(buf) && gv_sql_peek(buf)->type == GV_SQL_TOK_OR) {
        gv_sql_advance(buf);
        GV_SQLWhere *right = gv_sql_parse_where_and(buf);
        if (!right) { gv_sql_where_free(left); return NULL; }
        GV_SQLWhere *node = (GV_SQLWhere *)calloc(1, sizeof(GV_SQLWhere));
        if (!node) { gv_sql_where_free(left); gv_sql_where_free(right); return NULL; }
        node->type = GV_SQL_WHERE_OR;
        node->left = left;
        node->right = right;
        left = node;
    }
    return left;
}

/* parse_select: SELECT (* | COUNT(*)) FROM table [ANN(...)] [WHERE ...] [ORDER BY ...] [LIMIT n] */
static GV_SQLStmt *gv_sql_parse_select(GV_SQLTokenBuf *buf)
{
    /* SELECT already consumed */
    GV_SQLStmt *stmt = (GV_SQLStmt *)calloc(1, sizeof(GV_SQLStmt));
    if (!stmt) return NULL;
    stmt->type = GV_SQL_STMT_SELECT;

    GV_SQLToken *tok = gv_sql_peek(buf);
    if (!tok) { gv_sql_stmt_free(stmt); return NULL; }

    /* SELECT COUNT(*) or SELECT * */
    if (tok->type == GV_SQL_TOK_COUNT) {
        gv_sql_advance(buf);
        if (!gv_sql_expect(buf, GV_SQL_TOK_LPAREN) ||
            !gv_sql_expect(buf, GV_SQL_TOK_STAR) ||
            !gv_sql_expect(buf, GV_SQL_TOK_RPAREN)) {
            gv_sql_stmt_free(stmt);
            return NULL;
        }
        stmt->is_count = 1;
    } else if (tok->type == GV_SQL_TOK_STAR) {
        gv_sql_advance(buf);
    } else {
        gv_sql_stmt_free(stmt);
        return NULL;
    }

    /* FROM table */
    if (!gv_sql_expect(buf, GV_SQL_TOK_FROM)) { gv_sql_stmt_free(stmt); return NULL; }
    tok = gv_sql_peek(buf);
    if (!tok || tok->type != GV_SQL_TOK_IDENT) { gv_sql_stmt_free(stmt); return NULL; }
    stmt->table = gv_sql_strdup(tok->text);
    gv_sql_advance(buf);
    if (!stmt->table) { gv_sql_stmt_free(stmt); return NULL; }

    /* Optional ANN(...) */
    tok = gv_sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_ANN) {
        gv_sql_advance(buf);
        if (gv_sql_parse_ann(buf, &stmt->ann) != 0) {
            gv_sql_stmt_free(stmt);
            return NULL;
        }
        stmt->has_ann = 1;
    }

    /* Optional WHERE */
    tok = gv_sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_WHERE) {
        gv_sql_advance(buf);
        stmt->where = gv_sql_parse_where_expr(buf);
        if (!stmt->where) { gv_sql_stmt_free(stmt); return NULL; }
    }

    /* Optional ORDER BY field [ASC|DESC] */
    tok = gv_sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_ORDER) {
        gv_sql_advance(buf);
        if (!gv_sql_expect(buf, GV_SQL_TOK_BY)) { gv_sql_stmt_free(stmt); return NULL; }
        tok = gv_sql_peek(buf);
        if (!tok || tok->type != GV_SQL_TOK_IDENT) { gv_sql_stmt_free(stmt); return NULL; }
        stmt->order_field = gv_sql_strdup(tok->text);
        gv_sql_advance(buf);
        if (!stmt->order_field) { gv_sql_stmt_free(stmt); return NULL; }

        tok = gv_sql_peek(buf);
        if (tok && tok->type == GV_SQL_TOK_DESC) {
            stmt->order_desc = 1;
            gv_sql_advance(buf);
        } else if (tok && tok->type == GV_SQL_TOK_ASC) {
            stmt->order_desc = 0;
            gv_sql_advance(buf);
        }
    }

    /* Optional LIMIT n */
    tok = gv_sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_LIMIT) {
        gv_sql_advance(buf);
        tok = gv_sql_peek(buf);
        if (!tok || tok->type != GV_SQL_TOK_NUMBER) { gv_sql_stmt_free(stmt); return NULL; }
        stmt->limit = (size_t)tok->num_value;
        stmt->has_limit = 1;
        gv_sql_advance(buf);
    }

    return stmt;
}

/* parse_delete: DELETE FROM table WHERE ... */
static GV_SQLStmt *gv_sql_parse_delete(GV_SQLTokenBuf *buf)
{
    /* DELETE already consumed */
    GV_SQLStmt *stmt = (GV_SQLStmt *)calloc(1, sizeof(GV_SQLStmt));
    if (!stmt) return NULL;
    stmt->type = GV_SQL_STMT_DELETE;

    if (!gv_sql_expect(buf, GV_SQL_TOK_FROM)) { gv_sql_stmt_free(stmt); return NULL; }
    GV_SQLToken *tok = gv_sql_peek(buf);
    if (!tok || tok->type != GV_SQL_TOK_IDENT) { gv_sql_stmt_free(stmt); return NULL; }
    stmt->table = gv_sql_strdup(tok->text);
    gv_sql_advance(buf);
    if (!stmt->table) { gv_sql_stmt_free(stmt); return NULL; }

    if (!gv_sql_expect(buf, GV_SQL_TOK_WHERE)) { gv_sql_stmt_free(stmt); return NULL; }
    stmt->where = gv_sql_parse_where_expr(buf);
    if (!stmt->where) { gv_sql_stmt_free(stmt); return NULL; }

    return stmt;
}

/* parse_update: UPDATE table SET field=val, ... WHERE ... */
static GV_SQLStmt *gv_sql_parse_update(GV_SQLTokenBuf *buf)
{
    /* UPDATE already consumed */
    GV_SQLStmt *stmt = (GV_SQLStmt *)calloc(1, sizeof(GV_SQLStmt));
    if (!stmt) return NULL;
    stmt->type = GV_SQL_STMT_UPDATE;

    GV_SQLToken *tok = gv_sql_peek(buf);
    if (!tok || tok->type != GV_SQL_TOK_IDENT) { gv_sql_stmt_free(stmt); return NULL; }
    stmt->table = gv_sql_strdup(tok->text);
    gv_sql_advance(buf);
    if (!stmt->table) { gv_sql_stmt_free(stmt); return NULL; }

    if (!gv_sql_expect(buf, GV_SQL_TOK_SET)) { gv_sql_stmt_free(stmt); return NULL; }

    /* Parse SET clauses: field = value [, field = value ...] */
    size_t cap = 8;
    stmt->set_clauses = (GV_SQLSetClause *)calloc(cap, sizeof(GV_SQLSetClause));
    if (!stmt->set_clauses) { gv_sql_stmt_free(stmt); return NULL; }
    stmt->set_count = 0;

    for (;;) {
        tok = gv_sql_peek(buf);
        if (!tok || tok->type != GV_SQL_TOK_IDENT) break;
        char *field = gv_sql_strdup(tok->text);
        gv_sql_advance(buf);
        if (!field) { gv_sql_stmt_free(stmt); return NULL; }

        if (!gv_sql_expect(buf, GV_SQL_TOK_EQ)) { free(field); gv_sql_stmt_free(stmt); return NULL; }

        tok = gv_sql_peek(buf);
        if (!tok || (tok->type != GV_SQL_TOK_STRING &&
                     tok->type != GV_SQL_TOK_NUMBER &&
                     tok->type != GV_SQL_TOK_IDENT)) {
            free(field);
            gv_sql_stmt_free(stmt);
            return NULL;
        }
        char *value = gv_sql_strdup(tok->text ? tok->text : "");
        gv_sql_advance(buf);
        if (!value) { free(field); gv_sql_stmt_free(stmt); return NULL; }

        if (stmt->set_count >= cap) {
            cap *= 2;
            GV_SQLSetClause *tmp = (GV_SQLSetClause *)realloc(
                stmt->set_clauses, cap * sizeof(GV_SQLSetClause));
            if (!tmp) { free(field); free(value); gv_sql_stmt_free(stmt); return NULL; }
            stmt->set_clauses = tmp;
        }
        stmt->set_clauses[stmt->set_count].field = field;
        stmt->set_clauses[stmt->set_count].value = value;
        stmt->set_count++;

        /* Optional comma */
        if (gv_sql_peek(buf) && gv_sql_peek(buf)->type == GV_SQL_TOK_COMMA)
            gv_sql_advance(buf);
        else
            break;
    }

    if (stmt->set_count == 0) { gv_sql_stmt_free(stmt); return NULL; }

    /* WHERE clause (required for UPDATE) */
    if (!gv_sql_expect(buf, GV_SQL_TOK_WHERE)) { gv_sql_stmt_free(stmt); return NULL; }
    stmt->where = gv_sql_parse_where_expr(buf);
    if (!stmt->where) { gv_sql_stmt_free(stmt); return NULL; }

    return stmt;
}

/* Top-level parser */
static GV_SQLStmt *gv_sql_parse(GV_SQLTokenBuf *buf)
{
    GV_SQLToken *tok = gv_sql_peek(buf);
    if (!tok) return NULL;

    switch (tok->type) {
    case GV_SQL_TOK_SELECT:
        gv_sql_advance(buf);
        return gv_sql_parse_select(buf);
    case GV_SQL_TOK_DELETE:
        gv_sql_advance(buf);
        return gv_sql_parse_delete(buf);
    case GV_SQL_TOK_UPDATE:
        gv_sql_advance(buf);
        return gv_sql_parse_update(buf);
    default:
        snprintf(buf->error, GV_SQL_ERROR_SIZE,
                 "Expected SELECT, DELETE, or UPDATE; got '%s'",
                 tok->text ? tok->text : "(unknown)");
        return NULL;
    }
}

/*  WHERE clause evaluator (against vector metadata)  */

static int gv_sql_eval_where(const GV_SQLWhere *w, const GV_Vector *vec)
{
    if (!w) return 1; /* no WHERE => match all */

    switch (w->type) {
    case GV_SQL_WHERE_AND: {
        int l = gv_sql_eval_where(w->left, vec);
        if (l <= 0) return l;
        return gv_sql_eval_where(w->right, vec);
    }
    case GV_SQL_WHERE_OR: {
        int l = gv_sql_eval_where(w->left, vec);
        if (l < 0) return l;
        if (l == 1) return 1;
        return gv_sql_eval_where(w->right, vec);
    }
    case GV_SQL_WHERE_NOT: {
        int v = gv_sql_eval_where(w->child, vec);
        if (v < 0) return v;
        return v ? 0 : 1;
    }
    case GV_SQL_WHERE_CMP: {
        const char *meta_val = gv_vector_get_metadata(vec, w->field);
        if (!meta_val) return 0;

        if (w->op == GV_SQL_CMP_LIKE) {
            /* Simple LIKE: treat '%' as wildcard prefix/suffix */
            if (!w->value) return 0;
            size_t vlen = strlen(w->value);
            if (vlen == 0) return strlen(meta_val) == 0 ? 1 : 0;
            int prefix_wild = (w->value[0] == '%');
            int suffix_wild = (vlen > 1 && w->value[vlen - 1] == '%');
            const char *pattern = w->value + (prefix_wild ? 1 : 0);
            size_t plen = strlen(pattern);
            if (suffix_wild && plen > 0) plen--;
            if (prefix_wild && suffix_wild) {
                /* contains */
                char *tmp = gv_sql_strndup(pattern, plen);
                if (!tmp) return -1;
                int found = (strstr(meta_val, tmp) != NULL);
                free(tmp);
                return found;
            } else if (prefix_wild) {
                /* ends with */
                size_t mlen = strlen(meta_val);
                if (plen > mlen) return 0;
                return memcmp(meta_val + mlen - plen, pattern, plen) == 0;
            } else if (suffix_wild) {
                /* starts with */
                return strncmp(meta_val, pattern, plen) == 0;
            } else {
                return strcmp(meta_val, w->value) == 0;
            }
        }

        if (w->is_numeric) {
            char *endptr = NULL;
            double v = strtod(meta_val, &endptr);
            if (endptr == meta_val) return 0;
            switch (w->op) {
            case GV_SQL_CMP_EQ: return v == w->num_value;
            case GV_SQL_CMP_NE: return v != w->num_value;
            case GV_SQL_CMP_LT: return v <  w->num_value;
            case GV_SQL_CMP_LE: return v <= w->num_value;
            case GV_SQL_CMP_GT: return v >  w->num_value;
            case GV_SQL_CMP_GE: return v >= w->num_value;
            default: return 0;
            }
        } else {
            int cmp = strcmp(meta_val, w->value);
            switch (w->op) {
            case GV_SQL_CMP_EQ: return cmp == 0;
            case GV_SQL_CMP_NE: return cmp != 0;
            case GV_SQL_CMP_LT: return cmp <  0;
            case GV_SQL_CMP_LE: return cmp <= 0;
            case GV_SQL_CMP_GT: return cmp >  0;
            case GV_SQL_CMP_GE: return cmp >= 0;
            default: return 0;
            }
        }
    }
    }
    return 0;
}

/*  Metadata-to-JSON serialiser (lightweight, no dependency on gv_json.h)  */

static char *gv_sql_metadata_to_json(const GV_Metadata *meta)
{
    if (!meta) {
        char *s = (char *)malloc(3);
        if (s) { s[0] = '{'; s[1] = '}'; s[2] = '\0'; }
        return s;
    }

    /* Build JSON string incrementally */
    size_t cap = 256;
    size_t len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    buf[len++] = '{';

    int first = 1;
    for (const GV_Metadata *m = meta; m; m = m->next) {
        if (!first) {
            if (len + 2 > cap) { cap *= 2; char *t = realloc(buf, cap); if (!t) { free(buf); return NULL; } buf = t; }
            buf[len++] = ',';
        }
        first = 0;

        /* "key":"value" */
        size_t klen = m->key ? strlen(m->key) : 0;
        size_t vlen = m->value ? strlen(m->value) : 0;
        size_t needed = len + klen + vlen + 8; /* quotes, colon, etc */
        if (needed > cap) { while (cap < needed) cap *= 2; char *t = realloc(buf, cap); if (!t) { free(buf); return NULL; } buf = t; }

        buf[len++] = '"';
        if (m->key) { memcpy(buf + len, m->key, klen); len += klen; }
        buf[len++] = '"';
        buf[len++] = ':';
        buf[len++] = '"';
        if (m->value) { memcpy(buf + len, m->value, vlen); len += vlen; }
        buf[len++] = '"';
    }

    if (len + 2 > cap) { cap = len + 2; char *t = realloc(buf, cap); if (!t) { free(buf); return NULL; } buf = t; }
    buf[len++] = '}';
    buf[len] = '\0';
    return buf;
}

/*  Engine internals  */

struct GV_SQLEngine {
    GV_Database *db;
    pthread_mutex_t mutex;
    char last_error[GV_SQL_ERROR_SIZE];
};

static void gv_sql_set_error(GV_SQLEngine *eng, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(eng->last_error, GV_SQL_ERROR_SIZE, fmt, ap);
    va_end(ap);
}

/*  Executor: SELECT with ANN  */

static int gv_sql_exec_ann(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    const GV_SQLAnn *ann = &stmt->ann;

    if (ann->query_dim != db->dimension) {
        gv_sql_set_error(eng, "ANN query dimension %zu does not match database dimension %zu",
                         ann->query_dim, db->dimension);
        return -1;
    }

    size_t k = ann->k;
    GV_SearchResult *sr = (GV_SearchResult *)calloc(k, sizeof(GV_SearchResult));
    if (!sr) { gv_sql_set_error(eng, "Out of memory"); return -1; }

    int found;
    if (stmt->where) {
        /* Build a filter expression string from WHERE clause for the database API.
         * Use the lower-level approach: search with larger k, then post-filter. */
        size_t oversample = k * 4;
        if (oversample > db->count && db->count > 0) oversample = db->count;
        if (oversample < k) oversample = k;
        GV_SearchResult *all_sr = (GV_SearchResult *)calloc(oversample, sizeof(GV_SearchResult));
        if (!all_sr) { free(sr); gv_sql_set_error(eng, "Out of memory"); return -1; }

        found = gv_db_search(db, ann->query_vector, oversample, all_sr, ann->metric);
        if (found < 0) {
            free(all_sr);
            free(sr);
            gv_sql_set_error(eng, "ANN search failed");
            return -1;
        }

        /* Post-filter */
        int matched = 0;
        for (int i = 0; i < found && (size_t)matched < k; i++) {
            if (!all_sr[i].vector) continue;
            if (gv_sql_eval_where(stmt->where, all_sr[i].vector) == 1) {
                sr[matched++] = all_sr[i];
            }
        }
        free(all_sr);
        found = matched;
    } else {
        found = gv_db_search(db, ann->query_vector, k, sr, ann->metric);
        if (found < 0) {
            free(sr);
            gv_sql_set_error(eng, "ANN search failed");
            return -1;
        }
    }

    if (found < 0) found = 0;

    /* Apply LIMIT */
    size_t row_count = (size_t)found;
    if (stmt->has_limit && stmt->limit < row_count)
        row_count = stmt->limit;

    /* Build result */
    memset(result, 0, sizeof(*result));
    result->row_count = row_count;
    result->column_count = 3;
    result->column_names = (char **)calloc(3, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_sql_strdup("index");
        result->column_names[1] = gv_sql_strdup("distance");
        result->column_names[2] = gv_sql_strdup("metadata");
    }

    result->indices = (size_t *)calloc(row_count, sizeof(size_t));
    result->distances = (float *)calloc(row_count, sizeof(float));
    result->metadata_jsons = (char **)calloc(row_count, sizeof(char *));

    if (!result->indices || !result->distances || !result->metadata_jsons || !result->column_names) {
        gv_sql_free_result(result);
        free(sr);
        gv_sql_set_error(eng, "Out of memory building result set");
        return -1;
    }

    for (size_t i = 0; i < row_count; i++) {
        /* Determine vector index from the search result */
        if (sr[i].vector && db->soa_storage) {
            const float *vec_data = sr[i].vector->data;
            const float *base = db->soa_storage->data;
            if (vec_data && base && vec_data >= base) {
                size_t offset = (size_t)(vec_data - base);
                result->indices[i] = offset / db->dimension;
            }
        }
        result->distances[i] = sr[i].distance;
        result->metadata_jsons[i] = gv_sql_metadata_to_json(
            sr[i].vector ? sr[i].vector->metadata : NULL);
    }

    free(sr);
    return 0;
}

/*  Executor: SELECT with WHERE (metadata scan, no ANN)  */

static int gv_sql_exec_where_scan(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { gv_sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t max_results = stmt->has_limit ? stmt->limit : total;

    /* Collect matching indices */
    size_t cap = (max_results < 1024) ? max_results + 1 : 1024;
    size_t *match_idx = (size_t *)malloc(cap * sizeof(size_t));
    if (!match_idx) { gv_sql_set_error(eng, "Out of memory"); return -1; }
    size_t match_count = 0;

    for (size_t i = 0; i < total && match_count < max_results; i++) {
        if (gv_soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (gv_soa_storage_get_vector_view(soa, i, &view) != 0) continue;

        if (gv_sql_eval_where(stmt->where, &view) == 1) {
            if (match_count >= cap) {
                cap *= 2;
                size_t *tmp = (size_t *)realloc(match_idx, cap * sizeof(size_t));
                if (!tmp) { free(match_idx); gv_sql_set_error(eng, "Out of memory"); return -1; }
                match_idx = tmp;
            }
            match_idx[match_count++] = i;
        }
    }

    /* Build result */
    memset(result, 0, sizeof(*result));
    result->row_count = match_count;
    result->column_count = 2;
    result->column_names = (char **)calloc(2, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_sql_strdup("index");
        result->column_names[1] = gv_sql_strdup("metadata");
    }

    result->indices = (size_t *)calloc(match_count ? match_count : 1, sizeof(size_t));
    result->metadata_jsons = (char **)calloc(match_count ? match_count : 1, sizeof(char *));

    if (!result->indices || !result->metadata_jsons || !result->column_names) {
        gv_sql_free_result(result);
        free(match_idx);
        gv_sql_set_error(eng, "Out of memory building result set");
        return -1;
    }

    for (size_t i = 0; i < match_count; i++) {
        result->indices[i] = match_idx[i];
        GV_Metadata *meta = gv_soa_storage_get_metadata(soa, match_idx[i]);
        result->metadata_jsons[i] = gv_sql_metadata_to_json(meta);
    }

    free(match_idx);
    return 0;
}

/*  Executor: SELECT COUNT(*)  */

static int gv_sql_exec_count(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { gv_sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t count = 0;

    for (size_t i = 0; i < total; i++) {
        if (gv_soa_storage_is_deleted(soa, i)) continue;

        if (stmt->where) {
            GV_Vector view;
            if (gv_soa_storage_get_vector_view(soa, i, &view) != 0) continue;
            if (gv_sql_eval_where(stmt->where, &view) == 1)
                count++;
        } else {
            count++;
        }
    }

    /* Build single-row result with count */
    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)calloc(1, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_sql_strdup("count");
    }

    result->indices = (size_t *)calloc(1, sizeof(size_t));
    if (!result->indices || !result->column_names) {
        gv_sql_free_result(result);
        gv_sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->indices[0] = count;
    return 0;
}

/*  Executor: DELETE  */

static int gv_sql_exec_delete(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { gv_sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t deleted = 0;

    /* Collect matching indices first (iterate, then delete) */
    size_t cap = 256;
    size_t *del_idx = (size_t *)malloc(cap * sizeof(size_t));
    if (!del_idx) { gv_sql_set_error(eng, "Out of memory"); return -1; }

    for (size_t i = 0; i < total; i++) {
        if (gv_soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (gv_soa_storage_get_vector_view(soa, i, &view) != 0) continue;

        if (gv_sql_eval_where(stmt->where, &view) == 1) {
            if (deleted >= cap) {
                cap *= 2;
                size_t *tmp = (size_t *)realloc(del_idx, cap * sizeof(size_t));
                if (!tmp) { free(del_idx); gv_sql_set_error(eng, "Out of memory"); return -1; }
                del_idx = tmp;
            }
            del_idx[deleted++] = i;
        }
    }

    /* Perform deletions */
    for (size_t i = 0; i < deleted; i++) {
        gv_db_delete_vector_by_index(db, del_idx[i]);
    }
    free(del_idx);

    /* Build result: single row with delete count */
    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)calloc(1, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_sql_strdup("deleted_count");
    }
    result->indices = (size_t *)calloc(1, sizeof(size_t));
    if (!result->indices || !result->column_names) {
        gv_sql_free_result(result);
        gv_sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->indices[0] = deleted;
    return 0;
}

/*  Executor: UPDATE  */

static int gv_sql_exec_update(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { gv_sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t updated = 0;

    for (size_t i = 0; i < total; i++) {
        if (gv_soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (gv_soa_storage_get_vector_view(soa, i, &view) != 0) continue;

        if (gv_sql_eval_where(stmt->where, &view) != 1) continue;

        /* Apply SET clauses as metadata updates */
        const char **keys = (const char **)malloc(stmt->set_count * sizeof(char *));
        const char **vals = (const char **)malloc(stmt->set_count * sizeof(char *));
        if (!keys || !vals) {
            free(keys);
            free(vals);
            gv_sql_set_error(eng, "Out of memory");
            return -1;
        }
        for (size_t j = 0; j < stmt->set_count; j++) {
            keys[j] = stmt->set_clauses[j].field;
            vals[j] = stmt->set_clauses[j].value;
        }
        gv_db_update_vector_metadata(db, i, keys, vals, stmt->set_count);
        free(keys);
        free(vals);
        updated++;
    }

    /* Build result: single row with update count */
    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)calloc(1, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_sql_strdup("updated_count");
    }
    result->indices = (size_t *)calloc(1, sizeof(size_t));
    if (!result->indices || !result->column_names) {
        gv_sql_free_result(result);
        gv_sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->indices[0] = updated;
    return 0;
}

/*  Public API  */

GV_SQLEngine *gv_sql_create(void *db)
{
    if (!db) return NULL;

    GV_SQLEngine *eng = (GV_SQLEngine *)calloc(1, sizeof(GV_SQLEngine));
    if (!eng) return NULL;

    eng->db = (GV_Database *)db;
    if (pthread_mutex_init(&eng->mutex, NULL) != 0) {
        free(eng);
        return NULL;
    }
    eng->last_error[0] = '\0';
    return eng;
}

void gv_sql_destroy(GV_SQLEngine *eng)
{
    if (!eng) return;
    pthread_mutex_destroy(&eng->mutex);
    free(eng);
}

int gv_sql_execute(GV_SQLEngine *eng, const char *query, GV_SQLResult *result)
{
    if (!eng || !query || !result) return -1;

    pthread_mutex_lock(&eng->mutex);
    eng->last_error[0] = '\0';

    /* Tokenize */
    GV_SQLTokenBuf tbuf;
    memset(&tbuf, 0, sizeof(tbuf));
    if (gv_sql_tokenize(&tbuf, query) != 0) {
        gv_sql_set_error(eng, "%s", tbuf.error);
        gv_sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Parse */
    GV_SQLStmt *stmt = gv_sql_parse(&tbuf);
    if (!stmt) {
        if (tbuf.error[0])
            gv_sql_set_error(eng, "Parse error: %s", tbuf.error);
        else
            gv_sql_set_error(eng, "Parse error: invalid SQL syntax");
        gv_sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Check for trailing tokens */
    GV_SQLToken *remaining = gv_sql_peek(&tbuf);
    if (remaining && remaining->type != GV_SQL_TOK_EOF) {
        gv_sql_set_error(eng, "Parse error: unexpected tokens after statement");
        gv_sql_stmt_free(stmt);
        gv_sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Execute */
    int rc = -1;
    switch (stmt->type) {
    case GV_SQL_STMT_SELECT:
        if (stmt->is_count) {
            rc = gv_sql_exec_count(eng, stmt, result);
        } else if (stmt->has_ann) {
            rc = gv_sql_exec_ann(eng, stmt, result);
        } else {
            rc = gv_sql_exec_where_scan(eng, stmt, result);
        }
        break;
    case GV_SQL_STMT_DELETE:
        rc = gv_sql_exec_delete(eng, stmt, result);
        break;
    case GV_SQL_STMT_UPDATE:
        rc = gv_sql_exec_update(eng, stmt, result);
        break;
    }

    gv_sql_stmt_free(stmt);
    gv_sql_tokenbuf_free(&tbuf);
    pthread_mutex_unlock(&eng->mutex);
    return rc;
}

void gv_sql_free_result(GV_SQLResult *result)
{
    if (!result) return;

    free(result->indices);
    free(result->distances);

    if (result->metadata_jsons) {
        for (size_t i = 0; i < result->row_count; i++)
            free(result->metadata_jsons[i]);
        free(result->metadata_jsons);
    }

    if (result->column_names) {
        for (size_t i = 0; i < result->column_count; i++)
            free(result->column_names[i]);
        free(result->column_names);
    }

    memset(result, 0, sizeof(*result));
}

const char *gv_sql_last_error(const GV_SQLEngine *eng)
{
    if (!eng) return "";
    return eng->last_error;
}

int gv_sql_explain(GV_SQLEngine *eng, const char *query, char *plan, size_t plan_size)
{
    if (!eng || !query || !plan || plan_size == 0) return -1;

    pthread_mutex_lock(&eng->mutex);
    eng->last_error[0] = '\0';
    plan[0] = '\0';

    /* Tokenize */
    GV_SQLTokenBuf tbuf;
    memset(&tbuf, 0, sizeof(tbuf));
    if (gv_sql_tokenize(&tbuf, query) != 0) {
        gv_sql_set_error(eng, "%s", tbuf.error);
        gv_sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Parse */
    GV_SQLStmt *stmt = gv_sql_parse(&tbuf);
    if (!stmt) {
        if (tbuf.error[0])
            gv_sql_set_error(eng, "Parse error: %s", tbuf.error);
        else
            gv_sql_set_error(eng, "Parse error: invalid SQL syntax");
        gv_sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    GV_Database *db = eng->db;
    size_t total_vectors = db->count;

    /* Determine index type name */
    const char *index_name;
    switch (db->index_type) {
    case GV_INDEX_TYPE_KDTREE:  index_name = "KDTREE";  break;
    case GV_INDEX_TYPE_HNSW:    index_name = "HNSW";    break;
    case GV_INDEX_TYPE_IVFPQ:   index_name = "IVFPQ";   break;
    case GV_INDEX_TYPE_SPARSE:  index_name = "SPARSE";   break;
    case GV_INDEX_TYPE_FLAT:    index_name = "FLAT";     break;
    case GV_INDEX_TYPE_IVFFLAT: index_name = "IVFFLAT"; break;
    case GV_INDEX_TYPE_PQ:      index_name = "PQ";       break;
    case GV_INDEX_TYPE_LSH:     index_name = "LSH";      break;
    default:                    index_name = "UNKNOWN";  break;
    }

    size_t off = 0;

    switch (stmt->type) {
    case GV_SQL_STMT_SELECT:
        if (stmt->is_count) {
            off += (size_t)snprintf(plan + off, plan_size - off,
                "EXPLAIN: SELECT COUNT(*)\n"
                "  Strategy: FULL_SCAN\n"
                "  Index: %s\n"
                "  Estimated rows: %zu\n"
                "  Filter: %s\n",
                index_name, total_vectors,
                stmt->where ? "WHERE predicate (post-filter)" : "NONE");
        } else if (stmt->has_ann) {
            off += (size_t)snprintf(plan + off, plan_size - off,
                "EXPLAIN: SELECT with ANN\n"
                "  Strategy: INDEX_ANN_SEARCH\n"
                "  Index: %s\n"
                "  Metric: %s\n"
                "  k: %zu\n"
                "  Query dimension: %zu\n"
                "  Total vectors: %zu\n"
                "  Filter: %s\n",
                index_name,
                stmt->ann.metric == GV_DISTANCE_COSINE ? "COSINE" :
                stmt->ann.metric == GV_DISTANCE_EUCLIDEAN ? "EUCLIDEAN" : "DOT_PRODUCT",
                stmt->ann.k,
                stmt->ann.query_dim,
                total_vectors,
                stmt->where ? "WHERE predicate (post-filter on ANN results)" : "NONE");
            if (stmt->where) {
                off += (size_t)snprintf(plan + off, plan_size - off,
                    "  Oversample factor: 4x\n");
            }
        } else {
            off += (size_t)snprintf(plan + off, plan_size - off,
                "EXPLAIN: SELECT with WHERE\n"
                "  Strategy: FULL_SCAN\n"
                "  Index: %s (not used for metadata-only query)\n"
                "  Estimated rows: %zu\n"
                "  Filter: WHERE predicate\n",
                index_name, total_vectors);
        }
        if (stmt->has_limit) {
            off += (size_t)snprintf(plan + off, plan_size - off,
                "  Limit: %zu\n", stmt->limit);
        }
        if (stmt->order_field) {
            (void)snprintf(plan + off, plan_size - off,
                "  Order by: %s %s\n", stmt->order_field,
                stmt->order_desc ? "DESC" : "ASC");
        }
        break;

    case GV_SQL_STMT_DELETE:
        (void)snprintf(plan + off, plan_size - off,
            "EXPLAIN: DELETE\n"
            "  Strategy: FULL_SCAN + DELETE\n"
            "  Index: %s\n"
            "  Estimated rows scanned: %zu\n"
            "  Filter: WHERE predicate\n",
            index_name, total_vectors);
        break;

    case GV_SQL_STMT_UPDATE:
        (void)snprintf(plan + off, plan_size - off,
            "EXPLAIN: UPDATE\n"
            "  Strategy: FULL_SCAN + METADATA_UPDATE\n"
            "  Index: %s\n"
            "  Estimated rows scanned: %zu\n"
            "  Filter: WHERE predicate\n"
            "  SET clauses: %zu\n",
            index_name, total_vectors, stmt->set_count);
        break;
    }

    gv_sql_stmt_free(stmt);
    gv_sql_tokenbuf_free(&tbuf);
    pthread_mutex_unlock(&eng->mutex);
    return 0;
}
