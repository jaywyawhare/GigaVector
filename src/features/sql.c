#include <ctype.h>
#include "core/memory.h"
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "features/sql.h"
#include "storage/database.h"
#include "search/distance.h"
#include "schema/metadata.h"
#include "storage/soa_storage.h"
#include "core/utils.h"

/* Constants */

#define GV_SQL_ERROR_SIZE      1024
#define GV_SQL_MAX_QUERY_DIMS  16384
#define GV_SQL_MAX_TOKENS      4096

/* Token types */

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
    GV_SQL_TOK_SEMICOLON,
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
    GV_SQL_TOK_OFFSET,
    GV_SQL_TOK_IN,
    GV_SQL_TOK_BETWEEN,
    GV_SQL_TOK_IS,
    GV_SQL_TOK_NULL,
    GV_SQL_TOK_INSERT,
    GV_SQL_TOK_INTO,
    GV_SQL_TOK_VALUES,
    GV_SQL_TOK_SUM,
    GV_SQL_TOK_MIN,
    GV_SQL_TOK_MAX,
    GV_SQL_TOK_AVG,
    GV_SQL_TOK_ERROR
} GV_SQLTokenType;

typedef struct {
    GV_SQLTokenType type;
    char *text;
    double num_value;
} GV_SQLToken;

/* Tokeniser */

typedef struct {
    const char *input;
    size_t pos;
    char error[GV_SQL_ERROR_SIZE];
} GV_SQLLexer;

static void sql_lexer_init(GV_SQLLexer *lx, const char *input)
{
    lx->input = input;
    lx->pos = 0;
    lx->error[0] = '\0';
}

static void sql_token_free(GV_SQLToken *tok)
{
    if (tok && tok->text) {
        gv_free(tok->text);
        tok->text = NULL;
    }
}

static void sql_lexer_skip_ws(GV_SQLLexer *lx)
{
    while (lx->input[lx->pos] != '\0' &&
           (lx->input[lx->pos] == ' '  || lx->input[lx->pos] == '\t' ||
            lx->input[lx->pos] == '\n' || lx->input[lx->pos] == '\r')) {
        lx->pos++;
    }
}

static int sql_kw_match(const char *s, size_t len, const char *kw)
{
    size_t kwlen = strlen(kw);
    if (len != kwlen) return 0;
    for (size_t i = 0; i < len; i++) {
        if ((char)toupper((unsigned char)s[i]) != (char)toupper((unsigned char)kw[i]))
            return 0;
    }
    return 1;
}

static char *sql_strndup(const char *s, size_t n)
{
    char *p = (char *)gv_alloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static GV_SQLToken sql_lexer_next(GV_SQLLexer *lx)
{
    GV_SQLToken tok;
    memset(&tok, 0, sizeof(tok));

    sql_lexer_skip_ws(lx);
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
    if (c == ';') { lx->pos++; tok.type = GV_SQL_TOK_SEMICOLON; return tok; }

    /* Two-character operators */
    if (c == '!' && lx->input[lx->pos + 1] == '=') {
        lx->pos += 2; tok.type = GV_SQL_TOK_NE; return tok;
    }
    if (c == '<') {
        if (lx->input[lx->pos + 1] == '=') {
            lx->pos += 2; tok.type = GV_SQL_TOK_LE;
        } else if (lx->input[lx->pos + 1] == '>') {
            lx->pos += 2; tok.type = GV_SQL_TOK_NE;
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
        tok.text = sql_strndup(lx->input + start, len);
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
        tok.text = sql_strndup(lx->input + start, len);
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
        if (sql_kw_match(lx->input + start, len, "SELECT"))  { tok.type = GV_SQL_TOK_SELECT;  return tok; }
        if (sql_kw_match(lx->input + start, len, "FROM"))    { tok.type = GV_SQL_TOK_FROM;    return tok; }
        if (sql_kw_match(lx->input + start, len, "WHERE"))   { tok.type = GV_SQL_TOK_WHERE;   return tok; }
        if (sql_kw_match(lx->input + start, len, "AND"))     { tok.type = GV_SQL_TOK_AND;     return tok; }
        if (sql_kw_match(lx->input + start, len, "OR"))      { tok.type = GV_SQL_TOK_OR;      return tok; }
        if (sql_kw_match(lx->input + start, len, "NOT"))     { tok.type = GV_SQL_TOK_NOT;     return tok; }
        if (sql_kw_match(lx->input + start, len, "LIKE"))    { tok.type = GV_SQL_TOK_LIKE;    return tok; }
        if (sql_kw_match(lx->input + start, len, "LIMIT"))   { tok.type = GV_SQL_TOK_LIMIT;   return tok; }
        if (sql_kw_match(lx->input + start, len, "ORDER"))   { tok.type = GV_SQL_TOK_ORDER;   return tok; }
        if (sql_kw_match(lx->input + start, len, "BY"))      { tok.type = GV_SQL_TOK_BY;      return tok; }
        if (sql_kw_match(lx->input + start, len, "ASC"))     { tok.type = GV_SQL_TOK_ASC;     return tok; }
        if (sql_kw_match(lx->input + start, len, "DESC"))    { tok.type = GV_SQL_TOK_DESC;    return tok; }
        if (sql_kw_match(lx->input + start, len, "ANN"))     { tok.type = GV_SQL_TOK_ANN;     return tok; }
        if (sql_kw_match(lx->input + start, len, "DELETE"))  { tok.type = GV_SQL_TOK_DELETE;  return tok; }
        if (sql_kw_match(lx->input + start, len, "UPDATE"))  { tok.type = GV_SQL_TOK_UPDATE;  return tok; }
        if (sql_kw_match(lx->input + start, len, "SET"))     { tok.type = GV_SQL_TOK_SET;     return tok; }
        if (sql_kw_match(lx->input + start, len, "COUNT"))   { tok.type = GV_SQL_TOK_COUNT;   return tok; }
        if (sql_kw_match(lx->input + start, len, "OFFSET"))  { tok.type = GV_SQL_TOK_OFFSET;  return tok; }
        if (sql_kw_match(lx->input + start, len, "IN"))      { tok.type = GV_SQL_TOK_IN;      return tok; }
        if (sql_kw_match(lx->input + start, len, "BETWEEN")) { tok.type = GV_SQL_TOK_BETWEEN; return tok; }
        if (sql_kw_match(lx->input + start, len, "IS"))      { tok.type = GV_SQL_TOK_IS;      return tok; }
        if (sql_kw_match(lx->input + start, len, "NULL"))    { tok.type = GV_SQL_TOK_NULL;    return tok; }
        if (sql_kw_match(lx->input + start, len, "INSERT"))  { tok.type = GV_SQL_TOK_INSERT;  return tok; }
        if (sql_kw_match(lx->input + start, len, "INTO"))    { tok.type = GV_SQL_TOK_INTO;    return tok; }
        if (sql_kw_match(lx->input + start, len, "VALUES"))  { tok.type = GV_SQL_TOK_VALUES;  return tok; }
        if (sql_kw_match(lx->input + start, len, "SUM"))     { tok.type = GV_SQL_TOK_SUM;     return tok; }
        if (sql_kw_match(lx->input + start, len, "MIN"))     { tok.type = GV_SQL_TOK_MIN;     return tok; }
        if (sql_kw_match(lx->input + start, len, "MAX"))     { tok.type = GV_SQL_TOK_MAX;     return tok; }
        if (sql_kw_match(lx->input + start, len, "AVG"))     { tok.type = GV_SQL_TOK_AVG;     return tok; }

        tok.text = sql_strndup(lx->input + start, len);
        tok.type = tok.text ? GV_SQL_TOK_IDENT : GV_SQL_TOK_ERROR;
        return tok;
    }

    snprintf(lx->error, GV_SQL_ERROR_SIZE, "Unexpected character '%c' at position %zu", c, lx->pos);
    lx->pos++;
    tok.type = GV_SQL_TOK_ERROR;
    return tok;
}

/* Token buffer (pre-tokenise the full query) */

/* Cap on nested WHERE-expression productions (recursion-depth DoS guard). */
#define GV_SQL_MAX_RECURSION_DEPTH 256

typedef struct {
    GV_SQLToken *tokens;
    size_t count;
    size_t pos;
    size_t depth;   /* current WHERE-parse recursion depth (DoS guard) */
    char error[GV_SQL_ERROR_SIZE];
} GV_SQLTokenBuf;

static int sql_tokenize(GV_SQLTokenBuf *buf, const char *query)
{
    GV_SQLLexer lx;
    sql_lexer_init(&lx, query);

    buf->tokens = (GV_SQLToken *)gv_calloc(GV_SQL_MAX_TOKENS, sizeof(GV_SQLToken));
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
        GV_SQLToken tok = sql_lexer_next(&lx);
        if (tok.type == GV_SQL_TOK_ERROR) {
            snprintf(buf->error, GV_SQL_ERROR_SIZE, "Tokenization error: %s", lx.error);
            sql_token_free(&tok);
            return -1;
        }
        buf->tokens[buf->count++] = tok;
        if (tok.type == GV_SQL_TOK_EOF)
            break;
    }
    return 0;
}

static void sql_tokenbuf_free(GV_SQLTokenBuf *buf)
{
    if (!buf || !buf->tokens) return;
    for (size_t i = 0; i < buf->count; i++)
        sql_token_free(&buf->tokens[i]);
    gv_free(buf->tokens);
    buf->tokens = NULL;
    buf->count = 0;
}

static GV_SQLToken *sql_peek(GV_SQLTokenBuf *buf)
{
    if (buf->pos < buf->count)
        return &buf->tokens[buf->pos];
    return NULL;
}

static GV_SQLToken *sql_advance(GV_SQLTokenBuf *buf)
{
    if (buf->pos < buf->count)
        return &buf->tokens[buf->pos++];
    return NULL;
}

static int sql_consume_trailing_semicolons(GV_SQLTokenBuf *buf)
{
    GV_SQLToken *tok = sql_peek(buf);
    while (tok && tok->type == GV_SQL_TOK_SEMICOLON) {
        sql_advance(buf);
        tok = sql_peek(buf);
    }
    return (tok && tok->type == GV_SQL_TOK_EOF) ? 0 : -1;
}

static int sql_expect(GV_SQLTokenBuf *buf, GV_SQLTokenType type)
{
    GV_SQLToken *tok = sql_peek(buf);
    if (!tok || tok->type != type) return 0;
    buf->pos++;
    return 1;
}

/* Parsed AST types */

typedef enum {
    GV_SQL_STMT_SELECT,
    GV_SQL_STMT_DELETE,
    GV_SQL_STMT_UPDATE,
    GV_SQL_STMT_INSERT
} GV_SQLStmtType;

typedef enum {
    GV_SQL_AGG_NONE = 0,
    GV_SQL_AGG_COUNT,
    GV_SQL_AGG_SUM,
    GV_SQL_AGG_MIN,
    GV_SQL_AGG_MAX,
    GV_SQL_AGG_AVG
} GV_SQLAggKind;

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
    GV_SQL_CMP_LIKE,
    GV_SQL_CMP_IN,          /**< field IN (v1, v2, ...) */
    GV_SQL_CMP_BETWEEN,     /**< field BETWEEN low AND high */
    GV_SQL_CMP_IS_NULL,     /**< field IS NULL (metadata absent/empty) */
    GV_SQL_CMP_IS_NOT_NULL  /**< field IS NOT NULL */
} GV_SQLCmpOp;

typedef struct GV_SQLWhere {
    GV_SQLWhereType type;
    /* For CMP */
    char *field;
    GV_SQLCmpOp op;
    char *value;
    double num_value;
    int is_numeric;
    /* For BETWEEN: the upper bound (value is the lower bound) */
    char *value2;
    double num_value2;
    int is_numeric2;
    /* For IN: list of candidate values */
    char **list_values;
    double *list_nums;
    int *list_is_numeric;
    size_t list_count;
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
    int is_count;        /**< 1 if SELECT COUNT(*) (kept for compat; agg_kind==COUNT) */
    GV_SQLAggKind agg_kind;   /**< aggregate function for the projection, or NONE */
    char *agg_column;    /**< column argument for SUM/MIN/MAX/AVG (NULL for COUNT(*)) */
    int select_star;     /**< 1 if SELECT * */
    char **select_columns;
    size_t select_column_count;
    int has_ann;
    GV_SQLAnn ann;

    /* WHERE clause */
    GV_SQLWhere *where;

    /* ORDER BY */
    char *order_field;
    int order_desc;      /**< 1 if DESC, 0 if ASC */
    int has_order_ann;
    GV_SQLAnn order_ann;

    /* LIMIT / OFFSET */
    size_t limit;
    int has_limit;
    size_t offset;
    int has_offset;

    /* UPDATE SET clauses (also reused for INSERT metadata columns) */
    GV_SQLSetClause *set_clauses;
    size_t set_count;

    /* INSERT-specific */
    float *insert_vector;    /**< dense vector literal for INSERT */
    size_t insert_dim;       /**< dimension of insert_vector */
} GV_SQLStmt;

/* AST cleanup */

static void sql_where_free(GV_SQLWhere *w)
{
    if (!w) return;
    sql_where_free(w->left);
    sql_where_free(w->right);
    sql_where_free(w->child);
    gv_free(w->field);
    gv_free(w->value);
    gv_free(w->value2);
    if (w->list_values) {
        for (size_t i = 0; i < w->list_count; i++) gv_free(w->list_values[i]);
        gv_free(w->list_values);
    }
    gv_free(w->list_nums);
    gv_free(w->list_is_numeric);
    gv_free(w);
}

static void sql_stmt_free(GV_SQLStmt *s)
{
    if (!s) return;
    gv_free(s->table);
    gv_free(s->agg_column);
    gv_free(s->insert_vector);
    sql_where_free(s->where);
    gv_free(s->order_field);
    if (s->ann.query_vector) gv_free(s->ann.query_vector);
    if (s->order_ann.query_vector) gv_free(s->order_ann.query_vector);
    if (s->select_columns) {
        for (size_t i = 0; i < s->select_column_count; i++)
            gv_free(s->select_columns[i]);
        gv_free(s->select_columns);
    }
    if (s->set_clauses) {
        for (size_t i = 0; i < s->set_count; i++) {
            gv_free(s->set_clauses[i].field);
            gv_free(s->set_clauses[i].value);
        }
        gv_free(s->set_clauses);
    }
    gv_free(s);
}

/* Recursive descent parser */

/* Forward declarations */
static GV_SQLWhere *sql_parse_where_expr(GV_SQLTokenBuf *buf);

/* parse_ann_params: query=[...], k=N, metric=cosine|euclidean|dot inside parens */
static int sql_parse_ann_params(GV_SQLTokenBuf *buf, GV_SQLAnn *ann, int require_k)
{
    ann->query_vector = NULL;
    ann->query_dim = 0;
    ann->k = 10;
    ann->metric = GV_DISTANCE_COSINE;

    float tmp_vec[GV_SQL_MAX_QUERY_DIMS];
    size_t tmp_count = 0;

    while (sql_peek(buf) && sql_peek(buf)->type != GV_SQL_TOK_RPAREN) {
        GV_SQLToken *key = sql_peek(buf);
        if (!key || key->type != GV_SQL_TOK_IDENT) return -1;
        char *kname = key->text;
        sql_advance(buf);

        if (!sql_expect(buf, GV_SQL_TOK_EQ)) return -1;

        if (sql_kw_match(kname, strlen(kname), "query")) {
            if (!sql_expect(buf, GV_SQL_TOK_LBRACKET)) return -1;
            tmp_count = 0;
            while (sql_peek(buf) && sql_peek(buf)->type != GV_SQL_TOK_RBRACKET) {
                GV_SQLToken *num = sql_peek(buf);
                if (!num || num->type != GV_SQL_TOK_NUMBER) return -1;
                if (tmp_count >= GV_SQL_MAX_QUERY_DIMS) return -1;
                tmp_vec[tmp_count++] = (float)num->num_value;
                sql_advance(buf);
                if (sql_peek(buf) && sql_peek(buf)->type == GV_SQL_TOK_COMMA)
                    sql_advance(buf);
            }
            if (!sql_expect(buf, GV_SQL_TOK_RBRACKET)) return -1;
        } else if (sql_kw_match(kname, strlen(kname), "k")) {
            GV_SQLToken *num = sql_peek(buf);
            if (!num || num->type != GV_SQL_TOK_NUMBER) return -1;
            ann->k = (size_t)num->num_value;
            sql_advance(buf);
        } else if (sql_kw_match(kname, strlen(kname), "metric")) {
            GV_SQLToken *val = sql_peek(buf);
            if (!val || val->type != GV_SQL_TOK_IDENT) return -1;
            if (sql_kw_match(val->text, strlen(val->text), "cosine"))
                ann->metric = GV_DISTANCE_COSINE;
            else if (sql_kw_match(val->text, strlen(val->text), "euclidean"))
                ann->metric = GV_DISTANCE_EUCLIDEAN;
            else if (sql_kw_match(val->text, strlen(val->text), "dot"))
                ann->metric = GV_DISTANCE_DOT_PRODUCT;
            else
                return -1;
            sql_advance(buf);
        } else {
            return -1;
        }

        if (sql_peek(buf) && sql_peek(buf)->type == GV_SQL_TOK_COMMA)
            sql_advance(buf);
    }

    if (tmp_count == 0) return -1;
    if (require_k && ann->k == 0) return -1;

    ann->query_vector = (float *)gv_alloc(tmp_count * sizeof(float));
    if (!ann->query_vector) return -1;
    memcpy(ann->query_vector, tmp_vec, tmp_count * sizeof(float));
    ann->query_dim = tmp_count;
    return 0;
}

/* parse_ann: ANN(query=[...], k=N, metric=cosine|euclidean|dot) */
static int sql_parse_ann(GV_SQLTokenBuf *buf, GV_SQLAnn *ann)
{
    if (!sql_expect(buf, GV_SQL_TOK_LPAREN)) return -1;
    if (sql_parse_ann_params(buf, ann, 1) != 0) return -1;
    if (!sql_expect(buf, GV_SQL_TOK_RPAREN)) return -1;
    return 0;
}

/* parse_where primary: field CMP value | NOT expr | ( expr ) */
/* Whether a token can serve as a scalar literal in a predicate value slot. */
static int sql_tok_is_value(const GV_SQLToken *t)
{
    return t && (t->type == GV_SQL_TOK_STRING ||
                 t->type == GV_SQL_TOK_NUMBER ||
                 t->type == GV_SQL_TOK_IDENT);
}

/* field IN (v1, v2, ...) — takes ownership of `field`. */
static GV_SQLWhere *sql_parse_in_list(GV_SQLTokenBuf *buf, char *field)
{
    sql_advance(buf); /* consume IN */
    if (!sql_expect(buf, GV_SQL_TOK_LPAREN)) { gv_free(field); return NULL; }

    GV_SQLWhere *node = (GV_SQLWhere *)gv_calloc(1, sizeof(GV_SQLWhere));
    if (!node) { gv_free(field); return NULL; }
    node->type = GV_SQL_WHERE_CMP;
    node->op = GV_SQL_CMP_IN;
    node->field = field;

    size_t cap = 4;
    node->list_values = (char **)gv_calloc(cap, sizeof(char *));
    node->list_nums = (double *)gv_calloc(cap, sizeof(double));
    node->list_is_numeric = (int *)gv_calloc(cap, sizeof(int));
    if (!node->list_values || !node->list_nums || !node->list_is_numeric) {
        sql_where_free(node); return NULL;
    }
    for (;;) {
        GV_SQLToken *t = sql_peek(buf);
        if (!sql_tok_is_value(t)) { sql_where_free(node); return NULL; }
        if (node->list_count >= cap) {
            size_t ncap = cap * 2;
            char **nv = (char **)gv_realloc(node->list_values, ncap * sizeof(char *));
            if (nv) node->list_values = nv;
            double *nn = (double *)gv_realloc(node->list_nums, ncap * sizeof(double));
            if (nn) node->list_nums = nn;
            int *ni = (int *)gv_realloc(node->list_is_numeric, ncap * sizeof(int));
            if (ni) node->list_is_numeric = ni;
            if (!nv || !nn || !ni) { sql_where_free(node); return NULL; }
            cap = ncap;
        }
        node->list_values[node->list_count] = gv_dup_cstr(t->text ? t->text : "");
        if (!node->list_values[node->list_count]) { sql_where_free(node); return NULL; }
        node->list_nums[node->list_count] = t->num_value;
        node->list_is_numeric[node->list_count] = (t->type == GV_SQL_TOK_NUMBER) ? 1 : 0;
        node->list_count++;
        sql_advance(buf);
        t = sql_peek(buf);
        if (t && t->type == GV_SQL_TOK_COMMA) { sql_advance(buf); continue; }
        break;
    }
    if (!sql_expect(buf, GV_SQL_TOK_RPAREN) || node->list_count == 0) {
        sql_where_free(node); return NULL;
    }
    return node;
}

/* field BETWEEN low AND high — takes ownership of `field`. */
static GV_SQLWhere *sql_parse_between(GV_SQLTokenBuf *buf, char *field)
{
    sql_advance(buf); /* consume BETWEEN */
    GV_SQLWhere *node = (GV_SQLWhere *)gv_calloc(1, sizeof(GV_SQLWhere));
    if (!node) { gv_free(field); return NULL; }
    node->type = GV_SQL_WHERE_CMP;
    node->op = GV_SQL_CMP_BETWEEN;
    node->field = field;

    GV_SQLToken *t = sql_peek(buf);
    if (!sql_tok_is_value(t)) { sql_where_free(node); return NULL; }
    node->value = gv_dup_cstr(t->text ? t->text : "");
    node->num_value = t->num_value;
    node->is_numeric = (t->type == GV_SQL_TOK_NUMBER);
    if (!node->value) { sql_where_free(node); return NULL; }
    sql_advance(buf);

    if (!sql_expect(buf, GV_SQL_TOK_AND)) { sql_where_free(node); return NULL; }

    t = sql_peek(buf);
    if (!sql_tok_is_value(t)) { sql_where_free(node); return NULL; }
    node->value2 = gv_dup_cstr(t->text ? t->text : "");
    node->num_value2 = t->num_value;
    node->is_numeric2 = (t->type == GV_SQL_TOK_NUMBER);
    if (!node->value2) { sql_where_free(node); return NULL; }
    sql_advance(buf);
    return node;
}

/* field IS [NOT] NULL — takes ownership of `field`. */
static GV_SQLWhere *sql_parse_is_null(GV_SQLTokenBuf *buf, char *field)
{
    sql_advance(buf); /* consume IS */
    int negate = 0;
    GV_SQLToken *t = sql_peek(buf);
    if (t && t->type == GV_SQL_TOK_NOT) { negate = 1; sql_advance(buf); }
    if (!sql_expect(buf, GV_SQL_TOK_NULL)) { gv_free(field); return NULL; }

    GV_SQLWhere *node = (GV_SQLWhere *)gv_calloc(1, sizeof(GV_SQLWhere));
    if (!node) { gv_free(field); return NULL; }
    node->type = GV_SQL_WHERE_CMP;
    node->op = negate ? GV_SQL_CMP_IS_NOT_NULL : GV_SQL_CMP_IS_NULL;
    node->field = field;
    return node;
}

static GV_SQLWhere *sql_parse_where_primary_body(GV_SQLTokenBuf *buf);

/* Recursion-depth-guarded wrapper: all WHERE recursion funnels through
 * sql_parse_where_primary (expr->and->primary, primary->NOT->primary,
 * primary->'('->expr), so guarding here bounds total stack depth. */
static GV_SQLWhere *sql_parse_where_primary(GV_SQLTokenBuf *buf)
{
    if (buf->depth >= GV_SQL_MAX_RECURSION_DEPTH) {
        snprintf(buf->error, sizeof(buf->error),
                 "WHERE expression nesting too deep");
        return NULL;
    }
    buf->depth++;
    GV_SQLWhere *node = sql_parse_where_primary_body(buf);
    buf->depth--;
    return node;
}

static GV_SQLWhere *sql_parse_where_primary_body(GV_SQLTokenBuf *buf)
{
    GV_SQLToken *tok = sql_peek(buf);
    if (!tok) return NULL;

    /* NOT */
    if (tok->type == GV_SQL_TOK_NOT) {
        sql_advance(buf);
        GV_SQLWhere *child = sql_parse_where_primary(buf);
        if (!child) return NULL;
        GV_SQLWhere *node = (GV_SQLWhere *)gv_calloc(1, sizeof(GV_SQLWhere));
        if (!node) { sql_where_free(child); return NULL; }
        node->type = GV_SQL_WHERE_NOT;
        node->child = child;
        return node;
    }

    /* Parenthesized expression */
    if (tok->type == GV_SQL_TOK_LPAREN) {
        sql_advance(buf);
        GV_SQLWhere *expr = sql_parse_where_expr(buf);
        if (!expr) return NULL;
        if (!sql_expect(buf, GV_SQL_TOK_RPAREN)) {
            sql_where_free(expr);
            return NULL;
        }
        return expr;
    }

    /* field CMP value */
    if (tok->type != GV_SQL_TOK_IDENT) return NULL;
    char *field = gv_dup_cstr(tok->text);
    if (!field) return NULL;
    sql_advance(buf);

    tok = sql_peek(buf);
    if (!tok) { gv_free(field); return NULL; }

    /* Extended predicates: field IN (...) | field BETWEEN a AND b | field IS [NOT] NULL */
    if (tok->type == GV_SQL_TOK_IN)      return sql_parse_in_list(buf, field);
    if (tok->type == GV_SQL_TOK_BETWEEN) return sql_parse_between(buf, field);
    if (tok->type == GV_SQL_TOK_IS)      return sql_parse_is_null(buf, field);

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
        gv_free(field);
        return NULL;
    }
    sql_advance(buf);

    tok = sql_peek(buf);
    if (!tok || (tok->type != GV_SQL_TOK_STRING &&
                 tok->type != GV_SQL_TOK_NUMBER &&
                 tok->type != GV_SQL_TOK_IDENT)) {
        gv_free(field);
        return NULL;
    }

    GV_SQLWhere *node = (GV_SQLWhere *)gv_calloc(1, sizeof(GV_SQLWhere));
    if (!node) { gv_free(field); return NULL; }
    node->type = GV_SQL_WHERE_CMP;
    node->field = field;
    node->op = op;

    if (tok->type == GV_SQL_TOK_NUMBER) {
        node->value = gv_dup_cstr(tok->text);
        node->num_value = tok->num_value;
        node->is_numeric = 1;
    } else {
        node->value = gv_dup_cstr(tok->text ? tok->text : "");
        node->is_numeric = 0;
    }
    sql_advance(buf);

    if (!node->value) {
        sql_where_free(node);
        return NULL;
    }
    return node;
}

/* parse_where AND: primary (AND primary)* */
static GV_SQLWhere *sql_parse_where_and(GV_SQLTokenBuf *buf)
{
    GV_SQLWhere *left = sql_parse_where_primary(buf);
    if (!left) return NULL;

    while (sql_peek(buf) && sql_peek(buf)->type == GV_SQL_TOK_AND) {
        sql_advance(buf);
        GV_SQLWhere *right = sql_parse_where_primary(buf);
        if (!right) { sql_where_free(left); return NULL; }
        GV_SQLWhere *node = (GV_SQLWhere *)gv_calloc(1, sizeof(GV_SQLWhere));
        if (!node) { sql_where_free(left); sql_where_free(right); return NULL; }
        node->type = GV_SQL_WHERE_AND;
        node->left = left;
        node->right = right;
        left = node;
    }
    return left;
}

/* parse_where OR: and_expr (OR and_expr)* */
static GV_SQLWhere *sql_parse_where_expr(GV_SQLTokenBuf *buf)
{
    GV_SQLWhere *left = sql_parse_where_and(buf);
    if (!left) return NULL;

    while (sql_peek(buf) && sql_peek(buf)->type == GV_SQL_TOK_OR) {
        sql_advance(buf);
        GV_SQLWhere *right = sql_parse_where_and(buf);
        if (!right) { sql_where_free(left); return NULL; }
        GV_SQLWhere *node = (GV_SQLWhere *)gv_calloc(1, sizeof(GV_SQLWhere));
        if (!node) { sql_where_free(left); sql_where_free(right); return NULL; }
        node->type = GV_SQL_WHERE_OR;
        node->left = left;
        node->right = right;
        left = node;
    }
    return left;
}

/* parse_select: SELECT (* | COUNT(*)) FROM table [ANN(...)] [WHERE ...] [ORDER BY ...] [LIMIT n] */
static GV_SQLStmt *sql_parse_select(GV_SQLTokenBuf *buf)
{
    /* SELECT already consumed */
    GV_SQLStmt *stmt = (GV_SQLStmt *)gv_calloc(1, sizeof(GV_SQLStmt));
    if (!stmt) return NULL;
    stmt->type = GV_SQL_STMT_SELECT;

    GV_SQLToken *tok = sql_peek(buf);
    if (!tok) { sql_stmt_free(stmt); return NULL; }

    /* SELECT COUNT(*) | SUM|MIN|MAX|AVG(col) | * | field[,field...] */
    if (tok->type == GV_SQL_TOK_COUNT) {
        sql_advance(buf);
        if (!sql_expect(buf, GV_SQL_TOK_LPAREN) ||
            !sql_expect(buf, GV_SQL_TOK_STAR) ||
            !sql_expect(buf, GV_SQL_TOK_RPAREN)) {
            sql_stmt_free(stmt);
            return NULL;
        }
        stmt->is_count = 1;
        stmt->agg_kind = GV_SQL_AGG_COUNT;
    } else if (tok->type == GV_SQL_TOK_SUM || tok->type == GV_SQL_TOK_MIN ||
               tok->type == GV_SQL_TOK_MAX || tok->type == GV_SQL_TOK_AVG) {
        GV_SQLAggKind ak = (tok->type == GV_SQL_TOK_SUM) ? GV_SQL_AGG_SUM :
                           (tok->type == GV_SQL_TOK_MIN) ? GV_SQL_AGG_MIN :
                           (tok->type == GV_SQL_TOK_MAX) ? GV_SQL_AGG_MAX : GV_SQL_AGG_AVG;
        sql_advance(buf);
        if (!sql_expect(buf, GV_SQL_TOK_LPAREN)) { sql_stmt_free(stmt); return NULL; }
        GV_SQLToken *ct = sql_peek(buf);
        if (!ct || ct->type != GV_SQL_TOK_IDENT) { sql_stmt_free(stmt); return NULL; }
        stmt->agg_column = gv_dup_cstr(ct->text);
        if (!stmt->agg_column) { sql_stmt_free(stmt); return NULL; }
        sql_advance(buf);
        if (!sql_expect(buf, GV_SQL_TOK_RPAREN)) { sql_stmt_free(stmt); return NULL; }
        stmt->agg_kind = ak;
    } else if (tok->type == GV_SQL_TOK_STAR) {
        sql_advance(buf);
        stmt->select_star = 1;
    } else if (tok->type == GV_SQL_TOK_IDENT) {
        size_t cap = 4;
        stmt->select_columns = (char **)gv_calloc(cap, sizeof(char *));
        if (!stmt->select_columns) { sql_stmt_free(stmt); return NULL; }
        for (;;) {
            tok = sql_peek(buf);
            if (!tok || tok->type != GV_SQL_TOK_IDENT) {
                sql_stmt_free(stmt);
                return NULL;
            }
            if (stmt->select_column_count >= cap) {
                cap *= 2;
                char **tmp = (char **)gv_realloc(stmt->select_columns,
                                              cap * sizeof(char *));
                if (!tmp) { sql_stmt_free(stmt); return NULL; }
                stmt->select_columns = tmp;
            }
            stmt->select_columns[stmt->select_column_count] =
                gv_dup_cstr(tok->text);
            if (!stmt->select_columns[stmt->select_column_count]) {
                sql_stmt_free(stmt);
                return NULL;
            }
            stmt->select_column_count++;
            sql_advance(buf);
            tok = sql_peek(buf);
            if (tok && tok->type == GV_SQL_TOK_COMMA) {
                sql_advance(buf);
                continue;
            }
            break;
        }
    } else {
        sql_stmt_free(stmt);
        return NULL;
    }

    /* FROM table */
    if (!sql_expect(buf, GV_SQL_TOK_FROM)) { sql_stmt_free(stmt); return NULL; }
    tok = sql_peek(buf);
    if (!tok || tok->type != GV_SQL_TOK_IDENT) { sql_stmt_free(stmt); return NULL; }
    stmt->table = gv_dup_cstr(tok->text);
    sql_advance(buf);
    if (!stmt->table) { sql_stmt_free(stmt); return NULL; }

    /* Optional ANN(...) */
    tok = sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_ANN) {
        sql_advance(buf);
        if (sql_parse_ann(buf, &stmt->ann) != 0) {
            sql_stmt_free(stmt);
            return NULL;
        }
        stmt->has_ann = 1;
    }

    /* Optional WHERE */
    tok = sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_WHERE) {
        sql_advance(buf);
        stmt->where = sql_parse_where_expr(buf);
        if (!stmt->where) { sql_stmt_free(stmt); return NULL; }
    }

    /* Optional ORDER BY field [ASC|DESC] */
    tok = sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_ORDER) {
        sql_advance(buf);
        if (!sql_expect(buf, GV_SQL_TOK_BY)) { sql_stmt_free(stmt); return NULL; }
        tok = sql_peek(buf);
        if (!tok || tok->type != GV_SQL_TOK_IDENT) { sql_stmt_free(stmt); return NULL; }
        stmt->order_field = gv_dup_cstr(tok->text);
        sql_advance(buf);
        if (!stmt->order_field) { sql_stmt_free(stmt); return NULL; }

        tok = sql_peek(buf);
        if (tok && tok->type == GV_SQL_TOK_LPAREN) {
            if (sql_kw_match(stmt->order_field, strlen(stmt->order_field),
                             "vector_distance")) {
                sql_advance(buf);
                if (sql_parse_ann_params(buf, &stmt->order_ann, 0) != 0) {
                    sql_stmt_free(stmt);
                    return NULL;
                }
                if (!sql_expect(buf, GV_SQL_TOK_RPAREN)) {
                    sql_stmt_free(stmt);
                    return NULL;
                }
                stmt->has_order_ann = 1;
            } else {
                int depth = 0;
                do {
                    tok = sql_peek(buf);
                    if (!tok) { sql_stmt_free(stmt); return NULL; }
                    if (tok->type == GV_SQL_TOK_LPAREN) depth++;
                    else if (tok->type == GV_SQL_TOK_RPAREN) depth--;
                    sql_advance(buf);
                } while (depth > 0);
            }
        }

        tok = sql_peek(buf);
        if (tok && tok->type == GV_SQL_TOK_DESC) {
            stmt->order_desc = 1;
            sql_advance(buf);
        } else if (tok && tok->type == GV_SQL_TOK_ASC) {
            stmt->order_desc = 0;
            sql_advance(buf);
        }
    }

    /* Optional LIMIT n */
    tok = sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_LIMIT) {
        sql_advance(buf);
        tok = sql_peek(buf);
        if (!tok || tok->type != GV_SQL_TOK_NUMBER) { sql_stmt_free(stmt); return NULL; }
        stmt->limit = (size_t)tok->num_value;
        stmt->has_limit = 1;
        sql_advance(buf);
    }

    /* Optional OFFSET n (pagination; may appear with or without LIMIT) */
    tok = sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_OFFSET) {
        sql_advance(buf);
        tok = sql_peek(buf);
        if (!tok || tok->type != GV_SQL_TOK_NUMBER) { sql_stmt_free(stmt); return NULL; }
        stmt->offset = (size_t)tok->num_value;
        stmt->has_offset = 1;
        sql_advance(buf);
    }

    return stmt;
}

/* parse_delete: DELETE FROM table WHERE ... */
static GV_SQLStmt *sql_parse_delete(GV_SQLTokenBuf *buf)
{
    /* DELETE already consumed */
    GV_SQLStmt *stmt = (GV_SQLStmt *)gv_calloc(1, sizeof(GV_SQLStmt));
    if (!stmt) return NULL;
    stmt->type = GV_SQL_STMT_DELETE;

    if (!sql_expect(buf, GV_SQL_TOK_FROM)) { sql_stmt_free(stmt); return NULL; }
    GV_SQLToken *tok = sql_peek(buf);
    if (!tok || tok->type != GV_SQL_TOK_IDENT) { sql_stmt_free(stmt); return NULL; }
    stmt->table = gv_dup_cstr(tok->text);
    sql_advance(buf);
    if (!stmt->table) { sql_stmt_free(stmt); return NULL; }

    if (!sql_expect(buf, GV_SQL_TOK_WHERE)) { sql_stmt_free(stmt); return NULL; }
    stmt->where = sql_parse_where_expr(buf);
    if (!stmt->where) { sql_stmt_free(stmt); return NULL; }

    return stmt;
}

/* parse_update: UPDATE table SET field=val, ... WHERE ... */
static GV_SQLStmt *sql_parse_update(GV_SQLTokenBuf *buf)
{
    /* UPDATE already consumed */
    GV_SQLStmt *stmt = (GV_SQLStmt *)gv_calloc(1, sizeof(GV_SQLStmt));
    if (!stmt) return NULL;
    stmt->type = GV_SQL_STMT_UPDATE;

    GV_SQLToken *tok = sql_peek(buf);
    if (!tok || tok->type != GV_SQL_TOK_IDENT) { sql_stmt_free(stmt); return NULL; }
    stmt->table = gv_dup_cstr(tok->text);
    sql_advance(buf);
    if (!stmt->table) { sql_stmt_free(stmt); return NULL; }

    if (!sql_expect(buf, GV_SQL_TOK_SET)) { sql_stmt_free(stmt); return NULL; }

    /* Parse SET clauses: field = value [, field = value ...] */
    size_t cap = 8;
    stmt->set_clauses = (GV_SQLSetClause *)gv_calloc(cap, sizeof(GV_SQLSetClause));
    if (!stmt->set_clauses) { sql_stmt_free(stmt); return NULL; }
    stmt->set_count = 0;

    for (;;) {
        tok = sql_peek(buf);
        if (!tok || tok->type != GV_SQL_TOK_IDENT) break;
        char *field = gv_dup_cstr(tok->text);
        sql_advance(buf);
        if (!field) { sql_stmt_free(stmt); return NULL; }

        if (!sql_expect(buf, GV_SQL_TOK_EQ)) { gv_free(field); sql_stmt_free(stmt); return NULL; }

        tok = sql_peek(buf);
        if (!tok || (tok->type != GV_SQL_TOK_STRING &&
                     tok->type != GV_SQL_TOK_NUMBER &&
                     tok->type != GV_SQL_TOK_IDENT)) {
            gv_free(field);
            sql_stmt_free(stmt);
            return NULL;
        }
        char *value = gv_dup_cstr(tok->text ? tok->text : "");
        sql_advance(buf);
        if (!value) { gv_free(field); sql_stmt_free(stmt); return NULL; }

        if (stmt->set_count >= cap) {
            cap *= 2;
            GV_SQLSetClause *tmp = (GV_SQLSetClause *)gv_realloc(
                stmt->set_clauses, cap * sizeof(GV_SQLSetClause));
            if (!tmp) { gv_free(field); gv_free(value); sql_stmt_free(stmt); return NULL; }
            stmt->set_clauses = tmp;
        }
        stmt->set_clauses[stmt->set_count].field = field;
        stmt->set_clauses[stmt->set_count].value = value;
        stmt->set_count++;

        /* Optional comma */
        if (sql_peek(buf) && sql_peek(buf)->type == GV_SQL_TOK_COMMA)
            sql_advance(buf);
        else
            break;
    }

    if (stmt->set_count == 0) { sql_stmt_free(stmt); return NULL; }

    /* WHERE clause (required for UPDATE) */
    if (!sql_expect(buf, GV_SQL_TOK_WHERE)) { sql_stmt_free(stmt); return NULL; }
    stmt->where = sql_parse_where_expr(buf);
    if (!stmt->where) { sql_stmt_free(stmt); return NULL; }

    return stmt;
}

/* parse_insert: INSERT INTO table [(col, ...)] VALUES (value, ...)
 * Exactly one value must be a [f,f,...] vector literal (the embedding); any
 * other positions are stored as string metadata using their column names
 * (so scalar values require a preceding column list). */
static GV_SQLStmt *sql_parse_insert(GV_SQLTokenBuf *buf)
{
    /* INSERT already consumed */
    GV_SQLStmt *stmt = (GV_SQLStmt *)gv_calloc(1, sizeof(GV_SQLStmt));
    if (!stmt) return NULL;
    stmt->type = GV_SQL_STMT_INSERT;

    if (!sql_expect(buf, GV_SQL_TOK_INTO)) { sql_stmt_free(stmt); return NULL; }
    GV_SQLToken *tok = sql_peek(buf);
    if (!tok || tok->type != GV_SQL_TOK_IDENT) { sql_stmt_free(stmt); return NULL; }
    stmt->table = gv_dup_cstr(tok->text);
    if (!stmt->table) { sql_stmt_free(stmt); return NULL; }
    sql_advance(buf);

    /* Optional column list */
    char **cols = NULL;
    size_t col_count = 0, col_cap = 0, meta_cap = 0;
    tok = sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_LPAREN) {
        sql_advance(buf);
        for (;;) {
            tok = sql_peek(buf);
            if (!tok || tok->type != GV_SQL_TOK_IDENT) goto fail;
            if (col_count >= col_cap) {
                size_t nc = col_cap ? col_cap * 2 : 4;
                char **t = (char **)gv_realloc(cols, nc * sizeof(char *));
                if (!t) goto fail;
                cols = t; col_cap = nc;
            }
            cols[col_count] = gv_dup_cstr(tok->text);
            if (!cols[col_count]) goto fail;
            col_count++;
            sql_advance(buf);
            tok = sql_peek(buf);
            if (tok && tok->type == GV_SQL_TOK_COMMA) { sql_advance(buf); continue; }
            break;
        }
        if (!sql_expect(buf, GV_SQL_TOK_RPAREN)) goto fail;
    }

    if (!sql_expect(buf, GV_SQL_TOK_VALUES)) goto fail;
    if (!sql_expect(buf, GV_SQL_TOK_LPAREN)) goto fail;

    size_t vpos = 0;
    for (;;) {
        tok = sql_peek(buf);
        if (!tok) goto fail;
        if (tok->type == GV_SQL_TOK_LBRACKET) {
            if (stmt->insert_vector) goto fail; /* only one vector literal permitted */
            sql_advance(buf);
            float tmp[GV_SQL_MAX_QUERY_DIMS];
            size_t n = 0;
            while (sql_peek(buf) && sql_peek(buf)->type != GV_SQL_TOK_RBRACKET) {
                GV_SQLToken *num = sql_peek(buf);
                if (!num || num->type != GV_SQL_TOK_NUMBER || n >= GV_SQL_MAX_QUERY_DIMS) goto fail;
                tmp[n++] = (float)num->num_value;
                sql_advance(buf);
                if (sql_peek(buf) && sql_peek(buf)->type == GV_SQL_TOK_COMMA) sql_advance(buf);
            }
            if (!sql_expect(buf, GV_SQL_TOK_RBRACKET) || n == 0) goto fail;
            stmt->insert_vector = (float *)gv_alloc(n * sizeof(float));
            if (!stmt->insert_vector) goto fail;
            memcpy(stmt->insert_vector, tmp, n * sizeof(float));
            stmt->insert_dim = n;
        } else if (tok->type == GV_SQL_TOK_STRING || tok->type == GV_SQL_TOK_NUMBER ||
                   tok->type == GV_SQL_TOK_IDENT) {
            const char *cname = (vpos < col_count) ? cols[vpos] : NULL;
            if (!cname) goto fail; /* scalar value with no column name to bind to */
            if (stmt->set_count >= meta_cap) {
                size_t nc = meta_cap ? meta_cap * 2 : 4;
                GV_SQLSetClause *t = (GV_SQLSetClause *)gv_realloc(
                    stmt->set_clauses, nc * sizeof(GV_SQLSetClause));
                if (!t) goto fail;
                stmt->set_clauses = t; meta_cap = nc;
            }
            stmt->set_clauses[stmt->set_count].field = gv_dup_cstr(cname);
            stmt->set_clauses[stmt->set_count].value = gv_dup_cstr(tok->text ? tok->text : "");
            if (!stmt->set_clauses[stmt->set_count].field ||
                !stmt->set_clauses[stmt->set_count].value) goto fail;
            stmt->set_count++;
            sql_advance(buf);
        } else {
            goto fail;
        }
        vpos++;
        tok = sql_peek(buf);
        if (tok && tok->type == GV_SQL_TOK_COMMA) { sql_advance(buf); continue; }
        break;
    }
    if (!sql_expect(buf, GV_SQL_TOK_RPAREN)) goto fail;
    if (col_count && vpos != col_count) goto fail; /* column/value arity mismatch */
    if (!stmt->insert_vector) goto fail;            /* a vector literal is required */

    for (size_t i = 0; i < col_count; i++) gv_free(cols[i]);
    gv_free(cols);
    return stmt;

fail:
    for (size_t i = 0; i < col_count; i++) gv_free(cols[i]);
    gv_free(cols);
    sql_stmt_free(stmt);
    return NULL;
}

/* Top-level parser */
static GV_SQLStmt *sql_parse(GV_SQLTokenBuf *buf)
{
    GV_SQLToken *tok = sql_peek(buf);
    if (!tok) return NULL;

    switch (tok->type) {
    case GV_SQL_TOK_SELECT:
        sql_advance(buf);
        return sql_parse_select(buf);
    case GV_SQL_TOK_DELETE:
        sql_advance(buf);
        return sql_parse_delete(buf);
    case GV_SQL_TOK_UPDATE:
        sql_advance(buf);
        return sql_parse_update(buf);
    case GV_SQL_TOK_INSERT:
        sql_advance(buf);
        return sql_parse_insert(buf);
    default:
        snprintf(buf->error, GV_SQL_ERROR_SIZE,
                 "Expected SELECT, INSERT, DELETE, or UPDATE; got '%s'",
                 tok->text ? tok->text : "(unknown)");
        return NULL;
    }
}

/* WHERE clause evaluator (against vector metadata) */

static int sql_eval_where(const GV_SQLWhere *w, const GV_Vector *vec)
{
    if (!w) return 1; /* no WHERE => match all */

    switch (w->type) {
    case GV_SQL_WHERE_AND: {
        int l = sql_eval_where(w->left, vec);
        if (l <= 0) return l;
        return sql_eval_where(w->right, vec);
    }
    case GV_SQL_WHERE_OR: {
        int l = sql_eval_where(w->left, vec);
        if (l < 0) return l;
        if (l == 1) return 1;
        return sql_eval_where(w->right, vec);
    }
    case GV_SQL_WHERE_NOT: {
        int v = sql_eval_where(w->child, vec);
        if (v < 0) return v;
        return v ? 0 : 1;
    }
    case GV_SQL_WHERE_CMP: {
        const char *meta_val = vector_get_metadata(vec, w->field);

        /* Presence predicates define their own behaviour when the field is absent. */
        if (w->op == GV_SQL_CMP_IS_NULL)     return meta_val ? 0 : 1;
        if (w->op == GV_SQL_CMP_IS_NOT_NULL) return meta_val ? 1 : 0;

        if (!meta_val) return 0; /* every other predicate fails on an absent field */

        if (w->op == GV_SQL_CMP_IN) {
            for (size_t i = 0; i < w->list_count; i++) {
                if (w->list_is_numeric[i]) {
                    char *endptr = NULL;
                    double v = strtod(meta_val, &endptr);
                    if (endptr != meta_val && v == w->list_nums[i]) return 1;
                } else if (strcmp(meta_val, w->list_values[i]) == 0) {
                    return 1;
                }
            }
            return 0;
        }

        if (w->op == GV_SQL_CMP_BETWEEN) {
            if (w->is_numeric && w->is_numeric2) {
                char *endptr = NULL;
                double v = strtod(meta_val, &endptr);
                if (endptr == meta_val) return 0;
                return (v >= w->num_value && v <= w->num_value2) ? 1 : 0;
            }
            return (strcmp(meta_val, w->value) >= 0 &&
                    strcmp(meta_val, w->value2) <= 0) ? 1 : 0;
        }

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
                char *tmp = sql_strndup(pattern, plen);
                if (!tmp) return -1;
                int found = (strstr(meta_val, tmp) != NULL);
                gv_free(tmp);
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

/* Metadata-to-JSON serialiser (lightweight, no dependency on json.h) */

/* Ensure *buf has room for `extra` more bytes; grows via realloc.
 * Returns 1 on success, 0 on OOM (buf is freed on failure by caller). */
static int sql_json_reserve(char **buf, size_t *cap, size_t len, size_t extra)
{
    size_t needed = len + extra;
    if (needed <= *cap) return 1;
    size_t nc = *cap ? *cap : 256;
    while (nc < needed) nc *= 2;
    char *t = (char *)gv_realloc(*buf, nc);
    if (!t) return 0;
    *buf = t;
    *cap = nc;
    return 1;
}

/* Append `s` to the JSON buffer, escaping ", \, and control chars per RFC 8259.
 * Returns 1 on success, 0 on OOM. */
static int sql_json_append_escaped(char **buf, size_t *cap, size_t *len, const char *s)
{
    if (!s) return 1;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        /* Worst case one input byte expands to 6 output bytes (\u00XX). */
        if (!sql_json_reserve(buf, cap, *len, 6)) return 0;
        switch (c) {
        case '"':  (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = '"';  break;
        case '\\': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = '\\'; break;
        case '\b': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'b';  break;
        case '\f': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'f';  break;
        case '\n': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'n';  break;
        case '\r': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'r';  break;
        case '\t': (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 't';  break;
        default:
            if (c < 0x20) {
                static const char hex[] = "0123456789abcdef";
                (*buf)[(*len)++] = '\\'; (*buf)[(*len)++] = 'u';
                (*buf)[(*len)++] = '0';  (*buf)[(*len)++] = '0';
                (*buf)[(*len)++] = hex[(c >> 4) & 0xF];
                (*buf)[(*len)++] = hex[c & 0xF];
            } else {
                (*buf)[(*len)++] = (char)c;
            }
            break;
        }
    }
    return 1;
}

static char *sql_metadata_to_json(const GV_Metadata *meta)
{
    if (!meta) {
        char *s = (char *)gv_alloc(3);
        if (s) { s[0] = '{'; s[1] = '}'; s[2] = '\0'; }
        return s;
    }

    /* Build JSON string incrementally */
    size_t cap = 256;
    size_t len = 0;
    char *buf = (char *)gv_alloc(cap);
    if (!buf) return NULL;
    buf[len++] = '{';

    int first = 1;
    for (const GV_Metadata *m = meta; m; m = m->next) {
        if (!first) {
            if (!sql_json_reserve(&buf, &cap, len, 1)) { gv_free(buf); return NULL; }
            buf[len++] = ',';
        }
        first = 0;

        /* "key":"value" — key and value escaped for valid JSON. */
        if (!sql_json_reserve(&buf, &cap, len, 1)) { gv_free(buf); return NULL; }
        buf[len++] = '"';
        if (!sql_json_append_escaped(&buf, &cap, &len, m->key)) { gv_free(buf); return NULL; }
        if (!sql_json_reserve(&buf, &cap, len, 3)) { gv_free(buf); return NULL; }
        buf[len++] = '"';
        buf[len++] = ':';
        buf[len++] = '"';
        if (!sql_json_append_escaped(&buf, &cap, &len, m->value)) { gv_free(buf); return NULL; }
        if (!sql_json_reserve(&buf, &cap, len, 1)) { gv_free(buf); return NULL; }
        buf[len++] = '"';
    }

    if (!sql_json_reserve(&buf, &cap, len, 2)) { gv_free(buf); return NULL; }
    buf[len++] = '}';
    buf[len] = '\0';
    return buf;
}

/* Engine internals */

struct GV_SQLEngine {
    GV_Database *db;
    pthread_mutex_t mutex;
    char last_error[GV_SQL_ERROR_SIZE];
};

static void sql_set_error(GV_SQLEngine *eng, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(eng->last_error, GV_SQL_ERROR_SIZE, fmt, ap);
    va_end(ap);
}

typedef struct {
    size_t index;
    float distance;
    int has_distance;
} GV_SQLRow;

static const GV_SQLAnn *sql_order_distance_ann(const GV_SQLStmt *stmt)
{
    if (stmt->has_order_ann) return &stmt->order_ann;
    if (stmt->order_field &&
        sql_kw_match(stmt->order_field, strlen(stmt->order_field),
                     "vector_distance") &&
        stmt->has_ann)
        return &stmt->ann;
    return NULL;
}

static float sql_row_vector_distance(GV_Database *db, const GV_SQLStmt *stmt,
                                     const GV_SQLRow *row)
{
    const GV_SQLAnn *ann = sql_order_distance_ann(stmt);
    if (!ann || !ann->query_vector || ann->query_dim == 0) {
        return row->has_distance ? row->distance : 0.0f;
    }

    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) return row->has_distance ? row->distance : 0.0f;

    GV_Vector view;
    if (soa_storage_get_vector_view(soa, row->index, &view) != 0)
        return row->has_distance ? row->distance : 0.0f;

    GV_Vector query;
    query.data = (float *)ann->query_vector;
    query.dimension = ann->query_dim;
    query.metadata = NULL;
    return distance(&view, &query, ann->metric);
}

static int sql_get_metadata_value(GV_Database *db, size_t index,
                                  const char *field, char *out, size_t out_size)
{
    if (!out || out_size == 0) return -1;
    out[0] = '\0';

    GV_SoAStorage *soa = db->soa_storage;
    if (!soa || !field) return -1;

    if (sql_kw_match(field, strlen(field), "index")) {
        snprintf(out, out_size, "%zu", index);
        return 0;
    }

    GV_Vector view;
    if (soa_storage_get_vector_view(soa, index, &view) != 0) return -1;

    if (sql_kw_match(field, strlen(field), "metadata")) {
        char *json = sql_metadata_to_json(view.metadata);
        if (!json) return -1;
        snprintf(out, out_size, "%s", json);
        gv_free(json);
        return 0;
    }

    const char *val = vector_get_metadata(&view, field);
    if (!val) return -1;
    snprintf(out, out_size, "%s", val);
    return 0;
}

static int sql_compare_rows(GV_Database *db, const GV_SQLStmt *stmt,
                            const GV_SQLRow *a, const GV_SQLRow *b)
{
    if (!stmt->order_field) return 0;

    if (sql_order_distance_ann(stmt) ||
        sql_kw_match(stmt->order_field, strlen(stmt->order_field), "distance")) {
        float da = sql_row_vector_distance(db, stmt, a);
        float dbv = sql_row_vector_distance(db, stmt, b);
        if (da < dbv) return stmt->order_desc ? 1 : -1;
        if (da > dbv) return stmt->order_desc ? -1 : 1;
        return 0;
    }

    if (sql_kw_match(stmt->order_field, strlen(stmt->order_field), "index")) {
        if (a->index < b->index) return stmt->order_desc ? 1 : -1;
        if (a->index > b->index) return stmt->order_desc ? -1 : 1;
        return 0;
    }

    char va[512], vb[512];
    if (sql_get_metadata_value(db, a->index, stmt->order_field, va, sizeof(va)) != 0)
        va[0] = '\0';
    if (sql_get_metadata_value(db, b->index, stmt->order_field, vb, sizeof(vb)) != 0)
        vb[0] = '\0';

    char *ea = NULL, *eb = NULL;
    double na = strtod(va, &ea);
    double nb = strtod(vb, &eb);
    if (ea > va && eb > vb) {
        if (na < nb) return stmt->order_desc ? 1 : -1;
        if (na > nb) return stmt->order_desc ? -1 : 1;
        return 0;
    }

    int cmp = strcmp(va, vb);
    if (cmp < 0) return stmt->order_desc ? 1 : -1;
    if (cmp > 0) return stmt->order_desc ? -1 : 1;
    return 0;
}

static void sql_sort_rows(GV_Database *db, const GV_SQLStmt *stmt,
                          GV_SQLRow *rows, size_t count)
{
    if (!stmt->order_field || count < 2) return;
    for (size_t i = 0; i + 1 < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            if (sql_compare_rows(db, stmt, &rows[i], &rows[j]) > 0) {
                GV_SQLRow tmp = rows[i];
                rows[i] = rows[j];
                rows[j] = tmp;
            }
        }
    }
}

static void sql_apply_row_limit(GV_SQLRow *rows, size_t *count,
                                const GV_SQLStmt *stmt)
{
    /* OFFSET: drop the first `offset` rows (GV_SQLRow is POD, so shifting is safe). */
    if (stmt->has_offset && stmt->offset > 0) {
        if (stmt->offset >= *count) { *count = 0; return; }
        size_t remaining = *count - stmt->offset;
        memmove(rows, rows + stmt->offset, remaining * sizeof(GV_SQLRow));
        *count = remaining;
    }
    /* LIMIT: cap the row count. */
    if (stmt->has_limit && *count > stmt->limit) {
        *count = stmt->limit;
    }
}

static char *sql_dup_cell_value(GV_Database *db, const GV_SQLStmt *stmt,
                                const GV_SQLRow *row, const char *col,
                                int has_distances)
{
    char buf[4096];
    if (sql_kw_match(col, strlen(col), "index")) {
        snprintf(buf, sizeof(buf), "%zu", row->index);
        return gv_dup_cstr(buf);
    }
    if (sql_kw_match(col, strlen(col), "distance")) {
        float dist = row->has_distance ? row->distance
                                       : sql_row_vector_distance(db, stmt, row);
        (void)has_distances;
        snprintf(buf, sizeof(buf), "%g", (double)dist);
        return gv_dup_cstr(buf);
    }
    if (sql_kw_match(col, strlen(col), "metadata")) {
        GV_SoAStorage *soa = db->soa_storage;
        if (!soa) return gv_dup_cstr("{}");
        GV_Metadata *meta = soa_storage_get_metadata(soa, row->index);
        return sql_metadata_to_json(meta);
    }
    if (sql_get_metadata_value(db, row->index, col, buf, sizeof(buf)) != 0)
        return gv_dup_cstr("");
    return gv_dup_cstr(buf);
}

static size_t sql_default_column_count(const GV_SQLStmt *stmt, int has_distances)
{
    if (stmt->select_column_count > 0) return stmt->select_column_count;
    return has_distances ? 3 : 2;
}

static char *sql_default_column_name(const GV_SQLStmt *stmt, size_t idx,
                                     int has_distances)
{
    if (stmt->select_column_count > 0)
        return gv_dup_cstr(stmt->select_columns[idx]);
    if (has_distances) {
        if (idx == 0) return gv_dup_cstr("index");
        if (idx == 1) return gv_dup_cstr("distance");
        return gv_dup_cstr("metadata");
    }
    if (idx == 0) return gv_dup_cstr("index");
    return gv_dup_cstr("metadata");
}

static int sql_build_select_result(GV_SQLEngine *eng, const GV_SQLStmt *stmt,
                                   GV_SQLRow *rows, size_t row_count,
                                   int has_distances, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    size_t col_count = sql_default_column_count(stmt, has_distances);

    memset(result, 0, sizeof(*result));
    result->row_count = row_count;
    result->column_count = col_count;
    result->column_names = (char **)gv_calloc(col_count, sizeof(char *));
    result->indices = (size_t *)gv_calloc(row_count ? row_count : 1, sizeof(size_t));
    result->metadata_jsons =
        (char **)gv_calloc(row_count ? row_count : 1, sizeof(char *));
    result->column_values =
        (char **)gv_calloc((row_count ? row_count : 1) * col_count, sizeof(char *));
    if (has_distances)
        result->distances = (float *)gv_calloc(row_count ? row_count : 1, sizeof(float));

    if (!result->column_names || !result->indices || !result->metadata_jsons ||
        !result->column_values ||
        (has_distances && !result->distances)) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory building result set");
        return -1;
    }

    for (size_t c = 0; c < col_count; c++) {
        result->column_names[c] = sql_default_column_name(stmt, c, has_distances);
        if (!result->column_names[c]) {
            sql_free_result(result);
            sql_set_error(eng, "Out of memory building result set");
            return -1;
        }
    }

    for (size_t i = 0; i < row_count; i++) {
        result->indices[i] = rows[i].index;
        if (has_distances)
            result->distances[i] = rows[i].has_distance
                                       ? rows[i].distance
                                       : sql_row_vector_distance(db, stmt, &rows[i]);

        GV_SoAStorage *soa = db->soa_storage;
        GV_Metadata *meta = soa ? soa_storage_get_metadata(soa, rows[i].index) : NULL;
        result->metadata_jsons[i] = sql_metadata_to_json(meta);
        if (!result->metadata_jsons[i]) {
            sql_free_result(result);
            sql_set_error(eng, "Out of memory building result set");
            return -1;
        }

        for (size_t c = 0; c < col_count; c++) {
            const char *col = result->column_names[c];
            result->column_values[i * col_count + c] =
                sql_dup_cell_value(db, stmt, &rows[i], col, has_distances);
            if (!result->column_values[i * col_count + c]) {
                sql_free_result(result);
                sql_set_error(eng, "Out of memory building result set");
                return -1;
            }
        }
    }
    return 0;
}

/* Executor: SELECT with ANN */

static int sql_exec_ann(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    const GV_SQLAnn *ann = &stmt->ann;

    if (ann->query_dim != db->dimension) {
        sql_set_error(eng, "ANN query dimension %zu does not match database dimension %zu",
                      ann->query_dim, db->dimension);
        return -1;
    }

    size_t k = ann->k;
    GV_SearchResult *sr = (GV_SearchResult *)gv_calloc(k, sizeof(GV_SearchResult));
    if (!sr) { sql_set_error(eng, "Out of memory"); return -1; }

    int found;
    if (stmt->where) {
        size_t oversample = k * 4;
        if (oversample > db->count && db->count > 0) oversample = db->count;
        if (oversample < k) oversample = k;
        GV_SearchResult *all_sr =
            (GV_SearchResult *)gv_calloc(oversample, sizeof(GV_SearchResult));
        if (!all_sr) { gv_free(sr); sql_set_error(eng, "Out of memory"); return -1; }

        found = db_search(db, ann->query_vector, oversample, all_sr, ann->metric);
        if (found < 0) {
            gv_free(all_sr);
            gv_free(sr);
            sql_set_error(eng, "ANN search failed");
            return -1;
        }

        int matched = 0;
        for (int i = 0; i < found && (size_t)matched < k; i++) {
            if (!all_sr[i].vector) continue;
            if (sql_eval_where(stmt->where, all_sr[i].vector) == 1) {
                sr[matched] = all_sr[i];
                all_sr[i].vector = NULL; /* ownership transferred to sr */
                matched++;
            }
        }
        gv_search_results_free(all_sr, (size_t)found); /* free the unmatched result vectors */
        gv_free(all_sr);
        found = matched;
    } else {
        found = db_search(db, ann->query_vector, k, sr, ann->metric);
        if (found < 0) {
            gv_free(sr);
            sql_set_error(eng, "ANN search failed");
            return -1;
        }
    }

    if (found < 0) found = 0;

    GV_SQLRow *rows = (GV_SQLRow *)gv_calloc((size_t)found, sizeof(GV_SQLRow));
    if (!rows && found > 0) {
        gv_search_results_free(sr, (size_t)found);
        gv_free(sr);
        sql_set_error(eng, "Out of memory");
        return -1;
    }

    size_t row_count = (size_t)found;
    for (size_t i = 0; i < row_count; i++) {
        /* db_search stores the SoA storage index in .id (the vector payload is a
         * fresh heap copy, so pointer arithmetic against SoA would be invalid). */
        rows[i].index = sr[i].id;
        rows[i].distance = sr[i].distance;
        rows[i].has_distance = 1;
    }
    gv_search_results_free(sr, row_count); /* free owned result vectors */
    gv_free(sr);

    sql_sort_rows(db, stmt, rows, row_count);
    sql_apply_row_limit(rows, &row_count, stmt);

    int rc = sql_build_select_result(eng, stmt, rows, row_count, 1, result);
    gv_free(rows);
    return rc;
}

/* Executor: SELECT with WHERE (metadata scan, no ANN) */

static int sql_exec_where_scan(GV_SQLEngine *eng, const GV_SQLStmt *stmt,
                               GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t cap = 256;
    GV_SQLRow *rows = (GV_SQLRow *)gv_alloc(cap * sizeof(GV_SQLRow));
    if (!rows) { sql_set_error(eng, "Out of memory"); return -1; }
    size_t row_count = 0;

    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;
        if (sql_eval_where(stmt->where, &view) != 1) continue;

        if (row_count >= cap) {
            cap *= 2;
            GV_SQLRow *tmp = (GV_SQLRow *)gv_realloc(rows, cap * sizeof(GV_SQLRow));
            if (!tmp) { gv_free(rows); sql_set_error(eng, "Out of memory"); return -1; }
            rows = tmp;
        }
        rows[row_count].index = i;
        rows[row_count].distance = 0.0f;
        rows[row_count].has_distance = 0;
        row_count++;
    }

    sql_sort_rows(db, stmt, rows, row_count);
    sql_apply_row_limit(rows, &row_count, stmt);

    int rc = sql_build_select_result(eng, stmt, rows, row_count, 0, result);
    gv_free(rows);
    return rc;
}

/* Executor: SELECT COUNT(*) */

/* Executor: aggregate projection — COUNT(*) / SUM / MIN / MAX / AVG(column).
 * Produces a single-row, single-column result.  COUNT keeps its historical
 * contract (result->indices[0] holds the count); every aggregate also exposes
 * a formatted value in result->column_values[0]. */
static int sql_exec_aggregate(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t matched = 0;     /* rows passing WHERE */
    size_t numeric_n = 0;   /* rows with a numeric agg_column value */
    double acc = 0.0;       /* SUM / AVG accumulator */
    double best = 0.0;      /* MIN / MAX running value */
    int have_best = 0;
    int numeric_agg = (stmt->agg_kind != GV_SQL_AGG_COUNT && stmt->agg_kind != GV_SQL_AGG_NONE);

    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;

        if (stmt->where) {
            GV_Vector view;
            if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;
            if (sql_eval_where(stmt->where, &view) != 1) continue;
        }
        matched++;

        if (numeric_agg && stmt->agg_column) {
            char vbuf[4096];
            if (sql_get_metadata_value(db, i, stmt->agg_column, vbuf, sizeof(vbuf)) != 0) continue;
            char *endp = NULL;
            double v = strtod(vbuf, &endp);
            if (endp == vbuf) continue; /* non-numeric value: skipped like SQL NULLs */
            numeric_n++;
            acc += v;
            if (!have_best || (stmt->agg_kind == GV_SQL_AGG_MIN && v < best) ||
                              (stmt->agg_kind == GV_SQL_AGG_MAX && v > best)) {
                best = v;
                have_best = 1;
            }
        }
    }

    const char *col_name = "count";
    double dval = 0.0;
    int has_value = 1;
    switch (stmt->agg_kind) {
    case GV_SQL_AGG_SUM: col_name = "sum"; dval = acc; break;
    case GV_SQL_AGG_AVG: col_name = "avg"; if (numeric_n) dval = acc / (double)numeric_n; else has_value = 0; break;
    case GV_SQL_AGG_MIN: col_name = "min"; if (have_best) dval = best; else has_value = 0; break;
    case GV_SQL_AGG_MAX: col_name = "max"; if (have_best) dval = best; else has_value = 0; break;
    case GV_SQL_AGG_COUNT:
    default: col_name = "count"; dval = (double)matched; break;
    }

    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    result->column_values = (char **)gv_calloc(1, sizeof(char *));
    result->indices = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->column_names || !result->column_values || !result->indices) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->column_names[0] = gv_dup_cstr(col_name);
    char valbuf[64];
    if (!has_value) {
        result->column_values[0] = gv_dup_cstr("NULL");
    } else if (stmt->agg_kind == GV_SQL_AGG_COUNT || stmt->agg_kind == GV_SQL_AGG_NONE) {
        snprintf(valbuf, sizeof(valbuf), "%zu", matched);
        result->column_values[0] = gv_dup_cstr(valbuf);
    } else {
        snprintf(valbuf, sizeof(valbuf), "%g", dval);
        result->column_values[0] = gv_dup_cstr(valbuf);
    }
    result->indices[0] = matched; /* COUNT value; row count for other aggregates */
    if (!result->column_names[0] || !result->column_values[0]) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    return 0;
}

/* Executor: DELETE */

static int sql_exec_delete(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t deleted = 0;

    /* Collect matching indices first (iterate, then delete) */
    size_t cap = 256;
    size_t *del_idx = (size_t *)gv_alloc(cap * sizeof(size_t));
    if (!del_idx) { sql_set_error(eng, "Out of memory"); return -1; }

    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;

        if (sql_eval_where(stmt->where, &view) == 1) {
            if (deleted >= cap) {
                cap *= 2;
                size_t *tmp = (size_t *)gv_realloc(del_idx, cap * sizeof(size_t));
                if (!tmp) { gv_free(del_idx); sql_set_error(eng, "Out of memory"); return -1; }
                del_idx = tmp;
            }
            del_idx[deleted++] = i;
        }
    }

    /* Perform deletions */
    for (size_t i = 0; i < deleted; i++) {
        db_delete_vector_by_index(db, del_idx[i]);
    }
    gv_free(del_idx);

    /* Build result: single row with delete count */
    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_dup_cstr("deleted_count");
    }
    result->indices = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->indices || !result->column_names) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->indices[0] = deleted;
    return 0;
}

/* Executor: UPDATE */

static int sql_exec_update(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    GV_SoAStorage *soa = db->soa_storage;
    if (!soa) { sql_set_error(eng, "Database has no storage"); return -1; }

    size_t total = soa->count;
    size_t updated = 0;

    for (size_t i = 0; i < total; i++) {
        if (soa_storage_is_deleted(soa, i)) continue;

        GV_Vector view;
        if (soa_storage_get_vector_view(soa, i, &view) != 0) continue;

        if (sql_eval_where(stmt->where, &view) != 1) continue;

        /* Apply SET clauses as metadata updates */
        const char **keys = (const char **)gv_alloc(stmt->set_count * sizeof(char *));
        const char **vals = (const char **)gv_alloc(stmt->set_count * sizeof(char *));
        if (!keys || !vals) {
            gv_free(keys);
            gv_free(vals);
            sql_set_error(eng, "Out of memory");
            return -1;
        }
        for (size_t j = 0; j < stmt->set_count; j++) {
            keys[j] = stmt->set_clauses[j].field;
            vals[j] = stmt->set_clauses[j].value;
        }
        db_update_vector_metadata(db, i, keys, vals, stmt->set_count);
        gv_free(keys);
        gv_free(vals);
        updated++;
    }

    /* Build result: single row with update count */
    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    if (result->column_names) {
        result->column_names[0] = gv_dup_cstr("updated_count");
    }
    result->indices = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->indices || !result->column_names) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->indices[0] = updated;
    return 0;
}

/* Executor: INSERT */

static int sql_exec_insert(GV_SQLEngine *eng, const GV_SQLStmt *stmt, GV_SQLResult *result)
{
    GV_Database *db = eng->db;
    if (!stmt->insert_vector || stmt->insert_dim == 0) {
        sql_set_error(eng, "INSERT requires a vector value");
        return -1;
    }
    size_t dbdim = database_dimension(db);
    if (dbdim != 0 && stmt->insert_dim != dbdim) {
        sql_set_error(eng, "INSERT vector dimension %zu does not match database dimension %zu",
                      stmt->insert_dim, dbdim);
        return -1;
    }

    int rc;
    if (stmt->set_count > 0) {
        const char **keys = (const char **)gv_alloc(stmt->set_count * sizeof(char *));
        const char **vals = (const char **)gv_alloc(stmt->set_count * sizeof(char *));
        if (!keys || !vals) {
            gv_free(keys); gv_free(vals);
            sql_set_error(eng, "Out of memory");
            return -1;
        }
        for (size_t i = 0; i < stmt->set_count; i++) {
            keys[i] = stmt->set_clauses[i].field;
            vals[i] = stmt->set_clauses[i].value;
        }
        rc = db_add_vector_with_rich_metadata(db, stmt->insert_vector, stmt->insert_dim,
                                              keys, vals, stmt->set_count);
        gv_free(keys);
        gv_free(vals);
    } else {
        rc = db_add_vector(db, stmt->insert_vector, stmt->insert_dim);
    }
    if (rc < 0) {
        sql_set_error(eng, "INSERT failed");
        return -1;
    }

    memset(result, 0, sizeof(*result));
    result->row_count = 1;
    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    result->indices = (size_t *)gv_calloc(1, sizeof(size_t));
    if (!result->column_names || !result->indices) {
        sql_free_result(result);
        sql_set_error(eng, "Out of memory");
        return -1;
    }
    result->column_names[0] = gv_dup_cstr("inserted");
    result->indices[0] = 1;
    return 0;
}

/* Public API */

GV_SQLEngine *sql_create(void *db)
{
    if (!db) return NULL;

    GV_SQLEngine *eng = (GV_SQLEngine *)gv_calloc(1, sizeof(GV_SQLEngine));
    if (!eng) return NULL;

    eng->db = (GV_Database *)db;
    if (pthread_mutex_init(&eng->mutex, NULL) != 0) {
        gv_free(eng);
        return NULL;
    }
    eng->last_error[0] = '\0';
    return eng;
}

void sql_destroy(GV_SQLEngine *eng)
{
    if (!eng) return;
    pthread_mutex_destroy(&eng->mutex);
    gv_free(eng);
}

int sql_execute(GV_SQLEngine *eng, const char *query, GV_SQLResult *result)
{
    if (!eng || !query || !result) return -1;

    pthread_mutex_lock(&eng->mutex);
    eng->last_error[0] = '\0';

    /* Tokenize */
    GV_SQLTokenBuf tbuf;
    memset(&tbuf, 0, sizeof(tbuf));
    if (sql_tokenize(&tbuf, query) != 0) {
        sql_set_error(eng, "%s", tbuf.error);
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Parse */
    GV_SQLStmt *stmt = sql_parse(&tbuf);
    if (!stmt) {
        if (tbuf.error[0])
            sql_set_error(eng, "Parse error: %s", tbuf.error);
        else
            sql_set_error(eng, "Parse error: invalid SQL syntax");
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Check for trailing tokens */
    if (sql_consume_trailing_semicolons(&tbuf) != 0) {
        sql_set_error(eng, "Parse error: unexpected tokens after statement");
        sql_stmt_free(stmt);
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Execute */
    int rc = -1;
    switch (stmt->type) {
    case GV_SQL_STMT_SELECT:
        if (stmt->agg_kind != GV_SQL_AGG_NONE) {
            rc = sql_exec_aggregate(eng, stmt, result);
        } else if (stmt->has_ann) {
            rc = sql_exec_ann(eng, stmt, result);
        } else {
            rc = sql_exec_where_scan(eng, stmt, result);
        }
        break;
    case GV_SQL_STMT_DELETE:
        rc = sql_exec_delete(eng, stmt, result);
        break;
    case GV_SQL_STMT_UPDATE:
        rc = sql_exec_update(eng, stmt, result);
        break;
    case GV_SQL_STMT_INSERT:
        rc = sql_exec_insert(eng, stmt, result);
        break;
    }

    sql_stmt_free(stmt);
    sql_tokenbuf_free(&tbuf);
    pthread_mutex_unlock(&eng->mutex);
    return rc;
}

void sql_free_result(GV_SQLResult *result)
{
    if (!result) return;

    gv_free(result->indices);
    gv_free(result->distances);

    if (result->metadata_jsons) {
        for (size_t i = 0; i < result->row_count; i++)
            gv_free(result->metadata_jsons[i]);
        gv_free(result->metadata_jsons);
    }

    if (result->column_values) {
        size_t cells = result->row_count * result->column_count;
        for (size_t i = 0; i < cells; i++)
            gv_free(result->column_values[i]);
        gv_free(result->column_values);
    }

    if (result->column_names) {
        for (size_t i = 0; i < result->column_count; i++)
            gv_free(result->column_names[i]);
        gv_free(result->column_names);
    }

    memset(result, 0, sizeof(*result));
}

const char *sql_last_error(const GV_SQLEngine *eng)
{
    if (!eng) return "";
    return eng->last_error;
}

int sql_explain(GV_SQLEngine *eng, const char *query, char *plan, size_t plan_size)
{
    if (!eng || !query || !plan || plan_size == 0) return -1;

    pthread_mutex_lock(&eng->mutex);
    eng->last_error[0] = '\0';
    plan[0] = '\0';

    /* Tokenize */
    GV_SQLTokenBuf tbuf;
    memset(&tbuf, 0, sizeof(tbuf));
    if (sql_tokenize(&tbuf, query) != 0) {
        sql_set_error(eng, "%s", tbuf.error);
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    /* Parse */
    GV_SQLStmt *stmt = sql_parse(&tbuf);
    if (!stmt) {
        if (tbuf.error[0])
            sql_set_error(eng, "Parse error: %s", tbuf.error);
        else
            sql_set_error(eng, "Parse error: invalid SQL syntax");
        sql_tokenbuf_free(&tbuf);
        pthread_mutex_unlock(&eng->mutex);
        return -1;
    }

    if (sql_consume_trailing_semicolons(&tbuf) != 0) {
        sql_set_error(eng, "Parse error: unexpected tokens after statement");
        sql_stmt_free(stmt);
        sql_tokenbuf_free(&tbuf);
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
    case GV_INDEX_TYPE_IVFSQ8:  index_name = "IVFSQ8";  break;
    case GV_INDEX_TYPE_IVFTURBOQUANT: index_name = "IVFTURBOQUANT"; break;
    case GV_INDEX_TYPE_IVFDISK: index_name = "IVFDISK"; break;
    case GV_INDEX_TYPE_PQ:      index_name = "PQ";       break;
    case GV_INDEX_TYPE_LSH:     index_name = "LSH";      break;
    default:                    index_name = "UNKNOWN";  break;
    }

    size_t off = 0;

    switch (stmt->type) {
    case GV_SQL_STMT_SELECT:
        if (stmt->agg_kind != GV_SQL_AGG_NONE) {
            const char *aggname =
                stmt->agg_kind == GV_SQL_AGG_SUM ? "SUM" :
                stmt->agg_kind == GV_SQL_AGG_MIN ? "MIN" :
                stmt->agg_kind == GV_SQL_AGG_MAX ? "MAX" :
                stmt->agg_kind == GV_SQL_AGG_AVG ? "AVG" : "COUNT";
            off += (size_t)snprintf(plan + off, plan_size - off,
                "EXPLAIN: SELECT %s(%s)\n"
                "  Strategy: FULL_SCAN\n"
                "  Index: %s\n"
                "  Estimated rows: %zu\n"
                "  Filter: %s\n",
                aggname, stmt->agg_column ? stmt->agg_column : "*",
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

    case GV_SQL_STMT_INSERT:
        (void)snprintf(plan + off, plan_size - off,
            "EXPLAIN: INSERT\n"
            "  Strategy: SINGLE_ROW_INSERT\n"
            "  Index: %s\n"
            "  Vector dimension: %zu\n"
            "  Metadata columns: %zu\n",
            index_name, stmt->insert_dim, stmt->set_count);
        break;
    }

    sql_stmt_free(stmt);
    sql_tokenbuf_free(&tbuf);
    pthread_mutex_unlock(&eng->mutex);
    return 0;
}
