/**
 * @file sql_internal.h
 * @brief Internal SQL engine types and cross-file prototypes.
 *
 * Shared between sql_lexer.c, sql_parser.c and sql.c (executor/core).
 * Not a public API; do not include outside the SQL engine.
 */
#ifndef GV_FEATURES_SQL_INTERNAL_H
#define GV_FEATURES_SQL_INTERNAL_H

#include <stddef.h>
#include "features/sql.h"
#include "search/distance.h"

/* Constants */

#define GV_SQL_ERROR_SIZE      1024
#define GV_SQL_MAX_QUERY_DIMS  16384
#define GV_SQL_MAX_TOKENS      4096
#define GV_SQL_MAX_RECURSION_DEPTH 256

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
    GV_SQL_TOK_GROUP,
    GV_SQL_TOK_HAVING,
    GV_SQL_TOK_DISTINCT,
    GV_SQL_TOK_AS,
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

typedef struct {
    GV_SQLToken *tokens;
    size_t count;
    size_t pos;
    size_t depth;   /* current WHERE-parse recursion depth (DoS guard) */
    char error[GV_SQL_ERROR_SIZE];
} GV_SQLTokenBuf;

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

/* One output column of a (possibly grouped) SELECT: either a plain column
 * (agg == GV_SQL_AGG_NONE) or an aggregate over `column` (NULL for COUNT(*)). */
typedef struct {
    GV_SQLAggKind agg;
    char *column;
    char *alias;
} GV_SQLProjItem;

typedef struct {
    GV_SQLStmtType type;
    char *table;

    /* SELECT-specific */
    int is_count;        /**< 1 if SELECT COUNT(*) (kept for compat; agg_kind==COUNT) */
    GV_SQLAggKind agg_kind;   /**< aggregate function for the projection, or NONE */
    char *agg_column;    /**< column argument for SUM/MIN/MAX/AVG (NULL for COUNT(*)) */
    int select_star;     /**< 1 if SELECT * */
    int distinct;        /**< 1 if SELECT DISTINCT */
    char **select_columns;
    size_t select_column_count;
    int has_ann;
    GV_SQLAnn ann;

    /* Generalized projection + GROUP BY (mixed columns & aggregates) */
    GV_SQLProjItem *proj;
    size_t proj_count;
    char **group_by;
    size_t group_by_count;
    /* HAVING: a single aggregate comparison, e.g. HAVING COUNT(*) > 2 */
    int has_having;
    GV_SQLAggKind having_agg;
    char *having_column;
    GV_SQLTokenType having_op;   /* one of EQ/NE/LT/LE/GT/GE */
    double having_value;

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


/* --- Cross-file prototypes (split from sql.c) --- */
/* Lexer / token-buffer (sql_lexer.c) */
char *sql_strndup(const char *s, size_t n);
int sql_kw_match(const char *s, size_t len, const char *kw);
int sql_tokenize(GV_SQLTokenBuf *buf, const char *query);
void sql_tokenbuf_free(GV_SQLTokenBuf *buf);
GV_SQLToken *sql_peek(GV_SQLTokenBuf *buf);
GV_SQLToken *sql_advance(GV_SQLTokenBuf *buf);
int sql_consume_trailing_semicolons(GV_SQLTokenBuf *buf);
int sql_expect(GV_SQLTokenBuf *buf, GV_SQLTokenType type);

/* Parser (sql_parser.c) */
GV_SQLStmt *sql_parse(GV_SQLTokenBuf *buf);
void sql_stmt_free(GV_SQLStmt *s);

#endif /* GV_FEATURES_SQL_INTERNAL_H */
