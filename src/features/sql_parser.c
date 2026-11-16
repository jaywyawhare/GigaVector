/**
 * @file sql_parser.c
 * @brief SQL statement parser and AST destructors (split from sql.c).
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "features/sql_internal.h"
#include "storage/database.h"
#include "search/distance.h"
#include "core/memory.h"
#include "core/utils.h"

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

void sql_stmt_free(GV_SQLStmt *s)
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
    if (s->proj) {
        for (size_t i = 0; i < s->proj_count; i++) { gv_free(s->proj[i].column); gv_free(s->proj[i].alias); }
        gv_free(s->proj);
    }
    if (s->group_by) {
        for (size_t i = 0; i < s->group_by_count; i++) gv_free(s->group_by[i]);
        gv_free(s->group_by);
    }
    gv_free(s->having_column);
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

/* field IN (v1, v2, ...) - takes ownership of `field`. */
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

/* field BETWEEN low AND high - takes ownership of `field`. */
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

/* field IS [NOT] NULL - takes ownership of `field`. */
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

/* Parse one projection item: COUNT(*) | SUM|MIN|MAX|AVG(col) | column, with an
 * optional `AS alias`. Returns 0 on success (filling *out), -1 on syntax error. */
static int sql_parse_proj_item(GV_SQLTokenBuf *buf, GV_SQLProjItem *out)
{
    memset(out, 0, sizeof(*out));
    GV_SQLToken *tok = sql_peek(buf);
    if (!tok) return -1;
    if (tok->type == GV_SQL_TOK_COUNT) {
        sql_advance(buf);
        if (!sql_expect(buf, GV_SQL_TOK_LPAREN) ||
            !sql_expect(buf, GV_SQL_TOK_STAR) ||
            !sql_expect(buf, GV_SQL_TOK_RPAREN)) return -1;
        out->agg = GV_SQL_AGG_COUNT;
    } else if (tok->type == GV_SQL_TOK_SUM || tok->type == GV_SQL_TOK_MIN ||
               tok->type == GV_SQL_TOK_MAX || tok->type == GV_SQL_TOK_AVG) {
        out->agg = (tok->type == GV_SQL_TOK_SUM) ? GV_SQL_AGG_SUM :
                   (tok->type == GV_SQL_TOK_MIN) ? GV_SQL_AGG_MIN :
                   (tok->type == GV_SQL_TOK_MAX) ? GV_SQL_AGG_MAX : GV_SQL_AGG_AVG;
        sql_advance(buf);
        if (!sql_expect(buf, GV_SQL_TOK_LPAREN)) return -1;
        GV_SQLToken *ct = sql_peek(buf);
        if (!ct || ct->type != GV_SQL_TOK_IDENT) return -1;
        out->column = gv_dup_cstr(ct->text);
        if (!out->column) return -1;
        sql_advance(buf);
        if (!sql_expect(buf, GV_SQL_TOK_RPAREN)) { gv_free(out->column); out->column = NULL; return -1; }
    } else if (tok->type == GV_SQL_TOK_IDENT) {
        out->agg = GV_SQL_AGG_NONE;
        out->column = gv_dup_cstr(tok->text);
        if (!out->column) return -1;
        sql_advance(buf);
    } else {
        return -1;
    }
    tok = sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_AS) {
        sql_advance(buf);
        tok = sql_peek(buf);
        if (!tok || tok->type != GV_SQL_TOK_IDENT) { gv_free(out->column); out->column = NULL; return -1; }
        out->alias = gv_dup_cstr(tok->text);
        if (!out->alias) { gv_free(out->column); out->column = NULL; return -1; }
        sql_advance(buf);
    }
    return 0;
}

/* parse_select: SELECT [DISTINCT] (* | items) FROM table [ANN] [WHERE]
 * [GROUP BY cols [HAVING agg op n]] [ORDER BY ...] [LIMIT n] [OFFSET n] */
static GV_SQLStmt *sql_parse_select(GV_SQLTokenBuf *buf)
{
    /* SELECT already consumed */
    GV_SQLStmt *stmt = (GV_SQLStmt *)gv_calloc(1, sizeof(GV_SQLStmt));
    if (!stmt) return NULL;
    stmt->type = GV_SQL_STMT_SELECT;

    GV_SQLToken *tok = sql_peek(buf);
    if (!tok) { sql_stmt_free(stmt); return NULL; }

    /* Optional DISTINCT */
    if (tok->type == GV_SQL_TOK_DISTINCT) {
        stmt->distinct = 1;
        sql_advance(buf);
        tok = sql_peek(buf);
        if (!tok) { sql_stmt_free(stmt); return NULL; }
    }

    /* SELECT * | item[, item...] where item = COUNT(*)|AGG(col)|column [AS alias] */
    if (tok->type == GV_SQL_TOK_STAR) {
        sql_advance(buf);
        stmt->select_star = 1;
    } else {
        size_t cap = 4;
        stmt->proj = (GV_SQLProjItem *)gv_calloc(cap, sizeof(GV_SQLProjItem));
        if (!stmt->proj) { sql_stmt_free(stmt); return NULL; }
        for (;;) {
            if (stmt->proj_count >= cap) {
                cap *= 2;
                GV_SQLProjItem *tmp = (GV_SQLProjItem *)gv_realloc(stmt->proj, cap * sizeof(GV_SQLProjItem));
                if (!tmp) { sql_stmt_free(stmt); return NULL; }
                stmt->proj = tmp;
            }
            if (sql_parse_proj_item(buf, &stmt->proj[stmt->proj_count]) != 0) { sql_stmt_free(stmt); return NULL; }
            stmt->proj_count++;
            tok = sql_peek(buf);
            if (tok && tok->type == GV_SQL_TOK_COMMA) { sql_advance(buf); continue; }
            break;
        }
        /* Derive back-compat fields so the existing (non-grouped) executors work:
         *  - all plain columns  -> select_columns[]  (projection scan)
         *  - single aggregate   -> agg_kind/agg_column (scalar-aggregate path)
         *  - anything mixed     -> only valid with GROUP BY (grouped executor). */
        int all_plain = 1;
        for (size_t i = 0; i < stmt->proj_count; i++)
            if (stmt->proj[i].agg != GV_SQL_AGG_NONE) { all_plain = 0; break; }
        if (all_plain) {
            stmt->select_columns = (char **)gv_calloc(stmt->proj_count, sizeof(char *));
            if (!stmt->select_columns) { sql_stmt_free(stmt); return NULL; }
            for (size_t i = 0; i < stmt->proj_count; i++) {
                stmt->select_columns[i] = gv_dup_cstr(stmt->proj[i].column);
                if (!stmt->select_columns[i]) { sql_stmt_free(stmt); return NULL; }
                stmt->select_column_count++;
            }
        } else if (stmt->proj_count == 1) {
            stmt->agg_kind = stmt->proj[0].agg;
            stmt->is_count = (stmt->proj[0].agg == GV_SQL_AGG_COUNT);
            if (stmt->proj[0].column) {
                stmt->agg_column = gv_dup_cstr(stmt->proj[0].column);
                if (!stmt->agg_column) { sql_stmt_free(stmt); return NULL; }
            }
        }
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

    /* Optional GROUP BY col[, col...] [HAVING agg op number] */
    tok = sql_peek(buf);
    if (tok && tok->type == GV_SQL_TOK_GROUP) {
        sql_advance(buf);
        if (!sql_expect(buf, GV_SQL_TOK_BY)) { sql_stmt_free(stmt); return NULL; }
        size_t cap = 4;
        stmt->group_by = (char **)gv_calloc(cap, sizeof(char *));
        if (!stmt->group_by) { sql_stmt_free(stmt); return NULL; }
        for (;;) {
            tok = sql_peek(buf);
            if (!tok || tok->type != GV_SQL_TOK_IDENT) { sql_stmt_free(stmt); return NULL; }
            if (stmt->group_by_count >= cap) {
                cap *= 2;
                char **tmp = (char **)gv_realloc(stmt->group_by, cap * sizeof(char *));
                if (!tmp) { sql_stmt_free(stmt); return NULL; }
                stmt->group_by = tmp;
            }
            stmt->group_by[stmt->group_by_count] = gv_dup_cstr(tok->text);
            if (!stmt->group_by[stmt->group_by_count]) { sql_stmt_free(stmt); return NULL; }
            stmt->group_by_count++;
            sql_advance(buf);
            tok = sql_peek(buf);
            if (tok && tok->type == GV_SQL_TOK_COMMA) { sql_advance(buf); continue; }
            break;
        }
        /* Optional HAVING <aggregate> <cmp> <number> */
        tok = sql_peek(buf);
        if (tok && tok->type == GV_SQL_TOK_HAVING) {
            sql_advance(buf);
            GV_SQLProjItem hi;
            if (sql_parse_proj_item(buf, &hi) != 0) { sql_stmt_free(stmt); return NULL; }
            if (hi.agg == GV_SQL_AGG_NONE) {
                gv_free(hi.column); gv_free(hi.alias);
                snprintf(buf->error, GV_SQL_ERROR_SIZE, "HAVING requires an aggregate");
                sql_stmt_free(stmt); return NULL;
            }
            stmt->having_agg = hi.agg;
            stmt->having_column = hi.column; hi.column = NULL;
            gv_free(hi.alias);
            tok = sql_peek(buf);
            if (!tok || (tok->type != GV_SQL_TOK_EQ && tok->type != GV_SQL_TOK_NE &&
                         tok->type != GV_SQL_TOK_LT && tok->type != GV_SQL_TOK_LE &&
                         tok->type != GV_SQL_TOK_GT && tok->type != GV_SQL_TOK_GE)) {
                sql_stmt_free(stmt); return NULL;
            }
            stmt->having_op = tok->type;
            sql_advance(buf);
            tok = sql_peek(buf);
            if (!tok || tok->type != GV_SQL_TOK_NUMBER) { sql_stmt_free(stmt); return NULL; }
            stmt->having_value = tok->num_value;
            sql_advance(buf);
            stmt->has_having = 1;
        }
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
GV_SQLStmt *sql_parse(GV_SQLTokenBuf *buf)
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

