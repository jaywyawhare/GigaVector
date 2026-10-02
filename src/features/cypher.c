/**
 * @file cypher.c
 * @brief A broad openCypher subset executed over the knowledge-graph triple store.
 *
 * Pipeline: tokenize -> parse clauses (MATCH/OPTIONAL MATCH, WHERE, CREATE,
 * MERGE, SET, DELETE, RETURN) -> match patterns into binding rows via
 * backtracking DFS -> evaluate WHERE -> project (aggregate/distinct/order/limit)
 * or apply updates.
 */
#include "features/cypher.h"
#include "features/cypher_vector.h"
#include "core/memory.h"
#include "core/utils.h"
#include "core/compat.h"   /* strcasestr shim on MinGW/MSVC */

#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

/* Portable UTC breakdown: POSIX gmtime_r vs the Windows/ucrt gmtime_s (whose
 * argument order is dest-first, like C11 Annex K). MinGW lacks gmtime_r. */
static void gv_gmtime_utc(const time_t *t, struct tm *out) {
#ifdef _WIN32
    gmtime_s(out, t);
#else
    gmtime_r(t, out);
#endif
}

#define CY_MAXPROP  16
#define CY_MAXNODE  8
#define CY_MAXRET   32
#define CY_MAXSET   16
#define CY_MAXDEL   16
#define CY_MAXORD   8
#define CY_ERR      256
#define CY_MAX_RECURSION_DEPTH 256   /* cap on nested expr/pattern productions (DoS guard) */
#define CY_MAX_TOKENS  4096          /* cap on total token count (mirrors GV_SQL_MAX_TOKENS) */

/* lexer */

typedef enum {
    T_EOF, T_LP, T_RP, T_LB, T_RB, T_LC, T_RC, T_COLON, T_COMMA, T_DOT,
    T_EQ, T_NE, T_LT, T_GT, T_LE, T_GE, T_ARROW_R, T_ARROW_L, T_DASH,
    T_PIPE, T_STAR, T_PLUS, T_SLASH, T_PERCENT, T_PARAM, T_IDENT, T_STRING, T_NUMBER
} Tk;

typedef struct { Tk t; char *s; } Tok;
typedef struct { Tok *v; size_t n, cap, pos; size_t depth; char err[CY_ERR]; } Lex;

static char *sdup(const char *s, size_t n) {
    char *r = (char *)gv_alloc(n + 1);
    if (r) { memcpy(r, s, n); r[n] = 0; }
    return r;
}
static int push(Lex *lx, Tk t, const char *s, size_t n) {
    if (lx->n >= CY_MAX_TOKENS) {
        snprintf(lx->err, CY_ERR, "query too large (token limit exceeded)");
        return -1;
    }
    if (lx->n == lx->cap) {
        size_t nc = lx->cap ? lx->cap * 2 : 64;
        Tok *nv = (Tok *)gv_realloc(lx->v, nc * sizeof(Tok));
        if (!nv) return -1;
        lx->v = nv; lx->cap = nc;
    }
    char *dup = (s && n) ? sdup(s, n) : NULL;
    if (s && n && !dup) return -1;  /* OOM building token text — abort tokenize */
    lx->v[lx->n].t = t;
    lx->v[lx->n].s = dup;
    lx->n++;
    return 0;
}
static void lex_free(Lex *lx) {
    if (lx->v) { for (size_t i = 0; i < lx->n; i++) gv_free(lx->v[i].s); gv_free(lx->v); }
    lx->v = NULL;
}
static int tokenize(Lex *lx, const char *q) {
    size_t i = 0;
    /* PUSH aborts tokenization on any push() failure (token-limit or OOM);
     * push() has already set lx->err in that case, but guard here too. */
    #define PUSH(...) do { \
        if (push(lx, __VA_ARGS__) != 0) { \
            if (lx->err[0] == 0) snprintf(lx->err, CY_ERR, "out of memory tokenizing query"); \
            return -1; \
        } \
    } while (0)
    while (q[i]) {
        char c = q[i];
        if (isspace((unsigned char)c)) { i++; continue; }
        if (lx->n >= CY_MAX_TOKENS) { snprintf(lx->err, CY_ERR, "query too large (token limit exceeded)"); return -1; }
        switch (c) {
            case '(': PUSH(T_LP, 0, 0); i++; continue;
            case ')': PUSH(T_RP, 0, 0); i++; continue;
            case '[': PUSH(T_LB, 0, 0); i++; continue;
            case ']': PUSH(T_RB, 0, 0); i++; continue;
            case '{': PUSH(T_LC, 0, 0); i++; continue;
            case '}': PUSH(T_RC, 0, 0); i++; continue;
            case ':': PUSH(T_COLON, 0, 0); i++; continue;
            case ',': PUSH(T_COMMA, 0, 0); i++; continue;
            case '.': PUSH(T_DOT, 0, 0); i++; continue;
            case '|': PUSH(T_PIPE, 0, 0); i++; continue;
            case '*': PUSH(T_STAR, 0, 0); i++; continue;
            case '+': PUSH(T_PLUS, 0, 0); i++; continue;
            case '/': PUSH(T_SLASH, 0, 0); i++; continue;
            case '%': PUSH(T_PERCENT, 0, 0); i++; continue;
            case '$': { size_t s = ++i; while (q[i] && (isalnum((unsigned char)q[i]) || q[i] == '_')) i++;
                        PUSH(T_PARAM, q + s, i - s); continue; }
            case ';': i++; continue;
            case '=': PUSH(T_EQ, 0, 0); i++; continue;
            case '<':
                if (q[i+1] == '-') { PUSH(T_ARROW_L, 0, 0); i += 2; continue; }
                if (q[i+1] == '>') { PUSH(T_NE, 0, 0); i += 2; continue; }
                if (q[i+1] == '=') { PUSH(T_LE, 0, 0); i += 2; continue; }
                PUSH(T_LT, 0, 0); i++; continue;
            case '>':
                if (q[i+1] == '=') { PUSH(T_GE, 0, 0); i += 2; continue; }
                PUSH(T_GT, 0, 0); i++; continue;
            case '-':
                if (q[i+1] == '>') { PUSH(T_ARROW_R, 0, 0); i += 2; continue; }
                PUSH(T_DASH, 0, 0); i++; continue;
            case '\'': case '"': {
                char qt = c; size_t s = i + 1, j = s;
                while (q[j] && q[j] != qt) j++;
                if (!q[j]) { snprintf(lx->err, CY_ERR, "unterminated string"); return -1; }
                PUSH(T_STRING, q + s, j - s); i = j + 1; continue;
            }
            default: break;
        }
        if (isdigit((unsigned char)c)) {
            size_t s = i; i++;
            while (q[i] && isdigit((unsigned char)q[i])) i++;
            if (q[i] == '.' && isdigit((unsigned char)q[i + 1])) { /* decimal, not ".." */
                i++;
                while (q[i] && isdigit((unsigned char)q[i])) i++;
            }
            PUSH(T_NUMBER, q + s, i - s); continue;
        }
        if (isalpha((unsigned char)c) || c == '_') {
            size_t s = i;
            while (q[i] && (isalnum((unsigned char)q[i]) || q[i] == '_')) i++;
            PUSH(T_IDENT, q + s, i - s); continue;
        }
        snprintf(lx->err, CY_ERR, "unexpected character '%c'", c);
        return -1;
    }
    PUSH(T_EOF, 0, 0);
    #undef PUSH
    return 0;
}
static Tok *pk(Lex *lx) { return &lx->v[lx->pos]; }
static Tok *adv(Lex *lx) { return &lx->v[lx->pos < lx->n - 1 ? lx->pos++ : lx->pos]; }
static int kw(const Tok *t, const char *k) { return t->t == T_IDENT && t->s && strcasecmp(t->s, k) == 0; }
static int eat(Lex *lx, Tk t, const char *w) {
    if (pk(lx)->t != t) { snprintf(lx->err, CY_ERR, "expected %s", w); return -1; }
    adv(lx); return 0;
}

/* AST */

#define CY_MAXLABEL 4
typedef struct {
    char *var; char *label;                     /* label = first of labels[] (index probe) */
    char *labels[CY_MAXLABEL]; size_t nlabels;  /* all labels; node must match every one */
    char *pk[CY_MAXPROP]; char *pv[CY_MAXPROP]; size_t np;
} Node;
typedef struct { char *var; char *type; int dir; int varlen; int minh, maxh;
                 char *types[8]; size_t ntypes;               /* rel-type alternation :A|B|... */
                 char *pk[CY_MAXPROP]; char *pv[CY_MAXPROP]; size_t np; } Rel; /* dir 1 -> , -1 <- , 0 undirected */
#define CY_VARLEN_MAX 8
static const char *cy_param_lookup(const char *name); /* defined with the engine */
typedef struct { Node node[CY_MAXNODE]; Rel rel[CY_MAXNODE]; size_t nn; } Pattern;

typedef enum { OPD_VAR, OPD_PROP, OPD_LIT, OPD_TYPE, OPD_FUNC, OPD_BIN, OPD_NEG, OPD_CASE, OPD_PARAM,
               OPD_LIST, OPD_INDEX, OPD_LISTCOMP, OPD_PATCOMP, OPD_MAP } OpdK;
struct Expr;
typedef struct Opd {
    OpdK k;
    char *var, *prop, *lit;
    char *fname; struct Opd **args; size_t nargs;   /* function call / list literal elements */
    char binop; struct Opd *l, *r;                  /* arithmetic; OPD_INDEX: l[r] */
    struct Expr **cw; struct Opd **ct; size_t ncase; struct Opd *celse; /* CASE */
    char *lcvar; struct Opd *lclist; struct Expr *lcwhere; struct Opd *lcproj; /* list comprehension */
    Pattern *pcpat; struct Expr *pcwhere; struct Opd *pcproj;                  /* pattern comprehension */
    char **mk; size_t nmk;                                                     /* map literal keys (values in args) */
    int littype;  /* OPD_LIT type: 0=string 1=integer 2=float 3=boolean 4=null */
} Opd;
/* Internal list encoding: tag byte then elements joined by 0x1f. */
#define CY_LTAG '\x02'
#define CY_LSEP '\x1f'
/* Map encoding: tag then key\x06value pairs joined by \x07. */
#define CY_MTAG '\x05'
#define CY_MKV  '\x06'
#define CY_MPAIR '\x07'

typedef enum { EX_OR, EX_AND, EX_NOT, EX_CMP, EX_EXISTS } ExK;
typedef enum { C_EQ, C_NE, C_LT, C_GT, C_LE, C_GE, C_CONTAINS, C_STARTS, C_ENDS,
               C_IN, C_ISNULL, C_ISNOTNULL, C_TRUTHY } Cmp;
typedef struct Expr { ExK k; struct Expr *l, *r; Cmp op; Opd a, b;
                      char **inlist; size_t nin; Pattern *pat; } Expr;
static void pattern_clear(Pattern *p);

typedef enum { AG_NONE, AG_COUNT, AG_COUNTSTAR, AG_COLLECT, AG_SUM, AG_AVG, AG_MIN, AG_MAX, AG_STDEV, AG_STDEVP, AG_PCTCONT, AG_PCTDISC } Agg;
static int agg_has_pct(Agg a) { return a == AG_PCTCONT || a == AG_PCTDISC; }
typedef struct { Agg agg; Opd opd; char *alias; double aggarg; } Ret; /* aggarg: percentile p for percentileCont/Disc */
typedef struct { Opd opd; int desc; } Ord;
typedef struct { char *var; char *prop; Opd *valexpr; } SetItem;

/* parser */

static void node_clear(Node *n) {
    gv_free(n->var); gv_free(n->label);
    for (size_t i = 0; i < n->nlabels; i++) gv_free(n->labels[i]);
    for (size_t i = 0; i < n->np; i++) { gv_free(n->pk[i]); gv_free(n->pv[i]); }
    memset(n, 0, sizeof(*n));
}
static void expr_free(Expr *e);
static void opd_clear(Opd *o) {
    if (!o) return;
    gv_free(o->var); gv_free(o->prop); gv_free(o->lit); gv_free(o->fname);
    for (size_t i = 0; i < o->nargs; i++) { opd_clear(o->args[i]); gv_free(o->args[i]); }
    gv_free(o->args);
    if (o->l) { opd_clear(o->l); gv_free(o->l); }
    if (o->r) { opd_clear(o->r); gv_free(o->r); }
    for (size_t i = 0; i < o->ncase; i++) {
        expr_free(o->cw[i]);
        opd_clear(o->ct[i]); gv_free(o->ct[i]);
    }
    gv_free(o->cw); gv_free(o->ct);
    if (o->celse) { opd_clear(o->celse); gv_free(o->celse); }
    gv_free(o->lcvar);
    if (o->lclist) { opd_clear(o->lclist); gv_free(o->lclist); }
    if (o->lcwhere) expr_free(o->lcwhere);
    if (o->lcproj) { opd_clear(o->lcproj); gv_free(o->lcproj); }
    if (o->pcpat) { pattern_clear(o->pcpat); gv_free(o->pcpat); }
    if (o->pcwhere) expr_free(o->pcwhere);
    if (o->pcproj) { opd_clear(o->pcproj); gv_free(o->pcproj); }
    for (size_t i = 0; i < o->nmk; i++) gv_free(o->mk[i]);
    gv_free(o->mk);
    memset(o, 0, sizeof(*o));
}
static void setitem_free(SetItem *s) {
    gv_free(s->var); gv_free(s->prop);
    if (s->valexpr) { opd_clear(s->valexpr); gv_free(s->valexpr); }
    s->var = s->prop = NULL; s->valexpr = NULL;
}
static Opd opd_copy(const Opd *s) {
    Opd d; memset(&d, 0, sizeof(d));
    d.k = s->k; d.binop = s->binop;
    if (s->var)   d.var   = gv_dup_cstr(s->var);
    if (s->prop)  d.prop  = gv_dup_cstr(s->prop);
    if (s->lit)   d.lit   = gv_dup_cstr(s->lit);
    if (s->fname) d.fname = gv_dup_cstr(s->fname);
    if (s->nargs) {
        d.args = (Opd **)gv_alloc(s->nargs * sizeof(Opd *)); d.nargs = s->nargs;
        for (size_t i = 0; i < s->nargs; i++) { d.args[i] = (Opd *)gv_alloc(sizeof(Opd)); *d.args[i] = opd_copy(s->args[i]); }
    }
    /* Copy map-literal keys alongside their values (in args): val_of()'s OPD_MAP
     * branch reads mk[i], so a copy with nargs>0 but mk==NULL would NULL-deref. */
    if (s->nmk) {
        d.mk = (char **)gv_alloc(s->nmk * sizeof(char *)); d.nmk = s->nmk;
        for (size_t i = 0; i < s->nmk; i++) d.mk[i] = s->mk[i] ? gv_dup_cstr(s->mk[i]) : NULL;
    }
    /* Copy the list-comprehension source/var: val_of() dereferences lclist, so an
     * uncopied (NULL) lclist on a list-comp subject would NULL-deref. */
    if (s->lcvar)  d.lcvar  = gv_dup_cstr(s->lcvar);
    if (s->lclist) { d.lclist = (Opd *)gv_alloc(sizeof(Opd)); *d.lclist = opd_copy(s->lclist); }
    if (s->l) { d.l = (Opd *)gv_alloc(sizeof(Opd)); *d.l = opd_copy(s->l); }
    if (s->r) { d.r = (Opd *)gv_alloc(sizeof(Opd)); *d.r = opd_copy(s->r); }
    return d; /* CASE/pattern-comprehension subtrees still not copied (unsupported as copied subjects) */
}
static void expr_free(Expr *e) {
    if (!e) return;
    expr_free(e->l); expr_free(e->r);
    opd_clear(&e->a); opd_clear(&e->b);
    for (size_t i = 0; i < e->nin; i++) gv_free(e->inlist[i]);
    gv_free(e->inlist);
    if (e->pat) { pattern_clear(e->pat); gv_free(e->pat); }
    gv_free(e);
}

/* Parse a `{ key: value, ... }` map into parallel key/value arrays. Shared by
 * node and relationship patterns. Values may be string/number literals, $params
 * (substituted now), or bare identifiers (a '\x04'-tagged variable reference
 * resolved at CREATE time, e.g. a FOREACH loop var). */
static int parse_kv_props(Lex *lx, char **keys, char **vals, size_t *np, size_t cap) {
    if (pk(lx)->t != T_LC) return 0;
    adv(lx);
    while (pk(lx)->t != T_RC) {
        if (*np >= cap) { snprintf(lx->err, CY_ERR, "too many properties"); return -1; }
        if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected property key"); return -1; }
        char *key = gv_dup_cstr(adv(lx)->s);
        if (!key) { snprintf(lx->err, CY_ERR, "out of memory"); return -1; }
        if (eat(lx, T_COLON, "':'")) { gv_free(key); return -1; }
        Tok *v = pk(lx);
        char *val = NULL;
        if (v->t == T_STRING || v->t == T_NUMBER) val = gv_dup_cstr(adv(lx)->s);
        else if (v->t == T_PARAM) { const char *pv = cy_param_lookup(v->s); val = gv_dup_cstr(pv ? pv : ""); adv(lx); }
        else if (v->t == T_IDENT) { /* variable reference (resolved at CREATE time), e.g. FOREACH loop var */
            char *nm = adv(lx)->s; size_t l = strlen(nm);
            val = (char *)gv_alloc(l + 2);
            if (val) { val[0] = '\x04'; memcpy(val + 1, nm, l + 1); }
        }
        else { gv_free(key); snprintf(lx->err, CY_ERR, "expected property value"); return -1; }
        if (!val) { gv_free(key); snprintf(lx->err, CY_ERR, "out of memory"); return -1; }
        /* Commit key+value together only once both allocations succeed. */
        keys[*np] = key;
        vals[*np] = val;
        (*np)++;
        if (pk(lx)->t == T_COMMA) adv(lx); else break;
    }
    return eat(lx, T_RC, "'}'");
}
static int parse_props(Lex *lx, Node *n) {
    return parse_kv_props(lx, n->pk, n->pv, &n->np, CY_MAXPROP);
}
static int parse_node(Lex *lx, Node *n) {
    memset(n, 0, sizeof(*n));
    if (eat(lx, T_LP, "'('")) return -1;
    if (pk(lx)->t == T_IDENT) n->var = gv_dup_cstr(adv(lx)->s);
    while (pk(lx)->t == T_COLON) { adv(lx);
        if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected label"); node_clear(n); return -1; }
        /* first label -> n->label (fast type index); all labels -> n->labels[]
         * so a node with multiple labels must match ALL of them. */
        char *lab = gv_dup_cstr(adv(lx)->s);
        if (!n->label) n->label = gv_dup_cstr(lab);
        if (n->nlabels < CY_MAXLABEL) n->labels[n->nlabels++] = lab; else gv_free(lab);
    }
    if (parse_props(lx, n)) { node_clear(n); return -1; }
    if (eat(lx, T_RP, "')'")) { node_clear(n); return -1; }
    return 0;
}
static void rel_clear(Rel *r) {
    gv_free(r->var); gv_free(r->type);
    for (size_t i = 0; i < r->ntypes; i++) gv_free(r->types[i]);
    for (size_t i = 0; i < r->np; i++) { gv_free(r->pk[i]); gv_free(r->pv[i]); }
    memset(r, 0, sizeof(*r));
}
static int rel_fail(Rel *r) { rel_clear(r); return -1; }
static int parse_rel(Lex *lx, Rel *r) {
    memset(r, 0, sizeof(*r));
    Tk t = pk(lx)->t;
    if (t != T_DASH && t != T_ARROW_L) return 0;
    if (t == T_ARROW_L) { r->dir = -1; adv(lx); } else adv(lx);
    if (eat(lx, T_LB, "'['")) return rel_fail(r);
    if (pk(lx)->t == T_IDENT) r->var = gv_dup_cstr(adv(lx)->s);
    if (pk(lx)->t == T_COLON) { adv(lx);
        if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected relationship type"); return rel_fail(r); }
        r->type = gv_dup_cstr(adv(lx)->s);
        r->types[r->ntypes++] = gv_dup_cstr(r->type);
        /* rel type alternation :A|B|... — match ANY listed type */
        while (pk(lx)->t == T_PIPE) { adv(lx);
            if (pk(lx)->t == T_COLON) adv(lx);   /* tolerate :A|:B form */
            if (pk(lx)->t == T_IDENT) { char *tt = gv_dup_cstr(adv(lx)->s); if (r->ntypes < 8) r->types[r->ntypes++] = tt; else gv_free(tt); }
        }
    }
    if (pk(lx)->t == T_STAR) {
        adv(lx);
        r->varlen = 1; r->minh = 1; r->maxh = CY_VARLEN_MAX;
        if (pk(lx)->t == T_NUMBER) { r->minh = (int)strtol(adv(lx)->s, NULL, 10); r->maxh = r->minh; }
        /* range ".." tokenizes as two T_DOT */
        if (pk(lx)->t == T_DOT && lx->pos + 1 < lx->n && lx->v[lx->pos + 1].t == T_DOT) {
            adv(lx); adv(lx);
            r->maxh = CY_VARLEN_MAX;
            if (pk(lx)->t == T_NUMBER) r->maxh = (int)strtol(adv(lx)->s, NULL, 10);
        }
        if (r->maxh > CY_VARLEN_MAX) r->maxh = CY_VARLEN_MAX;
        if (r->minh < 1) r->minh = 1;
    }
    if (parse_kv_props(lx, r->pk, r->pv, &r->np, CY_MAXPROP)) return rel_fail(r);
    if (eat(lx, T_RB, "']'")) return rel_fail(r);
    if (r->dir == -1) { if (eat(lx, T_DASH, "'-'")) return rel_fail(r); }
    else if (pk(lx)->t == T_ARROW_R) { r->dir = 1; adv(lx); }
    else if (pk(lx)->t == T_DASH) { r->dir = 0; adv(lx); }
    else { snprintf(lx->err, CY_ERR, "expected '->' or '-'"); return rel_fail(r); }
    return 1;
}
static int parse_pattern_body(Lex *lx, Pattern *p);
static int parse_pattern(Lex *lx, Pattern *p) {
    memset(p, 0, sizeof(*p));
    if (lx->depth >= CY_MAX_RECURSION_DEPTH) { snprintf(lx->err, CY_ERR, "pattern nesting too deep"); return -1; }
    lx->depth++;
    int rc = parse_pattern_body(lx, p);
    lx->depth--;
    return rc;
}
static int parse_pattern_body(Lex *lx, Pattern *p) {
    if (parse_node(lx, &p->node[0])) return -1;
    p->nn = 1;
    for (;;) {
        Rel r;
        int hr = parse_rel(lx, &r);
        if (hr < 0) return -1;
        if (hr == 0) break;
        if (p->nn >= CY_MAXNODE) { snprintf(lx->err, CY_ERR, "pattern too long"); rel_clear(&r); return -1; }
        p->rel[p->nn - 1] = r;
        if (parse_node(lx, &p->node[p->nn])) return -1;
        p->nn++;
    }
    return 0;
}
static void pattern_clear(Pattern *p) {
    for (size_t i = 0; i < p->nn; i++) {
        node_clear(&p->node[i]);
        if (i + 1 < p->nn) rel_clear(&p->rel[i]);
    }
    memset(p, 0, sizeof(*p));
}

/* operand: var[.prop] | type(var) | 'string' | number | [list] | comprehension */
static Expr *parse_or(Lex *lx);       /* boolean expr (for CASE WHEN) */
static Opd *parse_add(Lex *lx);       /* value expr (arithmetic) */
static Opd *parse_atom(Lex *lx);
static Opd *parse_unary(Lex *lx);     /* recursion-depth-guarded wrapper */
static Expr *parse_cmp(Lex *lx);      /* recursion-depth-guarded wrapper */
/* primary = atom with postfix list indexing atom[expr] */
static Opd *parse_primary(Lex *lx) {
    Opd *o = parse_atom(lx);
    while (o && pk(lx)->t == T_LB) {
        adv(lx);
        Opd *idx = parse_add(lx);
        if (!idx || eat(lx, T_RB, "']'")) { opd_clear(o); gv_free(o); if (idx) { opd_clear(idx); gv_free(idx); } return NULL; }
        /* cppcheck-suppress nullPointerRedundantCheck */
        Opd *w = (Opd *)gv_calloc(1, sizeof(Opd));
        w->k = OPD_INDEX; w->l = o; w->r = idx; o = w;
    }
    return o;
}

/* atom: literal | [list]|[comprehension] | func(args) | type(v) | CASE..END | var[.prop] | ( expr ) */
static Opd *parse_atom_impl(Lex *lx) {
    Opd *o = (Opd *)gv_calloc(1, sizeof(Opd));
    Tok *t = pk(lx);
    if (t->t == T_STRING || t->t == T_NUMBER) {
        o->k = OPD_LIT;
        if (t->t == T_NUMBER) o->littype = strchr(t->s, '.') ? 2 : 1; /* float vs integer */
        else o->littype = 0; /* string */
        o->lit = gv_dup_cstr(adv(lx)->s);
        return o;
    }
    if (t->t == T_IDENT && (kw(t, "true") || kw(t, "false") || kw(t, "null"))) {
        o->k = OPD_LIT;
        if (kw(t, "null")) { o->littype = 4; o->lit = gv_dup_cstr(""); }
        else { o->littype = 3; o->lit = gv_dup_cstr(kw(t, "true") ? "1" : "0"); }
        adv(lx);
        return o;
    }
    if (t->t == T_PARAM) { o->k = OPD_PARAM; o->var = gv_dup_cstr(adv(lx)->s); return o; }
    if (t->t == T_LB) {
        adv(lx);
        /* pattern comprehension: [ (pattern) [WHERE cond] | proj ] */
        if (pk(lx)->t == T_LP) {
            o->k = OPD_PATCOMP;
            o->pcpat = (Pattern *)gv_calloc(1, sizeof(Pattern));
            if (parse_pattern(lx, o->pcpat)) { opd_clear(o); gv_free(o); return NULL; }
            if (kw(pk(lx), "where")) { adv(lx); o->pcwhere = parse_or(lx); if (!o->pcwhere) { opd_clear(o); gv_free(o); return NULL; } }
            if (eat(lx, T_PIPE, "'|'")) { opd_clear(o); gv_free(o); return NULL; }
            o->pcproj = parse_add(lx);
            if (!o->pcproj || eat(lx, T_RB, "']'")) { opd_clear(o); gv_free(o); return NULL; }
            return o;
        }
        /* comprehension: [ var IN listExpr [WHERE cond] | proj ] */
        if (pk(lx)->t == T_IDENT && lx->pos + 1 < lx->n && kw(&lx->v[lx->pos + 1], "in")) {
            o->k = OPD_LISTCOMP;
            o->lcvar = gv_dup_cstr(adv(lx)->s);
            adv(lx); /* IN */
            o->lclist = parse_add(lx);
            if (!o->lclist) { opd_clear(o); gv_free(o); return NULL; }
            if (kw(pk(lx), "where")) { adv(lx); o->lcwhere = parse_or(lx); if (!o->lcwhere) { opd_clear(o); gv_free(o); return NULL; } }
            if (eat(lx, T_PIPE, "'|'")) { opd_clear(o); gv_free(o); return NULL; }
            o->lcproj = parse_add(lx);
            if (!o->lcproj || eat(lx, T_RB, "']'")) { opd_clear(o); gv_free(o); return NULL; }
            return o;
        }
        /* list literal: [ e, e, ... ] */
        o->k = OPD_LIST;
        while (pk(lx)->t != T_RB && pk(lx)->t != T_EOF) {
            Opd *el = parse_add(lx);
            if (!el) { opd_clear(o); gv_free(o); return NULL; }
            o->args = (Opd **)gv_realloc(o->args, (o->nargs + 1) * sizeof(Opd *));
            o->args[o->nargs++] = el;
            if (pk(lx)->t == T_COMMA) adv(lx); else break;
        }
        if (eat(lx, T_RB, "']'")) { opd_clear(o); gv_free(o); return NULL; }
        return o;
    }
    if (t->t == T_LC) { /* map literal: { key: expr, ... } */
        adv(lx);
        o->k = OPD_MAP;
        while (pk(lx)->t != T_RC && pk(lx)->t != T_EOF) {
            if (pk(lx)->t != T_IDENT && pk(lx)->t != T_STRING) { snprintf(lx->err, CY_ERR, "expected map key"); opd_clear(o); gv_free(o); return NULL; }
            o->mk = (char **)gv_realloc(o->mk, (o->nmk + 1) * sizeof(char *));
            o->mk[o->nmk++] = gv_dup_cstr(adv(lx)->s);
            if (eat(lx, T_COLON, "':'")) { opd_clear(o); gv_free(o); return NULL; }
            Opd *v = parse_add(lx);
            if (!v) { opd_clear(o); gv_free(o); return NULL; }
            o->args = (Opd **)gv_realloc(o->args, (o->nargs + 1) * sizeof(Opd *));
            o->args[o->nargs++] = v;
            if (pk(lx)->t == T_COMMA) adv(lx); else break;
        }
        if (eat(lx, T_RC, "'}'")) { opd_clear(o); gv_free(o); return NULL; }
        return o;
    }
    if (t->t == T_LP) { /* grouping */
        adv(lx);
        Opd *inner = parse_add(lx);
        if (!inner || eat(lx, T_RP, "')'")) { opd_clear(o); gv_free(o); if (inner) { opd_clear(inner); gv_free(inner); } return NULL; }
        opd_clear(o); gv_free(o); return inner;
    }
    if (kw(t, "case")) {
        adv(lx);
        o->k = OPD_CASE;
        Opd *subject = NULL;
        if (!kw(pk(lx), "when")) { subject = parse_add(lx); if (!subject) { opd_clear(o); gv_free(o); return NULL; } }
        while (kw(pk(lx), "when")) {
            adv(lx);
            Expr *cond;
            if (subject) { /* simple CASE: subject = value */
                Opd *val = parse_add(lx);
                if (!val) { if (subject){opd_clear(subject);gv_free(subject);} opd_clear(o); gv_free(o); return NULL; }
                cond = (Expr *)gv_calloc(1, sizeof(Expr));
                cond->k = EX_CMP; cond->op = C_EQ;
                cond->a = opd_copy(subject);   /* fresh copy per WHEN */
                cond->b = *val; gv_free(val);
            } else {
                cond = parse_or(lx);
                if (!cond) { opd_clear(o); gv_free(o); return NULL; }
            }
            if (!kw(pk(lx), "then")) { snprintf(lx->err, CY_ERR, "expected THEN"); expr_free(cond); if(subject){opd_clear(subject);gv_free(subject);} opd_clear(o); gv_free(o); return NULL; }
            adv(lx);
            Opd *thenv = parse_add(lx);
            if (!thenv) { expr_free(cond); if(subject){opd_clear(subject);gv_free(subject);} opd_clear(o); gv_free(o); return NULL; }
            o->cw = (Expr **)gv_realloc(o->cw, (o->ncase + 1) * sizeof(Expr *));
            o->ct = (Opd **)gv_realloc(o->ct, (o->ncase + 1) * sizeof(Opd *));
            o->cw[o->ncase] = cond; o->ct[o->ncase] = thenv; o->ncase++;
        }
        if (kw(pk(lx), "else")) { adv(lx); o->celse = parse_add(lx); if (!o->celse) { if(subject){opd_clear(subject);gv_free(subject);} opd_clear(o); gv_free(o); return NULL; } }
        if (subject) { opd_clear(subject); gv_free(subject); }
        if (!kw(pk(lx), "end")) { snprintf(lx->err, CY_ERR, "expected END"); opd_clear(o); gv_free(o); return NULL; }
        adv(lx);
        return o;
    }
    if (t->t == T_IDENT && lx->pos + 1 < lx->n && lx->v[lx->pos + 1].t == T_LP) {
        /* function call: fname ( args ) ; type() maps to OPD_TYPE for a bare var */
        char *fn = gv_dup_cstr(adv(lx)->s);
        adv(lx); /* '(' */
        if (strcasecmp(fn, "type") == 0 && pk(lx)->t == T_IDENT && lx->v[lx->pos + 1].t == T_RP) {
            o->k = OPD_TYPE; o->var = gv_dup_cstr(adv(lx)->s); adv(lx); gv_free(fn); return o;
        }
        o->k = OPD_FUNC; o->fname = fn;
        if (pk(lx)->t == T_STAR) adv(lx); /* count(*)-like inside expr: ignore arg */
        while (pk(lx)->t != T_RP && pk(lx)->t != T_EOF) {
            Opd *a = parse_add(lx);
            if (!a) { opd_clear(o); gv_free(o); return NULL; }
            o->args = (Opd **)gv_realloc(o->args, (o->nargs + 1) * sizeof(Opd *));
            o->args[o->nargs++] = a;
            if (pk(lx)->t == T_COMMA) adv(lx); else break;
        }
        if (eat(lx, T_RP, "')'")) { opd_clear(o); gv_free(o); return NULL; }
        return o;
    }
    if (t->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected operand"); opd_clear(o); gv_free(o); return NULL; }
    o->var = gv_dup_cstr(adv(lx)->s);
    if (!o->var) { snprintf(lx->err, CY_ERR, "out of memory"); opd_clear(o); gv_free(o); return NULL; }
    if (pk(lx)->t == T_DOT) { adv(lx);
        if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected property"); opd_clear(o); gv_free(o); return NULL; }
        o->k = OPD_PROP; o->prop = gv_dup_cstr(adv(lx)->s);
        if (!o->prop) { snprintf(lx->err, CY_ERR, "out of memory"); opd_clear(o); gv_free(o); return NULL; }
    } else o->k = OPD_VAR;
    return o;
}
/* Depth-guarded wrapper around the atom production. Every recursive value-expr
 * cycle (grouping '(', lists/maps '[' '{', comprehensions, CASE) re-enters here. */
static Opd *parse_atom(Lex *lx) {
    if (lx->depth >= CY_MAX_RECURSION_DEPTH) { snprintf(lx->err, CY_ERR, "expression nesting too deep"); return NULL; }
    lx->depth++;
    Opd *o = parse_atom_impl(lx);
    lx->depth--;
    return o;
}
static Opd *parse_unary_impl(Lex *lx) {
    if (pk(lx)->t == T_DASH) {
        adv(lx);
        Opd *inner = parse_unary(lx);
        if (!inner) return NULL;
        Opd *o = (Opd *)gv_calloc(1, sizeof(Opd));
        o->k = OPD_NEG; o->l = inner; return o;
    }
    return parse_primary(lx);
}
/* Depth-guarded wrapper: unary minus chains ('- - - ...') recurse through here. */
static Opd *parse_unary(Lex *lx) {
    if (lx->depth >= CY_MAX_RECURSION_DEPTH) { snprintf(lx->err, CY_ERR, "expression nesting too deep"); return NULL; }
    lx->depth++;
    Opd *o = parse_unary_impl(lx);
    lx->depth--;
    return o;
}
static Opd *parse_mul(Lex *lx) {
    Opd *l = parse_unary(lx);
    while (l && (pk(lx)->t == T_STAR || pk(lx)->t == T_SLASH || pk(lx)->t == T_PERCENT)) {
        char op = pk(lx)->t == T_STAR ? '*' : pk(lx)->t == T_SLASH ? '/' : '%';
        adv(lx);
        Opd *r = parse_unary(lx);
        if (!r) { opd_clear(l); gv_free(l); return NULL; }
        /* cppcheck-suppress nullPointerRedundantCheck */
        Opd *o = (Opd *)gv_calloc(1, sizeof(Opd));
        o->k = OPD_BIN; o->binop = op; o->l = l; o->r = r; l = o;
    }
    return l;
}
static Opd *parse_add(Lex *lx) {
    Opd *l = parse_mul(lx);
    while (l && (pk(lx)->t == T_PLUS || pk(lx)->t == T_DASH)) {
        char op = pk(lx)->t == T_PLUS ? '+' : '-';
        adv(lx);
        Opd *r = parse_mul(lx);
        if (!r) { opd_clear(l); gv_free(l); return NULL; }
        /* cppcheck-suppress nullPointerRedundantCheck */
        Opd *o = (Opd *)gv_calloc(1, sizeof(Opd));
        o->k = OPD_BIN; o->binop = op; o->l = l; o->r = r; l = o;
    }
    return l;
}
/* Fill a by-value operand from the value-expression parser. */
static int parse_operand(Lex *lx, Opd *o) {
    memset(o, 0, sizeof(*o));
    Opd *e = parse_add(lx);
    if (!e) return -1;
    *o = *e; gv_free(e); /* move: children ownership transfers */
    return 0;
}

static Expr *parse_cmp_impl(Lex *lx) {
    if (pk(lx)->t == T_LP) {
        adv(lx);
        Expr *e = parse_or(lx);
        if (!e || eat(lx, T_RP, "')'")) { expr_free(e); return NULL; }
        return e;
    }
    if (kw(pk(lx), "not")) {
        adv(lx);
        Expr *c = parse_cmp(lx);
        if (!c) return NULL;
        Expr *e = (Expr *)gv_calloc(1, sizeof(Expr));
        e->k = EX_NOT; e->l = c; return e;
    }
    if (kw(pk(lx), "exists")) {
        adv(lx);
        if (eat(lx, T_LP, "'('")) return NULL;
        Pattern *p = (Pattern *)gv_calloc(1, sizeof(Pattern));
        if (parse_pattern(lx, p)) { pattern_clear(p); gv_free(p); return NULL; }
        if (eat(lx, T_RP, "')'")) { pattern_clear(p); gv_free(p); return NULL; }
        Expr *e = (Expr *)gv_calloc(1, sizeof(Expr));
        e->k = EX_EXISTS; e->pat = p; return e;
    }
    Expr *e = (Expr *)gv_calloc(1, sizeof(Expr));
    e->k = EX_CMP;
    if (parse_operand(lx, &e->a)) { expr_free(e); return NULL; }
    Tok *op = pk(lx);
    /* IS [NOT] NULL */
    if (kw(op, "is")) {
        adv(lx);
        int notnull = 0;
        if (kw(pk(lx), "not")) { notnull = 1; adv(lx); }
        if (!kw(pk(lx), "null")) { snprintf(lx->err, CY_ERR, "expected NULL"); expr_free(e); return NULL; }
        adv(lx);
        e->op = notnull ? C_ISNOTNULL : C_ISNULL;
        return e;
    }
    /* operand IN <listExpr>  (list literal, range(), variable, comprehension) */
    if (kw(op, "in")) {
        adv(lx);
        e->op = C_IN;
        if (parse_operand(lx, &e->b)) { expr_free(e); return NULL; }
        return e;
    }
    if      (op->t == T_EQ) { e->op = C_EQ; adv(lx); }
    else if (op->t == T_NE) { e->op = C_NE; adv(lx); }
    else if (op->t == T_LT) { e->op = C_LT; adv(lx); }
    else if (op->t == T_GT) { e->op = C_GT; adv(lx); }
    else if (op->t == T_LE) { e->op = C_LE; adv(lx); }
    else if (op->t == T_GE) { e->op = C_GE; adv(lx); }
    else if (kw(op, "contains")) { e->op = C_CONTAINS; adv(lx); }
    else if (kw(op, "starts")) { adv(lx); if (!kw(pk(lx), "with")) { snprintf(lx->err, CY_ERR, "expected WITH"); expr_free(e); return NULL; } adv(lx); e->op = C_STARTS; }
    else if (kw(op, "ends")) { adv(lx); if (!kw(pk(lx), "with")) { snprintf(lx->err, CY_ERR, "expected WITH"); expr_free(e); return NULL; } adv(lx); e->op = C_ENDS; }
    else { e->op = C_TRUTHY; return e; }  /* bare operand in boolean context: truthiness test */
    if (parse_operand(lx, &e->b)) { expr_free(e); return NULL; }
    return e;
}
/* Depth-guarded wrapper: boolean grouping '(', 'NOT ...', and nested WHERE all
 * re-enter the recursive-descent boolean cycle here. */
static Expr *parse_cmp(Lex *lx) {
    if (lx->depth >= CY_MAX_RECURSION_DEPTH) { snprintf(lx->err, CY_ERR, "expression nesting too deep"); return NULL; }
    lx->depth++;
    Expr *e = parse_cmp_impl(lx);
    lx->depth--;
    return e;
}
static Expr *parse_and(Lex *lx) {
    Expr *l = parse_cmp(lx);
    while (l && kw(pk(lx), "and")) {
        adv(lx);
        Expr *r = parse_cmp(lx);
        if (!r) { expr_free(l); return NULL; }
        /* cppcheck-suppress nullPointerRedundantCheck */
        Expr *e = (Expr *)gv_calloc(1, sizeof(Expr));
        e->k = EX_AND; e->l = l; e->r = r; l = e;
    }
    return l;
}
static Expr *parse_or(Lex *lx) {
    Expr *l = parse_and(lx);
    while (l && kw(pk(lx), "or")) {
        adv(lx);
        Expr *r = parse_and(lx);
        if (!r) { expr_free(l); return NULL; }
        /* cppcheck-suppress nullPointerRedundantCheck */
        Expr *e = (Expr *)gv_calloc(1, sizeof(Expr));
        e->k = EX_OR; e->l = l; e->r = r; l = e;
    }
    return l;
}

/* engine */

struct GV_CypherEngine {
    GV_KnowledgeGraph *kg; char err[CY_ERR];
    char **pname; char **pval; size_t nparam;   /* query parameters ($name) */
    char **idx_label; char **idx_prop; size_t nidx; /* tracked CREATE INDEX declarations */
};
/* Current engine during a run() (for parameters + EXISTS pattern eval). */
static GV_CypherEngine *g_eng = NULL;
static const char *cy_param_lookup(const char *name) {
    if (!g_eng || !name) return NULL;
    for (size_t i = 0; i < g_eng->nparam; i++)
        if (strcmp(g_eng->pname[i], name) == 0) return g_eng->pval[i];
    return NULL;
}

typedef struct { char *var; int is_rel; int is_val; uint64_t id; char *pred; } Bind; /* is_val: pred holds a scalar value (UNWIND/WITH) */
typedef struct { Bind *b; size_t n, cap; } Row;
typedef struct { Row *r; size_t n, cap; } RowSet;

static void row_free(Row *rw) {
    for (size_t i = 0; i < rw->n; i++) { gv_free(rw->b[i].var); gv_free(rw->b[i].pred); }
    gv_free(rw->b); rw->b = NULL; rw->n = rw->cap = 0;
}
static Row row_copy(const Row *s) {
    Row d; memset(&d, 0, sizeof(d));
    size_t cap = s->n ? s->n : 1;
    d.b = (Bind *)gv_alloc(cap * sizeof(Bind));
    if (!d.b) { d.cap = 0; d.n = 0; return d; } /* OOM: degrade to empty row; cap==0 keeps binds consistent */
    d.cap = cap;
    for (size_t i = 0; i < s->n; i++) {
        d.b[i].var = gv_dup_cstr(s->b[i].var);
        d.b[i].is_rel = s->b[i].is_rel;
        d.b[i].is_val = s->b[i].is_val;
        d.b[i].id = s->b[i].id;
        d.b[i].pred = s->b[i].pred ? gv_dup_cstr(s->b[i].pred) : NULL;
    }
    d.n = s->n;
    return d;
}
/* Structural equality of two rows' bindings (same vars, same bound values in
 * order) — used to enforce WITH DISTINCT. */
static int row_proj_eq(const Row *a, const Row *b) {
    if (a->n != b->n) return 0;
    for (size_t i = 0; i < a->n; i++) {
        const Bind *x = &a->b[i], *y = &b->b[i];
        if ((x->var == NULL) != (y->var == NULL)) return 0;
        if (x->var && strcmp(x->var, y->var) != 0) return 0;
        if (x->is_rel != y->is_rel || x->is_val != y->is_val) return 0;
        if (x->is_val || x->is_rel) { if (strcmp(x->pred ? x->pred : "", y->pred ? y->pred : "") != 0) return 0; }
        if (!x->is_val && x->id != y->id) return 0;
    }
    return 1;
}
static Bind *row_find(const Row *rw, const char *var) {
    if (!var) return NULL;
    for (size_t i = 0; i < rw->n; i++) if (rw->b[i].var && strcmp(rw->b[i].var, var) == 0) return &rw->b[i];
    return NULL;
}
static void row_bind_node(Row *rw, const char *var, uint64_t id) {
    if (!var) return;
    if (rw->n == rw->cap) {
        size_t nc = rw->cap ? rw->cap * 2 : 4;
        Bind *nb = (Bind *)gv_realloc(rw->b, nc * sizeof(Bind));
        if (!nb) return;  /* OOM: skip bind rather than deref NULL */
        rw->b = nb; rw->cap = nc;
    }
    rw->b[rw->n].var = gv_dup_cstr(var); rw->b[rw->n].is_rel = 0; rw->b[rw->n].is_val = 0; rw->b[rw->n].id = id; rw->b[rw->n].pred = NULL; rw->n++;
}
static void row_bind_rel(Row *rw, const char *var, const char *pred, uint64_t rel_id) {
    if (!var) return;
    if (rw->n == rw->cap) {
        size_t nc = rw->cap ? rw->cap * 2 : 4;
        Bind *nb = (Bind *)gv_realloc(rw->b, nc * sizeof(Bind));
        if (!nb) return;  /* OOM: skip bind rather than deref NULL */
        rw->b = nb; rw->cap = nc;
    }
    rw->b[rw->n].var = gv_dup_cstr(var); rw->b[rw->n].is_rel = 1; rw->b[rw->n].is_val = 0; rw->b[rw->n].id = rel_id; rw->b[rw->n].pred = gv_dup_cstr(pred); rw->n++;
}
static void row_bind_val(Row *rw, const char *var, const char *val) {
    if (!var) return;
    if (rw->n == rw->cap) {
        size_t nc = rw->cap ? rw->cap * 2 : 4;
        Bind *nb = (Bind *)gv_realloc(rw->b, nc * sizeof(Bind));
        if (!nb) return;  /* OOM: skip this bind rather than deref NULL */
        rw->b = nb; rw->cap = nc;
    }
    rw->b[rw->n].var = gv_dup_cstr(var); rw->b[rw->n].is_rel = 0; rw->b[rw->n].is_val = 1; rw->b[rw->n].id = 0; rw->b[rw->n].pred = gv_dup_cstr(val); rw->n++;
}
static void rs_add(RowSet *rs, Row rw) {
    if (rs->n == rs->cap) {
        size_t nc = rs->cap ? rs->cap * 2 : 16;
        Row *nr = (Row *)gv_realloc(rs->r, nc * sizeof(Row));
        if (!nr) { row_free(&rw); return; }  /* OOM: drop the row (freeing it) rather than deref NULL */
        rs->r = nr; rs->cap = nc;
    }
    rs->r[rs->n++] = rw;
}
static void rs_free(RowSet *rs) { for (size_t i = 0; i < rs->n; i++) row_free(&rs->r[i]); gv_free(rs->r); rs->r = NULL; rs->n = rs->cap = 0; }

/* Value of a relationship property, or NULL. Walks the relation's property bag. */
static const char *cy_rel_prop(GV_KnowledgeGraph *kg, uint64_t rel_id, const char *key) {
    const GV_KGRelation *rr = kg_get_relation(kg, rel_id);
    if (!rr) return NULL;
    for (const GV_KGProp *p = rr->properties; p; p = p->next)
        if (p->key && strcmp(p->key, key) == 0) return p->value.type == GV_PROP_STRING ? p->value.as.s : NULL;
    return NULL;
}
/* For alternation, kg_for_each_hop can't OR types, so scan all edges (NULL) and
 * filter here; for 0/1 types the hop layer already filtered. */
static const char *rel_scan_type(const Rel *r) { return r->ntypes > 1 ? NULL : r->type; }
static int rel_type_match(const Rel *r, const char *pred) {
    if (r->ntypes <= 1) return 1;
    if (!pred) return 0;
    for (size_t i = 0; i < r->ntypes; i++) if (r->types[i] && strcmp(r->types[i], pred) == 0) return 1;
    return 0;
}
/* A matched edge satisfies the pattern's inline relationship-property constraints. */
static int rel_ok(GV_KnowledgeGraph *kg, const Rel *r, uint64_t rel_id) {
    for (size_t i = 0; i < r->np; i++) {
        const char *pv = cy_rel_prop(kg, rel_id, r->pk[i]);
        if (!pv || strcmp(pv, r->pv[i]) != 0) return 0;
    }
    return 1;
}
static int node_ok(GV_KnowledgeGraph *kg, const Node *n, uint64_t id) {
    const GV_KGEntity *e = kg_get_entity(kg, id);
    if (!e) return 0;
    if (n->label && (!e->type || strcmp(e->type, n->label) != 0)) return 0;
    /* multi-label: entity type must equal every requested label (they must all
     * be identical in this single-type model, so extras beyond the first must match too). */
    for (size_t i = 0; i < n->nlabels; i++)
        if (n->labels[i] && (!e->type || strcmp(e->type, n->labels[i]) != 0)) return 0;
    for (size_t i = 0; i < n->np; i++) {
        if (strcmp(n->pk[i], "name") == 0) { if (!e->name || strcmp(e->name, n->pv[i]) != 0) return 0; }
        else { const char *pv = kg_get_entity_prop(kg, id, n->pk[i]); if (!pv || strcmp(pv, n->pv[i]) != 0) return 0; }
    }
    return 1;
}

/* Candidate ids for node i given current bindings. Returns count (<=cap). */
static int node_candidates(GV_CypherEngine *eng, const Node *n, const Row *row,
                           uint64_t *out, size_t cap) {
    Bind *b = row_find(row, n->var);
    if (b && !b->is_rel) { out[0] = b->id; return 1; }
    const char *nm = NULL;
    for (size_t i = 0; i < n->np; i++) if (strcmp(n->pk[i], "name") == 0) nm = n->pv[i];
    if (nm) return kg_find_entities_by_name(eng->kg, nm, out, cap);
    if (n->label) return kg_find_entities_by_type(eng->kg, n->label, out, cap);
    return kg_all_entity_ids(eng->kg, out, cap);
}

static int cy_in_arr(const uint64_t *a, int n, uint64_t x) {
    for (int i = 0; i < n; i++) if (a[i] == x) return 1;
    return 0;
}

/*
 * O(1)-per-hop edge expansion.
 *
 * Each pattern edge is expanded by walking the endpoint's pre-linked adjacency
 * arrays via kg_for_each_hop — a pure pointer dereference per neighbour, with no
 * hashing, id->node resolution, triple materialisation or string copies. A hop
 * yields the neighbour id and an interior pointer to the edge predicate (valid
 * for the read-only lifetime of the match), which is all the traversal needs.
 */
typedef struct { uint64_t id; uint64_t rel_id; const char *pred; } CyHop;
typedef struct {
    CyHop *buf;
    int    n;
    int    cap;
    int    oom;
} CyHopVec;

static int cy_hop_collect(const GV_KGHop *h, void *ctx) {
    CyHopVec *v = (CyHopVec *)ctx;
    if (v->n >= v->cap) {
        int nc = v->cap ? v->cap * 2 : 16;
        CyHop *tmp = (CyHop *)gv_realloc(v->buf, (size_t)nc * sizeof(CyHop));
        if (!tmp) { v->oom = 1; return 1; /* stop */ }
        v->buf = tmp;
        v->cap = nc;
    }
    v->buf[v->n].id = h->neighbor_id;
    v->buf[v->n].rel_id = h->relation_id;
    v->buf[v->n].pred = h->predicate;
    v->n++;
    return 0;
}

/* Gather the neighbours of `id` over an edge of `type`/`dir` into `vec`
 * (constant work per neighbour). Returns 0, or -1 on allocation failure. */
static int cy_expand(GV_KnowledgeGraph *kg, uint64_t id, const char *type,
                     int dir, CyHopVec *vec) {
    vec->buf = NULL; vec->n = 0; vec->cap = 0; vec->oom = 0;
    /* dir: 1=outgoing, -1=incoming, 0=both — maps directly onto kg_for_each_hop. */
    kg_for_each_hop(kg, id, dir, type, cy_hop_collect, vec);
    return vec->oom ? -1 : 0;
}

/* BFS: collect nodes reachable from `start` in [minh,maxh] hops via rel type/dir. */
static void cy_varlen_endpoints(GV_KnowledgeGraph *kg, const Rel *r, uint64_t start,
                                int dir, int minh, int maxh, uint64_t *out, int *nout, int cap) {
    *nout = 0;
    const char *scan_type = rel_scan_type(r);  /* NULL for alternation; filtered per hop below */
    uint64_t cur[1024]; int ncur = 0; cur[ncur++] = start;
    uint64_t visited[8192]; int nvis = 0; visited[nvis++] = start;
    for (int depth = 1; depth <= maxh && ncur > 0; depth++) {
        uint64_t next[1024]; int nnext = 0;
        for (int i = 0; i < ncur; i++) {
            CyHopVec vec;
            if (cy_expand(kg, cur[i], scan_type, dir, &vec) != 0) { gv_free(vec.buf); continue; }
            for (int ti = 0; ti < vec.n; ti++) {
                /* Apply the SAME per-hop filters as the single-hop path: without
                 * these a varlen edge like [:A|B {p:'x'}*1..2] traversed every
                 * edge type and ignored inline relationship-property constraints. */
                if (!rel_type_match(r, vec.buf[ti].pred)) continue;
                if (!rel_ok(kg, r, vec.buf[ti].rel_id)) continue;
                uint64_t other = vec.buf[ti].id;
                if (!cy_in_arr(visited, nvis, other)) {
                    if (nvis < 8192) visited[nvis++] = other;
                    if (nnext < 1024) next[nnext++] = other;
                }
            }
            gv_free(vec.buf);
        }
        if (depth >= minh)
            for (int i = 0; i < nnext; i++)
                if (*nout < cap && !cy_in_arr(out, *nout, next[i])) out[(*nout)++] = next[i];
        ncur = nnext;
        for (int i = 0; i < nnext; i++) cur[i] = next[i];
    }
}

/* DFS: bind node idx (forced id if forced!=0), traverse rels, emit complete rows. */
static void dfs(GV_CypherEngine *eng, const Pattern *p, size_t idx, uint64_t forced,
                Row *cur, RowSet *out) {
    const Node *n = &p->node[idx];
    uint64_t cbuf[256]; int cnt;
    if (forced) { cbuf[0] = forced; cnt = 1; }
    else cnt = node_candidates(eng, n, cur, cbuf, 256);
    for (int ci = 0; ci < cnt; ci++) {
        uint64_t id = cbuf[ci];
        Bind *bound = row_find(cur, n->var);
        if (bound && !bound->is_rel && bound->id != id) continue;
        if (!node_ok(eng->kg, n, id)) continue;
        Row row2 = row_copy(cur);
        int added_node = 0;
        if (n->var && !bound) { row_bind_node(&row2, n->var, id); added_node = 1; }
        (void)added_node;
        if (idx + 1 == p->nn) { rs_add(out, row2); continue; }
        const Rel *r = &p->rel[idx];

        /* Variable-length path: expand endpoints reachable in [minh,maxh] hops. */
        if (r->varlen) {
            uint64_t ends[512]; int nends = 0;
            cy_varlen_endpoints(eng->kg, r, id, r->dir, r->minh, r->maxh, ends, &nends, 512);
            for (int ei = 0; ei < nends; ei++) {
                Row row3 = row_copy(&row2);
                dfs(eng, p, idx + 1, ends[ei], &row3, out);
                row_free(&row3);
            }
            row_free(&row2);
            continue;
        }

        /* Expand this edge via O(1)-per-hop adjacency (pointer deref, no copy). */
        CyHopVec vec;
        if (cy_expand(eng->kg, id, rel_scan_type(r), r->dir, &vec) != 0) {
            gv_free(vec.buf); row_free(&row2); continue;
        }
        for (int ti = 0; ti < vec.n; ti++) {
            uint64_t other = vec.buf[ti].id;
            const char *pred = vec.buf[ti].pred;
            uint64_t rel_id = vec.buf[ti].rel_id;
            if (!rel_type_match(r, pred)) continue;              /* alternation :A|B filter */
            if (!rel_ok(eng->kg, r, rel_id)) continue;           /* inline rel-property filter */
            /* rel var conflict check */
            Bind *rb = row_find(&row2, r->var);
            if (rb && rb->is_rel && rb->pred && strcmp(rb->pred, pred) != 0) continue;
            Row row3 = row_copy(&row2);
            if (r->var && !rb) row_bind_rel(&row3, r->var, pred, rel_id);
            dfs(eng, p, idx + 1, other, &row3, out);
            row_free(&row3);
        }
        gv_free(vec.buf);
        row_free(&row2);
    }
}

/* Match a pattern, extending each input row; returns new RowSet. optional=keep unmatched. */
static RowSet match_pattern(GV_CypherEngine *eng, const Pattern *p, RowSet *in, int optional) {
    RowSet out; memset(&out, 0, sizeof(out));
    RowSet seed; memset(&seed, 0, sizeof(seed));
    int own_seed = 0;
    if (in->n == 0) { Row empty; memset(&empty, 0, sizeof(empty)); rs_add(&seed, empty); in = &seed; own_seed = 1; }
    for (size_t i = 0; i < in->n; i++) {
        size_t before = out.n;
        dfs(eng, p, 0, 0, &in->r[i], &out);
        if (optional && out.n == before) rs_add(&out, row_copy(&in->r[i]));
    }
    if (own_seed) rs_free(&seed);
    return out;
}

/* value eval */

static int eval_expr(GV_KnowledgeGraph *kg, const Expr *e, const Row *row);
static int is_num(const char *s, double *d);
static char *fmt_num(double v) {
    char b[48];
    if (v == (double)(long long)v && v < 1e15 && v > -1e15) snprintf(b, sizeof(b), "%lld", (long long)v);
    else snprintf(b, sizeof(b), "%g", v);
    return gv_dup_cstr(b);
}
/* ---- Typed value layer (Cypher scalar type system) ----
 * The engine renders values to strings for output/lists/grouping, but predicate
 * evaluation and arithmetic use a tagged CyVal so that NULL is distinct from the
 * empty string (three-valued logic), integers divide as integers, and booleans
 * are a first-class type. KG properties are schemaless strings that still coerce
 * to numbers in numeric contexts (preserving existing behaviour). */
typedef enum { CT_NULL, CT_INT, CT_FLOAT, CT_BOOL, CT_STR } CyType;
typedef struct { CyType t; long long i; double f; char *s; } CyVal;
static CyVal eval_val(GV_KnowledgeGraph *kg, const Opd *o, const Row *row);

static CyVal cv_null(void)      { CyVal v; v.t = CT_NULL; v.i = 0; v.f = 0; v.s = NULL; return v; }
static CyVal cv_int(long long x){ CyVal v = cv_null(); v.t = CT_INT;   v.i = x; v.f = (double)x; return v; }
static CyVal cv_float(double x) { CyVal v = cv_null(); v.t = CT_FLOAT; v.f = x; return v; }
static CyVal cv_bool(int x)     { CyVal v = cv_null(); v.t = CT_BOOL;  v.i = x ? 1 : 0; return v; }
static CyVal cv_str(char *own)  { CyVal v = cv_null(); v.t = CT_STR;   v.s = own; return v; } /* takes ownership */
static void  cv_free(CyVal *v)  { if (v->t == CT_STR) { gv_free(v->s); v->s = NULL; } }

/* Numeric view: 1 if v is a number (int/float/bool, or numeric string). */
static int cv_num(const CyVal *v, double *d) {
    switch (v->t) {
        case CT_INT:   *d = (double)v->i; return 1;
        case CT_FLOAT: *d = v->f; return 1;
        case CT_BOOL:  *d = (double)v->i; return 1;
        case CT_STR:   return is_num(v->s, d);
        default:       return 0;
    }
}
/* Render to display string (untagged); caller owns the result. */
static char *cv_render(const CyVal *v) {
    switch (v->t) {
        case CT_NULL:  return gv_dup_cstr("");
        case CT_INT:   { char b[32]; snprintf(b, sizeof(b), "%lld", v->i); return gv_dup_cstr(b); }
        case CT_FLOAT: return fmt_num(v->f);
        case CT_BOOL:  return gv_dup_cstr(v->i ? "true" : "false");
        case CT_STR:   return gv_dup_cstr(v->s ? v->s : "");
    }
    return gv_dup_cstr("");
}
/* Type-aware three-way comparison mirroring cmp_vals (numeric coercion else strcmp). */
static int cv_cmp(const CyVal *a, const CyVal *b) {
    double da, db;
    if (cv_num(a, &da) && cv_num(b, &db)) return (da < db) ? -1 : (da > db) ? 1 : 0;
    char *as = cv_render(a), *bs = cv_render(b);
    int r = strcmp(as, bs);
    gv_free(as); gv_free(bs);
    return (r < 0) ? -1 : (r > 0) ? 1 : 0;
}

static int cy_is_list(const char *s) { return s && s[0] == CY_LTAG; }
static char *cy_list_encode(char **elems, size_t n) {
    /* cppcheck-suppress uninitvar */
    size_t len = 1;
    for (size_t i = 0; i < n; i++) len += strlen(elems[i]) + 1;
    char *r = (char *)gv_alloc(len + 1);
    if (!r) return NULL;
    r[0] = CY_LTAG; size_t p = 1;
    for (size_t i = 0; i < n; i++) { if (i) r[p++] = CY_LSEP; size_t l = strlen(elems[i]); memcpy(r + p, elems[i], l); p += l; }
    r[p] = 0; return r;
}
/* Split an encoded list into heap element strings. Returns count. */
static size_t cy_list_split(const char *s, char ***out) {
    *out = NULL;
    if (!cy_is_list(s) || s[1] == 0) return 0;
    size_t cap = 8, n = 0; char **e = (char **)gv_alloc(cap * sizeof(char *));
    if (!e) return 0;
    const char *p = s + 1;
    for (;;) {
        const char *q = strchr(p, CY_LSEP);
        size_t l = q ? (size_t)(q - p) : strlen(p);
        if (n == cap) {
            char **ne = (char **)gv_realloc(e, cap * 2 * sizeof(char *));
            if (!ne) { for (size_t i = 0; i < n; i++) gv_free(e[i]); gv_free(e); return 0; }
            e = ne; cap *= 2;
        }
        e[n] = (char *)gv_alloc(l + 1);
        if (!e[n]) { for (size_t i = 0; i < n; i++) gv_free(e[i]); gv_free(e); return 0; }
        memcpy(e[n], p, l); e[n][l] = 0; n++;
        if (!q) break;
        p = q + 1;
    }
    *out = e; return n;
}
static int cy_is_map(const char *s) { return s && s[0] == CY_MTAG; }
/* Look up a key in an encoded map; returns value (owned) or "". */
static char *cy_map_get(const char *s, const char *key) {
    if (!cy_is_map(s)) return gv_dup_cstr("");
    const char *p = s + 1;
    while (*p) {
        const char *kv = strchr(p, CY_MKV); if (!kv) break;
        size_t kl = (size_t)(kv - p);
        const char *pe = strchr(kv + 1, CY_MPAIR);
        size_t vl = pe ? (size_t)(pe - (kv + 1)) : strlen(kv + 1);
        if (strlen(key) == kl && strncmp(p, key, kl) == 0) { char *r = gv_alloc(vl + 1); memcpy(r, kv + 1, vl); r[vl] = 0; return r; }
        if (!pe) break;
        p = pe + 1;
    }
    return gv_dup_cstr("");
}
/* Path encoding: '\x03' + <nodes list> + '\x1e' + <rels list>. */
#define CY_PTAG '\x03'
#define CY_PSEP '\x1e'
static int cy_is_path(const char *s) { return s && s[0] == CY_PTAG; }
static char *cy_path_encode(char **nodes, size_t nn, char **rels, size_t nr) {
    char *nl = cy_list_encode(nodes, nn), *rl = cy_list_encode(rels, nr);
    if (!nl || !rl) { gv_free(nl); gv_free(rl); return NULL; }
    size_t a = strlen(nl), b = strlen(rl);
    char *r = (char *)gv_alloc(1 + a + 1 + b + 1);
    if (!r) { gv_free(nl); gv_free(rl); return NULL; }
    r[0] = CY_PTAG; memcpy(r + 1, nl, a); r[1 + a] = CY_PSEP; memcpy(r + 2 + a, rl, b); r[2 + a + b] = 0;
    gv_free(nl); gv_free(rl); return r;
}
/* Extract the nodes-list (want_rels=0) or rels-list (want_rels=1) from a path. */
static char *cy_path_part(const char *s, int want_rels) {
    if (!cy_is_path(s)) return gv_dup_cstr("");
    const char *sep = strchr(s + 1, CY_PSEP);
    if (!sep) return gv_dup_cstr("");
    if (!want_rels) { size_t l = (size_t)(sep - (s + 1)); char *r = gv_alloc(l + 1); memcpy(r, s + 1, l); r[l] = 0; return r; }
    return gv_dup_cstr(sep + 1);
}
/* Render an (encoded) list/map to display form; passes scalars through. */
static char *cy_finalize(char *v) {
    if (cy_is_path(v)) { char *nl = cy_path_part(v, 0); char *r = cy_finalize(nl); gv_free(v); return r; }
    if (cy_is_map(v)) { /* render {k: val, ...} */
        size_t len = 2; const char *p = v + 1;
        for (const char *c = p; *c; c++) len += 3;
        char *r = (char *)gv_alloc(len + 1);
        if (!r) return v;   /* OOM: pass encoded value through unchanged */
        size_t w = 0; r[w++] = '{';
        int first = 1;
        while (*p) {
            const char *kv = strchr(p, CY_MKV); if (!kv) break;
            const char *pe = strchr(kv + 1, CY_MPAIR);
            size_t kl = (size_t)(kv - p), vl = pe ? (size_t)(pe - (kv + 1)) : strlen(kv + 1);
            if (!first) { r[w++] = ','; r[w++] = ' '; } first = 0;
            memcpy(r + w, p, kl); w += kl; r[w++] = ':'; r[w++] = ' ';
            memcpy(r + w, kv + 1, vl); w += vl;
            if (!pe) break;
            p = pe + 1;
        }
        r[w++] = '}'; r[w] = 0; gv_free(v); return r;
    }
    if (!cy_is_list(v)) return v;
    char **e; size_t n = cy_list_split(v, &e);
    size_t len = 2; for (size_t i = 0; i < n; i++) len += strlen(e[i]) + 2;
    char *r = (char *)gv_alloc(len + 1);
    if (!r) {   /* OOM: free split elems and pass encoded value through */
        for (size_t i = 0; i < n; i++) gv_free(e[i]);
        gv_free(e); return v;
    }
    size_t p = 0;
    r[p++] = '[';
    for (size_t i = 0; i < n; i++) { if (i) { r[p++] = ','; r[p++] = ' '; } size_t l = strlen(e[i]); memcpy(r + p, e[i], l); p += l; gv_free(e[i]); }
    r[p++] = ']'; r[p] = 0;
    gv_free(e); gv_free(v); return r;
}
static char *val_of(GV_KnowledgeGraph *kg, const Opd *o, const Row *row) {
    switch (o->k) {
        case OPD_LIT:
            if (o->littype == 4) return gv_dup_cstr("");                                  /* null -> empty display */
            if (o->littype == 3) return gv_dup_cstr(o->lit && o->lit[0] == '1' ? "true" : "false");
            return gv_dup_cstr(o->lit ? o->lit : "");
        case OPD_PARAM: { const char *v = cy_param_lookup(o->var); return gv_dup_cstr(v ? v : ""); }
        case OPD_NEG: case OPD_BIN: { CyVal v = eval_val(kg, o, row); char *s = cv_render(&v); cv_free(&v); return s; }
        case OPD_CASE: {
            for (size_t i = 0; i < o->ncase; i++)
                if (eval_expr(kg, o->cw[i], row)) return val_of(kg, o->ct[i], row);
            return o->celse ? val_of(kg, o->celse, row) : gv_dup_cstr("");
        }
        case OPD_LIST: {
            char **els = (char **)gv_alloc((o->nargs ? o->nargs : 1) * sizeof(char *));
            if (!els) return gv_dup_cstr("");
            for (size_t i = 0; i < o->nargs; i++) els[i] = val_of(kg, o->args[i], row);
            char *r = cy_list_encode(els, o->nargs);
            for (size_t i = 0; i < o->nargs; i++) gv_free(els[i]);
            gv_free(els); return r ? r : gv_dup_cstr("");
        }
        case OPD_INDEX: {
            char *lv = val_of(kg, o->l, row), *iv = val_of(kg, o->r, row);
            if (cy_is_map(lv)) { char *r = cy_map_get(lv, iv); gv_free(lv); gv_free(iv); return r; }
            double di = 0; is_num(iv, &di); gv_free(iv);
            char **e; size_t n = cy_list_split(lv, &e);
            char *r = gv_dup_cstr("");
            long idx = (long)di; if (idx < 0) idx += (long)n;
            if (idx >= 0 && (size_t)idx < n) { gv_free(r); r = gv_dup_cstr(e[idx]); }
            for (size_t i = 0; i < n; i++) gv_free(e[i]);
            gv_free(e); gv_free(lv); return r;
        }
        case OPD_MAP: {
            /* encode: tag + key\x06value\x07... */
            size_t len = 1;
            char **vals = (char **)gv_alloc((o->nargs ? o->nargs : 1) * sizeof(char *));
            for (size_t i = 0; i < o->nargs; i++) { vals[i] = val_of(kg, o->args[i], row); len += strlen(o->mk[i]) + strlen(vals[i]) + 2; }
            char *r = (char *)gv_alloc(len + 1); r[0] = CY_MTAG; size_t p = 1;
            for (size_t i = 0; i < o->nmk; i++) {
                if (i) r[p++] = CY_MPAIR;
                size_t kl = strlen(o->mk[i]); memcpy(r + p, o->mk[i], kl); p += kl;
                r[p++] = CY_MKV;
                size_t vl = strlen(vals[i]); memcpy(r + p, vals[i], vl); p += vl;
            }
            r[p] = 0;
            for (size_t i = 0; i < o->nargs; i++) gv_free(vals[i]);
            gv_free(vals); return r;
        }
        case OPD_PATCOMP: {
            if (!g_eng || !o->pcpat) return gv_dup_cstr("");
            RowSet seed; memset(&seed, 0, sizeof(seed));
            rs_add(&seed, row_copy(row));
            RowSet m = match_pattern(g_eng, o->pcpat, &seed, 0);
            char **outp = (char **)gv_alloc((m.n ? m.n : 1) * sizeof(char *)); size_t no = 0;
            for (size_t i = 0; i < m.n; i++) {
                if (o->pcwhere && !eval_expr(kg, o->pcwhere, &m.r[i])) continue;
                outp[no++] = val_of(kg, o->pcproj, &m.r[i]);
            }
            char *r = cy_list_encode(outp, no);
            for (size_t i = 0; i < no; i++) gv_free(outp[i]);
            gv_free(outp); rs_free(&m); rs_free(&seed);
            return r;
        }
        case OPD_LISTCOMP: {
            char *lv = val_of(kg, o->lclist, row);
            char **e; size_t n = cy_list_split(lv, &e); gv_free(lv);
            char **outp = (char **)gv_alloc((n ? n : 1) * sizeof(char *)); size_t no = 0;
            for (size_t i = 0; i < n; i++) {
                Row tmp = row_copy(row); row_bind_val(&tmp, o->lcvar, e[i]);
                int keep = o->lcwhere ? eval_expr(kg, o->lcwhere, &tmp) : 1;
                if (keep) outp[no++] = val_of(kg, o->lcproj ? o->lcproj : o->lclist, &tmp);
                row_free(&tmp); gv_free(e[i]);
            }
            gv_free(e);
            char *r = cy_list_encode(outp, no);
            for (size_t i = 0; i < no; i++) gv_free(outp[i]);
            gv_free(outp); return r;
        }
        case OPD_FUNC: {
            const char *fn = o->fname;
            char *a0 = o->nargs > 0 ? val_of(kg, o->args[0], row) : gv_dup_cstr("");
            char *res = NULL;
            if (strcasecmp(fn, "toupper") == 0) { for (char *p = a0; *p; p++) *p = (char)toupper((unsigned char)*p); res = a0; a0 = NULL; }
            else if (strcasecmp(fn, "tolower") == 0) { for (char *p = a0; *p; p++) *p = (char)tolower((unsigned char)*p); res = a0; a0 = NULL; }
            else if (strcasecmp(fn, "trim") == 0) { char *s = a0; while (*s && isspace((unsigned char)*s)) s++; char *e = s + strlen(s); while (e > s && isspace((unsigned char)e[-1])) e--; res = gv_alloc((size_t)(e - s) + 1); memcpy(res, s, (size_t)(e - s)); res[e - s] = 0; }
            else if (strcasecmp(fn, "size") == 0 || strcasecmp(fn, "length") == 0) {
                if (cy_is_path(a0)) { char *nl = cy_path_part(a0, 0); char **e; size_t n = cy_list_split(nl, &e); for (size_t i=0;i<n;i++) gv_free(e[i]); gv_free(e); gv_free(nl); res = fmt_num(n ? (double)(n - 1) : 0); }
                else if (cy_is_list(a0)) { char **e; size_t n = cy_list_split(a0, &e); for (size_t i=0;i<n;i++) gv_free(e[i]); gv_free(e); res = fmt_num((double)n); }
                else res = fmt_num((double)strlen(a0));
            }
            else if (strcasecmp(fn, "keys") == 0) {
                if (cy_is_map(a0)) { char *keys[256]; size_t nk = 0; const char *p = a0 + 1;
                    while (*p && nk < 256) { const char *kv = strchr(p, CY_MKV); if (!kv) break; size_t kl = (size_t)(kv - p);
                        keys[nk] = gv_alloc(kl + 1); memcpy(keys[nk], p, kl); keys[nk][kl] = 0; nk++;
                        const char *pe = strchr(kv + 1, CY_MPAIR); if (!pe) break; p = pe + 1; }
                    res = cy_list_encode(keys, nk); for (size_t i=0;i<nk;i++) gv_free(keys[i]); }
                else res = gv_dup_cstr("");
            }
            else if (strcasecmp(fn, "nodes") == 0) res = cy_path_part(a0, 0);
            else if (strcasecmp(fn, "relationships") == 0 || strcasecmp(fn, "rels") == 0) res = cy_path_part(a0, 1);
            else if (strcasecmp(fn, "range") == 0) {
                double lo = 0, hi = -1, step = 1;
                is_num(a0, &lo);
                if (o->nargs > 1) { char *s = val_of(kg, o->args[1], row); is_num(s, &hi); gv_free(s); }
                if (o->nargs > 2) { char *s = val_of(kg, o->args[2], row); is_num(s, &step); gv_free(s); }
                if (step == 0) step = 1;
                char **el = NULL; size_t n = 0, cap = 0;
                for (double v = lo; (step > 0) ? v <= hi : v >= hi; v += step) {
                    if (n == cap) { cap = cap ? cap * 2 : 8; el = (char **)gv_realloc(el, cap * sizeof(char *)); }
                    el[n++] = fmt_num(v);
                    if (n > 100000) break;
                }
                res = cy_list_encode(el ? el : (char **)&el, n);
                for (size_t i = 0; i < n; i++) gv_free(el[i]);
                gv_free(el);
            }
            else if (strcasecmp(fn, "head") == 0) { char **e; size_t n = cy_list_split(a0, &e); res = gv_dup_cstr(n ? e[0] : ""); for (size_t i=0;i<n;i++) gv_free(e[i]); gv_free(e); }
            else if (strcasecmp(fn, "last") == 0) { char **e; size_t n = cy_list_split(a0, &e); res = gv_dup_cstr(n ? e[n-1] : ""); for (size_t i=0;i<n;i++) gv_free(e[i]); gv_free(e); }
            else if (strcasecmp(fn, "tostring") == 0) { res = a0; a0 = NULL; }
            else if (strcasecmp(fn, "tointeger") == 0) { double d = 0; is_num(a0, &d); res = fmt_num((double)(long long)d); }
            else if (strcasecmp(fn, "tofloat") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(d); }
            else if (strcasecmp(fn, "abs") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(d < 0 ? -d : d); }
            else if (strcasecmp(fn, "ceil") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(ceil(d)); }
            else if (strcasecmp(fn, "floor") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(floor(d)); }
            else if (strcasecmp(fn, "round") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(floor(d + 0.5)); }
            else if (strcasecmp(fn, "sqrt") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(d > 0 ? sqrt(d) : 0); }
            else if (strcasecmp(fn, "sign") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(d > 0 ? 1 : d < 0 ? -1 : 0); }
            else if (strcasecmp(fn, "exp") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(exp(d)); }
            else if (strcasecmp(fn, "log") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(d > 0 ? log(d) : 0); }
            else if (strcasecmp(fn, "log10") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(d > 0 ? log10(d) : 0); }
            else if (strcasecmp(fn, "pow") == 0) {
                double base = 0, ex = 0; is_num(a0, &base);
                if (o->nargs > 1) { char *s = val_of(kg, o->args[1], row); is_num(s, &ex); gv_free(s); }
                res = fmt_num(pow(base, ex));
            }
            else if (strcasecmp(fn, "e") == 0) { res = fmt_num(2.718281828459045235); }
            else if (strcasecmp(fn, "pi") == 0) { res = fmt_num(3.141592653589793238); }
            else if (strcasecmp(fn, "sin") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(sin(d)); }
            else if (strcasecmp(fn, "cos") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(cos(d)); }
            else if (strcasecmp(fn, "tan") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(tan(d)); }
            else if (strcasecmp(fn, "asin") == 0) { double d = 0; is_num(a0, &d); res = fmt_num((d >= -1 && d <= 1) ? asin(d) : 0); }
            else if (strcasecmp(fn, "acos") == 0) { double d = 0; is_num(a0, &d); res = fmt_num((d >= -1 && d <= 1) ? acos(d) : 0); }
            else if (strcasecmp(fn, "atan") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(atan(d)); }
            else if (strcasecmp(fn, "atan2") == 0) {
                double y = 0, x = 0; is_num(a0, &y);
                if (o->nargs > 1) { char *s = val_of(kg, o->args[1], row); is_num(s, &x); gv_free(s); }
                res = fmt_num(atan2(y, x));
            }
            else if (strcasecmp(fn, "cot") == 0) { double d = 0; is_num(a0, &d); double t = tan(d); res = fmt_num(t != 0 ? 1.0 / t : 0); }
            else if (strcasecmp(fn, "degrees") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(d * (180.0 / 3.141592653589793238)); }
            else if (strcasecmp(fn, "radians") == 0) { double d = 0; is_num(a0, &d); res = fmt_num(d * (3.141592653589793238 / 180.0)); }
            else if (strcasecmp(fn, "haversin") == 0) { double d = 0; is_num(a0, &d); double s = sin(d / 2.0); res = fmt_num(s * s); }
            else if (strcasecmp(fn, "rand") == 0) { res = fmt_num((double)rand() / ((double)RAND_MAX + 1.0)); }
            else if (strcasecmp(fn, "toboolean") == 0) {
                res = gv_dup_cstr((a0 && (strcasecmp(a0, "true") == 0 || strcmp(a0, "1") == 0)) ? "true" : "false");
            }
            else if (strcasecmp(fn, "reverse") == 0) {
                size_t L = strlen(a0); res = gv_alloc(L + 1);
                for (size_t i = 0; i < L; i++) res[i] = a0[L - 1 - i];
                res[L] = 0;
            }
            else if (strcasecmp(fn, "left") == 0) {
                double n = 0; if (o->nargs > 1) { char *s = val_of(kg, o->args[1], row); is_num(s, &n); gv_free(s); }
                size_t L = strlen(a0), k = (size_t)(n < 0 ? 0 : n); if (k > L) k = L;
                res = gv_alloc(k + 1); memcpy(res, a0, k); res[k] = 0;
            }
            else if (strcasecmp(fn, "right") == 0) {
                double n = 0; if (o->nargs > 1) { char *s = val_of(kg, o->args[1], row); is_num(s, &n); gv_free(s); }
                size_t L = strlen(a0), k = (size_t)(n < 0 ? 0 : n); if (k > L) k = L;
                res = gv_alloc(k + 1); memcpy(res, a0 + (L - k), k); res[k] = 0;
            }
            else if (strcasecmp(fn, "replace") == 0) {
                char *from = o->nargs > 1 ? val_of(kg, o->args[1], row) : NULL;
                char *to   = o->nargs > 2 ? val_of(kg, o->args[2], row) : NULL;
                if (!from || !*from) { res = gv_dup_cstr(a0); }
                else {
                    size_t fl = strlen(from), tl = to ? strlen(to) : 0;
                    size_t cap = strlen(a0) + 1, len = 0; res = gv_alloc(cap);
                    for (const char *p = a0; *p; ) {
                        if (strncmp(p, from, fl) == 0) {
                            if (len + tl + 1 > cap) { cap = (len + tl + 1) * 2; res = gv_realloc(res, cap); }
                            if (to) { memcpy(res + len, to, tl); }
                            len += tl; p += fl;
                        } else {
                            if (len + 2 > cap) { cap = (len + 2) * 2; res = gv_realloc(res, cap); }
                            res[len++] = *p++;
                        }
                    }
                    res[len] = 0;
                }
                gv_free(from); gv_free(to);
            }
            else if (strcasecmp(fn, "split") == 0) {
                char *delim = o->nargs > 1 ? val_of(kg, o->args[1], row) : NULL;
                const char *d = (delim && *delim) ? delim : ",";
                size_t dl = strlen(d);
                char **parts = NULL; size_t np = 0, cap = 0;
                const char *start = a0;
                for (const char *p = a0; ; p++) {
                    if (strncmp(p, d, dl) == 0 || *p == '\0') {
                        size_t seg = (size_t)(p - start);
                        if (np == cap) { cap = cap ? cap * 2 : 8; parts = gv_realloc(parts, cap * sizeof(char *)); }
                        parts[np] = gv_alloc(seg + 1); memcpy(parts[np], start, seg); parts[np][seg] = 0; np++;
                        if (*p == '\0') break;
                        p += dl - 1; start = p + 1;
                    }
                }
                res = cy_list_encode(parts, np);
                for (size_t i = 0; i < np; i++) gv_free(parts[i]);
                gv_free(parts); gv_free(delim);
            }
            /* Temporal scalars. With an argument the ISO string is echoed (parsed);
             * with no argument the current wall-clock value is returned. */
            else if (strcasecmp(fn, "timestamp") == 0) { res = fmt_num((double)((long long)time(NULL) * 1000LL)); }
            else if (strcasecmp(fn, "date") == 0) {
                if (a0 && a0[0]) { res = gv_dup_cstr(a0); }
                else { char b[16]; time_t now = time(NULL); struct tm tmv; gv_gmtime_utc(&now, &tmv); strftime(b, sizeof(b), "%Y-%m-%d", &tmv); res = gv_dup_cstr(b); }
            }
            else if (strcasecmp(fn, "datetime") == 0) {
                if (a0 && a0[0]) { res = gv_dup_cstr(a0); }
                else { char b[32]; time_t now = time(NULL); struct tm tmv; gv_gmtime_utc(&now, &tmv); strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%S", &tmv); res = gv_dup_cstr(b); }
            }
            else if (strcasecmp(fn, "coalesce") == 0) {
                gv_free(a0); a0 = NULL; res = NULL;
                for (size_t i = 0; i < o->nargs; i++) {
                    CyVal cv = eval_val(kg, o->args[i], row);
                    if (cv.t != CT_NULL) { res = cv_render(&cv); cv_free(&cv); break; }
                    cv_free(&cv);
                }
                if (!res) res = gv_dup_cstr("");
            }
            else if (strcasecmp(fn, "substring") == 0) {
                double st = 0, ln = -1; if (o->nargs > 1) { char *s1 = val_of(kg, o->args[1], row); is_num(s1, &st); gv_free(s1); }
                if (o->nargs > 2) { char *s2 = val_of(kg, o->args[2], row); is_num(s2, &ln); gv_free(s2); }
                size_t L = strlen(a0); size_t s = (size_t)(st < 0 ? 0 : st); if (s > L) s = L;
                size_t n = (ln < 0) ? (L - s) : (size_t)ln; if (s + n > L) n = L - s;
                res = gv_alloc(n + 1); memcpy(res, a0 + s, n); res[n] = 0;
            }
            else if (strcasecmp(fn, "id") == 0) { Bind *b = o->nargs ? row_find(row, o->args[0]->var) : NULL; res = fmt_num(b ? (double)b->id : 0); }
            else if (strcasecmp(fn, "labels") == 0) { Bind *b = o->nargs ? row_find(row, o->args[0]->var) : NULL; const GV_KGEntity *e = b ? kg_get_entity(kg, b->id) : NULL; res = gv_dup_cstr(e && e->type ? e->type : ""); }
            else if (strcasecmp(fn, "type") == 0) { Bind *b = o->nargs ? row_find(row, o->args[0]->var) : NULL; res = gv_dup_cstr(b && b->pred ? b->pred : ""); }
            /* vector_distance(prop, $query) / vector_distance(prop, $query, 'cosine') */
            else if (strcasecmp(fn, "vector_distance") == 0 ||
                     strcasecmp(fn, "vector_distance_l2") == 0 ||
                     strcasecmp(fn, "vector_distance_cosine") == 0 ||
                     strcasecmp(fn, "vector_distance_dot") == 0 ||
                     strcasecmp(fn, "vector_distance_hamming") == 0) {
                char *query_val = o->nargs > 1 ? val_of(kg, o->args[1], row) : gv_dup_cstr("");
                const char *metric = NULL;
                if (strcasecmp(fn, "vector_distance_l2") == 0) metric = "l2";
                else if (strcasecmp(fn, "vector_distance_cosine") == 0) metric = "cosine";
                else if (strcasecmp(fn, "vector_distance_dot") == 0) metric = "dot";
                else if (strcasecmp(fn, "vector_distance_hamming") == 0) metric = "hamming";
                char *metric_buf = NULL;
                if (!metric && o->nargs > 2) {
                    metric_buf = val_of(kg, o->args[2], row);
                    metric = metric_buf;
                }
                res = cypher_eval_vector_distance(a0, query_val, metric);
                gv_free(query_val);
                gv_free(metric_buf);
            }
            else res = a0 ? gv_dup_cstr(a0) : gv_dup_cstr("");
            gv_free(a0);
            return res ? res : gv_dup_cstr("");
        }
        default: break;
    }
    /* VAR / PROP / TYPE */
    Bind *b = row_find(row, o->var);
    if (!b) return gv_dup_cstr("");
    if (b->is_val) {
        if (o->k == OPD_PROP && o->prop && cy_is_map(b->pred)) return cy_map_get(b->pred, o->prop); /* map.key */
        return gv_dup_cstr(o->k == OPD_PROP ? "" : (b->pred ? b->pred : ""));
    }
    if (b->is_rel) {
        /* r.prop -> relationship property; r / type(r) -> the type label */
        if (o->k == OPD_PROP && o->prop) {
            if (strcmp(o->prop, "type") == 0) return gv_dup_cstr(b->pred ? b->pred : "");
            const char *rv = cy_rel_prop(kg, b->id, o->prop);
            return gv_dup_cstr(rv ? rv : "");
        }
        return gv_dup_cstr(b->pred ? b->pred : "");
    }
    if (o->k == OPD_TYPE) return gv_dup_cstr(b->pred ? b->pred : "");
    const GV_KGEntity *e = kg_get_entity(kg, b->id);
    if (o->k == OPD_VAR) return gv_dup_cstr(e && e->name ? e->name : "");
    if (o->prop && strcmp(o->prop, "name") == 0) return gv_dup_cstr(e && e->name ? e->name : "");
    const char *pv = o->prop ? kg_get_entity_prop(kg, b->id, o->prop) : NULL;
    return gv_dup_cstr(pv ? pv : "");
}
static int is_num(const char *s, double *d) {
    if (!s || !*s) return 0;
    char *end; double v = strtod(s, &end);
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end) return 0;
    *d = v; return 1;
}
static int cmp_vals(const char *a, const char *b) {
    double da, db;
    if (is_num(a, &da) && is_num(b, &db)) return (da < db) ? -1 : (da > db) ? 1 : 0;
    return strcmp(a, b);
}
/* Typed scalar evaluation. Handles literals, arithmetic, type-conversion
 * functions, CASE and VAR/PROP resolution with NULL-on-absence. Compound values
 * (lists/maps/paths) and untyped functions fall back to the string renderer. */
static CyVal eval_val(GV_KnowledgeGraph *kg, const Opd *o, const Row *row) {
    switch (o->k) {
        case OPD_LIT:
            switch (o->littype) {
                case 1: return cv_int(strtoll(o->lit ? o->lit : "0", NULL, 10));
                case 2: return cv_float(o->lit ? strtod(o->lit, NULL) : 0);
                case 3: return cv_bool(o->lit && o->lit[0] == '1');
                case 4: return cv_null();
                default: return cv_str(gv_dup_cstr(o->lit ? o->lit : ""));
            }
        case OPD_PARAM: { const char *v = cy_param_lookup(o->var); return v ? cv_str(gv_dup_cstr(v)) : cv_null(); }
        case OPD_NEG: {
            CyVal a = eval_val(kg, o->l, row); double d = 0; int isn = cv_num(&a, &d);
            CyType t = a.t; cv_free(&a);
            if (!isn) return cv_null();
            return (t == CT_INT || t == CT_BOOL) ? cv_int(-(long long)d) : cv_float(-d);
        }
        case OPD_BIN: {
            CyVal a = eval_val(kg, o->l, row), b = eval_val(kg, o->r, row);
            if (a.t == CT_NULL || b.t == CT_NULL) { cv_free(&a); cv_free(&b); return cv_null(); }
            double da = 0, db = 0; int na = cv_num(&a, &da), nb = cv_num(&b, &db);
            if (o->binop == '+' && !(na && nb)) { /* string concatenation */
                char *as = cv_render(&a), *bs = cv_render(&b);
                size_t la = strlen(as), lb = strlen(bs);
                char *r = (char *)gv_alloc(la + lb + 1); memcpy(r, as, la); memcpy(r + la, bs, lb + 1);
                gv_free(as); gv_free(bs); cv_free(&a); cv_free(&b); return cv_str(r);
            }
            int bothint = (a.t == CT_INT || a.t == CT_BOOL) && (b.t == CT_INT || b.t == CT_BOOL);
            cv_free(&a); cv_free(&b);
            if (!(na && nb)) return cv_null();
            switch (o->binop) {
                case '+': return bothint ? cv_int((long long)da + (long long)db) : cv_float(da + db);
                case '-': return bothint ? cv_int((long long)da - (long long)db) : cv_float(da - db);
                case '*': return bothint ? cv_int((long long)da * (long long)db) : cv_float(da * db);
                case '/': if (bothint) { long long dd = (long long)db; return cv_int(dd ? (long long)da / dd : 0); }
                          return cv_float(db != 0 ? da / db : 0);
                case '%': { long long dd = (long long)db; return cv_int(dd ? (long long)da % dd : 0); }
            }
            return cv_null();
        }
        case OPD_CASE:
            for (size_t i = 0; i < o->ncase; i++)
                if (eval_expr(kg, o->cw[i], row)) return eval_val(kg, o->ct[i], row);
            return o->celse ? eval_val(kg, o->celse, row) : cv_null();
        case OPD_FUNC: {
            const char *fn = o->fname;
            if (strcasecmp(fn, "tointeger") == 0) { CyVal a = o->nargs ? eval_val(kg, o->args[0], row) : cv_null(); double d = 0; int n = cv_num(&a, &d); cv_free(&a); return n ? cv_int((long long)d) : cv_null(); }
            if (strcasecmp(fn, "tofloat") == 0)   { CyVal a = o->nargs ? eval_val(kg, o->args[0], row) : cv_null(); double d = 0; int n = cv_num(&a, &d); cv_free(&a); return n ? cv_float(d) : cv_null(); }
            if (strcasecmp(fn, "tostring") == 0)  { CyVal a = o->nargs ? eval_val(kg, o->args[0], row) : cv_null(); char *s = cv_render(&a); cv_free(&a); return cv_str(s); }
            if (strcasecmp(fn, "toboolean") == 0) {
                CyVal a = o->nargs ? eval_val(kg, o->args[0], row) : cv_null(); CyVal r;
                if (a.t == CT_BOOL) r = cv_bool((int)a.i);
                else { char *s = cv_render(&a); r = (strcasecmp(s, "true") == 0) ? cv_bool(1) : (strcasecmp(s, "false") == 0) ? cv_bool(0) : cv_null(); gv_free(s); }
                cv_free(&a); return r;
            }
            break; /* other functions: render via string path below */
        }
        case OPD_VAR: case OPD_PROP: case OPD_TYPE: {
            /* Resolve via the string renderer; an empty render means absent/NULL.
             * The KG stores "missing" as "" (REMOVE has no delete API), so an empty
             * string and an absent property are one and the same here. */
            char *s = val_of(kg, o, row);
            if (!s || !s[0]) { gv_free(s); return cv_null(); }
            return cv_str(s);
        }
        default: break;
    }
    /* Compound values and untyped functions: evaluate as a rendered string. */
    return cv_str(val_of(kg, o, row));
}
static int eval_expr(GV_KnowledgeGraph *kg, const Expr *e, const Row *row) {
    if (!e) return 1;
    switch (e->k) {
        case EX_OR:  return eval_expr(kg, e->l, row) || eval_expr(kg, e->r, row);
        case EX_AND: return eval_expr(kg, e->l, row) && eval_expr(kg, e->r, row);
        case EX_NOT: return !eval_expr(kg, e->l, row);
        case EX_EXISTS: {
            if (!g_eng || !e->pat) return 0;
            RowSet seed; memset(&seed, 0, sizeof(seed));
            rs_add(&seed, row_copy(row));
            RowSet m = match_pattern(g_eng, e->pat, &seed, 0);
            int res = m.n > 0;
            rs_free(&m); rs_free(&seed);
            return res;
        }
        case EX_CMP: {
            CyVal a = eval_val(kg, &e->a, row);
            int res = 0;
            if (e->op == C_ISNULL)    { res = (a.t == CT_NULL); cv_free(&a); return res; }
            if (e->op == C_ISNOTNULL) { res = (a.t != CT_NULL); cv_free(&a); return res; }
            if (e->op == C_TRUTHY) {
                /* Bare operand: true boolean, non-zero number, or non-empty string. */
                switch (a.t) {
                    case CT_NULL:  res = 0; break;
                    case CT_BOOL:  res = (a.i != 0); break;
                    case CT_INT:   res = (a.i != 0); break;
                    case CT_FLOAT: res = (a.f != 0); break;
                    default:       res = (a.s && a.s[0]); break;
                }
                cv_free(&a); return res;
            }
            if (e->op == C_IN) {
                CyVal b = eval_val(kg, &e->b, row);
                char *bv = cv_render(&b);
                if (cy_is_list(bv)) {
                    char **el; size_t n = cy_list_split(bv, &el);
                    for (size_t i = 0; i < n && !res; i++) {
                        CyVal ev; ev.t = CT_STR; ev.i = 0; ev.f = 0; ev.s = el[i]; /* borrowed */
                        if (a.t != CT_NULL && cv_cmp(&a, &ev) == 0) res = 1;
                    }
                    for (size_t i = 0; i < n; i++) gv_free(el[i]);
                    gv_free(el);
                } else res = (a.t != CT_NULL && b.t != CT_NULL && cv_cmp(&a, &b) == 0);
                gv_free(bv); cv_free(&b); cv_free(&a); return res;
            }
            CyVal b = eval_val(kg, &e->b, row);
            if (e->op == C_CONTAINS || e->op == C_STARTS || e->op == C_ENDS) {
                if (a.t != CT_NULL && b.t != CT_NULL) {
                    char *as = cv_render(&a), *bs = cv_render(&b);
                    if (e->op == C_CONTAINS) res = strstr(as, bs) != NULL;
                    else if (e->op == C_STARTS) res = strncmp(as, bs, strlen(bs)) == 0;
                    else { size_t la = strlen(as), lb = strlen(bs); res = la >= lb && strcmp(as + la - lb, bs) == 0; }
                    gv_free(as); gv_free(bs);
                }
                cv_free(&a); cv_free(&b); return res;
            }
            /* Three-valued logic: a comparison with NULL is never true. */
            if (a.t == CT_NULL || b.t == CT_NULL) { cv_free(&a); cv_free(&b); return 0; }
            int c = cv_cmp(&a, &b);
            cv_free(&a); cv_free(&b);
            switch (e->op) {
                case C_EQ: return c == 0;
                case C_NE: return c != 0;
                case C_LT: return c < 0;
                case C_GT: return c > 0;
                case C_LE: return c <= 0;
                case C_GE: return c >= 0;
                default: return 0;
            }
        }
    }
    return 0;
}

/* projection */

typedef struct { char **cells; size_t n, cap; } Cells;
static void cells_add(Cells *c, char *v) {
    if (c->n == c->cap) { c->cap = c->cap ? c->cap * 2 : 64; c->cells = (char **)gv_realloc(c->cells, c->cap * sizeof(char *)); }
    c->cells[c->n++] = v;
}

static int cy_dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}
/* percentileCont (linear interpolation) / percentileDisc (nearest-rank) over a
 * mutable value array; sorts in place. Matches Neo4j semantics. */
static double cy_percentile(double *vals, size_t n, double p, int cont) {
    if (n == 0) return 0;
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    qsort(vals, n, sizeof(double), cy_dcmp);
    if (cont) {
        double rank = p * (double)(n - 1);
        size_t lo = (size_t)floor(rank), hi = (size_t)ceil(rank);
        if (lo == hi) return vals[lo];
        return vals[lo] + (rank - (double)lo) * (vals[hi] - vals[lo]);
    }
    size_t idx = (size_t)ceil(p * (double)n);
    if (idx > 0) idx--;
    if (idx >= n) idx = n - 1;
    return vals[idx];
}

static char *col_name(const Ret *r) {
    char buf[160];
    if (r->alias) return gv_dup_cstr(r->alias);
    const char *fn = NULL;
    switch (r->agg) {
        case AG_COUNTSTAR: return gv_dup_cstr("count(*)");
        case AG_COUNT: fn = "count"; break;
        case AG_COLLECT: fn = "collect"; break;
        case AG_SUM: fn = "sum"; break; case AG_AVG: fn = "avg"; break;
        case AG_MIN: fn = "min"; break; case AG_MAX: fn = "max"; break;
        case AG_STDEV: fn = "stDev"; break; case AG_STDEVP: fn = "stDevP"; break;
        case AG_PCTCONT: fn = "percentileCont"; break; case AG_PCTDISC: fn = "percentileDisc"; break;
        case AG_NONE: default: fn = NULL; break;
    }
    char base[128];
    if (r->opd.k == OPD_TYPE) snprintf(base, sizeof(base), "type(%s)", r->opd.var);
    else if (r->opd.k == OPD_PROP) snprintf(base, sizeof(base), "%s.%s", r->opd.var, r->opd.prop);
    else if (r->opd.k == OPD_LIT) snprintf(base, sizeof(base), "%s", r->opd.lit ? r->opd.lit : "");
    else snprintf(base, sizeof(base), "%s", r->opd.var ? r->opd.var : "");
    if (fn) snprintf(buf, sizeof(buf), "%s(%s)", fn, base);
    else snprintf(buf, sizeof(buf), "%s", base);
    return gv_dup_cstr(buf);
}

/* Order rows (array of row-value arrays) by ORDER BY operands. */
typedef struct { char **v; } PRow;

static int has_agg(const Ret *items, size_t ni) {
    for (size_t i = 0; i < ni; i++) if (items[i].agg != AG_NONE) return 1;
    return 0;
}

static int build_projection(GV_CypherEngine *eng, RowSet *rows, Ret *items, size_t ni,
                            Ord *ords, size_t no, int distinct, long skip, long limit,
                            GV_CypherResult *res) {
    GV_KnowledgeGraph *kg = eng->kg;
    /* Compute a value matrix: prows[r][c] for non-aggregate projection first. */
    int agg = has_agg(items, ni);

    /* Build output rows as arrays of strings (ni columns). */
    Cells out; memset(&out, 0, sizeof(out));
    size_t out_rows = 0;

    if (!agg) {
        for (size_t r = 0; r < rows->n; r++) {
            for (size_t c = 0; c < ni; c++) {
                char *v = (items[c].opd.k == OPD_TYPE)
                    ? val_of(kg, &items[c].opd, &rows->r[r])
                    : val_of(kg, &items[c].opd, &rows->r[r]);
                cells_add(&out, v);
            }
            out_rows++;
        }
    } else {
        /* group by the non-aggregate items; aggregate the rest. */
        /* Simple approach: build group key = concatenation of non-agg values. */
        typedef struct { char *key; size_t *rowidx; size_t nr, cap; } Grp;
        Grp *groups = NULL; size_t ng = 0, gcap = 0;
        for (size_t r = 0; r < rows->n; r++) {
            char keybuf[1024]; keybuf[0] = 0;
            for (size_t c = 0; c < ni; c++) {
                if (items[c].agg != AG_NONE) continue;
                char *v = val_of(kg, &items[c].opd, &rows->r[r]);
                strncat(keybuf, v, sizeof(keybuf) - strlen(keybuf) - 1);
                strncat(keybuf, "\x1f", sizeof(keybuf) - strlen(keybuf) - 1);
                gv_free(v);
            }
            size_t gi;
            for (gi = 0; gi < ng; gi++) if (strcmp(groups[gi].key, keybuf) == 0) break;
            if (gi == ng) {
                if (ng == gcap) { gcap = gcap ? gcap * 2 : 8; groups = (Grp *)gv_realloc(groups, gcap * sizeof(Grp)); }
                groups[ng].key = gv_dup_cstr(keybuf);
                groups[ng].rowidx = NULL; groups[ng].nr = groups[ng].cap = 0;
                ng++;
            }
            Grp *g = &groups[gi];
            if (g->nr == g->cap) { g->cap = g->cap ? g->cap * 2 : 8; g->rowidx = (size_t *)gv_realloc(g->rowidx, g->cap * sizeof(size_t)); }
            g->rowidx[g->nr++] = r;
        }
        for (size_t gi = 0; gi < ng; gi++) {
            Grp *g = &groups[gi];
            for (size_t c = 0; c < ni; c++) {
                if (items[c].agg == AG_NONE) {
                    cells_add(&out, val_of(kg, &items[c].opd, &rows->r[g->rowidx[0]]));
                } else if (items[c].agg == AG_COUNTSTAR) {
                    char b[32]; snprintf(b, sizeof(b), "%zu", g->nr); cells_add(&out, gv_dup_cstr(b));
                } else if (items[c].agg == AG_COUNT) {
                    char b[32]; snprintf(b, sizeof(b), "%zu", g->nr); cells_add(&out, gv_dup_cstr(b));
                } else if (items[c].agg == AG_COLLECT) {
                    char **el = (char **)gv_alloc((g->nr ? g->nr : 1) * sizeof(char *));
                    for (size_t k = 0; k < g->nr; k++) el[k] = val_of(kg, &items[c].opd, &rows->r[g->rowidx[k]]);
                    cells_add(&out, cy_list_encode(el, g->nr)); /* rendered as [..] at finalize */
                    for (size_t k = 0; k < g->nr; k++) gv_free(el[k]);
                    gv_free(el);
                } else { /* sum/avg/min/max */
                    double acc = 0, mn = 0, mx = 0; size_t cntn = 0;
                    for (size_t k = 0; k < g->nr; k++) {
                        char *v = val_of(kg, &items[c].opd, &rows->r[g->rowidx[k]]);
                        double d; if (is_num(v, &d)) { if (cntn == 0) { mn = mx = d; } else { if (d < mn) mn = d; if (d > mx) mx = d; } acc += d; cntn++; }
                        gv_free(v);
                    }
                    double outv = 0;
                    if (items[c].agg == AG_SUM) outv = acc;
                    else if (items[c].agg == AG_AVG) outv = cntn ? acc / cntn : 0;
                    else if (items[c].agg == AG_MIN) outv = mn;
                    else if (items[c].agg == AG_MAX) outv = mx;
                    else if (items[c].agg == AG_STDEV || items[c].agg == AG_STDEVP) {
                        double mean = cntn ? acc / cntn : 0, ss = 0;
                        for (size_t k = 0; k < g->nr; k++) {
                            char *v = val_of(kg, &items[c].opd, &rows->r[g->rowidx[k]]);
                            double d; if (is_num(v, &d)) { double dv = d - mean; ss += dv * dv; }
                            gv_free(v);
                        }
                        if (items[c].agg == AG_STDEV) outv = cntn > 1 ? sqrt(ss / (cntn - 1)) : 0;
                        else outv = cntn > 0 ? sqrt(ss / cntn) : 0;
                    } else { /* AG_PCTCONT / AG_PCTDISC */
                        double *vals = (double *)gv_alloc((g->nr ? g->nr : 1) * sizeof(double));
                        size_t nv = 0;
                        for (size_t k = 0; k < g->nr; k++) {
                            char *v = val_of(kg, &items[c].opd, &rows->r[g->rowidx[k]]);
                            double d; if (is_num(v, &d)) vals[nv++] = d;
                            gv_free(v);
                        }
                        outv = cy_percentile(vals, nv, items[c].aggarg, items[c].agg == AG_PCTCONT);
                        gv_free(vals);
                    }
                    char b[48]; snprintf(b, sizeof(b), "%g", outv); cells_add(&out, gv_dup_cstr(b));
                }
            }
            out_rows++;
        }
        for (size_t gi = 0; gi < ng; gi++) { gv_free(groups[gi].key); gv_free(groups[gi].rowidx); }
        gv_free(groups);
    }

    /* Column names (needed to map ORDER BY keys to projected columns). */
    char **colnames = (char **)gv_alloc(ni * sizeof(char *));
    for (size_t c = 0; c < ni; c++) colnames[c] = col_name(&items[c]);

    /* ORDER BY: map each key to a projected column when possible; otherwise
     * (non-aggregate) evaluate the operand against the corresponding match row. */
    if (no > 0 && out_rows > 1) {
        int ocol[CY_MAXORD];
        for (size_t oi = 0; oi < no; oi++) {
            char key[160];
            if (ords[oi].opd.k == OPD_TYPE) snprintf(key, sizeof(key), "type(%s)", ords[oi].opd.var);
            else if (ords[oi].opd.k == OPD_PROP) snprintf(key, sizeof(key), "%s.%s", ords[oi].opd.var, ords[oi].opd.prop);
            else snprintf(key, sizeof(key), "%s", ords[oi].opd.var ? ords[oi].opd.var : "");
            ocol[oi] = -1;
            for (size_t c = 0; c < ni; c++) {
                if (items[c].alias && ords[oi].opd.k == OPD_VAR && strcmp(items[c].alias, ords[oi].opd.var) == 0) { ocol[oi] = (int)c; break; }
                if (strcmp(colnames[c], key) == 0) { ocol[oi] = (int)c; break; }
            }
        }
        for (size_t i = 1; i < out_rows; i++) {
            for (size_t j = i; j > 0; j--) {
                int c = 0;
                for (size_t oi = 0; oi < no && c == 0; oi++) {
                    if (ocol[oi] >= 0) {
                        c = cmp_vals(out.cells[j * ni + ocol[oi]], out.cells[(j - 1) * ni + ocol[oi]]);
                    } else if (!agg) {
                        char *va = val_of(kg, &ords[oi].opd, &rows->r[j]);
                        char *vb = val_of(kg, &ords[oi].opd, &rows->r[j - 1]);
                        c = cmp_vals(va, vb); gv_free(va); gv_free(vb);
                    }
                    if (ords[oi].desc) c = -c;
                }
                if (c < 0) {
                    for (size_t cc = 0; cc < ni; cc++) {
                        char *tmp = out.cells[j * ni + cc];
                        out.cells[j * ni + cc] = out.cells[(j - 1) * ni + cc];
                        out.cells[(j - 1) * ni + cc] = tmp;
                    }
                    if (!agg) { Row tr = rows->r[j]; rows->r[j] = rows->r[j - 1]; rows->r[j - 1] = tr; }
                } else break;
            }
        }
    }

    /* DISTINCT after ordering (removes later duplicate rows). */
    if (distinct && out_rows > 1) {
        for (size_t i = 0; i < out_rows; i++) {
            for (size_t j = i + 1; j < out_rows; ) {
                int same = 1;
                for (size_t c = 0; c < ni; c++)
                    if (strcmp(out.cells[i * ni + c], out.cells[j * ni + c]) != 0) { same = 0; break; }
                if (same) {
                    for (size_t c = 0; c < ni; c++) gv_free(out.cells[j * ni + c]);
                    memmove(&out.cells[j * ni], &out.cells[(j + 1) * ni], (out_rows - j - 1) * ni * sizeof(char *));
                    out_rows--;
                    out.n -= ni;
                } else j++;
            }
        }
    }

    /* SKIP / LIMIT */
    long start = skip > 0 ? skip : 0;
    long end = (limit >= 0) ? (start + limit) : (long)out_rows;
    if (start > (long)out_rows) start = out_rows;
    if (end > (long)out_rows) end = out_rows;
    long kept = end - start; if (kept < 0) kept = 0;

    res->column_count = ni;
    res->column_names = colnames; /* transfer ownership */
    res->row_count = (size_t)kept;
    res->column_values = (char **)gv_alloc((kept ? (size_t)kept : 1) * ni * sizeof(char *));
    for (long r = 0; r < kept; r++)
        for (size_t c = 0; c < ni; c++)
            res->column_values[r * ni + c] = cy_finalize(out.cells[(start + r) * ni + c]);
    /* free the cells outside [start,end) */
    for (long r = 0; r < (long)out_rows; r++)
        if (r < start || r >= end)
            for (size_t c = 0; c < ni; c++) gv_free(out.cells[r * ni + c]);
    gv_free(out.cells);
    return 0;
}

/* RETURN parse */

static Agg agg_of(const Tok *t) {
    if (kw(t, "count")) return AG_COUNT;
    if (kw(t, "collect")) return AG_COLLECT;
    if (kw(t, "sum")) return AG_SUM;
    if (kw(t, "avg")) return AG_AVG;
    if (kw(t, "min")) return AG_MIN;
    if (kw(t, "max")) return AG_MAX;
    if (kw(t, "stdev")) return AG_STDEV;
    if (kw(t, "stdevp")) return AG_STDEVP;
    if (kw(t, "percentilecont")) return AG_PCTCONT;
    if (kw(t, "percentiledisc")) return AG_PCTDISC;
    return AG_NONE;
}
static int parse_return(Lex *lx, Ret *items, size_t *ni, int *distinct, int *star) {
    *ni = 0; *distinct = 0; *star = 0;
    if (kw(pk(lx), "distinct")) { *distinct = 1; adv(lx); }
    if (pk(lx)->t == T_STAR) { *star = 1; adv(lx); return 0; } /* RETURN * */
    for (;;) {
        if (*ni >= CY_MAXRET) { snprintf(lx->err, CY_ERR, "too many RETURN items"); return -1; }
        Ret *it = &items[*ni]; memset(it, 0, sizeof(*it));
        Agg a = agg_of(pk(lx));
        if (a != AG_NONE) {
            adv(lx);
            if (eat(lx, T_LP, "'('")) return -1;
            if (a == AG_COUNT && pk(lx)->t == T_STAR) { it->agg = AG_COUNTSTAR; adv(lx); }
            else {
                if (kw(pk(lx), "distinct")) adv(lx); /* count(DISTINCT x) ~ count(x) here */
                it->agg = a;
                if (parse_operand(lx, &it->opd)) return -1;
                if (agg_has_pct(a)) { /* percentileCont/Disc(expr, fraction) */
                    if (eat(lx, T_COMMA, "','")) return -1;
                    if (pk(lx)->t != T_NUMBER) { snprintf(lx->err, CY_ERR, "percentile expects a numeric fraction"); return -1; }
                    it->aggarg = strtod(adv(lx)->s, NULL);
                }
            }
            if (eat(lx, T_RP, "')'")) return -1;
        } else {
            if (parse_operand(lx, &it->opd)) return -1;
        }
        if (kw(pk(lx), "as")) { adv(lx);
            if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected alias"); return -1; }
            it->alias = gv_dup_cstr(adv(lx)->s);
        }
        (*ni)++;
        if (pk(lx)->t == T_COMMA) { adv(lx); continue; }
        break;
    }
    return 0;
}

/* CREATE/MERGE */

/* Resolve a prop value: a '\x04'-prefixed value is a variable reference into sym. */
static char *resolve_pv(GV_CypherEngine *eng, Row *sym, const char *pv) {
    if (pv && pv[0] == '\x04') {
        Bind *b = row_find(sym, pv + 1);
        if (!b) return gv_dup_cstr("");
        if (b->is_rel || b->is_val) return gv_dup_cstr(b->pred ? b->pred : "");
        const GV_KGEntity *e = kg_get_entity(eng->kg, b->id);
        return gv_dup_cstr(e && e->name ? e->name : "");
    }
    return gv_dup_cstr(pv ? pv : "");
}
static uint64_t ensure_node(GV_CypherEngine *eng, const Node *n, Row *sym, int *created) {
    if (created) *created = 0;
    Bind *b = row_find(sym, n->var);
    if (b && !b->is_rel && !b->is_val) return b->id;
    char *nm = NULL;
    for (size_t i = 0; i < n->np; i++) if (strcmp(n->pk[i], "name") == 0) nm = resolve_pv(eng, sym, n->pv[i]);
    uint64_t id = kg_add_entity(eng->kg, nm ? nm : (n->var ? n->var : "node"), n->label ? n->label : "Node", NULL, 0);
    gv_free(nm);
    if (id == 0) return 0;
    if (created) *created = 1;
    for (size_t i = 0; i < n->np; i++)
        if (strcmp(n->pk[i], "name") != 0) { char *v = resolve_pv(eng, sym, n->pv[i]); kg_set_entity_prop(eng->kg, id, n->pk[i], v); gv_free(v); }
    if (n->var) row_bind_node(sym, n->var, id);
    return id;
}
/* Create a whole pattern (chain) into the graph; sym seeds/collects var bindings. */
static void create_pattern(GV_CypherEngine *eng, const Pattern *p, Row *sym, GV_CypherResult *res) {
    uint64_t prev = 0; int c;
    for (size_t i = 0; i < p->nn; i++) {
        uint64_t id = ensure_node(eng, &p->node[i], sym, &c);
        if (id && c) res->nodes_created++;
        if (i > 0 && id && prev) {
            const Rel *r = &p->rel[i - 1];
            uint64_t s = (r->dir == -1) ? id : prev;
            uint64_t o = (r->dir == -1) ? prev : id;
            const char *rtype = r->type ? r->type : "RELATED";
            uint64_t rid = kg_add_relation(eng->kg, s, rtype, o, 1.0f);
            if (rid != 0) {
                res->relationships_created++;
                for (size_t k = 0; k < r->np; k++) {   /* inline relationship properties */
                    char *v = resolve_pv(eng, sym, r->pv[k]);
                    kg_set_relation_prop(eng->kg, rid, r->pk[k], v);
                    gv_free(v);
                }
                if (r->var && !row_find(sym, r->var)) row_bind_rel(sym, r->var, rtype, rid);
            }
        }
        prev = id;
    }
}

/* executor */

/* Parse `var.prop = expr [, ...]` into a SetItem list. The value is a full value
 * expression (arithmetic, property refs, functions), evaluated per-row at apply. */
static int parse_setlist(Lex *lx, SetItem *out, size_t *n, size_t cap) {
    for (;;) {
        if (*n >= cap) { snprintf(lx->err, CY_ERR, "too many SET items"); return -1; }
        if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected variable in SET"); return -1; }
        out[*n].var = gv_dup_cstr(adv(lx)->s); out[*n].prop = NULL; out[*n].valexpr = NULL;
        if (eat(lx, T_DOT, "'.'")) { setitem_free(&out[*n]); return -1; }
        out[*n].prop = (pk(lx)->t == T_IDENT) ? gv_dup_cstr(adv(lx)->s) : NULL;
        if (eat(lx, T_EQ, "'='")) { setitem_free(&out[*n]); return -1; }
        out[*n].valexpr = parse_add(lx);
        if (!out[*n].valexpr) { snprintf(lx->err, CY_ERR, "expected value in SET"); setitem_free(&out[*n]); return -1; }
        (*n)++;
        if (pk(lx)->t == T_COMMA) { adv(lx); continue; }
        break;
    }
    return 0;
}
static void apply_sets(GV_CypherEngine *eng, Row *row, SetItem *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        Bind *b = row_find(row, s[i].var);
        if (!b || !s[i].prop || !s[i].valexpr) continue;
        char *v = val_of(eng->kg, s[i].valexpr, row);
        if (b->is_rel)        kg_set_relation_prop(eng->kg, b->id, s[i].prop, v ? v : "");
        else if (!b->is_val)  kg_set_entity_prop(eng->kg, b->id, s[i].prop, v ? v : "");
        gv_free(v);
    }
}
static long parse_int(Lex *lx) {
    if (pk(lx)->t == T_NUMBER) return strtol(adv(lx)->s, NULL, 10);
    return -1;
}

/* Bind a path variable to the encoded path for each matched row (named elements). */
static void bind_paths(RowSet *rs, const Pattern *p, GV_KnowledgeGraph *kg, const char *pvar) {
    for (size_t r = 0; r < rs->n; r++) {
        char *nodes[CY_MAXNODE]; size_t nn = 0; char *rels[CY_MAXNODE]; size_t nr = 0;
        for (size_t i = 0; i < p->nn; i++) {
            if (p->node[i].var) { Bind *b = row_find(&rs->r[r], p->node[i].var);
                if (b && !b->is_rel && !b->is_val) { const GV_KGEntity *e = kg_get_entity(kg, b->id); nodes[nn++] = gv_dup_cstr(e && e->name ? e->name : ""); } }
            if (i + 1 < p->nn && p->rel[i].var) { Bind *b = row_find(&rs->r[r], p->rel[i].var); if (b) rels[nr++] = gv_dup_cstr(b->pred ? b->pred : ""); }
        }
        char *enc = cy_path_encode(nodes, nn, rels, nr);
        row_bind_val(&rs->r[r], pvar, enc); gv_free(enc);
        for (size_t i = 0; i < nn; i++) gv_free(nodes[i]);
        for (size_t i = 0; i < nr; i++) gv_free(rels[i]);
    }
}

/* CALL db.labels() | db.relationshipTypes() [YIELD col] — built-in procedures. */
static int run_call(GV_CypherEngine *eng, Lex *lx, GV_CypherResult *res) {
    memset(res, 0, sizeof(*res));
    adv(lx); /* CALL */
    char proc[128]; proc[0] = 0;
    while (pk(lx)->t == T_IDENT) {
        strncat(proc, adv(lx)->s, sizeof(proc) - strlen(proc) - 1);
        if (pk(lx)->t == T_DOT) { adv(lx); strncat(proc, ".", sizeof(proc) - strlen(proc) - 1); } else break;
    }
    if (pk(lx)->t == T_LP) { adv(lx); while (pk(lx)->t != T_RP && pk(lx)->t != T_EOF) adv(lx); if (eat(lx, T_RP, "')'")) return -1; }
    char *colname = NULL;
    if (kw(pk(lx), "yield")) { adv(lx); if (pk(lx)->t == T_IDENT) colname = gv_dup_cstr(adv(lx)->s); }

    char *buf[4096]; int n = 0;
    const char *defcol = "value";
    if (strcasecmp(proc, "db.labels") == 0) { n = kg_get_entity_types(eng->kg, buf, 4096); defcol = "label"; }
    else if (strcasecmp(proc, "db.relationshiptypes") == 0) { n = kg_get_predicates(eng->kg, buf, 4096); defcol = "relationshipType"; }
    else if (strcasecmp(proc, "db.indexes") == 0) {
        for (size_t k = 0; k < eng->nidx && n < 4096; k++) {
            size_t ln = strlen(eng->idx_label[k]) + strlen(eng->idx_prop[k]) + 4;
            char *s = (char *)gv_alloc(ln); snprintf(s, ln, "%s(%s)", eng->idx_label[k], eng->idx_prop[k]);
            buf[n++] = s;
        }
        defcol = "index";
    }
    else { snprintf(eng->err, CY_ERR, "unknown procedure '%s'", proc); gv_free(colname); return -1; }
    if (n < 0) n = 0;

    res->column_count = 1;
    res->column_names = (char **)gv_alloc(sizeof(char *));
    res->column_names[0] = colname ? colname : gv_dup_cstr(defcol);
    res->row_count = (size_t)n;
    res->column_values = (char **)gv_alloc((n ? (size_t)n : 1) * sizeof(char *));
    for (int i = 0; i < n; i++) { res->column_values[i] = buf[i]; } /* transfer ownership */
    return 0;
}

/* GCC -O3 emits a false-positive maybe-uninitialized for `no` here (it is
 * initialised to 0 at its declaration); the strict build already whitelists it. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
static int run(GV_CypherEngine *eng, Lex *lx, GV_CypherResult *res) {
    memset(res, 0, sizeof(*res));
    RowSet rows; memset(&rows, 0, sizeof(rows));
    Expr *pending_where = NULL;
    int rc = -1;

    /* Reading clauses: (OPTIONAL)? MATCH pattern (WHERE expr)? , repeatable */
    for (;;) {
        int optional = 0;
        if (kw(pk(lx), "optional")) { optional = 1; adv(lx); }
        if (kw(pk(lx), "match")) {
            adv(lx);
            for (;;) {
                /* optional path variable:  p = (pattern) */
                char *pathvar = NULL;
                if (pk(lx)->t == T_IDENT && lx->pos + 1 < lx->n && lx->v[lx->pos + 1].t == T_EQ) {
                    pathvar = gv_dup_cstr(adv(lx)->s); adv(lx); /* '=' */
                }
                Pattern p;
                if (parse_pattern(lx, &p)) { pattern_clear(&p); gv_free(pathvar); goto done; }
                RowSet nr = match_pattern(eng, &p, &rows, optional);
                if (pathvar) bind_paths(&nr, &p, eng->kg, pathvar);
                pattern_clear(&p); gv_free(pathvar);
                rs_free(&rows); rows = nr;
                if (pk(lx)->t == T_COMMA) { adv(lx); continue; }
                break;
            }
            if (kw(pk(lx), "where")) {
                adv(lx);
                pending_where = parse_or(lx);
                if (!pending_where) goto done;
                /* filter rows */
                RowSet f; memset(&f, 0, sizeof(f));
                for (size_t i = 0; i < rows.n; i++)
                    if (eval_expr(eng->kg, pending_where, &rows.r[i])) rs_add(&f, row_copy(&rows.r[i]));
                rs_free(&rows); rows = f;
                expr_free(pending_where); pending_where = NULL;
            }
            continue;
        }
        if (kw(pk(lx), "unwind")) {
            adv(lx);
            /* UNWIND <expr> AS var  (expr evaluates to a list per input row) */
            Opd uexpr; if (parse_operand(lx, &uexpr)) goto done;
            if (!kw(pk(lx), "as")) { snprintf(lx->err, CY_ERR, "expected AS in UNWIND"); opd_clear(&uexpr); goto done; }
            adv(lx);
            if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected UNWIND variable"); opd_clear(&uexpr); goto done; }
            char *uvar = gv_dup_cstr(adv(lx)->s);
            RowSet nr; memset(&nr, 0, sizeof(nr));
            RowSet seed; memset(&seed, 0, sizeof(seed)); int own = 0;
            RowSet *src = &rows;
            if (rows.n == 0) { Row e; memset(&e, 0, sizeof(e)); rs_add(&seed, e); src = &seed; own = 1; }
            for (size_t i = 0; i < src->n; i++) {
                char *lv = val_of(eng->kg, &uexpr, &src->r[i]);
                char **el; size_t ne = cy_list_split(lv, &el);
                if (!cy_is_list(lv)) { /* scalar -> single-element */
                    Row rw = row_copy(&src->r[i]); row_bind_val(&rw, uvar, lv); rs_add(&nr, rw);
                } else {
                    for (size_t k = 0; k < ne; k++) { Row rw = row_copy(&src->r[i]); row_bind_val(&rw, uvar, el[k]); rs_add(&nr, rw); }
                }
                for (size_t k = 0; k < ne; k++) gv_free(el[k]);
                gv_free(el); gv_free(lv);
            }
            if (own) rs_free(&seed);
            rs_free(&rows); rows = nr;
            opd_clear(&uexpr); gv_free(uvar);
            continue;
        }
        if (kw(pk(lx), "with")) {
            adv(lx);
            int wdistinct = 0;
            if (kw(pk(lx), "distinct")) { wdistinct = 1; adv(lx); }
            /* WITH item [AS alias] , ... [WHERE expr] — supports aggregation. */
            Opd wopd[CY_MAXRET]; char *walias[CY_MAXRET]; Agg wagg[CY_MAXRET]; double wpct[CY_MAXRET]; size_t nw2 = 0;
            int werr = 0, anyagg = 0;
            for (;;) {
                if (nw2 >= CY_MAXRET) { werr = 1; break; }
                memset(&wopd[nw2], 0, sizeof(wopd[nw2]));
                Agg a = agg_of(pk(lx));
                if (a != AG_NONE) {
                    adv(lx);
                    if (eat(lx, T_LP, "'('")) { werr = 1; break; }
                    if (a == AG_COUNT && pk(lx)->t == T_STAR) { wagg[nw2] = AG_COUNTSTAR; adv(lx); }
                    else { if (kw(pk(lx), "distinct")) adv(lx); wagg[nw2] = a; if (parse_operand(lx, &wopd[nw2])) { werr = 1; break; }
                           if (agg_has_pct(a)) { if (eat(lx, T_COMMA, "','")) { werr = 1; break; } if (pk(lx)->t != T_NUMBER) { werr = 1; break; } wpct[nw2] = strtod(adv(lx)->s, NULL); } }
                    if (eat(lx, T_RP, "')'")) { werr = 1; break; }
                    anyagg = 1;
                } else { wagg[nw2] = AG_NONE; if (parse_operand(lx, &wopd[nw2])) { werr = 1; break; } }
                walias[nw2] = NULL;
                if (kw(pk(lx), "as")) { adv(lx); if (pk(lx)->t == T_IDENT) walias[nw2] = gv_dup_cstr(adv(lx)->s); }
                nw2++;
                if (pk(lx)->t == T_COMMA) { adv(lx); continue; }
                break;
            }
            if (werr) { for (size_t i=0;i<=nw2 && i<CY_MAXRET;i++){ opd_clear(&wopd[i]); if(i<nw2) gv_free(walias[i]); } snprintf(lx->err, CY_ERR, "invalid WITH"); goto done; }
            RowSet nr; memset(&nr, 0, sizeof(nr));
            if (!anyagg) {
                for (size_t i = 0; i < rows.n; i++) {
                    Row nwr; memset(&nwr, 0, sizeof(nwr));
                    for (size_t c = 0; c < nw2; c++) {
                        if (wopd[c].k == OPD_VAR && !walias[c]) {
                            Bind *b = row_find(&rows.r[i], wopd[c].var);
                            if (b) { if (b->is_rel) row_bind_rel(&nwr, b->var, b->pred, b->id);
                                     else if (b->is_val) row_bind_val(&nwr, b->var, b->pred);
                                     else row_bind_node(&nwr, b->var, b->id);
                                     continue; }
                        }
                        char *v = val_of(eng->kg, &wopd[c], &rows.r[i]);
                        const char *nm = walias[c] ? walias[c] : wopd[c].var ? wopd[c].var : "expr";
                        row_bind_val(&nwr, nm, v); gv_free(v);
                    }
                    if (wdistinct) {   /* WITH DISTINCT: skip a row equal to one already emitted */
                        int dup = 0;
                        for (size_t e = 0; e < nr.n && !dup; e++) if (row_proj_eq(&nr.r[e], &nwr)) dup = 1;
                        if (dup) { row_free(&nwr); continue; }
                    }
                    rs_add(&nr, nwr);
                }
            } else {
                /* group by non-agg item values */
                size_t *grp = (size_t *)gv_alloc((rows.n ? rows.n : 1) * sizeof(size_t)); /* group id per row */
                char **gkey = NULL; size_t ng = 0;
                for (size_t i = 0; i < rows.n; i++) {
                    char kb[1024]; kb[0] = 0;
                    for (size_t c = 0; c < nw2; c++) { if (wagg[c] != AG_NONE) continue;
                        char *v = val_of(eng->kg, &wopd[c], &rows.r[i]); strncat(kb, v, sizeof(kb)-strlen(kb)-1); strncat(kb, "\x1f", sizeof(kb)-strlen(kb)-1); gv_free(v); }
                    size_t gi; for (gi = 0; gi < ng; gi++) if (strcmp(gkey[gi], kb) == 0) break;
                    if (gi == ng) { gkey = (char **)gv_realloc(gkey, (ng+1)*sizeof(char*)); gkey[ng] = gv_dup_cstr(kb); ng++; }
                    grp[i] = gi;
                }
                for (size_t gi = 0; gi < ng; gi++) {
                    size_t first = 0; int found = 0;
                    for (size_t i = 0; i < rows.n; i++) if (grp[i] == gi) { first = i; found = 1; break; }
                    if (!found) continue;
                    Row nwr; memset(&nwr, 0, sizeof(nwr));
                    for (size_t c = 0; c < nw2; c++) {
                        const char *nm = walias[c] ? walias[c] : wopd[c].var ? wopd[c].var : "agg";
                        if (wagg[c] == AG_NONE) {
                            if (wopd[c].k == OPD_VAR && !walias[c]) { Bind *b = row_find(&rows.r[first], wopd[c].var);
                                if (b) { if (b->is_rel) row_bind_rel(&nwr,b->var,b->pred,b->id); else if (b->is_val) row_bind_val(&nwr,b->var,b->pred); else row_bind_node(&nwr,b->var,b->id); continue; } }
                            char *v = val_of(eng->kg, &wopd[c], &rows.r[first]); row_bind_val(&nwr, nm, v); gv_free(v);
                        } else if (wagg[c] == AG_COUNTSTAR || wagg[c] == AG_COUNT) {
                            size_t cnt = 0; for (size_t i = 0; i < rows.n; i++) if (grp[i] == gi) cnt++;
                            char b[32]; snprintf(b, sizeof(b), "%zu", cnt); row_bind_val(&nwr, nm, b);
                        } else if (wagg[c] == AG_COLLECT) {
                            char **el = NULL; size_t ne = 0;
                            for (size_t i = 0; i < rows.n; i++) if (grp[i]==gi) { el = (char**)gv_realloc(el,(ne+1)*sizeof(char*)); el[ne++] = val_of(eng->kg,&wopd[c],&rows.r[i]); }
                            char *enc = cy_list_encode(el, ne); row_bind_val(&nwr, nm, enc); gv_free(enc);
                            for (size_t i=0;i<ne;i++) gv_free(el[i]);
                            gv_free(el);
                        } else { double acc=0,mn=0,mx=0; size_t cn=0;
                            for (size_t i = 0; i < rows.n; i++) if (grp[i]==gi) { char *v=val_of(eng->kg,&wopd[c],&rows.r[i]); double d; if(is_num(v,&d)){ if(!cn){mn=mx=d;} else { if(d<mn)mn=d; if(d>mx)mx=d;} acc+=d; cn++; } gv_free(v); }
                            double ov;
                            if (wagg[c]==AG_STDEV || wagg[c]==AG_STDEVP) {
                                double mean = cn?acc/cn:0, ss=0;
                                for (size_t i = 0; i < rows.n; i++) if (grp[i]==gi) { char *v=val_of(eng->kg,&wopd[c],&rows.r[i]); double d; if(is_num(v,&d)){ double dv=d-mean; ss+=dv*dv; } gv_free(v); }
                                ov = wagg[c]==AG_STDEV ? (cn>1?sqrt(ss/(cn-1)):0) : (cn>0?sqrt(ss/cn):0);
                            } else if (agg_has_pct(wagg[c])) {
                                double *pv = (double*)gv_alloc((cn?cn:1)*sizeof(double)); size_t pn=0;
                                for (size_t i = 0; i < rows.n; i++) if (grp[i]==gi) { char *v=val_of(eng->kg,&wopd[c],&rows.r[i]); double d; if(is_num(v,&d)) pv[pn++]=d; gv_free(v); }
                                ov = cy_percentile(pv, pn, wpct[c], wagg[c]==AG_PCTCONT); gv_free(pv);
                            } else {
                                ov = wagg[c]==AG_SUM?acc : wagg[c]==AG_AVG?(cn?acc/cn:0) : wagg[c]==AG_MIN?mn:mx;
                            }
                            char *v = fmt_num(ov); row_bind_val(&nwr, nm, v); gv_free(v);
                        }
                    }
                    rs_add(&nr, nwr);
                }
                for (size_t i = 0; i < ng; i++) gv_free(gkey[i]);
                gv_free(gkey); gv_free(grp);
            }
            rs_free(&rows); rows = nr;
            for (size_t i = 0; i < nw2; i++) { opd_clear(&wopd[i]); gv_free(walias[i]); }
            if (kw(pk(lx), "where")) {
                adv(lx);
                Expr *we = parse_or(lx);
                if (!we) goto done;
                RowSet f; memset(&f, 0, sizeof(f));
                for (size_t i = 0; i < rows.n; i++)
                    if (eval_expr(eng->kg, we, &rows.r[i])) rs_add(&f, row_copy(&rows.r[i]));
                rs_free(&rows); rows = f;
                expr_free(we);
            }
            continue;
        }
        if (kw(pk(lx), "foreach")) {
            adv(lx);
            /* FOREACH ( var IN listExpr | CREATE pattern | SET ... ) */
            if (eat(lx, T_LP, "'('")) goto done;
            if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected FOREACH variable"); goto done; }
            char *fvar = gv_dup_cstr(adv(lx)->s);
            if (!kw(pk(lx), "in")) { snprintf(lx->err, CY_ERR, "expected IN"); gv_free(fvar); goto done; }
            adv(lx);
            Opd flist; if (parse_operand(lx, &flist)) { gv_free(fvar); goto done; }
            if (eat(lx, T_PIPE, "'|'")) { gv_free(fvar); opd_clear(&flist); goto done; }
            /* inner clause: CREATE pattern  OR  SET items */
            int inner_create = 0; Pattern fp; memset(&fp, 0, sizeof(fp));
            SetItem fsets[CY_MAXSET]; size_t fns = 0;
            if (kw(pk(lx), "create")) { adv(lx); inner_create = 1; if (parse_pattern(lx, &fp)) { pattern_clear(&fp); gv_free(fvar); opd_clear(&flist); goto done; } }
            else if (kw(pk(lx), "set")) { adv(lx); if (parse_setlist(lx, fsets, &fns, CY_MAXSET)) { for (size_t i=0;i<fns;i++) setitem_free(&fsets[i]); gv_free(fvar); opd_clear(&flist); goto done; } }
            else { snprintf(lx->err, CY_ERR, "FOREACH body must be CREATE or SET"); gv_free(fvar); opd_clear(&flist); goto done; }
            if (eat(lx, T_RP, "')'")) { if (inner_create) pattern_clear(&fp); for (size_t i=0;i<fns;i++) setitem_free(&fsets[i]); gv_free(fvar); opd_clear(&flist); goto done; }
            /* execute side effects per input row per list element (rows pass through unchanged) */
            RowSet seed; memset(&seed, 0, sizeof(seed)); int own = 0; RowSet *src = &rows;
            if (rows.n == 0) { Row e; memset(&e, 0, sizeof(e)); rs_add(&seed, e); src = &seed; own = 1; }
            for (size_t i = 0; i < src->n; i++) {
                char *lv = val_of(eng->kg, &flist, &src->r[i]);
                char **el; size_t ne = cy_list_split(lv, &el);
                for (size_t k = 0; k < ne; k++) {
                    Row sym = row_copy(&src->r[i]); row_bind_val(&sym, fvar, el[k]);
                    if (inner_create) create_pattern(eng, &fp, &sym, res);
                    else apply_sets(eng, &sym, fsets, fns);
                    row_free(&sym); gv_free(el[k]);
                }
                gv_free(el); gv_free(lv);
            }
            if (own) rs_free(&seed);
            if (inner_create) pattern_clear(&fp);
            for (size_t i = 0; i < fns; i++) setitem_free(&fsets[i]);
            gv_free(fvar); opd_clear(&flist);
            continue;
        }
        break;
    }

    /* Terminal clause */
    if (kw(pk(lx), "remove")) {
        adv(lx);
        /* REMOVE var.prop [, ...] — deletes the property outright. */
        for (;;) {
            if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected variable in REMOVE"); goto done; }
            char *rv = gv_dup_cstr(adv(lx)->s);
            char *rp = NULL;
            if (pk(lx)->t == T_DOT) { adv(lx); if (pk(lx)->t == T_IDENT) rp = gv_dup_cstr(adv(lx)->s); }
            for (size_t i = 0; i < rows.n; i++) {
                Bind *b = row_find(&rows.r[i], rv);
                if (b && rp && !b->is_val) {
                    if (b->is_rel) kg_remove_relation_prop(eng->kg, b->id, rp);
                    else           kg_remove_entity_prop(eng->kg, b->id, rp);
                }
            }
            gv_free(rv); gv_free(rp);
            if (pk(lx)->t == T_COMMA) { adv(lx); continue; }
            break;
        }
        rc = 0; goto done;
    }
    if (kw(pk(lx), "create")) {
        adv(lx);
        for (;;) {
            Pattern p;
            if (parse_pattern(lx, &p)) { pattern_clear(&p); goto done; }
            if (rows.n == 0) { Row sym; memset(&sym, 0, sizeof(sym)); create_pattern(eng, &p, &sym, res); row_free(&sym); }
            else for (size_t i = 0; i < rows.n; i++) { Row sym = row_copy(&rows.r[i]); create_pattern(eng, &p, &sym, res); row_free(&sym); }
            pattern_clear(&p);
            if (pk(lx)->t == T_COMMA) { adv(lx); continue; }
            break;
        }
        rc = 0; goto done;
    }
    if (kw(pk(lx), "merge")) {
        adv(lx);
        Pattern p;
        if (parse_pattern(lx, &p)) { pattern_clear(&p); goto done; }
        /* optional ON CREATE SET / ON MATCH SET */
        SetItem oc[8] = {0}; size_t noc = 0; SetItem om[8] = {0}; size_t nom = 0; int perr = 0;
        while (kw(pk(lx), "on")) {
            adv(lx);
            int cr = kw(pk(lx), "create"), mt = kw(pk(lx), "match");
            if (!cr && !mt) { snprintf(lx->err, CY_ERR, "expected CREATE or MATCH after ON"); perr = 1; break; }
            adv(lx);
            if (!kw(pk(lx), "set")) { snprintf(lx->err, CY_ERR, "expected SET"); perr = 1; break; }
            /* cppcheck-suppress legacyUninitvar */
            adv(lx);
            if (parse_setlist(lx, cr ? oc : om, cr ? &noc : &nom, 8)) { perr = 1; break; }
        }
        if (!perr) {
            RowSet m = match_pattern(eng, &p, &rows, 0);
            if (m.n == 0) {
                Row sym; memset(&sym, 0, sizeof(sym));
                create_pattern(eng, &p, &sym, res);
                apply_sets(eng, &sym, oc, noc);
                row_free(&sym);
            } else {
                for (size_t i = 0; i < m.n; i++) apply_sets(eng, &m.r[i], om, nom);
            }
            rs_free(&m);
            rc = 0;
        }
        for (size_t i = 0; i < noc; i++) setitem_free(&oc[i]);
        for (size_t i = 0; i < nom; i++) setitem_free(&om[i]);
        pattern_clear(&p);
        goto done;
    }
    if (kw(pk(lx), "set")) {
        adv(lx);
        SetItem sets[CY_MAXSET]; size_t ns = 0;
        if (parse_setlist(lx, sets, &ns, CY_MAXSET)) { for (size_t s = 0; s < ns; s++) setitem_free(&sets[s]); goto done; }
        for (size_t i = 0; i < rows.n; i++) apply_sets(eng, &rows.r[i], sets, ns);
        for (size_t s = 0; s < ns; s++) setitem_free(&sets[s]);
        rc = 0; goto done;
    }
    if (kw(pk(lx), "detach")) { adv(lx); if (!kw(pk(lx), "delete")) { snprintf(lx->err, CY_ERR, "expected DELETE"); goto done; } }
    if (kw(pk(lx), "delete")) {
        adv(lx);
        char *vars[CY_MAXDEL]; size_t nv = 0;
        for (;;) {
            if (nv >= CY_MAXDEL || pk(lx)->t != T_IDENT) break;
            vars[nv++] = gv_dup_cstr(adv(lx)->s);
            if (pk(lx)->t == T_COMMA) { adv(lx); continue; }
            break;
        }
        /* collect unique ids to delete (kg_remove_entity cascades relations = DETACH) */
        for (size_t i = 0; i < rows.n; i++)
            for (size_t v = 0; v < nv; v++) {
                Bind *b = row_find(&rows.r[i], vars[v]);
                if (b && !b->is_rel) kg_remove_entity(eng->kg, b->id);
            }
        for (size_t v = 0; v < nv; v++) gv_free(vars[v]);
        rc = 0; goto done;
    }
    if (kw(pk(lx), "return")) {
        adv(lx);
        Ret items[CY_MAXRET]; size_t ni = 0; int distinct = 0; int star = 0;
        Ord ords[CY_MAXORD]; size_t no = 0; long skip = -1, limit = -1;
        if (parse_return(lx, items, &ni, &distinct, &star)) goto done_ret;
        if (star && rows.n > 0) { /* RETURN * -> all bound variables */
            for (size_t bi = 0; bi < rows.r[0].n && ni < CY_MAXRET; bi++) {
                memset(&items[ni], 0, sizeof(items[ni]));
                items[ni].opd.k = OPD_VAR;
                items[ni].opd.var = gv_dup_cstr(rows.r[0].b[bi].var);
                ni++;
            }
        }
        if (kw(pk(lx), "order")) { adv(lx);
            if (!kw(pk(lx), "by")) { snprintf(lx->err, CY_ERR, "expected BY"); goto done_ret; }
            adv(lx);
            for (;;) {
                if (no >= CY_MAXORD) break;
                if (parse_operand(lx, &ords[no].opd)) goto done_ret;
                ords[no].desc = 0;
                if (kw(pk(lx), "desc")) { ords[no].desc = 1; adv(lx); }
                else if (kw(pk(lx), "asc")) adv(lx);
                no++;
                if (pk(lx)->t == T_COMMA) { adv(lx); continue; }
                break;
            }
        }
        if (kw(pk(lx), "skip")) { adv(lx); skip = parse_int(lx); }
        if (kw(pk(lx), "limit")) { adv(lx); limit = parse_int(lx); }

        rc = build_projection(eng, &rows, items, ni, ords, no, distinct, skip, limit, res);
    done_ret:
        for (size_t i = 0; i < ni; i++) { opd_clear(&items[i].opd); gv_free(items[i].alias); }
        for (size_t i = 0; i < no; i++) opd_clear(&ords[i].opd);
        goto done;
    }

    if (pk(lx)->t == T_EOF) { rc = 0; goto done; } /* reading/update-only query (e.g. FOREACH, standalone MATCH) */
    snprintf(lx->err, CY_ERR, "expected RETURN, CREATE, MERGE, SET or DELETE");

done:
    expr_free(pending_where);
    rs_free(&rows);
    if (rc != 0 && lx->err[0] && !eng->err[0]) snprintf(eng->err, CY_ERR, "%s", lx->err);
    return rc;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

/* public API */

GV_CypherEngine *cypher_create(GV_KnowledgeGraph *kg) {
    if (!kg) return NULL;
    GV_CypherEngine *e = (GV_CypherEngine *)gv_calloc(1, sizeof(GV_CypherEngine));
    if (e) e->kg = kg;
    return e;
}
void cypher_destroy(GV_CypherEngine *eng) {
    if (!eng) return;
    for (size_t i = 0; i < eng->nparam; i++) { gv_free(eng->pname[i]); gv_free(eng->pval[i]); }
    gv_free(eng->pname); gv_free(eng->pval);
    for (size_t i = 0; i < eng->nidx; i++) { gv_free(eng->idx_label[i]); gv_free(eng->idx_prop[i]); }
    gv_free(eng->idx_label); gv_free(eng->idx_prop);
    gv_free(eng);
}

int cypher_set_parameter(GV_CypherEngine *eng, const char *name, const char *value) {
    if (!eng || !name || !value) return -1;
    for (size_t i = 0; i < eng->nparam; i++)
        if (strcmp(eng->pname[i], name) == 0) { gv_free(eng->pval[i]); eng->pval[i] = gv_dup_cstr(value); return 0; }
    char **nn = (char **)gv_realloc(eng->pname, (eng->nparam + 1) * sizeof(char *));
    char **nv = (char **)gv_realloc(eng->pval, (eng->nparam + 1) * sizeof(char *));
    if (!nn || !nv) { gv_free(nn); gv_free(nv); return -1; }
    eng->pname = nn; eng->pval = nv;
    eng->pname[eng->nparam] = gv_dup_cstr(name);
    eng->pval[eng->nparam] = gv_dup_cstr(value);
    eng->nparam++;
    return 0;
}

/* Build a one-row plan description for "EXPLAIN <query>": the clause pipeline
 * plus, per MATCH pattern, whether label/name predicates resolve through the
 * KG's type/name hash indexes or fall back to scans. */
static int cypher_explain(GV_CypherEngine *eng, const char *query,
                          GV_CypherResult *result) {
    (void)eng;
    memset(result, 0, sizeof(*result));
    static const char *clauses[] = {
        "MATCH", "OPTIONAL", "UNWIND", "WITH", "WHERE",
        "CREATE", "MERGE", "SET", "REMOVE", "DELETE", "FOREACH",
        "RETURN", "CALL", NULL
    };
    char plan[2048];
    size_t len = 0;
#define PLAN_APPEND(s_) do { \
        size_t n_ = strlen(s_); \
        if (len + n_ < sizeof(plan)) { memcpy(plan + len, (s_), n_); len += n_; } \
    } while (0)

    const char *p = query + 7;   /* skip "EXPLAIN" */
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    int first_clause = 1;
    for (const char **cl = clauses; *cl; cl++) {
        const char *hit = strcasestr(p, *cl);
        if (!hit) continue;
        if (!first_clause) PLAN_APPEND(" -> ");
        PLAN_APPEND(*cl);
        first_clause = 0;
        /* MATCH patterns with a :Label resolve via the type index. */
        if (strcmp(*cl, "MATCH") == 0 || strcmp(*cl, "OPTIONAL") == 0) {
            const char *q = hit + strlen(*cl);
            const char *lbl = strchr(q, ':');
            const char *stop = strstr(q, "RETURN");
            if (lbl && (!stop || lbl < stop)) {
                PLAN_APPEND(" [label predicate -> type_index lookup]");
            } else {
                PLAN_APPEND(" [full enumeration]");
            }
        }
    }
    if (first_clause) PLAN_APPEND("(no recognized clauses)");
#undef PLAN_APPEND
    plan[len] = '\0';

    result->column_count = 1;
    result->column_names = (char **)gv_calloc(1, sizeof(char *));
    result->column_values = (char **)gv_calloc(1, sizeof(char *));
    if (!result->column_names || !result->column_values) {
        gv_free(result->column_names); gv_free(result->column_values);
        memset(result, 0, sizeof(*result));
        return -1;
    }
    result->column_names[0] = gv_dup_cstr("plan");
    result->column_values[0] = gv_dup_cstr(plan);
    if (!result->column_names[0] || !result->column_values[0]) {
        cypher_free_result(result);
        return -1;
    }
    result->row_count = 1;
    return 0;
}

/* Append src's rows onto dst (UNION). Column counts must match. When !all,
 * a src row is skipped if every cell equals an existing dst row's (set union);
 * O(n^2) which is fine at the scales this engine targets. dst keeps its own
 * column_names (from the first query segment). Returns 0, or -1 on a column
 * mismatch / allocation failure. */
static int result_union_merge(GV_CypherResult *dst, const GV_CypherResult *src, int all) {
    if (dst->column_count != src->column_count) return -1;
    size_t cc = dst->column_count;
    for (size_t sr = 0; sr < src->row_count; sr++) {
        char **srow = &src->column_values[sr * cc];
        if (!all) {
            int dup = 0;
            for (size_t dr = 0; dr < dst->row_count && !dup; dr++) {
                int eq = 1;
                for (size_t c = 0; c < cc; c++) {
                    const char *a = dst->column_values[dr * cc + c];
                    const char *b = srow[c];
                    if (strcmp(a ? a : "", b ? b : "") != 0) { eq = 0; break; }
                }
                if (eq) dup = 1;
            }
            if (dup) continue;
        }
        char **nv = (char **)gv_realloc(dst->column_values, (dst->row_count + 1) * cc * sizeof(char *));
        if (!nv) return -1;
        dst->column_values = nv;
        for (size_t c = 0; c < cc; c++)
            dst->column_values[dst->row_count * cc + c] = gv_dup_cstr(srow[c] ? srow[c] : "");
        dst->row_count++;
    }
    return 0;
}

/* Execute one top-level query segment (no UNION handling). Dispatches the
 * leading clause, including the CALL { subquery } brace form. */
/* Extract (label, property) from an index DDL statement, tolerating both the
 * legacy form  CREATE INDEX ON :Label(prop)  and the modern descriptor form
 * CREATE INDEX FOR (n:Label) ON (n.prop).  Returns 0 and heap-dups *label and
 * *prop on success; -1 otherwise. Leftover tokens (to end of statement) are
 * consumed either way. */
static int parse_index_target(Lex *lx, char **label, char **prop) {
    *label = NULL; *prop = NULL;
    char *dot_prop = NULL, *lp_prop = NULL; /* n.prop form vs legacy :Label(prop) form */
    Tk prev = T_EOF;
    while (pk(lx)->t != T_EOF && pk(lx)->t != T_RC) {
        Tok *t = pk(lx);
        if (kw(t, "union")) break;
        if (t->t == T_IDENT) {
            if (prev == T_COLON && !*label) *label = gv_dup_cstr(t->s);
            else if (prev == T_DOT && !dot_prop) dot_prop = gv_dup_cstr(t->s);
            else if (prev == T_LP && !lp_prop && *label) lp_prop = gv_dup_cstr(t->s);
        }
        prev = t->t;
        adv(lx);
    }
    /* A dotted property (descriptor form `ON (n.prop)`) wins over the bare
     * ident inside parens, which for that form is just the pattern variable. */
    *prop = dot_prop ? dot_prop : lp_prop;
    if (dot_prop && lp_prop) gv_free(lp_prop);
    if (!*label || !*prop) { gv_free(*label); gv_free(*prop); *label = *prop = NULL; return -1; }
    return 0;
}

/* Build a single-column result with `label(prop)` rows describing tracked indexes. */
static int indexes_result(GV_CypherEngine *eng, GV_CypherResult *res) {
    memset(res, 0, sizeof(*res));
    res->column_count = 1;
    res->column_names = (char **)gv_alloc(sizeof(char *));
    res->column_names[0] = gv_dup_cstr("index");
    res->row_count = eng->nidx;
    res->column_values = (char **)gv_alloc((eng->nidx ? eng->nidx : 1) * sizeof(char *));
    for (size_t i = 0; i < eng->nidx; i++) {
        size_t n = strlen(eng->idx_label[i]) + strlen(eng->idx_prop[i]) + 4;
        char *s = (char *)gv_alloc(n);
        snprintf(s, n, "%s(%s)", eng->idx_label[i], eng->idx_prop[i]);
        res->column_values[i] = s;
    }
    return 0;
}

/* CREATE/DROP INDEX: advisory DDL. The KG already resolves :Label and name
 * predicates through hash indexes, so a property index does not change query
 * results — we track declarations so clients can round-trip index DDL and list
 * them via SHOW INDEXES / CALL db.indexes. */
static int run_create_index(GV_CypherEngine *eng, Lex *lx, GV_CypherResult *res) {
    adv(lx); /* CREATE */ adv(lx); /* INDEX */
    char *label = NULL, *prop = NULL;
    if (parse_index_target(lx, &label, &prop) != 0) {
        snprintf(lx->err, CY_ERR, "CREATE INDEX: expected :Label(property)"); return -1;
    }
    for (size_t i = 0; i < eng->nidx; i++) /* idempotent: ignore duplicates */
        if (strcmp(eng->idx_label[i], label) == 0 && strcmp(eng->idx_prop[i], prop) == 0) {
            gv_free(label); gv_free(prop); memset(res, 0, sizeof(*res)); return 0;
        }
    char **nl = (char **)gv_realloc(eng->idx_label, (eng->nidx + 1) * sizeof(char *));
    char **np = (char **)gv_realloc(eng->idx_prop, (eng->nidx + 1) * sizeof(char *));
    if (!nl || !np) { gv_free(nl); gv_free(np); gv_free(label); gv_free(prop); snprintf(lx->err, CY_ERR, "OOM"); return -1; }
    eng->idx_label = nl; eng->idx_prop = np;
    eng->idx_label[eng->nidx] = label; eng->idx_prop[eng->nidx] = prop; eng->nidx++;
    memset(res, 0, sizeof(*res));
    return 0;
}

static int run_drop_index(GV_CypherEngine *eng, Lex *lx, GV_CypherResult *res) {
    adv(lx); /* DROP */ adv(lx); /* INDEX */
    char *label = NULL, *prop = NULL;
    if (parse_index_target(lx, &label, &prop) != 0) {
        snprintf(lx->err, CY_ERR, "DROP INDEX: expected :Label(property)"); return -1;
    }
    for (size_t i = 0; i < eng->nidx; i++)
        if (strcmp(eng->idx_label[i], label) == 0 && strcmp(eng->idx_prop[i], prop) == 0) {
            gv_free(eng->idx_label[i]); gv_free(eng->idx_prop[i]);
            memmove(&eng->idx_label[i], &eng->idx_label[i + 1], (eng->nidx - i - 1) * sizeof(char *));
            memmove(&eng->idx_prop[i], &eng->idx_prop[i + 1], (eng->nidx - i - 1) * sizeof(char *));
            eng->nidx--; break;
        }
    gv_free(label); gv_free(prop);
    memset(res, 0, sizeof(*res));
    return 0;
}

static int run_segment(GV_CypherEngine *eng, Lex *lx, GV_CypherResult *res) {
    Tok *first = pk(lx);
    if (kw(first, "call") && lx->pos + 1 < lx->n && lx->v[lx->pos + 1].t == T_LC) {
        /* CALL { <subquery> }: run the braced read-query and surface its rows as
         * this statement's result (uncorrelated form only — the subquery does not
         * see outer variables, and a trailing outer RETURN is not supported). */
        adv(lx); /* CALL */
        adv(lx); /* '{' */
        int rc = run(eng, lx, res);
        if (rc != 0) return rc;
        if (pk(lx)->t != T_RC) { snprintf(lx->err, CY_ERR, "expected '}' to close CALL subquery"); cypher_free_result(res); return -1; }
        adv(lx); /* '}' */
        return 0;
    }
    if (kw(first, "call")) return run_call(eng, lx, res);
    /* Index DDL: CREATE INDEX .. / DROP INDEX .. (distinguished from node CREATE
     * by the INDEX keyword in the second position). */
    if (kw(first, "create") && lx->pos + 1 < lx->n && kw(&lx->v[lx->pos + 1], "index"))
        return run_create_index(eng, lx, res);
    if (kw(first, "drop") && lx->pos + 1 < lx->n && kw(&lx->v[lx->pos + 1], "index"))
        return run_drop_index(eng, lx, res);
    if (kw(first, "show") && lx->pos + 1 < lx->n && kw(&lx->v[lx->pos + 1], "indexes"))
        return indexes_result(eng, res);
    if (kw(first, "match") || kw(first, "optional") || kw(first, "create") || kw(first, "merge") || kw(first, "unwind") || kw(first, "foreach"))
        return run(eng, lx, res);
    snprintf(lx->err, CY_ERR, "expected MATCH, OPTIONAL, CREATE, MERGE, UNWIND, CALL, CREATE/DROP INDEX or SHOW INDEXES");
    return -1;
}

int cypher_execute(GV_CypherEngine *eng, const char *query, GV_CypherResult *result) {
    if (!eng || !query || !result) return -1;
    eng->err[0] = 0;

    /* EXPLAIN prefix: describe the plan without executing. */
    if (strncasecmp(query, "EXPLAIN", 7) == 0 &&
        (query[7] == ' ' || query[7] == '\n' || query[7] == '\r' || query[7] == '\t')) {
        return cypher_explain(eng, query, result);
    }

    Lex lx; memset(&lx, 0, sizeof(lx));
    if (tokenize(&lx, query)) { snprintf(eng->err, CY_ERR, "%s", lx.err); lex_free(&lx); return -1; }
    GV_CypherEngine *prev = g_eng; g_eng = eng;  /* for parameters + EXISTS */
    int rc = run_segment(eng, &lx, result);
    /* UNION [ALL]: fold each following segment into the first result. */
    while (rc == 0 && kw(pk(&lx), "union")) {
        adv(&lx);
        int all = 0;
        if (kw(pk(&lx), "all")) { adv(&lx); all = 1; }
        GV_CypherResult r2; memset(&r2, 0, sizeof(r2));
        if (run_segment(eng, &lx, &r2) != 0) { cypher_free_result(&r2); rc = -1; break; }
        if (result_union_merge(result, &r2, all) != 0) {
            snprintf(eng->err, CY_ERR, "UNION column count mismatch");
            cypher_free_result(&r2); cypher_free_result(result); rc = -1; break;
        }
        cypher_free_result(&r2);
    }
    g_eng = prev;
    if (rc != 0 && lx.err[0] && !eng->err[0]) snprintf(eng->err, CY_ERR, "%s", lx.err);
    lex_free(&lx);
    return rc;
}
void cypher_free_result(GV_CypherResult *result) {
    if (!result) return;
    if (result->column_names) {
        for (size_t i = 0; i < result->column_count; i++) gv_free(result->column_names[i]);
        gv_free(result->column_names);
    }
    if (result->column_values) {
        for (size_t i = 0; i < result->row_count * result->column_count; i++) gv_free(result->column_values[i]);
        gv_free(result->column_values);
    }
    memset(result, 0, sizeof(*result));
}
const char *cypher_last_error(const GV_CypherEngine *eng) { return (eng && eng->err[0]) ? eng->err : ""; }
