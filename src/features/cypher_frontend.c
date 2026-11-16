/**
  * @file cypher_frontend.c
  * @brief Cypher tokenizer and clause/expression parser (split from cypher.c).
  */
#include "features/cypher.h"
#include "features/cypher_internal.h"
#include "core/memory.h"
#include "core/utils.h"
#include "core/compat.h"
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

/* Portable UTC breakdown: POSIX gmtime_r vs the Windows/ucrt gmtime_s (whose
 * argument order is dest-first, like C11 Annex K). MinGW lacks gmtime_r. */
void gv_gmtime_utc(const time_t *t, struct tm *out) {
#ifdef _WIN32
    gmtime_s(out, t);
#else
    gmtime_r(t, out);
#endif
}


int agg_has_pct(Agg a) { return a == AG_PCTCONT || a == AG_PCTDISC; }

static char *sdup(const char *s, size_t n) {
    char *r = (char *)gv_alloc(n + 1);
    if (r) { memcpy(r, s, n); r[n] = 0; }
    return r;
}
int push(Lex *lx, Tk t, const char *s, size_t n) {
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
    if (s && n && !dup) return -1;  /* OOM building token text - abort tokenize */
    lx->v[lx->n].t = t;
    lx->v[lx->n].s = dup;
    lx->n++;
    return 0;
}
void lex_free(Lex *lx) {
    if (lx->v) { for (size_t i = 0; i < lx->n; i++) gv_free(lx->v[i].s); gv_free(lx->v); }
    lx->v = NULL;
}
int tokenize(Lex *lx, const char *q) {
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
Tok *pk(Lex *lx) { return &lx->v[lx->pos]; }
Tok *adv(Lex *lx) { return &lx->v[lx->pos < lx->n - 1 ? lx->pos++ : lx->pos]; }
int kw(const Tok *t, const char *k) { return t->t == T_IDENT && t->s && strcasecmp(t->s, k) == 0; }
int eat(Lex *lx, Tk t, const char *w) {
    if (pk(lx)->t != t) { snprintf(lx->err, CY_ERR, "expected %s", w); return -1; }
    adv(lx); return 0;
}

/* parser */

static void node_clear(Node *n) {
    gv_free(n->var); gv_free(n->label);
    for (size_t i = 0; i < n->nlabels; i++) gv_free(n->labels[i]);
    for (size_t i = 0; i < n->np; i++) { gv_free(n->pk[i]); gv_free(n->pv[i]); }
    memset(n, 0, sizeof(*n));
}
void expr_free(Expr *e);
void opd_clear(Opd *o) {
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
void setitem_free(SetItem *s) {
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
    if (s->lcproj) { d.lcproj = (Opd *)gv_alloc(sizeof(Opd)); *d.lcproj = opd_copy(s->lcproj); }
    if (s->l) { d.l = (Opd *)gv_alloc(sizeof(Opd)); *d.l = opd_copy(s->l); }
    if (s->r) { d.r = (Opd *)gv_alloc(sizeof(Opd)); *d.r = opd_copy(s->r); }
    return d; /* CASE/pattern-comprehension subtrees still not copied (unsupported as copied subjects) */
}
void expr_free(Expr *e) {
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
        /* rel type alternation :A|B|... - match ANY listed type */
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
int parse_pattern(Lex *lx, Pattern *p) {
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
void pattern_clear(Pattern *p) {
    for (size_t i = 0; i < p->nn; i++) {
        node_clear(&p->node[i]);
        if (i + 1 < p->nn) rel_clear(&p->rel[i]);
    }
    memset(p, 0, sizeof(*p));
}

/* operand: var[.prop] | type(var) | 'string' | number | [list] | comprehension */
Expr *parse_or(Lex *lx);       /* boolean expr (for CASE WHEN) */
Opd *parse_add(Lex *lx);       /* value expr (arithmetic) */
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
        /* reduce( acc = init, var IN listExpr | expr ) - fold a list into one
         * value. Stored as: var=acc name, l=init, lcvar=loop var, lclist=list,
         * lcproj=body expr. */
        if (strcasecmp(fn, "reduce") == 0 && pk(lx)->t == T_IDENT) {
            o->var = gv_dup_cstr(adv(lx)->s);        /* accumulator variable */
            if (eat(lx, T_EQ, "'=' in reduce")) { opd_clear(o); gv_free(o); return NULL; }
            o->l = parse_add(lx);                    /* initial value */
            if (!o->l) { opd_clear(o); gv_free(o); return NULL; }
            if (eat(lx, T_COMMA, "',' in reduce")) { opd_clear(o); gv_free(o); return NULL; }
            if (pk(lx)->t != T_IDENT) { snprintf(lx->err, CY_ERR, "expected loop var in reduce"); opd_clear(o); gv_free(o); return NULL; }
            o->lcvar = gv_dup_cstr(adv(lx)->s);      /* loop variable */
            if (!kw(pk(lx), "in")) { snprintf(lx->err, CY_ERR, "expected IN in reduce"); opd_clear(o); gv_free(o); return NULL; }
            adv(lx); /* IN */
            o->lclist = parse_add(lx);
            if (!o->lclist) { opd_clear(o); gv_free(o); return NULL; }
            if (eat(lx, T_PIPE, "'|' in reduce")) { opd_clear(o); gv_free(o); return NULL; }
            o->lcproj = parse_add(lx);               /* body expression */
            if (!o->lcproj || eat(lx, T_RP, "')'")) { opd_clear(o); gv_free(o); return NULL; }
            return o;
        }
        /* list predicate: any/all/none/single ( var IN listExpr WHERE pred ) -
         * reuses the list-comprehension fields (lcvar/lclist/lcwhere). */
        if ((strcasecmp(fn, "any") == 0 || strcasecmp(fn, "all") == 0 ||
             strcasecmp(fn, "none") == 0 || strcasecmp(fn, "single") == 0) &&
            pk(lx)->t == T_IDENT && lx->pos + 1 < lx->n && kw(&lx->v[lx->pos + 1], "in")) {
            o->lcvar = gv_dup_cstr(adv(lx)->s);
            adv(lx); /* IN */
            o->lclist = parse_add(lx);
            if (!o->lclist) { opd_clear(o); gv_free(o); return NULL; }
            if (kw(pk(lx), "where")) { adv(lx); o->lcwhere = parse_or(lx); if (!o->lcwhere) { opd_clear(o); gv_free(o); return NULL; } }
            if (eat(lx, T_RP, "')'")) { opd_clear(o); gv_free(o); return NULL; }
            return o;
        }
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
Opd *parse_add(Lex *lx) {
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
int parse_operand(Lex *lx, Opd *o) {
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
Expr *parse_or(Lex *lx) {
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

