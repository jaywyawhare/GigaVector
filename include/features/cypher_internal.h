/**
  * @file cypher_internal.h
  * @brief Internal Cypher engine types, macros and cross-file prototypes.
  *
  * Shared between cypher_frontend.c (lexer + parser) and cypher.c (executor).
  * Not a public API; do not include outside the Cypher engine.
  */
#ifndef GV_FEATURES_CYPHER_INTERNAL_H
#define GV_FEATURES_CYPHER_INTERNAL_H

#include <stddef.h>
#include <time.h>

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

typedef enum { AG_NONE, AG_COUNT, AG_COUNTSTAR, AG_COLLECT, AG_SUM, AG_AVG, AG_MIN, AG_MAX, AG_STDEV, AG_STDEVP, AG_PCTCONT, AG_PCTDISC } Agg;
typedef struct { Agg agg; Opd opd; char *alias; double aggarg; } Ret; /* aggarg: percentile p for percentileCont/Disc */
typedef struct { Opd opd; int desc; } Ord;
typedef struct { char *var; char *prop; Opd *valexpr; } SetItem;


/* --- Cross-file prototypes (split from cypher.c) --- */
/* Portable UTC helper (cypher_frontend.c) */
void gv_gmtime_utc(const time_t *t, struct tm *out);
/* Lexer (cypher_frontend.c) */
int push(Lex *lx, Tk t, const char *s, size_t n);
void lex_free(Lex *lx);
int tokenize(Lex *lx, const char *q);
Tok *pk(Lex *lx);
Tok *adv(Lex *lx);
int kw(const Tok *t, const char *k);
int eat(Lex *lx, Tk t, const char *w);
/* Parser (cypher_frontend.c) */
void opd_clear(Opd *o);
void setitem_free(SetItem *s);
void expr_free(Expr *e);
int parse_pattern(Lex *lx, Pattern *p);
void pattern_clear(Pattern *p);
Opd *parse_add(Lex *lx);
int parse_operand(Lex *lx, Opd *o);
Expr *parse_or(Lex *lx);
int agg_has_pct(Agg a);
/* Engine parameter lookup (cypher.c) */
const char *cy_param_lookup(const char *name);

#endif /* GV_FEATURES_CYPHER_INTERNAL_H */
