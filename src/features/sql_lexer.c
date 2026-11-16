/**
 * @file sql_lexer.c
 * @brief SQL tokenizer and token-buffer cursor (split from sql.c).
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "features/sql_internal.h"
#include "core/memory.h"
#include "core/utils.h"

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

int sql_kw_match(const char *s, size_t len, const char *kw)
{
    size_t kwlen = strlen(kw);
    if (len != kwlen) return 0;
    for (size_t i = 0; i < len; i++) {
        if ((char)toupper((unsigned char)s[i]) != (char)toupper((unsigned char)kw[i]))
            return 0;
    }
    return 1;
}

char *sql_strndup(const char *s, size_t n)
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
        if (sql_kw_match(lx->input + start, len, "GROUP"))   { tok.type = GV_SQL_TOK_GROUP;   return tok; }
        if (sql_kw_match(lx->input + start, len, "HAVING"))  { tok.type = GV_SQL_TOK_HAVING;  return tok; }
        if (sql_kw_match(lx->input + start, len, "DISTINCT")){ tok.type = GV_SQL_TOK_DISTINCT;return tok; }
        if (sql_kw_match(lx->input + start, len, "AS"))      { tok.type = GV_SQL_TOK_AS;      return tok; }

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

int sql_tokenize(GV_SQLTokenBuf *buf, const char *query)
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
            snprintf(buf->error, GV_SQL_ERROR_SIZE, "Tokenization error: %.1000s", lx.error);
            sql_token_free(&tok);
            return -1;
        }
        buf->tokens[buf->count++] = tok;
        if (tok.type == GV_SQL_TOK_EOF)
            break;
    }
    return 0;
}

void sql_tokenbuf_free(GV_SQLTokenBuf *buf)
{
    if (!buf || !buf->tokens) return;
    for (size_t i = 0; i < buf->count; i++)
        sql_token_free(&buf->tokens[i]);
    gv_free(buf->tokens);
    buf->tokens = NULL;
    buf->count = 0;
}

GV_SQLToken *sql_peek(GV_SQLTokenBuf *buf)
{
    if (buf->pos < buf->count)
        return &buf->tokens[buf->pos];
    return NULL;
}

GV_SQLToken *sql_advance(GV_SQLTokenBuf *buf)
{
    if (buf->pos < buf->count)
        return &buf->tokens[buf->pos++];
    return NULL;
}

int sql_consume_trailing_semicolons(GV_SQLTokenBuf *buf)
{
    GV_SQLToken *tok = sql_peek(buf);
    while (tok && tok->type == GV_SQL_TOK_SEMICOLON) {
        sql_advance(buf);
        tok = sql_peek(buf);
    }
    return (tok && tok->type == GV_SQL_TOK_EOF) ? 0 : -1;
}

int sql_expect(GV_SQLTokenBuf *buf, GV_SQLTokenType type)
{
    GV_SQLToken *tok = sql_peek(buf);
    if (!tok || tok->type != type) return 0;
    buf->pos++;
    return 1;
}

