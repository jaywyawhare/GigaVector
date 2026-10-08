/**
 * @file document_extract.c
 * @brief Self-contained plain-text extraction for Markdown, HTML and PDF
 *        (uncompressed content streams), plus a plain-text passthrough.
 */
#include "storage/document_extract.h"
#include "core/memory.h"

#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>

typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
    int    oom;
} TextBuf;

static void tb_init(TextBuf *t) { t->buf = NULL; t->len = 0; t->cap = 0; t->oom = 0; }

static int tb_reserve(TextBuf *t, size_t extra) {
    if (t->oom) return -1;
    if (t->len + extra + 1 <= t->cap) return 0;
    size_t ncap = t->cap ? t->cap * 2 : 256;
    while (ncap < t->len + extra + 1) ncap *= 2;
    char *nb = (char *)gv_realloc(t->buf, ncap);
    if (!nb) { t->oom = 1; return -1; }
    t->buf = nb;
    t->cap = ncap;
    return 0;
}

static void tb_putc(TextBuf *t, char c) {
    if (tb_reserve(t, 1) != 0) return;
    t->buf[t->len++] = c;
}

static void tb_put(TextBuf *t, const char *s, size_t n) {
    if (tb_reserve(t, n) != 0) return;
    memcpy(t->buf + t->len, s, n);
    t->len += n;
}

/* Append a space unless the buffer already ends with whitespace (collapses
   runs of inter-token whitespace that formatting would otherwise introduce). */
static void tb_sep(TextBuf *t) {
    if (t->len == 0) return;
    char last = t->buf[t->len - 1];
    if (last == ' ' || last == '\n' || last == '\t') return;
    tb_putc(t, ' ');
}

static char *tb_finish(TextBuf *t) {
    if (t->oom) { gv_free(t->buf); return NULL; }
    if (tb_reserve(t, 0) != 0) { gv_free(t->buf); return NULL; }
    if (!t->buf) { /* empty document -> empty string */
        t->buf = (char *)gv_alloc(1);
        if (!t->buf) return NULL;
    }
    t->buf[t->len] = '\0';
    return t->buf;
}


static int ci_ends_with(const char *s, const char *suffix) {
    size_t ls = strlen(s), lx = strlen(suffix);
    if (lx > ls) return 0;
    const char *tail = s + (ls - lx);
    for (size_t i = 0; i < lx; i++) {
        if (tolower((unsigned char)tail[i]) != tolower((unsigned char)suffix[i])) return 0;
    }
    return 1;
}

GV_DocFormat gv_doc_format_from_extension(const char *path) {
    if (!path) return GV_DOC_FORMAT_TEXT;
    if (ci_ends_with(path, ".md") || ci_ends_with(path, ".markdown")) return GV_DOC_FORMAT_MARKDOWN;
    if (ci_ends_with(path, ".html") || ci_ends_with(path, ".htm")) return GV_DOC_FORMAT_HTML;
    if (ci_ends_with(path, ".pdf")) return GV_DOC_FORMAT_PDF;
    return GV_DOC_FORMAT_TEXT;
}

static GV_DocFormat sniff_format(const char *data, size_t len) {
    if (len >= 5 && memcmp(data, "%PDF-", 5) == 0) return GV_DOC_FORMAT_PDF;
    /* A leading '<' that opens an html/doctype tag -> HTML. */
    for (size_t i = 0; i < len && i < 256; i++) {
        unsigned char c = (unsigned char)data[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        if (c == '<') {
            if ((len - i) >= 2) {
                char n = (char)tolower((unsigned char)data[i + 1]);
                if (n == '!' || n == 'h' || n == 'b' || n == 'p' || n == 'd') return GV_DOC_FORMAT_HTML;
            }
        }
        break;
    }
    return GV_DOC_FORMAT_TEXT;
}


static char *extract_text(const char *data, size_t len) {
    char *out = (char *)gv_alloc(len + 1);
    if (!out) return NULL;
    memcpy(out, data, len);
    out[len] = '\0';
    return out;
}


/* Decode a single entity starting at s[*i] == '&'. Writes decoded bytes to t,
   advances *i past the entity (including ';'). Falls back to emitting '&'. */
static void decode_entity(TextBuf *t, const char *s, size_t len, size_t *i) {
    size_t start = *i + 1;
    size_t j = start;
    while (j < len && s[j] != ';' && (j - start) < 10 &&
           (isalnum((unsigned char)s[j]) || s[j] == '#')) j++;
    if (j >= len || s[j] != ';') { tb_putc(t, '&'); (*i)++; return; }

    size_t nlen = j - start;
    char name[12];
    memcpy(name, s + start, nlen);
    name[nlen] = '\0';

    if (name[0] == '#') {
        long code = (name[1] == 'x' || name[1] == 'X')
            ? strtol(name + 2, NULL, 16)
            : strtol(name + 1, NULL, 10);
        if (code <= 0) { tb_putc(t, '&'); (*i)++; return; }
        if (code < 0x80) {
            tb_putc(t, (char)code);
        } else if (code < 0x800) {
            tb_putc(t, (char)(0xC0 | (code >> 6)));
            tb_putc(t, (char)(0x80 | (code & 0x3F)));
        } else {
            tb_putc(t, (char)(0xE0 | (code >> 12)));
            tb_putc(t, (char)(0x80 | ((code >> 6) & 0x3F)));
            tb_putc(t, (char)(0x80 | (code & 0x3F)));
        }
    } else if (strcmp(name, "amp") == 0)  tb_putc(t, '&');
    else if (strcmp(name, "lt") == 0)     tb_putc(t, '<');
    else if (strcmp(name, "gt") == 0)     tb_putc(t, '>');
    else if (strcmp(name, "quot") == 0)   tb_putc(t, '"');
    else if (strcmp(name, "apos") == 0)   tb_putc(t, '\'');
    else if (strcmp(name, "nbsp") == 0)   tb_putc(t, ' ');
    else { tb_put(t, "&", 1); tb_put(t, name, nlen); tb_putc(t, ';'); }

    *i = j + 1;
}

/* Case-insensitive match of s[i..] against a lowercase literal. */
static int ci_match(const char *s, size_t len, size_t i, const char *lit) {
    size_t n = strlen(lit);
    if (i + n > len) return 0;
    for (size_t k = 0; k < n; k++)
        if (tolower((unsigned char)s[i + k]) != lit[k]) return 0;
    return 1;
}

static char *extract_html(const char *s, size_t len) {
    TextBuf t; tb_init(&t);
    size_t i = 0;
    while (i < len) {
        if (s[i] == '<') {
            if (ci_match(s, len, i, "<!--")) {
                i += 4;
                while (i < len && !(i + 2 < len && s[i] == '-' && s[i+1] == '-' && s[i+2] == '>')) i++;
                i = (i + 3 <= len) ? i + 3 : len;
                continue;
            }
            /* script/style: skip through the matching close tag. */
            const char *skip_close = NULL;
            if (ci_match(s, len, i, "<script")) skip_close = "</script";
            else if (ci_match(s, len, i, "<style")) skip_close = "</style";
            if (skip_close) {
                i++;
                while (i < len && !ci_match(s, len, i, skip_close)) i++;
                while (i < len && s[i] != '>') i++;
                if (i < len) i++;
                tb_sep(&t);
                continue;
            }
            /* Block-level tags act as separators (word boundaries / newlines). */
            int is_break = ci_match(s, len, i, "<p") || ci_match(s, len, i, "</p") ||
                           ci_match(s, len, i, "<br") || ci_match(s, len, i, "<div") ||
                           ci_match(s, len, i, "</div") || ci_match(s, len, i, "<li") ||
                           ci_match(s, len, i, "<tr") || ci_match(s, len, i, "<h");
            while (i < len && s[i] != '>') i++;
            if (i < len) i++;
            if (is_break) tb_sep(&t); else tb_sep(&t);
            continue;
        }
        if (s[i] == '&') { decode_entity(&t, s, len, &i); continue; }
        if (s[i] == '\r') { i++; continue; }
        if (s[i] == '\n' || s[i] == '\t') { tb_sep(&t); i++; continue; }
        if (s[i] == ' ') { tb_sep(&t); i++; continue; }
        tb_putc(&t, s[i]);
        i++;
    }
    return tb_finish(&t);
}


static char *extract_markdown(const char *s, size_t len) {
    TextBuf t; tb_init(&t);
    int in_fence = 0;
    size_t i = 0;
    while (i < len) {
        size_t ls = i;
        size_t le = ls;
        while (le < len && s[le] != '\n') le++;
        size_t line_len = le - ls;

        /* fenced code block toggles on ``` or ~~~ */
        size_t p = ls;
        while (p < le && (s[p] == ' ' || s[p] == '\t')) p++;
        if (line_len - (p - ls) >= 3 &&
            ((s[p] == '`' && s[p+1] == '`' && s[p+2] == '`') ||
             (s[p] == '~' && s[p+1] == '~' && s[p+2] == '~'))) {
            in_fence = !in_fence;
            i = (le < len) ? le + 1 : le;
            continue;
        }
        if (in_fence) {
            tb_put(&t, s + ls, line_len);
            tb_putc(&t, '\n');
            i = (le < len) ? le + 1 : le;
            continue;
        }

        /* strip leading blockquote '>' , list markers and ATX heading '#' */
        while (p < le && (s[p] == '>' || s[p] == ' ' || s[p] == '\t')) p++;
        while (p < le && s[p] == '#') p++;
        if (p < le && (s[p] == ' ')) p++;
        if (p + 1 < le && (s[p] == '-' || s[p] == '*' || s[p] == '+') && s[p+1] == ' ') p += 2;
        else {
            /* ordered list "1. " */
            size_t q = p;
            while (q < le && isdigit((unsigned char)s[q])) q++;
            if (q > p && q + 1 < le && s[q] == '.' && s[q+1] == ' ') p = q + 2;
        }

        /* inline pass: drop emphasis/backticks, unwrap links/images */
        size_t k = p;
        while (k < le) {
            char c = s[k];
            if (c == '\\' && k + 1 < le) { tb_putc(&t, s[k+1]); k += 2; continue; }
            if (c == '*' || c == '_' || c == '`') { k++; continue; }
            if (c == '!' && k + 1 < le && s[k+1] == '[') { k++; continue; } /* image -> keep alt */
            if (c == '[') {
                /* [text](url) or [text][ref] -> text */
                size_t tstart = k + 1;
                size_t tend = tstart;
                while (tend < le && s[tend] != ']') tend++;
                if (tend < le) {
                    tb_put(&t, s + tstart, tend - tstart);
                    k = tend + 1;
                    if (k < le && s[k] == '(') { while (k < le && s[k] != ')') k++; if (k < le) k++; }
                    else if (k < le && s[k] == '[') { while (k < le && s[k] != ']') k++; if (k < le) k++; }
                    continue;
                }
            }
            tb_putc(&t, c);
            k++;
        }
        tb_putc(&t, '\n');
        i = (le < len) ? le + 1 : le;
    }
    return tb_finish(&t);
}


/* Emit a PDF literal string (already inside the parentheses) with escape
   handling into t. Returns index just past the closing ')'. */
static size_t pdf_emit_literal(TextBuf *t, const char *s, size_t len, size_t i) {
    int depth = 1;
    while (i < len && depth > 0) {
        char c = s[i];
        if (c == '\\' && i + 1 < len) {
            char e = s[i + 1];
            switch (e) {
                case 'n': tb_putc(t, '\n'); break;
                case 'r': tb_putc(t, ' '); break;
                case 't': tb_putc(t, '\t'); break;
                case 'b': case 'f': tb_putc(t, ' '); break;
                case '(': tb_putc(t, '('); break;
                case ')': tb_putc(t, ')'); break;
                case '\\': tb_putc(t, '\\'); break;
                default:
                    if (e >= '0' && e <= '7') {
                        int oct = 0, n = 0;
                        size_t j = i + 1;
                        while (j < len && n < 3 && s[j] >= '0' && s[j] <= '7') { oct = oct * 8 + (s[j]-'0'); j++; n++; }
                        tb_putc(t, (char)oct);
                        i = j;
                        continue;
                    }
                    tb_putc(t, e);
            }
            i += 2;
            continue;
        }
        if (c == '(') { depth++; tb_putc(t, c); i++; continue; }
        if (c == ')') { depth--; if (depth > 0) tb_putc(t, c); i++; continue; }
        tb_putc(t, c);
        i++;
    }
    return i;
}

static char *extract_pdf(const char *s, size_t len) {
    TextBuf t; tb_init(&t);
    size_t i = 0;
    while (i < len) {
        if (!ci_match(s, len, i, "stream")) { i++; continue; }
        i += 6;
        if (i < len && s[i] == '\r') i++;
        if (i < len && s[i] == '\n') i++;
        size_t send = i;
        while (send < len && !ci_match(s, len, send, "endstream")) send++;

        /* Only parse streams that look like text content (contain text ops and
           are not FlateDecode binary). Heuristic: must contain "Tj" or "TJ". */
        size_t seg_len = send - i;
        int has_text_op = 0;
        for (size_t p = i; p + 1 < send; p++) {
            if (s[p] == 'T' && (s[p+1] == 'j' || s[p+1] == 'J')) { has_text_op = 1; break; }
        }
        if (has_text_op) {
            size_t p = i;
            while (p < send) {
                if (s[p] == '(') {
                    p = pdf_emit_literal(&t, s, send, p + 1);
                    continue;
                }
                /* End of a text-showing array/operator -> separator. */
                if (s[p] == ']' || (s[p] == 'T' && p + 1 < send && (s[p+1] == 'j' || s[p+1] == 'J'))) {
                    tb_sep(&t);
                }
                p++;
            }
            tb_putc(&t, '\n');
        }
        (void)seg_len;
        i = (send < len) ? send + 9 : len; /* skip past "endstream" */
    }
    return tb_finish(&t);
}

char *gv_document_extract_text(GV_DocFormat fmt, const void *data, size_t len) {
    if (!data) return NULL;
    const char *s = (const char *)data;

    if (fmt == GV_DOC_FORMAT_AUTO) fmt = sniff_format(s, len);

    switch (fmt) {
        case GV_DOC_FORMAT_MARKDOWN: return extract_markdown(s, len);
        case GV_DOC_FORMAT_HTML:     return extract_html(s, len);
        case GV_DOC_FORMAT_PDF:      return extract_pdf(s, len);
        case GV_DOC_FORMAT_TEXT:
        default:                     return extract_text(s, len);
    }
}
