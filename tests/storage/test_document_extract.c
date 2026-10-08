/* Plain-text extraction from Markdown / HTML / PDF (uncompressed) documents. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage/document_extract.h"
#include "core/memory.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static int contains(const char *hay, const char *needle) {
    return hay && strstr(hay, needle) != NULL;
}

static void test_format_from_extension(void) {
    ASSERT(gv_doc_format_from_extension("a.md") == GV_DOC_FORMAT_MARKDOWN, "md -> markdown");
    ASSERT(gv_doc_format_from_extension("a.MARKDOWN") == GV_DOC_FORMAT_MARKDOWN, "MARKDOWN -> markdown");
    ASSERT(gv_doc_format_from_extension("page.html") == GV_DOC_FORMAT_HTML, "html -> html");
    ASSERT(gv_doc_format_from_extension("INDEX.HTML") == GV_DOC_FORMAT_HTML, "HTML (upper) -> html");
    ASSERT(gv_doc_format_from_extension("p.htm") == GV_DOC_FORMAT_HTML, "htm -> html");
    ASSERT(gv_doc_format_from_extension("doc.pdf") == GV_DOC_FORMAT_PDF, "pdf -> pdf");
    ASSERT(gv_doc_format_from_extension("notes.txt") == GV_DOC_FORMAT_TEXT, "txt -> text");
    ASSERT(gv_doc_format_from_extension(NULL) == GV_DOC_FORMAT_TEXT, "NULL -> text");
}

static void test_plain_passthrough(void) {
    const char *src = "hello world";
    char *out = gv_document_extract_text(GV_DOC_FORMAT_TEXT, src, strlen(src));
    ASSERT(out && strcmp(out, "hello world") == 0, "plain text passes through");
    gv_free(out);
}

static void test_html(void) {
    const char *html =
        "<!DOCTYPE html><html><head><style>.x{color:red}</style>"
        "<script>var a = 1 < 2;</script></head>"
        "<body><h1>Title</h1><p>Hello&nbsp;&amp; welcome to <b>GigaVector</b>.</p>"
        "<p>Price &lt; 100 &#38; rising &#x41;.</p></body></html>";
    char *out = gv_document_extract_text(GV_DOC_FORMAT_HTML, html, strlen(html));
    ASSERT(out != NULL, "html extraction returns text");
    ASSERT(contains(out, "Title"), "heading text kept");
    ASSERT(contains(out, "Hello & welcome"), "entities decoded (&nbsp; &amp;)");
    ASSERT(contains(out, "GigaVector"), "bold inline text kept");
    ASSERT(contains(out, "Price < 100 & rising A"), "lt/numeric entities decoded");
    ASSERT(!contains(out, "color:red"), "style content removed");
    ASSERT(!contains(out, "var a"), "script content removed");
    /* note: a literal '<' legitimately appears via the decoded &lt; entity */
    ASSERT(!contains(out, "<b>") && !contains(out, "</") && !contains(out, "<p>"),
           "no residual markup tags");
    gv_free(out);
}

static void test_markdown(void) {
    const char *md =
        "# Heading One\n\n"
        "Some **bold** and _italic_ and `code` text.\n\n"
        "- item one\n- item two\n\n"
        "See [the site](https://example.com) for more.\n\n"
        "```\nraw_code_block(kept);\n```\n"
        "> a quote\n";
    char *out = gv_document_extract_text(GV_DOC_FORMAT_MARKDOWN, md, strlen(md));
    ASSERT(out != NULL, "markdown extraction returns text");
    ASSERT(contains(out, "Heading One"), "heading marker stripped, text kept");
    ASSERT(contains(out, "bold") && !contains(out, "**"), "emphasis markers removed");
    ASSERT(contains(out, "the site") && !contains(out, "example.com"), "link text kept, url dropped");
    ASSERT(contains(out, "item one") && !contains(out, "- item"), "list markers stripped");
    ASSERT(contains(out, "raw_code_block(kept);"), "fenced code content kept");
    ASSERT(contains(out, "a quote") && !contains(out, "> a quote"), "blockquote marker stripped");
    gv_free(out);
}

static void test_pdf_uncompressed(void) {
    /* Minimal PDF with an uncompressed content stream using Tj and TJ. */
    const char *pdf =
        "%PDF-1.4\n"
        "4 0 obj\n<< /Length 60 >>\nstream\n"
        "BT /F1 12 Tf 72 720 Td (Hello PDF World) Tj\n"
        "[(Array ) -250 (text) ] TJ ET\n"
        "endstream\nendobj\n"
        "%%EOF\n";
    char *out = gv_document_extract_text(GV_DOC_FORMAT_PDF, pdf, strlen(pdf));
    ASSERT(out != NULL, "pdf extraction returns text");
    ASSERT(contains(out, "Hello PDF World"), "Tj literal extracted");
    ASSERT(contains(out, "Array") && contains(out, "text"), "TJ array parts extracted");
    gv_free(out);
}

static void test_auto_sniff(void) {
    const char *pdf = "%PDF-1.7\nstuff";
    char *pdf_out = gv_document_extract_text(GV_DOC_FORMAT_AUTO, pdf, strlen(pdf));
    ASSERT(pdf_out != NULL, "auto pdf ok");
    gv_free(pdf_out);

    const char *html = "  <html><body>Hi there</body></html>";
    char *out = gv_document_extract_text(GV_DOC_FORMAT_AUTO, html, strlen(html));
    ASSERT(contains(out, "Hi there") && !contains(out, "<body>"), "auto-detected html stripped");
    gv_free(out);

    ASSERT(gv_document_extract_text(GV_DOC_FORMAT_TEXT, NULL, 0) == NULL, "NULL data -> NULL");
}

int main(void) {
    test_format_from_extension();
    test_plain_passthrough();
    test_html();
    test_markdown();
    test_pdf_uncompressed();
    test_auto_sniff();
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL DOCUMENT-EXTRACT TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
