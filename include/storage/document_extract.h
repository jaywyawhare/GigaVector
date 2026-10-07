/**
 * @file document_extract.h
 * @brief Plain-text extraction from common document formats, so the ingest
 *        pipeline is no longer limited to pre-extracted text.
 *
 * Supported formats are self-contained (no external parser/decompression
 * dependencies): plain text, Markdown, HTML, and PDF text operators from
 * *uncompressed* content streams. Formats that require a decompression
 * library (DOCX/ZIP, FlateDecode-only PDFs) are intentionally not handled
 * here; see gv_document_extract_text() for the exact contract.
 */
#ifndef GIGAVECTOR_GV_DOCUMENT_EXTRACT_H
#define GIGAVECTOR_GV_DOCUMENT_EXTRACT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_DOC_FORMAT_AUTO = 0, /**< Sniff from extension/content. */
    GV_DOC_FORMAT_TEXT,     /**< Plain UTF-8 text (passthrough). */
    GV_DOC_FORMAT_MARKDOWN, /**< Markdown; markup is stripped to prose. */
    GV_DOC_FORMAT_HTML,     /**< HTML; tags/scripts/styles removed, entities decoded. */
    GV_DOC_FORMAT_PDF       /**< PDF; text from uncompressed content streams. */
} GV_DocFormat;

/**
 * @brief Map a filename/path extension to a document format.
 * @return The matching format, or GV_DOC_FORMAT_TEXT when unrecognised.
 */
GV_DocFormat gv_doc_format_from_extension(const char *path);

/**
 * @brief Extract plain text from an in-memory document.
 *
 * @param fmt  Source format. GV_DOC_FORMAT_AUTO sniffs from @p data (falls
 *             back to plain text).
 * @param data Document bytes (need not be NUL-terminated).
 * @param len  Length of @p data in bytes.
 * @return A newly allocated NUL-terminated string (free with gv_free), or
 *         NULL on allocation failure or when @p data is NULL. For a PDF with
 *         only compressed (FlateDecode) streams the result may be empty — the
 *         extractor reads uncompressed text operators only, never guessing.
 */
char *gv_document_extract_text(GV_DocFormat fmt, const void *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_DOCUMENT_EXTRACT_H */
