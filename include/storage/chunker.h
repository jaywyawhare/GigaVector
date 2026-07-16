/**
 * @file chunker.h
 * @brief Document chunker: splits text into overlapping chunks, each with a
 *        deterministic shared identity ("{doc_id}:{index:04d}") — the primary
 *        key that binds the embedding, graph, and memory layers.
 */
#ifndef GIGAVECTOR_GV_CHUNKER_H
#define GIGAVECTOR_GV_CHUNKER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char    chunk_id[64];   /**< "{doc_id}:{index:04d}" — the cross-layer PK. */
    char    doc_id[32];     /**< Document id (sha/uuid prefix). */
    size_t  chunk_index;    /**< Position within the document. */
    char   *text;           /**< Heap-allocated chunk text (NUL-terminated). */
    size_t  char_start;     /**< Byte offset of the chunk start in the source. */
    size_t  char_end;       /**< Byte offset one past the chunk end. */
} GV_Chunk;

/**
 * @brief Split a document into overlapping word-based chunks.
 *
 * @param text           Source document (NUL-terminated).
 * @param chunk_tokens   Target words per chunk (0 -> default 512).
 * @param overlap_tokens Overlap in words between consecutive chunks (0 -> 128).
 * @param doc_id         Document id; if NULL, a uuid-derived id is generated.
 * @param out_chunks     Receives a heap array of GV_Chunk (caller frees via gv_chunks_free).
 * @param out_count      Receives the number of chunks.
 * @return 0 on success, -1 on error.
 */
int gv_chunk_document(const char *text, size_t chunk_tokens, size_t overlap_tokens,
                      const char *doc_id, GV_Chunk **out_chunks, size_t *out_count);

/** @brief Free a chunk array returned by gv_chunk_document. */
void gv_chunks_free(GV_Chunk *chunks, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CHUNKER_H */
