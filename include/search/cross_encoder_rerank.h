#ifndef GIGAVECTOR_GV_CROSS_ENCODER_RERANK_H
#define GIGAVECTOR_GV_CROSS_ENCODER_RERANK_H

#include <stddef.h>
#include <stdint.h>
#include "search/distance.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_CrossEncoder GV_CrossEncoder;

/*
 * Create a cross-encoder reranker from an ONNX model file.
 * The model should take (query_text, document_text) pairs and output a relevance score.
 *
 * @param model_path   Path to ONNX model file (NULL for fallback BM25-like scoring)
 * @param max_seq_len  Maximum sequence length for tokenizer (default 512)
 * @return Cross-encoder handle, or NULL on error
 */
GV_CrossEncoder *cross_encoder_create(const char *model_path, size_t max_seq_len);

/*
 * Destroy a cross-encoder reranker and free all resources.
 *
 * @param ce Cross-encoder handle (safe to call with NULL).
 */
void cross_encoder_destroy(GV_CrossEncoder *ce);

/*
 * Callback to retrieve the text associated with a search result.
 *
 * @param result_index  Index into the results array.
 * @param user_data     Opaque pointer supplied to cross_encoder_rerank().
 * @return Pointer to a NUL-terminated UTF-8 string; the string must remain
 *         valid for the duration of the rerank call.
 */
typedef const char *(*GV_CrossEncoderTextGetter)(size_t result_index, void *user_data);

/*
 * Rerank search results using the cross-encoder.
 *
 * Scores each (query, document_text) pair, then sorts results in-place
 * by descending relevance score.
 *
 * @param ce            Cross-encoder handle
 * @param query         Query text (the cross-encoder compares this to each result's text)
 * @param results       Input/output array of search results (reranked in-place by score)
 * @param count         Number of results
 * @param text_getter   Callback to get the text for a result (by index into results array)
 * @param user_data     Opaque pointer for text_getter
 * @return 0 on success, -1 on error
 */
int cross_encoder_rerank(GV_CrossEncoder *ce, const char *query,
                         GV_SearchResult *results, size_t count,
                         GV_CrossEncoderTextGetter text_getter, void *user_data);

/*
 * Batch rerank with pre-extracted texts (avoids callback overhead).
 *
 * @param ce              Cross-encoder handle
 * @param query           Query text
 * @param document_texts  Array of document text strings (length == doc_count)
 * @param doc_count       Number of documents
 * @param out_scores      Output array of relevance scores (length >= doc_count)
 * @return 0 on success, -1 on error
 */
int cross_encoder_rerank_batch(GV_CrossEncoder *ce, const char *query,
                               const char **document_texts, size_t doc_count,
                               float *out_scores);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_CROSS_ENCODER_RERANK_H */
