#ifndef GIGAVECTOR_GV_ONNX_H
#define GIGAVECTOR_GV_ONNX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_onnx.h
 * @brief ONNX model serving for re-ranking and embedding.
 *
 * Load and run ONNX models inside the GigaVector search/indexing pipeline.
 * Supports cross-encoder re-ranking and bi-encoder embedding generation.
 * When compiled with GV_HAVE_ONNX the full ONNX Runtime C API is used;
 * otherwise every function degrades to a safe stub.
 */

/* Opaque Types */

/**
 * @brief Opaque ONNX model handle.
 *
 * Wraps an ONNX Runtime session together with its environment, thread-pool
 * settings, and an optional vocabulary for text tokenization.
 */
typedef struct GV_ONNXModel GV_ONNXModel;

/* Configuration */

/**
 * @brief ONNX model configuration.
 */
typedef struct {
    const char *model_path;         /**< Path to the .onnx model file. */
    int         num_threads;        /**< Intra-op thread count (default: 4). */
    int         use_gpu;            /**< Use CUDA execution provider (default: 0). */
    size_t      max_batch_size;     /**< Maximum inference batch size (default: 32). */
    int         optimization_level; /**< Graph optimization: 0=none, 1=basic,
                                         2=extended, 3=all (default: 2). */
} GV_ONNXConfig;

/* Tensor */

/**
 * @brief Dense tensor exchanged with ONNX Runtime.
 *
 * All elements are stored contiguously in row-major order.
 */
typedef struct {
    float  *data;           /**< Flat data buffer. */
    size_t *shape;          /**< Dimension sizes array (length = ndim). */
    size_t  ndim;           /**< Number of dimensions. */
    size_t  total_elements; /**< Product of all shape dimensions. */
} GV_ONNXTensor;

/* Runtime Query */

/**
 * @brief Check whether the ONNX Runtime is linked.
 *
 * @return 1 if compiled with GV_HAVE_ONNX, 0 otherwise.
 */
int gv_onnx_available(void);

/* Model Lifecycle */

/**
 * @brief Load an ONNX model.
 *
 * Creates an OrtEnv and OrtSession (when GV_HAVE_ONNX is defined) or prints
 * a warning and returns NULL in the stub path.
 *
 * @param config Model configuration (must not be NULL).
 * @return Model handle, or NULL on error.
 */
GV_ONNXModel *gv_onnx_load(const GV_ONNXConfig *config);

/**
 * @brief Destroy an ONNX model and release all resources.
 *
 * Safe to call with NULL.
 *
 * @param model Model handle.
 */
void gv_onnx_destroy(GV_ONNXModel *model);

/* Inference */

/**
 * @brief Run raw tensor inference.
 *
 * Maps GV_ONNXTensor inputs to OrtValues, executes the session, and writes
 * results into the pre-allocated output tensors.
 *
 * @param model        Model handle.
 * @param inputs       Array of input tensors.
 * @param input_count  Number of input tensors.
 * @param outputs      Array of output tensors (pre-allocated).
 * @param output_count Number of output tensors.
 * @return 0 on success, -1 on error.
 */
int gv_onnx_infer(GV_ONNXModel *model, const GV_ONNXTensor *inputs,
                   size_t input_count, GV_ONNXTensor *outputs,
                   size_t output_count);

/* High-Level Pipelines */

/**
 * @brief Cross-encoder re-ranking.
 *
 * Tokenizes the query and each document, runs the cross-encoder model, and
 * writes a relevance score per document into @p scores.
 *
 * @param model      Model handle (must be a cross-encoder).
 * @param query_text Query string.
 * @param doc_texts  Array of document strings.
 * @param doc_count  Number of documents.
 * @param scores     Output array of length @p doc_count.
 * @return 0 on success, -1 on error.
 */
int gv_onnx_rerank(GV_ONNXModel *model, const char *query_text,
                    const char **doc_texts, size_t doc_count, float *scores);

/**
 * @brief Bi-encoder embedding generation.
 *
 * Tokenizes each text, runs the encoder model, and writes the resulting
 * embeddings contiguously into @p embeddings (row-major, text_count x dimension).
 *
 * @param model      Model handle (must be a bi-encoder).
 * @param texts      Array of text strings.
 * @param text_count Number of texts.
 * @param embeddings Output buffer (text_count * dimension floats).
 * @param dimension  Expected embedding dimension.
 * @return 0 on success, -1 on error.
 */
int gv_onnx_embed(GV_ONNXModel *model, const char **texts,
                   size_t text_count, float *embeddings, size_t dimension);

/* Tensor Helpers */

/**
 * @brief Create a tensor with the given shape.
 *
 * Allocates the data buffer (zero-filled) and the shape array.
 *
 * @param shape Dimension sizes.
 * @param ndim  Number of dimensions.
 * @return Initialized tensor (data is NULL on allocation failure).
 */
GV_ONNXTensor gv_onnx_tensor_create(const size_t *shape, size_t ndim);

/**
 * @brief Free a tensor's internal buffers.
 *
 * Safe to call with a tensor whose data is already NULL.
 *
 * @param tensor Tensor to free.
 */
void gv_onnx_tensor_destroy(GV_ONNXTensor *tensor);

/* Model Introspection */

/**
 * @brief Query input node names and count.
 *
 * @param model       Model handle.
 * @param input_count Output: number of inputs.
 * @param input_names Output: allocated array of name strings (caller frees
 *                    each string and the array itself with free()).
 * @return 0 on success, -1 on error.
 */
int gv_onnx_get_input_info(const GV_ONNXModel *model, size_t *input_count,
                            char ***input_names);

/**
 * @brief Query output node names and count.
 *
 * @param model        Model handle.
 * @param output_count Output: number of outputs.
 * @param output_names Output: allocated array of name strings (caller frees
 *                     each string and the array itself with free()).
 * @return 0 on success, -1 on error.
 */
int gv_onnx_get_output_info(const GV_ONNXModel *model, size_t *output_count,
                             char ***output_names);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_ONNX_H */
