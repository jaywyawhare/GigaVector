/**
 * @file gv_onnx.c
 * @brief ONNX model serving implementation.
 *
 * When GV_HAVE_ONNX is defined the full ONNX Runtime C API is used to load
 * models, run inference, and drive the cross-encoder / bi-encoder pipelines.
 * Without GV_HAVE_ONNX every public function degrades to a safe stub that
 * returns an error code or NULL.
 */

#include "gigavector/gv_onnx.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <ctype.h>

#ifdef GV_HAVE_ONNX
#include <onnxruntime_c_api.h>
#endif

/*  Constants  */

#define GV_ONNX_MAX_TOKEN_LEN   256
#define GV_ONNX_MAX_SEQ_LEN     512
#define GV_ONNX_VOCAB_BUCKETS   8192
#define GV_ONNX_PAD_TOKEN_ID    0
#define GV_ONNX_UNK_TOKEN_ID    1
#define GV_ONNX_CLS_TOKEN_ID    2
#define GV_ONNX_SEP_TOKEN_ID    3

/*  Internal Structures  */

/** Vocabulary hash-table entry for whitespace tokenizer. */
typedef struct GV_VocabEntry {
    char                  *token;
    int64_t                id;
    struct GV_VocabEntry  *next;
} GV_VocabEntry;

/** Simple vocabulary loaded from vocab.txt alongside the model. */
typedef struct {
    GV_VocabEntry **buckets;
    size_t          bucket_count;
    size_t          size;
} GV_Vocab;

struct GV_ONNXModel {
#ifdef GV_HAVE_ONNX
    const OrtApi       *api;
    OrtEnv             *env;
    OrtSession         *session;
    OrtSessionOptions  *session_opts;
    OrtMemoryInfo      *memory_info;

    size_t              input_count;
    char              **input_names;
    size_t              output_count;
    char              **output_names;
#endif

    GV_ONNXConfig       config;
    GV_Vocab           *vocab;
    pthread_mutex_t     mutex;
    char                last_error[512];
};

/*  Vocabulary Helpers (shared by both compile paths)  */

static size_t vocab_hash(const char *str, size_t bucket_count) {
    size_t h = 5381;
    int c;
    while ((c = (unsigned char)*str++)) {
        h = ((h << 5) + h) + c;
    }
    return h % bucket_count;
}

static GV_Vocab *vocab_create(void) {
    GV_Vocab *v = calloc(1, sizeof(GV_Vocab));
    if (!v) return NULL;
    v->bucket_count = GV_ONNX_VOCAB_BUCKETS;
    v->buckets = calloc(v->bucket_count, sizeof(GV_VocabEntry *));
    if (!v->buckets) {
        free(v);
        return NULL;
    }
    return v;
}

static void vocab_destroy(GV_Vocab *v) {
    if (!v) return;
    for (size_t i = 0; i < v->bucket_count; i++) {
        GV_VocabEntry *e = v->buckets[i];
        while (e) {
            GV_VocabEntry *next = e->next;
            free(e->token);
            free(e);
            e = next;
        }
    }
    free(v->buckets);
    free(v);
}

static int vocab_insert(GV_Vocab *v, const char *token, int64_t id) {
    size_t idx = vocab_hash(token, v->bucket_count);
    GV_VocabEntry *e = calloc(1, sizeof(GV_VocabEntry));
    if (!e) return -1;
    e->token = strdup(token);
    if (!e->token) { free(e); return -1; }
    e->id = id;
    e->next = v->buckets[idx];
    v->buckets[idx] = e;
    v->size++;
    return 0;
}

static int64_t vocab_lookup(const GV_Vocab *v, const char *token) {
    size_t idx = vocab_hash(token, v->bucket_count);
    for (const GV_VocabEntry *e = v->buckets[idx]; e; e = e->next) {
        if (strcmp(e->token, token) == 0) return e->id;
    }
    return GV_ONNX_UNK_TOKEN_ID;
}

/**
 * Load vocab.txt from the same directory as the model file.
 * Format: one token per line, ID equals the zero-based line number.
 */
static GV_Vocab *vocab_load(const char *model_path) {
    if (!model_path) return NULL;

    /* Build vocab path: replace the filename with vocab.txt */
    size_t path_len = strlen(model_path);
    char *vocab_path = malloc(path_len + 16);
    if (!vocab_path) return NULL;

    strcpy(vocab_path, model_path);
    char *slash = strrchr(vocab_path, '/');
    if (!slash) slash = strrchr(vocab_path, '\\');
    if (slash) {
        strcpy(slash + 1, "vocab.txt");
    } else {
        strcpy(vocab_path, "vocab.txt");
    }

    FILE *fp = fopen(vocab_path, "r");
    free(vocab_path);
    if (!fp) return NULL;

    GV_Vocab *v = vocab_create();
    if (!v) { fclose(fp); return NULL; }

    char line[GV_ONNX_MAX_TOKEN_LEN];
    int64_t id = 0;
    while (fgets(line, sizeof(line), fp)) {
        /* Strip trailing newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) { id++; continue; }
        vocab_insert(v, line, id);
        id++;
    }

    fclose(fp);
    return v;
}

/*  Whitespace Tokenizer  */

/**
 * Tokenize @p text into integer IDs using a whitespace split and vocabulary
 * lookup.  Writes at most @p max_len IDs into @p out_ids and returns the
 * actual token count.  The sequence is wrapped with [CLS] ... [SEP].
 */
static size_t tokenize_text(const GV_Vocab *vocab, const char *text,
                             int64_t *out_ids, size_t max_len) {
    if (!vocab || !text || !out_ids || max_len < 3) return 0;

    size_t pos = 0;
    out_ids[pos++] = GV_ONNX_CLS_TOKEN_ID;

    const char *p = text;
    while (*p && pos < max_len - 1) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        /* Collect token */
        char token[GV_ONNX_MAX_TOKEN_LEN];
        size_t tlen = 0;
        while (*p && !isspace((unsigned char)*p) &&
               tlen < GV_ONNX_MAX_TOKEN_LEN - 1) {
            token[tlen++] = (char)tolower((unsigned char)*p);
            p++;
        }
        token[tlen] = '\0';

        out_ids[pos++] = vocab_lookup(vocab, token);
    }

    out_ids[pos++] = GV_ONNX_SEP_TOKEN_ID;
    return pos;
}

/*  Tensor Helpers  */

GV_ONNXTensor gv_onnx_tensor_create(const size_t *shape, size_t ndim) {
    GV_ONNXTensor t;
    memset(&t, 0, sizeof(t));

    if (!shape || ndim == 0) return t;

    t.ndim = ndim;
    t.shape = malloc(ndim * sizeof(size_t));
    if (!t.shape) return t;
    memcpy(t.shape, shape, ndim * sizeof(size_t));

    t.total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        t.total_elements *= shape[i];
    }

    t.data = calloc(t.total_elements, sizeof(float));
    if (!t.data) {
        free(t.shape);
        memset(&t, 0, sizeof(t));
    }

    return t;
}

void gv_onnx_tensor_destroy(GV_ONNXTensor *tensor) {
    if (!tensor) return;
    free(tensor->data);
    free(tensor->shape);
    tensor->data = NULL;
    tensor->shape = NULL;
    tensor->ndim = 0;
    tensor->total_elements = 0;
}

/* ############################################################################
 * ===  GV_HAVE_ONNX  — full ONNX Runtime implementation  ====================
 * ############################################################################ */

#ifdef GV_HAVE_ONNX

/*  Internal Helpers (ONNX)  */

static void set_error(GV_ONNXModel *m, const char *msg) {
    if (!m) return;
    strncpy(m->last_error, msg, sizeof(m->last_error) - 1);
    m->last_error[sizeof(m->last_error) - 1] = '\0';
}

static int check_status(GV_ONNXModel *m, OrtStatus *status) {
    if (status == NULL) return 0;  /* success */
    const char *msg = m->api->GetErrorMessage(status);
    set_error(m, msg);
    m->api->ReleaseStatus(status);
    return -1;
}

/*  Runtime Query  */

int gv_onnx_available(void) {
    return 1;
}

/*  Model Lifecycle  */

GV_ONNXModel *gv_onnx_load(const GV_ONNXConfig *config) {
    if (!config || !config->model_path) {
        fprintf(stderr, "gv_onnx_load: NULL config or model_path\n");
        return NULL;
    }

    GV_ONNXModel *m = calloc(1, sizeof(GV_ONNXModel));
    if (!m) return NULL;

    m->config = *config;
    pthread_mutex_init(&m->mutex, NULL);

    /* Obtain the global API handle */
    m->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!m->api) {
        set_error(m, "Failed to obtain ORT API");
        goto fail;
    }

    /* Create environment */
    if (check_status(m, m->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                           "gv_onnx", &m->env))) {
        goto fail;
    }

    /* Session options */
    if (check_status(m, m->api->CreateSessionOptions(&m->session_opts))) {
        goto fail;
    }

    /* Thread pool */
    int threads = config->num_threads > 0 ? config->num_threads : 4;
    m->api->SetIntraOpNumThreads(m->session_opts, threads);
    m->api->SetInterOpNumThreads(m->session_opts, 1);

    /* Optimization level */
    GraphOptimizationLevel opt;
    switch (config->optimization_level) {
        case 0:  opt = ORT_DISABLE_ALL;               break;
        case 1:  opt = ORT_ENABLE_BASIC;               break;
        case 3:  opt = ORT_ENABLE_ALL;                 break;
        default: opt = ORT_ENABLE_EXTENDED;             break;
    }
    m->api->SetSessionGraphOptimizationLevel(m->session_opts, opt);

    /* GPU execution provider (optional) */
    if (config->use_gpu) {
        OrtCUDAProviderOptions cuda_opts;
        memset(&cuda_opts, 0, sizeof(cuda_opts));
        cuda_opts.device_id = 0;
        /* Ignore failure — will fall back to CPU */
        m->api->SessionOptionsAppendExecutionProvider_CUDA(m->session_opts,
                                                           &cuda_opts);
    }

    /* Create session */
    if (check_status(m, m->api->CreateSession(m->env, config->model_path,
                                               m->session_opts,
                                               &m->session))) {
        goto fail;
    }

    /* Memory info for tensor allocation */
    if (check_status(m, m->api->CreateCpuMemoryInfo(
                         OrtArenaAllocator, OrtMemTypeDefault,
                         &m->memory_info))) {
        goto fail;
    }

    /* Query input / output names */
    OrtAllocator *allocator = NULL;
    m->api->GetAllocatorWithDefaultOptions(&allocator);

    m->api->SessionGetInputCount(m->session, &m->input_count);
    m->input_names = calloc(m->input_count, sizeof(char *));
    for (size_t i = 0; i < m->input_count; i++) {
        char *name = NULL;
        m->api->SessionGetInputName(m->session, i, allocator, &name);
        m->input_names[i] = name ? strdup(name) : NULL;
        if (name) allocator->Free(allocator, name);
    }

    m->api->SessionGetOutputCount(m->session, &m->output_count);
    m->output_names = calloc(m->output_count, sizeof(char *));
    for (size_t i = 0; i < m->output_count; i++) {
        char *name = NULL;
        m->api->SessionGetOutputName(m->session, i, allocator, &name);
        m->output_names[i] = name ? strdup(name) : NULL;
        if (name) allocator->Free(allocator, name);
    }

    /* Try to load vocabulary */
    m->vocab = vocab_load(config->model_path);

    return m;

fail:
    gv_onnx_destroy(m);
    return NULL;
}

void gv_onnx_destroy(GV_ONNXModel *model) {
    if (!model) return;

    pthread_mutex_destroy(&model->mutex);

    if (model->api) {
        if (model->memory_info) model->api->ReleaseMemoryInfo(model->memory_info);
        if (model->session)     model->api->ReleaseSession(model->session);
        if (model->session_opts) model->api->ReleaseSessionOptions(model->session_opts);
        if (model->env)         model->api->ReleaseEnv(model->env);
    }

    if (model->input_names) {
        for (size_t i = 0; i < model->input_count; i++) free(model->input_names[i]);
        free(model->input_names);
    }
    if (model->output_names) {
        for (size_t i = 0; i < model->output_count; i++) free(model->output_names[i]);
        free(model->output_names);
    }

    vocab_destroy(model->vocab);
    free(model);
}

/*  Inference  */

int gv_onnx_infer(GV_ONNXModel *model, const GV_ONNXTensor *inputs,
                   size_t input_count, GV_ONNXTensor *outputs,
                   size_t output_count) {
    if (!model || !inputs || !outputs) return -1;
    if (input_count == 0 || output_count == 0) return -1;
    if (input_count > model->input_count || output_count > model->output_count) {
        set_error(model, "Input/output count mismatch");
        return -1;
    }

    int rc = -1;
    pthread_mutex_lock(&model->mutex);

    /* Build OrtValue inputs */
    OrtValue **ort_inputs = calloc(input_count, sizeof(OrtValue *));
    if (!ort_inputs) goto unlock;

    for (size_t i = 0; i < input_count; i++) {
        int64_t *shape64 = malloc(inputs[i].ndim * sizeof(int64_t));
        if (!shape64) goto cleanup;
        for (size_t d = 0; d < inputs[i].ndim; d++) {
            shape64[d] = (int64_t)inputs[i].shape[d];
        }
        OrtStatus *s = model->api->CreateTensorWithDataAsOrtValue(
            model->memory_info, inputs[i].data,
            inputs[i].total_elements * sizeof(float),
            shape64, inputs[i].ndim,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_inputs[i]);
        free(shape64);
        if (check_status(model, s)) goto cleanup;
    }

    /* Run session */
    OrtValue **ort_outputs = calloc(output_count, sizeof(OrtValue *));
    if (!ort_outputs) goto cleanup;

    const char **in_names  = (const char **)model->input_names;
    const char **out_names = (const char **)model->output_names;

    OrtStatus *run_status = model->api->Run(
        model->session, NULL,
        in_names,  (const OrtValue *const *)ort_inputs, input_count,
        out_names, output_count, ort_outputs);

    if (check_status(model, run_status)) {
        free(ort_outputs);
        goto cleanup;
    }

    /* Copy output data */
    for (size_t i = 0; i < output_count; i++) {
        if (!ort_outputs[i]) continue;
        float *out_data = NULL;
        model->api->GetTensorMutableData(ort_outputs[i], (void **)&out_data);
        if (out_data && outputs[i].data) {
            memcpy(outputs[i].data, out_data,
                   outputs[i].total_elements * sizeof(float));
        }
        model->api->ReleaseValue(ort_outputs[i]);
    }
    free(ort_outputs);

    rc = 0;

cleanup:
    if (ort_inputs) {
        for (size_t i = 0; i < input_count; i++) {
            if (ort_inputs[i]) model->api->ReleaseValue(ort_inputs[i]);
        }
        free(ort_inputs);
    }

unlock:
    pthread_mutex_unlock(&model->mutex);
    return rc;
}

/*  Cross-Encoder Re-ranking  */

int gv_onnx_rerank(GV_ONNXModel *model, const char *query_text,
                    const char **doc_texts, size_t doc_count, float *scores) {
    if (!model || !query_text || !doc_texts || !scores) return -1;
    if (doc_count == 0) return -1;
    if (!model->vocab) {
        set_error(model, "No vocabulary loaded — cannot tokenize");
        return -1;
    }

    size_t batch = model->config.max_batch_size > 0
                       ? model->config.max_batch_size : 32;
    int64_t *ids   = calloc(GV_ONNX_MAX_SEQ_LEN, sizeof(int64_t));
    int64_t *attn  = calloc(GV_ONNX_MAX_SEQ_LEN, sizeof(int64_t));
    if (!ids || !attn) { free(ids); free(attn); return -1; }

    /* Tokenize query once */
    int64_t q_ids[GV_ONNX_MAX_SEQ_LEN];
    size_t q_len = tokenize_text(model->vocab, query_text,
                                  q_ids, GV_ONNX_MAX_SEQ_LEN);

    int rc = 0;
    for (size_t base = 0; base < doc_count; base += batch) {
        size_t chunk = doc_count - base;
        if (chunk > batch) chunk = batch;

        for (size_t d = 0; d < chunk; d++) {
            /* Build [CLS] query tokens [SEP] doc tokens [SEP] */
            memset(ids, GV_ONNX_PAD_TOKEN_ID,
                   GV_ONNX_MAX_SEQ_LEN * sizeof(int64_t));
            memset(attn, 0, GV_ONNX_MAX_SEQ_LEN * sizeof(int64_t));

            size_t pos = 0;
            /* Copy query tokens (already has CLS..SEP) */
            for (size_t t = 0; t < q_len && pos < GV_ONNX_MAX_SEQ_LEN; t++) {
                ids[pos] = q_ids[t];
                attn[pos] = 1;
                pos++;
            }

            /* Tokenize document and append */
            int64_t d_ids[GV_ONNX_MAX_SEQ_LEN];
            size_t d_len = tokenize_text(model->vocab, doc_texts[base + d],
                                          d_ids, GV_ONNX_MAX_SEQ_LEN);
            /* Skip the leading CLS of the doc sequence */
            for (size_t t = 1; t < d_len && pos < GV_ONNX_MAX_SEQ_LEN; t++) {
                ids[pos] = d_ids[t];
                attn[pos] = 1;
                pos++;
            }

            /* Run inference for this pair — shape [1, seq_len] */
            size_t seq_len = pos;
            size_t in_shape[2] = { 1, seq_len };

            /* Build float tensors from int64 (ONNX expects float for some
               cross-encoder exports; adjust to int64 tensor if needed) */
            GV_ONNXTensor input_ids  = gv_onnx_tensor_create(in_shape, 2);
            GV_ONNXTensor attn_mask  = gv_onnx_tensor_create(in_shape, 2);
            if (!input_ids.data || !attn_mask.data) {
                gv_onnx_tensor_destroy(&input_ids);
                gv_onnx_tensor_destroy(&attn_mask);
                rc = -1;
                break;
            }

            for (size_t t = 0; t < seq_len; t++) {
                input_ids.data[t] = (float)ids[t];
                attn_mask.data[t] = (float)attn[t];
            }

            size_t out_shape[2] = { 1, 1 };
            GV_ONNXTensor output = gv_onnx_tensor_create(out_shape, 2);
            if (!output.data) {
                gv_onnx_tensor_destroy(&input_ids);
                gv_onnx_tensor_destroy(&attn_mask);
                rc = -1;
                break;
            }

            GV_ONNXTensor ins[2]  = { input_ids, attn_mask };
            GV_ONNXTensor outs[1] = { output };

            if (gv_onnx_infer(model, ins, 2, outs, 1) != 0) {
                gv_onnx_tensor_destroy(&input_ids);
                gv_onnx_tensor_destroy(&attn_mask);
                gv_onnx_tensor_destroy(&output);
                rc = -1;
                break;
            }

            scores[base + d] = outs[0].data[0];

            gv_onnx_tensor_destroy(&input_ids);
            gv_onnx_tensor_destroy(&attn_mask);
            gv_onnx_tensor_destroy(&output);
        }
        if (rc != 0) break;
    }

    free(ids);
    free(attn);
    return rc;
}

/*  Bi-Encoder Embedding  */

int gv_onnx_embed(GV_ONNXModel *model, const char **texts,
                   size_t text_count, float *embeddings, size_t dimension) {
    if (!model || !texts || !embeddings) return -1;
    if (text_count == 0 || dimension == 0) return -1;
    if (!model->vocab) {
        set_error(model, "No vocabulary loaded — cannot tokenize");
        return -1;
    }

    size_t batch = model->config.max_batch_size > 0
                       ? model->config.max_batch_size : 32;
    int rc = 0;

    for (size_t base = 0; base < text_count; base += batch) {
        size_t chunk = text_count - base;
        if (chunk > batch) chunk = batch;

        for (size_t i = 0; i < chunk; i++) {
            int64_t tok_ids[GV_ONNX_MAX_SEQ_LEN];
            size_t tok_len = tokenize_text(model->vocab, texts[base + i],
                                            tok_ids, GV_ONNX_MAX_SEQ_LEN);

            size_t in_shape[2] = { 1, tok_len };
            GV_ONNXTensor input_ids = gv_onnx_tensor_create(in_shape, 2);
            GV_ONNXTensor attn_mask = gv_onnx_tensor_create(in_shape, 2);
            if (!input_ids.data || !attn_mask.data) {
                gv_onnx_tensor_destroy(&input_ids);
                gv_onnx_tensor_destroy(&attn_mask);
                rc = -1;
                break;
            }

            for (size_t t = 0; t < tok_len; t++) {
                input_ids.data[t] = (float)tok_ids[t];
                attn_mask.data[t] = 1.0f;
            }

            size_t out_shape[2] = { 1, dimension };
            GV_ONNXTensor output = gv_onnx_tensor_create(out_shape, 2);
            if (!output.data) {
                gv_onnx_tensor_destroy(&input_ids);
                gv_onnx_tensor_destroy(&attn_mask);
                rc = -1;
                break;
            }

            GV_ONNXTensor ins[2]  = { input_ids, attn_mask };
            GV_ONNXTensor outs[1] = { output };

            if (gv_onnx_infer(model, ins, 2, outs, 1) != 0) {
                gv_onnx_tensor_destroy(&input_ids);
                gv_onnx_tensor_destroy(&attn_mask);
                gv_onnx_tensor_destroy(&output);
                rc = -1;
                break;
            }

            memcpy(embeddings + (base + i) * dimension, outs[0].data,
                   dimension * sizeof(float));

            gv_onnx_tensor_destroy(&input_ids);
            gv_onnx_tensor_destroy(&attn_mask);
            gv_onnx_tensor_destroy(&output);
        }
        if (rc != 0) break;
    }

    return rc;
}

/*  Model Introspection  */

int gv_onnx_get_input_info(const GV_ONNXModel *model, size_t *input_count,
                            char ***input_names) {
    if (!model || !input_count || !input_names) return -1;

    *input_count = model->input_count;
    *input_names = calloc(model->input_count, sizeof(char *));
    if (!*input_names) return -1;

    for (size_t i = 0; i < model->input_count; i++) {
        (*input_names)[i] = model->input_names[i]
                                ? strdup(model->input_names[i])
                                : NULL;
    }
    return 0;
}

int gv_onnx_get_output_info(const GV_ONNXModel *model, size_t *output_count,
                             char ***output_names) {
    if (!model || !output_count || !output_names) return -1;

    *output_count = model->output_count;
    *output_names = calloc(model->output_count, sizeof(char *));
    if (!*output_names) return -1;

    for (size_t i = 0; i < model->output_count; i++) {
        (*output_names)[i] = model->output_names[i]
                                 ? strdup(model->output_names[i])
                                 : NULL;
    }
    return 0;
}

/* ############################################################################
 * ===  Stub implementation (no ONNX Runtime)  ================================
 * ############################################################################ */

#else /* !GV_HAVE_ONNX */

int gv_onnx_available(void) {
    return 0;
}

GV_ONNXModel *gv_onnx_load(const GV_ONNXConfig *config) {
    (void)config;
    fprintf(stderr,
            "gv_onnx_load: ONNX Runtime not available "
            "(compile with -DGV_HAVE_ONNX)\n");
    return NULL;
}

void gv_onnx_destroy(GV_ONNXModel *model) {
    if (!model) return;
    pthread_mutex_destroy(&model->mutex);
    vocab_destroy(model->vocab);
    free(model);
}

int gv_onnx_infer(GV_ONNXModel *model, const GV_ONNXTensor *inputs,
                   size_t input_count, GV_ONNXTensor *outputs,
                   size_t output_count) {
    (void)model; (void)inputs; (void)input_count;
    (void)outputs; (void)output_count;
    return -1;
}

int gv_onnx_rerank(GV_ONNXModel *model, const char *query_text,
                    const char **doc_texts, size_t doc_count, float *scores) {
    (void)model; (void)query_text; (void)doc_texts;
    (void)doc_count; (void)scores;
    return -1;
}

int gv_onnx_embed(GV_ONNXModel *model, const char **texts,
                   size_t text_count, float *embeddings, size_t dimension) {
    (void)model; (void)texts; (void)text_count;
    (void)embeddings; (void)dimension;
    return -1;
}

int gv_onnx_get_input_info(const GV_ONNXModel *model, size_t *input_count,
                            char ***input_names) {
    (void)model; (void)input_count; (void)input_names;
    return -1;
}

int gv_onnx_get_output_info(const GV_ONNXModel *model, size_t *output_count,
                             char ***output_names) {
    (void)model; (void)output_count; (void)output_names;
    return -1;
}

#endif /* GV_HAVE_ONNX */
