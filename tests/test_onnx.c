#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_onnx.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_onnx_available(void) {
    int avail = gv_onnx_available();
    ASSERT(avail == 0 || avail == 1, "gv_onnx_available should return 0 or 1");
    return 0;
}

static int test_load_nonexistent(void) {
    GV_ONNXConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.model_path = "/tmp/nonexistent_model_file_that_does_not_exist.onnx";
    cfg.num_threads = 1;
    cfg.use_gpu = 0;
    cfg.max_batch_size = 1;
    cfg.optimization_level = 0;

    GV_ONNXModel *model = gv_onnx_load(&cfg);
    ASSERT(model == NULL, "gv_onnx_load with nonexistent file should return NULL");
    return 0;
}

static int test_load_null_path(void) {
    GV_ONNXConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.model_path = NULL;
    cfg.num_threads = 1;
    cfg.use_gpu = 0;
    cfg.max_batch_size = 1;
    cfg.optimization_level = 0;

    GV_ONNXModel *model = gv_onnx_load(&cfg);
    ASSERT(model == NULL, "gv_onnx_load with NULL path should return NULL");
    return 0;
}

static int test_destroy_null(void) {
    /* Destroying NULL must not crash */
    gv_onnx_destroy(NULL);
    return 0;
}

static int test_tensor_create_1d(void) {
    size_t shape[] = {10};
    GV_ONNXTensor t = gv_onnx_tensor_create(shape, 1);
    ASSERT(t.data != NULL, "tensor data should be allocated");
    ASSERT(t.shape != NULL, "tensor shape should be allocated");
    ASSERT(t.ndim == 1, "tensor ndim should be 1");
    ASSERT(t.total_elements == 10, "tensor total_elements should be 10");
    ASSERT(t.shape[0] == 10, "tensor shape[0] should be 10");

    /* Data should be zero-initialized */
    for (size_t i = 0; i < t.total_elements; i++) {
        ASSERT(t.data[i] == 0.0f, "tensor data should be zero-initialized");
    }

    gv_onnx_tensor_destroy(&t);
    ASSERT(t.data == NULL, "tensor data should be NULL after destroy");
    ASSERT(t.shape == NULL, "tensor shape should be NULL after destroy");
    return 0;
}

static int test_tensor_create_2d(void) {
    size_t shape[] = {3, 4};
    GV_ONNXTensor t = gv_onnx_tensor_create(shape, 2);
    ASSERT(t.data != NULL, "2D tensor data should be allocated");
    ASSERT(t.shape != NULL, "2D tensor shape should be allocated");
    ASSERT(t.ndim == 2, "tensor ndim should be 2");
    ASSERT(t.total_elements == 12, "tensor total_elements should be 3*4=12");
    ASSERT(t.shape[0] == 3, "shape[0] should be 3");
    ASSERT(t.shape[1] == 4, "shape[1] should be 4");

    gv_onnx_tensor_destroy(&t);
    return 0;
}

static int test_tensor_create_3d(void) {
    size_t shape[] = {2, 3, 5};
    GV_ONNXTensor t = gv_onnx_tensor_create(shape, 3);
    ASSERT(t.data != NULL, "3D tensor data should be allocated");
    ASSERT(t.ndim == 3, "tensor ndim should be 3");
    ASSERT(t.total_elements == 30, "tensor total_elements should be 2*3*5=30");
    ASSERT(t.shape[0] == 2, "shape[0]");
    ASSERT(t.shape[1] == 3, "shape[1]");
    ASSERT(t.shape[2] == 5, "shape[2]");

    gv_onnx_tensor_destroy(&t);
    return 0;
}

static int test_tensor_create_single_element(void) {
    size_t shape[] = {1};
    GV_ONNXTensor t = gv_onnx_tensor_create(shape, 1);
    ASSERT(t.data != NULL, "single element tensor data should be allocated");
    ASSERT(t.total_elements == 1, "total_elements should be 1");

    /* Write and read back */
    t.data[0] = 42.0f;
    ASSERT(t.data[0] == 42.0f, "should be able to write to tensor data");

    gv_onnx_tensor_destroy(&t);
    return 0;
}

static int test_tensor_destroy_null_data(void) {
    /* tensor_destroy with NULL data should not crash */
    GV_ONNXTensor t;
    memset(&t, 0, sizeof(t));
    t.data = NULL;
    t.shape = NULL;
    t.ndim = 0;
    t.total_elements = 0;
    gv_onnx_tensor_destroy(&t);
    return 0;
}

static int test_tensor_write_read(void) {
    size_t shape[] = {2, 3};
    GV_ONNXTensor t = gv_onnx_tensor_create(shape, 2);
    ASSERT(t.data != NULL, "tensor data");

    /* Fill with known values */
    for (size_t i = 0; i < t.total_elements; i++) {
        t.data[i] = (float)i * 1.5f;
    }

    /* Verify values */
    for (size_t i = 0; i < t.total_elements; i++) {
        float expected = (float)i * 1.5f;
        ASSERT(t.data[i] == expected, "tensor data should match written values");
    }

    gv_onnx_tensor_destroy(&t);
    return 0;
}

static int test_infer_null_model(void) {
    size_t shape[] = {1, 4};
    GV_ONNXTensor input = gv_onnx_tensor_create(shape, 2);
    GV_ONNXTensor output = gv_onnx_tensor_create(shape, 2);

    /* Infer with NULL model should fail */
    int rc = gv_onnx_infer(NULL, &input, 1, &output, 1);
    ASSERT(rc != 0, "gv_onnx_infer with NULL model should fail");

    gv_onnx_tensor_destroy(&input);
    gv_onnx_tensor_destroy(&output);
    return 0;
}

static int test_rerank_null_model(void) {
    const char *docs[] = {"hello", "world"};
    float scores[2] = {0};
    int rc = gv_onnx_rerank(NULL, "query", docs, 2, scores);
    ASSERT(rc != 0, "gv_onnx_rerank with NULL model should fail");
    return 0;
}

static int test_embed_null_model(void) {
    const char *texts[] = {"hello"};
    float embeddings[4] = {0};
    int rc = gv_onnx_embed(NULL, texts, 1, embeddings, 4);
    ASSERT(rc != 0, "gv_onnx_embed with NULL model should fail");
    return 0;
}

static int test_get_input_info_null(void) {
    size_t count = 0;
    char **names = NULL;
    int rc = gv_onnx_get_input_info(NULL, &count, &names);
    ASSERT(rc != 0, "gv_onnx_get_input_info with NULL model should fail");
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing onnx_available...",        test_onnx_available},
        {"Testing load_nonexistent...",      test_load_nonexistent},
        {"Testing load_null_path...",        test_load_null_path},
        {"Testing destroy_null...",          test_destroy_null},
        {"Testing tensor_create_1d...",      test_tensor_create_1d},
        {"Testing tensor_create_2d...",      test_tensor_create_2d},
        {"Testing tensor_create_3d...",      test_tensor_create_3d},
        {"Testing tensor_single_element...", test_tensor_create_single_element},
        {"Testing tensor_destroy_null...",   test_tensor_destroy_null_data},
        {"Testing tensor_write_read...",     test_tensor_write_read},
        {"Testing infer_null_model...",      test_infer_null_model},
        {"Testing rerank_null_model...",     test_rerank_null_model},
        {"Testing embed_null_model...",      test_embed_null_model},
        {"Testing get_input_info_null...",   test_get_input_info_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
