#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "multimodal/muvera.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define TOKEN_DIM 16

static void fill_tokens(float *tokens, size_t num_tokens, size_t dim, float seed) {
    for (size_t i = 0; i < num_tokens; i++) {
        for (size_t j = 0; j < dim; j++) {
            tokens[i * dim + j] = sinf(seed + (float)(i * dim + j) * 0.3f);
        }
    }
}

static int test_muvera_config_init(void) {
    GV_MuveraConfig config;
    memset(&config, 0xFF, sizeof(config));

    muvera_config_init(&config);

    ASSERT(config.token_dimension == 128, "default token_dimension should be 128");
    ASSERT(config.num_projections == 64, "default num_projections should be 64");
    ASSERT(config.output_dimension == 0, "default output_dimension should be 0 (auto)");
    ASSERT(config.seed == 42, "default seed should be 42");
    ASSERT(config.normalize == 1, "default normalize should be 1");

    return 0;
}

static int test_muvera_create_destroy(void) {
    GV_MuveraConfig config;
    muvera_config_init(&config);
    config.token_dimension = TOKEN_DIM;
    config.num_projections = 8;
    config.output_dimension = 0;

    GV_MuveraEncoder *enc = muvera_create(&config);
    ASSERT(enc != NULL, "muvera_create returned NULL");

    size_t out_dim = muvera_output_dimension(enc);
    ASSERT(out_dim > 0, "output dimension should be > 0");

    muvera_destroy(enc);
    return 0;
}

static int test_muvera_create_defaults(void) {
    GV_MuveraEncoder *enc = muvera_create(NULL);
    ASSERT(enc != NULL, "create with NULL config returned NULL");

    size_t out_dim = muvera_output_dimension(enc);
    ASSERT(out_dim > 0, "output dimension should be > 0");

    muvera_destroy(enc);
    return 0;
}

static int test_muvera_encode(void) {
    GV_MuveraConfig config;
    muvera_config_init(&config);
    config.token_dimension = TOKEN_DIM;
    config.num_projections = 8;

    GV_MuveraEncoder *enc = muvera_create(&config);
    ASSERT(enc != NULL, "create failed");

    size_t out_dim = muvera_output_dimension(enc);
    float *output = (float *)calloc(out_dim, sizeof(float));
    ASSERT(output != NULL, "calloc failed");

    size_t num_tokens = 5;
    float *tokens = (float *)malloc(num_tokens * TOKEN_DIM * sizeof(float));
    ASSERT(tokens != NULL, "malloc failed");
    fill_tokens(tokens, num_tokens, TOKEN_DIM, 1.0f);

    int rc = muvera_encode(enc, tokens, num_tokens, output);
    ASSERT(rc == 0, "muvera_encode failed");

    int nonzero = 0;
    for (size_t i = 0; i < out_dim; i++) {
        if (fabsf(output[i]) > 1e-9f) { nonzero = 1; break; }
    }
    ASSERT(nonzero, "encoded output should not be all zeros");

    free(tokens);
    free(output);
    muvera_destroy(enc);
    return 0;
}

static int test_muvera_encode_deterministic(void) {
    GV_MuveraConfig config;
    muvera_config_init(&config);
    config.token_dimension = TOKEN_DIM;
    config.num_projections = 8;
    config.seed = 123;

    GV_MuveraEncoder *enc1 = muvera_create(&config);
    GV_MuveraEncoder *enc2 = muvera_create(&config);
    ASSERT(enc1 != NULL && enc2 != NULL, "create failed");

    size_t out_dim = muvera_output_dimension(enc1);
    float *out1 = (float *)malloc(out_dim * sizeof(float));
    float *out2 = (float *)malloc(out_dim * sizeof(float));
    ASSERT(out1 != NULL && out2 != NULL, "malloc failed");

    size_t num_tokens = 3;
    float *tokens = (float *)malloc(num_tokens * TOKEN_DIM * sizeof(float));
    ASSERT(tokens != NULL, "malloc failed");
    fill_tokens(tokens, num_tokens, TOKEN_DIM, 2.0f);

    ASSERT(muvera_encode(enc1, tokens, num_tokens, out1) == 0, "encode1 failed");
    ASSERT(muvera_encode(enc2, tokens, num_tokens, out2) == 0, "encode2 failed");

    ASSERT(memcmp(out1, out2, out_dim * sizeof(float)) == 0,
           "same seed should produce identical encodings");

    free(tokens);
    free(out1);
    free(out2);
    muvera_destroy(enc1);
    muvera_destroy(enc2);
    return 0;
}

static int test_muvera_encode_batch(void) {
    GV_MuveraConfig config;
    muvera_config_init(&config);
    config.token_dimension = TOKEN_DIM;
    config.num_projections = 8;

    GV_MuveraEncoder *enc = muvera_create(&config);
    ASSERT(enc != NULL, "create failed");

    size_t out_dim = muvera_output_dimension(enc);
    size_t batch_size = 3;

    size_t counts[3] = { 4, 2, 6 };
    float *set0 = (float *)malloc(counts[0] * TOKEN_DIM * sizeof(float));
    float *set1 = (float *)malloc(counts[1] * TOKEN_DIM * sizeof(float));
    float *set2 = (float *)malloc(counts[2] * TOKEN_DIM * sizeof(float));
    ASSERT(set0 && set1 && set2, "malloc failed");

    fill_tokens(set0, counts[0], TOKEN_DIM, 0.0f);
    fill_tokens(set1, counts[1], TOKEN_DIM, 1.0f);
    fill_tokens(set2, counts[2], TOKEN_DIM, 2.0f);

    const float *token_sets[3] = { set0, set1, set2 };
    float *outputs = (float *)calloc(batch_size * out_dim, sizeof(float));
    ASSERT(outputs != NULL, "calloc failed");

    int rc = muvera_encode_batch(enc, token_sets, counts, batch_size, outputs);
    ASSERT(rc == 0, "muvera_encode_batch failed");

    free(set0);
    free(set1);
    free(set2);
    free(outputs);
    muvera_destroy(enc);
    return 0;
}

static int test_muvera_output_dimension(void) {
    GV_MuveraConfig config;
    muvera_config_init(&config);
    config.token_dimension = TOKEN_DIM;
    config.num_projections = 8;
    config.output_dimension = 64;

    GV_MuveraEncoder *enc = muvera_create(&config);
    ASSERT(enc != NULL, "create failed");

    size_t out_dim = muvera_output_dimension(enc);
    ASSERT(out_dim == 64, "explicit output_dimension should be honored");

    muvera_destroy(enc);
    return 0;
}

static int test_muvera_destroy_null(void) {
    muvera_destroy(NULL);

    ASSERT(muvera_output_dimension(NULL) == 0,
           "output_dimension of NULL encoder should be 0");

    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing muvera config init...",            test_muvera_config_init},
        {"Testing muvera create/destroy...",         test_muvera_create_destroy},
        {"Testing muvera create defaults...",        test_muvera_create_defaults},
        {"Testing muvera encode...",                 test_muvera_encode},
        {"Testing muvera encode deterministic...",   test_muvera_encode_deterministic},
        {"Testing muvera encode batch...",           test_muvera_encode_batch},
        {"Testing muvera output dimension...",       test_muvera_output_dimension},
        {"Testing muvera destroy null...",           test_muvera_destroy_null},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
