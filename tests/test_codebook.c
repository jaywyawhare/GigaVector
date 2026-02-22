#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gigavector/gv_codebook.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_codebook_create_destroy(void) {
    /* dimension=4, m=2 subspaces, nbits=8 (256 centroids per subspace) */
    GV_Codebook *cb = gv_codebook_create(4, 2, 8);
    ASSERT(cb != NULL, "codebook creation dim=4 m=2 nbits=8");
    ASSERT(cb->dimension == 4, "dimension is 4");
    ASSERT(cb->m == 2, "m is 2");
    ASSERT(cb->nbits == 8, "nbits is 8");
    ASSERT(cb->ksub == 256, "ksub is 256 (1 << 8)");
    ASSERT(cb->dsub == 2, "dsub is 2 (4 / 2)");
    ASSERT(cb->trained == 0, "initially untrained");

    gv_codebook_destroy(cb);

    /* Destroy NULL should be safe */
    gv_codebook_destroy(NULL);
    return 0;
}

static int test_codebook_create_invalid(void) {
    /* dimension not divisible by m should fail */
    GV_Codebook *cb = gv_codebook_create(5, 2, 8);
    ASSERT(cb == NULL, "create with dim=5 m=2 should fail (5 not divisible by 2)");

    /* nbits > 8 should fail */
    cb = gv_codebook_create(4, 2, 9);
    ASSERT(cb == NULL, "create with nbits=9 should fail");

    /* m=0 should fail */
    cb = gv_codebook_create(4, 0, 8);
    ASSERT(cb == NULL, "create with m=0 should fail");

    return 0;
}

static int test_codebook_train(void) {
    GV_Codebook *cb = gv_codebook_create(4, 2, 4); /* 16 centroids per subspace */
    ASSERT(cb != NULL, "codebook creation");

    /* Generate training data: 64 random-ish vectors */
    float data[64 * 4];
    for (int i = 0; i < 64; i++) {
        data[i * 4 + 0] = (float)(i % 7) * 0.1f;
        data[i * 4 + 1] = (float)(i % 5) * 0.2f;
        data[i * 4 + 2] = (float)(i % 3) * 0.3f;
        data[i * 4 + 3] = (float)(i % 11) * 0.05f;
    }

    ASSERT(gv_codebook_train(cb, data, 64, 5) == 0, "train codebook with 64 vectors, 5 iters");
    ASSERT(cb->trained != 0, "codebook is trained after training");

    gv_codebook_destroy(cb);
    return 0;
}

static int test_codebook_encode_decode(void) {
    GV_Codebook *cb = gv_codebook_create(4, 2, 4);
    ASSERT(cb != NULL, "codebook creation");

    /* Train with some data */
    float data[32 * 4];
    for (int i = 0; i < 32; i++) {
        data[i * 4 + 0] = (float)(i % 4);
        data[i * 4 + 1] = (float)(i % 3);
        data[i * 4 + 2] = (float)(i % 5);
        data[i * 4 + 3] = (float)(i % 2);
    }
    ASSERT(gv_codebook_train(cb, data, 32, 5) == 0, "train codebook");

    /* Encode a vector */
    float vec[4] = {1.0f, 2.0f, 3.0f, 0.0f};
    uint8_t codes[2]; /* m=2 */
    ASSERT(gv_codebook_encode(cb, vec, codes) == 0, "encode vector");

    /* Each code should be within [0, ksub) */
    ASSERT(codes[0] < cb->ksub, "code[0] in range");
    ASSERT(codes[1] < cb->ksub, "code[1] in range");

    /* Decode back to an approximate vector */
    float decoded[4];
    ASSERT(gv_codebook_decode(cb, codes, decoded) == 0, "decode codes");

    /* The decoded vector should be an approximation (not necessarily exact) */
    /* Just check that values are finite */
    for (int i = 0; i < 4; i++) {
        ASSERT(!isnan(decoded[i]) && !isinf(decoded[i]), "decoded value is finite");
    }

    gv_codebook_destroy(cb);
    return 0;
}

static int test_codebook_distance_adc(void) {
    GV_Codebook *cb = gv_codebook_create(4, 2, 4);
    ASSERT(cb != NULL, "codebook creation");

    float data[32 * 4];
    for (int i = 0; i < 32; i++) {
        data[i * 4 + 0] = (float)(i % 4);
        data[i * 4 + 1] = (float)(i % 3);
        data[i * 4 + 2] = (float)(i % 5);
        data[i * 4 + 3] = (float)(i % 2);
    }
    ASSERT(gv_codebook_train(cb, data, 32, 5) == 0, "train codebook");

    /* Encode a vector */
    float vec[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    uint8_t codes[2];
    gv_codebook_encode(cb, vec, codes);

    /* ADC distance from itself should be small */
    float dist = gv_codebook_distance_adc(cb, vec, codes);
    ASSERT(dist >= 0.0f, "ADC distance is non-negative");

    /* ADC distance from a very different query should be larger */
    float far_query[4] = {100.0f, 100.0f, 100.0f, 100.0f};
    float dist_far = gv_codebook_distance_adc(cb, far_query, codes);
    ASSERT(dist_far > dist, "far query has larger ADC distance");

    gv_codebook_destroy(cb);
    return 0;
}

static int test_codebook_copy(void) {
    GV_Codebook *cb = gv_codebook_create(4, 2, 4);
    ASSERT(cb != NULL, "codebook creation");

    float data[32 * 4];
    for (int i = 0; i < 32; i++) {
        data[i * 4 + 0] = (float)i;
        data[i * 4 + 1] = (float)(i * 2);
        data[i * 4 + 2] = (float)(i + 1);
        data[i * 4 + 3] = (float)(i % 7);
    }
    gv_codebook_train(cb, data, 32, 3);

    GV_Codebook *copy = gv_codebook_copy(cb);
    ASSERT(copy != NULL, "codebook copy succeeded");
    ASSERT(copy->dimension == cb->dimension, "copy has same dimension");
    ASSERT(copy->m == cb->m, "copy has same m");
    ASSERT(copy->ksub == cb->ksub, "copy has same ksub");
    ASSERT(copy->nbits == cb->nbits, "copy has same nbits");
    ASSERT(copy->trained == cb->trained, "copy has same trained state");
    ASSERT(copy->centroids != cb->centroids, "copy has separate centroid storage");

    /* Verify centroid data matches */
    size_t num_floats = cb->m * cb->ksub * cb->dsub;
    ASSERT(memcmp(copy->centroids, cb->centroids, num_floats * sizeof(float)) == 0,
           "copy centroid data matches original");

    gv_codebook_destroy(cb);
    gv_codebook_destroy(copy);
    return 0;
}

static int test_codebook_save_load(void) {
    const char *path = "/tmp/test_codebook_save_load.bin";
    GV_Codebook *cb = gv_codebook_create(4, 2, 4);
    ASSERT(cb != NULL, "codebook creation");

    float data[32 * 4];
    for (int i = 0; i < 32; i++) {
        data[i * 4 + 0] = (float)(i % 4);
        data[i * 4 + 1] = (float)(i % 3);
        data[i * 4 + 2] = (float)(i % 5);
        data[i * 4 + 3] = (float)(i % 2);
    }
    gv_codebook_train(cb, data, 32, 3);

    ASSERT(gv_codebook_save(cb, path) == 0, "save codebook to file");

    GV_Codebook *loaded = gv_codebook_load(path);
    ASSERT(loaded != NULL, "load codebook from file");
    ASSERT(loaded->dimension == 4, "loaded dimension is 4");
    ASSERT(loaded->m == 2, "loaded m is 2");
    ASSERT(loaded->trained != 0, "loaded codebook is trained");

    /* Encode with both and compare */
    float vec[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    uint8_t codes_orig[2], codes_loaded[2];
    gv_codebook_encode(cb, vec, codes_orig);
    gv_codebook_encode(loaded, vec, codes_loaded);
    ASSERT(codes_orig[0] == codes_loaded[0] && codes_orig[1] == codes_loaded[1],
           "encoding matches between original and loaded");

    gv_codebook_destroy(cb);
    gv_codebook_destroy(loaded);
    remove(path);
    return 0;
}

static int test_codebook_save_load_fp(void) {
    const char *path = "/tmp/test_codebook_fp.bin";
    GV_Codebook *cb = gv_codebook_create(4, 4, 4);
    ASSERT(cb != NULL, "codebook creation dim=4 m=4");

    float data[64 * 4];
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 4; j++) {
            data[i * 4 + j] = (float)((i + j) % 10);
        }
    }
    gv_codebook_train(cb, data, 64, 3);

    /* Save via FILE* */
    FILE *fout = fopen(path, "wb");
    ASSERT(fout != NULL, "open file for writing");
    ASSERT(gv_codebook_save_fp(cb, fout) == 0, "save codebook via FILE*");
    fclose(fout);

    /* Load via FILE* */
    FILE *fin = fopen(path, "rb");
    ASSERT(fin != NULL, "open file for reading");
    GV_Codebook *loaded = gv_codebook_load_fp(fin);
    fclose(fin);
    ASSERT(loaded != NULL, "load codebook via FILE*");
    ASSERT(loaded->dimension == 4, "loaded dimension matches");
    ASSERT(loaded->m == 4, "loaded m matches");

    gv_codebook_destroy(cb);
    gv_codebook_destroy(loaded);
    remove(path);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing codebook create/destroy...", test_codebook_create_destroy},
        {"Testing codebook create invalid params...", test_codebook_create_invalid},
        {"Testing codebook train...", test_codebook_train},
        {"Testing codebook encode/decode...", test_codebook_encode_decode},
        {"Testing codebook ADC distance...", test_codebook_distance_adc},
        {"Testing codebook copy...", test_codebook_copy},
        {"Testing codebook save/load (filepath)...", test_codebook_save_load},
        {"Testing codebook save/load (FILE*)...", test_codebook_save_load_fp},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
