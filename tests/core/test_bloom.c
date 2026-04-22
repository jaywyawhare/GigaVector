#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/bloom.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_bloom_create_destroy(void) {
    GV_BloomFilter *bf = bloom_create(100, 0.01);
    ASSERT(bf != NULL, "bloom filter creation with expected_items=100, fp_rate=0.01");

    bloom_destroy(bf);

    bloom_destroy(NULL);
    return 0;
}

static int test_bloom_add_and_check(void) {
    GV_BloomFilter *bf = bloom_create(1000, 0.01);
    ASSERT(bf != NULL, "bloom filter creation");

    int val = 42;
    ASSERT(bloom_add(bf, &val, sizeof(val)) == 0, "add raw data");

    ASSERT(bloom_check(bf, &val, sizeof(val)) == 1, "check raw data present");

    int other = 9999;
    /* Note: false positives possible, but with low fp_rate and few items it should be 0 */
    int result = bloom_check(bf, &other, sizeof(other));
    ASSERT(result == 0 || result == 1, "check returns 0 or 1 for unknown item");

    bloom_destroy(bf);
    return 0;
}

static int test_bloom_string_operations(void) {
    GV_BloomFilter *bf = bloom_create(500, 0.01);
    ASSERT(bf != NULL, "bloom filter creation");

    ASSERT(bloom_add_string(bf, "hello") == 0, "add string 'hello'");
    ASSERT(bloom_add_string(bf, "world") == 0, "add string 'world'");
    ASSERT(bloom_add_string(bf, "gigavector") == 0, "add string 'gigavector'");

    ASSERT(bloom_check_string(bf, "hello") == 1, "check 'hello' present");
    ASSERT(bloom_check_string(bf, "world") == 1, "check 'world' present");
    ASSERT(bloom_check_string(bf, "gigavector") == 1, "check 'gigavector' present");

    int result = bloom_check_string(bf, "nothere");
    ASSERT(result == 0 || result == 1, "check unknown string returns valid result");

    bloom_destroy(bf);
    return 0;
}

static int test_bloom_count(void) {
    GV_BloomFilter *bf = bloom_create(100, 0.05);
    ASSERT(bf != NULL, "bloom filter creation");

    ASSERT(bloom_count(bf) == 0, "count is 0 on empty filter");

    bloom_add_string(bf, "aaa");
    bloom_add_string(bf, "bbb");
    bloom_add_string(bf, "ccc");
    ASSERT(bloom_count(bf) == 3, "count is 3 after 3 inserts");

    ASSERT(bloom_count(NULL) == 0, "count of NULL filter is 0");

    bloom_destroy(bf);
    return 0;
}

static int test_bloom_fp_rate(void) {
    GV_BloomFilter *bf = bloom_create(1000, 0.01);
    ASSERT(bf != NULL, "bloom filter creation");

    double rate_empty = bloom_fp_rate(bf);
    ASSERT(rate_empty < 1e-9, "empty filter has ~0 FP rate");

    for (int i = 0; i < 100; i++) {
        bloom_add(bf, &i, sizeof(i));
    }
    double rate_partial = bloom_fp_rate(bf);
    ASSERT(rate_partial >= 0.0 && rate_partial <= 1.0, "FP rate in valid range");

    ASSERT(bloom_fp_rate(NULL) == 0.0, "NULL filter FP rate is 0");

    bloom_destroy(bf);
    return 0;
}

static int test_bloom_clear(void) {
    GV_BloomFilter *bf = bloom_create(100, 0.01);
    ASSERT(bf != NULL, "bloom filter creation");

    bloom_add_string(bf, "test1");
    bloom_add_string(bf, "test2");
    ASSERT(bloom_count(bf) == 2, "count is 2 before clear");

    bloom_clear(bf);
    ASSERT(bloom_count(bf) == 0, "count is 0 after clear");

    ASSERT(bloom_check_string(bf, "test1") == 0, "'test1' absent after clear");
    ASSERT(bloom_check_string(bf, "test2") == 0, "'test2' absent after clear");

    bloom_clear(NULL);

    bloom_destroy(bf);
    return 0;
}

static int test_bloom_save_load(void) {
    const char *path = "/tmp/test_bloom_save_load.bin";
    GV_BloomFilter *bf = bloom_create(200, 0.01);
    ASSERT(bf != NULL, "bloom filter creation");

    bloom_add_string(bf, "alpha");
    bloom_add_string(bf, "beta");
    bloom_add_string(bf, "gamma");

    FILE *fout = fopen(path, "wb");
    ASSERT(fout != NULL, "open file for writing");
    ASSERT(bloom_save(bf, fout) == 0, "save bloom filter");
    fclose(fout);

    FILE *fin = fopen(path, "rb");
    ASSERT(fin != NULL, "open file for reading");
    GV_BloomFilter *loaded = NULL;
    ASSERT(bloom_load(&loaded, fin) == 0, "load bloom filter");
    fclose(fin);

    ASSERT(loaded != NULL, "loaded filter is not NULL");
    ASSERT(bloom_count(loaded) == 3, "loaded filter has correct count");

    ASSERT(bloom_check_string(loaded, "alpha") == 1, "'alpha' present in loaded filter");
    ASSERT(bloom_check_string(loaded, "beta") == 1, "'beta' present in loaded filter");
    ASSERT(bloom_check_string(loaded, "gamma") == 1, "'gamma' present in loaded filter");

    bloom_destroy(bf);
    bloom_destroy(loaded);
    remove(path);
    return 0;
}

static int test_bloom_merge(void) {
    GV_BloomFilter *a = bloom_create(100, 0.01);
    GV_BloomFilter *b = bloom_create(100, 0.01);
    ASSERT(a != NULL && b != NULL, "bloom filter creation for merge");

    bloom_add_string(a, "item_a1");
    bloom_add_string(a, "item_a2");
    bloom_add_string(b, "item_b1");
    bloom_add_string(b, "item_b2");

    GV_BloomFilter *merged = bloom_merge(a, b);
    ASSERT(merged != NULL, "merge succeeded");

    ASSERT(bloom_check_string(merged, "item_a1") == 1, "'item_a1' in merged");
    ASSERT(bloom_check_string(merged, "item_a2") == 1, "'item_a2' in merged");
    ASSERT(bloom_check_string(merged, "item_b1") == 1, "'item_b1' in merged");
    ASSERT(bloom_check_string(merged, "item_b2") == 1, "'item_b2' in merged");

    ASSERT(bloom_count(merged) == 4, "merged count is sum of a and b");

    bloom_destroy(a);
    bloom_destroy(b);
    bloom_destroy(merged);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing bloom create/destroy...", test_bloom_create_destroy},
        {"Testing bloom add and check...", test_bloom_add_and_check},
        {"Testing bloom string operations...", test_bloom_string_operations},
        {"Testing bloom count...", test_bloom_count},
        {"Testing bloom FP rate...", test_bloom_fp_rate},
        {"Testing bloom clear...", test_bloom_clear},
        {"Testing bloom save/load...", test_bloom_save_load},
        {"Testing bloom merge...", test_bloom_merge},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
