#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "multimodal/payload_inverted.h"
#include "core/id_bitmap.h"
#include "core/memory.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_create_destroy(void) {
    GV_PayloadInvertedIndex *idx = payload_inverted_create();
    ASSERT(idx != NULL, "create index");
    ASSERT(payload_inverted_field_count(idx) == 0, "initially 0 fields");
    ASSERT(payload_inverted_entry_count(idx) == 0, "initially 0 entries");
    payload_inverted_destroy(idx);
    payload_inverted_destroy(NULL);
    return 0;
}

static int test_add_fields(void) {
    GV_PayloadInvertedIndex *idx = payload_inverted_create();
    ASSERT(idx != NULL, "create");

    ASSERT(payload_inverted_add_field(idx, "color", 0) == 0, "add string field");
    ASSERT(payload_inverted_add_field(idx, "age", 1) == 0, "add int field");
    ASSERT(payload_inverted_add_field(idx, "score", 2) == 0, "add float field");
    ASSERT(payload_inverted_add_field(idx, "active", 3) == 0, "add bool field");
    ASSERT(payload_inverted_field_count(idx) == 4, "4 fields");

    /* Adding same field again should be idempotent */
    ASSERT(payload_inverted_add_field(idx, "color", 0) == 0, "idempotent add");
    ASSERT(payload_inverted_field_count(idx) == 4, "still 4 fields");

    payload_inverted_destroy(idx);
    return 0;
}

static int test_exact_match_string(void) {
    GV_PayloadInvertedIndex *idx = payload_inverted_create();
    payload_inverted_add_field(idx, "category", 0);

    payload_inverted_index(idx, "category", 0, "electronics");
    payload_inverted_index(idx, "category", 1, "clothing");
    payload_inverted_index(idx, "category", 2, "electronics");
    payload_inverted_index(idx, "category", 3, "food");
    payload_inverted_index(idx, "category", 4, "clothing");

    ASSERT(payload_inverted_entry_count(idx) == 5, "5 entries");

    GV_IdBitmap *result = payload_inverted_find(idx, "category", "electronics");
    ASSERT(result != NULL, "find electronics");
    ASSERT(gv_id_bitmap_cardinality(result) == 2, "2 electronics");
    ASSERT(gv_id_bitmap_contains(result, 0), "id 0 is electronics");
    ASSERT(gv_id_bitmap_contains(result, 2), "id 2 is electronics");
    gv_id_bitmap_free(result);

    result = payload_inverted_find(idx, "category", "clothing");
    ASSERT(gv_id_bitmap_cardinality(result) == 2, "2 clothing");
    ASSERT(gv_id_bitmap_contains(result, 1), "id 1 is clothing");
    ASSERT(gv_id_bitmap_contains(result, 4), "id 4 is clothing");
    gv_id_bitmap_free(result);

    result = payload_inverted_find(idx, "category", "nonexistent");
    ASSERT(result == NULL || gv_id_bitmap_cardinality(result) == 0, "nonexistent returns empty");

    payload_inverted_destroy(idx);
    return 0;
}

static int test_prefix_match(void) {
    GV_PayloadInvertedIndex *idx = payload_inverted_create();
    payload_inverted_add_field(idx, "name", 0);

    payload_inverted_index(idx, "name", 0, "hello world");
    payload_inverted_index(idx, "name", 1, "help yourself");
    payload_inverted_index(idx, "name", 2, "hero journey");
    payload_inverted_index(idx, "name", 3, "goodbye");

    /* Prefix "he" should match tokens "hello", "help", "hero" */
    GV_IdBitmap *result = payload_inverted_find_prefix(idx, "name", "he");
    ASSERT(result != NULL, "prefix find returned");
    size_t card = gv_id_bitmap_cardinality(result);
    ASSERT(card >= 2, "prefix 'he' should match at least ids 0 and 1");
    ASSERT(gv_id_bitmap_contains(result, 0), "id 0 has 'hello'");
    ASSERT(gv_id_bitmap_contains(result, 1), "id 1 has 'help'");
    ASSERT(gv_id_bitmap_contains(result, 2), "id 2 has 'hero'");
    ASSERT(!gv_id_bitmap_contains(result, 3), "id 3 'goodbye' not matched");
    gv_id_bitmap_free(result);

    payload_inverted_destroy(idx);
    return 0;
}

static int test_range_query(void) {
    GV_PayloadInvertedIndex *idx = payload_inverted_create();
    payload_inverted_add_field(idx, "price", 2);

    payload_inverted_index(idx, "price", 0, "10.5");
    payload_inverted_index(idx, "price", 1, "25.0");
    payload_inverted_index(idx, "price", 2, "5.0");
    payload_inverted_index(idx, "price", 3, "50.0");
    payload_inverted_index(idx, "price", 4, "15.0");

    /* Range [10, 20] should match ids 0 (10.5) and 4 (15.0) */
    GV_IdBitmap *result = payload_inverted_find_range(idx, "price", 10.0, 20.0);
    ASSERT(result != NULL, "range find returned");
    ASSERT(gv_id_bitmap_cardinality(result) == 2, "2 items in [10,20]");
    ASSERT(gv_id_bitmap_contains(result, 0), "id 0 (10.5)");
    ASSERT(gv_id_bitmap_contains(result, 4), "id 4 (15.0)");
    ASSERT(!gv_id_bitmap_contains(result, 1), "id 1 (25.0) out of range");
    ASSERT(!gv_id_bitmap_contains(result, 2), "id 2 (5.0) out of range");
    gv_id_bitmap_free(result);

    payload_inverted_destroy(idx);
    return 0;
}

static int test_remove(void) {
    GV_PayloadInvertedIndex *idx = payload_inverted_create();
    payload_inverted_add_field(idx, "tag", 0);

    payload_inverted_index(idx, "tag", 0, "alpha");
    payload_inverted_index(idx, "tag", 1, "beta");
    payload_inverted_index(idx, "tag", 2, "alpha");

    GV_IdBitmap *r = payload_inverted_find(idx, "tag", "alpha");
    ASSERT(r && gv_id_bitmap_cardinality(r) == 2, "2 alpha before remove");
    gv_id_bitmap_free(r);

    ASSERT(payload_inverted_remove(idx, 0) == 0, "remove id 0");

    r = payload_inverted_find(idx, "tag", "alpha");
    ASSERT(r && gv_id_bitmap_cardinality(r) == 1, "1 alpha after remove");
    ASSERT(gv_id_bitmap_contains(r, 2), "id 2 still alpha");
    gv_id_bitmap_free(r);

    r = payload_inverted_find(idx, "tag", "beta");
    ASSERT(r && gv_id_bitmap_cardinality(r) == 1, "beta unaffected");
    ASSERT(gv_id_bitmap_contains(r, 1), "id 1 is beta");
    gv_id_bitmap_free(r);

    payload_inverted_destroy(idx);
    return 0;
}

static int test_persistence(void) {
    GV_PayloadInvertedIndex *idx = payload_inverted_create();
    payload_inverted_add_field(idx, "color", 0);
    payload_inverted_add_field(idx, "score", 2);

    payload_inverted_index(idx, "color", 0, "red");
    payload_inverted_index(idx, "color", 1, "blue");
    payload_inverted_index(idx, "color", 2, "red");
    payload_inverted_index(idx, "score", 0, "99.5");
    payload_inverted_index(idx, "score", 1, "75.0");

    /* Save to a temp file */
    char path[256];
    snprintf(path, sizeof(path), "/tmp/gv_inv_test_%d.bin", (int)getpid());
    FILE *f = fopen(path, "wb");
    ASSERT(f != NULL, "open file for write");
    ASSERT(payload_inverted_save(idx, f) == 0, "save");
    fclose(f);

    /* Load back */
    f = fopen(path, "rb");
    ASSERT(f != NULL, "open file for read");
    GV_PayloadInvertedIndex *loaded = NULL;
    ASSERT(payload_inverted_load(&loaded, f) == 0, "load");
    fclose(f);
    remove(path);

    ASSERT(loaded != NULL, "loaded index not null");
    ASSERT(payload_inverted_field_count(loaded) == 2, "2 fields after load");

    GV_IdBitmap *r = payload_inverted_find(loaded, "color", "red");
    ASSERT(r && gv_id_bitmap_cardinality(r) == 2, "2 red after load");
    ASSERT(gv_id_bitmap_contains(r, 0), "id 0 red");
    ASSERT(gv_id_bitmap_contains(r, 2), "id 2 red");
    gv_id_bitmap_free(r);

    r = payload_inverted_find(loaded, "score", "99.5");
    ASSERT(r && gv_id_bitmap_cardinality(r) == 1, "score 99.5 after load");
    ASSERT(gv_id_bitmap_contains(r, 0), "id 0 has 99.5");
    gv_id_bitmap_free(r);

    payload_inverted_destroy(idx);
    payload_inverted_destroy(loaded);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_create_destroy();
    rc |= test_add_fields();
    rc |= test_exact_match_string();
    rc |= test_prefix_match();
    rc |= test_range_query();
    rc |= test_remove();
    rc |= test_persistence();
    if (rc == 0) printf("All payload inverted index tests passed.\n");
    return rc;
}
