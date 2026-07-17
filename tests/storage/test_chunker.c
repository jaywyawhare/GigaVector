/**
 * test_chunker.c — edge cases for the document chunker (src/storage/chunker.c):
 * empty/whitespace-only text, single word, overlap>=chunk_tokens clamp,
 * chunk_tokens==0 default, long doc_id truncation, and normal multi-chunk
 * with overlap.
 */
#include <stdio.h>
#include <string.h>
#include "storage/chunker.h"

#define ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); return -1; } } while (0)

/* Empty and whitespace-only inputs tokenize to zero words -> 0 chunks, rc 0. */
static int test_empty_and_whitespace(void) {
    GV_Chunk *chunks = NULL;
    size_t count = 999;

    ASSERT(gv_chunk_document("", 8, 2, "doc", &chunks, &count) == 0);
    ASSERT(count == 0);
    ASSERT(chunks == NULL);
    gv_chunks_free(chunks, count);

    chunks = NULL; count = 999;
    ASSERT(gv_chunk_document("   \t\n  \r ", 8, 2, "doc", &chunks, &count) == 0);
    ASSERT(count == 0);
    ASSERT(chunks == NULL);
    gv_chunks_free(chunks, count);
    return 0;
}

/* NULL args are rejected. */
static int test_null_args(void) {
    GV_Chunk *chunks = NULL;
    size_t count = 0;
    ASSERT(gv_chunk_document(NULL, 8, 2, "doc", &chunks, &count) == -1);
    ASSERT(gv_chunk_document("hello", 8, 2, "doc", NULL, &count) == -1);
    ASSERT(gv_chunk_document("hello", 8, 2, "doc", &chunks, NULL) == -1);
    return 0;
}

/* A single word yields exactly one chunk containing that word. */
static int test_single_word(void) {
    GV_Chunk *chunks = NULL;
    size_t count = 0;
    ASSERT(gv_chunk_document("hello", 8, 2, "doc", &chunks, &count) == 0);
    ASSERT(count == 1);
    ASSERT(chunks != NULL);
    ASSERT(strcmp(chunks[0].text, "hello") == 0);
    ASSERT(chunks[0].chunk_index == 0);
    ASSERT(strcmp(chunks[0].doc_id, "doc") == 0);
    ASSERT(strcmp(chunks[0].chunk_id, "doc:0000") == 0);
    gv_chunks_free(chunks, count);
    return 0;
}

/* overlap >= chunk_tokens is clamped to chunk_tokens/4 so stride stays > 0
 * and chunking terminates (does not loop forever). */
static int test_overlap_clamp(void) {
    GV_Chunk *chunks = NULL;
    size_t count = 0;
    /* 10 words, chunk_tokens=4, overlap=4 (>= chunk_tokens) -> clamped to 1. */
    const char *text = "a b c d e f g h i j";
    ASSERT(gv_chunk_document(text, 4, 4, "doc", &chunks, &count) == 0);
    ASSERT(count > 1); /* must have made forward progress, multiple chunks */
    /* chunk indices are contiguous starting at 0 */
    for (size_t i = 0; i < count; i++) ASSERT(chunks[i].chunk_index == i);
    gv_chunks_free(chunks, count);
    return 0;
}

/* chunk_tokens == 0 uses the default (512); a short doc -> a single chunk. */
static int test_zero_chunk_tokens(void) {
    GV_Chunk *chunks = NULL;
    size_t count = 0;
    ASSERT(gv_chunk_document("one two three four", 0, 0, "doc", &chunks, &count) == 0);
    ASSERT(count == 1); /* 4 words << default 512 -> one chunk */
    ASSERT(strcmp(chunks[0].text, "one two three four") == 0);
    gv_chunks_free(chunks, count);
    return 0;
}

/* A doc_id longer than the 32-byte doc_id field is truncated by snprintf.
 * Document that truncation: doc_id/chunk_id are NUL-terminated and bounded. */
static int test_long_doc_id_truncation(void) {
    GV_Chunk *chunks = NULL;
    size_t count = 0;
    /* 40-char doc_id; doc_id[] is char[32] so only 31 chars survive. */
    const char *long_id = "0123456789abcdef0123456789abcdef01234567";
    ASSERT(gv_chunk_document("hello world", 8, 2, long_id, &chunks, &count) == 0);
    ASSERT(count == 1);
    /* doc_id is truncated to 31 chars + NUL. */
    ASSERT(strlen(chunks[0].doc_id) == 31);
    ASSERT(strncmp(chunks[0].doc_id, long_id, 31) == 0);
    /* chunk_id fits in 64 bytes and is NUL-terminated. */
    ASSERT(strlen(chunks[0].chunk_id) < sizeof(chunks[0].chunk_id));
    ASSERT(strncmp(chunks[0].chunk_id, chunks[0].doc_id, 31) == 0);
    gv_chunks_free(chunks, count);
    return 0;
}

/* Normal multi-chunk path: overlap means consecutive chunks share words, so
 * chunk char ranges advance but overlap. */
static int test_multi_chunk_overlap(void) {
    GV_Chunk *chunks = NULL;
    size_t count = 0;
    const char *text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11";
    /* chunk_tokens=4, overlap=2 -> stride 2, 12 words -> multiple chunks. */
    ASSERT(gv_chunk_document(text, 4, 2, "doc", &chunks, &count) == 0);
    ASSERT(count >= 3);
    /* Each chunk's char range is non-empty and within the source. */
    size_t src_len = strlen(text);
    for (size_t i = 0; i < count; i++) {
        ASSERT(chunks[i].char_start < chunks[i].char_end);
        ASSERT(chunks[i].char_end <= src_len);
        ASSERT(chunks[i].text != NULL && chunks[i].text[0] != '\0');
    }
    /* Starts are non-decreasing across chunks. */
    for (size_t i = 1; i < count; i++)
        ASSERT(chunks[i].char_start >= chunks[i - 1].char_start);
    gv_chunks_free(chunks, count);
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_empty_and_whitespace();
    rc |= test_null_args();
    rc |= test_single_word();
    rc |= test_overlap_clamp();
    rc |= test_zero_chunk_tokens();
    rc |= test_long_doc_id_truncation();
    rc |= test_multi_chunk_overlap();
    if (rc == 0) printf("All chunker tests PASSED.\n");
    return rc != 0;
}
