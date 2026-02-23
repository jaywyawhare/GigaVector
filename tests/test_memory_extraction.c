/**
 * @file test_memory_extraction.c
 * @brief Unit tests for memory extraction (gv_memory_extraction.h).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_memory_layer.h"
#include "gigavector/gv_memory_extraction.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

/* Test 1: Extract candidates from simple text */
static int test_extract_from_text_simple(void) {
    GV_MemoryCandidate candidates[10];
    memset(candidates, 0, sizeof(candidates));
    size_t actual_count = 0;

    const char *text = "The user prefers Python over Java. "
                       "They work at a technology company. "
                       "Their favorite color is blue.";

    int ret = gv_memory_extract_candidates_from_text(
        text, "test-source", 0.1, candidates, 10, &actual_count);

    ASSERT(ret == 0, "extract from text should succeed");
    /* Heuristic extraction should find at least some candidates */

    for (size_t i = 0; i < actual_count; i++) {
        ASSERT(candidates[i].content != NULL, "candidate content should not be NULL");
        ASSERT(candidates[i].importance_score >= 0.0 && candidates[i].importance_score <= 1.0,
               "importance score should be in [0,1]");
    }

    gv_memory_candidates_free(candidates, actual_count);
    return 0;
}

/* Test 2: Extract candidates from conversation text */
static int test_extract_from_conversation(void) {
    GV_MemoryCandidate candidates[10];
    memset(candidates, 0, sizeof(candidates));
    size_t actual_count = 0;

    const char *conversation =
        "User: I really like hiking in the mountains.\n"
        "Assistant: That sounds great! Do you have a favorite trail?\n"
        "User: Yes, I love the Pacific Crest Trail. I hiked it last summer.\n"
        "Assistant: That's amazing! How long did it take?\n"
        "User: About 5 months. I prefer long distance hiking.";

    int ret = gv_memory_extract_candidates_from_conversation(
        conversation, "conv-001", 0.1, candidates, 10, &actual_count);

    ASSERT(ret == 0, "extract from conversation should succeed");

    for (size_t i = 0; i < actual_count; i++) {
        ASSERT(candidates[i].content != NULL, "candidate content should not be NULL");
    }

    gv_memory_candidates_free(candidates, actual_count);
    return 0;
}

/* Test 3: Score a constructed candidate */
static int test_score_candidate(void) {
    GV_MemoryCandidate candidate;
    memset(&candidate, 0, sizeof(candidate));

    candidate.content = strdup("User prefers Python programming language");
    candidate.memory_type = GV_MEMORY_TYPE_PREFERENCE;
    candidate.importance_score = 0.5;

    double score = gv_memory_score_candidate(&candidate);
    ASSERT(score >= 0.0 && score <= 1.0, "score should be in [0,1]");

    free(candidate.content);
    return 0;
}

/* Test 4: Score candidate with different content lengths */
static int test_score_candidate_varied(void) {
    /* Short content */
    GV_MemoryCandidate short_candidate;
    memset(&short_candidate, 0, sizeof(short_candidate));
    short_candidate.content = strdup("Hi");
    short_candidate.memory_type = GV_MEMORY_TYPE_FACT;
    short_candidate.importance_score = 0.2;

    double score_short = gv_memory_score_candidate(&short_candidate);
    ASSERT(score_short >= 0.0 && score_short <= 1.0, "short content score in range");

    /* Long content */
    GV_MemoryCandidate long_candidate;
    memset(&long_candidate, 0, sizeof(long_candidate));
    long_candidate.content = strdup(
        "The user is a software engineer with 10 years of experience "
        "specializing in machine learning and natural language processing "
        "who currently works at a major technology company");
    long_candidate.memory_type = GV_MEMORY_TYPE_FACT;
    long_candidate.importance_score = 0.9;

    double score_long = gv_memory_score_candidate(&long_candidate);
    ASSERT(score_long >= 0.0 && score_long <= 1.0, "long content score in range");

    free(short_candidate.content);
    free(long_candidate.content);
    return 0;
}

/* Test 5: Detect memory type for various content */
static int test_detect_type_fact(void) {
    GV_MemoryType t;

    t = gv_memory_detect_type("The capital of France is Paris");
    /* Should return some valid type */
    ASSERT(t >= GV_MEMORY_TYPE_FACT && t <= GV_MEMORY_TYPE_EVENT,
           "detected type should be valid");

    return 0;
}

/* Test 6: Detect type for preference content */
static int test_detect_type_preference(void) {
    GV_MemoryType t;

    t = gv_memory_detect_type("I prefer dark mode over light mode");
    ASSERT(t >= GV_MEMORY_TYPE_FACT && t <= GV_MEMORY_TYPE_EVENT,
           "detected type should be valid for preference");

    t = gv_memory_detect_type("My favorite language is Rust");
    ASSERT(t >= GV_MEMORY_TYPE_FACT && t <= GV_MEMORY_TYPE_EVENT,
           "detected type should be valid for favorite");

    return 0;
}

/* Test 7: Detect type for relationship content */
static int test_detect_type_relationship(void) {
    GV_MemoryType t;

    t = gv_memory_detect_type("Alice is Bob's manager");
    ASSERT(t >= GV_MEMORY_TYPE_FACT && t <= GV_MEMORY_TYPE_EVENT,
           "detected type should be valid for relationship");

    t = gv_memory_detect_type("John works with Sarah on the AI project");
    ASSERT(t >= GV_MEMORY_TYPE_FACT && t <= GV_MEMORY_TYPE_EVENT,
           "detected type should be valid for work relationship");

    return 0;
}

/* Test 8: Detect type for event content */
static int test_detect_type_event(void) {
    GV_MemoryType t;

    t = gv_memory_detect_type("The user graduated from MIT in 2020");
    ASSERT(t >= GV_MEMORY_TYPE_FACT && t <= GV_MEMORY_TYPE_EVENT,
           "detected type should be valid for event");

    t = gv_memory_detect_type("Meeting scheduled for next Tuesday");
    ASSERT(t >= GV_MEMORY_TYPE_FACT && t <= GV_MEMORY_TYPE_EVENT,
           "detected type should be valid for scheduled event");

    return 0;
}

/* Test 9: Free NULL candidate (should not crash) */
static int test_candidate_free_null(void) {
    gv_memory_candidate_free(NULL);
    gv_memory_candidates_free(NULL, 0);
    gv_memory_candidates_free(NULL, 5);
    /* If we reach here, no crash */
    return 0;
}

/* Test 10: Free candidate with allocated fields */
static int test_candidate_free_allocated(void) {
    GV_MemoryCandidate candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.content = strdup("Test content");
    candidate.extraction_context = strdup("Test context");
    candidate.importance_score = 0.5;
    candidate.memory_type = GV_MEMORY_TYPE_FACT;

    gv_memory_candidate_free(&candidate);
    /* Should not crash, fields should be freed */
    return 0;
}

/* Test 11: Extract from empty text */
static int test_extract_empty_text(void) {
    GV_MemoryCandidate candidates[5];
    memset(candidates, 0, sizeof(candidates));
    size_t actual_count = 999;

    int ret = gv_memory_extract_candidates_from_text(
        "", NULL, 0.1, candidates, 5, &actual_count);

    /* Empty text should either succeed with 0 candidates or return error */
    if (ret == 0) {
        ASSERT(actual_count == 0, "empty text should produce 0 candidates");
    }
    /* ret == -1 is also acceptable for empty input */

    gv_memory_candidates_free(candidates, actual_count);
    return 0;
}

/* Test 12: Extract with max_candidates = 1 limit */
static int test_extract_max_candidates_limit(void) {
    GV_MemoryCandidate candidates[1];
    memset(candidates, 0, sizeof(candidates));
    size_t actual_count = 0;

    const char *text = "User likes cats. User likes dogs. User likes birds. "
                       "User likes fish. User likes hamsters.";

    int ret = gv_memory_extract_candidates_from_text(
        text, "source", 0.0, candidates, 1, &actual_count);

    ASSERT(ret == 0, "extract with limit=1 should succeed");
    ASSERT(actual_count <= 1, "should not exceed max_candidates limit");

    gv_memory_candidates_free(candidates, actual_count);
    return 0;
}

/* Test 13: Extract from conversation with NULL conversation_id */
static int test_extract_conversation_null_id(void) {
    GV_MemoryCandidate candidates[5];
    memset(candidates, 0, sizeof(candidates));
    size_t actual_count = 0;

    const char *conversation = "User: I enjoy reading science fiction books.";

    int ret = gv_memory_extract_candidates_from_conversation(
        conversation, NULL, 0.1, candidates, 5, &actual_count);

    ASSERT(ret == 0, "extract with NULL conversation_id should succeed");

    gv_memory_candidates_free(candidates, actual_count);
    return 0;
}

/* Test 14: Extract with high threshold (fewer results) */
static int test_extract_high_threshold(void) {
    GV_MemoryCandidate candidates[10];
    memset(candidates, 0, sizeof(candidates));
    size_t actual_count = 0;

    const char *text = "The weather is nice today. User is a PhD in physics.";

    int ret = gv_memory_extract_candidates_from_text(
        text, "test", 0.99, candidates, 10, &actual_count);

    ASSERT(ret == 0, "extract with high threshold should succeed");
    /* High threshold should filter out most candidates */

    gv_memory_candidates_free(candidates, actual_count);
    return 0;
}

int main(void) {
    int failed = 0;
    int passed = 0;

    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_extract_from_text_simple",     test_extract_from_text_simple},
        {"test_extract_from_conversation",    test_extract_from_conversation},
        {"test_score_candidate",              test_score_candidate},
        {"test_score_candidate_varied",       test_score_candidate_varied},
        {"test_detect_type_fact",             test_detect_type_fact},
        {"test_detect_type_preference",       test_detect_type_preference},
        {"test_detect_type_relationship",     test_detect_type_relationship},
        {"test_detect_type_event",            test_detect_type_event},
        {"test_candidate_free_null",          test_candidate_free_null},
        {"test_candidate_free_allocated",     test_candidate_free_allocated},
        {"test_extract_empty_text",           test_extract_empty_text},
        {"test_extract_max_candidates_limit", test_extract_max_candidates_limit},
        {"test_extract_conversation_null_id", test_extract_conversation_null_id},
        {"test_extract_high_threshold",       test_extract_high_threshold},
    };

    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < num_tests; i++) {
        int result = tests[i].fn();
        if (result == 0) {
            printf("  OK   %s\n", tests[i].name);
            passed++;
        } else {
            printf("  FAILED %s\n", tests[i].name);
            failed++;
        }
    }

    printf("\n%d/%d tests passed\n", passed, num_tests);
    return failed > 0 ? 1 : 0;
}
