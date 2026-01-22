/**
 * @file test_importance.c
 * @brief Tests for the SOTA importance scoring algorithm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "gigavector/gv_importance.h"

#define TEST_PASS() printf("  ✓ %s passed\n", __func__)
#define TEST_FAIL(msg) do { printf("  ✗ %s failed: %s\n", __func__, msg); return 1; } while(0)
#define ASSERT_RANGE(val, min, max, name) \
    if ((val) < (min) || (val) > (max)) { \
        printf("    %s = %.4f, expected [%.4f, %.4f]\n", name, val, min, max); \
        TEST_FAIL("value out of range"); \
    }
#define ASSERT_GT(a, b, name) \
    if ((a) <= (b)) { \
        printf("    %s: %.4f should be > %.4f\n", name, a, b); \
        TEST_FAIL("comparison failed"); \
    }

/* ============================================================================
 * Content Analysis Tests
 * ============================================================================ */

int test_informativeness_empty(void) {
    double score = gv_importance_informativeness(NULL, 0);
    ASSERT_RANGE(score, 0.0, 0.0, "null input");

    score = gv_importance_informativeness("", 0);
    ASSERT_RANGE(score, 0.0, 0.0, "empty input");

    TEST_PASS();
    return 0;
}

int test_informativeness_simple(void) {
    const char *simple = "The cat sat on the mat.";
    double score = gv_importance_informativeness(simple, strlen(simple));
    ASSERT_RANGE(score, 0.1, 0.6, "simple sentence");

    TEST_PASS();
    return 0;
}

int test_informativeness_complex(void) {
    const char *complex = "The sophisticated algorithm demonstrates remarkable "
                          "computational efficiency through innovative parallelization "
                          "strategies and optimized memory management techniques.";
    double score = gv_importance_informativeness(complex, strlen(complex));
    ASSERT_RANGE(score, 0.3, 0.9, "complex sentence");

    /* Complex should score higher than simple */
    const char *simple = "I like cats.";
    double simple_score = gv_importance_informativeness(simple, strlen(simple));
    ASSERT_GT(score, simple_score, "complex vs simple");

    TEST_PASS();
    return 0;
}

int test_specificity_numbers(void) {
    const char *with_numbers = "The meeting is scheduled for January 15, 2025 at 3pm.";
    const char *without_numbers = "The meeting is scheduled for next week sometime.";

    double score_with = gv_importance_specificity(with_numbers, strlen(with_numbers));
    double score_without = gv_importance_specificity(without_numbers, strlen(without_numbers));

    ASSERT_GT(score_with, score_without, "numbers increase specificity");
    ASSERT_RANGE(score_with, 0.5, 1.0, "with numbers");

    TEST_PASS();
    return 0;
}

int test_specificity_proper_nouns(void) {
    const char *with_names = "John Smith met with Sarah Johnson at Microsoft headquarters.";
    const char *without_names = "someone met with another person at some company.";

    double score_with = gv_importance_specificity(with_names, strlen(with_names));
    double score_without = gv_importance_specificity(without_names, strlen(without_names));

    ASSERT_GT(score_with, score_without, "proper nouns increase specificity");

    TEST_PASS();
    return 0;
}

int test_specificity_vague_words(void) {
    /* Test statistical specificity detection:
     * - Pronoun-heavy text = less specific
     * - Text with numbers, dates, proper nouns = more specific */
    const char *vague = "It happened there and they did that with it.";
    const char *specific = "A database crash occurred in the production server at 3:45 PM.";

    double score_vague = gv_importance_specificity(vague, strlen(vague));
    double score_specific = gv_importance_specificity(specific, strlen(specific));

    ASSERT_GT(score_specific, score_vague, "specific vs vague");
    /* Vague text with many pronouns should score lower */
    ASSERT_RANGE(score_vague, 0.0, 0.55, "vague content");

    TEST_PASS();
    return 0;
}

int test_salience_emotional(void) {
    /* Test statistical salience detection based on structural features:
     * - Emphasis markers (!, ?) indicate emotional/important content
     * - ALL CAPS words indicate emphasis
     * Note: We do NOT use keyword lists - purely structural analysis */
    const char *emotional = "I absolutely LOVE this new feature! It makes me SO happy!";
    const char *neutral = "The feature has been implemented according to specifications.";

    double score_emotional = gv_importance_salience(emotional, strlen(emotional));
    double score_neutral = gv_importance_salience(neutral, strlen(neutral));

    ASSERT_GT(score_emotional, score_neutral, "emotional vs neutral");
    /* Lower threshold since we use structural features, not keyword matching */
    ASSERT_RANGE(score_emotional, 0.15, 1.0, "emotional content");

    TEST_PASS();
    return 0;
}

int test_salience_sentence_emphasis(void) {
    /* Test statistical salience detection based on structural emphasis:
     * - Multiple exclamation marks indicate urgency/importance
     * - Question marks indicate interactive content
     * - ALL CAPS words indicate emphasis
     * Note: Language-agnostic - works for any language with punctuation */
    const char *emphasized = "This is URGENT! Please respond IMMEDIATELY! Is this clear?";
    const char *plain = "This is urgent. Please respond immediately. Is this clear.";

    double score_emphasized = gv_importance_salience(emphasized, strlen(emphasized));
    double score_plain = gv_importance_salience(plain, strlen(plain));

    ASSERT_GT(score_emphasized, score_plain, "emphasized vs plain text");

    TEST_PASS();
    return 0;
}

int test_salience_important_markers(void) {
    /* Test statistical salience detection:
     * - Emphasis markers (!, ?, ALL CAPS) = more salient
     * - Future tense markers (will, going to) = more salient
     * - Superlatives (-est, -iest) = more salient */
    const char *important = "This is CRITICAL! You MUST back up the database before deployment!";
    const char *normal = "Back up the database before deployment.";

    double score_important = gv_importance_salience(important, strlen(important));
    double score_normal = gv_importance_salience(normal, strlen(normal));

    ASSERT_GT(score_important, score_normal, "important markers");

    TEST_PASS();
    return 0;
}

int test_entity_density(void) {
    const char *high_entity = "John Smith (john@example.com) works at Microsoft in Seattle, WA 98101.";
    const char *low_entity = "someone works at a company in a city.";

    double score_high = gv_importance_entity_density(high_entity, strlen(high_entity));
    double score_low = gv_importance_entity_density(low_entity, strlen(low_entity));

    ASSERT_GT(score_high, score_low, "entity density");
    ASSERT_RANGE(score_high, 0.3, 1.0, "high entity content");

    TEST_PASS();
    return 0;
}

int test_content_score_combined(void) {
    const char *high_quality = "My birthday is on March 15th, and I always celebrate "
                               "with my family at Giovanni's Italian Restaurant in downtown "
                               "Seattle. I love their margherita pizza!";

    const char *low_quality = "stuff happened.";

    double score_high = gv_importance_score_content(high_quality, strlen(high_quality));
    double score_low = gv_importance_score_content(low_quality, strlen(low_quality));

    ASSERT_GT(score_high, score_low, "combined content score");
    ASSERT_RANGE(score_high, 0.4, 1.0, "high quality content");
    ASSERT_RANGE(score_low, 0.0, 0.4, "low quality content");

    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Temporal Decay Tests
 * ============================================================================ */

int test_temporal_decay_immediate(void) {
    double decay = gv_importance_temporal_decay(NULL, 0);
    ASSERT_RANGE(decay, 0.99, 1.0, "zero age");

    TEST_PASS();
    return 0;
}

int test_temporal_decay_one_hour(void) {
    double decay = gv_importance_temporal_decay(NULL, 3600);  /* 1 hour in seconds */
    ASSERT_RANGE(decay, 0.9, 1.0, "one hour");

    TEST_PASS();
    return 0;
}

int test_temporal_decay_one_day(void) {
    double decay = gv_importance_temporal_decay(NULL, 86400);  /* 24 hours */
    ASSERT_RANGE(decay, 0.7, 0.95, "one day");

    TEST_PASS();
    return 0;
}

int test_temporal_decay_one_week(void) {
    double decay = gv_importance_temporal_decay(NULL, 604800);  /* 7 days */
    /* Default half-life is 168 hours (1 week), so decay should be ~0.5 */
    ASSERT_RANGE(decay, 0.4, 0.6, "one week (half-life)");

    TEST_PASS();
    return 0;
}

int test_temporal_decay_one_month(void) {
    double decay = gv_importance_temporal_decay(NULL, 2592000);  /* 30 days */
    /* Should be decayed but above minimum */
    ASSERT_RANGE(decay, 0.1, 0.3, "one month");

    TEST_PASS();
    return 0;
}

int test_temporal_decay_minimum(void) {
    double decay = gv_importance_temporal_decay(NULL, 31536000);  /* 1 year */
    /* Should hit minimum floor */
    ASSERT_RANGE(decay, 0.1, 0.15, "one year (minimum floor)");

    TEST_PASS();
    return 0;
}

int test_temporal_recency_boost(void) {
    /* Memories within recency window should score higher than those outside */
    /* Default recency boost window is 24 hours */
    double decay_12hours = gv_importance_temporal_decay(NULL, 43200);  /* 12 hours - within window */
    double decay_48hours = gv_importance_temporal_decay(NULL, 172800); /* 48 hours - outside window */

    /* 12 hours (within recency window) should score higher than 48 hours (outside) */
    ASSERT_GT(decay_12hours, decay_48hours, "recency boost");

    /* Also verify 12 hours gets some recency boost (should be > base decay) */
    ASSERT_RANGE(decay_12hours, 0.85, 1.0, "12 hour decay with recency");

    TEST_PASS();
    return 0;
}

int test_temporal_custom_config(void) {
    GV_TemporalDecayConfig config = {
        .half_life_hours = 24.0,      /* 1 day half-life */
        .min_decay_factor = 0.2,
        .recency_boost_hours = 1.0,
        .recency_boost_factor = 2.0
    };

    double decay = gv_importance_temporal_decay(&config, 86400);  /* 1 day */
    ASSERT_RANGE(decay, 0.45, 0.55, "custom config at half-life");

    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Access Pattern Tests
 * ============================================================================ */

int test_access_history_init(void) {
    GV_AccessHistory history;
    int result = gv_access_history_init(&history, 16);
    if (result != 0) TEST_FAIL("init failed");

    if (history.event_capacity != 16) TEST_FAIL("wrong capacity");
    if (history.event_count != 0) TEST_FAIL("wrong count");
    if (history.total_accesses != 0) TEST_FAIL("wrong total");

    gv_access_history_free(&history);
    TEST_PASS();
    return 0;
}

int test_access_record(void) {
    GV_AccessHistory history;
    gv_access_history_init(&history, 16);

    time_t now = time(NULL);
    int result = gv_importance_record_access(&history, now, 0.8, 0);
    if (result != 0) TEST_FAIL("record failed");

    if (history.total_accesses != 1) TEST_FAIL("wrong total");
    if (history.last_access != now) TEST_FAIL("wrong last_access");
    if (fabs(history.avg_relevance - 0.8) > 0.01) TEST_FAIL("wrong avg_relevance");

    gv_access_history_free(&history);
    TEST_PASS();
    return 0;
}

int test_access_multiple_records(void) {
    GV_AccessHistory history;
    gv_access_history_init(&history, 16);

    time_t now = time(NULL);
    gv_importance_record_access(&history, now - 3600, 0.6, 0);
    gv_importance_record_access(&history, now - 1800, 0.8, 0);
    gv_importance_record_access(&history, now, 1.0, 0);

    if (history.total_accesses != 3) TEST_FAIL("wrong total");
    if (history.event_count != 3) TEST_FAIL("wrong event count");

    /* Average of 0.6, 0.8, 1.0 = 0.8 */
    if (fabs(history.avg_relevance - 0.8) > 0.01) TEST_FAIL("wrong avg_relevance");

    gv_access_history_free(&history);
    TEST_PASS();
    return 0;
}

int test_access_score_no_history(void) {
    double score = gv_importance_access_score(NULL, NULL, time(NULL));
    ASSERT_RANGE(score, 0.0, 0.0, "no history");

    TEST_PASS();
    return 0;
}

int test_access_score_with_history(void) {
    GV_AccessHistory history;
    gv_access_history_init(&history, 16);

    time_t now = time(NULL);
    /* Multiple accesses at good intervals */
    gv_importance_record_access(&history, now - 172800, 0.7, 0);  /* 2 days ago */
    gv_importance_record_access(&history, now - 86400, 0.8, 0);   /* 1 day ago */
    gv_importance_record_access(&history, now, 0.9, 0);           /* now */

    double score = gv_importance_access_score(NULL, &history, now);
    ASSERT_RANGE(score, 0.3, 0.9, "with history");

    gv_access_history_free(&history);
    TEST_PASS();
    return 0;
}

int test_access_score_frequent_better(void) {
    GV_AccessHistory history1, history2;
    gv_access_history_init(&history1, 16);
    gv_access_history_init(&history2, 16);

    time_t now = time(NULL);

    /* History 1: single access */
    gv_importance_record_access(&history1, now, 0.8, 0);

    /* History 2: multiple accesses */
    for (int i = 0; i < 10; i++) {
        gv_importance_record_access(&history2, now - i * 3600, 0.8, 0);
    }

    double score1 = gv_importance_access_score(NULL, &history1, now);
    double score2 = gv_importance_access_score(NULL, &history2, now);

    ASSERT_GT(score2, score1, "frequent access scores higher");

    gv_access_history_free(&history1);
    gv_access_history_free(&history2);
    TEST_PASS();
    return 0;
}

int test_access_history_serialization(void) {
    GV_AccessHistory history;
    gv_access_history_init(&history, 16);

    time_t now = time(NULL);
    gv_importance_record_access(&history, now - 3600, 0.6, 0);
    gv_importance_record_access(&history, now, 0.8, 1);

    char *json = gv_access_history_serialize(&history);
    if (json == NULL) TEST_FAIL("serialize failed");

    GV_AccessHistory restored;
    int result = gv_access_history_deserialize(json, &restored);
    if (result != 0) {
        free(json);
        TEST_FAIL("deserialize failed");
    }

    if (restored.total_accesses != history.total_accesses) {
        free(json);
        gv_access_history_free(&restored);
        TEST_FAIL("total_accesses mismatch");
    }

    if (fabs(restored.avg_relevance - history.avg_relevance) > 0.01) {
        free(json);
        gv_access_history_free(&restored);
        TEST_FAIL("avg_relevance mismatch");
    }

    free(json);
    gv_access_history_free(&history);
    gv_access_history_free(&restored);
    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Full Importance Calculation Tests
 * ============================================================================ */

int test_importance_calculate_basic(void) {
    GV_ImportanceContext ctx = {
        .content = "My favorite restaurant is Giovanni's in Seattle.",
        .content_length = strlen("My favorite restaurant is Giovanni's in Seattle."),
        .creation_time = time(NULL) - 3600,  /* 1 hour ago */
        .current_time = time(NULL),
        .access_history = NULL,
        .relationship_count = 0,
        .incoming_links = 0,
        .outgoing_links = 0,
        .query_context = NULL,
        .semantic_similarity = 0.0
    };

    GV_ImportanceResult result;
    int ret = gv_importance_calculate(NULL, &ctx, &result);
    if (ret != 0) TEST_FAIL("calculate failed");

    ASSERT_RANGE(result.final_score, 0.3, 0.9, "basic importance");
    ASSERT_RANGE(result.content_score, 0.2, 0.8, "content score");
    ASSERT_RANGE(result.temporal_score, 0.8, 1.0, "temporal score (recent)");

    if (!(result.factors_used & GV_FACTOR_CONTENT)) TEST_FAIL("content factor missing");
    if (!(result.factors_used & GV_FACTOR_TEMPORAL)) TEST_FAIL("temporal factor missing");

    TEST_PASS();
    return 0;
}

int test_importance_calculate_with_access(void) {
    GV_AccessHistory history;
    gv_access_history_init(&history, 16);

    time_t now = time(NULL);
    gv_importance_record_access(&history, now - 86400, 0.8, 0);
    gv_importance_record_access(&history, now, 0.9, 0);

    GV_ImportanceContext ctx = {
        .content = "Important deadline for the project is next Friday.",
        .content_length = strlen("Important deadline for the project is next Friday."),
        .creation_time = now - 172800,  /* 2 days ago */
        .current_time = now,
        .access_history = &history,
        .relationship_count = 2,
        .incoming_links = 1,
        .outgoing_links = 1,
        .query_context = NULL,
        .semantic_similarity = 0.0
    };

    GV_ImportanceResult result;
    int ret = gv_importance_calculate(NULL, &ctx, &result);
    if (ret != 0) {
        gv_access_history_free(&history);
        TEST_FAIL("calculate failed");
    }

    ASSERT_RANGE(result.final_score, 0.4, 0.9, "importance with access");
    ASSERT_RANGE(result.access_score, 0.2, 0.8, "access score");
    ASSERT_RANGE(result.structural_score, 0.1, 0.8, "structural score");

    if (!(result.factors_used & GV_FACTOR_ACCESS)) TEST_FAIL("access factor missing");
    if (!(result.factors_used & GV_FACTOR_STRUCTURAL)) TEST_FAIL("structural factor missing");

    gv_access_history_free(&history);
    TEST_PASS();
    return 0;
}

int test_importance_calculate_with_query(void) {
    GV_ImportanceContext ctx = {
        .content = "My favorite pizza topping is pepperoni.",
        .content_length = strlen("My favorite pizza topping is pepperoni."),
        .creation_time = time(NULL) - 86400,
        .current_time = time(NULL),
        .access_history = NULL,
        .relationship_count = 0,
        .incoming_links = 0,
        .outgoing_links = 0,
        .query_context = "What pizza toppings do I like?",
        .semantic_similarity = 0.95
    };

    GV_ImportanceResult result;
    int ret = gv_importance_calculate(NULL, &ctx, &result);
    if (ret != 0) TEST_FAIL("calculate failed");

    /* High similarity should boost final score */
    ASSERT_RANGE(result.final_score, 0.5, 1.0, "importance with query context");
    if (!(result.factors_used & GV_FACTOR_QUERY)) TEST_FAIL("query factor missing");

    TEST_PASS();
    return 0;
}

int test_importance_old_vs_new(void) {
    time_t now = time(NULL);

    GV_ImportanceContext ctx_new = {
        .content = "Meeting scheduled for tomorrow at 2pm.",
        .content_length = strlen("Meeting scheduled for tomorrow at 2pm."),
        .creation_time = now - 60,  /* 1 minute ago */
        .current_time = now,
        .access_history = NULL,
        .relationship_count = 0,
        .incoming_links = 0,
        .outgoing_links = 0,
        .query_context = NULL,
        .semantic_similarity = 0.0
    };

    GV_ImportanceContext ctx_old = {
        .content = "Meeting scheduled for tomorrow at 2pm.",
        .content_length = strlen("Meeting scheduled for tomorrow at 2pm."),
        .creation_time = now - 2592000,  /* 30 days ago */
        .current_time = now,
        .access_history = NULL,
        .relationship_count = 0,
        .incoming_links = 0,
        .outgoing_links = 0,
        .query_context = NULL,
        .semantic_similarity = 0.0
    };

    GV_ImportanceResult result_new, result_old;
    gv_importance_calculate(NULL, &ctx_new, &result_new);
    gv_importance_calculate(NULL, &ctx_old, &result_old);

    ASSERT_GT(result_new.final_score, result_old.final_score, "new memory vs old");
    ASSERT_GT(result_new.temporal_score, result_old.temporal_score, "temporal scores");

    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Batch and Rerank Tests
 * ============================================================================ */

int test_importance_batch(void) {
    time_t now = time(NULL);

    GV_ImportanceContext contexts[3] = {
        {
            .content = "First memory content.",
            .content_length = strlen("First memory content."),
            .creation_time = now - 3600,
            .current_time = now,
        },
        {
            .content = "Second memory with more detailed information about preferences.",
            .content_length = strlen("Second memory with more detailed information about preferences."),
            .creation_time = now - 7200,
            .current_time = now,
        },
        {
            .content = "Third.",
            .content_length = strlen("Third."),
            .creation_time = now - 1800,
            .current_time = now,
        }
    };

    GV_ImportanceResult results[3];
    int scored = gv_importance_calculate_batch(NULL, contexts, results, 3);

    if (scored != 3) TEST_FAIL("batch scoring failed");

    for (int i = 0; i < 3; i++) {
        ASSERT_RANGE(results[i].final_score, 0.0, 1.0, "batch result range");
    }

    TEST_PASS();
    return 0;
}

int test_importance_rerank(void) {
    time_t now = time(NULL);

    /* Create contexts with varying similarity but different importance */
    GV_ImportanceContext contexts[4] = {
        {
            .content = "stuff.",  /* Low quality */
            .content_length = strlen("stuff."),
            .creation_time = now - 86400,
            .current_time = now,
            .semantic_similarity = 0.95  /* High similarity */
        },
        {
            .content = "My birthday is March 15th and I always celebrate at home.",  /* High quality */
            .content_length = strlen("My birthday is March 15th and I always celebrate at home."),
            .creation_time = now - 3600,
            .current_time = now,
            .semantic_similarity = 0.75  /* Medium similarity */
        },
        {
            .content = "I love Italian food, especially pasta carbonara!",  /* Medium quality, emotional */
            .content_length = strlen("I love Italian food, especially pasta carbonara!"),
            .creation_time = now - 60,
            .current_time = now,
            .semantic_similarity = 0.80  /* Medium similarity */
        },
        {
            .content = "random",  /* Very low quality */
            .content_length = strlen("random"),
            .creation_time = now - 604800,
            .current_time = now,
            .semantic_similarity = 0.90  /* High similarity */
        }
    };

    GV_ImportanceResult results[4];
    size_t indices[4];

    int ret = gv_importance_rerank(NULL, contexts, results, indices, 4, 0.5);
    if (ret != 0) TEST_FAIL("rerank failed");

    /* The high-quality, recent memory should rank well even with lower similarity */
    printf("    Rerank order: ");
    for (int i = 0; i < 4; i++) {
        printf("%zu ", indices[i]);
    }
    printf("\n");

    /* Verify ordering is valid */
    for (int i = 0; i < 4; i++) {
        if (indices[i] >= 4) TEST_FAIL("invalid index in rerank");
    }

    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Configuration Tests
 * ============================================================================ */

int test_config_default(void) {
    GV_ImportanceConfig config = gv_importance_config_default();

    /* Verify weights sum to approximately 1.0 */
    double weight_sum = config.weights.content_weight +
                        config.weights.temporal_weight +
                        config.weights.access_weight +
                        config.weights.salience_weight +
                        config.weights.structural_weight;

    if (fabs(weight_sum - 1.0) > 0.01) {
        printf("    Weight sum: %.4f\n", weight_sum);
        TEST_FAIL("weights don't sum to 1.0");
    }

    ASSERT_RANGE(config.temporal.half_life_hours, 1.0, 10000.0, "half_life");
    ASSERT_RANGE(config.base_score, 0.0, 1.0, "base_score");

    TEST_PASS();
    return 0;
}

int test_config_custom_weights(void) {
    GV_ImportanceConfig config = gv_importance_config_default();

    /* Override weights - focus on content only */
    config.weights.content_weight = 1.0;
    config.weights.temporal_weight = 0.0;
    config.weights.access_weight = 0.0;
    config.weights.salience_weight = 0.0;
    config.weights.structural_weight = 0.0;

    GV_ImportanceContext ctx = {
        .content = "High quality informative content with specific details.",
        .content_length = strlen("High quality informative content with specific details."),
        .creation_time = time(NULL) - 2592000,  /* 30 days ago - would normally decay */
        .current_time = time(NULL),
    };

    GV_ImportanceResult result;
    gv_importance_calculate(&config, &ctx, &result);

    /* With only content weight, temporal decay shouldn't affect final score much */
    /* (it will still be computed but not weighted) */
    ASSERT_RANGE(result.content_score, 0.3, 0.8, "content-only score");

    TEST_PASS();
    return 0;
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    printf("=== Importance Scoring Tests ===\n\n");

    int failures = 0;

    printf("Content Analysis Tests:\n");
    failures += test_informativeness_empty();
    failures += test_informativeness_simple();
    failures += test_informativeness_complex();
    failures += test_specificity_numbers();
    failures += test_specificity_proper_nouns();
    failures += test_specificity_vague_words();
    failures += test_salience_emotional();
    failures += test_salience_sentence_emphasis();
    failures += test_salience_important_markers();
    failures += test_entity_density();
    failures += test_content_score_combined();

    printf("\nTemporal Decay Tests:\n");
    failures += test_temporal_decay_immediate();
    failures += test_temporal_decay_one_hour();
    failures += test_temporal_decay_one_day();
    failures += test_temporal_decay_one_week();
    failures += test_temporal_decay_one_month();
    failures += test_temporal_decay_minimum();
    failures += test_temporal_recency_boost();
    failures += test_temporal_custom_config();

    printf("\nAccess Pattern Tests:\n");
    failures += test_access_history_init();
    failures += test_access_record();
    failures += test_access_multiple_records();
    failures += test_access_score_no_history();
    failures += test_access_score_with_history();
    failures += test_access_score_frequent_better();
    failures += test_access_history_serialization();

    printf("\nFull Importance Calculation Tests:\n");
    failures += test_importance_calculate_basic();
    failures += test_importance_calculate_with_access();
    failures += test_importance_calculate_with_query();
    failures += test_importance_old_vs_new();

    printf("\nBatch and Rerank Tests:\n");
    failures += test_importance_batch();
    failures += test_importance_rerank();

    printf("\nConfiguration Tests:\n");
    failures += test_config_default();
    failures += test_config_custom_weights();

    printf("\n=== Results ===\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed.\n", failures);
        return 1;
    }
}
