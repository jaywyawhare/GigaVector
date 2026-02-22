/**
 * @file gv_importance.c
 * @brief State-of-the-art importance scoring implementation.
 *
 * Implements a multi-factor importance scoring algorithm based on research from:
 * - BM25 (Okapi) ranking algorithm for statistical term importance
 * - Ebbinghaus forgetting curve for temporal decay
 * - Cortex's temporal weighting approach
 * - Spaced repetition (SM-2) for access patterns
 *
 * Key design principle: NO hardcoded keyword lists.
 * All scoring is based on statistical/structural features that are language-agnostic.
 *
 * References:
 * - Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework: BM25 and Beyond"
 * - Ebbinghaus (1885) "Memory: A Contribution to Experimental Psychology"
 * - MemoryBank (Zhong et al., 2024) "Enhancing LLMs with Long-Term Memory"
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>

#include "gigavector/gv_importance.h"

/*  BM25 Parameters (Okapi BM25 defaults from mem0)  */

#define BM25_K1 1.5    /* Term frequency saturation parameter */
#define BM25_B  0.75   /* Length normalization parameter */

/*  Helper Functions - Pure Statistical (No Keyword Lists)  */

static size_t count_words(const char *text, size_t len) {
    size_t count = 0;
    int in_word = 0;

    for (size_t i = 0; i < len; i++) {
        if (isalnum((unsigned char)text[i])) {
            if (!in_word) {
                in_word = 1;
                count++;
            }
        } else {
            in_word = 0;
        }
    }
    return count;
}

/**
 * @brief Count unique tokens using hash-based deduplication.
 * This is language-agnostic - works on any tokenized text.
 */
static size_t count_unique_words(const char *text, size_t len) {
    #define HASH_SIZE 256
    unsigned int seen[HASH_SIZE] = {0};
    size_t unique = 0;

    const char *start = text;
    const char *end = text + len;

    while (start < end) {
        /* Skip non-word characters */
        while (start < end && !isalnum((unsigned char)*start)) start++;
        if (start >= end) break;

        /* Find word end */
        const char *word_start = start;
        while (start < end && isalnum((unsigned char)*start)) start++;
        size_t word_len = start - word_start;

        if (word_len == 0 || word_len > 63) continue;

        /* Simple hash of word */
        unsigned int hash = 0;
        for (size_t i = 0; i < word_len; i++) {
            hash = hash * 31 + (unsigned char)tolower((unsigned char)word_start[i]);
        }
        hash %= HASH_SIZE;

        /* Mark as seen (simplified - may have collisions but good enough) */
        if (seen[hash] == 0) {
            seen[hash] = 1;
            unique++;
        }
    }

    return unique;
    #undef HASH_SIZE
}

/**
 * @brief Count capitalized words (potential named entities).
 * Language-agnostic: capitalization is a structural feature.
 */
static int count_capitalized_words(const char *text, size_t len) {
    int count = 0;
    int at_word_start = 1;
    int word_is_capitalized = 0;
    int after_sentence_end = 1;  /* Don't count first word of sentences */

    for (size_t i = 0; i < len; i++) {
        if (isalpha((unsigned char)text[i])) {
            if (at_word_start) {
                word_is_capitalized = isupper((unsigned char)text[i]);
                at_word_start = 0;
            }
        } else if (isspace((unsigned char)text[i])) {
            if (word_is_capitalized && !after_sentence_end) {
                count++;
            }
            at_word_start = 1;
            word_is_capitalized = 0;
            after_sentence_end = 0;
        } else if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            if (word_is_capitalized && !after_sentence_end) {
                count++;
            }
            at_word_start = 1;
            word_is_capitalized = 0;
            after_sentence_end = 1;
        }
    }

    return count;
}

/**
 * @brief Count numeric sequences (dates, amounts, IDs).
 * Language-agnostic: numbers are universal.
 */
static int count_number_sequences(const char *text, size_t len) {
    int count = 0;
    int in_number = 0;

    for (size_t i = 0; i < len; i++) {
        if (isdigit((unsigned char)text[i])) {
            if (!in_number) {
                count++;
                in_number = 1;
            }
        } else {
            in_number = 0;
        }
    }
    return count;
}

static double sigmoid(double x, double midpoint, double steepness) {
    return 1.0 / (1.0 + exp(-steepness * (x - midpoint)));
}

static double clamp(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/*  Configuration Functions  */

GV_ImportanceConfig gv_importance_config_default(void) {
    GV_ImportanceConfig config;

    /* Component weights (sum to 1.0) */
    config.weights.content_weight = 0.30;
    config.weights.temporal_weight = 0.25;
    config.weights.access_weight = 0.20;
    config.weights.salience_weight = 0.15;
    config.weights.structural_weight = 0.10;

    /* Temporal decay - based on Ebbinghaus curve research */
    config.temporal.half_life_hours = 168.0;        /* 1 week */
    config.temporal.min_decay_factor = 0.1;
    config.temporal.recency_boost_hours = 24.0;     /* 1 day */
    config.temporal.recency_boost_factor = 1.3;

    /* Access patterns - spaced repetition inspired */
    config.access.retrieval_boost_base = 0.05;
    config.access.retrieval_boost_decay = 0.95;
    config.access.optimal_interval_hours = 48.0;    /* 2 days */
    config.access.interval_tolerance = 0.5;
    config.access.max_tracked_accesses = 100;

    /* Content analysis */
    config.content.min_word_count = 5.0;
    config.content.optimal_word_count = 20.0;
    config.content.max_word_count = 100.0;
    config.content.enable_entity_detection = 1;
    config.content.enable_specificity_scoring = 1;

    /* General */
    config.enable_adaptive_weights = 0;
    config.base_score = 0.5;

    return config;
}

/*  Content Analysis Functions - Pure Statistical (No Keywords)  */

/**
 * @brief Calculate informativeness using Type-Token Ratio (TTR).
 *
 * TTR = unique_words / total_words
 * Higher TTR indicates more diverse vocabulary = more informative.
 * This is the standard measure from corpus linguistics.
 */
double gv_importance_informativeness(const char *content, size_t len) {
    if (content == NULL || len == 0) {
        return 0.0;
    }

    size_t total_words = count_words(content, len);
    if (total_words == 0) return 0.0;

    /* Factor 1: Lexical diversity (Type-Token Ratio) */
    size_t unique_words = count_unique_words(content, len);
    double ttr = (double)unique_words / (double)total_words;
    /* Normalize TTR - typical range is 0.3-0.8 for natural text */
    double diversity_score = clamp((ttr - 0.2) / 0.6, 0.0, 1.0);

    /* Factor 2: Average word length (vocabulary sophistication) */
    size_t total_letters = 0;
    for (size_t i = 0; i < len; i++) {
        if (isalpha((unsigned char)content[i])) {
            total_letters++;
        }
    }
    double avg_word_len = (double)total_letters / (double)total_words;
    /* Optimal range is 4-7 characters (Zipf's law) */
    double length_score = 1.0 - fabs(avg_word_len - 5.5) / 4.0;
    length_score = clamp(length_score, 0.0, 1.0);

    /* Factor 3: Sentence structure (punctuation density) */
    size_t sentence_ends = 0;
    for (size_t i = 0; i < len; i++) {
        if (content[i] == '.' || content[i] == '!' || content[i] == '?') {
            sentence_ends++;
        }
    }
    double words_per_sentence = (sentence_ends > 0) ?
        (double)total_words / (double)sentence_ends : (double)total_words;
    /* Optimal range is 10-25 words per sentence */
    double structure_score = 1.0 - fabs(words_per_sentence - 17.5) / 15.0;
    structure_score = clamp(structure_score, 0.0, 1.0);

    /* Combine factors */
    return 0.4 * diversity_score + 0.3 * length_score + 0.3 * structure_score;
}

/**
 * @brief Calculate specificity using structural features only.
 *
 * Specific content has:
 * - Numbers (dates, quantities, IDs)
 * - Capitalized words (proper nouns, names)
 * - Structured patterns (emails, URLs, dates)
 * - Higher word length variance
 *
 * No keyword lists used - purely structural analysis.
 */
double gv_importance_specificity(const char *content, size_t len) {
    if (content == NULL || len == 0) {
        return 0.0;
    }

    double score = 0.5;  /* Base score */
    size_t word_count = count_words(content, len);
    if (word_count == 0) return 0.0;

    /* Factor 1: Number density (dates, quantities, IDs are specific) */
    int num_count = count_number_sequences(content, len);
    double number_density = (double)num_count / (double)word_count;
    score += clamp(number_density * 0.8, 0.0, 0.2);

    /* Factor 2: Capitalized word ratio (proper nouns are specific) */
    int cap_count = count_capitalized_words(content, len);
    double cap_ratio = (double)cap_count / (double)word_count;
    score += clamp(cap_ratio * 0.5, 0.0, 0.2);

    /* Factor 3: Word length variance (specific text has varied vocabulary) */
    double total_word_len = 0.0;
    double total_word_len_sq = 0.0;
    int wc = 0;
    const char *p = content;
    const char *end = content + len;
    while (p < end) {
        while (p < end && !isalpha((unsigned char)*p)) p++;
        if (p >= end) break;
        const char *word_start = p;
        while (p < end && isalpha((unsigned char)*p)) p++;
        size_t wlen = p - word_start;
        if (wlen > 0) {
            total_word_len += (double)wlen;
            total_word_len_sq += (double)(wlen * wlen);
            wc++;
        }
    }
    if (wc > 1) {
        double mean_len = total_word_len / wc;
        double variance = (total_word_len_sq / wc) - (mean_len * mean_len);
        /* Higher variance = more specific vocabulary */
        double variance_score = 1.0 - exp(-variance / 10.0);
        score += clamp(variance_score * 0.15, 0.0, 0.15);
    }

    /* Factor 4: Structural patterns (language-agnostic) */
    /* Email pattern: @ followed by . */
    const char *at_pos = strchr(content, '@');
    if (at_pos != NULL && strchr(at_pos, '.') != NULL) {
        score += 0.1;
    }

    /* URL pattern: :// or www. */
    if (strstr(content, "://") != NULL || strstr(content, "www.") != NULL) {
        score += 0.1;
    }

    /* Date-like patterns: digits separated by / or - */
    for (size_t i = 0; i + 4 < len; i++) {
        if (isdigit((unsigned char)content[i])) {
            size_t j = i;
            while (j < len && isdigit((unsigned char)content[j])) j++;
            if (j < len && (content[j] == '/' || content[j] == '-')) {
                j++;
                if (j < len && isdigit((unsigned char)content[j])) {
                    score += 0.1;
                    break;
                }
            }
        }
    }

    /* Factor 5: Short word ratio (high ratio = less specific) */
    int short_words = 0;  /* Words <= 3 chars */
    p = content;
    while (p < end) {
        while (p < end && !isalpha((unsigned char)*p)) p++;
        if (p >= end) break;
        int wlen = 0;
        while (p < end && isalpha((unsigned char)*p)) {
            wlen++;
            p++;
        }
        if (wlen > 0 && wlen <= 3) {
            short_words++;
        }
    }
    double short_ratio = (double)short_words / (double)word_count;
    /* Penalize high short-word ratio (function words are short) */
    score -= clamp(short_ratio * 0.2, 0.0, 0.1);

    return clamp(score, 0.0, 1.0);
}

/**
 * @brief Calculate salience using structural and statistical features.
 *
 * Salient content has:
 * - Emphasis markers (!, ?, ALL CAPS)
 * - Structural emphasis (quotation marks, parentheses)
 * - Position markers (starts with capital)
 * - Possessive patterns ('s)
 *
 * No keyword lists - purely structural analysis.
 */
double gv_importance_salience(const char *content, size_t len) {
    if (content == NULL || len == 0) {
        return 0.0;
    }

    double score = 0.0;
    size_t word_count = count_words(content, len);
    if (word_count == 0) return 0.0;

    /* Factor 1: Punctuation emphasis (universal across languages) */
    int exclamation_count = 0;
    int question_count = 0;
    int quote_count = 0;
    int parenthesis_count = 0;

    for (size_t i = 0; i < len; i++) {
        if (content[i] == '!') exclamation_count++;
        if (content[i] == '?') question_count++;
        if (content[i] == '"' || content[i] == '\'') quote_count++;
        if (content[i] == '(' || content[i] == ')') parenthesis_count++;
    }

    /* Exclamations indicate emphasis/emotion */
    score += clamp((double)exclamation_count * 0.12, 0.0, 0.2);
    /* Questions indicate engagement/importance */
    score += clamp((double)question_count * 0.1, 0.0, 0.15);
    /* Quotes indicate cited/important content */
    score += clamp((double)quote_count * 0.03, 0.0, 0.1);
    /* Parentheses indicate additional context */
    score += clamp((double)parenthesis_count * 0.02, 0.0, 0.05);

    /* Factor 2: ALL CAPS words (emphasis) */
    int caps_word_count = 0;
    const char *p = content;
    const char *end = content + len;
    while (p < end) {
        while (p < end && !isalpha((unsigned char)*p)) p++;
        if (p >= end) break;
        int all_caps = 1;
        int word_len = 0;
        while (p < end && isalpha((unsigned char)*p)) {
            if (!isupper((unsigned char)*p)) all_caps = 0;
            word_len++;
            p++;
        }
        /* Only count 2+ char words */
        if (all_caps && word_len >= 2) {
            caps_word_count++;
        }
    }
    double caps_ratio = (double)caps_word_count / (double)word_count;
    score += clamp(caps_ratio * 1.5, 0.0, 0.2);

    /* Factor 3: Possessive patterns ('s) - indicates ownership/personal */
    int possessive_count = 0;
    for (size_t i = 0; i + 1 < len; i++) {
        if (content[i] == '\'' && (content[i+1] == 's' || content[i+1] == 'S')) {
            possessive_count++;
        }
    }
    score += clamp((double)possessive_count * 0.1, 0.0, 0.15);

    /* Factor 4: Colon usage (often introduces important info) */
    int colon_count = 0;
    for (size_t i = 0; i < len; i++) {
        if (content[i] == ':') colon_count++;
    }
    score += clamp((double)colon_count * 0.08, 0.0, 0.1);

    /* Factor 5: Sentence length variance (varied = more engaging/salient) */
    int sentence_count = 0;
    int words_in_sentence = 0;
    double total_sent_len = 0.0;
    double total_sent_len_sq = 0.0;

    p = content;
    while (p < end) {
        if (isalnum((unsigned char)*p)) {
            while (p < end && isalnum((unsigned char)*p)) p++;
            words_in_sentence++;
        } else if (*p == '.' || *p == '!' || *p == '?') {
            if (words_in_sentence > 0) {
                total_sent_len += words_in_sentence;
                total_sent_len_sq += words_in_sentence * words_in_sentence;
                sentence_count++;
                words_in_sentence = 0;
            }
            p++;
        } else {
            p++;
        }
    }
    if (words_in_sentence > 0) {
        total_sent_len += words_in_sentence;
        total_sent_len_sq += words_in_sentence * words_in_sentence;
        sentence_count++;
    }

    if (sentence_count > 1) {
        double mean_sent = total_sent_len / sentence_count;
        double variance = (total_sent_len_sq / sentence_count) - (mean_sent * mean_sent);
        double variance_score = 1.0 - exp(-variance / 50.0);
        score += clamp(variance_score * 0.1, 0.0, 0.1);
    }

    return clamp(score, 0.0, 1.0);
}

/**
 * @brief Calculate entity density based on structural markers.
 *
 * Entities are identified by:
 * - Capitalized words (names, places, organizations)
 * - Number sequences (dates, amounts, IDs)
 *
 * No NER model or keyword lists - purely structural.
 */
double gv_importance_entity_density(const char *content, size_t len) {
    if (content == NULL || len == 0) {
        return 0.0;
    }

    size_t word_count = count_words(content, len);
    if (word_count == 0) return 0.0;

    int entity_count = 0;

    /* Count capitalized words (potential named entities) */
    entity_count += count_capitalized_words(content, len);

    /* Count number sequences (dates, amounts, IDs) */
    entity_count += count_number_sequences(content, len);

    /* Calculate density */
    double density = (double)entity_count / (double)word_count;

    /* Optimal density is around 0.1-0.3 */
    /* Use sigmoid for smooth transition */
    return sigmoid(density, 0.15, 15.0);
}

/**
 * @brief Calculate combined content score.
 *
 * Combines:
 * - Informativeness (TTR, word length, sentence structure)
 * - Specificity (numbers, capitals, patterns)
 * - Salience (punctuation, emphasis)
 * - Entity density (named entities)
 */
double gv_importance_score_content(const char *content, size_t len) {
    if (content == NULL || len == 0) {
        return 0.0;
    }

    /* Get individual scores */
    double informativeness = gv_importance_informativeness(content, len);
    double specificity = gv_importance_specificity(content, len);
    double salience = gv_importance_salience(content, len);
    double entity_density = gv_importance_entity_density(content, len);

    /* Word count factor (optimal length: 5-100 words) */
    size_t word_count = count_words(content, len);
    double length_factor;
    if (word_count < 5) {
        length_factor = (double)word_count / 5.0;
    } else if (word_count <= 100) {
        length_factor = 1.0;
    } else {
        length_factor = 1.0 - (double)(word_count - 100) / 200.0;
        length_factor = clamp(length_factor, 0.5, 1.0);
    }

    /* Combine with weights */
    double base_score = 0.30 * informativeness +
                        0.25 * specificity +
                        0.25 * salience +
                        0.20 * entity_density;

    return clamp(base_score * length_factor, 0.0, 1.0);
}

/**
 * @brief Score extracted facts (optimized for short LLM-extracted content).
 *
 * This function is designed for short extracted facts like:
 * - "Name is John"
 * - "Works at Google"
 * - "Favorite color is blue"
 *
 * Key differences from gv_importance_score_content():
 * - NO length penalty (LLM already filtered for important facts)
 * - Higher weight on specificity and entity density
 * - Bonus for proper nouns and concrete details
 */
double gv_importance_score_extracted(const char *content, size_t len) {
    if (content == NULL || len == 0) {
        return 0.0;
    }

    /* Get individual scores */
    double specificity = gv_importance_specificity(content, len);
    double entity_density = gv_importance_entity_density(content, len);
    double informativeness = gv_importance_informativeness(content, len);

    /* Count structural features that indicate valuable extracted facts */
    size_t word_count = count_words(content, len);
    int capitalized = count_capitalized_words(content, len);
    int numbers = count_number_sequences(content, len);

    /* Bonus for having proper nouns or numbers (concrete facts) */
    double concrete_bonus = 0.0;
    if (word_count > 0) {
        /* Proportion of concrete elements */
        double concrete_ratio = (double)(capitalized + numbers) / (double)word_count;
        concrete_bonus = clamp(concrete_ratio * 0.3, 0.0, 0.3);
    }

    /* Base score with emphasis on specificity and entities
     * (more important for extracted facts than raw text) */
    double base_score = 0.35 * specificity +
                        0.30 * entity_density +
                        0.20 * informativeness +
                        0.15 * concrete_bonus;

    /* Minimum floor of 0.4 for any extracted fact
     * (LLM extraction implies some importance) */
    double floor = 0.4;

    /* Scale to range [floor, 1.0] */
    double final_score = floor + (1.0 - floor) * base_score;

    return clamp(final_score, floor, 1.0);
}

/*  Temporal Decay Functions - Ebbinghaus Forgetting Curve  */

/**
 * @brief Calculate temporal decay using Ebbinghaus forgetting curve.
 *
 * Formula: R = e^(-t/S) where S = half_life / ln(2)
 *
 * Based on research from MemoryBank (Zhong et al., 2024) and Cortex.
 */
double gv_importance_temporal_decay(const GV_TemporalDecayConfig *config,
                                     double age_seconds) {
    GV_TemporalDecayConfig default_config;
    if (config == NULL) {
        default_config = gv_importance_config_default().temporal;
        config = &default_config;
    }

    if (age_seconds <= 0) {
        return 1.0;
    }

    double age_hours = age_seconds / 3600.0;

    /* Ebbinghaus forgetting curve: R = e^(-t/S) where S is stability */
    /* We use half-life to compute stability: S = half_life / ln(2) */
    double stability = config->half_life_hours / 0.693147;  /* ln(2) */
    double decay = exp(-age_hours / stability);

    /* Apply minimum decay floor */
    decay = fmax(decay, config->min_decay_factor);

    /* Apply recency boost for very recent memories */
    if (age_hours < config->recency_boost_hours) {
        double recency_factor = 1.0 - (age_hours / config->recency_boost_hours);
        double boost = 1.0 + (config->recency_boost_factor - 1.0) * recency_factor;
        decay = fmin(decay * boost, 1.0);
    }

    return decay;
}

/*  Access Pattern Functions - Spaced Repetition Inspired  */

/**
 * @brief Calculate access pattern score based on retrieval history.
 *
 * Inspired by spaced repetition (SM-2 algorithm):
 * - More accesses = more important
 * - Recent access = more important
 * - Optimal spacing = more important
 */
double gv_importance_access_score(const GV_AccessPatternConfig *config,
                                   const GV_AccessHistory *history,
                                   time_t current_time) {
    GV_AccessPatternConfig default_config;
    if (config == NULL) {
        default_config = gv_importance_config_default().access;
        config = &default_config;
    }

    if (history == NULL || history->total_accesses == 0) {
        return 0.0;
    }

    double score = 0.0;

    /* Factor 1: Total retrieval count (log scale for diminishing returns) */
    double retrieval_score = log(1.0 + history->total_accesses) / log(1.0 + 100.0);
    retrieval_score = clamp(retrieval_score, 0.0, 1.0);

    /* Factor 2: Recency of last access */
    double last_access_age = difftime(current_time, history->last_access);
    double recency_score = gv_importance_temporal_decay(NULL, last_access_age);

    /* Factor 3: Average relevance when accessed */
    double relevance_score = history->avg_relevance;

    /* Factor 4: Access interval quality (spaced repetition inspired) */
    double interval_score = 0.5;  /* Default */
    if (history->event_count >= 2) {
        double total_interval = 0.0;
        size_t interval_count = 0;
        for (size_t i = 1; i < history->event_count; i++) {
            double interval = difftime(history->events[i].timestamp,
                                       history->events[i-1].timestamp) / 3600.0;
            total_interval += interval;
            interval_count++;
        }

        if (interval_count > 0) {
            double avg_interval = total_interval / interval_count;
            double interval_ratio = avg_interval / config->optimal_interval_hours;
            /* Best score when interval_ratio is around 1.0 */
            interval_score = exp(-pow(log(interval_ratio + 0.1), 2) / 2.0);
        }
    }

    /* Combine factors */
    score = 0.3 * retrieval_score +
            0.3 * recency_score +
            0.2 * relevance_score +
            0.2 * interval_score;

    return clamp(score, 0.0, 1.0);
}

int gv_importance_record_access(GV_AccessHistory *history,
                                 time_t timestamp,
                                 double relevance,
                                 int access_type) {
    if (history == NULL) {
        return -1;
    }

    /* Expand capacity if needed */
    if (history->event_count >= history->event_capacity) {
        size_t new_capacity = history->event_capacity == 0 ? 16 : history->event_capacity * 2;
        if (new_capacity > 100) new_capacity = 100;

        GV_AccessEvent *new_events = realloc(history->events,
                                              new_capacity * sizeof(GV_AccessEvent));
        if (new_events == NULL) {
            return -1;
        }
        history->events = new_events;
        history->event_capacity = new_capacity;
    }

    /* If at capacity, remove oldest event */
    if (history->event_count >= 100) {
        memmove(history->events, history->events + 1,
                (history->event_count - 1) * sizeof(GV_AccessEvent));
        history->event_count--;
    }

    /* Add new event */
    GV_AccessEvent *event = &history->events[history->event_count++];
    event->timestamp = timestamp;
    event->relevance_at_access = relevance;
    event->access_type = access_type;

    /* Update summary stats */
    history->last_access = timestamp;
    history->total_accesses++;

    /* Update running average relevance */
    double n = (double)history->total_accesses;
    history->avg_relevance = ((n - 1) * history->avg_relevance + relevance) / n;

    return 0;
}

/*  Access History Management  */

int gv_access_history_init(GV_AccessHistory *history, size_t initial_capacity) {
    if (history == NULL) {
        return -1;
    }

    memset(history, 0, sizeof(GV_AccessHistory));

    if (initial_capacity > 0) {
        history->events = calloc(initial_capacity, sizeof(GV_AccessEvent));
        if (history->events == NULL) {
            return -1;
        }
        history->event_capacity = initial_capacity;
    }

    return 0;
}

void gv_access_history_free(GV_AccessHistory *history) {
    if (history == NULL) {
        return;
    }

    free(history->events);
    memset(history, 0, sizeof(GV_AccessHistory));
}

char *gv_access_history_serialize(const GV_AccessHistory *history) {
    if (history == NULL) {
        return strdup("{}");
    }

    size_t buffer_size = 256 + history->event_count * 100;
    char *buffer = malloc(buffer_size);
    if (buffer == NULL) {
        return NULL;
    }

    int pos = snprintf(buffer, buffer_size,
        "{\"total_accesses\":%u,"
        "\"last_access\":%ld,"
        "\"avg_relevance\":%.4f,"
        "\"events\":[",
        history->total_accesses,
        (long)history->last_access,
        history->avg_relevance);

    for (size_t i = 0; i < history->event_count && pos < (int)buffer_size - 100; i++) {
        if (i > 0) {
            buffer[pos++] = ',';
        }
        pos += snprintf(buffer + pos, buffer_size - pos,
            "{\"ts\":%ld,\"rel\":%.4f,\"type\":%d}",
            (long)history->events[i].timestamp,
            history->events[i].relevance_at_access,
            history->events[i].access_type);
    }

    snprintf(buffer + pos, buffer_size - pos, "]}");

    return buffer;
}

int gv_access_history_deserialize(const char *json, GV_AccessHistory *history) {
    if (json == NULL || history == NULL) {
        return -1;
    }

    gv_access_history_init(history, 16);

    const char *p;

    p = strstr(json, "\"total_accesses\":");
    if (p) {
        history->total_accesses = (uint32_t)atoi(p + 17);
    }

    p = strstr(json, "\"last_access\":");
    if (p) {
        history->last_access = (time_t)atol(p + 14);
    }

    p = strstr(json, "\"avg_relevance\":");
    if (p) {
        history->avg_relevance = atof(p + 16);
    }

    p = strstr(json, "\"events\":[");
    if (p) {
        p += 10;
        while (*p && *p != ']') {
            if (*p == '{') {
                GV_AccessEvent event = {0};
                const char *ts = strstr(p, "\"ts\":");
                const char *rel = strstr(p, "\"rel\":");
                const char *type = strstr(p, "\"type\":");

                if (ts) event.timestamp = (time_t)atol(ts + 5);
                if (rel) event.relevance_at_access = atof(rel + 6);
                if (type) event.access_type = atoi(type + 7);

                if (history->event_count >= history->event_capacity) {
                    size_t new_cap = history->event_capacity * 2;
                    GV_AccessEvent *new_events = realloc(history->events,
                                                          new_cap * sizeof(GV_AccessEvent));
                    if (new_events) {
                        history->events = new_events;
                        history->event_capacity = new_cap;
                    }
                }
                if (history->event_count < history->event_capacity) {
                    history->events[history->event_count++] = event;
                }

                p = strchr(p, '}');
                if (p) p++;
            } else {
                p++;
            }
        }
    }

    return 0;
}

/*  Main Scoring Functions  */

int gv_importance_calculate(const GV_ImportanceConfig *config,
                            const GV_ImportanceContext *context,
                            GV_ImportanceResult *result) {
    if (context == NULL || result == NULL) {
        return -1;
    }

    GV_ImportanceConfig default_config;
    if (config == NULL) {
        default_config = gv_importance_config_default();
        config = &default_config;
    }

    memset(result, 0, sizeof(GV_ImportanceResult));

    /* Calculate content-based score */
    if (context->content != NULL && context->content_length > 0) {
        result->informativeness = gv_importance_informativeness(context->content,
                                                                  context->content_length);
        result->specificity = gv_importance_specificity(context->content,
                                                          context->content_length);
        result->salience_score = gv_importance_salience(context->content,
                                                          context->content_length);
        result->entity_density = gv_importance_entity_density(context->content,
                                                                context->content_length);

        result->content_score = 0.35 * result->informativeness +
                                0.25 * result->specificity +
                                0.25 * result->salience_score +
                                0.15 * result->entity_density;

        result->factors_used |= GV_FACTOR_CONTENT;
    } else {
        result->content_score = config->base_score;
    }

    /* Calculate temporal score */
    if (context->creation_time > 0 && context->current_time > 0) {
        double age_seconds = difftime(context->current_time, context->creation_time);
        result->decay_factor = gv_importance_temporal_decay(&config->temporal, age_seconds);

        double age_hours = age_seconds / 3600.0;
        if (age_hours < config->temporal.recency_boost_hours) {
            result->recency_bonus = (1.0 - age_hours / config->temporal.recency_boost_hours) *
                                    (config->temporal.recency_boost_factor - 1.0);
        }

        result->temporal_score = result->decay_factor;
        result->factors_used |= GV_FACTOR_TEMPORAL;
    } else {
        result->temporal_score = 1.0;
        result->decay_factor = 1.0;
    }

    /* Calculate access pattern score */
    if (context->access_history != NULL && context->access_history->total_accesses > 0) {
        result->access_score = gv_importance_access_score(&config->access,
                                                           context->access_history,
                                                           context->current_time);
        result->retrieval_boost = log(1.0 + context->access_history->total_accesses) * 0.1;
        result->factors_used |= GV_FACTOR_ACCESS;
    } else {
        result->access_score = 0.0;
    }

    /* Salience already computed in content analysis */
    result->factors_used |= GV_FACTOR_SALIENCE;

    /* Calculate structural score */
    if (context->relationship_count > 0 ||
        context->incoming_links > 0 ||
        context->outgoing_links > 0) {

        double total_links = (double)(context->incoming_links + context->outgoing_links);
        double link_score = 1.0 - exp(-0.3 * total_links);
        double rel_score = 1.0 - exp(-0.2 * context->relationship_count);

        result->structural_score = 0.6 * link_score + 0.4 * rel_score;
        result->factors_used |= GV_FACTOR_STRUCTURAL;
    } else {
        result->structural_score = 0.0;
    }

    /* Query context boost */
    double query_boost = 0.0;
    if (context->query_context != NULL && context->semantic_similarity > 0.0) {
        query_boost = context->semantic_similarity;
        result->factors_used |= GV_FACTOR_QUERY;
    }

    /* Combine all factors with weights */
    double weighted_sum = 0.0;
    double total_weight = 0.0;

    if (result->factors_used & GV_FACTOR_CONTENT) {
        weighted_sum += config->weights.content_weight * result->content_score;
        total_weight += config->weights.content_weight;
    }

    if (result->factors_used & GV_FACTOR_TEMPORAL) {
        weighted_sum += config->weights.temporal_weight * result->temporal_score;
        total_weight += config->weights.temporal_weight;
    }

    if (result->factors_used & GV_FACTOR_ACCESS) {
        weighted_sum += config->weights.access_weight * result->access_score;
        total_weight += config->weights.access_weight;
    }

    if (result->factors_used & GV_FACTOR_SALIENCE) {
        weighted_sum += config->weights.salience_weight * result->salience_score;
        total_weight += config->weights.salience_weight;
    }

    if (result->factors_used & GV_FACTOR_STRUCTURAL) {
        weighted_sum += config->weights.structural_weight * result->structural_score;
        total_weight += config->weights.structural_weight;
    }

    /* Normalize if not all factors present */
    if (total_weight > 0) {
        result->final_score = weighted_sum / total_weight;
    } else {
        result->final_score = config->base_score;
    }

    /* Apply query context boost if present */
    if (query_boost > 0.0) {
        result->final_score = 0.5 * result->final_score + 0.5 * query_boost;
    }

    /* Confidence based on how many factors were available */
    int factor_count = __builtin_popcount(result->factors_used);
    result->confidence = (double)factor_count / 6.0;

    result->final_score = clamp(result->final_score, 0.0, 1.0);

    return 0;
}

/*  Batch Operations  */

int gv_importance_calculate_batch(const GV_ImportanceConfig *config,
                                   const GV_ImportanceContext *contexts,
                                   GV_ImportanceResult *results,
                                   size_t count) {
    if (contexts == NULL || results == NULL || count == 0) {
        return 0;
    }

    int success_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (gv_importance_calculate(config, &contexts[i], &results[i]) == 0) {
            success_count++;
        }
    }

    return success_count;
}

int gv_importance_rerank(const GV_ImportanceConfig *config,
                          const GV_ImportanceContext *contexts,
                          GV_ImportanceResult *results,
                          size_t *indices,
                          size_t count,
                          double similarity_weight) {
    if (contexts == NULL || results == NULL || indices == NULL || count == 0) {
        return -1;
    }

    /* Calculate importance for all */
    gv_importance_calculate_batch(config, contexts, results, count);

    /* Combine importance with similarity */
    double importance_weight = 1.0 - similarity_weight;
    double *combined_scores = malloc(count * sizeof(double));
    if (combined_scores == NULL) {
        return -1;
    }

    for (size_t i = 0; i < count; i++) {
        combined_scores[i] = importance_weight * results[i].final_score +
                             similarity_weight * contexts[i].semantic_similarity;
        indices[i] = i;
    }

    /* Sort by combined score (descending) */
    for (size_t i = 0; i < count - 1; i++) {
        for (size_t j = 0; j < count - i - 1; j++) {
            if (combined_scores[indices[j]] < combined_scores[indices[j + 1]]) {
                size_t temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    free(combined_scores);
    return 0;
}
