#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#include "gigavector/gv_memory_extraction.h"
#include "gigavector/gv_importance.h"
#include "gigavector/gv_llm.h"
#include "gigavector/gv_json.h"

static int is_sentence_end(char c) {
    return c == '.' || c == '!' || c == '?' || c == '\n';
}

/**
 * @brief Calculate importance score using the SOTA multi-factor algorithm.
 *
 * This replaces the old hardcoded heuristics with the comprehensive
 * importance scoring system that considers:
 * - Content informativeness and lexical diversity
 * - Specificity (numbers, proper nouns, concrete details)
 * - Salience (emotional markers, personal relevance)
 * - Entity density (named entities, structured data)
 */
static double calculate_importance(const char *content, size_t len) {
    if (content == NULL || len == 0) {
        return 0.0;
    }

    /* Use the new SOTA importance scoring algorithm */
    return gv_importance_score_content(content, len);
}

static GV_MemoryType detect_type_from_keywords(const char *content) {
    if (content == NULL) {
        return GV_MEMORY_TYPE_FACT;
    }
    
    char lower_content[1024];
    size_t len = strlen(content);
    if (len >= sizeof(lower_content)) {
        len = sizeof(lower_content) - 1;
    }
    
    for (size_t i = 0; i < len; i++) {
        lower_content[i] = tolower(content[i]);
    }
    lower_content[len] = '\0';
    
    if (strstr(lower_content, "prefer") || strstr(lower_content, "like") ||
        strstr(lower_content, "favorite") || strstr(lower_content, "dislike")) {
        return GV_MEMORY_TYPE_PREFERENCE;
    }
    
    if (strstr(lower_content, "know") || strstr(lower_content, "friend") ||
        strstr(lower_content, "colleague") || strstr(lower_content, "relationship")) {
        return GV_MEMORY_TYPE_RELATIONSHIP;
    }
    
    if (strstr(lower_content, "happened") || strstr(lower_content, "event") ||
        strstr(lower_content, "occurred") || strstr(lower_content, "meeting")) {
        return GV_MEMORY_TYPE_EVENT;
    }
    
    return GV_MEMORY_TYPE_FACT;
}

int gv_memory_extract_candidates_from_conversation(const char *conversation,
                                                    const char *conversation_id,
                                                    double threshold,
                                                    GV_MemoryCandidate *candidates,
                                                    size_t max_candidates,
                                                    size_t *actual_count) {
    if (conversation == NULL || candidates == NULL || actual_count == NULL) {
        return -1;
    }
    
    *actual_count = 0;
    
    size_t len = strlen(conversation);
    if (len == 0) {
        return 0;
    }
    
    const char *start = conversation;
    const char *end = conversation + len;
    size_t candidate_idx = 0;
    
    while (start < end && candidate_idx < max_candidates) {
        while (start < end && (isspace(*start) || ispunct(*start))) {
            start++;
        }
        
        if (start >= end) {
            break;
        }
        
        const char *sentence_start = start;
        const char *sentence_end = start;
        
        while (sentence_end < end && !is_sentence_end(*sentence_end)) {
            sentence_end++;
        }
        
        if (sentence_end > sentence_start) {
            size_t sentence_len = sentence_end - sentence_start;
            if (sentence_len > 10) {
                char *sentence = (char *)malloc(sentence_len + 1);
                if (sentence != NULL) {
                    memcpy(sentence, sentence_start, sentence_len);
                    sentence[sentence_len] = '\0';
                    
                    double importance = calculate_importance(sentence, sentence_len);
                    
                    if (importance >= threshold) {
                        candidates[candidate_idx].content = sentence;
                        candidates[candidate_idx].importance_score = importance;
                        candidates[candidate_idx].memory_type = detect_type_from_keywords(sentence);
                        candidates[candidate_idx].extraction_context = conversation_id ?
                            strdup(conversation_id) : NULL;
                        candidate_idx++;
                    } else {
                        free(sentence);
                    }
                }
            }
        }
        
        start = sentence_end + 1;
    }
    
    *actual_count = candidate_idx;
    return 0;
}

int gv_memory_extract_candidates_from_text(const char *text,
                                            const char *source,
                                            double threshold,
                                            GV_MemoryCandidate *candidates,
                                            size_t max_candidates,
                                            size_t *actual_count) {
    if (text == NULL || candidates == NULL || actual_count == NULL) {
        return -1;
    }
    
    return gv_memory_extract_candidates_from_conversation(text, source, threshold,
                                                          candidates, max_candidates, actual_count);
}

double gv_memory_score_candidate(const GV_MemoryCandidate *candidate) {
    if (candidate == NULL) {
        return 0.0;
    }
    return candidate->importance_score;
}

GV_MemoryType gv_memory_detect_type(const char *content) {
    if (content == NULL) {
        return GV_MEMORY_TYPE_FACT;
    }
    return detect_type_from_keywords(content);
}

void gv_memory_candidate_free(GV_MemoryCandidate *candidate) {
    if (candidate == NULL) {
        return;
    }
    
    free(candidate->content);
    free(candidate->extraction_context);
    memset(candidate, 0, sizeof(GV_MemoryCandidate));
}

void gv_memory_candidates_free(GV_MemoryCandidate *candidates, size_t count) {
    if (candidates == NULL) {
        return;
    }
    
    for (size_t i = 0; i < count; i++) {
        gv_memory_candidate_free(&candidates[i]);
    }
}

static const char *get_default_user_extraction_prompt(void) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char date_str[32];
    strftime(date_str, sizeof(date_str), "%Y-%m-%d", tm_info);
    
    static char prompt[4096];
    snprintf(prompt, sizeof(prompt),
        "You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences. "
        "Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts. "
        "This allows for easy retrieval and personalization in future interactions.\n\n"
        "# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES. DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.\n"
        "# [IMPORTANT]: YOU WILL BE PENALIZED IF YOU INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.\n\n"
        "Types of Information to Remember:\n"
        "1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.\n"
        "2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.\n"
        "3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.\n"
        "4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.\n"
        "5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.\n"
        "6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information.\n"
        "7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares.\n\n"
        "Examples:\n"
        "User: Hi.\n"
        "Assistant: Hello! How can I help today?\n"
        "Output: {\"facts\": []}\n\n"
        "User: Hi, my name is John. I am a software engineer.\n"
        "Assistant: Nice to meet you, John!\n"
        "Output: {\"facts\": [\"Name is John\", \"Is a Software engineer\"]}\n\n"
        "User: My favourite movies are Inception and Interstellar.\n"
        "Assistant: Great choices!\n"
        "Output: {\"facts\": [\"Favourite movies are Inception and Interstellar\"]}\n\n"
        "Return the facts in JSON format with a \"facts\" key containing an array of strings.\n"
        "Today's date is %s.\n"
        "If you do not find anything relevant, return an empty list.\n"
        "Extract facts ONLY from user messages, not assistant messages.\n\n"
        "Conversation:\n", date_str);
    return prompt;
}

static const char *get_default_agent_extraction_prompt(void) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char date_str[32];
    strftime(date_str, sizeof(date_str), "%Y-%m-%d", tm_info);
    
    static char prompt[4096];
    snprintf(prompt, sizeof(prompt),
        "You are an Assistant Information Organizer, specialized in accurately storing facts, preferences, and characteristics about the AI assistant from conversations. "
        "Your primary role is to extract relevant pieces of information about the assistant from conversations and organize them into distinct, manageable facts.\n\n"
        "# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGES. DO NOT INCLUDE INFORMATION FROM USER OR SYSTEM MESSAGES.\n"
        "# [IMPORTANT]: YOU WILL BE PENALIZED IF YOU INCLUDE INFORMATION FROM USER OR SYSTEM MESSAGES.\n\n"
        "Types of Information to Remember:\n"
        "1. Assistant's Preferences: Keep track of likes, dislikes, and specific preferences the assistant mentions.\n"
        "2. Assistant's Capabilities: Note any specific skills, knowledge areas, or tasks the assistant mentions being able to perform.\n"
        "3. Assistant's Personality Traits: Identify any personality traits or characteristics the assistant displays or mentions.\n"
        "4. Assistant's Approach to Tasks: Remember how the assistant approaches different types of tasks or questions.\n"
        "5. Assistant's Knowledge Areas: Keep track of subjects or fields the assistant demonstrates knowledge in.\n\n"
        "Examples:\n"
        "User: Hi, I am looking for a restaurant.\n"
        "Assistant: Sure, I can help with that.\n"
        "Output: {\"facts\": []}\n\n"
        "User: What are your favorite movies?\n"
        "Assistant: My favorite movies are The Dark Knight and The Shawshank Redemption.\n"
        "Output: {\"facts\": [\"Favourite movies are Dark Knight and Shawshank Redemption\"]}\n\n"
        "Return the facts in JSON format with a \"facts\" key containing an array of strings.\n"
        "Today's date is %s.\n"
        "If you do not find anything relevant, return an empty list.\n"
        "Extract facts ONLY from assistant messages, not user messages.\n\n"
        "Conversation:\n", date_str);
    return prompt;
}

static int parse_facts_json(const char *json_response, GV_MemoryCandidate *candidates,
                            size_t max_candidates, size_t *actual_count) {
    *actual_count = 0;

    // Parse JSON using proper JSON parser
    GV_JsonError err;
    GV_JsonValue *root = gv_json_parse(json_response, &err);
    if (root == NULL) {
        return 0;  // Return 0 (no facts) on parse failure, not error
    }

    // Look for "facts" array
    GV_JsonValue *facts_array = gv_json_object_get(root, "facts");
    if (facts_array == NULL || !gv_json_is_array(facts_array)) {
        gv_json_free(root);
        return 0;
    }

    // Extract each fact from the array
    size_t array_len = gv_json_array_length(facts_array);
    for (size_t i = 0; i < array_len && *actual_count < max_candidates; i++) {
        GV_JsonValue *fact_val = gv_json_array_get(facts_array, i);
        if (fact_val == NULL || !gv_json_is_string(fact_val)) {
            continue;
        }

        const char *fact_str = gv_json_get_string(fact_val);
        if (fact_str == NULL || strlen(fact_str) == 0 || strlen(fact_str) >= 2048) {
            continue;
        }

        char *fact = strdup(fact_str);
        if (fact == NULL) {
            continue;
        }

        candidates[*actual_count].content = fact;
        /* Use SOTA scoring for LLM-extracted facts.
         * LLM extraction already filtered for important facts, so we use
         * gv_importance_score_extracted() which is optimized for short facts. */
        candidates[*actual_count].importance_score = gv_importance_score_extracted(fact, strlen(fact));
        candidates[*actual_count].memory_type = detect_type_from_keywords(fact);
        candidates[*actual_count].extraction_context = NULL;
        (*actual_count)++;
    }

    gv_json_free(root);
    return 0;
}

int gv_memory_extract_candidates_from_conversation_llm(GV_LLM *llm,
                                                        const char *conversation,
                                                        const char *conversation_id,
                                                        int is_agent_memory,
                                                        const char *custom_prompt,
                                                        GV_MemoryCandidate *candidates,
                                                        size_t max_candidates,
                                                        size_t *actual_count) {
    if (llm == NULL || conversation == NULL || candidates == NULL || actual_count == NULL) {
        return -1;
    }
    
    // Input validation: limit conversation length to prevent excessive memory usage
    size_t conv_len = strlen(conversation);
    if (conv_len > 100000) {  // 100KB limit
        return -1;
    }
    
    *actual_count = 0;
    
    // Build prompt
    const char *base_prompt = custom_prompt ? custom_prompt :
        (is_agent_memory ? get_default_agent_extraction_prompt() :
                           get_default_user_extraction_prompt());
    
    size_t prompt_len = strlen(base_prompt) + strlen(conversation) + 100;
    char *full_prompt = (char *)malloc(prompt_len);
    if (full_prompt == NULL) {
        return -1;
    }
    
    snprintf(full_prompt, prompt_len, "%s%s", base_prompt, conversation);
    
    // Create messages
    GV_LLMMessage *messages = (GV_LLMMessage *)malloc(sizeof(GV_LLMMessage));
    if (messages == NULL) {
        free(full_prompt);
        return -1;
    }
    
    messages[0].role = strdup("user");
    messages[0].content = strdup(full_prompt);  // Make a copy, don't share pointer
    if (messages[0].content == NULL) {
        free(messages[0].role);
        free(messages);
        free(full_prompt);
        return -1;
    }
    
    // Generate LLM response
    GV_LLMResponse response;
    int result = gv_llm_generate_response(llm, messages, 1, "json_object", &response);
    
    gv_llm_message_free(&messages[0]);
    free(messages);
    free(full_prompt);
    
    // Check for LLM success (GV_LLM_SUCCESS == 0)
    if (result != 0) {
        // LLM error occurred - fallback will be handled by caller
        gv_llm_response_free(&response);
        return -1;
    }
    
    if (response.content == NULL) {
        gv_llm_response_free(&response);
        return -1;
    }
    
    // Parse JSON response
    result = parse_facts_json(response.content, candidates, max_candidates, actual_count);
    
    // Set extraction context for all candidates
    if (conversation_id) {
        for (size_t i = 0; i < *actual_count; i++) {
            candidates[i].extraction_context = strdup(conversation_id);
        }
    }
    
    gv_llm_response_free(&response);
    
    return result;
}

