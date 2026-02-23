/**
 * @file test_agent.c
 * @brief Unit tests for the AI agent framework (gv_agent.h).
 * Agent creation requires a real LLM provider, so most tests verify
 * graceful failure when no LLM is available, plus config/struct fields.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gigavector/gv_database.h"
#include "gigavector/gv_agent.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static const char *TEST_DB = "tmp_test_agent.bin";

static GV_Database *create_test_db(void) {
    remove(TEST_DB);
    return gv_db_open(TEST_DB, 4, GV_INDEX_TYPE_FLAT);
}

/* Test 1: Create agent returns NULL without real LLM (expected) */
static int test_agent_create_no_llm(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "database creation");

    GV_AgentConfig config = {0};
    config.agent_type = GV_AGENT_QUERY;
    config.llm_provider = "openai";
    config.api_key = "test-key-not-real";
    config.model = "test-model";
    config.temperature = 0.5f;
    config.max_retries = 3;

    /* Without a real LLM endpoint, creation may return NULL. That's OK. */
    GV_Agent *agent = gv_agent_create(db, &config);
    if (agent != NULL) {
        gv_agent_destroy(agent);
    }
    /* Either way — no crash is success */

    gv_db_close(db);
    remove(TEST_DB);
    return 0;
}

/* Test 2: Destroy NULL agent (should not crash) */
static int test_agent_destroy_null(void) {
    gv_agent_destroy(NULL);
    return 0;
}

/* Test 3: Create agent with NULL db or NULL config */
static int test_agent_create_null_params(void) {
    GV_AgentConfig config = {0};
    config.agent_type = GV_AGENT_QUERY;
    config.llm_provider = "openai";
    config.api_key = "test-key";
    config.model = "test-model";

    GV_Agent *agent = gv_agent_create(NULL, &config);
    ASSERT(agent == NULL, "agent creation with NULL db should fail");

    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "database creation");
    agent = gv_agent_create(db, NULL);
    ASSERT(agent == NULL, "agent creation with NULL config should fail");

    gv_db_close(db);
    remove(TEST_DB);
    return 0;
}

/* Test 4: Create with NULL api_key fails */
static int test_agent_create_null_api_key(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "database creation");

    GV_AgentConfig config = {0};
    config.agent_type = GV_AGENT_QUERY;
    config.llm_provider = "openai";
    config.api_key = NULL; /* NULL key */
    config.model = "test-model";

    GV_Agent *agent = gv_agent_create(db, &config);
    ASSERT(agent == NULL, "agent creation with NULL api_key should fail");

    gv_db_close(db);
    remove(TEST_DB);
    return 0;
}

/* Test 5: Free NULL result (should not crash) */
static int test_agent_free_result_null(void) {
    gv_agent_free_result(NULL);
    return 0;
}

/* Test 6: Agent config defaults and fields */
static int test_agent_config_fields(void) {
    GV_AgentConfig config;
    memset(&config, 0, sizeof(config));

    ASSERT(config.agent_type == 0, "default agent_type should be 0 (QUERY)");
    ASSERT(config.llm_provider == NULL, "default llm_provider should be NULL");
    ASSERT(config.api_key == NULL, "default api_key should be NULL");
    ASSERT(config.model == NULL, "default model should be NULL");
    ASSERT(config.temperature == 0.0f, "default temperature should be 0.0");
    ASSERT(config.max_retries == 0, "default max_retries should be 0");
    ASSERT(config.system_prompt_override == NULL, "default system_prompt_override should be NULL");

    config.agent_type = GV_AGENT_PERSONALIZE;
    config.llm_provider = "anthropic";
    config.api_key = "sk-test-key-12345";
    config.model = "claude-3";
    config.temperature = 0.7f;
    config.max_retries = 5;
    config.system_prompt_override = "You are a helpful assistant.";

    ASSERT(config.agent_type == GV_AGENT_PERSONALIZE, "agent_type should be PERSONALIZE");
    ASSERT(strcmp(config.llm_provider, "anthropic") == 0, "llm_provider check");
    ASSERT(strcmp(config.api_key, "sk-test-key-12345") == 0, "api_key check");
    ASSERT(strcmp(config.model, "claude-3") == 0, "model check");
    ASSERT(config.temperature > 0.69f && config.temperature < 0.71f, "temperature check");
    ASSERT(config.max_retries == 5, "max_retries check");
    ASSERT(strcmp(config.system_prompt_override, "You are a helpful assistant.") == 0, "system_prompt check");

    return 0;
}

/* Test 7: Agent result structure fields */
static int test_agent_result_structure(void) {
    GV_AgentResult result;
    memset(&result, 0, sizeof(result));

    ASSERT(result.success == 0, "default success should be 0");
    ASSERT(result.response_text == NULL, "default response_text should be NULL");
    ASSERT(result.result_indices == NULL, "default result_indices should be NULL");
    ASSERT(result.result_distances == NULL, "default result_distances should be NULL");
    ASSERT(result.result_count == 0, "default result_count should be 0");
    ASSERT(result.generated_filter == NULL, "default generated_filter should be NULL");
    ASSERT(result.error_message == NULL, "default error_message should be NULL");

    return 0;
}

/* Test 8: Agent type enum values */
static int test_agent_type_enums(void) {
    ASSERT(GV_AGENT_QUERY == 0, "GV_AGENT_QUERY == 0");
    ASSERT(GV_AGENT_TRANSFORM == 1, "GV_AGENT_TRANSFORM == 1");
    ASSERT(GV_AGENT_PERSONALIZE == 2, "GV_AGENT_PERSONALIZE == 2");
    return 0;
}

/* Test 9: Schema hint on NULL agent (should not crash) */
static int test_agent_schema_hint_null(void) {
    gv_agent_set_schema_hint(NULL, "{}");
    gv_agent_set_schema_hint(NULL, NULL);
    return 0;
}

/* Test 10: All agent types with no-real-llm */
static int test_agent_all_types_no_llm(void) {
    GV_Database *db = create_test_db();
    ASSERT(db != NULL, "database creation");

    int types[] = {GV_AGENT_QUERY, GV_AGENT_TRANSFORM, GV_AGENT_PERSONALIZE};
    for (int i = 0; i < 3; i++) {
        GV_AgentConfig config = {0};
        config.agent_type = types[i];
        config.llm_provider = "openai";
        config.api_key = "fake-key";
        config.model = "test-model";
        config.temperature = 0.0f;
        config.max_retries = 1;

        GV_Agent *agent = gv_agent_create(db, &config);
        /* May be NULL if LLM not available — that's expected */
        if (agent != NULL) {
            gv_agent_destroy(agent);
        }
    }

    gv_db_close(db);
    remove(TEST_DB);
    return 0;
}

int main(void) {
    int failed = 0, passed = 0;
    remove(TEST_DB);

    struct { const char *name; int (*fn)(void); } tests[] = {
        {"test_agent_create_no_llm",       test_agent_create_no_llm},
        {"test_agent_destroy_null",        test_agent_destroy_null},
        {"test_agent_create_null_params",  test_agent_create_null_params},
        {"test_agent_create_null_api_key", test_agent_create_null_api_key},
        {"test_agent_free_result_null",    test_agent_free_result_null},
        {"test_agent_config_fields",       test_agent_config_fields},
        {"test_agent_result_structure",    test_agent_result_structure},
        {"test_agent_type_enums",          test_agent_type_enums},
        {"test_agent_schema_hint_null",    test_agent_schema_hint_null},
        {"test_agent_all_types_no_llm",    test_agent_all_types_no_llm},
    };

    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < num_tests; i++) {
        int result = tests[i].fn();
        if (result == 0) { printf("  OK   %s\n", tests[i].name); passed++; }
        else { printf("  FAILED %s\n", tests[i].name); failed++; }
    }

    printf("\n%d/%d tests passed\n", passed, num_tests);
    remove(TEST_DB);
    return failed > 0 ? 1 : 0;
}
