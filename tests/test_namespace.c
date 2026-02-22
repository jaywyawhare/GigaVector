#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_namespace.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_config_init(void) {
    GV_NamespaceConfig cfg;
    memset(&cfg, 0xFF, sizeof(cfg));
    gv_namespace_config_init(&cfg);
    ASSERT(cfg.name == NULL,                    "name should be NULL");
    ASSERT(cfg.dimension == 0,                  "dimension should be 0");
    ASSERT(cfg.index_type == GV_NS_INDEX_HNSW,  "index_type should be HNSW");
    ASSERT(cfg.max_vectors == 0,                "max_vectors should be 0");
    ASSERT(cfg.max_memory_bytes == 0,           "max_memory_bytes should be 0");
    return 0;
}

static int test_manager_create_destroy(void) {
    GV_NamespaceManager *mgr = gv_namespace_manager_create(NULL);
    ASSERT(mgr != NULL, "manager_create(NULL) should succeed");
    gv_namespace_manager_destroy(mgr);

    /* Destroy NULL is safe */
    gv_namespace_manager_destroy(NULL);
    return 0;
}

static int test_create_and_get(void) {
    GV_NamespaceManager *mgr = gv_namespace_manager_create(NULL);
    ASSERT(mgr != NULL, "create manager");

    GV_NamespaceConfig cfg;
    gv_namespace_config_init(&cfg);
    cfg.name = "vectors_v1";
    cfg.dimension = 128;

    GV_Namespace *ns = gv_namespace_create(mgr, &cfg);
    ASSERT(ns != NULL, "namespace_create should succeed");

    GV_Namespace *ns2 = gv_namespace_get(mgr, "vectors_v1");
    ASSERT(ns2 != NULL, "namespace_get should find the namespace");

    GV_Namespace *ns3 = gv_namespace_get(mgr, "nonexistent");
    ASSERT(ns3 == NULL, "namespace_get for unknown should return NULL");

    gv_namespace_manager_destroy(mgr);
    return 0;
}

static int test_exists_and_delete(void) {
    GV_NamespaceManager *mgr = gv_namespace_manager_create(NULL);
    ASSERT(mgr != NULL, "create manager");

    GV_NamespaceConfig cfg;
    gv_namespace_config_init(&cfg);
    cfg.name = "temp_ns";
    cfg.dimension = 4;
    ASSERT(gv_namespace_create(mgr, &cfg) != NULL, "create ns");

    ASSERT(gv_namespace_exists(mgr, "temp_ns") == 1, "should exist");
    ASSERT(gv_namespace_exists(mgr, "nope") == 0,    "should not exist");

    ASSERT(gv_namespace_delete(mgr, "temp_ns") == 0, "delete should succeed");
    ASSERT(gv_namespace_exists(mgr, "temp_ns") == 0, "should not exist after delete");

    gv_namespace_manager_destroy(mgr);
    return 0;
}

static int test_list_namespaces(void) {
    GV_NamespaceManager *mgr = gv_namespace_manager_create(NULL);
    ASSERT(mgr != NULL, "create manager");

    GV_NamespaceConfig cfg;
    gv_namespace_config_init(&cfg);
    cfg.dimension = 8;

    cfg.name = "alpha";
    ASSERT(gv_namespace_create(mgr, &cfg) != NULL, "create alpha");
    cfg.name = "beta";
    ASSERT(gv_namespace_create(mgr, &cfg) != NULL, "create beta");

    char **names = NULL;
    size_t count = 0;
    ASSERT(gv_namespace_list(mgr, &names, &count) == 0, "list");
    ASSERT(count == 2, "should have 2 namespaces");

    /* Free returned names */
    if (names) {
        for (size_t i = 0; i < count; i++) free(names[i]);
        free(names);
    }

    gv_namespace_manager_destroy(mgr);
    return 0;
}

static int test_add_vector_and_count(void) {
    GV_NamespaceManager *mgr = gv_namespace_manager_create(NULL);
    ASSERT(mgr != NULL, "create manager");

    GV_NamespaceConfig cfg;
    gv_namespace_config_init(&cfg);
    cfg.name = "vec_ns";
    cfg.dimension = 3;
    GV_Namespace *ns = gv_namespace_create(mgr, &cfg);
    ASSERT(ns != NULL, "create ns");

    float v1[] = {1.0f, 2.0f, 3.0f};
    float v2[] = {4.0f, 5.0f, 6.0f};
    ASSERT(gv_namespace_add_vector(ns, v1, 3) == 0, "add v1");
    ASSERT(gv_namespace_add_vector(ns, v2, 3) == 0, "add v2");
    ASSERT(gv_namespace_count(ns) == 2, "count should be 2");

    gv_namespace_manager_destroy(mgr);
    return 0;
}

static int test_get_info(void) {
    GV_NamespaceManager *mgr = gv_namespace_manager_create(NULL);
    ASSERT(mgr != NULL, "create manager");

    GV_NamespaceConfig cfg;
    gv_namespace_config_init(&cfg);
    cfg.name = "info_ns";
    cfg.dimension = 16;
    cfg.index_type = GV_NS_INDEX_HNSW;
    GV_Namespace *ns = gv_namespace_create(mgr, &cfg);
    ASSERT(ns != NULL, "create ns");

    GV_NamespaceInfo info;
    memset(&info, 0, sizeof(info));
    ASSERT(gv_namespace_get_info(ns, &info) == 0, "get_info");
    ASSERT(info.dimension == 16, "dimension should be 16");
    ASSERT(info.index_type == GV_NS_INDEX_HNSW, "index type should be HNSW");
    gv_namespace_free_info(&info);

    gv_namespace_manager_destroy(mgr);
    return 0;
}

static int test_get_db(void) {
    GV_NamespaceManager *mgr = gv_namespace_manager_create(NULL);
    ASSERT(mgr != NULL, "create manager");

    GV_NamespaceConfig cfg;
    gv_namespace_config_init(&cfg);
    cfg.name = "db_ns";
    cfg.dimension = 4;
    GV_Namespace *ns = gv_namespace_create(mgr, &cfg);
    ASSERT(ns != NULL, "create ns");

    GV_Database *db = gv_namespace_get_db(ns);
    ASSERT(db != NULL, "get_db should return non-NULL");

    gv_namespace_manager_destroy(mgr);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing config_init...",          test_config_init},
        {"Testing manager_create_destroy..",test_manager_create_destroy},
        {"Testing create_and_get...",       test_create_and_get},
        {"Testing exists_and_delete...",    test_exists_and_delete},
        {"Testing list_namespaces...",      test_list_namespaces},
        {"Testing add_vector_and_count...", test_add_vector_and_count},
        {"Testing get_info...",             test_get_info},
        {"Testing get_db...",               test_get_db},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
