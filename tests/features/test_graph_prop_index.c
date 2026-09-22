#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "features/graph_prop_index.h"
#include "core/prop_value.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_create_destroy(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();
    ASSERT(idx != NULL, "create index");
    ASSERT(graph_prop_index_node_count(idx) == 0, "empty node count");
    ASSERT(graph_prop_index_edge_count(idx) == 0, "empty edge count");
    graph_prop_index_destroy(idx);

    graph_prop_index_destroy(NULL);
    return 0;
}

static int test_add_find_string(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue v1 = gv_prop_string("Alice");
    GV_PropValue v2 = gv_prop_string("Bob");
    GV_PropValue v3 = gv_prop_string("Alice");

    ASSERT(graph_prop_index_add_node(idx, "name", v1, 1) == 0, "add Alice");
    ASSERT(graph_prop_index_add_node(idx, "name", v2, 2) == 0, "add Bob");
    ASSERT(graph_prop_index_add_node(idx, "name", v3, 3) == 0, "add Alice dup");
    ASSERT(graph_prop_index_add_node(idx, "city", v1, 1) == 0, "add city=Alice");

    ASSERT(graph_prop_index_node_count(idx) == 4, "total node entries");

    uint64_t ids[16];
    int n = graph_prop_index_find_exact(idx, "name", v1, ids, 16);
    ASSERT(n == 2, "find name=Alice count");
    ASSERT((ids[0] == 1 && ids[1] == 3) || (ids[0] == 3 && ids[1] == 1),
           "find name=Alice ids");

    n = graph_prop_index_find_exact(idx, "name", v2, ids, 16);
    ASSERT(n == 1, "find name=Bob count");
    ASSERT(ids[0] == 2, "find name=Bob id");

    n = graph_prop_index_find_exact(idx, "name", v3, ids, 16);
    ASSERT(n == 2, "find name=Alice via dup key");

    n = graph_prop_index_find_exact(idx, "city", v1, ids, 16);
    ASSERT(n == 1, "find city=Alice");
    ASSERT(ids[0] == 1, "find city=Alice id");

    n = graph_prop_index_find_exact(idx, "missing", v1, ids, 16);
    ASSERT(n == 0, "find missing key");

    gv_prop_free(&v1);
    gv_prop_free(&v2);
    gv_prop_free(&v3);
    graph_prop_index_destroy(idx);
    return 0;
}

static int test_add_find_int64(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue v10 = gv_prop_int64(10);
    GV_PropValue v20 = gv_prop_int64(20);
    GV_PropValue v30 = gv_prop_int64(30);

    graph_prop_index_add_node(idx, "age", v10, 1);
    graph_prop_index_add_node(idx, "age", v20, 2);
    graph_prop_index_add_node(idx, "age", v30, 3);
    graph_prop_index_add_node(idx, "age", v10, 4);

    uint64_t ids[16];
    int n = graph_prop_index_find_exact(idx, "age", v20, ids, 16);
    ASSERT(n == 1, "find age=20");
    ASSERT(ids[0] == 2, "find age=20 id");

    n = graph_prop_index_find_exact(idx, "age", v10, ids, 16);
    ASSERT(n == 2, "find age=10 count");

    graph_prop_index_destroy(idx);
    return 0;
}

static int test_add_find_float64(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue v1 = gv_prop_float64(1.5);
    GV_PropValue v2 = gv_prop_float64(2.5);
    GV_PropValue v3 = gv_prop_float64(3.5);

    graph_prop_index_add_node(idx, "score", v1, 10);
    graph_prop_index_add_node(idx, "score", v2, 20);
    graph_prop_index_add_node(idx, "score", v3, 30);

    uint64_t ids[16];
    int n = graph_prop_index_find_exact(idx, "score", v2, ids, 16);
    ASSERT(n == 1, "find score=2.5");
    ASSERT(ids[0] == 20, "find score=2.5 id");

    graph_prop_index_destroy(idx);
    return 0;
}

static int test_add_find_bool(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue vt = gv_prop_bool(1);
    GV_PropValue vf = gv_prop_bool(0);

    graph_prop_index_add_node(idx, "active", vt, 1);
    graph_prop_index_add_node(idx, "active", vf, 2);
    graph_prop_index_add_node(idx, "active", vt, 3);

    uint64_t ids[16];
    int n = graph_prop_index_find_exact(idx, "active", vt, ids, 16);
    ASSERT(n == 2, "find active=true count");
    ASSERT((ids[0] == 1 && ids[1] == 3) || (ids[0] == 3 && ids[1] == 1),
           "find active=true ids");

    n = graph_prop_index_find_exact(idx, "active", vf, ids, 16);
    ASSERT(n == 1, "find active=false count");

    graph_prop_index_destroy(idx);
    return 0;
}

static int test_range_int64(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    for (int64_t i = 1; i <= 10; i++) {
        GV_PropValue v = gv_prop_int64(i);
        graph_prop_index_add_node(idx, "val", v, (uint64_t)i);
    }

    GV_PropValue min = gv_prop_int64(3);
    GV_PropValue max = gv_prop_int64(7);
    uint64_t ids[16];

    int n = graph_prop_index_find_range(idx, "val", min, max, ids, 16);
    ASSERT(n == 5, "range [3,7] count");
    ASSERT(ids[0] == 3, "range [3,7] first");
    ASSERT(ids[4] == 7, "range [3,7] last");

    /* Open-ended ranges with unbounded min/max */
    GV_PropValue null = gv_prop_null();
    min = gv_prop_int64(8);
    n = graph_prop_index_find_range(idx, "val", min, null, ids, 16);
    ASSERT(n == 3, "range [8,] count");

    max = gv_prop_int64(4);
    n = graph_prop_index_find_range(idx, "val", null, max, ids, 16);
    ASSERT(n == 4, "range [,4] count");

    graph_prop_index_destroy(idx);
    return 0;
}

static int test_range_float64(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue v1 = gv_prop_float64(1.0);
    GV_PropValue v2 = gv_prop_float64(2.0);
    GV_PropValue v3 = gv_prop_float64(3.0);
    GV_PropValue v4 = gv_prop_float64(4.0);
    GV_PropValue v5 = gv_prop_float64(5.0);

    graph_prop_index_add_node(idx, "score", v1, 10);
    graph_prop_index_add_node(idx, "score", v2, 20);
    graph_prop_index_add_node(idx, "score", v3, 30);
    graph_prop_index_add_node(idx, "score", v4, 40);
    graph_prop_index_add_node(idx, "score", v5, 50);

    GV_PropValue min = gv_prop_float64(2.0);
    GV_PropValue max = gv_prop_float64(4.0);
    uint64_t ids[16];

    int n = graph_prop_index_find_range(idx, "score", min, max, ids, 16);
    ASSERT(n == 3, "float range [2,4] count");
    ASSERT(ids[0] == 20, "float range first");
    ASSERT(ids[2] == 40, "float range last");

    graph_prop_index_destroy(idx);
    return 0;
}

static int test_remove_node(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue v1 = gv_prop_string("hello");
    GV_PropValue v2 = gv_prop_string("world");

    graph_prop_index_add_node(idx, "key", v1, 1);
    graph_prop_index_add_node(idx, "key", v1, 2);
    graph_prop_index_add_node(idx, "key", v2, 3);

    uint64_t ids[16];
    ASSERT(graph_prop_index_find_exact(idx, "key", v1, ids, 16) == 2, "before remove");

    ASSERT(graph_prop_index_remove_node(idx, "key", v1, 1) == 0, "remove node 1");
    ASSERT(graph_prop_index_find_exact(idx, "key", v1, ids, 16) == 1, "after remove 1");
    ASSERT(ids[0] == 2, "remaining id is 2");

    ASSERT(graph_prop_index_remove_node(idx, "key", v1, 2) == 0, "remove node 2");
    ASSERT(graph_prop_index_find_exact(idx, "key", v1, ids, 16) == 0, "all gone");

    /* Entry should be fully deleted when both node and edge counts are 0 */
    ASSERT(graph_prop_index_find_exact(idx, "key", v2, ids, 16) == 1, "other key intact");

    /* Removing nonexistent */
    ASSERT(graph_prop_index_remove_node(idx, "key", v1, 99) != 0, "remove missing id");
    ASSERT(graph_prop_index_remove_node(idx, "missing", v1, 1) != 0, "remove missing key");

    gv_prop_free(&v1);
    gv_prop_free(&v2);
    graph_prop_index_destroy(idx);
    return 0;
}

static int test_edge_index(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue vk = gv_prop_string("KNOWS");
    GV_PropValue wf = gv_prop_string("WORKS_FOR");

    graph_prop_index_add_edge(idx, "type", vk, 100);
    graph_prop_index_add_edge(idx, "type", vk, 101);
    graph_prop_index_add_edge(idx, "type", wf, 102);

    uint64_t ids[16];
    int n = graph_prop_index_find_edges_exact(idx, "type", vk, ids, 16);
    ASSERT(n == 2, "find edges type=KNOWS");

    n = graph_prop_index_find_edges_exact(idx, "type", wf, ids, 16);
    ASSERT(n == 1, "find edges type=WORKS_FOR");
    ASSERT(ids[0] == 102, "find edges type=WORKS_FOR id");

    ASSERT(graph_prop_index_edge_count(idx) == 3, "total edge count");

    ASSERT(graph_prop_index_remove_edge(idx, "type", vk, 100) == 0, "remove edge");
    n = graph_prop_index_find_edges_exact(idx, "type", vk, ids, 16);
    ASSERT(n == 1, "after remove edge");

    /* Range on edges */
    GV_PropValue v100 = gv_prop_int64(100);
    GV_PropValue v300 = gv_prop_int64(300);
    GV_PropValue ve1 = gv_prop_int64(10);
    GV_PropValue ve2 = gv_prop_int64(20);
    GV_PropValue ve3 = gv_prop_int64(30);

    graph_prop_index_add_edge(idx, "weight", ve1, 1);
    graph_prop_index_add_edge(idx, "weight", ve2, 2);
    graph_prop_index_add_edge(idx, "weight", ve3, 3);

    n = graph_prop_index_find_edges_range(idx, "weight", ve1, ve3, ids, 16);
    ASSERT(n == 3, "edge range all");

    n = graph_prop_index_find_edges_range(idx, "weight", ve2, ve2, ids, 16);
    ASSERT(n == 1, "edge range single");
    ASSERT(ids[0] == 2, "edge range single id");

    gv_prop_free(&vk);
    gv_prop_free(&wf);
    gv_prop_free(&v100);
    gv_prop_free(&v300);
    gv_prop_free(&ve1);
    gv_prop_free(&ve2);
    gv_prop_free(&ve3);
    graph_prop_index_destroy(idx);
    return 0;
}

static int test_persistence(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue vs = gv_prop_string("test");
    GV_PropValue vi = gv_prop_int64(42);
    GV_PropValue vf = gv_prop_float64(3.14);
    GV_PropValue vb = gv_prop_bool(1);

    graph_prop_index_add_node(idx, "s", vs, 1);
    graph_prop_index_add_node(idx, "i", vi, 2);
    graph_prop_index_add_node(idx, "f", vf, 3);
    graph_prop_index_add_node(idx, "b", vb, 4);

    graph_prop_index_add_edge(idx, "e", vs, 10);
    graph_prop_index_add_edge(idx, "e", vi, 20);

    char path[256];
    gv_test_make_temp_path(path, sizeof(path), "test_prop_idx", ".bin");
    FILE *f = fopen(path, "wb");
    ASSERT(f != NULL, "open file for write");
    ASSERT(graph_prop_index_save(idx, f) == 0, "save index");
    fclose(f);
    graph_prop_index_destroy(idx);

    f = fopen(path, "rb");
    ASSERT(f != NULL, "open file for read");
    GV_GraphPropIndex *idx2 = NULL;
    ASSERT(graph_prop_index_load(&idx2, f) == 0, "load index");
    fclose(f);
    ASSERT(idx2 != NULL, "loaded index not null");

    ASSERT(graph_prop_index_node_count(idx2) == 4, "loaded node count");
    ASSERT(graph_prop_index_edge_count(idx2) == 2, "loaded edge count");

    uint64_t ids[16];
    int n = graph_prop_index_find_exact(idx2, "s", vs, ids, 16);
    ASSERT(n == 1, "loaded string find");
    ASSERT(ids[0] == 1, "loaded string id");

    n = graph_prop_index_find_exact(idx2, "i", vi, ids, 16);
    ASSERT(n == 1, "loaded int64 find");
    ASSERT(ids[0] == 2, "loaded int64 id");

    n = graph_prop_index_find_exact(idx2, "f", vf, ids, 16);
    ASSERT(n == 1, "loaded float64 find");
    ASSERT(ids[0] == 3, "loaded float64 id");

    n = graph_prop_index_find_exact(idx2, "b", vb, ids, 16);
    ASSERT(n == 1, "loaded bool find");
    ASSERT(ids[0] == 4, "loaded bool id");

    n = graph_prop_index_find_edges_exact(idx2, "e", vs, ids, 16);
    ASSERT(n == 1, "loaded edge string find");
    ASSERT(ids[0] == 10, "loaded edge string id");

    n = graph_prop_index_find_edges_exact(idx2, "e", vi, ids, 16);
    ASSERT(n == 1, "loaded edge int64 find");
    ASSERT(ids[0] == 20, "loaded edge int64 id");

    gv_prop_free(&vs);
    gv_prop_free(&vi);
    gv_prop_free(&vf);
    gv_prop_free(&vb);
    graph_prop_index_destroy(idx2);
    unlink(path);
    return 0;
}

static int test_stress(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    uint32_t seed = 42;
    for (int i = 1; i <= 1000; i++) {
        seed = seed * 1103515245u + 12345u;
        int64_t val = (int64_t)(seed % 1000);
        GV_PropValue v = gv_prop_int64(val);
        graph_prop_index_add_node(idx, "rand", v, (uint64_t)i);
    }

    /* Verify each node's value is findable */
    seed = 42;
    for (int i = 1; i <= 1000; i++) {
        seed = seed * 1103515245u + 12345u;
        int64_t val = (int64_t)(seed % 1000);
        GV_PropValue v = gv_prop_int64(val);
        uint64_t ids[2048];
        int n = graph_prop_index_find_exact(idx, "rand", v, ids, 2048);
        ASSERT(n >= 1, "stress: each value has at least 1 hit");
    }

    /* Range query */
    GV_PropValue min = gv_prop_int64(100);
    GV_PropValue max = gv_prop_int64(200);
    uint64_t ids[2048];
    int n = graph_prop_index_find_range(idx, "rand", min, max, ids, 2048);
    ASSERT(n > 0, "stress: range has results");

    /* Verify all results are in range */
    for (int i = 0; i < n; i++) {
        ASSERT(ids[i] >= 1 && ids[i] <= 1000, "stress: id in valid range");
    }

    graph_prop_index_destroy(idx);
    return 0;
}

static int test_range_no_match(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue v1 = gv_prop_int64(1);

    graph_prop_index_add_node(idx, "val", v1, 1);

    GV_PropValue min = gv_prop_int64(10);
    GV_PropValue max = gv_prop_int64(20);
    uint64_t ids[16];

    int n = graph_prop_index_find_range(idx, "val", min, max, ids, 16);
    ASSERT(n == 0, "range no match");

    /* Single-element range */
    n = graph_prop_index_find_range(idx, "val", v1, v1, ids, 16);
    ASSERT(n == 1, "range exact match");

    graph_prop_index_destroy(idx);
    return 0;
}

static int test_limit_output(void)
{
    GV_GraphPropIndex *idx = graph_prop_index_create();

    GV_PropValue v = gv_prop_int64(1);
    for (int i = 1; i <= 10; i++) {
        graph_prop_index_add_node(idx, "key", v, (uint64_t)i);
    }

    uint64_t ids[3];
    int n = graph_prop_index_find_exact(idx, "key", v, ids, 3);
    ASSERT(n == 3, "limit: returns only max_count");
    ASSERT(ids[0] >= 1 && ids[0] <= 10, "limit: id valid");
    ASSERT(ids[1] >= 1 && ids[1] <= 10, "limit: id valid");
    ASSERT(ids[2] >= 1 && ids[2] <= 10, "limit: id valid");

    graph_prop_index_destroy(idx);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestEntry;

int main(void)
{
    TestEntry tests[] = {
        {"create/destroy",     test_create_destroy},
        {"add/find string",    test_add_find_string},
        {"add/find int64",     test_add_find_int64},
        {"add/find float64",   test_add_find_float64},
        {"add/find bool",      test_add_find_bool},
        {"range int64",        test_range_int64},
        {"range float64",      test_range_float64},
        {"range no match",     test_range_no_match},
        {"limit output",       test_limit_output},
        {"remove node",        test_remove_node},
        {"edge index",         test_edge_index},
        {"persistence",        test_persistence},
        {"stress",             test_stress},
    };

    int total = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;

    for (int i = 0; i < total; i++) {
        if (tests[i].fn() == 0) {
            passed++;
        }
    }
    return (passed == total) ? 0 : 1;
}
