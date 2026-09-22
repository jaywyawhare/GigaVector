#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "features/knowledge_graph.h"
#include "features/cypher.h"
#include "features/cypher_vector.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static const char *cell(const GV_CypherResult *r, size_t row, size_t col) {
    return r->column_values[row * r->column_count + col];
}

/* ---------- cypher_vector.h standalone tests ---------- */

static void test_vector_distance_l2(void) {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    float d = cypher_vector_distance(a, 3, b, 3, GV_VECDIST_L2);
    ASSERT(fabsf(d - sqrtf(2.0f)) < 1e-5f, "L2 distance (1,0,0)-(0,1,0) = sqrt(2)");
}

static void test_vector_distance_l2_identical(void) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float d = cypher_vector_distance(a, 3, a, 3, GV_VECDIST_L2);
    ASSERT(fabsf(d) < 1e-6f, "L2 distance of identical vectors = 0");
}

static void test_vector_distance_cosine(void) {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    float d = cypher_vector_distance(a, 2, b, 2, GV_VECDIST_COSINE);
    ASSERT(fabsf(d - 1.0f) < 1e-5f, "Cosine distance orthogonal = 1.0");
}

static void test_vector_distance_cosine_identical(void) {
    float a[] = {3.0f, 4.0f};
    float d = cypher_vector_distance(a, 2, a, 2, GV_VECDIST_COSINE);
    ASSERT(fabsf(d) < 1e-5f, "Cosine distance identical = 0");
}

static void test_vector_distance_dot(void) {
    float a[] = {1.0f, 2.0f};
    float b[] = {3.0f, 4.0f};
    float d = cypher_vector_distance(a, 2, b, 2, GV_VECDIST_DOT);
    ASSERT(fabsf(d - (-11.0f)) < 1e-5f, "Negative dot product (1,2).(3,4) = -11");
}

static void test_vector_distance_hamming(void) {
    float a[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float b[] = {0.0f, 0.0f, 1.0f, 1.0f};
    float d = cypher_vector_distance(a, 4, b, 4, GV_VECDIST_HAMMING);
    ASSERT(fabsf(d - 2.0f) < 1e-5f, "Hamming distance: 2 positions differ");
}

static void test_parse_vecdist_type(void) {
    ASSERT(cypher_parse_vecdist_type("l2") == GV_VECDIST_L2, "parse 'l2'");
    ASSERT(cypher_parse_vecdist_type("euclidean") == GV_VECDIST_L2, "parse 'euclidean'");
    ASSERT(cypher_parse_vecdist_type("cosine") == GV_VECDIST_COSINE, "parse 'cosine'");
    ASSERT(cypher_parse_vecdist_type("dot") == GV_VECDIST_DOT, "parse 'dot'");
    ASSERT(cypher_parse_vecdist_type("hamming") == GV_VECDIST_HAMMING, "parse 'hamming'");
    ASSERT(cypher_parse_vecdist_type(NULL) == GV_VECDIST_L2, "parse NULL -> L2 default");
    ASSERT(cypher_parse_vecdist_type("unknown") == GV_VECDIST_L2, "parse unknown -> L2 default");
}

static void test_vector_distance_mismatched_dims(void) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f};
    /* Should use min(3,2) = 2 dimensions */
    float d = cypher_vector_distance(a, 3, b, 2, GV_VECDIST_L2);
    ASSERT(d >= 0.0f, "Mismatched dims still returns valid distance");
    ASSERT(fabsf(d - sqrtf(18.0f)) < 1e-5f, "Uses min dims: sqrt((1-4)^2+(2-5)^2)=sqrt(18)");
}

/* ---------- Cypher integration tests ---------- */

static void test_cypher_vector_distance_l2(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    GV_CypherEngine *cy = cypher_create(kg);
    GV_CypherResult r;

    /* Create a node with an embedding property (comma-separated floats) */
    ASSERT(cypher_execute(cy,
        "CREATE (a:Item {embedding:'1.0,0.0,0.0'})", &r) == 0, "create Item");
    cypher_free_result(&r);

    /* vector_distance_l2 with $q parameter */
    cypher_set_parameter(cy, "q", "0.0,1.0,0.0");
    ASSERT(cypher_execute(cy,
        "MATCH (a:Item) RETURN vector_distance_l2(a.embedding, $q)", &r) == 0,
        "vector_distance_l2 query");
    ASSERT(r.row_count == 1, "one row");
    /* L2: sqrt((1-0)^2 + (0-1)^2 + 0) = sqrt(2) ≈ 1.41421 */
    double dist = strtod(cell(&r, 0, 0), NULL);
    ASSERT(fabs(dist - 1.41421) < 0.01, "L2 distance sqrt(2)");
    cypher_free_result(&r);

    cypher_destroy(cy);
    kg_destroy(kg);
}

static void test_cypher_vector_distance_generic(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    GV_CypherEngine *cy = cypher_create(kg);
    GV_CypherResult r;

    ASSERT(cypher_execute(cy,
        "CREATE (a:Vec {embedding:'1.0,0.0'})", &r) == 0, "create Vec");
    cypher_free_result(&r);

    cypher_set_parameter(cy, "q", "0.0,1.0");
    /* vector_distance with explicit 'cosine' metric */
    ASSERT(cypher_execute(cy,
        "MATCH (a:Vec) RETURN vector_distance(a.embedding, $q, 'cosine')", &r) == 0,
        "vector_distance cosine");
    double dist = strtod(cell(&r, 0, 0), NULL);
    ASSERT(fabs(dist - 1.0) < 0.01, "Cosine distance orthogonal = 1.0");
    cypher_free_result(&r);

    /* vector_distance with explicit 'dot' metric */
    ASSERT(cypher_execute(cy,
        "MATCH (a:Vec) RETURN vector_distance(a.embedding, $q, 'dot')", &r) == 0,
        "vector_distance dot");
    dist = strtod(cell(&r, 0, 0), NULL);
    ASSERT(fabs(dist - 0.0) < 0.01, "Dot: -(1*0 + 0*1) = 0");
    cypher_free_result(&r);

    cypher_destroy(cy);
    kg_destroy(kg);
}

static void test_cypher_vector_distance_in_where(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    GV_CypherEngine *cy = cypher_create(kg);
    GV_CypherResult r;

    /* Create two nodes with different embeddings */
    ASSERT(cypher_execute(cy,
        "CREATE (a:Point {name:'A', emb:'1.0,0.0,0.0'})", &r) == 0, "create A");
    cypher_free_result(&r);
    ASSERT(cypher_execute(cy,
        "CREATE (b:Point {name:'B', emb:'0.0,1.0,0.0'})", &r) == 0, "create B");
    cypher_free_result(&r);

    cypher_set_parameter(cy, "q", "1.0,0.0,0.0");

    /* WHERE clause: only A should match (distance 0) */
    ASSERT(cypher_execute(cy,
        "MATCH (p:Point) WHERE vector_distance_l2(p.emb, $q) < 0.5 RETURN p.name", &r) == 0,
        "WHERE vector_distance");
    ASSERT(r.row_count == 1 && strcmp(cell(&r, 0, 0), "A") == 0,
           "Only A within 0.5 of query");
    cypher_free_result(&r);

    cypher_destroy(cy);
    kg_destroy(kg);
}

static void test_cypher_vector_distance_order_by(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    GV_CypherEngine *cy = cypher_create(kg);
    GV_CypherResult r;

    ASSERT(cypher_execute(cy,
        "CREATE (:P {name:'X', v:'1.0,0.0'})", &r) == 0, "create X");
    cypher_free_result(&r);
    ASSERT(cypher_execute(cy,
        "CREATE (:P {name:'Y', v:'0.5,0.5'})", &r) == 0, "create Y");
    cypher_free_result(&r);
    ASSERT(cypher_execute(cy,
        "CREATE (:P {name:'Z', v:'0.0,1.0'})", &r) == 0, "create Z");
    cypher_free_result(&r);

    cypher_set_parameter(cy, "q", "1.0,0.0");
    ASSERT(cypher_execute(cy,
        "MATCH (p:P) RETURN p.name, vector_distance_l2(p.v, $q) AS dist ORDER BY dist",
        &r) == 0, "ORDER BY vector_distance");
    ASSERT(r.row_count == 3, "3 results");
    /* X closest (dist=0), then Y (dist≈0.707), then Z (dist=1) */
    ASSERT(strcmp(cell(&r, 0, 0), "X") == 0, "X is closest");
    ASSERT(strcmp(cell(&r, 1, 0), "Y") == 0, "Y is second");
    ASSERT(strcmp(cell(&r, 2, 0), "Z") == 0, "Z is farthest");
    cypher_free_result(&r);

    cypher_destroy(cy);
    kg_destroy(kg);
}

static void test_cypher_vector_distance_as_alias(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    GV_CypherEngine *cy = cypher_create(kg);
    GV_CypherResult r;

    ASSERT(cypher_execute(cy,
        "CREATE (n:Rec {emb:'3.0,4.0'})", &r) == 0, "create Rec");
    cypher_free_result(&r);

    cypher_set_parameter(cy, "q", "0.0,0.0");
    ASSERT(cypher_execute(cy,
        "MATCH (n:Rec) RETURN vector_distance_l2(n.emb, $q) AS distance", &r) == 0,
        "AS alias");
    double dist = strtod(cell(&r, 0, 0), NULL);
    /* L2: sqrt(9+16) = 5 */
    ASSERT(fabs(dist - 5.0) < 0.01, "distance = 5.0 via AS alias");
    cypher_free_result(&r);

    cypher_destroy(cy);
    kg_destroy(kg);
}

/* ---------- test runner ---------- */

typedef void (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        /* Standalone cypher_vector.h tests */
        {"vector_distance_l2",                  test_vector_distance_l2},
        {"vector_distance_l2_identical",        test_vector_distance_l2_identical},
        {"vector_distance_cosine",              test_vector_distance_cosine},
        {"vector_distance_cosine_identical",    test_vector_distance_cosine_identical},
        {"vector_distance_dot",                 test_vector_distance_dot},
        {"vector_distance_hamming",             test_vector_distance_hamming},
        {"parse_vecdist_type",                  test_parse_vecdist_type},
        {"vector_distance_mismatched_dims",     test_vector_distance_mismatched_dims},
        /* Cypher integration tests */
        {"cypher_vector_distance_l2",           test_cypher_vector_distance_l2},
        {"cypher_vector_distance_generic",      test_cypher_vector_distance_generic},
        {"cypher_vector_distance_in_where",     test_cypher_vector_distance_in_where},
        {"cypher_vector_distance_order_by",     test_cypher_vector_distance_order_by},
        {"cypher_vector_distance_as_alias",     test_cypher_vector_distance_as_alias},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    for (int i = 0; i < n; i++) {
        tests[i].fn();
    }
    printf(failures ? "\n%d/%d TESTS FAILED\n" : "\nALL %d TESTS PASSED\n", failures, n);
    return failures ? 1 : 0;
}
