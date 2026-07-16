/* Comprehensive Cypher-over-KG tests: multi-hop, WHERE, aggregation, ORDER BY,
   DISTINCT, SET, DELETE, MERGE, OPTIONAL MATCH. */
#include <stdio.h>
#include <string.h>
#include "features/knowledge_graph.h"
#include "features/cypher.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

static const char *cell(const GV_CypherResult *r, size_t row, size_t col) {
    return r->column_values[row * r->column_count + col];
}
static int row_with(const GV_CypherResult *r, size_t col, const char *v) {
    for (size_t i = 0; i < r->row_count; i++) if (strcmp(cell(r, i, col), v) == 0) return (int)i;
    return -1;
}
static int q(GV_CypherEngine *cy, const char *s, GV_CypherResult *r) {
    return cypher_execute(cy, s, r);
}

int main(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);
    GV_CypherEngine *cy = cypher_create(kg);
    GV_CypherResult r;

    /* Build: Alice-KNOWS->Bob-KNOWS->Carol  (multi-hop CREATE) */
    ASSERT(q(cy, "CREATE (a:Person {name:'Alice', age:'30'})-[:KNOWS]->(b:Person {name:'Bob', age:'25'})"
               "-[:KNOWS]->(c:Person {name:'Carol', age:'35'})", &r) == 0, "multi-hop CREATE");
    ASSERT(r.nodes_created == 3 && r.relationships_created == 2, "3 nodes + 2 rels created");
    cypher_free_result(&r);

    /* CREATE after MATCH (reuse bound var) */
    ASSERT(q(cy, "MATCH (a {name:'Alice'}) CREATE (a)-[:WORKS_AT]->(acme:Company {name:'Acme'})", &r) == 0,
           "CREATE after MATCH");
    ASSERT(r.nodes_created == 1 && r.relationships_created == 1, "reused Alice, created Acme+rel");
    cypher_free_result(&r);

    /* Multi-hop MATCH */
    ASSERT(q(cy, "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a.name, b.name, c.name", &r) == 0, "multi-hop MATCH");
    ASSERT(r.row_count == 1 && strcmp(cell(&r,0,0),"Alice")==0 && strcmp(cell(&r,0,1),"Bob")==0
           && strcmp(cell(&r,0,2),"Carol")==0, "A-KNOWS->B-KNOWS->C");
    cypher_free_result(&r);

    /* WHERE numeric comparison + ORDER BY */
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.age > '28' RETURN p.name ORDER BY p.name", &r) == 0, "WHERE numeric");
    ASSERT(r.row_count == 2 && strcmp(cell(&r,0,0),"Alice")==0 && strcmp(cell(&r,1,0),"Carol")==0,
           "age>28 -> Alice,Carol (numeric-aware)");
    cypher_free_result(&r);

    /* WHERE OR */
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.name = 'Alice' OR p.name = 'Bob' RETURN p.name", &r) == 0, "WHERE OR");
    ASSERT(r.row_count == 2, "OR -> 2 rows");
    cypher_free_result(&r);

    /* WHERE CONTAINS */
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.name CONTAINS 'o' RETURN p.name ORDER BY p.name", &r) == 0, "WHERE CONTAINS");
    ASSERT(r.row_count == 2 && row_with(&r,0,"Bob")>=0 && row_with(&r,0,"Carol")>=0, "CONTAINS 'o' -> Bob,Carol");
    cypher_free_result(&r);

    /* Aggregation count(*) */
    ASSERT(q(cy, "MATCH (p:Person) RETURN count(*)", &r) == 0, "count(*)");
    ASSERT(r.row_count == 1 && strcmp(cell(&r,0,0),"3")==0, "3 persons");
    cypher_free_result(&r);

    /* Grouped aggregation */
    ASSERT(q(cy, "MATCH (a)-[:KNOWS]->(b) RETURN a.name, count(*) ORDER BY a.name", &r) == 0, "grouped count");
    ASSERT(r.column_count==2 && r.row_count == 2, "2 groups (Alice, Bob)");
    ASSERT(strcmp(cell(&r,0,0),"Alice")==0 && strcmp(cell(&r,0,1),"1")==0, "Alice:1");
    cypher_free_result(&r);

    /* collect */
    ASSERT(q(cy, "MATCH (a)-[:KNOWS]->(b) RETURN collect(b.name)", &r) == 0, "collect");
    ASSERT(r.row_count == 1 && strstr(cell(&r,0,0),"Bob") && strstr(cell(&r,0,0),"Carol"), "collect has Bob,Carol");
    cypher_free_result(&r);

    /* DISTINCT */
    ASSERT(q(cy, "MATCH (a)-[r]->(b) RETURN a.name", &r) == 0, "all outgoing");
    ASSERT(r.row_count == 3, "3 outgoing edges (Alice x2, Bob x1)");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (a)-[r]->(b) RETURN DISTINCT a.name", &r) == 0, "DISTINCT");
    ASSERT(r.row_count == 2, "DISTINCT a.name -> 2");
    cypher_free_result(&r);

    /* ORDER BY DESC + LIMIT / SKIP */
    ASSERT(q(cy, "MATCH (p:Person) RETURN p.name ORDER BY p.age DESC LIMIT 1", &r) == 0, "order desc limit");
    ASSERT(r.row_count == 1 && strcmp(cell(&r,0,0),"Carol")==0, "oldest = Carol");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person) RETURN p.name ORDER BY p.age SKIP 1 LIMIT 1", &r) == 0, "order skip limit");
    ASSERT(r.row_count == 1 && strcmp(cell(&r,0,0),"Alice")==0, "skip Bob(25) -> Alice(30)");
    cypher_free_result(&r);

    /* AS alias + type() */
    ASSERT(q(cy, "MATCH (a)-[r:KNOWS]->(b) RETURN type(r) AS rel, b.name AS who ORDER BY who", &r) == 0, "alias + type");
    ASSERT(r.column_count==2 && strcmp(r.column_names[0],"rel")==0 && strcmp(r.column_names[1],"who")==0, "aliases named");
    cypher_free_result(&r);

    /* SET */
    ASSERT(q(cy, "MATCH (p {name:'Bob'}) SET p.age = '26'", &r) == 0, "SET"); cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p {name:'Bob'}) RETURN p.age", &r) == 0, "read after SET");
    ASSERT(r.row_count == 1 && strcmp(cell(&r,0,0),"26")==0, "Bob.age = 26 after SET");
    cypher_free_result(&r);

    /* MERGE idempotent (existing) then MERGE new */
    ASSERT(q(cy, "MERGE (x:Person {name:'Alice'})", &r) == 0 && r.nodes_created == 0, "MERGE existing -> no create");
    cypher_free_result(&r);
    ASSERT(q(cy, "MERGE (z:Person {name:'Zed'})", &r) == 0 && r.nodes_created == 1, "MERGE new -> create");
    cypher_free_result(&r);

    /* DELETE */
    ASSERT(q(cy, "MATCH (p {name:'Zed'}) DELETE p", &r) == 0, "DELETE"); cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p {name:'Zed'}) RETURN p.name", &r) == 0 && r.row_count == 0, "Zed gone after DELETE");
    cypher_free_result(&r);

    /* OPTIONAL MATCH (Carol has no outgoing KNOWS) */
    ASSERT(q(cy, "MATCH (p:Person {name:'Carol'}) OPTIONAL MATCH (p)-[:KNOWS]->(x) RETURN p.name, x.name", &r) == 0,
           "OPTIONAL MATCH");
    ASSERT(r.row_count == 1 && strcmp(cell(&r,0,0),"Carol")==0 && strcmp(cell(&r,0,1),"")==0,
           "Carol kept, x unbound (empty)");
    cypher_free_result(&r);

    /* Unlabeled MATCH enumerates all nodes */
    ASSERT(q(cy, "MATCH (n) RETURN n.name", &r) == 0, "unlabeled MATCH");
    ASSERT(r.row_count == 4, "4 nodes total (Alice,Bob,Carol,Acme)");
    cypher_free_result(&r);

    /* ---- variable-length paths ---- */
    ASSERT(q(cy, "MATCH (a {name:'Alice'})-[:KNOWS*1..2]->(x) RETURN x.name ORDER BY x.name", &r) == 0
           && r.row_count == 2 && strcmp(cell(&r,0,0),"Bob")==0 && strcmp(cell(&r,1,0),"Carol")==0,
           "varlen *1..2 Alice -> Bob,Carol");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (a {name:'Alice'})-[:KNOWS*2]->(x) RETURN x.name", &r) == 0
           && r.row_count == 1 && strcmp(cell(&r,0,0),"Carol")==0, "varlen *2 -> Carol");
    cypher_free_result(&r);

    /* ---- RETURN * ---- */
    ASSERT(q(cy, "MATCH (a:Person {name:'Alice'})-[:KNOWS]->(b) RETURN *", &r) == 0 && r.column_count >= 2,
           "RETURN * returns bound vars");
    cypher_free_result(&r);

    /* ---- IN / IS NULL ---- */
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.name IN ['Alice','Carol'] RETURN p.name ORDER BY p.name", &r) == 0
           && r.row_count == 2, "IN list -> 2");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.age IS NOT NULL RETURN count(*)", &r) == 0
           && strcmp(cell(&r,0,0),"3")==0, "IS NOT NULL age -> 3 persons");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.city IS NULL RETURN count(*)", &r) == 0
           && strcmp(cell(&r,0,0),"3")==0, "IS NULL city -> all 3 (none have city)");
    cypher_free_result(&r);

    /* ---- scalar functions + arithmetic ---- */
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN toUpper(p.name)", &r) == 0
           && strcmp(cell(&r,0,0),"ALICE")==0, "toUpper");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN p.age + 5 AS a", &r) == 0
           && strcmp(cell(&r,0,0),"35")==0, "arithmetic p.age+5");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN 2 + 3 * 4", &r) == 0
           && strcmp(cell(&r,0,0),"14")==0, "arithmetic precedence");
    cypher_free_result(&r);

    /* ---- CASE (generic + simple) ---- */
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.name='Alice' RETURN CASE WHEN p.age > '28' THEN 'old' ELSE 'young' END", &r) == 0
           && strcmp(cell(&r,0,0),"old")==0, "generic CASE");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN CASE p.name WHEN 'Alice' THEN 'A' ELSE '?' END", &r) == 0
           && strcmp(cell(&r,0,0),"A")==0, "simple CASE");
    cypher_free_result(&r);

    /* ---- UNWIND ---- */
    ASSERT(q(cy, "UNWIND ['x','y','z'] AS v RETURN v ORDER BY v", &r) == 0 && r.row_count == 3
           && strcmp(cell(&r,0,0),"x")==0, "UNWIND list -> 3 rows");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['1','2','3'] AS n RETURN sum(n)", &r) == 0 && strcmp(cell(&r,0,0),"6")==0,
           "UNWIND + sum -> 6");
    cypher_free_result(&r);

    /* ---- WITH pipelining ---- */
    ASSERT(q(cy, "MATCH (a:Person {name:'Alice'}) WITH a MATCH (a)-[:KNOWS]->(b) RETURN b.name", &r) == 0
           && r.row_count == 1 && strcmp(cell(&r,0,0),"Bob")==0, "WITH pass-through var");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) WITH toUpper(p.name) AS u RETURN u", &r) == 0
           && strcmp(cell(&r,0,0),"ALICE")==0, "WITH computed alias carried to RETURN");
    cypher_free_result(&r);

    /* ---- REMOVE ---- */
    ASSERT(q(cy, "MATCH (p:Person {name:'Carol'}) REMOVE p.age", &r) == 0, "REMOVE p.age");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Carol'}) WHERE p.age IS NULL RETURN p.name", &r) == 0
           && r.row_count == 1, "Carol.age removed -> IS NULL");
    cypher_free_result(&r);

    /* ---- MERGE ON CREATE / ON MATCH SET ---- */
    ASSERT(q(cy, "MERGE (m:Marker {name:'M'}) ON CREATE SET m.state = 'new'", &r) == 0, "MERGE ON CREATE");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (m:Marker {name:'M'}) RETURN m.state", &r) == 0 && strcmp(cell(&r,0,0),"new")==0,
           "ON CREATE SET applied");
    cypher_free_result(&r);
    ASSERT(q(cy, "MERGE (m:Marker {name:'M'}) ON MATCH SET m.state = 'seen'", &r) == 0, "MERGE ON MATCH");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (m:Marker {name:'M'}) RETURN m.state", &r) == 0 && strcmp(cell(&r,0,0),"seen")==0,
           "ON MATCH SET applied");
    cypher_free_result(&r);

    /* ---- query parameters ---- */
    cypher_set_parameter(cy, "who", "Alice");
    ASSERT(q(cy, "MATCH (p:Person {name:$who}) RETURN p.name", &r) == 0 && r.row_count == 1
           && strcmp(cell(&r,0,0),"Alice")==0, "$param in pattern");
    cypher_free_result(&r);

    /* ---- EXISTS / NOT EXISTS ---- */
    ASSERT(q(cy, "MATCH (p:Person) WHERE EXISTS((p)-[:KNOWS]->()) RETURN p.name ORDER BY p.name", &r) == 0
           && r.row_count == 2 && strcmp(cell(&r,0,0),"Alice")==0, "EXISTS -> Alice,Bob");
    cypher_free_result(&r);

    /* ---- lists / range / indexing / comprehension ---- */
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN range(1,3)", &r) == 0
           && strcmp(cell(&r,0,0),"[1, 2, 3]")==0, "range()");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN [x IN range(1,5) WHERE x > '3' | x * 2]", &r) == 0
           && strcmp(cell(&r,0,0),"[8, 10]")==0, "list comprehension");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN [1,2,3][2]", &r) == 0
           && strcmp(cell(&r,0,0),"3")==0, "list index");
    cypher_free_result(&r);

    /* ---- maps ---- */
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN {a:'1', b:'2'}['b']", &r) == 0
           && strcmp(cell(&r,0,0),"2")==0, "map index");
    cypher_free_result(&r);

    /* ---- pattern comprehension ---- */
    ASSERT(q(cy, "MATCH (a:Person {name:'Alice'}) RETURN [(a)-[:KNOWS]->(x) | x.name] AS f", &r) == 0
           && strstr(cell(&r,0,0),"Bob"), "pattern comprehension");
    cypher_free_result(&r);

    /* ---- aggregating WITH ---- */
    ASSERT(q(cy, "MATCH (a)-[:KNOWS]->(b) WITH a, count(*) AS c WHERE c > '0' RETURN count(*)", &r) == 0,
           "aggregating WITH + post-filter");
    cypher_free_result(&r);

    /* ---- CALL ---- */
    ASSERT(q(cy, "CALL db.labels()", &r) == 0 && r.column_count == 1
           && strcmp(r.column_names[0],"label")==0, "CALL db.labels()");
    cypher_free_result(&r);

    /* ---- FOREACH + path variable ---- */
    ASSERT(q(cy, "FOREACH (x IN ['a','b'] | CREATE (:Tag {name: x}))", &r) == 0, "FOREACH create");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (t:Tag) RETURN count(*)", &r) == 0 && strcmp(cell(&r,0,0),"2")==0, "FOREACH made 2 Tags");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH pth = (a:Person {name:'Alice'})-[:KNOWS]->(b) RETURN length(pth)", &r) == 0
           && strcmp(cell(&r,0,0),"1")==0, "path variable length");
    cypher_free_result(&r);

    /* Syntax error still reported */
    ASSERT(q(cy, "MATCH (n:Person RETURN n.name", &r) == -1 && strlen(cypher_last_error(cy)) > 0, "syntax error");

    cypher_destroy(cy);
    kg_destroy(kg);
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL CYPHER TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
