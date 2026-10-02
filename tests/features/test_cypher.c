/* Comprehensive Cypher-over-KG tests: multi-hop, WHERE, aggregation, ORDER BY,
   DISTINCT, SET, DELETE, MERGE, OPTIONAL MATCH. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
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
    /* same comparison with an unquoted numeric literal */
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.age > 28 RETURN p.name ORDER BY p.name", &r) == 0, "WHERE numeric literal");
    ASSERT(r.row_count == 2 && strcmp(cell(&r,0,0),"Alice")==0 && strcmp(cell(&r,1,0),"Carol")==0,
           "age>28 (unquoted literal) -> Alice,Carol");
    cypher_free_result(&r);
    /* lexical trap: '9' > '30' as strings but not numerically */
    ASSERT(q(cy, "MATCH (p:Person {name:'Bob'}) WHERE p.age > 9 RETURN p.name", &r) == 0, "numeric trap setup");
    ASSERT(r.row_count == 1, "25 > 9 numerically (would fail lexically)");
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

    /* ---- edge direction over the O(1) adjacency ---- */
    /* Reverse pattern: who KNOWS Carol?  (Bob-KNOWS->Carol) — uses incoming adjacency. */
    ASSERT(q(cy, "MATCH (c:Person {name:'Carol'})<-[:KNOWS]-(x) RETURN x.name ORDER BY x.name", &r) == 0
           && r.row_count == 1 && strcmp(cell(&r,0,0),"Bob")==0, "reverse <-[:KNOWS]- Carol -> Bob");
    cypher_free_result(&r);
    /* Undirected: Bob's KNOWS neighbours in either direction (Alice in, Carol out). */
    ASSERT(q(cy, "MATCH (b:Person {name:'Bob'})-[:KNOWS]-(x) RETURN x.name ORDER BY x.name", &r) == 0
           && r.row_count == 2 && strcmp(cell(&r,0,0),"Alice")==0 && strcmp(cell(&r,1,0),"Carol")==0,
           "undirected -[:KNOWS]- Bob -> Alice,Carol");
    cypher_free_result(&r);
    /* Reverse multi-hop chain: Carol <- Bob <- Alice. */
    ASSERT(q(cy, "MATCH (c {name:'Carol'})<-[:KNOWS]-(b)<-[:KNOWS]-(a) RETURN a.name", &r) == 0
           && r.row_count == 1 && strcmp(cell(&r,0,0),"Alice")==0, "reverse 2-hop -> Alice");
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

    /* ---- additional scalar functions (replace/reverse/left/right/toBoolean/pow/split) ---- */
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN replace('hello','l','L')", &r) == 0
           && strcmp(cell(&r,0,0),"heLLo")==0, "replace");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN reverse('abc')", &r) == 0
           && strcmp(cell(&r,0,0),"cba")==0, "reverse");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN left('hello',2)", &r) == 0
           && strcmp(cell(&r,0,0),"he")==0, "left");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN right('hello',2)", &r) == 0
           && strcmp(cell(&r,0,0),"lo")==0, "right");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN toBoolean('true')", &r) == 0
           && strcmp(cell(&r,0,0),"true")==0, "toBoolean");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN pow(2,10)", &r) == 0
           && strncmp(cell(&r,0,0),"1024",4)==0, "pow");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN size(split('a,b,c',','))", &r) == 0
           && strcmp(cell(&r,0,0),"3")==0, "split+size");
    cypher_free_result(&r);
    /* trigonometric / angle math functions */
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN sin(0)", &r) == 0
           && strncmp(cell(&r,0,0),"0",1)==0, "sin(0)=0");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN cos(0)", &r) == 0
           && strncmp(cell(&r,0,0),"1",1)==0, "cos(0)=1");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN degrees(pi())", &r) == 0
           && strncmp(cell(&r,0,0),"180",3)==0, "degrees(pi)=180");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND ['x'] AS v RETURN atan2(1,1)", &r) == 0
           && strncmp(cell(&r,0,0),"0.785",5)==0, "atan2(1,1)=pi/4");
    cypher_free_result(&r);

    /* stDev / stDevP aggregates over a fresh, isolated node set {10,20,30} */
    ASSERT(q(cy, "CREATE (:Measure {v:'10'}),(:Measure {v:'20'}),(:Measure {v:'30'})", &r) == 0, "create Measure nodes");
    cypher_free_result(&r);
    /* population stdev of {10,20,30}: mean 20, var=(100+0+100)/3=66.67, sqrt~=8.165 */
    ASSERT(q(cy, "MATCH (m:Measure) RETURN stDevP(m.v)", &r) == 0
           && r.row_count == 1 && strncmp(cell(&r,0,0),"8.16",4)==0, "stDevP({10,20,30}) ~= 8.16");
    cypher_free_result(&r);
    /* sample stdev of {10,20,30} = sqrt(200/2) = 10 */
    ASSERT(q(cy, "MATCH (m:Measure) RETURN stDev(m.v)", &r) == 0
           && r.row_count == 1 && strncmp(cell(&r,0,0),"10",2)==0, "stDev({10,20,30}) = 10");
    cypher_free_result(&r);
    /* same aggregate through a WITH projection (separate aggregation path) */
    ASSERT(q(cy, "MATCH (m:Measure) WITH stDevP(m.v) AS s RETURN s", &r) == 0
           && r.row_count == 1 && strncmp(cell(&r,0,0),"8.16",4)==0, "WITH stDevP ~= 8.16");
    cypher_free_result(&r);
    /* percentileCont(0.5) over {10,20,30} = 20 (interpolated median) */
    ASSERT(q(cy, "MATCH (m:Measure) RETURN percentileCont(m.v, 0.5)", &r) == 0
           && r.row_count == 1 && strncmp(cell(&r,0,0),"20",2)==0, "percentileCont(0.5)=20");
    cypher_free_result(&r);
    /* percentileDisc(0.0) = min = 10; percentileDisc(1.0) = max = 30 */
    ASSERT(q(cy, "MATCH (m:Measure) RETURN percentileDisc(m.v, 0)", &r) == 0
           && strncmp(cell(&r,0,0),"10",2)==0, "percentileDisc(0)=10");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (m:Measure) RETURN percentileDisc(m.v, 1)", &r) == 0
           && strncmp(cell(&r,0,0),"30",2)==0, "percentileDisc(1)=30");
    cypher_free_result(&r);
    /* percentileCont through WITH projection (separate aggregation path) */
    ASSERT(q(cy, "MATCH (m:Measure) WITH percentileCont(m.v, 0.5) AS p RETURN p", &r) == 0
           && r.row_count == 1 && strncmp(cell(&r,0,0),"20",2)==0, "WITH percentileCont(0.5)=20");
    cypher_free_result(&r);

    /* ---- list predicate functions any/all/none/single ---- */
    ASSERT(q(cy, "UNWIND [1] AS z RETURN any(x IN [1,2,3] WHERE x > 2)", &r) == 0
           && strcmp(cell(&r,0,0),"true")==0, "any(x>2)=true");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND [1] AS z RETURN all(x IN [1,2,3] WHERE x > 1)", &r) == 0
           && strcmp(cell(&r,0,0),"false")==0, "all(x>1)=false");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND [1] AS z RETURN none(x IN [1,2,3] WHERE x > 5)", &r) == 0
           && strcmp(cell(&r,0,0),"true")==0, "none(x>5)=true");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND [1] AS z RETURN single(x IN [1,2,3] WHERE x = 2)", &r) == 0
           && strcmp(cell(&r,0,0),"true")==0, "single(x=2)=true");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND [1] AS z RETURN single(x IN [1,2,3] WHERE x > 1)", &r) == 0
           && strcmp(cell(&r,0,0),"false")==0, "single(x>1)=false (two match)");
    cypher_free_result(&r);
    /* reduce: left fold */
    ASSERT(q(cy, "UNWIND [1] AS z RETURN reduce(s = 0, x IN [1,2,3,4] | s + x)", &r) == 0
           && strncmp(cell(&r,0,0),"10",2)==0, "reduce sum = 10");
    cypher_free_result(&r);
    ASSERT(q(cy, "UNWIND [1] AS z RETURN reduce(p = 1, x IN [1,2,3,4] | p * x)", &r) == 0
           && strncmp(cell(&r,0,0),"24",2)==0, "reduce product = 24");
    cypher_free_result(&r);

    /* ---- vector distance predicates + variable-length paths ---- */
    ASSERT(q(cy, "CREATE (va:Vec {name:'va', emb:'[1,0,0]'})-[:LINK]->(vb:Vec {name:'vb', emb:'[0,1,0]'})"
                 "-[:LINK]->(vc:Vec {name:'vc', emb:'[0.9,0.1,0]'})", &r) == 0, "create vec chain");
    cypher_free_result(&r);
    /* bracketed vectors parse; L2 distance computed */
    ASSERT(q(cy, "MATCH (v:Vec {name:'va'}) RETURN vector_distance(v.emb, '[1,0,0]')", &r) == 0
           && strncmp(cell(&r,0,0),"0",1)==0, "vector_distance(self)=0");
    cypher_free_result(&r);
    /* vector predicate in WHERE filters correctly (excludes the far vb) */
    ASSERT(q(cy, "MATCH (v:Vec) WHERE vector_distance(v.emb, '[1,0,0]') < 0.5 RETURN v.name ORDER BY v.name", &r) == 0
           && r.row_count == 2 && strcmp(cell(&r,0,0),"va")==0 && strcmp(cell(&r,1,0),"vc")==0,
           "WHERE vector_distance < 0.5 -> va,vc (not vb)");
    cypher_free_result(&r);
    /* cosine metric: parallel vector has distance 0 */
    ASSERT(q(cy, "MATCH (v:Vec {name:'va'}) RETURN vector_distance_cosine(v.emb, '[2,0,0]')", &r) == 0
           && strncmp(cell(&r,0,0),"0",1)==0, "cosine of parallel vectors = 0");
    cypher_free_result(&r);
    /* HEADLINE: variable-length path + vector predicate combined */
    ASSERT(q(cy, "MATCH (a:Vec {name:'va'})-[:LINK*1..2]->(x) WHERE vector_distance(x.emb, '[1,0,0]') < 0.5 "
                 "RETURN x.name ORDER BY x.name", &r) == 0
           && r.row_count == 1 && strcmp(cell(&r,0,0),"vc")==0,
           "varlen path + vector predicate -> only vc");
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

    /* ---- relationship properties ---- */
    /* CREATE an edge carrying properties */
    ASSERT(q(cy, "MATCH (a:Person {name:'Alice'}), (b:Person {name:'Bob'}) "
                 "CREATE (a)-[:RATED {score:'5', note:'great'}]->(b)", &r) == 0, "CREATE rel with props");
    cypher_free_result(&r);
    /* read a relationship property via r.prop */
    ASSERT(q(cy, "MATCH (a:Person {name:'Alice'})-[r:RATED]->(b) RETURN r.score, r.note", &r) == 0
           && r.row_count == 1 && strcmp(cell(&r,0,0),"5")==0 && strcmp(cell(&r,0,1),"great")==0,
           "read rel props r.score, r.note");
    cypher_free_result(&r);
    /* filter on a relationship property inside the pattern */
    ASSERT(q(cy, "MATCH (a)-[r:RATED {score:'5'}]->(b) RETURN a.name, b.name", &r) == 0
           && r.row_count == 1 && strcmp(cell(&r,0,0),"Alice")==0, "inline rel-prop filter matches");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (a)-[r:RATED {score:'1'}]->(b) RETURN a.name", &r) == 0 && r.row_count == 0,
           "inline rel-prop filter excludes");
    cypher_free_result(&r);
    /* SET a relationship property, then read it back */
    ASSERT(q(cy, "MATCH (a:Person {name:'Alice'})-[r:RATED]->(b) SET r.score = '9'", &r) == 0, "SET rel prop");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH ()-[r:RATED]->() RETURN r.score", &r) == 0 && strcmp(cell(&r,0,0),"9")==0,
           "rel prop updated to 9");
    cypher_free_result(&r);
    /* type(r) still yields the relationship type */
    ASSERT(q(cy, "MATCH ()-[r:RATED]->() RETURN type(r)", &r) == 0 && strcmp(cell(&r,0,0),"RATED")==0,
           "type(r) still works");
    cypher_free_result(&r);

    /* ---- computed SET (expression, not just literal) ---- */
    ASSERT(q(cy, "MATCH (p:Person {name:'Bob'}) SET p.age = '25'", &r) == 0, "reset Bob age");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Bob'}) SET p.age = p.age + '10'", &r) == 0, "computed SET p.age+10");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Bob'}) RETURN p.age", &r) == 0 && strcmp(cell(&r,0,0),"35")==0,
           "Bob age 25 -> 35 via expression");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Bob'}) SET p.label = toUpper(p.name)", &r) == 0, "computed SET with function");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Bob'}) RETURN p.label", &r) == 0 && strcmp(cell(&r,0,0),"BOB")==0,
           "SET p.label = toUpper(name)");
    cypher_free_result(&r);

    /* ---- rel-type alternation :A|B ---- */
    ASSERT(q(cy, "MATCH (a:Person {name:'Alice'})-[r:KNOWS|RATED]->(b) RETURN b.name ORDER BY b.name", &r) == 0
           && r.row_count == 2, "alternation KNOWS|RATED -> 2 edges");
    cypher_free_result(&r);

    /* ---- multi-label node ---- */
    ASSERT(q(cy, "CREATE (:Person:Person {name:'Zed'})", &r) == 0, "create multi-label node");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (n:Person:Person {name:'Zed'}) RETURN n.name", &r) == 0
           && r.row_count == 1 && strcmp(cell(&r,0,0),"Zed")==0, "multi-label match (all labels)");
    cypher_free_result(&r);

    /* ---- WITH DISTINCT dedup ---- */
    /* Many :Person nodes all share the label "Person"; projecting it via
     * WITH DISTINCT must collapse to a single row, vs one row per node without. */
    {
        GV_CypherResult r2;
        ASSERT(q(cy, "MATCH (p:Person) WITH labels(p) AS l RETURN count(*)", &r2) == 0, "count persons");
        long persons = atol(cell(&r2, 0, 0));
        cypher_free_result(&r2);
        ASSERT(persons > 1, "more than one Person exists");
        ASSERT(q(cy, "MATCH (p:Person) WITH DISTINCT labels(p) AS l RETURN count(*)", &r2) == 0,
               "WITH DISTINCT label");
        ASSERT(r2.row_count == 1 && strcmp(cell(&r2,0,0),"1")==0, "DISTINCT label collapses to 1 group");
        cypher_free_result(&r2);
    }

    /* ---- Typed value system ---- (anchored on a single-row MATCH for a projection context) */
    #define A "MATCH (p:Person {name:'Alice'}) RETURN "
    /* Integer arithmetic: division and modulo are integer-valued when both operands are integers. */
    ASSERT(q(cy, A "7/2 AS x", &r) == 0 && strcmp(cell(&r,0,0),"3")==0, "integer division 7/2 -> 3");
    cypher_free_result(&r);
    ASSERT(q(cy, A "7.0/2 AS x", &r) == 0 && strcmp(cell(&r,0,0),"3.5")==0, "float division 7.0/2 -> 3.5");
    cypher_free_result(&r);
    ASSERT(q(cy, A "7 % 3 AS x", &r) == 0 && strcmp(cell(&r,0,0),"1")==0, "modulo 7 % 3 -> 1");
    cypher_free_result(&r);
    ASSERT(q(cy, A "2 + 3 * 4 AS x", &r) == 0 && strcmp(cell(&r,0,0),"14")==0, "integer 2+3*4 -> 14");
    cypher_free_result(&r);
    /* Boolean literal renders as true/false. */
    ASSERT(q(cy, A "true AS x", &r) == 0 && strcmp(cell(&r,0,0),"true")==0, "boolean literal true");
    cypher_free_result(&r);
    /* NULL literal is distinct from empty string: IS NULL true, coalesce skips it. */
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) WHERE null IS NULL RETURN p.name", &r) == 0
           && r.row_count == 1, "null literal IS NULL");
    cypher_free_result(&r);
    ASSERT(q(cy, A "coalesce(null, 'fallback') AS x", &r) == 0 && strcmp(cell(&r,0,0),"fallback")==0, "coalesce skips null");
    cypher_free_result(&r);
    /* Three-valued logic: comparison against null is never true (all rows filtered out). */
    ASSERT(q(cy, "MATCH (p:Person) WHERE p.age > null RETURN p.name", &r) == 0
           && r.row_count == 0, "comparison with null -> no rows");
    cypher_free_result(&r);
    /* Type conversions. */
    ASSERT(q(cy, A "toInteger('3.7') AS x", &r) == 0 && strcmp(cell(&r,0,0),"3")==0, "toInteger('3.7') -> 3");
    cypher_free_result(&r);
    ASSERT(q(cy, A "toString(42) AS x", &r) == 0 && strcmp(cell(&r,0,0),"42")==0, "toString(42) -> '42'");
    cypher_free_result(&r);
    ASSERT(q(cy, A "toFloat('3') / 2 AS x", &r) == 0 && strcmp(cell(&r,0,0),"1.5")==0, "toFloat('3')/2 -> 1.5");
    cypher_free_result(&r);
    /* CASE returning a typed integer participates in integer arithmetic downstream. */
    ASSERT(q(cy, A "(CASE WHEN true THEN 10 ELSE 0 END) / 4 AS x", &r) == 0
           && strcmp(cell(&r,0,0),"2")==0, "typed CASE integer division");
    cypher_free_result(&r);
    #undef A
    /* Bare boolean in predicate position (truthiness). */
    GV_CypherResult rall;
    ASSERT(q(cy, "MATCH (p:Person) RETURN p.name", &rall) == 0 && rall.row_count > 0, "person rows exist");
    ASSERT(q(cy, "MATCH (p:Person) WHERE true RETURN p.name", &r) == 0
           && r.row_count == rall.row_count, "WHERE true keeps all rows");
    cypher_free_result(&r);
    cypher_free_result(&rall);
    ASSERT(q(cy, "MATCH (p:Person) WHERE false RETURN p.name", &r) == 0
           && r.row_count == 0, "WHERE false drops all rows");
    cypher_free_result(&r);

    /* Baseline person count for the UNION/CALL assertions below. */
    GV_CypherResult rbase;
    ASSERT(q(cy, "MATCH (p:Person) RETURN p.name", &rbase) == 0 && rbase.row_count > 0, "person baseline rows exist");
    size_t np = rbase.row_count;
    cypher_free_result(&rbase);

    /* UNION: the combined set is de-duplicated back to the baseline. */
    ASSERT(q(cy, "MATCH (p:Person) RETURN p.name UNION MATCH (p:Person) RETURN p.name", &r) == 0
           && r.row_count == np, "UNION dedups to baseline persons");
    cypher_free_result(&r);
    /* UNION ALL: duplicates are kept (2 x baseline). */
    ASSERT(q(cy, "MATCH (p:Person) RETURN p.name UNION ALL MATCH (p:Person) RETURN p.name", &r) == 0
           && r.row_count == 2 * np, "UNION ALL keeps all rows");
    cypher_free_result(&r);
    /* UNION with mismatched column counts is an error. */
    ASSERT(q(cy, "MATCH (p:Person) RETURN p.name UNION MATCH (p:Person) RETURN p.name, p.age", &r) == -1
           && strlen(cypher_last_error(cy)) > 0, "UNION column count mismatch errors");

    /* CALL { subquery }: inner rows become the statement result. */
    ASSERT(q(cy, "CALL { MATCH (p:Person) RETURN p.name }", &r) == 0
           && r.row_count == np && r.column_count == 1, "CALL { subquery } returns baseline rows");
    cypher_free_result(&r);

    /* Temporal scalar functions. */
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN timestamp() AS t", &r) == 0
           && atoll(cell(&r,0,0)) > 0, "timestamp() > 0");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN date('2020-01-02') AS d", &r) == 0
           && strcmp(cell(&r,0,0),"2020-01-02")==0, "date('2020-01-02') echoes ISO date");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN datetime('2020-01-02T03:04:05') AS d", &r) == 0
           && strcmp(cell(&r,0,0),"2020-01-02T03:04:05")==0, "datetime(iso) echoes ISO datetime");
    cypher_free_result(&r);
    ASSERT(q(cy, "MATCH (p:Person {name:'Alice'}) RETURN date() AS d", &r) == 0
           && strlen(cell(&r,0,0))==10, "date() -> YYYY-MM-DD (10 chars)");
    cypher_free_result(&r);

    /* Index DDL: CREATE/DROP INDEX (advisory) + SHOW INDEXES / CALL db.indexes */
    ASSERT(q(cy, "CREATE INDEX ON :Person(name)", &r) == 0, "CREATE INDEX legacy form");
    cypher_free_result(&r);
    ASSERT(q(cy, "CREATE INDEX FOR (p:Person) ON (p.age)", &r) == 0, "CREATE INDEX descriptor form");
    cypher_free_result(&r);
    ASSERT(q(cy, "CREATE INDEX ON :Person(name)", &r) == 0, "CREATE INDEX idempotent");
    cypher_free_result(&r);
    ASSERT(q(cy, "SHOW INDEXES", &r) == 0 && r.row_count == 2, "SHOW INDEXES lists 2");
    ASSERT(row_with(&r,0,"Person(name)") >= 0 && row_with(&r,0,"Person(age)") >= 0, "both indexes present");
    cypher_free_result(&r);
    ASSERT(q(cy, "CALL db.indexes()", &r) == 0 && r.row_count == 2, "CALL db.indexes -> 2");
    cypher_free_result(&r);
    ASSERT(q(cy, "DROP INDEX ON :Person(name)", &r) == 0, "DROP INDEX");
    cypher_free_result(&r);
    ASSERT(q(cy, "SHOW INDEXES", &r) == 0 && r.row_count == 1 && row_with(&r,0,"Person(age)") >= 0, "1 index after drop");
    cypher_free_result(&r);

    /* Syntax error still reported */
    ASSERT(q(cy, "MATCH (n:Person RETURN n.name", &r) == -1 && strlen(cypher_last_error(cy)) > 0, "syntax error");

    cypher_destroy(cy);
    kg_destroy(kg);
    printf(failures ? "\nSOME TESTS FAILED (%d)\n" : "\nALL CYPHER TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
