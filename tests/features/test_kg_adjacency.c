/*
 * Direct-adjacency (constant-time hop) tests for the knowledge graph.
 *
 * Validates the kg_for_each_hop() pointer-deref traversal primitive and proves
 * the adjacency stays consistent across every structural mutation:
 * add/remove relation, remove entity (cascade), and merge entities.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "features/knowledge_graph.h"

static int failures = 0;
#define ASSERT(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } \
                          else { printf("ok: %s\n", (m)); } } while (0)

/* --- hop collector ------------------------------------------------------- */
typedef struct {
    uint64_t ids[64];
    char     preds[64][32];
    int      outgoing[64];
    int      n;
} HopSink;

static int sink(const GV_KGHop *h, void *ctx) {
    HopSink *s = (HopSink *)ctx;
    if (s->n >= 64) return 1;
    s->ids[s->n] = h->neighbor_id;
    snprintf(s->preds[s->n], sizeof(s->preds[s->n]), "%s", h->predicate ? h->predicate : "");
    s->outgoing[s->n] = h->outgoing;
    s->n++;
    return 0;
}

static int has_id(const HopSink *s, uint64_t id) {
    for (int i = 0; i < s->n; i++) if (s->ids[i] == id) return 1;
    return 0;
}
static int has_id_pred(const HopSink *s, uint64_t id, const char *pred) {
    for (int i = 0; i < s->n; i++)
        if (s->ids[i] == id && strcmp(s->preds[i], pred) == 0) return 1;
    return 0;
}

static HopSink hops(GV_KnowledgeGraph *kg, uint64_t id, int dir, const char *pred) {
    HopSink s; s.n = 0;
    kg_for_each_hop(kg, id, dir, pred, sink, &s);
    return s;
}

int main(void) {
    GV_KnowledgeGraph *kg = kg_create(NULL);

    /* Star + chain:  A -KNOWS-> B, A -KNOWS-> C, A -WORKS_AT-> Org,
     *                B -KNOWS-> C,  C -KNOWS-> A  (creates a cycle). */
    uint64_t A   = kg_add_entity(kg, "Alice", "Person", NULL, 0);
    uint64_t B   = kg_add_entity(kg, "Bob",   "Person", NULL, 0);
    uint64_t C   = kg_add_entity(kg, "Carol", "Person", NULL, 0);
    uint64_t Org = kg_add_entity(kg, "Acme",  "Company", NULL, 0);
    ASSERT(A && B && C && Org, "entities created");

    uint64_t rAB = kg_add_relation(kg, A, "KNOWS", B, 1.0f);
    uint64_t rAC = kg_add_relation(kg, A, "KNOWS", C, 1.0f);
    uint64_t rAO = kg_add_relation(kg, A, "WORKS_AT", Org, 1.0f);
    uint64_t rBC = kg_add_relation(kg, B, "KNOWS", C, 1.0f);
    uint64_t rCA = kg_add_relation(kg, C, "KNOWS", A, 1.0f);
    ASSERT(rAB && rAC && rAO && rBC && rCA, "relations created");

    /* Outgoing hops of A: KNOWS->B, KNOWS->C, WORKS_AT->Org */
    HopSink s = hops(kg, A, 1, NULL);
    ASSERT(s.n == 3, "A has 3 outgoing hops");
    ASSERT(has_id_pred(&s, B, "KNOWS"), "A -KNOWS-> B (out)");
    ASSERT(has_id_pred(&s, C, "KNOWS"), "A -KNOWS-> C (out)");
    ASSERT(has_id_pred(&s, Org, "WORKS_AT"), "A -WORKS_AT-> Org (out)");

    /* Predicate filter: only KNOWS out-edges of A */
    s = hops(kg, A, 1, "KNOWS");
    ASSERT(s.n == 2 && has_id(&s, B) && has_id(&s, C), "A KNOWS-filter -> B,C only");

    /* Incoming hops of A: only C -KNOWS-> A */
    s = hops(kg, A, -1, NULL);
    ASSERT(s.n == 1 && has_id(&s, C) && s.outgoing[0] == 0, "A has 1 incoming (from C)");

    /* Both directions on C: out C->A ; in A->C, B->C */
    s = hops(kg, C, 0, NULL);
    ASSERT(s.n == 3, "C has 3 hops both-dir (1 out + 2 in)");
    int outc = 0, inc = 0;
    for (int i = 0; i < s.n; i++) { if (s.outgoing[i]) outc++; else inc++; }
    ASSERT(outc == 1 && inc == 2, "C: 1 out, 2 in");

    /* --- remove a single relation: adjacency updates on both endpoints --- */
    ASSERT(kg_remove_relation(kg, rAC) == 0, "remove A-KNOWS->C");
    s = hops(kg, A, 1, "KNOWS");
    ASSERT(s.n == 1 && has_id(&s, B) && !has_id(&s, C), "A KNOWS-out now only B");
    s = hops(kg, C, -1, "KNOWS");
    ASSERT(s.n == 1 && has_id(&s, B) && !has_id(&s, A), "C KNOWS-in now only from B");

    /* --- remove an entity: cascade removes its edges from neighbours too --- */
    ASSERT(kg_remove_entity(kg, B) == 0, "remove entity B");
    /* A had A-KNOWS->B (gone). A out now: WORKS_AT->Org only. */
    s = hops(kg, A, 1, NULL);
    ASSERT(s.n == 1 && has_id(&s, Org), "after B removed, A out = Org only");
    /* C had B-KNOWS->C incoming (gone) and C-KNOWS->A outgoing (kept). */
    s = hops(kg, C, 0, NULL);
    ASSERT(s.n == 1 && has_id(&s, A) && s.outgoing[0] == 1, "after B removed, C = C->A only");

    /* --- rebuild graph and test merge --------------------------------- */
    kg_destroy(kg);
    kg = kg_create(NULL);
    uint64_t X = kg_add_entity(kg, "X", "N", NULL, 0);
    uint64_t Y = kg_add_entity(kg, "Y", "N", NULL, 0);
    uint64_t Z = kg_add_entity(kg, "Z", "N", NULL, 0);
    uint64_t W = kg_add_entity(kg, "W", "N", NULL, 0);
    kg_add_relation(kg, X, "R", Z, 1.0f);   /* X -> Z */
    kg_add_relation(kg, Z, "R", Y, 1.0f);   /* Z -> Y */
    kg_add_relation(kg, W, "R", Y, 1.0f);   /* W -> Y */
    /* Merge Y into X: Z->Y and W->Y become Z->X and W->X (X inherits both). */
    ASSERT(kg_merge_entities(kg, X, Y) == 0, "merge Y into X");
    s = hops(kg, X, -1, NULL);
    ASSERT(s.n == 2 && has_id(&s, Z) && has_id(&s, W),
           "post-merge: X inherited incoming from Z and W");
    s = hops(kg, Z, 1, NULL);
    ASSERT(s.n == 1 && has_id(&s, X), "post-merge: Z->Y is now Z->X");
    /* Y is gone: querying it yields no hops (and no crash). */
    s = hops(kg, Y, 0, NULL);
    ASSERT(s.n == 0, "merged-away Y has no hops");

    /* --- persistence: adjacency rebuilt on load ----------------------- */
    const char *path = "test_kg_adj.db";
    ASSERT(kg_save(kg, path) == 0, "kg_save");
    kg_destroy(kg);
    kg = kg_load(path);
    ASSERT(kg != NULL, "kg_load");
    if (kg) {
        s = hops(kg, X, -1, NULL);
        ASSERT(s.n == 2 && has_id(&s, Z) && has_id(&s, W),
               "loaded: X incoming from Z and W (adjacency rebuilt)");
    }
    remove(path);

    kg_destroy(kg);

    if (failures == 0) { printf("\nALL KG ADJACENCY TESTS PASSED\n"); return 0; }
    printf("\n%d FAILURE(S)\n", failures);
    return 1;
}
