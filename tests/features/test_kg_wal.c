#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "features/knowledge_graph.h"
#include "../test_tmp.h"

#define ASSERT(cond, msg)         \
    do {                          \
        if (!(cond)) {            \
            fprintf(stderr, "FAIL: %s\n", msg); \
            return -1;            \
        }                         \
    } while (0)

static int test_kg_wal_replay_after_crash(void) {
    char path[512];
    ASSERT(gv_test_make_temp_path(path, sizeof(path), "gv_kgwal", ".gvkg") == 0,
           "temp path");

    GV_KnowledgeGraph *kg = kg_create(NULL);
    ASSERT(kg != NULL, "create");
    ASSERT(kg_save(kg, path) == 0, "initial snapshot save");
    ASSERT(kg_wal_attach(kg, path) == 0, "wal attach");

    uint64_t e1 = kg_add_entity(kg, "Alice", "Person", NULL, 0);
    uint64_t e2 = kg_add_entity(kg, "ACME", "Company", NULL, 0);
    uint64_t e3 = kg_add_entity(kg, "Bob", "Person", NULL, 0);
    ASSERT(e1 && e2 && e3, "add entities");
    float emb[4] = {1.0f, 0.0f, 0.5f, 0.25f};
    uint64_t e4 = kg_add_entity(kg, "Embed", "Thing", emb, 4);
    ASSERT(e4, "add entity with embedding");

    uint64_t r1 = kg_add_relation(kg, e1, "works_at", e2, 1.0f);
    uint64_t r2 = kg_add_relation(kg, e3, "knows", e1, 0.9f);
    ASSERT(r1 && r2, "add relations");
    ASSERT(kg_set_entity_prop(kg, e1, "role", "engineer") == 0, "entity prop");
    ASSERT(kg_set_relation_prop(kg, r1, "chunk_id", "doc:0001") == 0,
           "relation chunk prop");

    /* Mutations that must also survive replay. */
    uint64_t e5 = kg_add_entity(kg, "Temp", "Person", NULL, 0);
    ASSERT(e5, "add temp entity");
    uint64_t r3 = kg_add_relation(kg, e5, "likes", e1, 1.0f);
    ASSERT(r3, "add temp relation");
    ASSERT(kg_remove_relation(kg, r2) == 0, "remove relation r2");
    ASSERT(kg_remove_entity(kg, e5) == 0, "remove entity e5 (cascades r3)");

    /* Crash: destroy without saving. */
    kg_destroy(kg);

    GV_KnowledgeGraph *r = kg_load(path);
    ASSERT(r != NULL, "load + replay");

    /* Survivors: Alice, ACME, Bob, Embed (Temp removed). */
    uint64_t by_name[8];
    int n = kg_find_entities_by_name(r, "Alice", by_name, 8);
    ASSERT(n == 1 && by_name[0] == e1, "Alice survived with same id");
    n = kg_find_entities_by_name(r, "Temp", by_name, 8);
    ASSERT(n == 0, "Temp stayed removed");
    n = kg_find_entities_by_type(r, "Company", by_name, 8);
    ASSERT(n == 1 && by_name[0] == e2, "type index intact after replay");

    /* r2 was explicitly removed; r3 cascaded away with e5. */
    GV_KGTriple tr[8];
    int nt = kg_query_triples(r, &e3, NULL, NULL, tr, 8);
    ASSERT(nt == 0, "removed relation stayed removed");
    nt = kg_query_triples(r, &e1, NULL, NULL, tr, 8);
    ASSERT(nt == 1, "r1 survived");
    kg_free_triples(tr, nt);

    const GV_KGRelation *rel = kg_get_relation(r, r1);
    ASSERT(rel != NULL, "relation id preserved");
    ASSERT(strcmp(rel->predicate, "works_at") == 0, "predicate preserved");
    ASSERT(kg_remove_relations_by_chunk(r, "doc:0001") == 1,
           "chunk index rebuilt on replayed prop");

    /* Replayed embedding is queryable. */
    const GV_KGEntity *ent = kg_get_entity(r, e4);
    ASSERT(ent != NULL && ent->dimension == 4 &&
           ent->embedding && ent->embedding[0] == 1.0f,
           "replayed embedding intact");

    /* WAL stays attached post-recovery. */
    uint64_t e6 = kg_add_entity(r, "PostRecovery", "Person", NULL, 0);
    ASSERT(e6 > 0, "post-recovery mutation");
    kg_destroy(r);

    GV_KnowledgeGraph *kg2 = kg_load(path);
    ASSERT(kg2 != NULL, "second reload");
    n = kg_find_entities_by_name(kg2, "PostRecovery", by_name, 8);
    ASSERT(n == 1, "second reload includes recovered+new ops");
    kg_destroy(kg2);

    unlink(path);
    char wal_path[540];
    snprintf(wal_path, sizeof(wal_path), "%s.wal", path);
    unlink(wal_path);
    return 0;
}

static int test_kg_wal_checkpoint_on_save(void) {
    char path[512];
    ASSERT(gv_test_make_temp_path(path, sizeof(path), "gv_kgwal_ckpt", ".gvkg") == 0,
           "temp path");

    GV_KnowledgeGraph *kg = kg_create(NULL);
    ASSERT(kg != NULL, "create");
    ASSERT(kg_save(kg, path) == 0, "save empty");
    ASSERT(kg_wal_attach(kg, path) == 0, "attach");

    ASSERT(kg_add_entity(kg, "X", "Y", NULL, 0) > 0, "add node");
    ASSERT(kg_save(kg, path) == 0, "checkpointing save");

    kg_destroy(kg);

    GV_KnowledgeGraph *r = kg_load(path);
    ASSERT(r != NULL, "reload after checkpoint");
    uint64_t ids[8];
    ASSERT(kg_find_entities_by_type(r, "Y", ids, 8) == 1,
           "state intact after checkpoint");
    kg_destroy(r);

    unlink(path);
    char wal_path[540];
    snprintf(wal_path, sizeof(wal_path), "%s.wal", path);
    unlink(wal_path);
    return 0;
}

int main(void) {
    if (test_kg_wal_replay_after_crash() != 0) return 1;
    printf("ok: KG WAL replay after crash\n");
    if (test_kg_wal_checkpoint_on_save() != 0) return 1;
    printf("ok: KG WAL checkpoint on save\n");
    printf("All KG-WAL tests PASSED\n");
    return 0;
}
