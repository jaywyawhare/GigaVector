/**
 * test_posting_bitmap.c — validates the Dgraph-style roaring-bitmap posting
 * conversions: the metadata inverted index (key=value -> roaring UID set, with
 * AND-based multi-filter) and the knowledge-graph typed sets (predicate ->
 * relation set, entity -> neighbour set, joined by bitmap intersection).
 */
#include <stdio.h>
#include <string.h>

#include "multimodal/metadata_index.h"
#include "features/knowledge_graph.h"
#include "core/id_bitmap.h"

static int failures = 0;
#define ASSERT(c, m) do { \
    if (!(c)) { printf("FAIL: %s:%d: %s\n", __FILE__, __LINE__, (m)); failures++; } \
    else { printf("ok: %s\n", (m)); } } while (0)

static int test_metadata_bitmap(void) {
    GV_MetadataIndex *idx = metadata_index_create();
    ASSERT(idx != NULL, "metadata index created");
    /* color=red -> {1,3,5}; size=big -> {3,5,7} */
    metadata_index_add(idx, "color", "red", 1);
    metadata_index_add(idx, "color", "red", 3);
    metadata_index_add(idx, "color", "red", 5);
    metadata_index_add(idx, "color", "red", 3);   /* dup -> bitmap dedups */
    metadata_index_add(idx, "size", "big", 3);
    metadata_index_add(idx, "size", "big", 5);
    metadata_index_add(idx, "size", "big", 7);

    ASSERT(metadata_index_count(idx, "color", "red") == 3, "color=red posting has 3 (deduped)");

    const GV_IdBitmap *red = metadata_index_query_bitmap(idx, "color", "red");
    const GV_IdBitmap *big = metadata_index_query_bitmap(idx, "size", "big");
    ASSERT(red && big, "both posting bitmaps present");

    /* Dgraph-style AND filter: color=red AND size=big -> {3,5} */
    GV_IdBitmap *both = gv_id_bitmap_and(red, big);
    ASSERT(both != NULL, "bitmap AND produced");
    ASSERT(gv_id_bitmap_cardinality(both) == 2, "red AND big == {3,5} (card 2)");
    ASSERT(gv_id_bitmap_contains(both, 3) && gv_id_bitmap_contains(both, 5), "AND contains 3 and 5");
    ASSERT(!gv_id_bitmap_contains(both, 1) && !gv_id_bitmap_contains(both, 7), "AND excludes 1 and 7");
    gv_id_bitmap_free(both);

    /* absent pair -> NULL bitmap */
    ASSERT(metadata_index_query_bitmap(idx, "color", "green") == NULL, "absent pair -> NULL");

    /* remove_vector drops it from all postings */
    metadata_index_remove_vector(idx, 3);
    ASSERT(metadata_index_count(idx, "color", "red") == 2, "after remove(3), red has 2");
    ASSERT(!gv_id_bitmap_contains(metadata_index_query_bitmap(idx, "size", "big"), 3),
           "3 removed from size=big too");

    metadata_index_destroy(idx);
    return 0;
}

static int test_kg_typed_sets(void) {
    GV_KGConfig cfg; kg_config_init(&cfg); cfg.embedding_dimension = 0;
    GV_KnowledgeGraph *kg = kg_create(&cfg);
    ASSERT(kg != NULL, "kg created");

    uint64_t A = kg_add_entity(kg, "A", "Node", NULL, 0);
    uint64_t B = kg_add_entity(kg, "B", "Node", NULL, 0);
    uint64_t C = kg_add_entity(kg, "C", "Node", NULL, 0);
    uint64_t D = kg_add_entity(kg, "D", "Node", NULL, 0);
    /* A-knows->C, B-knows->C, A-likes->D */
    kg_add_relation(kg, A, "knows", C, 1.0f);
    kg_add_relation(kg, B, "knows", C, 1.0f);
    kg_add_relation(kg, A, "likes", D, 1.0f);

    /* predicate posting: "knows" has 2 relations */
    GV_IdBitmap *knows = kg_predicate_relation_set(kg, "knows");
    ASSERT(knows != NULL && gv_id_bitmap_cardinality(knows) == 2, "predicate 'knows' -> 2 relations");
    gv_id_bitmap_free(knows);
    GV_IdBitmap *likes = kg_predicate_relation_set(kg, "likes");
    ASSERT(likes && gv_id_bitmap_cardinality(likes) == 1, "predicate 'likes' -> 1 relation");
    gv_id_bitmap_free(likes);
    GV_IdBitmap *none = kg_predicate_relation_set(kg, "nope");
    ASSERT(none && gv_id_bitmap_cardinality(none) == 0, "unused predicate -> empty set");
    gv_id_bitmap_free(none);

    /* neighbour sets: A out -> {C,D}, B out -> {C}; join (AND) -> common neighbour {C} */
    GV_IdBitmap *na = kg_entity_neighbor_set(kg, A, 0);
    GV_IdBitmap *nb = kg_entity_neighbor_set(kg, B, 0);
    ASSERT(na && gv_id_bitmap_cardinality(na) == 2, "A out-neighbours == {C,D}");
    ASSERT(nb && gv_id_bitmap_cardinality(nb) == 1, "B out-neighbours == {C}");
    GV_IdBitmap *common = gv_id_bitmap_and(na, nb);
    ASSERT(common && gv_id_bitmap_cardinality(common) == 1 && gv_id_bitmap_contains(common, C),
           "common neighbour of A,B is C (Dgraph-style set join)");
    /* C incoming -> {A,B} */
    GV_IdBitmap *cin = kg_entity_neighbor_set(kg, C, 1);
    ASSERT(cin && gv_id_bitmap_cardinality(cin) == 2, "C in-neighbours == {A,B}");
    gv_id_bitmap_free(na); gv_id_bitmap_free(nb); gv_id_bitmap_free(common); gv_id_bitmap_free(cin);

    kg_destroy(kg);
    return 0;
}

int main(void) {
    test_metadata_bitmap();
    test_kg_typed_sets();
    if (failures == 0) printf("All posting-bitmap tests PASSED.\n");
    else printf("%d posting-bitmap test(s) FAILED.\n", failures);
    return failures ? 1 : 0;
}
