"""End-to-end integration test for the high-level Python bindings.

Walks one realistic workflow through the public Python API (the CFFI layer):
vector DB add/search/filter/range/persist, dense+sparse hybrid fusion, and the
knowledge graph (entities, relations, triple queries, vector similarity). This
is the Python-surface counterpart to tests/test_e2e.c and guards that the
bindings work together, not just in isolation.
"""

from __future__ import annotations

import sys
import tempfile
import unittest

import gigavector as gv

# Some bindings (hybrid, KG similarity) exchange heap-allocated result arrays
# across the DLL boundary, which hits a known Windows cross-CRT heap issue (see
# test_bindings_smoke). The vector-DB phase uses the same GV_SearchResult path
# as the Windows-green test_api/test_ivfdisk suites, so it stays enabled
# everywhere; the cross-boundary phases are skipped on Windows (covered there by
# the cross-platform C integration test tests/test_e2e.c).
_skip_win = unittest.skipIf(sys.platform == "win32",
                            "binding exchanges heap arrays across the DLL boundary (Windows cross-CRT)")


class TestVectorDBLifecycle(unittest.TestCase):
    def test_add_search_filter_range_persist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/e2e.gvdb"
            db = gv.Database.open(path, dimension=4, index=gv.IndexType.FLAT)
            db.add_vector([1.0, 0.0, 0.0, 0.0], metadata={"color": "red"})
            db.add_vector([0.0, 1.0, 0.0, 0.0], metadata={"color": "blue"})
            db.add_vector([0.0, 0.0, 1.0, 0.0], metadata={"color": "green"})
            db.add_vector([0.9, 0.1, 0.0, 0.0], metadata={"color": "red"})
            self.assertEqual(db.count, 4)

            hits = db.search([1.0, 0.0, 0.0, 0.0], k=3, distance=gv.DistanceType.EUCLIDEAN)
            self.assertEqual(len(hits), 3)
            self.assertEqual(hits[0].id, 0)  # exact match is nearest
            self.assertLessEqual(hits[0].distance, hits[1].distance)
            self.assertEqual(hits[0].vector.metadata.get("color"), "red")

            # Payload-filtered search -> only the two reds.
            red_hits = db.search([1.0, 0.0, 0.0, 0.0], k=4,
                                 distance=gv.DistanceType.EUCLIDEAN,
                                 filter_metadata=("color", "red"))
            self.assertEqual(len(red_hits), 2)
            self.assertTrue(all(h.vector.metadata.get("color") == "red" for h in red_hits))

            # Range search within 0.5 of red -> red + red2.
            in_radius = db.range_search([1.0, 0.0, 0.0, 0.0], radius=0.5)
            self.assertEqual(len(in_radius), 2)

            db.save(path)
            db.close()

            # Reopen and confirm persistence.
            db2 = gv.Database.open(path, dimension=4, index=gv.IndexType.FLAT)
            self.assertEqual(db2.count, 4)
            again = db2.search([1.0, 0.0, 0.0, 0.0], k=1)
            self.assertEqual(again[0].vector.metadata.get("color"), "red")
            db2.close()


@_skip_win
class TestHybridFusion(unittest.TestCase):
    def test_dense_plus_sparse(self) -> None:
        db = gv.Database.open(None, dimension=3, index=gv.IndexType.FLAT)
        bm = gv.BM25Index()
        vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]]
        texts = ["quick brown fox jumps", "lazy dog sleeps", "quick fox runs fast"]
        for i, (v, t) in enumerate(zip(vecs, texts)):
            db.add_vector(v)
            bm.add_document(i, t)

        hs = gv.HybridSearcher(db, bm)
        res = hs.search(query_vector=[1.0, 0.0, 0.0], query_text="quick fox", k=3)
        self.assertGreaterEqual(len(res), 1)
        # Doc 0 matches both the query vector and the text best.
        self.assertEqual(res[0].vector_index, 0)
        self.assertGreaterEqual(res[0].combined_score, res[1].combined_score)


@_skip_win
class TestKnowledgeGraph(unittest.TestCase):
    def test_entities_relations_triples_similarity(self) -> None:
        kg = gv.KnowledgeGraph(gv.KGConfig(embedding_dimension=3))
        alice = kg.add_entity("Alice", "Person", embedding=[1.0, 0.0, 0.0])
        bob = kg.add_entity("Bob", "Person", embedding=[0.0, 1.0, 0.0])
        acme = kg.add_entity("Acme", "Company", embedding=[0.0, 0.0, 1.0])
        kg.add_relation(alice, "KNOWS", bob)
        kg.add_relation(alice, "WORKS_AT", acme)

        # Triple query: Alice's outgoing edges.
        triples = kg.query_triples(subject=alice)
        preds = {t.predicate for t in triples}
        self.assertIn("KNOWS", preds)
        self.assertIn("WORKS_AT", preds)

        # Vector similarity over entity embeddings -> Alice is nearest to her own vector.
        sim = kg.search_similar([1.0, 0.0, 0.0], k=1)
        self.assertGreaterEqual(len(sim), 1)
        self.assertEqual(sim[0].entity_id, alice)

        # Typed property round-trip.
        kg.set_entity_prop(alice, "team", "platform")
        self.assertEqual(kg.get_entity_prop(alice, "team"), "platform")


if __name__ == "__main__":
    unittest.main()
