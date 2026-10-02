"""Smoke tests for subsystems whose gv_* CFFI symbols were previously unexported.

These exercise the forwarding wrappers end-to-end so a future regression (a
subsystem's gv_ symbol going missing again) fails loudly instead of only
surfacing when a user calls the affected class.
"""

from __future__ import annotations

import unittest

import gigavector as gv


class TestPayloadIndex(unittest.TestCase):
    def test_add_insert_counts(self) -> None:
        pi = gv.PayloadIndex()
        pi.add_field("age", gv.FieldType.INT)
        self.assertEqual(pi.field_count, 1)
        pi.insert_int(1, "age", 30)
        pi.insert_int(2, "age", 40)
        self.assertEqual(pi.total_entries, 2)


class TestPointIDMap(unittest.TestCase):
    def test_set_get_contains(self) -> None:
        m = gv.PointIDMap()
        m.set("doc-a", 10)
        m.set("doc-b", 20)
        self.assertEqual(m.get("doc-a"), 10)
        self.assertIn("doc-b", m)
        self.assertEqual(len(m), 2)
        self.assertEqual(m.reverse_lookup(20), "doc-b")

    def test_generate_uuid(self) -> None:
        u = gv.PointIDMap.generate_uuid()
        self.assertIsInstance(u, str)
        self.assertGreater(len(u), 0)


class TestSchema(unittest.TestCase):
    def test_add_field(self) -> None:
        s = gv.Schema()
        s.add_field("name", gv.SchemaFieldType.STRING)
        self.assertEqual(s.field_count, 1)
        self.assertTrue(s.has_field("name"))


class TestHybridSearch(unittest.TestCase):
    def test_dense_sparse_fusion(self) -> None:
        db = gv.Database.open(None, dimension=3, index=gv.IndexType.FLAT)
        docs = {0: "the quick brown fox", 1: "lazy dog sleeps", 2: "quick fox runs fast"}
        vecs = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0], 2: [0.9, 0.1, 0.0]}
        for i in sorted(docs):
            db.add_vector(vecs[i])
        bm = gv.BM25Index()
        for i in sorted(docs):
            bm.add_document(i, docs[i])
        hs = gv.HybridSearcher(db, bm)
        res = hs.search(query_vector=[1.0, 0.0, 0.0], query_text="quick fox", k=3)
        self.assertEqual(len(res), 3)
        # Doc 0 matches both the query vector and the text best -> ranks first.
        self.assertEqual(res[0].vector_index, 0)
        self.assertGreater(res[0].combined_score, res[1].combined_score)


if __name__ == "__main__":
    unittest.main()
