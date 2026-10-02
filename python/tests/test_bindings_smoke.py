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


if __name__ == "__main__":
    unittest.main()
