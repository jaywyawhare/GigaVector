"""Collection alias manager - exercises the gv_alias_* CFFI bindings."""

from __future__ import annotations

import unittest

from gigavector import AliasManager


class TestAliasManager(unittest.TestCase):
    def test_create_and_resolve(self) -> None:
        m = AliasManager()
        m.create("prod", "collection_v1")
        self.assertEqual(m.resolve("prod"), "collection_v1")
        self.assertTrue(m.exists("prod"))
        self.assertFalse(m.exists("missing"))

    def test_update(self) -> None:
        m = AliasManager()
        m.create("prod", "collection_v1")
        m.update("prod", "collection_v2")
        self.assertEqual(m.resolve("prod"), "collection_v2")

    def test_swap(self) -> None:
        m = AliasManager()
        m.create("a", "col_a")
        m.create("b", "col_b")
        m.swap("a", "b")
        self.assertEqual(m.resolve("a"), "col_b")
        self.assertEqual(m.resolve("b"), "col_a")

    def test_delete(self) -> None:
        m = AliasManager()
        m.create("tmp", "col")
        self.assertTrue(m.exists("tmp"))
        m.delete("tmp")
        self.assertFalse(m.exists("tmp"))


if __name__ == "__main__":
    unittest.main()
