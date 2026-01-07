import unittest

from gigavector import Database, IndexType, DistanceType


class TestRichMetadata(unittest.TestCase):
    def test_add_and_search_with_multiple_metadata_entries(self):
        # In-memory DB; WAL disabled so this only validates in-memory flow.
        with Database.open(None, dimension=2, index=IndexType.KDTREE) as db:
            vec = [1.0, 2.0]
            meta = {"tag": "a", "owner": "b"}
            db.add_vector(vec, metadata=meta)

            hits = db.search(vec, k=1, distance=DistanceType.EUCLIDEAN)
            self.assertEqual(len(hits), 1)
            hit = hits[0]
            self.assertAlmostEqual(hit.distance, 0.0)


if __name__ == "__main__":
    unittest.main()

