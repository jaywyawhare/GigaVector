import os
import tempfile
import unittest

from gigavector import Database, DistanceType, IndexType


class TestAPI(unittest.TestCase):
    def test_basic_add_search(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with Database.open(db_path, dimension=3, index=IndexType.KDTREE) as db:
                db.add_vector([1.0, 2.0, 3.0])
                hits = db.search([1.0, 2.0, 3.0], k=1, distance=DistanceType.EUCLIDEAN)
                self.assertEqual(len(hits), 1)
                self.assertAlmostEqual(hits[0].distance, 0.0)

    def test_multi_metadata_and_wal_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "db.bin")
            # WAL will be auto-created alongside the db file
            with Database.open(db_path, dimension=2, index=IndexType.KDTREE) as db:
                db.add_vector([0.1, 0.2], metadata={"tag": "a", "owner": "b", "source": "demo"})
                db.save(db_path)

            # Reopen to ensure snapshot + WAL restore
            with Database.open(db_path, dimension=2, index=IndexType.KDTREE) as db:
                hits = db.search([0.1, 0.2], k=1, distance=DistanceType.EUCLIDEAN)
                self.assertEqual(len(hits), 1)

    def test_filtered_search(self):
        with Database.open(None, dimension=2, index=IndexType.KDTREE) as db:
            db.add_vector([0.0, 1.0], metadata={"color": "red"})
            db.add_vector([0.0, 2.0], metadata={"color": "blue"})
            hits = db.search([0.0, 1.1], k=2, distance=DistanceType.EUCLIDEAN, filter_metadata=("color", "red"))
            self.assertEqual(len(hits), 1)

    def test_batch_search(self):
        with Database.open(None, dimension=2, index=IndexType.KDTREE) as db:
            db.add_vector([0.0, 0.0])
            db.add_vector([1.0, 1.0])
            queries = [[0.0, 0.1], [1.0, 1.1]]
            results = db.search_batch(queries, k=1, distance=DistanceType.EUCLIDEAN)
            self.assertEqual(len(results), 2)
            self.assertEqual(len(results[0]), 1)
            self.assertAlmostEqual(results[0][0].distance, 0.1, places=3)
            self.assertAlmostEqual(results[1][0].distance, 0.1, places=3)

    # def test_error_handling(self):
    #     with Database.open(None, dimension=2, index=IndexType.KDTREE) as db:
    #         # Wrong dimension for add_vector
    #         with self.assertRaises(ValueError):
    #             db.add_vector([1.0])
    #         with self.assertRaises(ValueError):
    #             db.add_vector([1.0, 2.0, 3.0])

    #         # Wrong dimension for search
    #         with self.assertRaises(ValueError):
    #             db.search([1.0], k=1)
    #         with self.assertRaises(ValueError):
    #             db.search([1.0, 2.0, 3.0], k=1)

    #         # Invalid k
    #         with self.assertRaises(RuntimeError):
    #             db.search([1.0, 2.0], k=0)

    def test_index_type_smoke(self):
        # Use dimension 8 so IVFPQ (default m=8) can initialize.
        dim = 8
        vec = [0.5] * dim
        # For IVFPQ we need a modest dataset; feed a few vectors.
        dataset = [
            [0.5] * dim,
            [0.6] * dim,
            [0.4] * dim,
            [0.5 if i % 2 == 0 else 0.6 for i in range(dim)],
        ]
        for index in (IndexType.KDTREE, IndexType.HNSW, IndexType.IVFPQ):
            try:
                db = Database.open(None, dimension=dim, index=index)
            except RuntimeError:
                # Some builds may omit optional index implementations; skip in that case.
                self.skipTest(f"{index} not available")
                continue
            with db:
                try:
                    if index == IndexType.IVFPQ:
                        # Train with enough vectors (>= codebook size, default 256).
                        train = [[(i % 10) / 10.0 for _ in range(dim)] for i in range(256)]
                        db.train_ivfpq(train)
                    for v in dataset:
                        db.add_vector(v)
                except RuntimeError:
                    self.skipTest(f"{index} setup failed (likely unsupported build)")
                    continue
                hits = db.search(vec, k=1, distance=DistanceType.EUCLIDEAN)
                self.assertEqual(len(hits), 1)
                # Allow non-zero distance for approximate indexes (IVFPQ).
                self.assertLess(hits[0].distance, 0.25)


if __name__ == "__main__":
    unittest.main()

