import os
import tempfile
import unittest

from gigavector import Database, DistanceType, IndexType, IVFDiskConfig


class TestIVFDisk(unittest.TestCase):
    def test_train_insert_search_save(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "db.gvdb")
            cfg = IVFDiskConfig(nlist=8, nprobe=4, train_iters=8)
            db = Database.open(
                path,
                dimension=8,
                index=IndexType.IVFDISK,
                ivfdisk_config=cfg,
            )
            self.assertIsNotNone(db)

            dim = 8
            train = [
                [float(i + d) * 0.01 for d in range(dim)]
                for i in range(64)
            ]
            db.train_ivfdisk(train)

            vectors = train[:32]
            for vec in vectors:
                db.add_vector(vec)

            hits = db.search(list(vectors[0]), k=5, distance=DistanceType.EUCLIDEAN)
            self.assertGreaterEqual(len(hits), 1)

            db.save()
            db.close()

            db2 = Database.open(path, dimension=8, index=IndexType.IVFDISK)
            self.assertEqual(db2.count, 32)
            hits2 = db2.search(list(vectors[0]), k=5, distance=DistanceType.EUCLIDEAN)
            self.assertGreaterEqual(len(hits2), 1)
            db2.close()

    def test_delete_and_update(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "db.gvdb")
            cfg = IVFDiskConfig(nlist=4, nprobe=4, use_sq8=True)
            db = Database.open(
                path,
                dimension=4,
                index=IndexType.IVFDISK,
                ivfdisk_config=cfg,
            )
            dim = 4
            train = [[float(i + d) * 0.1 for d in range(dim)] for i in range(32)]
            db.train_ivfdisk(train)
            for vec in train[:16]:
                db.add_vector(vec)

            db.delete_vector(0)
            updated = [9.0, 9.1, 9.2, 9.3]
            db.update_vector(1, updated)

            hits = db.search(train[2], k=10, distance=DistanceType.EUCLIDEAN)
            ids = {h.id for h in hits}
            self.assertNotIn(0, ids)
            db.close()

    def test_open_mmap_readonly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "db.gvdb")
            dim = 4
            cfg = IVFDiskConfig(nlist=4, nprobe=2)
            db = Database.open(
                path,
                dimension=dim,
                index=IndexType.IVFDISK,
                ivfdisk_config=cfg,
            )
            train = [[float(i + d) * 0.1 for d in range(dim)] for i in range(24)]
            db.train_ivfdisk(train)
            for vec in train[:6]:
                db.add_vector(vec)
            db.save()
            db.close()

            ro = Database.open_mmap(path, dimension=dim, index=IndexType.IVFDISK)
            self.assertEqual(ro.count, 6)
            hits = ro.search(train[0], k=3, distance=DistanceType.EUCLIDEAN)
            self.assertGreaterEqual(len(hits), 1)
            ro.close()

    def test_wal_replay_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "db.gvdb")
            dim = 4
            cfg = IVFDiskConfig(nlist=4, nprobe=2)
            db = Database.open(
                path,
                dimension=dim,
                index=IndexType.IVFDISK,
                ivfdisk_config=cfg,
            )
            train = [[float(i + d) / 16.0 for d in range(dim)] for i in range(32)]
            db.train_ivfdisk(train)
            db.save()

            vectors = train[:8]
            for vec in vectors:
                db.add_vector(vec)
            db.close()

            db2 = Database.open(path, dimension=dim, index=IndexType.IVFDISK)
            self.assertEqual(db2.count, len(vectors))
            hits = db2.search(vectors[0], k=4, distance=DistanceType.EUCLIDEAN)
            self.assertGreaterEqual(len(hits), 1)
            db2.close()

    def test_border_ratio_config(self) -> None:
        """High border_ratio with two-cluster data should not break insert/search."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "db.gvdb")
            dim = 2
            cfg = IVFDiskConfig(nlist=2, nprobe=2, border_ratio=10.0)
            db = Database.open(
                path,
                dimension=dim,
                index=IndexType.IVFDISK,
                ivfdisk_config=cfg,
            )
            train: list[list[float]] = []
            for i in range(50):
                train.append([0.0, float(i) * 0.01])
            for i in range(50):
                train.append([10.0, float(i) * 0.01])
            db.train_ivfdisk(train)

            border = [5.0, 0.0]
            db.add_vector(border)
            for vec in train[:10]:
                db.add_vector(vec)

            hits = db.search(border, k=5, distance=DistanceType.EUCLIDEAN)
            self.assertGreaterEqual(len(hits), 1)
            db.close()

    def test_grpc_train_ivfdisk(self) -> None:
        import sys

        if sys.platform == "win32":
            self.skipTest("POSIX gRPC client not available on Windows")

        from gigavector import GrpcConfig, GrpcServer, RemoteShardClient

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "db.gvdb")
            dim = 4
            cfg = IVFDiskConfig(nlist=4, nprobe=2)
            db = Database.open(
                path,
                dimension=dim,
                index=IndexType.IVFDISK,
                ivfdisk_config=cfg,
            )
            train = [[float(i + d) * 0.1 for d in range(dim)] for i in range(16)]
            server = GrpcServer(
                db._db,
                GrpcConfig(port=50219, bind_address="127.0.0.1"),
            )
            try:
                server.start()
            except RuntimeError:
                self.skipTest("grpc port unavailable")
            try:
                client = RemoteShardClient("127.0.0.1", 50219, dim)
                client.train_ivfdisk(train)
                db.add_vector(train[0])
                hits = db.search(train[0], k=3, distance=DistanceType.EUCLIDEAN)
                self.assertGreaterEqual(len(hits), 1)
            finally:
                server.stop()
                server.close()
                db.close()


if __name__ == "__main__":
    unittest.main()
