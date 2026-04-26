from __future__ import annotations

import asyncio
import unittest

from gigavector import Database, DistanceType, IndexType, Benchmark, BenchmarkResult
from gigavector.async_api import AsyncDatabase
from gigavector.pool import DatabasePool


def run(coro):
    return asyncio.run(coro)


class TestAsyncDatabase(unittest.TestCase):
    def test_open_close(self):
        async def _():
            db = await AsyncDatabase.async_open(None, dimension=3, index=IndexType.FLAT)
            await db.aclose()
        run(_())

    def test_context_manager(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=3, index=IndexType.FLAT) as db:
                self.assertIsInstance(db, AsyncDatabase)
        run(_())

    def test_add_and_search(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=3, index=IndexType.FLAT) as db:
                await db.add_vector([1.0, 2.0, 3.0])
                hits = await db.search([1.0, 2.0, 3.0], k=1, distance=DistanceType.EUCLIDEAN)
                self.assertEqual(len(hits), 1)
                self.assertAlmostEqual(hits[0].distance, 0.0, places=4)
        run(_())

    def test_add_vectors(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.add_vectors([[1.0, 0.0], [0.0, 1.0]])
                self.assertEqual(await db.count(), 2)
        run(_())

    def test_count(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                self.assertEqual(await db.count(), 0)
                await db.add_vector([1.0, 2.0])
                self.assertEqual(await db.count(), 1)
        run(_())

    def test_delete_vector(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.add_vector([1.0, 2.0])
                await db.delete_vector(0)
        run(_())

    def test_update_vector(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.add_vector([1.0, 2.0])
                await db.update_vector(0, [3.0, 4.0])
        run(_())

    def test_update_metadata(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.add_vector([1.0, 2.0], metadata={"k": "v"})
                await db.update_metadata(0, {"k": "updated"})
        run(_())

    def test_get_stats(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                stats = await db.get_stats()
                self.assertIsNotNone(stats)
        run(_())

    def test_health_check(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                rc = await db.health_check()
                self.assertEqual(rc, 0)
        run(_())

    def test_scroll(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.add_vector([1.0, 2.0])
                entries = await db.scroll(offset=0, limit=10)
                self.assertEqual(len(entries), 1)
        run(_())

    def test_range_search(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.add_vector([0.0, 0.0])
                await db.add_vector([10.0, 10.0])
                hits = await db.range_search([0.0, 0.0], radius=1.0)
                self.assertEqual(len(hits), 1)
        run(_())

    def test_search_batch(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.add_vector([1.0, 0.0])
                await db.add_vector([0.0, 1.0])
                results = await db.search_batch([[1.0, 0.0], [0.0, 1.0]], k=1)
                self.assertEqual(len(results), 2)
        run(_())

    def test_compact(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.compact()
        run(_())

    def test_upsert(self):
        async def _():
            async with await AsyncDatabase.async_open(None, dimension=2, index=IndexType.FLAT) as db:
                await db.add_vector([1.0, 2.0])
                await db.upsert(0, [3.0, 4.0])
        run(_())


class TestDatabasePool(unittest.TestCase):
    def test_context_manager(self):
        async def _():
            async with DatabasePool(None, dimension=2, index_type=IndexType.FLAT, size=2) as pool:
                self.assertEqual(pool.size, 2)
                self.assertEqual(pool.available, 2)
        run(_())

    def test_acquire_and_release(self):
        async def _():
            async with DatabasePool(None, dimension=2, index_type=IndexType.FLAT, size=2) as pool:
                async with pool.acquire() as db:
                    self.assertIsInstance(db, Database)
                    self.assertEqual(pool.available, 1)
                self.assertEqual(pool.available, 2)
        run(_())

    def test_multiple_acquires(self):
        async def _():
            async with DatabasePool(None, dimension=2, index_type=IndexType.FLAT, size=3) as pool:
                async with pool.acquire() as db1:
                    async with pool.acquire() as db2:
                        self.assertIsNot(db1, db2)
                        self.assertEqual(pool.available, 1)
        run(_())

    def test_use_db_from_pool(self):
        async def _():
            async with DatabasePool(None, dimension=2, index_type=IndexType.FLAT, size=1) as pool:
                async with pool.acquire() as db:
                    db.add_vector([1.0, 2.0])
                    hits = db.search([1.0, 2.0], k=1)
                    self.assertEqual(len(hits), 1)
        run(_())

    def test_timeout(self):
        async def _():
            async with DatabasePool(None, dimension=2, index_type=IndexType.FLAT, size=1) as pool:
                async with pool.acquire():
                    with self.assertRaises(TimeoutError):
                        async with pool.acquire(timeout=0.05):
                            pass
        run(_())


class TestBenchmark(unittest.TestCase):
    def _make_db(self):
        return Database.open(None, dimension=4, index=IndexType.FLAT)

    def test_run_insert(self):
        with self._make_db() as db:
            bench = Benchmark(db)
            vectors = [[float(i)] * 4 for i in range(30)]
            result = bench.run_insert(vectors, warmup=5)
            self.assertIsInstance(result, BenchmarkResult)
            self.assertEqual(result.operation, "insert")
            self.assertEqual(result.count, 25)
            self.assertGreater(result.qps, 0)
            self.assertGreater(result.p50_ms, 0)
            self.assertGreaterEqual(result.p99_ms, result.p95_ms)
            self.assertGreaterEqual(result.p95_ms, result.p50_ms)

    def test_run_search(self):
        with self._make_db() as db:
            for i in range(20):
                db.add_vector([float(i)] * 4)
            bench = Benchmark(db)
            queries = [[float(i)] * 4 for i in range(20)]
            result = bench.run_search(queries, k=3, warmup=5)
            self.assertEqual(result.operation, "search")
            self.assertEqual(result.count, 15)
            self.assertGreater(result.qps, 0)

    def test_run_batch_search(self):
        with self._make_db() as db:
            for i in range(20):
                db.add_vector([float(i)] * 4)
            bench = Benchmark(db)
            queries = [[float(i)] * 4 for i in range(32)]
            result = bench.run_batch_search(queries, k=3, batch_size=8)
            self.assertEqual(result.operation, "batch_search")
            self.assertEqual(result.count, 32)

    def test_report(self):
        with self._make_db() as db:
            for i in range(20):
                db.add_vector([float(i)] * 4)
            bench = Benchmark(db)
            queries = [[float(i)] * 4 for i in range(20)]
            result = bench.run_search(queries, k=1, warmup=5)
            report = bench.report(result)
            self.assertIn("search", report)
            self.assertIn("QPS", report)
            self.assertIn("p99", report)

    def test_zero_warmup(self):
        with self._make_db() as db:
            bench = Benchmark(db)
            vectors = [[1.0, 2.0, 3.0, 4.0]] * 10
            result = bench.run_insert(vectors, warmup=0)
            self.assertEqual(result.count, 10)


if __name__ == "__main__":
    unittest.main()
