from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence

from ._core import Database, DistanceType, IndexType, ScrollEntry, SearchHit


class AsyncDatabase:
    def __init__(self, db: Database, executor: ThreadPoolExecutor) -> None:
        self._db = db
        self._executor = executor

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    @classmethod
    async def async_open(
        cls,
        path: str | None,
        dimension: int,
        index: IndexType = IndexType.KDTREE,
        max_workers: int = 4,
        **kwargs,
    ) -> AsyncDatabase:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        loop = asyncio.get_running_loop()
        db = await loop.run_in_executor(
            executor, lambda: Database.open(path, dimension, index, **kwargs)
        )
        return cls(db, executor)

    async def aclose(self) -> None:
        await self._run(self._db.close)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._executor.shutdown, True)

    async def __aenter__(self) -> AsyncDatabase:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def add_vector(
        self, vector: Sequence[float], metadata: dict[str, str] | None = None
    ) -> None:
        await self._run(self._db.add_vector, vector, metadata)

    async def add_vectors(self, vectors: Iterable[Sequence[float]]) -> None:
        await self._run(self._db.add_vectors, vectors)

    async def search(
        self,
        query: Sequence[float],
        k: int,
        distance: DistanceType = DistanceType.EUCLIDEAN,
        **kwargs,
    ) -> list[SearchHit]:
        return await self._run(self._db.search, query, k, distance, **kwargs)

    async def delete_vector(self, vector_index: int) -> None:
        await self._run(self._db.delete_vector, vector_index)

    async def update_vector(self, vector_index: int, new_data: Sequence[float]) -> None:
        await self._run(self._db.update_vector, vector_index, new_data)

    async def update_metadata(self, vector_index: int, metadata: dict[str, str]) -> None:
        await self._run(self._db.update_metadata, vector_index, metadata)

    async def search_with_filter_expr(
        self,
        query: Sequence[float],
        k: int,
        filter_expr: Any,
        **kwargs,
    ) -> list[SearchHit]:
        return await self._run(self._db.search_with_filter_expr, query, k, filter_expr, **kwargs)

    async def range_search(
        self,
        query: Sequence[float],
        radius: float,
        max_results: int = 1000,
        **kwargs,
    ) -> list[SearchHit]:
        return await self._run(self._db.range_search, query, radius, max_results, **kwargs)

    async def search_batch(
        self,
        queries: Iterable[Sequence[float]],
        k: int,
        **kwargs,
    ) -> list[list[SearchHit]]:
        return await self._run(self._db.search_batch, queries, k, **kwargs)

    async def count(self) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: self._db.count)

    async def get_stats(self) -> Any:
        return await self._run(self._db.get_stats)

    async def save(self, path: str | None = None) -> None:
        await self._run(self._db.save, path)

    async def compact(self) -> None:
        await self._run(self._db.compact)

    async def upsert(
        self,
        vector_index: int,
        vector: Sequence[float],
        metadata: dict[str, str] | None = None,
    ) -> None:
        await self._run(self._db.upsert, vector_index, vector, metadata)

    async def scroll(self, offset: int = 0, limit: int = 100) -> list[ScrollEntry]:
        return await self._run(self._db.scroll, offset, limit)

    async def export_json(self, filepath: str) -> int:
        return await self._run(self._db.export_json, filepath)

    async def import_json(self, filepath: str) -> int:
        return await self._run(self._db.import_json, filepath)

    async def health_check(self) -> int:
        return await self._run(self._db.health_check)
