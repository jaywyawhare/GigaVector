from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ._core import Database, IndexType


class DatabasePool:
    def __init__(
        self,
        path: str | None,
        dimension: int,
        index_type: IndexType,
        size: int = 5,
        **open_kwargs,
    ) -> None:
        self._path = path
        self._dimension = dimension
        self._index_type = index_type
        self._size = size
        self._open_kwargs = open_kwargs
        self._queue: asyncio.Queue[Database] = asyncio.Queue(maxsize=size)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            for _ in range(self._size):
                db = await asyncio.to_thread(
                    Database.open,
                    self._path,
                    self._dimension,
                    self._index_type,
                    **self._open_kwargs,
                )
                await self._queue.put(db)
            self._initialized = True

    async def __aenter__(self) -> DatabasePool:
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @asynccontextmanager
    async def acquire(self, timeout: float = 30.0) -> AsyncIterator[Database]:
        if not self._initialized:
            await self._initialize()
        try:
            db = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError("Timed out waiting for a database connection from the pool")
        try:
            yield db
        finally:
            await self._queue.put(db)

    async def close(self) -> None:
        while not self._queue.empty():
            try:
                db = self._queue.get_nowait()
                db.close()
            except asyncio.QueueEmpty:
                break
        self._initialized = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        return self._queue.qsize()
