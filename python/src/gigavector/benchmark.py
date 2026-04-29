from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ._core import Database, DistanceType


@dataclass
class BenchmarkResult:
    operation: str
    count: int
    total_time_s: float
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (len(sorted_data) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[lo]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


class Benchmark:
    def __init__(self, db: Database) -> None:
        self._db = db

    def run_insert(
        self,
        vectors: Iterable[Sequence[float]],
        warmup: int = 10,
    ) -> BenchmarkResult:
        vectors = list(vectors)
        for vec in vectors[:warmup]:
            self._db.add_vector(vec)

        latencies: list[float] = []
        t_start = time.perf_counter()
        for vec in vectors[warmup:]:
            t0 = time.perf_counter()
            self._db.add_vector(vec)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        total = time.perf_counter() - t_start

        count = len(latencies)
        latencies.sort()
        return BenchmarkResult(
            operation="insert",
            count=count,
            total_time_s=total,
            qps=count / total if total > 0 else 0.0,
            p50_ms=_percentile(latencies, 50),
            p95_ms=_percentile(latencies, 95),
            p99_ms=_percentile(latencies, 99),
            mean_ms=sum(latencies) / count if count > 0 else 0.0,
        )

    def run_search(
        self,
        queries: Iterable[Sequence[float]],
        k: int = 10,
        warmup: int = 10,
    ) -> BenchmarkResult:
        queries = list(queries)
        for q in queries[:warmup]:
            self._db.search(q, k, DistanceType.EUCLIDEAN)

        latencies: list[float] = []
        t_start = time.perf_counter()
        for q in queries[warmup:]:
            t0 = time.perf_counter()
            self._db.search(q, k, DistanceType.EUCLIDEAN)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        total = time.perf_counter() - t_start

        count = len(latencies)
        latencies.sort()
        return BenchmarkResult(
            operation="search",
            count=count,
            total_time_s=total,
            qps=count / total if total > 0 else 0.0,
            p50_ms=_percentile(latencies, 50),
            p95_ms=_percentile(latencies, 95),
            p99_ms=_percentile(latencies, 99),
            mean_ms=sum(latencies) / count if count > 0 else 0.0,
        )

    def run_batch_search(
        self,
        queries: Iterable[Sequence[float]],
        k: int = 10,
        batch_size: int = 32,
    ) -> BenchmarkResult:
        queries = list(queries)
        batches = [queries[i : i + batch_size] for i in range(0, len(queries), batch_size)]

        latencies: list[float] = []
        t_start = time.perf_counter()
        for batch in batches:
            t0 = time.perf_counter()
            self._db.search_batch(batch, k)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        total = time.perf_counter() - t_start

        count = len(queries)
        latencies.sort()
        return BenchmarkResult(
            operation="batch_search",
            count=count,
            total_time_s=total,
            qps=count / total if total > 0 else 0.0,
            p50_ms=_percentile(latencies, 50),
            p95_ms=_percentile(latencies, 95),
            p99_ms=_percentile(latencies, 99),
            mean_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        )

    def report(self, result: BenchmarkResult) -> str:
        return (
            f"Operation : {result.operation}\n"
            f"Count     : {result.count}\n"
            f"Total time: {result.total_time_s:.3f}s\n"
            f"QPS       : {result.qps:.1f}\n"
            f"Mean      : {result.mean_ms:.3f}ms\n"
            f"p50       : {result.p50_ms:.3f}ms\n"
            f"p95       : {result.p95_ms:.3f}ms\n"
            f"p99       : {result.p99_ms:.3f}ms"
        )
