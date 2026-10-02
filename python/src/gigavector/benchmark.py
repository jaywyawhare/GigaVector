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


@dataclass
class RecallResult:
    """Recall@k measured against ground-truth, alongside query throughput/latency
    so the accuracy-vs-speed trade-off (the ann-benchmarks metric) is captured in
    one run."""

    count: int
    k: int
    recall_at_k: float
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

    def run_recall(
        self,
        queries: Sequence[Sequence[float]],
        ground_truth: Sequence[Sequence[int]],
        k: int = 10,
        distance: DistanceType = DistanceType.EUCLIDEAN,
    ) -> RecallResult:
        """Measure recall@k against caller-supplied ground truth (e.g. computed
        with an exact FLAT index). recall@k = mean over queries of
        |returned_ids ∩ true_ids| / k. Also records query latency/QPS so the
        accuracy-vs-throughput trade-off is visible in a single run.
        """
        if len(queries) != len(ground_truth):
            raise ValueError("queries and ground_truth must be the same length")

        latencies: list[float] = []
        hits = 0
        denom = 0
        t_start = time.perf_counter()
        for q, truth in zip(queries, ground_truth):
            t0 = time.perf_counter()
            results = self._db.search(q, k, distance)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            returned = {getattr(r, "index", getattr(r, "id", r)) for r in results}
            truth_k = set(list(truth)[:k])
            hits += len(returned & truth_k)
            denom += len(truth_k)
        total = time.perf_counter() - t_start

        count = len(queries)
        latencies.sort()
        return RecallResult(
            count=count,
            k=k,
            recall_at_k=hits / denom if denom > 0 else 0.0,
            qps=count / total if total > 0 else 0.0,
            p50_ms=_percentile(latencies, 50),
            p95_ms=_percentile(latencies, 95),
            p99_ms=_percentile(latencies, 99),
            mean_ms=sum(latencies) / count if count > 0 else 0.0,
        )

    def report_recall(self, result: RecallResult) -> str:
        return (
            f"Operation : recall@{result.k}\n"
            f"Count     : {result.count}\n"
            f"Recall@{result.k} : {result.recall_at_k:.4f}\n"
            f"QPS       : {result.qps:.1f}\n"
            f"Mean      : {result.mean_ms:.3f}ms\n"
            f"p50       : {result.p50_ms:.3f}ms\n"
            f"p95       : {result.p95_ms:.3f}ms\n"
            f"p99       : {result.p99_ms:.3f}ms"
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


def _cli(argv: Sequence[str] | None = None) -> int:
    """Reproducible insert/search/recall benchmark harness.

    Generates seeded random data, builds the chosen index plus an exact FLAT
    index for ground truth, and reports insert/search throughput and recall@k —
    the standard vector-database (ann-benchmarks-style) metrics.
    """
    import argparse
    import random

    from ._core import Database, IndexType

    parser = argparse.ArgumentParser(prog="python -m gigavector.benchmark")
    parser.add_argument("--n", type=int, default=10000, help="number of vectors")
    parser.add_argument("--dim", type=int, default=128, help="vector dimension")
    parser.add_argument("--queries", type=int, default=1000, help="number of query vectors")
    parser.add_argument("--k", type=int, default=10, help="top-k")
    parser.add_argument("--index", default="HNSW", help="index type (HNSW, IVFFLAT, FLAT, ...)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    index = getattr(IndexType, args.index.upper())
    rng = random.Random(args.seed)
    data = [[rng.random() for _ in range(args.dim)] for _ in range(args.n)]
    queries = [[rng.random() for _ in range(args.dim)] for _ in range(args.queries)]

    index_db = Database.open(None, dimension=args.dim, index=index)
    truth_db = Database.open(None, dimension=args.dim, index=IndexType.FLAT)
    for v in data:
        index_db.add_vector(v)
        truth_db.add_vector(v)
    ground_truth = [[h.id for h in truth_db.search(q, args.k)] for q in queries]

    bench = Benchmark(index_db)
    search = bench.run_search(queries, k=args.k)
    recall = bench.run_recall(queries, ground_truth, k=args.k)

    print(f"GigaVector benchmark — index={args.index.upper()} n={args.n} dim={args.dim} k={args.k}")
    print(bench.report(search))
    print(bench.report_recall(recall))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))
