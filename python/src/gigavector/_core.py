from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

from ._ffi import ffi, lib


class IndexType(IntEnum):
    KDTREE = 0
    HNSW = 1
    IVFPQ = 2
    SPARSE = 3


class DistanceType(IntEnum):
    EUCLIDEAN = 0
    COSINE = 1
    DOT_PRODUCT = 2
    MANHATTAN = 3


@dataclass(frozen=True)
class Vector:
    data: list[float]
    metadata: dict[str, str]


@dataclass(frozen=True)
class SearchHit:
    distance: float
    vector: Vector


@dataclass(frozen=True)
class DBStats:
    total_inserts: int
    total_queries: int
    total_range_queries: int
    total_wal_records: int


@dataclass
class HNSWConfig:
    """Configuration for HNSW index."""
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    max_level: int = 16
    use_binary_quant: bool = False
    quant_rerank: int = 0
    use_acorn: bool = False
    acorn_hops: int = 1


@dataclass
class ScalarQuantConfig:
    """Configuration for scalar quantization."""
    bits: int = 8
    per_dimension: bool = False


@dataclass
class IVFPQConfig:
    """Configuration for IVFPQ index."""
    nlist: int = 64
    m: int = 8
    nbits: int = 8
    nprobe: int = 4
    train_iters: int = 15
    default_rerank: int = 32
    use_cosine: bool = False
    use_scalar_quant: bool = False
    scalar_quant_config: ScalarQuantConfig = None
    oversampling_factor: float = 1.0
    
    def __post_init__(self):
        if self.scalar_quant_config is None:
            self.scalar_quant_config = ScalarQuantConfig()


def _choose_index_type(dimension: int, expected_count: int | None) -> IndexType:
    """
    Heuristic index selector based on dimension and estimated collection size.

    - For small collections (<= 20k) and low/moderate dimensions (<= 64), use KDTREE.
    - For very large collections (>= 500k) and high dimensions (>= 128), prefer IVFPQ.
    - Otherwise, default to HNSW.
    """
    if expected_count is None or expected_count < 0:
        # Fall back to HNSW when we don't know collection size.
        return IndexType.HNSW
    val = lib.gv_index_suggest(dimension, int(expected_count))
    return IndexType(int(val))


def _metadata_to_dict(meta_ptr) -> dict[str, str]:
    if meta_ptr == ffi.NULL:
        return {}
    out: dict[str, str] = {}
    cur = meta_ptr
    while cur != ffi.NULL:
        try:
            key = ffi.string(cur.key).decode("utf-8") if cur.key != ffi.NULL else ""
            value = ffi.string(cur.value).decode("utf-8") if cur.value != ffi.NULL else ""
            if key:
                out[key] = value
        except (UnicodeDecodeError, AttributeError):
            pass
        cur = cur.next
    return out


def _copy_vector(vec_ptr) -> Vector:
    try:
        if vec_ptr == ffi.NULL:
            return Vector(data=[], metadata={})
        dim = int(vec_ptr.dimension)
        if dim <= 0 or dim > 100000:
            raise ValueError(f"Invalid vector dimension: {dim}")
        if dim == 0:
            return Vector(data=[], metadata={})
        if vec_ptr.data == ffi.NULL:
            return Vector(data=[], metadata={})
        data = [vec_ptr.data[i] for i in range(dim)]
        metadata = _metadata_to_dict(vec_ptr.metadata)
        return Vector(data=data, metadata=metadata)
    except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
        return Vector(data=[], metadata={})

def _copy_sparse_vector(sv_ptr, dim: int) -> Vector:
    if sv_ptr == ffi.NULL:
        return Vector(data=[], metadata={})
    nnz = int(sv_ptr.nnz)
    data = [0.0] * dim
    for i in range(nnz):
        ent = sv_ptr.entries[i]
        idx = int(ent.index)
        if 0 <= idx < dim:
            data[idx] = float(ent.value)
    metadata = _metadata_to_dict(sv_ptr.metadata)
    return Vector(data=data, metadata=metadata)


class Database:
    def __init__(self, handle, dimension: int):
        self._db = handle
        self.dimension = int(dimension)
        self._closed = False

    @classmethod
    def open(cls, path: str | None, dimension: int, index: IndexType = IndexType.KDTREE, 
             hnsw_config: HNSWConfig | None = None, ivfpq_config: IVFPQConfig | None = None):
        """
        Open a database instance.
        
        Args:
            path: File path for persistent storage. Use None for in-memory database.
            dimension: Vector dimension (must be consistent for all vectors).
            index: Index type to use. Defaults to KDTREE.
            hnsw_config: Optional HNSW configuration. Only used when index is HNSW.
            ivfpq_config: Optional IVFPQ configuration. Only used when index is IVFPQ.
        
        Returns:
            Database instance
        """
        c_path = path.encode("utf-8") if path is not None else ffi.NULL
        
        if hnsw_config is not None and index == IndexType.HNSW:
            config = ffi.new("GV_HNSWConfig *", {
                "M": hnsw_config.M,
                "efConstruction": hnsw_config.ef_construction,
                "efSearch": hnsw_config.ef_search,
                "maxLevel": hnsw_config.max_level,
                "use_binary_quant": 1 if hnsw_config.use_binary_quant else 0,
                "quant_rerank": hnsw_config.quant_rerank,
                "use_acorn": 1 if hnsw_config.use_acorn else 0,
                "acorn_hops": hnsw_config.acorn_hops,
            })
            db = lib.gv_db_open_with_hnsw_config(c_path, dimension, int(index), config)
        elif ivfpq_config is not None and index == IndexType.IVFPQ:
            sq_config = ffi.new("GV_ScalarQuantConfig *", {
                "bits": ivfpq_config.scalar_quant_config.bits,
                "per_dimension": 1 if ivfpq_config.scalar_quant_config.per_dimension else 0
            })
            config = ffi.new("GV_IVFPQConfig *", {
                "nlist": ivfpq_config.nlist,
                "m": ivfpq_config.m,
                "nbits": ivfpq_config.nbits,
                "nprobe": ivfpq_config.nprobe,
                "train_iters": ivfpq_config.train_iters,
                "default_rerank": ivfpq_config.default_rerank,
                "use_cosine": 1 if ivfpq_config.use_cosine else 0,
                "use_scalar_quant": 1 if ivfpq_config.use_scalar_quant else 0,
                "scalar_quant_config": sq_config[0],
                "oversampling_factor": ivfpq_config.oversampling_factor
            })
            db = lib.gv_db_open_with_ivfpq_config(c_path, dimension, int(index), config)
        else:
            db = lib.gv_db_open(c_path, dimension, int(index))
        
        if db == ffi.NULL:
            raise RuntimeError("gv_db_open failed")
        return cls(db, dimension)

    @classmethod
    def open_auto(cls, path: str | None, dimension: int,
                  expected_count: int | None = None,
                  hnsw_config: HNSWConfig | None = None,
                  ivfpq_config: IVFPQConfig | None = None):
        """
        Open a database and automatically choose a reasonable index type.

        Args:
            path: Optional path for persistence (None for in-memory).
            dimension: Vector dimensionality.
            expected_count: Optional estimate of the number of vectors.
            hnsw_config: Optional HNSW configuration (used if HNSW is selected).
            ivfpq_config: Optional IVFPQ configuration (used if IVFPQ is selected).
        """
        index = _choose_index_type(dimension, expected_count)
        return cls.open(path, dimension, index=index,
                        hnsw_config=hnsw_config, ivfpq_config=ivfpq_config)

    @classmethod
    def open_mmap(cls, path: str, dimension: int, index: IndexType = IndexType.KDTREE):
        """
        Open a read-only database by memory-mapping an existing snapshot file.

        This is a thin wrapper around gv_db_open_mmap(). The returned Database
        instance shares the mapped file; modifications are not persisted.
        """
        if not path:
            raise ValueError("path must be non-empty")
        c_path = path.encode("utf-8")
        db = lib.gv_db_open_mmap(c_path, dimension, int(index))
        if db == ffi.NULL:
            raise RuntimeError("gv_db_open_mmap failed")
        return cls(db, dimension)

    def close(self):
        if self._closed:
            return
        lib.gv_db_close(self._db)
        self._closed = True

    def get_stats(self) -> DBStats:
        """
        Return aggregate runtime statistics for this database.
        """
        stats_c = ffi.new("GV_DBStats *")
        lib.gv_db_get_stats(self._db, stats_c)
        return DBStats(
            total_inserts=int(stats_c.total_inserts),
            total_queries=int(stats_c.total_queries),
            total_range_queries=int(stats_c.total_range_queries),
            total_wal_records=int(stats_c.total_wal_records),
        )

    def save(self, path: str | None = None):
        """Persist the database to a binary snapshot file."""
        c_path = path.encode("utf-8") if path is not None else ffi.NULL
        rc = lib.gv_db_save(self._db, c_path)
        if rc != 0:
            raise RuntimeError("gv_db_save failed")
        # Truncate WAL to avoid replaying already-saved inserts
        if self._db.wal != ffi.NULL:
            lib.gv_wal_truncate(self._db.wal)
        # Truncate WAL to avoid replaying already-saved inserts
        if self._db.wal != ffi.NULL:
            lib.gv_wal_truncate(self._db.wal)

    def set_exact_search_threshold(self, threshold: int) -> None:
        """
        Configure the exact-search fallback threshold.

        When the number of stored vectors is <= threshold, the database may
        use a brute-force exact search path instead of the index (for
        supported index types). A threshold of 0 disables automatic fallback.
        """
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        lib.gv_db_set_exact_search_threshold(self._db, int(threshold))

    def set_force_exact_search(self, enabled: bool) -> None:
        """
        Force or disable exact search regardless of collection size.
        This is mainly intended for testing and benchmarking.
        """
        lib.gv_db_set_force_exact_search(self._db, 1 if enabled else 0)

    def set_cosine_normalized(self, enabled: bool) -> None:
        """
        Enable or disable L2 pre-normalization for subsequently inserted dense vectors.

        When enabled, all new inserts are normalized to unit length. For cosine
        distance, this allows treating similarity as negative dot product.
        """
        lib.gv_db_set_cosine_normalized(self._db, 1 if enabled else 0)

    def train_ivfpq(self, data: Sequence[Sequence[float]]):
        """Train IVF-PQ index with provided vectors (only for IVFPQ index)."""
        flat = [item for vec in data for item in vec]
        count = len(data)
        if count == 0:
            raise ValueError("training data empty")
        if len(flat) % count != 0:
            raise ValueError("inconsistent training data")
        if (len(flat) // count) != self.dimension:
            raise ValueError("training vectors must match db dimension")
        buf = ffi.new("float[]", flat)
        rc = lib.gv_db_ivfpq_train(self._db, buf, count, self.dimension)
        if rc != 0:
            raise RuntimeError("gv_db_ivfpq_train failed")

    def start_background_compaction(self) -> None:
        """
        Start background compaction thread.

        The compaction thread periodically:
        - Removes deleted vectors from storage
        - Rebuilds indexes to remove gaps
        - Compacts WAL when it grows too large
        """
        rc = lib.gv_db_start_background_compaction(self._db)
        if rc != 0:
            raise RuntimeError("gv_db_start_background_compaction failed")

    def stop_background_compaction(self) -> None:
        """
        Stop background compaction thread gracefully.
        """
        lib.gv_db_stop_background_compaction(self._db)

    def compact(self) -> None:
        """
        Manually trigger compaction (runs synchronously).

        This performs the same compaction operations as the background thread
        but runs synchronously in the current thread.
        """
        rc = lib.gv_db_compact(self._db)
        if rc != 0:
            raise RuntimeError("gv_db_compact failed")

    def set_compaction_interval(self, interval_sec: int) -> None:
        """
        Set compaction interval in seconds.

        Args:
            interval_sec: Compaction interval in seconds (default: 300).
        """
        if interval_sec < 0:
            raise ValueError("interval_sec must be non-negative")
        lib.gv_db_set_compaction_interval(self._db, int(interval_sec))

    def set_wal_compaction_threshold(self, threshold_bytes: int) -> None:
        """
        Set WAL compaction threshold in bytes.

        Args:
            threshold_bytes: WAL size threshold for compaction (default: 10MB).
        """
        if threshold_bytes < 0:
            raise ValueError("threshold_bytes must be non-negative")
        lib.gv_db_set_wal_compaction_threshold(self._db, int(threshold_bytes))

    def set_deleted_ratio_threshold(self, ratio: float) -> None:
        """
        Set deleted vector ratio threshold for triggering compaction.

        Compaction is triggered when the ratio of deleted vectors exceeds this threshold.

        Args:
            ratio: Threshold ratio (0.0 to 1.0, default: 0.1).
        """
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("ratio must be between 0.0 and 1.0")
        lib.gv_db_set_deleted_ratio_threshold(self._db, float(ratio))

    def set_resource_limits(
        self,
        max_memory_bytes: int | None = None,
        max_vectors: int | None = None,
        max_concurrent_operations: int | None = None,
    ) -> None:
        """
        Set resource limits for the database.

        Args:
            max_memory_bytes: Maximum memory usage in bytes (0 or None = unlimited).
            max_vectors: Maximum number of vectors (0 or None = unlimited).
            max_concurrent_operations: Maximum concurrent operations (0 or None = unlimited).
        """
        limits = ffi.new("GV_ResourceLimits *")
        limits.max_memory_bytes = max_memory_bytes if max_memory_bytes is not None else 0
        limits.max_vectors = max_vectors if max_vectors is not None else 0
        limits.max_concurrent_operations = max_concurrent_operations if max_concurrent_operations is not None else 0

        rc = lib.gv_db_set_resource_limits(self._db, limits)
        if rc != 0:
            raise RuntimeError("gv_db_set_resource_limits failed")

    def get_resource_limits(self) -> dict[str, int]:
        """
        Get current resource limits.

        Returns:
            Dictionary with 'max_memory_bytes', 'max_vectors', 'max_concurrent_operations'.
        """
        limits = ffi.new("GV_ResourceLimits *")
        lib.gv_db_get_resource_limits(self._db, limits)
        return {
            "max_memory_bytes": limits.max_memory_bytes,
            "max_vectors": limits.max_vectors,
            "max_concurrent_operations": limits.max_concurrent_operations,
        }

    def get_memory_usage(self) -> int:
        """
        Get current estimated memory usage in bytes.

        Returns:
            Current memory usage in bytes.
        """
        return lib.gv_db_get_memory_usage(self._db)

    def get_concurrent_operations(self) -> int:
        """
        Get current number of concurrent operations.

        Returns:
            Current number of concurrent operations.
        """
        return lib.gv_db_get_concurrent_operations(self._db)

    def get_detailed_stats(self) -> dict:
        """
        Get detailed statistics for the database.

        Returns:
            Dictionary containing detailed statistics including:
            - basic_stats: Basic aggregated statistics
            - insert_latency: Insert operation latency histogram
            - search_latency: Search operation latency histogram
            - queries_per_second: Current QPS
            - inserts_per_second: Current IPS
            - memory: Memory usage breakdown
            - recall: Recall metrics for approximate search
            - health_status: Health status (0=healthy, -1=degraded, -2=unhealthy)
            - deleted_vector_count: Number of deleted vectors
            - deleted_ratio: Ratio of deleted vectors
        """
        stats = ffi.new("GV_DetailedStats *")
        rc = lib.gv_db_get_detailed_stats(self._db, stats)
        if rc != 0:
            raise RuntimeError("gv_db_get_detailed_stats failed")

        result = {
            "basic_stats": {
                "total_inserts": stats.basic_stats.total_inserts,
                "total_queries": stats.basic_stats.total_queries,
                "total_range_queries": stats.basic_stats.total_range_queries,
                "total_wal_records": stats.basic_stats.total_wal_records,
            },
            "queries_per_second": stats.queries_per_second,
            "inserts_per_second": stats.inserts_per_second,
            "memory": {
                "soa_storage_bytes": stats.memory.soa_storage_bytes,
                "index_bytes": stats.memory.index_bytes,
                "metadata_index_bytes": stats.memory.metadata_index_bytes,
                "wal_bytes": stats.memory.wal_bytes,
                "total_bytes": stats.memory.total_bytes,
            },
            "recall": {
                "total_queries": stats.recall.total_queries,
                "avg_recall": stats.recall.avg_recall,
                "min_recall": stats.recall.min_recall,
                "max_recall": stats.recall.max_recall,
            },
            "health_status": stats.health_status,
            "deleted_vector_count": stats.deleted_vector_count,
            "deleted_ratio": stats.deleted_ratio,
        }

        # Add latency histograms if available
        if stats.insert_latency.buckets != ffi.NULL and stats.insert_latency.bucket_count > 0:
            buckets = []
            for i in range(stats.insert_latency.bucket_count):
                buckets.append({
                    "count": stats.insert_latency.buckets[i],
                    "boundary_us": stats.insert_latency.bucket_boundaries[i],
                })
            result["insert_latency"] = {
                "buckets": buckets,
                "total_samples": stats.insert_latency.total_samples,
                "sum_latency_us": stats.insert_latency.sum_latency_us,
            }

        if stats.search_latency.buckets != ffi.NULL and stats.search_latency.bucket_count > 0:
            buckets = []
            for i in range(stats.search_latency.bucket_count):
                buckets.append({
                    "count": stats.search_latency.buckets[i],
                    "boundary_us": stats.search_latency.bucket_boundaries[i],
                })
            result["search_latency"] = {
                "buckets": buckets,
                "total_samples": stats.search_latency.total_samples,
                "sum_latency_us": stats.search_latency.sum_latency_us,
            }

        lib.gv_db_free_detailed_stats(stats)
        return result

    def health_check(self) -> int:
        """
        Perform health check on the database.

        Returns:
            0 if healthy, -1 if degraded, -2 if unhealthy.
        """
        return lib.gv_db_health_check(self._db)

    def record_recall(self, recall: float) -> None:
        """
        Record recall for a search operation.

        Args:
            recall: Recall value (0.0 to 1.0).
        """
        if recall < 0.0 or recall > 1.0:
            raise ValueError("recall must be between 0.0 and 1.0")
        lib.gv_db_record_recall(self._db, float(recall))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _check_dimension(self, vec: Sequence[float]):
        if len(vec) != self.dimension:
            raise ValueError(f"expected vector of dim {self.dimension}, got {len(vec)}")

    def add_vector(self, vector: Sequence[float], metadata: dict[str, str] | None = None):
        """
        Add a vector to the database with optional metadata.
        
        Args:
            vector: Vector data as a sequence of floats
            metadata: Optional dictionary of key-value metadata pairs.
                     Supports multiple entries; all entries are persisted via WAL when enabled.
        
        Raises:
            ValueError: If vector dimension doesn't match database dimension
            RuntimeError: If insertion fails
        """
        self._check_dimension(vector)
        buf = ffi.new("float[]", list(vector))
        
        if not metadata:
            # No metadata - use simple add
            rc = lib.gv_db_add_vector(self._db, buf, self.dimension)
            if rc != 0:
                raise RuntimeError("gv_db_add_vector failed")
            return
        
        metadata_items = list(metadata.items())
        if len(metadata_items) == 1:
            # Single entry - use optimized path (handles WAL and locking properly)
            k, v = metadata_items[0]
            rc = lib.gv_db_add_vector_with_metadata(self._db, buf, self.dimension, k.encode(), v.encode())
            if rc != 0:
                raise RuntimeError("gv_db_add_vector_with_metadata failed")
            return
        
        # Multiple metadata entries: use the rich C API (handles WAL + locking)
        key_cdatas = [ffi.new("char[]", k.encode()) for k, _ in metadata_items]
        val_cdatas = [ffi.new("char[]", v.encode()) for _, v in metadata_items]
        keys_c = ffi.new("const char * []", key_cdatas)
        vals_c = ffi.new("const char * []", val_cdatas)
        rc = lib.gv_db_add_vector_with_rich_metadata(
            self._db, buf, self.dimension, keys_c, vals_c, len(metadata_items)
        )
        if rc != 0:
            raise RuntimeError("gv_db_add_vector_with_rich_metadata failed")

    def add_vectors(self, vectors: Iterable[Sequence[float]]):
        data = [item for vec in vectors for item in vec]
        count = len(data) // self.dimension if self.dimension else 0
        if count * self.dimension != len(data):
            raise ValueError("all vectors must have the configured dimension")
        buf = ffi.new("float[]", data)
        rc = lib.gv_db_add_vectors(self._db, buf, count, self.dimension)
        if rc != 0:
            raise RuntimeError("gv_db_add_vectors failed")

    def delete_vector(self, vector_index: int):
        """
        Delete a vector from the database by its index (insertion order).
        
        Args:
            vector_index: Index of the vector to delete (0-based insertion order)
        
        Raises:
            RuntimeError: If deletion fails
        """
        rc = lib.gv_db_delete_vector_by_index(self._db, vector_index)
        if rc != 0:
            raise RuntimeError(f"gv_db_delete_vector_by_index failed for index {vector_index}")

    def update_vector(self, vector_index: int, new_data: Sequence[float]):
        """
        Update a vector in the database by its index (insertion order).
        
        Args:
            vector_index: Index of the vector to update (0-based insertion order)
            new_data: New vector data as a sequence of floats
        
        Raises:
            ValueError: If vector dimension doesn't match database dimension
            RuntimeError: If update fails
        """
        self._check_dimension(new_data)
        buf = ffi.new("float[]", list(new_data))
        rc = lib.gv_db_update_vector(self._db, vector_index, buf, self.dimension)
        if rc != 0:
            raise RuntimeError(f"gv_db_update_vector failed for index {vector_index}")

    def update_metadata(self, vector_index: int, metadata: dict[str, str]):
        """
        Update metadata for a vector in the database by its index.
        
        Args:
            vector_index: Index of the vector to update (0-based insertion order)
            metadata: Dictionary of key-value metadata pairs to set
        
        Raises:
            RuntimeError: If update fails
        """
        if not metadata:
            return
        
        metadata_items = list(metadata.items())
        key_cdatas = [ffi.new("char[]", k.encode()) for k, _ in metadata_items]
        val_cdatas = [ffi.new("char[]", v.encode()) for _, v in metadata_items]
        keys_c = ffi.new("const char * []", key_cdatas)
        vals_c = ffi.new("const char * []", val_cdatas)
        rc = lib.gv_db_update_vector_metadata(
            self._db, vector_index, keys_c, vals_c, len(metadata_items)
        )
        if rc != 0:
            raise RuntimeError(f"gv_db_update_vector_metadata failed for index {vector_index}")

    def search(self, query: Sequence[float], k: int, distance: DistanceType = DistanceType.EUCLIDEAN,
               filter_metadata: tuple[str, str] | None = None) -> list[SearchHit]:
        self._check_dimension(query)
        qbuf = ffi.new("float[]", list(query))
        results = ffi.new("GV_SearchResult[]", k)
        if filter_metadata:
            key, value = filter_metadata
            n = lib.gv_db_search_filtered(self._db, qbuf, k, results, int(distance), key.encode(), value.encode())
        else:
            n = lib.gv_db_search(self._db, qbuf, k, results, int(distance))
        if n < 0:
            raise RuntimeError("gv_db_search failed")
        out: list[SearchHit] = []
        for i in range(n):
            res = results[i]
            try:
                if res.is_sparse:
                    if res.sparse_vector != ffi.NULL:
                        vec = _copy_sparse_vector(res.sparse_vector, self.dimension)
                        out.append(SearchHit(distance=float(res.distance), vector=vec))
                else:
                    if res.vector != ffi.NULL:
                        vec = _copy_vector(res.vector)
                        out.append(SearchHit(distance=float(res.distance), vector=vec))
            except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
                continue
        return out

    def search_with_filter_expr(self, query: Sequence[float], k: int,
                                distance: DistanceType = DistanceType.EUCLIDEAN,
                                filter_expr: str | None = None) -> list[SearchHit]:
        """
        Advanced search with a metadata filter expression.

        The filter expression supports logical operators (AND, OR, NOT),
        comparison operators (==, !=, >, >=, <, <=) on numeric or string
        metadata, and string matching (CONTAINS, PREFIX).

        Example:
            db.search_with_filter_expr(
                [0.1] * 128,
                k=10,
                distance=DistanceType.EUCLIDEAN,
                filter_expr='category == "A" AND score >= 0.5'
            )
        """
        if filter_expr is None:
            raise ValueError("filter_expr must be provided")
        self._check_dimension(query)
        qbuf = ffi.new("float[]", list(query))
        results = ffi.new("GV_SearchResult[]", k)
        n = lib.gv_db_search_with_filter_expr(self._db, qbuf, k, results, int(distance), filter_expr.encode())
        if n < 0:
            raise RuntimeError("gv_db_search_with_filter_expr failed")
        out: list[SearchHit] = []
        for i in range(n):
            res = results[i]
            if res.is_sparse and res.sparse_vector != ffi.NULL:
                out.append(SearchHit(distance=float(res.distance),
                                     vector=_copy_sparse_vector(res.sparse_vector, self.dimension)))
            else:
                out.append(SearchHit(distance=float(res.distance), vector=_copy_vector(res.vector)))
        return out

    def add_sparse_vector(self, indices: Sequence[int], values: Sequence[float],
                          metadata: dict[str, str] | None = None) -> None:
        if self._db is None or self._closed:
            raise RuntimeError("database is closed")
        if len(indices) != len(values):
            raise ValueError("indices and values must have same length")
        nnz = len(indices)
        idx_buf = ffi.new("uint32_t[]", [int(i) for i in indices])
        val_buf = ffi.new("float[]", [float(v) for v in values])
        key = None
        val = None
        if metadata:
            if len(metadata) != 1:
                raise ValueError("only one metadata key/value supported in this helper")
            key, val = next(iter(metadata.items()))
        rc = lib.gv_db_add_sparse_vector(self._db, idx_buf, val_buf, nnz, self.dimension,
                                         key.encode() if key else ffi.NULL,
                                         val.encode() if val else ffi.NULL)
        if rc != 0:
            raise RuntimeError("gv_db_add_sparse_vector failed")

    def search_sparse(self, indices: Sequence[int], values: Sequence[float], k: int,
                      distance: DistanceType = DistanceType.DOT_PRODUCT) -> list[SearchHit]:
        if len(indices) != len(values):
            raise ValueError("indices and values must have same length")
        nnz = len(indices)
        idx_buf = ffi.new("uint32_t[]", [int(i) for i in indices])
        val_buf = ffi.new("float[]", [float(v) for v in values])
        results = ffi.new("GV_SearchResult[]", k)
        n = lib.gv_db_search_sparse(self._db, idx_buf, val_buf, nnz, k, results, int(distance))
        if n < 0:
            raise RuntimeError("gv_db_search_sparse failed")
        out: list[SearchHit] = []
        for i in range(n):
            res = results[i]
            if res.sparse_vector != ffi.NULL:
                out.append(SearchHit(distance=float(res.distance),
                                     vector=_copy_sparse_vector(res.sparse_vector, self.dimension)))
        return out

    def range_search(self, query: Sequence[float], radius: float, max_results: int = 1000,
                     distance: DistanceType = DistanceType.EUCLIDEAN,
                     filter_metadata: tuple[str, str] | None = None) -> list[SearchHit]:
        """
        Range search: find all vectors within a distance threshold.
        
        Args:
            query: Query vector.
            radius: Maximum distance threshold (inclusive).
            max_results: Maximum number of results to return.
            distance: Distance metric to use.
            filter_metadata: Optional (key, value) tuple for metadata filtering.
        
        Returns:
            List of search hits within the radius.
        """
        self._check_dimension(query)
        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        
        qbuf = ffi.new("float[]", list(query))
        results = ffi.new("GV_SearchResult[]", max_results)
        if filter_metadata:
            key, value = filter_metadata
            n = lib.gv_db_range_search_filtered(self._db, qbuf, radius, results, max_results,
                                                int(distance), key.encode(), value.encode())
        else:
            n = lib.gv_db_range_search(self._db, qbuf, radius, results, max_results, int(distance))
        if n < 0:
            raise RuntimeError("gv_db_range_search failed")
        return [SearchHit(distance=float(results[i].distance), vector=_copy_vector(results[i].vector)) for i in range(n)]

    def search_batch(self, queries: Iterable[Sequence[float]], k: int,
                     distance: DistanceType = DistanceType.EUCLIDEAN) -> list[list[SearchHit]]:
        queries_list = list(queries)
        if not queries_list:
            return []
        for q in queries_list:
            self._check_dimension(q)
        flat = [item for q in queries_list for item in q]
        qbuf = ffi.new("float[]", flat)
        results = ffi.new("GV_SearchResult[]", len(queries_list) * k)
        n = lib.gv_db_search_batch(self._db, qbuf, len(queries_list), k, results, int(distance))
        if n < 0:
            raise RuntimeError("gv_db_search_batch failed")
        out: list[list[SearchHit]] = []
        for qi in range(len(queries_list)):
            hits = []
            for hi in range(k):
                res = results[qi * k + hi]
                hits.append(SearchHit(distance=float(res.distance), vector=_copy_vector(res.vector)))
            out.append(hits)
        return out

    def search_ivfpq_opts(self, query: Sequence[float], k: int,
                          distance: DistanceType = DistanceType.EUCLIDEAN,
                          nprobe_override: int | None = None, rerank_top: int | None = None) -> list[SearchHit]:
        self._check_dimension(query)
        qbuf = ffi.new("float[]", list(query))
        results = ffi.new("GV_SearchResult[]", k)
        nprobe = nprobe_override if nprobe_override is not None else 4
        rerank = rerank_top if rerank_top is not None else 32
        n = lib.gv_db_search_ivfpq_opts(self._db, qbuf, k, results, int(distance), nprobe, rerank)
        if n < 0:
            raise RuntimeError("gv_db_search_ivfpq_opts failed")
        out: list[SearchHit] = []
        for i in range(n):
            res = results[i]
            if res.vector != ffi.NULL:
                vec = _copy_vector(res.vector)
                out.append(SearchHit(distance=float(res.distance), vector=vec))
        return out

    def record_latency(self, latency_us: int, is_insert: bool):
        lib.gv_db_record_latency(self._db, latency_us, 1 if is_insert else 0)

    def record_recall(self, recall: float):
        if not (0.0 <= recall <= 1.0):
            raise ValueError("recall must be between 0.0 and 1.0")
        lib.gv_db_record_recall(self._db, recall)

    def health_check(self) -> int:
        return lib.gv_db_health_check(self._db)

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Avoid raising during interpreter shutdown
            pass

