from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import TracebackType
from typing import Any, Callable, Iterable, Optional, Sequence, TypeAlias

from ._ffi import ffi, lib

# Type alias for CFFI pointer types.
# At runtime, this is Any to allow cffi's dynamic CData objects.
# For type checking, this documents that the value is a CFFI pointer.
CData: TypeAlias = Any


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
    scalar_quant_config: Optional[ScalarQuantConfig] = None
    oversampling_factor: float = 1.0

    def __post_init__(self) -> None:
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


def _metadata_to_dict(meta_ptr: CData) -> dict[str, str]:
    """Convert C metadata linked list to Python dict.

    Args:
        meta_ptr: CFFI pointer to GV_Metadata structure (linked list).

    Returns:
        Dictionary of key-value metadata pairs.
    """
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


def _copy_vector(vec_ptr: CData) -> Vector:
    """Copy a C vector structure to a Python Vector object.

    Args:
        vec_ptr: CFFI pointer to GV_Vector structure.

    Returns:
        Python Vector object with copied data and metadata.
    """
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


def _copy_sparse_vector(sv_ptr: CData, dim: int) -> Vector:
    """Copy a C sparse vector structure to a dense Python Vector object.

    Args:
        sv_ptr: CFFI pointer to GV_SparseVector structure.
        dim: Target dimension for the dense vector.

    Returns:
        Python Vector object with sparse values expanded to dense representation.
    """
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
    """GigaVector database for storing and querying high-dimensional vectors."""

    _db: CData
    dimension: int
    _closed: bool

    def __init__(self, handle: CData, dimension: int) -> None:
        self._db = handle
        self.dimension = int(dimension)
        self._closed = False

    @classmethod
    def open(cls, path: str | None, dimension: int, index: IndexType = IndexType.KDTREE,
             hnsw_config: HNSWConfig | None = None, ivfpq_config: IVFPQConfig | None = None) -> Database:
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
            sq_cfg = ivfpq_config.scalar_quant_config or ScalarQuantConfig()
            sq_config = ffi.new("GV_ScalarQuantConfig *", {
                "bits": sq_cfg.bits,
                "per_dimension": 1 if sq_cfg.per_dimension else 0
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
                  ivfpq_config: IVFPQConfig | None = None) -> Database:
        """Open a database and automatically choose a reasonable index type.

        Args:
            path: Optional path for persistence (None for in-memory).
            dimension: Vector dimensionality.
            expected_count: Optional estimate of the number of vectors.
            hnsw_config: Optional HNSW configuration (used if HNSW is selected).
            ivfpq_config: Optional IVFPQ configuration (used if IVFPQ is selected).

        Returns:
            Database instance with automatically selected index type.
        """
        index = _choose_index_type(dimension, expected_count)
        return cls.open(path, dimension, index=index,
                        hnsw_config=hnsw_config, ivfpq_config=ivfpq_config)

    @classmethod
    def open_mmap(cls, path: str, dimension: int, index: IndexType = IndexType.KDTREE) -> Database:
        """Open a read-only database by memory-mapping an existing snapshot file.

        This is a thin wrapper around gv_db_open_mmap(). The returned Database
        instance shares the mapped file; modifications are not persisted.

        Args:
            path: Path to the snapshot file.
            dimension: Vector dimensionality.
            index: Index type to use.

        Returns:
            Database instance backed by memory-mapped file.
        """
        if not path:
            raise ValueError("path must be non-empty")
        c_path = path.encode("utf-8")
        db = lib.gv_db_open_mmap(c_path, dimension, int(index))
        if db == ffi.NULL:
            raise RuntimeError("gv_db_open_mmap failed")
        return cls(db, dimension)

    def close(self) -> None:
        """Close the database and release resources."""
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

    def save(self, path: str | None = None) -> None:
        """Persist the database to a binary snapshot file.

        Args:
            path: Output path. If None, uses the path from open().
        """
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

    def train_ivfpq(self, data: Sequence[Sequence[float]]) -> None:
        """Train IVF-PQ index with provided vectors (only for IVFPQ index).

        Args:
            data: Training vectors, each must have the same dimension as the database.

        Raises:
            ValueError: If training data is empty or has inconsistent dimensions.
            RuntimeError: If training fails.
        """
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

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _check_dimension(self, vec: Sequence[float]) -> None:
        if len(vec) != self.dimension:
            raise ValueError(f"expected vector of dim {self.dimension}, got {len(vec)}")

    def add_vector(self, vector: Sequence[float], metadata: dict[str, str] | None = None) -> None:
        """Add a vector to the database with optional metadata.

        Args:
            vector: Vector data as a sequence of floats.
            metadata: Optional dictionary of key-value metadata pairs.
                Supports multiple entries; all entries are persisted via WAL when enabled.

        Raises:
            ValueError: If vector dimension doesn't match database dimension.
            RuntimeError: If insertion fails.
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

    def add_vectors(self, vectors: Iterable[Sequence[float]]) -> None:
        """Add multiple vectors to the database in batch.

        Args:
            vectors: Iterable of vectors, each must match the database dimension.

        Raises:
            ValueError: If any vector has incorrect dimension.
            RuntimeError: If insertion fails.
        """
        data = [item for vec in vectors for item in vec]
        count = len(data) // self.dimension if self.dimension else 0
        if count * self.dimension != len(data):
            raise ValueError("all vectors must have the configured dimension")
        buf = ffi.new("float[]", data)
        rc = lib.gv_db_add_vectors(self._db, buf, count, self.dimension)
        if rc != 0:
            raise RuntimeError("gv_db_add_vectors failed")

    def delete_vector(self, vector_index: int) -> None:
        """Delete a vector from the database by its index (insertion order).

        Args:
            vector_index: Index of the vector to delete (0-based insertion order).

        Raises:
            RuntimeError: If deletion fails.
        """
        rc = lib.gv_db_delete_vector_by_index(self._db, vector_index)
        if rc != 0:
            raise RuntimeError(f"gv_db_delete_vector_by_index failed for index {vector_index}")

    def update_vector(self, vector_index: int, new_data: Sequence[float]) -> None:
        """Update a vector in the database by its index (insertion order).

        Args:
            vector_index: Index of the vector to update (0-based insertion order).
            new_data: New vector data as a sequence of floats.

        Raises:
            ValueError: If vector dimension doesn't match database dimension.
            RuntimeError: If update fails.
        """
        self._check_dimension(new_data)
        buf = ffi.new("float[]", list(new_data))
        rc = lib.gv_db_update_vector(self._db, vector_index, buf, self.dimension)
        if rc != 0:
            raise RuntimeError(f"gv_db_update_vector failed for index {vector_index}")

    def update_metadata(self, vector_index: int, metadata: dict[str, str]) -> None:
        """Update metadata for a vector in the database by its index.

        Args:
            vector_index: Index of the vector to update (0-based insertion order).
            metadata: Dictionary of key-value metadata pairs to set.

        Raises:
            RuntimeError: If update fails.
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

    def record_latency(self, latency_us: int, is_insert: bool) -> None:
        """Record operation latency for monitoring.

        Args:
            latency_us: Latency in microseconds.
            is_insert: True for insert operations, False for search operations.
        """
        lib.gv_db_record_latency(self._db, latency_us, 1 if is_insert else 0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Avoid raising during interpreter shutdown
            pass


# ============================================================================
# LLM Module
# ============================================================================

class LLMError(IntEnum):
    SUCCESS = 0
    NULL_POINTER = -1
    INVALID_CONFIG = -2
    INVALID_API_KEY = -3
    INVALID_URL = -4
    MEMORY_ALLOCATION = -5
    CURL_INIT = -6
    NETWORK = -7
    TIMEOUT = -8
    RESPONSE_TOO_LARGE = -9
    PARSE_FAILED = -10
    INVALID_RESPONSE = -11
    CUSTOM_URL_REQUIRED = -12


class LLMProvider(IntEnum):
    """LLM provider enumeration.

    Supported providers:
    - OPENAI: OpenAI GPT models (tested, recommended)
    - GOOGLE: Google Gemini models (tested)
    - CUSTOM: Custom OpenAI-compatible endpoints

    Internal/experimental (not exposed to end users):
    - ANTHROPIC: Claude models (not yet tested due to API key unavailability)
    """
    OPENAI = 0
    ANTHROPIC = 1      # Internal: not yet tested, API keys unavailable
    GOOGLE = 2
    # AZURE_OPENAI removed - use CUSTOM with Azure endpoint instead
    CUSTOM = 3


@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout_seconds: int = 30
    custom_prompt: Optional[str] = None

    def _to_c_config(self) -> CData:
        """Convert to C configuration structure.

        Returns:
            CFFI pointer to GV_LLMConfig structure.
        """
        c_config = ffi.new("GV_LLMConfig *")
        c_config.provider = int(self.provider)
        c_config.api_key = ffi.new("char[]", self.api_key.encode())
        c_config.model = ffi.new("char[]", self.model.encode())
        c_config.base_url = ffi.new("char[]", self.base_url.encode()) if self.base_url else ffi.NULL
        c_config.temperature = self.temperature
        c_config.max_tokens = self.max_tokens
        c_config.timeout_seconds = self.timeout_seconds
        c_config.custom_prompt = ffi.new("char[]", self.custom_prompt.encode()) if self.custom_prompt else ffi.NULL
        return c_config


@dataclass
class LLMMessage:
    role: str
    content: str

    def _to_c_message(self) -> tuple[CData, bytes, bytes]:
        """Convert to C message structure.

        Returns:
            Tuple of (c_msg, role_bytes, content_bytes) to keep references alive.
        """
        role_bytes = self.role.encode()
        content_bytes = self.content.encode()
        c_msg = ffi.new("GV_LLMMessage *")
        c_msg.role = ffi.new("char[]", role_bytes)
        c_msg.content = ffi.new("char[]", content_bytes)
        return (c_msg, role_bytes, content_bytes)


@dataclass
class LLMResponse:
    content: str
    finish_reason: int
    token_count: int


class LLM:
    """LLM client for generating responses using various providers."""

    _llm: CData
    _closed: bool

    def __init__(self, config: LLMConfig) -> None:
        c_config = config._to_c_config()
        self._llm = lib.gv_llm_create(c_config)
        if self._llm == ffi.NULL:
            raise RuntimeError("Failed to create LLM instance. Make sure libcurl is installed and API key is valid.")
        self._closed = False

    def __enter__(self) -> LLM:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the LLM client and release resources."""
        if not self._closed and self._llm != ffi.NULL:
            lib.gv_llm_destroy(self._llm)
            self._llm = ffi.NULL
            self._closed = True

    def generate_response(
        self,
        messages: Sequence[LLMMessage],
        response_format: Optional[str] = None
    ) -> LLMResponse:
        if self._closed:
            raise ValueError("LLM instance is closed")

        # Allocate array of messages
        c_messages = ffi.new("GV_LLMMessage[]", len(messages))
        message_refs = []  # Keep references alive
        
        for i, msg in enumerate(messages):
            c_msg, role_bytes, content_bytes = msg._to_c_message()
            c_messages[i] = c_msg[0]
            message_refs.append((c_msg, role_bytes, content_bytes))

        response_format_bytes = response_format.encode() if response_format else ffi.NULL
        c_response = ffi.new("GV_LLMResponse *")

        result = lib.gv_llm_generate_response(
            self._llm, c_messages, len(messages), response_format_bytes, c_response
        )

        # Free message copies
        for c_msg, _, _ in message_refs:
            lib.gv_llm_message_free(c_msg)

        if result != 0:
            error_msg = lib.gv_llm_get_last_error(self._llm)
            error_str = lib.gv_llm_error_string(result)
            lib.gv_llm_response_free(c_response)
            error_detail = ffi.string(error_msg).decode("utf-8") if error_msg != ffi.NULL else error_str.decode("utf-8") if error_str != ffi.NULL else "Unknown error"
            raise RuntimeError(f"Failed to generate LLM response: {error_detail} (code: {result})")

        content = ffi.string(c_response.content).decode("utf-8") if c_response.content != ffi.NULL else ""
        response = LLMResponse(
            content=content,
            finish_reason=int(c_response.finish_reason),
            token_count=int(c_response.token_count)
        )

        lib.gv_llm_response_free(c_response)
        return response
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message from the LLM instance."""
        if self._closed or self._llm == ffi.NULL:
            return None
        error_msg = lib.gv_llm_get_last_error(self._llm)
        if error_msg == ffi.NULL:
            return None
        return ffi.string(error_msg).decode("utf-8")
    
    @staticmethod
    def error_string(error_code: int) -> str:
        """Get human-readable error description for an error code."""
        error_str = lib.gv_llm_error_string(error_code)
        if error_str == ffi.NULL:
            return "Unknown error"
        return ffi.string(error_str).decode("utf-8")


# ============================================================================
# Embedding Service Module
# ============================================================================

class EmbeddingProvider(IntEnum):
    OPENAI = 0
    HUGGINGFACE = 1
    CUSTOM = 2
    NONE = 3


@dataclass
class EmbeddingConfig:
    provider: EmbeddingProvider = EmbeddingProvider.NONE
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    embedding_dimension: int = 0
    batch_size: int = 100
    enable_cache: bool = True
    cache_size: int = 1000
    timeout_seconds: int = 30
    huggingface_model_path: Optional[str] = None

    def _to_c_config(self) -> CData:
        """Convert to C configuration structure.

        Returns:
            CFFI pointer to GV_EmbeddingConfig structure.
        """
        c_config = ffi.new("GV_EmbeddingConfig *")
        c_config.provider = int(self.provider)
        if self.api_key:
            c_config.api_key = ffi.new("char[]", self.api_key.encode())
        else:
            c_config.api_key = ffi.NULL
        if self.model:
            c_config.model = ffi.new("char[]", self.model.encode())
        else:
            c_config.model = ffi.NULL
        if self.base_url:
            c_config.base_url = ffi.new("char[]", self.base_url.encode())
        else:
            c_config.base_url = ffi.NULL
        c_config.embedding_dimension = self.embedding_dimension
        c_config.batch_size = self.batch_size
        c_config.enable_cache = 1 if self.enable_cache else 0
        c_config.cache_size = self.cache_size
        c_config.timeout_seconds = self.timeout_seconds
        if self.huggingface_model_path:
            c_config.huggingface_model_path = ffi.new("char[]", self.huggingface_model_path.encode())
        else:
            c_config.huggingface_model_path = ffi.NULL
        return c_config


class EmbeddingService:
    """Embedding service for generating vector embeddings from text."""

    _service: CData
    _closed: bool

    def __init__(self, config: EmbeddingConfig) -> None:
        c_config = config._to_c_config()
        self._service = lib.gv_embedding_service_create(c_config)
        if self._service == ffi.NULL:
            raise RuntimeError("Failed to create embedding service")
        self._closed = False

    def __enter__(self) -> EmbeddingService:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the embedding service and release resources."""
        if not self._closed and self._service != ffi.NULL:
            lib.gv_embedding_service_destroy(self._service)
            self._service = ffi.NULL
            self._closed = True
    
    def generate(self, text: str) -> Optional[Sequence[float]]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None on error
        """
        if self._closed:
            raise ValueError("Embedding service is closed")
        
        embedding_dim_ptr = ffi.new("size_t *")
        embedding_ptr = ffi.new("float **")
        
        result = lib.gv_embedding_generate(
            self._service, text.encode(), embedding_dim_ptr, embedding_ptr
        )
        
        if result != 0:
            return None
        
        if embedding_ptr[0] == ffi.NULL:
            return None
        
        embedding = [embedding_ptr[0][i] for i in range(embedding_dim_ptr[0])]
        lib.free(embedding_ptr[0])
        
        return embedding
    
    def generate_batch(self, texts: Sequence[str]) -> list[Optional[Sequence[float]]]:
        """Generate embeddings for multiple texts (batch operation).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        if self._closed:
            raise ValueError("Embedding service is closed")
        
        if not texts:
            return []
        
        text_ptrs = [ffi.new("char[]", text.encode()) for text in texts]
        text_array = ffi.new("char *[]", text_ptrs)
        
        embedding_dims_ptr = ffi.new("size_t **")
        embeddings_ptr = ffi.new("float ***")
        
        result = lib.gv_embedding_generate_batch(
            self._service, text_array, len(texts), embedding_dims_ptr, embeddings_ptr
        )
        
        if result < 0:
            return [None] * len(texts)

        embeddings: list[Optional[Sequence[float]]] = []
        if embeddings_ptr[0] != ffi.NULL:
            for i in range(len(texts)):
                if embeddings_ptr[0][i] != ffi.NULL and embedding_dims_ptr[0][i] > 0:
                    emb: list[float] = [embeddings_ptr[0][i][j] for j in range(embedding_dims_ptr[0][i])]
                    lib.free(embeddings_ptr[0][i])
                    embeddings.append(emb)
                else:
                    embeddings.append(None)
            lib.free(embeddings_ptr[0])
            lib.free(embedding_dims_ptr[0])

        return embeddings


class EmbeddingCache:
    """Embedding cache for storing and retrieving embeddings."""

    _cache: CData
    _closed: bool

    def __init__(self, max_size: int = 1000) -> None:
        self._cache = lib.gv_embedding_cache_create(max_size)
        if self._cache == ffi.NULL:
            raise RuntimeError("Failed to create embedding cache")
        self._closed = False

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the cache and release resources."""
        if not self._closed and self._cache != ffi.NULL:
            lib.gv_embedding_cache_destroy(self._cache)
            self._cache = ffi.NULL
            self._closed = True
    
    def get(self, text: str) -> Optional[Sequence[float]]:
        """Get embedding from cache.
        
        Args:
            text: Text key
            
        Returns:
            Embedding vector if found, None otherwise
        """
        if self._closed:
            raise ValueError("Cache is closed")
        
        embedding_dim_ptr = ffi.new("size_t *")
        embedding_ptr = ffi.new("const float **")
        
        result = lib.gv_embedding_cache_get(
            self._cache, text.encode(), embedding_dim_ptr, embedding_ptr
        )
        
        if result != 1:
            return None
        
        if embedding_ptr[0] == ffi.NULL:
            return None
        
        embedding = [embedding_ptr[0][i] for i in range(embedding_dim_ptr[0])]
        return embedding
    
    def put(self, text: str, embedding: Sequence[float]) -> bool:
        """Store embedding in cache.
        
        Args:
            text: Text key
            embedding: Embedding vector
            
        Returns:
            True on success, False on error
        """
        if self._closed:
            raise ValueError("Cache is closed")
        
        c_embedding = ffi.new("float[]", embedding)
        result = lib.gv_embedding_cache_put(
            self._cache, text.encode(), len(embedding), c_embedding
        )
        
        return result == 0
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        if self._closed:
            raise ValueError("Cache is closed")
        lib.gv_embedding_cache_clear(self._cache)
    
    def stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with 'size', 'hits', 'misses'
        """
        if self._closed:
            raise ValueError("Cache is closed")
        
        size_ptr = ffi.new("size_t *")
        hits_ptr = ffi.new("uint64_t *")
        misses_ptr = ffi.new("uint64_t *")
        
        lib.gv_embedding_cache_stats(self._cache, size_ptr, hits_ptr, misses_ptr)
        
        return {
            'size': size_ptr[0],
            'hits': hits_ptr[0],
            'misses': misses_ptr[0]
        }


# ============================================================================
# Memory Layer Module
# ============================================================================

class MemoryType(IntEnum):
    FACT = 0
    PREFERENCE = 1
    RELATIONSHIP = 2
    EVENT = 3


class ConsolidationStrategy(IntEnum):
    MERGE = 0
    UPDATE = 1
    LINK = 2
    ARCHIVE = 3


@dataclass(frozen=True)
class MemoryMetadata:
    memory_id: Optional[str] = None
    memory_type: MemoryType = MemoryType.FACT
    source: Optional[str] = None
    timestamp: Optional[int] = None
    importance_score: float = 0.5
    extraction_metadata: Optional[str] = None
    related_memory_ids: Sequence[str] = ()
    consolidated: bool = False


@dataclass(frozen=True)
class MemoryResult:
    memory_id: str
    content: str
    relevance_score: float
    distance: float
    metadata: Optional[MemoryMetadata] = None
    related: Sequence[MemoryResult] = ()


@dataclass
class MemoryLayerConfig:
    extraction_threshold: float = 0.5
    consolidation_threshold: float = 0.85
    default_strategy: ConsolidationStrategy = ConsolidationStrategy.MERGE
    enable_temporal_weighting: bool = True
    enable_relationship_retrieval: bool = True
    max_related_memories: int = 5
    llm_config: Optional[LLMConfig] = None
    use_llm_extraction: bool = True
    use_llm_consolidation: bool = False


def _create_c_metadata(meta: Optional[MemoryMetadata]) -> CData:
    """Convert Python MemoryMetadata to C structure.

    Args:
        meta: Python memory metadata object, or None.

    Returns:
        CFFI pointer to GV_MemoryMetadata, or ffi.NULL if meta is None.
    """
    if meta is None:
        return ffi.NULL

    c_meta = ffi.new("GV_MemoryMetadata *")
    c_meta.memory_id = ffi.new("char[]", meta.memory_id.encode()) if meta.memory_id else ffi.NULL
    c_meta.memory_type = int(meta.memory_type)
    c_meta.source = ffi.new("char[]", meta.source.encode()) if meta.source else ffi.NULL
    c_meta.timestamp = meta.timestamp if meta.timestamp else 0
    c_meta.importance_score = meta.importance_score
    c_meta.extraction_metadata = ffi.new("char[]", meta.extraction_metadata.encode()) if meta.extraction_metadata else ffi.NULL
    c_meta.related_count = len(meta.related_memory_ids) if meta.related_memory_ids else 0
    c_meta.consolidated = 1 if meta.consolidated else 0

    if meta.related_memory_ids:
        c_meta.related_memory_ids = ffi.new("char*[]", [ffi.new("char[]", id.encode()) for id in meta.related_memory_ids])
    else:
        c_meta.related_memory_ids = ffi.NULL

    return c_meta


def _copy_memory_metadata(c_meta_ptr: CData) -> Optional[MemoryMetadata]:
    """Copy C memory metadata structure to Python object.

    Args:
        c_meta_ptr: CFFI pointer to GV_MemoryMetadata structure.

    Returns:
        Python MemoryMetadata object, or None on error.
    """
    if c_meta_ptr == ffi.NULL:
        return None

    try:
        memory_id = ffi.string(c_meta_ptr.memory_id).decode("utf-8") if c_meta_ptr.memory_id != ffi.NULL else None
        source = ffi.string(c_meta_ptr.source).decode("utf-8") if c_meta_ptr.source != ffi.NULL else None
        extraction_metadata = ffi.string(c_meta_ptr.extraction_metadata).decode("utf-8") if c_meta_ptr.extraction_metadata != ffi.NULL else None

        related_ids = []
        if c_meta_ptr.related_memory_ids != ffi.NULL and c_meta_ptr.related_count > 0:
            for i in range(c_meta_ptr.related_count):
                if c_meta_ptr.related_memory_ids[i] != ffi.NULL:
                    related_ids.append(ffi.string(c_meta_ptr.related_memory_ids[i]).decode("utf-8"))

        return MemoryMetadata(
            memory_id=memory_id,
            memory_type=MemoryType(int(c_meta_ptr.memory_type)),
            source=source,
            timestamp=int(c_meta_ptr.timestamp) if c_meta_ptr.timestamp > 0 else None,
            importance_score=float(c_meta_ptr.importance_score),
            extraction_metadata=extraction_metadata,
            related_memory_ids=tuple(related_ids),
            consolidated=bool(c_meta_ptr.consolidated)
        )
    except (AttributeError, TypeError, ValueError, UnicodeDecodeError):
        return None


def _copy_memory_result(c_result_ptr: CData) -> Optional[MemoryResult]:
    """Copy C memory result structure to Python object.

    Args:
        c_result_ptr: CFFI pointer to GV_MemoryResult structure.

    Returns:
        Python MemoryResult object, or None on error.
    """
    if c_result_ptr == ffi.NULL:
        return None

    try:
        memory_id = ffi.string(c_result_ptr.memory_id).decode("utf-8") if c_result_ptr.memory_id != ffi.NULL else ""
        content = ffi.string(c_result_ptr.content).decode("utf-8") if c_result_ptr.content != ffi.NULL else ""
        
        metadata = _copy_memory_metadata(c_result_ptr.metadata) if c_result_ptr.metadata != ffi.NULL else None
        
        related = []
        if c_result_ptr.related != ffi.NULL and c_result_ptr.related_count > 0:
            for i in range(c_result_ptr.related_count):
                if c_result_ptr.related[i] != ffi.NULL:
                    rel = _copy_memory_result(c_result_ptr.related[i])
                    if rel:
                        related.append(rel)
        
        return MemoryResult(
            memory_id=memory_id,
            content=content,
            relevance_score=float(c_result_ptr.relevance_score),
            distance=float(c_result_ptr.distance),
            metadata=metadata,
            related=tuple(related)
        )
    except (AttributeError, TypeError, ValueError, UnicodeDecodeError):
        return None


class MemoryLayer:
    """Memory layer for semantic memory storage and retrieval."""

    _layer: CData
    _db: Database
    _closed: bool

    def __init__(self, db: Database, config: Optional[MemoryLayerConfig] = None) -> None:
        if db._closed:
            raise ValueError("Database is closed")

        if config is None:
            config = MemoryLayerConfig()

        c_config = ffi.new("GV_MemoryLayerConfig *")
        c_config.extraction_threshold = config.extraction_threshold
        c_config.consolidation_threshold = config.consolidation_threshold
        c_config.default_strategy = int(config.default_strategy)
        c_config.enable_temporal_weighting = 1 if config.enable_temporal_weighting else 0
        c_config.enable_relationship_retrieval = 1 if config.enable_relationship_retrieval else 0
        c_config.max_related_memories = config.max_related_memories
        c_config.use_llm_extraction = 1 if config.use_llm_extraction else 0
        c_config.use_llm_consolidation = 1 if config.use_llm_consolidation else 0
        
        # Set LLM config if provided
        if config.llm_config:
            c_llm_config = config.llm_config._to_c_config()
            c_config.llm_config = c_llm_config
        else:
            c_config.llm_config = ffi.NULL
        
        self._layer = lib.gv_memory_layer_create(db._db, c_config)
        if self._layer == ffi.NULL:
            raise RuntimeError("Failed to create memory layer")
        
        self._db = db
        self._closed = False

    def __enter__(self) -> MemoryLayer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the memory layer and release resources."""
        if not self._closed and self._layer != ffi.NULL:
            lib.gv_memory_layer_destroy(self._layer)
            self._layer = ffi.NULL
            self._closed = True
    
    def add(self, content: str, embedding: Sequence[float], metadata: Optional[MemoryMetadata] = None) -> str:
        if self._closed:
            raise ValueError("Memory layer is closed")
        
        if len(embedding) != self._db.dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} does not match database dimension {self._db.dimension}")
        
        c_embedding = ffi.new("float[]", embedding)
        c_meta = _create_c_metadata(metadata) if metadata else ffi.NULL
        
        memory_id_ptr = lib.gv_memory_add(self._layer, content.encode(), c_embedding, c_meta)
        if memory_id_ptr == ffi.NULL:
            raise RuntimeError("Failed to add memory")
        
        memory_id = ffi.string(memory_id_ptr).decode("utf-8")
        lib.free(memory_id_ptr)
        
        return memory_id
    
    def extract_from_conversation(self, conversation: str, conversation_id: Optional[str] = None) -> list[str]:
        if self._closed:
            raise ValueError("Memory layer is closed")
        
        conv_id_bytes = conversation_id.encode() if conversation_id else ffi.NULL
        embeddings_ptr = ffi.new("float**")
        count_ptr = ffi.new("size_t *")
        
        memory_ids_ptr = lib.gv_memory_extract_from_conversation(
            self._layer, conversation.encode(), conv_id_bytes, embeddings_ptr, count_ptr
        )
        
        if memory_ids_ptr == ffi.NULL:
            return []
        
        count = int(count_ptr[0])
        memory_ids = []
        for i in range(count):
            if memory_ids_ptr[i] != ffi.NULL:
                memory_ids.append(ffi.string(memory_ids_ptr[i]).decode("utf-8"))
                lib.free(memory_ids_ptr[i])
        
        lib.free(memory_ids_ptr)
        if embeddings_ptr[0] != ffi.NULL:
            lib.free(embeddings_ptr[0])
        
        return memory_ids
    
    def extract_from_text(self, text: str, source: Optional[str] = None) -> list[str]:
        if self._closed:
            raise ValueError("Memory layer is closed")
        
        source_bytes = source.encode() if source else ffi.NULL
        embeddings_ptr = ffi.new("float**")
        count_ptr = ffi.new("size_t *")
        
        memory_ids_ptr = lib.gv_memory_extract_from_text(
            self._layer, text.encode(), source_bytes, embeddings_ptr, count_ptr
        )
        
        if memory_ids_ptr == ffi.NULL:
            return []
        
        count = int(count_ptr[0])
        memory_ids = []
        for i in range(count):
            if memory_ids_ptr[i] != ffi.NULL:
                memory_ids.append(ffi.string(memory_ids_ptr[i]).decode("utf-8"))
                lib.free(memory_ids_ptr[i])
        
        lib.free(memory_ids_ptr)
        if embeddings_ptr[0] != ffi.NULL:
            lib.free(embeddings_ptr[0])
        
        return memory_ids
    
    def consolidate(self, threshold: Optional[float] = None, strategy: Optional[ConsolidationStrategy] = None) -> int:
        if self._closed:
            raise ValueError("Memory layer is closed")
        
        actual_threshold = threshold if threshold is not None else -1.0
        actual_strategy = int(strategy) if strategy is not None else -1
        
        result = lib.gv_memory_consolidate(self._layer, actual_threshold, actual_strategy)
        if result < 0:
            raise RuntimeError("Failed to consolidate memories")
        
        return result
    
    def search(self, query_embedding: Sequence[float], k: int = 10, distance: DistanceType = DistanceType.COSINE) -> list[MemoryResult]:
        if self._closed:
            raise ValueError("Memory layer is closed")
        
        if len(query_embedding) != self._db.dimension:
            raise ValueError(f"Query embedding dimension {len(query_embedding)} does not match database dimension {self._db.dimension}")
        
        c_embedding = ffi.new("float[]", query_embedding)
        c_results = ffi.new("GV_MemoryResult[]", k)
        
        count = lib.gv_memory_search(self._layer, c_embedding, k, c_results, int(distance))
        if count < 0:
            raise RuntimeError("Failed to search memories")
        
        results = []
        for i in range(count):
            result = _copy_memory_result(c_results + i)
            if result:
                results.append(result)
            lib.gv_memory_result_free(c_results + i)
        
        return results
    
    def get(self, memory_id: str) -> Optional[MemoryResult]:
        if self._closed:
            raise ValueError("Memory layer is closed")
        
        c_result = ffi.new("GV_MemoryResult *")
        ret = lib.gv_memory_get(self._layer, memory_id.encode(), c_result)
        
        if ret != 0:
            return None
        
        result = _copy_memory_result(c_result)
        lib.gv_memory_result_free(c_result)
        return result
    
    def delete(self, memory_id: str) -> bool:
        if self._closed:
            raise ValueError("Memory layer is closed")
        
        result = lib.gv_memory_delete(self._layer, memory_id.encode())
        return result == 0


# ============================================================================
# Context Graph Module
# ============================================================================

class EntityType(IntEnum):
    PERSON = 0
    ORGANIZATION = 1
    LOCATION = 2
    EVENT = 3
    OBJECT = 4
    CONCEPT = 5
    USER = 6


@dataclass
class GraphEntity:
    entity_id: Optional[str] = None
    name: str = ""
    entity_type: EntityType = EntityType.PERSON
    embedding: Optional[Sequence[float]] = None
    embedding_dim: int = 0
    created: int = 0
    updated: int = 0
    mentions: int = 0
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None

    def _to_c_entity(self) -> CData:
        """Convert to C entity structure.

        Returns:
            CFFI pointer to GV_GraphEntity structure.
        """
        c_entity = ffi.new("GV_GraphEntity *")
        if self.entity_id:
            c_entity.entity_id = ffi.new("char[]", self.entity_id.encode())
        if self.name:
            c_entity.name = ffi.new("char[]", self.name.encode())
        c_entity.entity_type = int(self.entity_type)
        if self.embedding:
            c_entity.embedding = ffi.new("float[]", self.embedding)
            c_entity.embedding_dim = len(self.embedding)
        c_entity.created = self.created
        c_entity.updated = self.updated
        c_entity.mentions = self.mentions
        if self.user_id:
            c_entity.user_id = ffi.new("char[]", self.user_id.encode())
        if self.agent_id:
            c_entity.agent_id = ffi.new("char[]", self.agent_id.encode())
        if self.run_id:
            c_entity.run_id = ffi.new("char[]", self.run_id.encode())
        return c_entity


@dataclass
class GraphRelationship:
    relationship_id: Optional[str] = None
    source_entity_id: str = ""
    destination_entity_id: str = ""
    relationship_type: str = ""
    created: int = 0
    updated: int = 0
    mentions: int = 0

    def _to_c_relationship(self) -> CData:
        """Convert to C relationship structure.

        Returns:
            CFFI pointer to GV_GraphRelationship structure.
        """
        c_rel = ffi.new("GV_GraphRelationship *")
        if self.relationship_id:
            c_rel.relationship_id = ffi.new("char[]", self.relationship_id.encode())
        else:
            c_rel.relationship_id = ffi.NULL
        if self.source_entity_id:
            c_rel.source_entity_id = ffi.new("char[]", self.source_entity_id.encode())
        else:
            c_rel.source_entity_id = ffi.NULL
        if self.destination_entity_id:
            c_rel.destination_entity_id = ffi.new("char[]", self.destination_entity_id.encode())
        else:
            c_rel.destination_entity_id = ffi.NULL
        if self.relationship_type:
            c_rel.relationship_type = ffi.new("char[]", self.relationship_type.encode())
        else:
            c_rel.relationship_type = ffi.NULL
        c_rel.created = self.created
        c_rel.updated = self.updated
        c_rel.mentions = self.mentions
        return c_rel


@dataclass
class GraphQueryResult:
    source_name: str = ""
    relationship_type: str = ""
    destination_name: str = ""
    similarity: float = 0.0


@dataclass
class ContextGraphConfig:
    llm: Optional[LLM] = None  # LLM instance
    similarity_threshold: float = 0.7
    enable_entity_extraction: bool = True
    enable_relationship_extraction: bool = True
    max_traversal_depth: int = 3
    max_results: int = 100
    embedding_callback: Optional[Callable[[str], Sequence[float]]] = None
    embedding_dimension: int = 0
    embedding_service: Optional[EmbeddingService] = None  # EmbeddingService instance

    def _to_c_config(self) -> CData:
        """Convert to C configuration structure.

        Returns:
            CFFI pointer to GV_ContextGraphConfig structure.
        """
        c_config = ffi.new("GV_ContextGraphConfig *")
        if self.llm is not None:
            # Access internal C pointer from LLM object
            if hasattr(self.llm, '_llm'):
                c_config.llm = self.llm._llm
            else:
                c_config.llm = ffi.NULL
        else:
            c_config.llm = ffi.NULL

        # Set embedding service if provided
        if self.embedding_service is not None:
            if hasattr(self.embedding_service, '_service'):
                c_config.embedding_service = self.embedding_service._service
            else:
                c_config.embedding_service = ffi.NULL
        else:
            c_config.embedding_service = ffi.NULL

        c_config.similarity_threshold = self.similarity_threshold
        c_config.enable_entity_extraction = 1 if self.enable_entity_extraction else 0
        c_config.enable_relationship_extraction = 1 if self.enable_relationship_extraction else 0
        c_config.max_traversal_depth = self.max_traversal_depth
        c_config.max_results = self.max_results

        # Embedding callback setup
        # Note: C callbacks from Python are complex, so we'll handle this differently
        # For now, embeddings should be provided when adding entities
        c_config.embedding_callback = ffi.NULL
        c_config.embedding_user_data = ffi.NULL
        c_config.embedding_dimension = self.embedding_dimension

        return c_config


class ContextGraph:
    """Context graph for entity and relationship extraction and querying."""

    _graph: CData
    _config: ContextGraphConfig
    _closed: bool

    def __init__(self, config: Optional[ContextGraphConfig] = None) -> None:
        if config is None:
            config = ContextGraphConfig()

        c_config = config._to_c_config()
        self._graph = lib.gv_context_graph_create(c_config)
        if self._graph == ffi.NULL:
            raise RuntimeError("Failed to create context graph")
        self._config = config
        self._closed = False

    def __enter__(self) -> ContextGraph:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the context graph and release resources."""
        if not self._closed and self._graph != ffi.NULL:
            lib.gv_context_graph_destroy(self._graph)
            self._graph = ffi.NULL
            self._closed = True
    
    def extract(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        generate_embeddings: Optional[Callable[[str], Sequence[float]]] = None,
        use_batch_embeddings: bool = True,
    ) -> tuple[list[GraphEntity], list[GraphRelationship]]:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from.
            user_id: Optional user ID filter.
            agent_id: Optional agent ID filter.
            run_id: Optional run ID filter.
            generate_embeddings: Optional callback to generate embeddings for extracted entities.
            use_batch_embeddings: If True and embedding_service is set, use batch generation.

        Returns:
            Tuple of (entities, relationships) extracted from the text.
        """
        entities_ptr = ffi.new("GV_GraphEntity **")
        entity_count_ptr = ffi.new("size_t *")
        relationships_ptr = ffi.new("GV_GraphRelationship **")
        relationship_count_ptr = ffi.new("size_t *")
        
        user_id_bytes = user_id.encode() if user_id else ffi.NULL
        agent_id_bytes = agent_id.encode() if agent_id else ffi.NULL
        run_id_bytes = run_id.encode() if run_id else ffi.NULL
        
        result = lib.gv_context_graph_extract(
            self._graph, text.encode(), 
            user_id_bytes, agent_id_bytes, run_id_bytes,
            entities_ptr, entity_count_ptr,
            relationships_ptr, relationship_count_ptr
        )
        
        if result != 0:
            raise RuntimeError("Failed to extract entities and relationships")
        
        entities = []
        if entities_ptr[0] != ffi.NULL:
            # Collect all entity names for batch embedding generation
            entity_names = []
            entity_indices = []
            for i in range(entity_count_ptr[0]):
                c_entity = entities_ptr[0] + i
                name = ffi.string(c_entity.name).decode("utf-8") if c_entity.name else ""
                if name:
                    entity_names.append(name)
                    entity_indices.append(i)
            
            # Generate embeddings in batch if service is available
            embeddings_map: dict[str, Sequence[float]] = {}
            if use_batch_embeddings and self._config.embedding_service and entity_names:
                try:
                    batch_embeddings = self._config.embedding_service.generate_batch(entity_names)
                    embeddings_map = {name: emb for name, emb in zip(entity_names, batch_embeddings) if emb is not None}
                except Exception:
                    pass

            # Process entities
            for i in range(entity_count_ptr[0]):
                c_entity = entities_ptr[0] + i
                name = ffi.string(c_entity.name).decode("utf-8") if c_entity.name else ""

                # Get embedding from C entity, batch service, or callback
                embedding: Optional[Sequence[float]] = None
                embedding_dim = 0
                if c_entity.embedding != ffi.NULL and c_entity.embedding_dim > 0:
                    embedding = [c_entity.embedding[j] for j in range(c_entity.embedding_dim)]
                    embedding_dim = c_entity.embedding_dim
                elif name in embeddings_map:
                    embedding = embeddings_map[name]
                    embedding_dim = len(embedding)
                elif generate_embeddings and name:
                    try:
                        emb_result = generate_embeddings(name)
                        if emb_result:
                            embedding = emb_result
                            embedding_dim = len(embedding)
                    except Exception:
                        pass
                elif self._config.embedding_service and name:
                    try:
                        service_emb = self._config.embedding_service.generate(name)
                        if service_emb:
                            embedding = service_emb
                            embedding_dim = len(embedding)
                    except Exception:
                        pass
                
                entity = GraphEntity(
                    entity_id=ffi.string(c_entity.entity_id).decode("utf-8") if c_entity.entity_id else None,
                    name=name,
                    entity_type=EntityType(c_entity.entity_type),
                    embedding=embedding,
                    embedding_dim=embedding_dim,
                    created=c_entity.created,
                    updated=c_entity.updated,
                    mentions=c_entity.mentions,
                    user_id=ffi.string(c_entity.user_id).decode("utf-8") if c_entity.user_id else None,
                    agent_id=ffi.string(c_entity.agent_id).decode("utf-8") if c_entity.agent_id else None,
                    run_id=ffi.string(c_entity.run_id).decode("utf-8") if c_entity.run_id else None,
                )
                entities.append(entity)
                lib.gv_graph_entity_free(c_entity)
            lib.free(entities_ptr[0])
        
        relationships = []
        if relationships_ptr[0] != ffi.NULL:
            for i in range(relationship_count_ptr[0]):
                c_rel = relationships_ptr[0] + i
                rel = GraphRelationship(
                    relationship_id=ffi.string(c_rel.relationship_id).decode("utf-8") if c_rel.relationship_id else None,
                    source_entity_id=ffi.string(c_rel.source_entity_id).decode("utf-8") if c_rel.source_entity_id else "",
                    destination_entity_id=ffi.string(c_rel.destination_entity_id).decode("utf-8") if c_rel.destination_entity_id else "",
                    relationship_type=ffi.string(c_rel.relationship_type).decode("utf-8") if c_rel.relationship_type else "",
                    created=c_rel.created,
                    updated=c_rel.updated,
                    mentions=c_rel.mentions,
                )
                relationships.append(rel)
                lib.gv_graph_relationship_free(c_rel)
            lib.free(relationships_ptr[0])
        
        return entities, relationships
    
    def add_entities(
        self,
        entities: Sequence[GraphEntity],
        generate_embeddings: Optional[Callable[[str], Sequence[float]]] = None,
        use_batch_embeddings: bool = True,
    ) -> None:
        """Add entities to the graph.

        Args:
            entities: List of entities to add.
            generate_embeddings: Optional callback to generate embeddings for entities without them.
            use_batch_embeddings: If True and embedding_service is set, use batch generation.
        """
        if not entities:
            return
        
        # Collect entities that need embeddings
        entities_needing_embeddings = [
            (i, entity) for i, entity in enumerate(entities)
            if entity.embedding is None and entity.name
        ]
        
        # Generate embeddings in batch if service is available
        if use_batch_embeddings and self._config.embedding_service and entities_needing_embeddings:
            try:
                entity_names = [entity.name for _, entity in entities_needing_embeddings]
                batch_embeddings = self._config.embedding_service.generate_batch(entity_names)
                for (i, entity), embedding in zip(entities_needing_embeddings, batch_embeddings):
                    if embedding is not None:
                        entity.embedding = embedding
                        entity.embedding_dim = len(embedding)
            except Exception:
                pass
        
        # Generate embeddings individually if callback provided or service available
        if generate_embeddings or (self._config.embedding_service and not use_batch_embeddings):
            for i, entity in entities_needing_embeddings:
                if entity.embedding is None and entity.name:
                    try:
                        if generate_embeddings:
                            entity.embedding = generate_embeddings(entity.name)
                        elif self._config.embedding_service:
                            entity.embedding = self._config.embedding_service.generate(entity.name)
                        if entity.embedding:
                            entity.embedding_dim = len(entity.embedding)
                    except Exception:
                        pass
        
        c_entities = ffi.new("GV_GraphEntity[]", len(entities))
        for i, entity in enumerate(entities):
            c_entity = entity._to_c_entity()
            c_entities[i] = c_entity[0]
        
        result = lib.gv_context_graph_add_entities(self._graph, c_entities, len(entities))
        if result != 0:
            raise RuntimeError("Failed to add entities")
    
    def add_relationships(self, relationships: Sequence[GraphRelationship]) -> None:
        """Add relationships to the graph.

        Args:
            relationships: List of relationships to add.
        """
        if not relationships:
            return
        
        c_rels = ffi.new("GV_GraphRelationship[]", len(relationships))
        for i, rel in enumerate(relationships):
            c_rel = rel._to_c_relationship()
            c_rels[i] = c_rel[0]
        
        result = lib.gv_context_graph_add_relationships(self._graph, c_rels, len(relationships))
        if result != 0:
            raise RuntimeError("Failed to add relationships")
    
    def search(
        self,
        query_embedding: Sequence[float],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        max_results: int = 10,
    ) -> list[GraphQueryResult]:
        """Search for related entities in the graph.

        Args:
            query_embedding: Query embedding vector.
            user_id: Optional user ID filter.
            agent_id: Optional agent ID filter.
            run_id: Optional run ID filter.
            max_results: Maximum number of results to return.

        Returns:
            List of graph query results ordered by similarity.
        """
        c_embedding = ffi.new("float[]", query_embedding)
        c_results = ffi.new("GV_GraphQueryResult[]", max_results)
        
        user_id_bytes = user_id.encode() if user_id else ffi.NULL
        agent_id_bytes = agent_id.encode() if agent_id else ffi.NULL
        run_id_bytes = run_id.encode() if run_id else ffi.NULL
        
        count = lib.gv_context_graph_search(
            self._graph, c_embedding, len(query_embedding),
            user_id_bytes, agent_id_bytes, run_id_bytes,
            c_results, max_results
        )
        
        if count < 0:
            raise RuntimeError("Failed to search graph")
        
        results = []
        for i in range(count):
            result = GraphQueryResult(
                source_name=ffi.string(c_results[i].source_name).decode("utf-8") if c_results[i].source_name else "",
                relationship_type=ffi.string(c_results[i].relationship_type).decode("utf-8") if c_results[i].relationship_type else "",
                destination_name=ffi.string(c_results[i].destination_name).decode("utf-8") if c_results[i].destination_name else "",
                similarity=c_results[i].similarity,
            )
            results.append(result)
            lib.gv_graph_query_result_free(c_results + i)
        
        return results
    
    def get_related(self, entity_id: str, max_depth: int = 3, max_results: int = 10) -> list[GraphQueryResult]:
        """Get related entities for a given entity.

        Args:
            entity_id: ID of the entity to find relationships for.
            max_depth: Maximum graph traversal depth.
            max_results: Maximum number of results to return.

        Returns:
            List of graph query results representing related entities.
        """
        c_results = ffi.new("GV_GraphQueryResult[]", max_results)
        
        count = lib.gv_context_graph_get_related(
            self._graph, entity_id.encode(), max_depth, c_results, max_results
        )
        
        if count < 0:
            raise RuntimeError("Failed to get related entities")
        
        results = []
        for i in range(count):
            result = GraphQueryResult(
                source_name=ffi.string(c_results[i].source_name).decode("utf-8") if c_results[i].source_name else "",
                relationship_type=ffi.string(c_results[i].relationship_type).decode("utf-8") if c_results[i].relationship_type else "",
                destination_name=ffi.string(c_results[i].destination_name).decode("utf-8") if c_results[i].destination_name else "",
                similarity=c_results[i].similarity,
            )
            results.append(result)
            lib.gv_graph_query_result_free(c_results + i)
        
        return results

