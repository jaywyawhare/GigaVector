from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

from ._ffi import ffi, lib


class IndexType(IntEnum):
    KDTREE = 0
    HNSW = 1
    IVFPQ = 2


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


def _metadata_to_dict(meta_ptr) -> dict[str, str]:
    if meta_ptr == ffi.NULL:
        return {}
    out: dict[str, str] = {}
    cur = meta_ptr
    while cur != ffi.NULL:
        key = ffi.string(cur.key).decode("utf-8")
        value = ffi.string(cur.value).decode("utf-8")
        out[key] = value
        cur = cur.next
    return out


def _copy_vector(vec_ptr) -> Vector:
    dim = int(vec_ptr.dimension)
    data = list(ffi.unpack(vec_ptr.data, dim))
    metadata = _metadata_to_dict(vec_ptr.metadata)
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

    def close(self):
        if self._closed:
            return
        lib.gv_db_close(self._db)
        self._closed = True

    def save(self, path: str | None = None):
        """Persist the database to a binary snapshot file."""
        c_path = path.encode("utf-8") if path is not None else ffi.NULL
        rc = lib.gv_db_save(self._db, c_path)
        if rc != 0:
            raise RuntimeError("gv_db_save failed")

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
        return [SearchHit(distance=float(results[i].distance), vector=_copy_vector(results[i].vector)) for i in range(n)]

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
        return [SearchHit(distance=float(results[i].distance), vector=_copy_vector(results[i].vector)) for i in range(n)]

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

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Avoid raising during interpreter shutdown
            pass

