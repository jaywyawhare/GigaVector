"""GigaVector: High-performance vector database with LLM integration.

GigaVector is a vector database library designed for efficient storage and
retrieval of high-dimensional vectors with support for multiple index types,
distance metrics, and advanced features like LLM-based memory extraction.

Core Components:
    Database: Main vector storage and search interface.
    LLM: Language model integration for text processing.
    EmbeddingService: Text-to-vector embedding generation.
    ContextGraph: Entity and relationship extraction.
    MemoryLayer: Semantic memory storage and retrieval.

Example:
    >>> from gigavector import Database, IndexType, DistanceType
    >>> db = Database.open(None, dimension=128, index=IndexType.HNSW)
    >>> db.add_vector([0.1] * 128, metadata={"category": "example"})
    >>> results = db.search([0.1] * 128, k=10, distance=DistanceType.COSINE)
    >>> db.close()
"""
from ._core import (
    # Database core types
    Database,
    DBStats,
    DistanceType,
    IndexType,
    SearchHit,
    Vector,
    # Configuration types
    HNSWConfig,
    IVFPQConfig,
    IVFFlatConfig,
    PQConfig,
    LSHConfig,
    ScalarQuantConfig,
    SearchParams,
    ScrollEntry,
    # LLM types
    LLM,
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    # Embedding service types
    EmbeddingCache,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingService,
    # Memory layer types
    ConsolidationStrategy,
    MemoryLayer,
    MemoryLayerConfig,
    MemoryMetadata,
    MemoryResult,
    MemoryType,
    # Context graph types
    ContextGraph,
    ContextGraphConfig,
    EntityType,
    GraphEntity,
    GraphQueryResult,
    GraphRelationship,
    # GPU acceleration
    GPUConfig,
    GPUContext,
    GPUDeviceInfo,
    GPUDistanceMetric,
    GPUIndex,
    GPUSearchParams,
    GPUStats,
    gpu_available,
    gpu_device_count,
    gpu_get_device_info,
    # HTTP Server
    Server,
    ServerConfig,
    ServerError,
    ServerStats,
    # Backup & Restore
    BackupCompression,
    BackupHeader,
    BackupOptions,
    BackupResult,
    RestoreOptions,
    backup_create,
    backup_read_header,
    backup_restore,
    backup_restore_to_db,
    backup_verify,
    # Shard management
    ShardConfig,
    ShardInfo,
    ShardManager,
    ShardState,
    ShardStrategy,
    # Replication
    ReplicaInfo,
    ReplicationConfig,
    ReplicationManager,
    ReplicationRole,
    ReplicationState,
    ReplicationStats,
    # Cluster management
    Cluster,
    ClusterConfig,
    ClusterStats,
    NodeInfo,
    NodeRole,
    NodeState,
    # Namespace / Multi-tenancy
    Namespace,
    NamespaceConfig,
    NamespaceInfo,
    NamespaceManager,
    NSIndexType,
    # TTL (Time-to-Live)
    TTLConfig,
    TTLManager,
    TTLStats,
    # BM25 Full-text Search
    BM25Config,
    BM25Index,
    BM25Result,
    BM25Stats,
    # Hybrid Search
    FusionType,
    HybridConfig,
    HybridResult,
    HybridSearcher,
    HybridStats,
    # Authentication
    APIKey,
    AuthConfig,
    AuthManager,
    AuthResult,
    AuthType,
    Identity,
    JWTConfig,
)

__all__ = [
    # Database core
    "Database",
    "DBStats",
    "DistanceType",
    "IndexType",
    "SearchHit",
    "Vector",
    # Configuration
    "HNSWConfig",
    "IVFPQConfig",
    "IVFFlatConfig",
    "PQConfig",
    "LSHConfig",
    "ScalarQuantConfig",
    "SearchParams",
    "ScrollEntry",
    # LLM
    "LLM",
    "LLMConfig",
    "LLMError",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    # Embedding service
    "EmbeddingCache",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingService",
    # Memory layer
    "ConsolidationStrategy",
    "MemoryLayer",
    "MemoryLayerConfig",
    "MemoryMetadata",
    "MemoryResult",
    "MemoryType",
    # Context graph
    "ContextGraph",
    "ContextGraphConfig",
    "EntityType",
    "GraphEntity",
    "GraphQueryResult",
    "GraphRelationship",
    # GPU acceleration
    "GPUConfig",
    "GPUContext",
    "GPUDeviceInfo",
    "GPUDistanceMetric",
    "GPUIndex",
    "GPUSearchParams",
    "GPUStats",
    "gpu_available",
    "gpu_device_count",
    "gpu_get_device_info",
    # HTTP Server
    "Server",
    "ServerConfig",
    "ServerError",
    "ServerStats",
    # Backup & Restore
    "BackupCompression",
    "BackupHeader",
    "BackupOptions",
    "BackupResult",
    "RestoreOptions",
    "backup_create",
    "backup_read_header",
    "backup_restore",
    "backup_restore_to_db",
    "backup_verify",
    # Shard management
    "ShardConfig",
    "ShardInfo",
    "ShardManager",
    "ShardState",
    "ShardStrategy",
    # Replication
    "ReplicaInfo",
    "ReplicationConfig",
    "ReplicationManager",
    "ReplicationRole",
    "ReplicationState",
    "ReplicationStats",
    # Cluster management
    "Cluster",
    "ClusterConfig",
    "ClusterStats",
    "NodeInfo",
    "NodeRole",
    "NodeState",
    # Namespace / Multi-tenancy
    "Namespace",
    "NamespaceConfig",
    "NamespaceInfo",
    "NamespaceManager",
    "NSIndexType",
    # TTL (Time-to-Live)
    "TTLConfig",
    "TTLManager",
    "TTLStats",
    # BM25 Full-text Search
    "BM25Config",
    "BM25Index",
    "BM25Result",
    "BM25Stats",
    # Hybrid Search
    "FusionType",
    "HybridConfig",
    "HybridResult",
    "HybridSearcher",
    "HybridStats",
    # Authentication
    "APIKey",
    "AuthConfig",
    "AuthManager",
    "AuthResult",
    "AuthType",
    "Identity",
    "JWTConfig",
]

__version__ = "0.8.0"
