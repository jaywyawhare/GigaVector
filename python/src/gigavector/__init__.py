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
    serve_with_dashboard,
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
    # Multi-vector documents
    DocAggregation,
    DocSearchResult,
    MultiVecConfig,
    MultiVecIndex,
    # Snapshots
    SnapshotInfo,
    SnapshotManager,
    # MVCC Transactions
    MVCCManager,
    Transaction,
    TxnStatus,
    # Query Optimizer
    CollectionStats,
    PlanStrategy,
    QueryOptimizer,
    QueryPlan,
    # Payload Indexing
    FieldType,
    PayloadIndex,
    PayloadOp,
    # Vector Deduplication
    DedupConfig,
    DedupIndex,
    DedupResult,
    # Index Migration
    Migration,
    MigrationInfo,
    MigrationStatus,
    # Collection Versioning
    VersionInfo,
    VersionManager,
    # Read Replica Load Balancing
    ReadPolicy,
    # Bloom Filter
    BloomFilter,
    # Query Tracing
    QueryTrace,
    TraceSpan,
    # Client-Side Caching
    Cache,
    CacheConfig,
    CachePolicy,
    CacheStats,
    # Schema Evolution
    Schema,
    SchemaDiff,
    SchemaField,
    SchemaFieldType,
    # Codebook Sharing
    Codebook,
    # Point ID Mapping
    PointIDMap,
    # TLS/HTTPS
    TLSConfig,
    TLSContext,
    TLSVersion,
    # Score Threshold
    ThresholdResult,
    search_with_threshold,
    # Named Vectors
    NamedVectorStore,
    VectorFieldConfig,
    # Filter Operations
    delete_by_filter,
    update_metadata_by_filter,
    count_by_filter,
    # gRPC
    GrpcConfig,
    GrpcServer,
    GrpcStats,
    # Auto-Embedding
    AutoEmbedConfig,
    AutoEmbedder,
    AutoEmbedProvider,
    AutoEmbedStats,
    # DiskANN
    DiskANNConfig,
    DiskANNIndex,
    DiskANNStats,
    # Grouped Search
    GroupedSearch,
    GroupHit,
    GroupSearchConfig,
    SearchGroup,
    # Geo-Spatial
    GeoIndex,
    GeoPoint,
    GeoResult,
    # Late Interaction
    LateInteractionConfig,
    LateInteractionIndex,
    LateInteractionResult,
    # Recommendation
    RecommendConfig,
    RecommendResult,
    Recommender,
    # Aliases
    AliasInfo,
    AliasManager,
    # Vacuum
    VacuumConfig,
    VacuumManager,
    VacuumState,
    VacuumStats,
    # Consistency
    ConsistencyLevel,
    ConsistencyManager,
    # Quotas
    QuotaConfig,
    QuotaManager,
    QuotaResult,
    QuotaUsage,
    # Compression
    CompressionConfig,
    CompressionStats,
    CompressionType,
    Compressor,
    # Webhooks
    EventType,
    WebhookConfig,
    WebhookManager,
    WebhookStats,
    # RBAC
    Permission,
    RBACManager,
    # MMR Reranking
    MMRConfig,
    MMRResult,
    mmr_rerank,
    # Custom Ranking
    RankExpr,
    RankSignal,
    RankedResult,
    # Advanced Quantization
    QuantCodebook,
    QuantConfig,
    QuantMode,
    QuantType,
    # Full-Text Search (Enhanced)
    FTConfig,
    FTIndex,
    FTLanguage,
    FTResult,
    ft_stem,
    # Optimized HNSW
    HNSWInlineConfig,
    HNSWInlineIndex,
    HNSWRebuildConfig,
    HNSWRebuildStats,
    # ONNX Model Serving
    ONNXConfig,
    ONNXModel,
    # Agentic Interfaces
    Agent,
    AgentConfig,
    AgentResult,
    AgentType,
    # MUVERA Encoder
    MuveraConfig,
    MuveraEncoder,
    # Enterprise SSO
    SSOConfig,
    SSOManager,
    SSOProvider,
    SSOToken,
    # Tiered Multitenancy
    TenantInfo,
    TenantTier,
    TierThresholds,
    TieredManager,
    TieredTenantConfig,
    # Integrated Inference
    InferenceConfig,
    InferenceEngine,
    InferenceResult,
    # JSON Path Indexing
    JSONPathConfig,
    JSONPathIndex,
    JSONPathType,
    # CDC Stream
    CDCConfig,
    CDCCursor,
    CDCEvent,
    CDCEventType,
    CDCStream,
    # Embedded / Edge Mode
    EmbeddedConfig,
    EmbeddedDB,
    EmbeddedIndexType,
    EmbeddedResult,
    # Conditional Updates
    ConditionType,
    ConditionalResult,
    Condition,
    CondManager,
    # Time-Travel
    TimeTravelConfig,
    TimeTravelManager,
    TTVersionEntry,
    # Multimodal Storage
    MediaConfig,
    MediaEntry,
    MediaStore,
    MediaType,
    # SQL Interface
    SQLEngine,
    SQLResult,
    # Phased Ranking Pipeline
    PhaseType,
    PhasedResult,
    Pipeline,
    PipelineStats,
    # Learned Sparse
    LearnedSparseConfig,
    LearnedSparseEntry,
    LearnedSparseIndex,
    LearnedSparseResult,
    LearnedSparseStats,
    # Graph Database
    GraphDBConfig,
    GraphPath,
    GraphDB,
    # Knowledge Graph
    KGConfig,
    KGSearchResult,
    KGTriple,
    KGLinkPrediction,
    KGSubgraph,
    KGStats,
    KnowledgeGraph,
)
from .dashboard.server import DashboardServer

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
    "serve_with_dashboard",
    # Dashboard
    "DashboardServer",
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
    # Multi-vector documents
    "DocAggregation",
    "DocSearchResult",
    "MultiVecConfig",
    "MultiVecIndex",
    # Snapshots
    "SnapshotInfo",
    "SnapshotManager",
    # MVCC Transactions
    "MVCCManager",
    "Transaction",
    "TxnStatus",
    # Query Optimizer
    "CollectionStats",
    "PlanStrategy",
    "QueryOptimizer",
    "QueryPlan",
    # Payload Indexing
    "FieldType",
    "PayloadIndex",
    "PayloadOp",
    # Vector Deduplication
    "DedupConfig",
    "DedupIndex",
    "DedupResult",
    # Index Migration
    "Migration",
    "MigrationInfo",
    "MigrationStatus",
    # Collection Versioning
    "VersionInfo",
    "VersionManager",
    # Read Replica Load Balancing
    "ReadPolicy",
    # Bloom Filter
    "BloomFilter",
    # Query Tracing
    "QueryTrace",
    "TraceSpan",
    # Client-Side Caching
    "Cache",
    "CacheConfig",
    "CachePolicy",
    "CacheStats",
    # Schema Evolution
    "Schema",
    "SchemaDiff",
    "SchemaField",
    "SchemaFieldType",
    # Codebook Sharing
    "Codebook",
    # Point ID Mapping
    "PointIDMap",
    # TLS/HTTPS
    "TLSConfig",
    "TLSContext",
    "TLSVersion",
    # Score Threshold
    "ThresholdResult",
    "search_with_threshold",
    # Named Vectors
    "NamedVectorStore",
    "VectorFieldConfig",
    # Filter Operations
    "delete_by_filter",
    "update_metadata_by_filter",
    "count_by_filter",
    # gRPC
    "GrpcConfig",
    "GrpcServer",
    "GrpcStats",
    # Auto-Embedding
    "AutoEmbedConfig",
    "AutoEmbedder",
    "AutoEmbedProvider",
    "AutoEmbedStats",
    # DiskANN
    "DiskANNConfig",
    "DiskANNIndex",
    "DiskANNStats",
    # Grouped Search
    "GroupedSearch",
    "GroupHit",
    "GroupSearchConfig",
    "SearchGroup",
    # Geo-Spatial
    "GeoIndex",
    "GeoPoint",
    "GeoResult",
    # Late Interaction
    "LateInteractionConfig",
    "LateInteractionIndex",
    "LateInteractionResult",
    # Recommendation
    "RecommendConfig",
    "RecommendResult",
    "Recommender",
    # Aliases
    "AliasInfo",
    "AliasManager",
    # Vacuum
    "VacuumConfig",
    "VacuumManager",
    "VacuumState",
    "VacuumStats",
    # Consistency
    "ConsistencyLevel",
    "ConsistencyManager",
    # Quotas
    "QuotaConfig",
    "QuotaManager",
    "QuotaResult",
    "QuotaUsage",
    # Compression
    "CompressionConfig",
    "CompressionStats",
    "CompressionType",
    "Compressor",
    # Webhooks
    "EventType",
    "WebhookConfig",
    "WebhookManager",
    "WebhookStats",
    # RBAC
    "Permission",
    "RBACManager",
    # MMR Reranking
    "MMRConfig",
    "MMRResult",
    "mmr_rerank",
    # Custom Ranking
    "RankExpr",
    "RankSignal",
    "RankedResult",
    # Advanced Quantization
    "QuantCodebook",
    "QuantConfig",
    "QuantMode",
    "QuantType",
    # Full-Text Search (Enhanced)
    "FTConfig",
    "FTIndex",
    "FTLanguage",
    "FTResult",
    "ft_stem",
    # Optimized HNSW
    "HNSWInlineConfig",
    "HNSWInlineIndex",
    "HNSWRebuildConfig",
    "HNSWRebuildStats",
    # ONNX Model Serving
    "ONNXConfig",
    "ONNXModel",
    # Agentic Interfaces
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentType",
    # MUVERA Encoder
    "MuveraConfig",
    "MuveraEncoder",
    # Enterprise SSO
    "SSOConfig",
    "SSOManager",
    "SSOProvider",
    "SSOToken",
    # Tiered Multitenancy
    "TenantInfo",
    "TenantTier",
    "TierThresholds",
    "TieredManager",
    "TieredTenantConfig",
    # Integrated Inference
    "InferenceConfig",
    "InferenceEngine",
    "InferenceResult",
    # JSON Path Indexing
    "JSONPathConfig",
    "JSONPathIndex",
    "JSONPathType",
    # CDC Stream
    "CDCConfig",
    "CDCCursor",
    "CDCEvent",
    "CDCEventType",
    "CDCStream",
    # Embedded / Edge Mode
    "EmbeddedConfig",
    "EmbeddedDB",
    "EmbeddedIndexType",
    "EmbeddedResult",
    # Conditional Updates
    "ConditionType",
    "ConditionalResult",
    "Condition",
    "CondManager",
    # Time-Travel
    "TimeTravelConfig",
    "TimeTravelManager",
    "TTVersionEntry",
    # Multimodal Storage
    "MediaConfig",
    "MediaEntry",
    "MediaStore",
    "MediaType",
    # SQL Interface
    "SQLEngine",
    "SQLResult",
    # Phased Ranking Pipeline
    "PhaseType",
    "PhasedResult",
    "Pipeline",
    "PipelineStats",
    # Learned Sparse
    "LearnedSparseConfig",
    "LearnedSparseEntry",
    "LearnedSparseIndex",
    "LearnedSparseResult",
    "LearnedSparseStats",
    # Graph Database
    "GraphDBConfig",
    "GraphPath",
    "GraphDB",
    # Knowledge Graph
    "KGConfig",
    "KGSearchResult",
    "KGTriple",
    "KGLinkPrediction",
    "KGSubgraph",
    "KGStats",
    "KnowledgeGraph",
]

__version__ = "0.8.0"
