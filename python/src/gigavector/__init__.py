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
    ScalarQuantConfig,
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
    "ScalarQuantConfig",
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
]

__version__ = "0.6.0"
