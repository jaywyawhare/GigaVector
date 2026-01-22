# GigaVector Architecture Documentation

> A comprehensive high-performance vector database library with semantic memory capabilities

## Table of Contents

1. [Overview](#overview)
2. [Vector Database Algorithms](#vector-database-algorithms)
   - [HNSW (Hierarchical Navigable Small Worlds)](#hnsw-hierarchical-navigable-small-worlds)
   - [IVF-PQ (Inverted File with Product Quantization)](#ivf-pq-inverted-file-with-product-quantization)
   - [K-D Tree](#k-d-tree)
   - [Sparse Index](#sparse-index)
3. [Distance Metrics](#distance-metrics)
4. [Memory Layer](#memory-layer)
   - [Architecture](#memory-layer-architecture)
   - [Memory Links and Relationships](#memory-links-and-relationships)
   - [Memory Extraction](#memory-extraction)
   - [Memory Consolidation](#memory-consolidation)
5. [Importance Scoring System](#importance-scoring-system)
6. [LLM Integration](#llm-integration)
7. [Embedding System](#embedding-system)
8. [Context Graph](#context-graph)
9. [Core Data Structures](#core-data-structures)
10. [Storage and Persistence](#storage-and-persistence)
11. [Configuration and Optimization](#configuration-and-optimization)
12. [Integration Patterns](#integration-patterns)

---

## Overview

GigaVector is a high-performance vector database library written in C, designed for:

- **Approximate Nearest Neighbor (ANN) Search**: Multiple algorithms (HNSW, IVF-PQ, K-D Tree)
- **Semantic Memory**: Store, retrieve, and consolidate memories with importance scoring
- **LLM Integration**: Extract memories from conversations using large language models
- **Knowledge Graphs**: Build entity-relationship graphs for context-aware retrieval
- **Production Ready**: WAL, compaction, monitoring, and resource limits

### Key Design Principles

1. **Zero Dependencies**: Core library has no external dependencies (SQLite/cURL optional for LLM)
2. **Memory Efficient**: Structure-of-Arrays storage, quantization options
3. **Language Agnostic**: All scoring/analysis uses statistical methods, not keyword lists
4. **Modular**: Use only the components you need

---

## Vector Database Algorithms

### HNSW (Hierarchical Navigable Small Worlds)

**File**: `include/gigavector/gv_hnsw.h`, `src/gv_hnsw.c`

HNSW is a graph-based approximate nearest neighbor algorithm that builds a hierarchical structure of navigable small world graphs.

#### How It Works

```
Level 2:    [A] -------- [B]
             |
Level 1:    [A] -- [C] -- [B] -- [D]
             |      |      |      |
Level 0:    [A]-[E]-[C]-[F]-[B]-[G]-[D]-[H]
```

1. **Hierarchy**: Vectors are inserted at multiple levels with decreasing probability
2. **Navigation**: Search starts at top level, greedily navigates to nearest neighbor
3. **Refinement**: Descends through levels, refining results at each layer

#### Configuration

```c
typedef struct {
    size_t M;                // Connections per node (default: 16)
    size_t efConstruction;   // Candidate list during build (default: 200)
    size_t efSearch;         // Candidate list during search (default: 50)
    size_t maxLevel;         // Max hierarchy levels (auto-calculated)
    int use_binary_quant;    // Binary quantization for fast candidate selection
    size_t quant_rerank;     // Candidates to rerank with exact distance
    int use_acorn;           // ACORN-style exploration for filtered search
    size_t acorn_hops;       // Exploration depth (1-2 hops)
} GV_HNSWConfig;
```

#### Trade-offs

| Parameter | Higher Value | Lower Value |
|-----------|--------------|-------------|
| `M` | Better recall, more memory | Faster, less memory |
| `efConstruction` | Better graph quality, slower build | Faster build |
| `efSearch` | Better recall, slower search | Faster search |

#### When to Use

- **Best for**: Medium to large datasets (10K - 10M vectors)
- **Memory**: ~1.2x raw data size
- **Latency**: Sub-millisecond for most queries
- **Recall**: 95%+ with default settings

---

### IVF-PQ (Inverted File with Product Quantization)

**File**: `include/gigavector/gv_ivfpq.h`, `src/gv_ivfpq.c`

IVF-PQ combines coarse quantization with product quantization for memory-efficient search on very large datasets.

#### How It Works

```
Step 1: Coarse Quantization (IVF)
┌─────────────────────────────────────┐
│  Vectors partitioned into clusters  │
│  [C1] [C2] [C3] ... [Cn]           │
│   │    │    │        │              │
│  [v1] [v4] [v7]    [vN]            │
│  [v2] [v5] [v8]                    │
│  [v3] [v6]                         │
└─────────────────────────────────────┘

Step 2: Product Quantization (PQ)
┌─────────────────────────────────────┐
│  Vector: [d1 d2 d3 d4 d5 d6 d7 d8] │
│           ↓     ↓     ↓     ↓       │
│  Sub-vectors: [s1] [s2] [s3] [s4]  │
│           ↓     ↓     ↓     ↓       │
│  Codes:    3    7    2    9        │
│  (8 bits each = 4 bytes total)     │
└─────────────────────────────────────┘
```

#### Configuration

```c
typedef struct {
    size_t nlist;              // Coarse centroids (default: sqrt(n))
    size_t m;                  // Subquantizers (must divide dimension)
    size_t nbits;              // Bits per code (typically 8)
    size_t nprobe;             // Lists to search (default: 1)
    size_t train_iters;        // K-means iterations
    size_t default_rerank;     // Rerank pool size
    int use_cosine;            // Cosine normalization
    int use_scalar_quant;      // Additional scalar quantization
    float oversampling_factor; // Candidate oversampling
} GV_IVFPQConfig;
```

#### Workflow

```c
// 1. Create index
GV_IVFPQ *index = gv_ivfpq_create(dimension, &config);

// 2. Train on representative data
gv_ivfpq_train(index, training_vectors, num_training, GV_DISTANCE_EUCLIDEAN);

// 3. Insert vectors
for (int i = 0; i < num_vectors; i++) {
    gv_ivfpq_insert(index, vectors[i], &metadata[i]);
}

// 4. Search
gv_ivfpq_search(index, query, k, results, GV_DISTANCE_EUCLIDEAN);
```

#### Memory Calculation

```
Memory = n_vectors * (m * nbits / 8) + codebook_size
       = 1M vectors * (8 * 8 / 8) + negligible
       = 8 MB (vs 512 MB for raw float32 with d=128)
```

#### When to Use

- **Best for**: Very large datasets (1M+ vectors)
- **Memory**: 32x+ compression vs raw data
- **Latency**: Higher than HNSW but scalable
- **Recall**: 90%+ with proper nprobe/rerank

---

### K-D Tree

**File**: `include/gigavector/gv_kdtree.h`, `src/gv_kdtree.c`

K-D Tree is a space-partitioning data structure for exact nearest neighbor search in low dimensions.

#### How It Works

```
                    [x=5]
                   /     \
              [y=3]       [y=7]
             /    \      /    \
          [x=2]  [x=4] [x=6] [x=8]
```

Each level splits space along alternating dimensions.

#### When to Use

- **Best for**: Small datasets (<20K) OR low dimensions (<64)
- **Memory**: ~2x raw data size (tree structure)
- **Latency**: O(log n) average case
- **Recall**: 100% (exact search)

#### Limitations

- **Curse of dimensionality**: Performance degrades significantly above 20-30 dimensions
- Not suitable for high-dimensional embeddings (768, 1536 dimensions)

---

### Sparse Index

**File**: `include/gigavector/gv_sparse_index.h`, `src/gv_sparse_index.c`

Inverted index for sparse vectors (BM25, TF-IDF, high-dimensional feature vectors).

#### Sparse Vector Representation

```c
typedef struct {
    size_t dimension;           // Total dimensions
    size_t nnz;                 // Non-zero count
    GV_SparseEntry *entries;    // Index-value pairs
    GV_Metadata *metadata;      // Optional metadata
} GV_SparseVector;

typedef struct {
    uint32_t index;             // Dimension index
    float value;                // Value at dimension
} GV_SparseEntry;
```

#### Example: BM25 Vectors

```c
// Document: "the quick brown fox"
// Vocabulary: {"the": 0, "quick": 1, "brown": 2, "fox": 3, ...}
GV_SparseEntry entries[] = {
    {0, 0.1},   // "the" (low IDF)
    {1, 0.8},   // "quick"
    {2, 0.7},   // "brown"
    {3, 0.9}    // "fox"
};
GV_SparseVector doc = {
    .dimension = 50000,  // Vocabulary size
    .nnz = 4,
    .entries = entries
};
```

#### When to Use

- **Best for**: Text search (BM25), feature vectors with many zero values
- **Memory**: Proportional to non-zero entries
- **Latency**: Fast for sparse queries

---

## Distance Metrics

**File**: `include/gigavector/gv_distance.h`, `src/gv_distance.c`

### Supported Metrics

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| **Euclidean** | `√Σ(aᵢ - bᵢ)²` | [0, ∞) | General purpose, images |
| **Cosine** | `1 - (a·b)/(‖a‖‖b‖)` | [0, 2] | Text embeddings, normalized vectors |
| **Dot Product** | `-a·b` | (-∞, ∞) | When magnitude matters |
| **Manhattan** | `Σ|aᵢ - bᵢ|` | [0, ∞) | Sparse data, grid-based |

### SIMD Optimization

GigaVector auto-detects CPU features and uses optimized implementations:

```c
// Detected at runtime
if (gv_cpu_has_feature(GV_CPU_AVX2)) {
    // Use AVX2 implementation (8 floats at once)
} else if (gv_cpu_has_feature(GV_CPU_SSE)) {
    // Use SSE implementation (4 floats at once)
} else {
    // Fallback to scalar
}
```

---

## Memory Layer

### Memory Layer Architecture

**File**: `include/gigavector/gv_memory_layer.h`, `src/gv_memory_layer.c`

The memory layer provides semantic memory storage with intelligent retrieval, extraction, and consolidation.

```
┌─────────────────────────────────────────────────────────────┐
│                      Memory Layer                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌────────────┐  ┌─────────────┐              │
│  │Extraction│──│ Importance │──│Consolidation│              │
│  │  (LLM)   │  │  Scoring   │  │   (Merge)   │              │
│  └──────────┘  └────────────┘  └─────────────┘              │
│        │              │               │                      │
│        ▼              ▼               ▼                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Vector Database (HNSW/IVF-PQ)          │    │
│  │  [Memory 1] [Memory 2] [Memory 3] ... [Memory N]    │    │
│  └─────────────────────────────────────────────────────┘    │
│        │                                                     │
│        ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Context Graph (Entities)                │    │
│  │  [Person]──knows──[Person]                          │    │
│  │     │                 │                             │    │
│  │  works_at          lives_in                         │    │
│  │     │                 │                             │    │
│  │  [Org]            [Location]                        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Memory Types

```c
typedef enum {
    GV_MEMORY_TYPE_FACT = 0,        // Factual information
    GV_MEMORY_TYPE_PREFERENCE = 1,   // User preferences
    GV_MEMORY_TYPE_RELATIONSHIP = 2, // Entity relationships
    GV_MEMORY_TYPE_EVENT = 3         // Events/occurrences
} GV_MemoryType;
```

### Memory Metadata

```c
typedef struct {
    char *memory_id;                // Unique identifier (mem_xxxx)
    GV_MemoryType memory_type;      // Classification
    char *source;                   // Original source (conversation_id)
    time_t timestamp;               // Creation time
    time_t last_accessed;           // For decay calculation
    uint32_t access_count;          // Retrieval frequency
    double importance_score;        // Computed importance (0.0-1.0)
    char *extraction_metadata;      // JSON with extraction details
    GV_MemoryLink *links;           // Typed relationships
    size_t link_count;              // Number of links
    int consolidated;               // Consolidation flag
} GV_MemoryMetadata;
```

### Search Options (Cortex-Style Temporal Weighting)

```c
typedef struct {
    float temporal_weight;      // 0.0=semantic only, 1.0=recency only
    float importance_weight;    // Weight for importance (default: 0.4)
    int include_linked;         // Include linked memories
    float link_boost;           // Boost for linked memories (default: 0.1)
    time_t min_timestamp;       // Filter: minimum time
    time_t max_timestamp;       // Filter: maximum time
    int memory_type;            // Filter by type (-1 = all)
    const char *source;         // Filter by source (NULL = all)
} GV_MemorySearchOptions;
```

**Temporal Blending Formula** (from Cortex):
```
combined_score = semantic * (1 - temporal_weight) + recency * temporal_weight
recency = e^(-days_ago / 7.0)  // ~5-day half-life
```

---

### Memory Links and Relationships

Build a knowledge graph connecting related memories.

#### Link Types

```c
typedef enum {
    GV_LINK_SIMILAR = 0,        // Semantically similar
    GV_LINK_SUPPORTS = 1,       // Supports/reinforces target
    GV_LINK_CONTRADICTS = 2,    // Contradicts target (symmetric)
    GV_LINK_EXTENDS = 3,        // Elaborates/extends target
    GV_LINK_CAUSAL = 4,         // Caused by target
    GV_LINK_EXAMPLE = 5,        // Example of target
    GV_LINK_PREREQUISITE = 6,   // Target depends on this
    GV_LINK_TEMPORAL = 7        // Before/after relationship
} GV_MemoryLinkType;
```

#### Link Structure

```c
typedef struct {
    char *target_memory_id;      // ID of linked memory
    GV_MemoryLinkType link_type; // Relationship type
    float strength;              // Link strength (0.0-1.0)
    time_t created_at;           // When link was created
    char *reason;                // Why link was created
} GV_MemoryLink;
```

#### Bidirectional Links

When creating a link A→B, a reciprocal link B→A is automatically created with:
- Reciprocal link type (SUPPORTS ↔ SUPPORTS, etc.)
- Reduced strength (0.9x of original)

---

### Memory Extraction

**File**: `include/gigavector/gv_memory_extraction.h`, `src/gv_memory_extraction.c`

Extract factual memories from conversations and text.

#### Extraction Methods

1. **LLM-Based Extraction** (recommended):
   ```c
   // Uses LLM to understand context and extract key facts
   gv_memory_extract_candidates_from_conversation_llm(
       layer, conversation, conversation_id,
       candidates, max_candidates, &actual_count
   );
   ```

2. **Heuristic-Based Extraction** (fallback):
   ```c
   // Uses importance scoring and sentence analysis
   gv_memory_extract_candidates_from_text(
       text, candidates, max_candidates, &actual_count
   );
   ```

#### Memory Candidate

```c
typedef struct {
    char *content;              // Extracted fact
    double importance_score;    // Importance (0.0-1.0)
    GV_MemoryType memory_type;  // Detected type
    char *extraction_context;   // Source identifier
} GV_MemoryCandidate;
```

---

### Memory Consolidation

**File**: `include/gigavector/gv_memory_consolidation.h`, `src/gv_memory_consolidation.c`

Merge similar memories to reduce duplication.

#### Consolidation Strategies

```c
typedef enum {
    GV_CONSOLIDATION_MERGE = 0,    // Combine into one memory
    GV_CONSOLIDATION_UPDATE = 1,   // Update existing with new info
    GV_CONSOLIDATION_LINK = 2,     // Create relationship link
    GV_CONSOLIDATION_ARCHIVE = 3   // Mark redundant, keep history
} GV_ConsolidationStrategy;
```

#### Workflow

```c
// 1. Find similar memories above threshold
GV_MemoryPair pairs[100];
int pair_count = gv_memory_find_similar(layer, 0.85, pairs, 100);

// 2. Consolidate each pair
for (int i = 0; i < pair_count; i++) {
    gv_memory_consolidate_pair(layer, &pairs[i], GV_CONSOLIDATION_MERGE);
}
```

---

## Importance Scoring System

**File**: `include/gigavector/gv_importance.h`, `src/gv_importance.c`

Multi-factor importance scoring based on cognitive science and ML ranking research.

### Design Philosophy

**NO hardcoded keyword lists**. All scoring uses statistical and structural analysis:
- Language-agnostic (works for any language)
- Based on SOTA research from mem0, Cortex, supermemory
- Combines multiple signals for robust scoring

### Scoring Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Final Importance Score                    │
│  ═══════════════════════════════════════════════════════════│
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Content  │ │ Temporal │ │  Access  │ │Structural│       │
│  │  (30%)   │ │  (25%)   │ │  (20%)   │ │  (10%)   │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       │            │            │            │              │
│  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐        │
│  │Informtv.│  │Ebbinghaus│  │ SM-2    │  │ Links   │        │
│  │Specific.│  │ Curve    │  │ Style   │  │ Graph   │        │
│  │Salience │  │ Recency  │  │ Pattern │  │ Density │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Component Weights

```c
typedef struct {
    double content_weight;      // Content features (default: 0.30)
    double temporal_weight;     // Recency/decay (default: 0.25)
    double access_weight;       // Retrieval patterns (default: 0.20)
    double salience_weight;     // Emotional relevance (default: 0.15)
    double structural_weight;   // Relationships (default: 0.10)
} GV_ImportanceWeights;
```

### Temporal Decay (Ebbinghaus Forgetting Curve)

Based on memory retention research:

```
R = e^(-t/S)

Where:
- R = retention (0.0-1.0)
- t = time since creation
- S = stability = half_life / ln(2)
```

```c
typedef struct {
    double half_life_hours;         // Time to 50% (default: 168 = 1 week)
    double min_decay_factor;        // Minimum floor (default: 0.1)
    double recency_boost_hours;     // Recent window (default: 24)
    double recency_boost_factor;    // Boost multiplier (default: 1.5)
} GV_TemporalDecayConfig;
```

### Content Analysis (Language-Agnostic)

| Feature | Method | What It Measures |
|---------|--------|------------------|
| **Informativeness** | Type-Token Ratio | Lexical diversity |
| **Specificity** | Number density, capitals, patterns | Concrete vs vague |
| **Salience** | Punctuation (!?), ALL CAPS | Emphasis/emotion |
| **Entity Density** | Capitalized + number sequences | Named entities |

#### Specificity Detection

Instead of keyword matching, uses structural signals:
- **Number density**: "Meeting at 3:45 PM on Jan 15" → high specificity
- **Capitalized words**: "John works at Google" → proper nouns
- **Structural patterns**: URLs, emails, dates, times
- **Word length variance**: Technical terms tend to be longer
- **Short word ratio**: Vague text has many short words (it, that, this)

#### Salience Detection

Uses emphasis markers, not keyword lists:
- **Exclamation/question marks**: "This is important!" → higher salience
- **ALL CAPS words**: "URGENT: Please respond" → emphasis
- **Possessive patterns**: "my preference", "John's idea" → personal relevance
- **Sentence length variance**: Varied structure indicates importance

### Access Pattern Scoring (Spaced Repetition)

Inspired by SM-2 algorithm:

```c
typedef struct {
    double retrieval_boost_base;    // Boost per retrieval (default: 0.05)
    double retrieval_boost_decay;   // Decay for old retrievals (default: 0.95)
    double optimal_interval_hours;  // Ideal retrieval gap (default: 48)
    double interval_tolerance;      // Tolerance range (default: 0.5)
    size_t max_tracked_accesses;    // Max events to track (default: 100)
} GV_AccessPatternConfig;
```

### Result Structure

```c
typedef struct {
    double final_score;         // Combined score (0.0-1.0)

    // Component scores
    double content_score;
    double temporal_score;
    double access_score;
    double salience_score;
    double structural_score;

    // Sub-components
    double informativeness;
    double specificity;
    double entity_density;
    double decay_factor;
    double retrieval_boost;
    double recency_bonus;

    double confidence;          // Score confidence
    int factors_used;           // Bitmask of factors
} GV_ImportanceResult;
```

### Usage

```c
// Quick content-only scoring
double score = gv_importance_score_content(text, strlen(text));

// Full importance calculation
GV_ImportanceContext ctx = {
    .content = text,
    .content_length = strlen(text),
    .creation_time = memory_timestamp,
    .current_time = time(NULL),
    .semantic_similarity = query_similarity,
    .relationship_count = num_links
};

GV_ImportanceConfig config = gv_importance_config_default();
GV_ImportanceResult result;
gv_importance_calculate(&config, &ctx, &result);

printf("Final score: %.2f (confidence: %.2f)\n",
       result.final_score, result.confidence);
```

---

## LLM Integration

**File**: `include/gigavector/gv_llm.h`, `src/gv_llm.c`

### Supported Providers

```c
typedef enum {
    GV_LLM_PROVIDER_OPENAI = 0,     // GPT models
    GV_LLM_PROVIDER_GOOGLE = 1,     // Gemini
    GV_LLM_PROVIDER_ANTHROPIC = 2,  // Claude
    GV_LLM_PROVIDER_CUSTOM = 3      // OpenAI-compatible API
} GV_LLMProvider;
```

### Configuration

```c
typedef struct {
    GV_LLMProvider provider;
    char *api_key;              // API key
    char *model;                // Model name (e.g., "gpt-4")
    char *base_url;             // Custom endpoint
    double temperature;         // Generation temperature
    int max_tokens;             // Max response tokens
    int timeout_seconds;        // Request timeout
    char *custom_prompt;        // Custom extraction prompt
} GV_LLMConfig;
```

### Usage

```c
// Create LLM instance
GV_LLMConfig config = {
    .provider = GV_LLM_PROVIDER_OPENAI,
    .api_key = getenv("OPENAI_API_KEY"),
    .model = "gpt-4",
    .temperature = 0.7,
    .max_tokens = 1000,
    .timeout_seconds = 30
};
GV_LLM *llm = gv_llm_create(&config);

// Generate response
GV_LLMMessage messages[] = {
    {"system", "Extract key facts from the conversation."},
    {"user", conversation_text}
};
GV_LLMResponse response;
int result = gv_llm_generate_response(llm, messages, 2, &response);

if (result == 0) {
    printf("Response: %s\n", response.content);
    gv_llm_response_free(&response);
}

gv_llm_destroy(llm);
```

---

## Embedding System

**File**: `include/gigavector/gv_embedding.h`, `src/gv_embedding.c`

### Supported Providers

```c
typedef enum {
    GV_EMBEDDING_PROVIDER_OPENAI = 0,
    GV_EMBEDDING_PROVIDER_HUGGINGFACE = 1,
    GV_EMBEDDING_PROVIDER_GOOGLE = 2,
    GV_EMBEDDING_PROVIDER_CUSTOM = 3,
    GV_EMBEDDING_PROVIDER_NONE = 4
} GV_EmbeddingProvider;
```

### Configuration

```c
typedef struct {
    GV_EmbeddingProvider provider;
    char *api_key;
    char *model;                    // e.g., "text-embedding-3-small"
    char *base_url;                 // For custom providers
    size_t embedding_dimension;     // Expected output dimension
    size_t batch_size;              // Batch size (default: 100)
    int enable_cache;               // Enable caching
    size_t cache_size;              // Max cached entries
    int timeout_seconds;
    char *huggingface_model_path;   // Local model path
} GV_EmbeddingConfig;
```

### Caching

```c
// Create cache
GV_EmbeddingCache *cache = gv_embedding_cache_create(10000);

// Check cache
float *cached = gv_embedding_cache_get(cache, "some text");
if (cached) {
    // Use cached embedding
} else {
    // Generate and cache
    float *embedding = generate_embedding("some text");
    gv_embedding_cache_put(cache, "some text", embedding, dimension);
}

// Get stats
GV_EmbeddingCacheStats stats = gv_embedding_cache_stats(cache);
printf("Hit rate: %.2f%%\n", 100.0 * stats.hits / (stats.hits + stats.misses));
```

---

## Context Graph

**File**: `include/gigavector/gv_context_graph.h`, `src/gv_context_graph.c`

Build knowledge graphs of entities and their relationships.

### Entity Types

```c
typedef enum {
    GV_ENTITY_TYPE_PERSON = 0,
    GV_ENTITY_TYPE_ORGANIZATION = 1,
    GV_ENTITY_TYPE_LOCATION = 2,
    GV_ENTITY_TYPE_EVENT = 3,
    GV_ENTITY_TYPE_OBJECT = 4,
    GV_ENTITY_TYPE_CONCEPT = 5,
    GV_ENTITY_TYPE_USER = 6         // Self-reference
} GV_EntityType;
```

### Entity Structure

```c
typedef struct {
    char *entity_id;
    char *name;
    GV_EntityType entity_type;
    float *embedding;               // Entity embedding
    size_t embedding_dim;
    time_t created;
    time_t updated;
    uint64_t mentions;              // Reference count
    char *user_id;                  // User scope
    char *agent_id;                 // Agent scope (optional)
    char *run_id;                   // Run scope (optional)
} GV_GraphEntity;
```

### Relationship Structure

```c
typedef struct {
    char *relationship_id;
    char *source_entity_id;
    char *destination_entity_id;
    char *relationship_type;        // e.g., "knows", "works_at"
    time_t created;
    time_t updated;
    uint64_t mentions;
} GV_GraphRelationship;
```

### Usage

```c
// Create context graph
GV_ContextGraphConfig config = {
    .llm = llm,
    .embedding_service = embedding_service,
    .similarity_threshold = 0.85,
    .enable_entity_extraction = 1,
    .enable_relationship_extraction = 1,
    .max_traversal_depth = 3,
    .max_results = 100
};
GV_ContextGraph *graph = gv_context_graph_create(db, &config);

// Extract entities from text
GV_GraphEntity entities[10];
GV_GraphRelationship rels[20];
size_t entity_count, rel_count;
gv_context_graph_extract(
    graph, "John works at Google in San Francisco",
    entities, 10, &entity_count,
    rels, 20, &rel_count
);

// Query related entities
GV_GraphEntity related[10];
int count = gv_context_graph_get_related(graph, "entity_john", related, 10);
```

---

## Core Data Structures

### Vector Types

```c
// Dense vector
typedef struct {
    size_t dimension;
    float *data;
    GV_Metadata *metadata;
} GV_Vector;

// Sparse vector
typedef struct {
    size_t dimension;           // Total dimensions
    size_t nnz;                 // Non-zero count
    GV_SparseEntry *entries;    // Index-value pairs
    GV_Metadata *metadata;
} GV_SparseVector;

typedef struct {
    uint32_t index;
    float value;
} GV_SparseEntry;
```

### Metadata

```c
// Linked list of key-value pairs
typedef struct GV_Metadata {
    char *key;
    char *value;
    struct GV_Metadata *next;
} GV_Metadata;

// Operations
gv_vector_set_metadata(vec, "category", "document");
const char *cat = gv_vector_get_metadata(vec, "category");
gv_vector_remove_metadata(vec, "category");
```

### Search Results

```c
typedef struct {
    const GV_Vector *vector;    // Pointer to matched vector
    float distance;             // Distance/similarity
} GV_SearchResult;

typedef struct {
    char *memory_id;
    char *content;
    float relevance_score;
    float distance;
    GV_MemoryMetadata *metadata;
    GV_MemoryResult *related;   // Related memories
    size_t related_count;
} GV_MemoryResult;
```

---

## Storage and Persistence

### Structure-of-Arrays (SoA) Storage

**File**: `include/gigavector/gv_soa_storage.h`

Optimized for cache efficiency:

```
Traditional (AoS):        SoA Storage:
┌─────────────────┐      ┌───────────────────────────┐
│ Vec1: [data][meta]│    │ Data:  [v1 v2 v3 v4 ...]  │
│ Vec2: [data][meta]│    │ Meta:  [m1 m2 m3 m4 ...]  │
│ Vec3: [data][meta]│    │ Flags: [f1 f2 f3 f4 ...]  │
└─────────────────┘      └───────────────────────────┘
```

Benefits:
- Better cache locality during search
- Efficient SIMD operations
- Reduced memory fragmentation

### Write-Ahead Log (WAL)

**File**: `include/gigavector/gv_wal.h`, `src/gv_wal.c`

Durability through transaction logging:

```c
// Operations logged
gv_wal_append_insert(wal, vector, dimension);
gv_wal_append_insert_rich(wal, vector, dimension, keys, values, count);
gv_wal_append_delete(wal, index);
gv_wal_append_update(wal, index, new_vector, dimension, keys, values, count);

// Recovery
gv_wal_replay(wal, db);

// Maintenance
gv_wal_truncate(wal);  // After checkpoint
```

### Quantization

#### Binary Quantization (32x compression)

```c
// 1 bit per dimension
uint8_t *binary = gv_binary_quantize(vector, dimension);
uint32_t hamming = gv_binary_hamming_distance(binary1, binary2, dimension);
```

#### Scalar Quantization (4-8x compression)

```c
// Quantize to 8-bit
uint8_t *quantized = gv_scalar_quantize(vector, dimension, min, max, 8);
float *restored = gv_scalar_dequantize(quantized, dimension, min, max, 8);
```

---

## Configuration and Optimization

### Index Selection Heuristic

```c
GV_IndexType type = gv_index_suggest(dimension, expected_count);

// Logic:
// - Small (<=20k) + low dim (<=64): KD-Tree
// - Very large (>=500k) + high dim (>=128): IVF-PQ
// - Otherwise: HNSW
```

### Resource Limits

```c
typedef struct {
    size_t max_memory_bytes;
    size_t max_vectors;
    size_t max_concurrent_operations;
} GV_ResourceLimits;

gv_db_set_resource_limits(db, &limits);
```

### Monitoring

```c
// Basic stats
GV_Stats stats;
gv_db_get_stats(db, &stats);

// Detailed stats
GV_DetailedStats detailed;
gv_db_get_detailed_stats(db, &detailed);
printf("QPS: %.2f, Avg latency: %.2fms\n",
       detailed.qps, detailed.avg_search_latency_ms);

// Health check
int health = gv_db_health_check(db);  // 0=healthy, -1=degraded, -2=unhealthy
```

### Background Compaction

```c
// Configure
gv_db_set_compaction_interval(db, 300);           // Every 5 minutes
gv_db_set_wal_compaction_threshold(db, 10485760); // 10MB WAL
gv_db_set_deleted_ratio_threshold(db, 0.1);       // 10% deleted

// Start/stop
gv_db_start_background_compaction(db);
gv_db_stop_background_compaction(db);

// Manual
gv_db_compact(db);
```

---

## Integration Patterns

### Complete Memory System Workflow

```c
// 1. Initialize components
GV_Database *db = gv_db_open("memories.db", 1536, GV_INDEX_TYPE_HNSW);

GV_EmbeddingConfig embed_config = {
    .provider = GV_EMBEDDING_PROVIDER_OPENAI,
    .api_key = getenv("OPENAI_API_KEY"),
    .model = "text-embedding-3-small",
    .embedding_dimension = 1536
};
GV_EmbeddingService *embeddings = gv_embedding_service_create(&embed_config);

GV_LLMConfig llm_config = {
    .provider = GV_LLM_PROVIDER_OPENAI,
    .api_key = getenv("OPENAI_API_KEY"),
    .model = "gpt-4"
};
GV_LLM *llm = gv_llm_create(&llm_config);

GV_MemoryLayerConfig mem_config = gv_memory_layer_config_default();
mem_config.use_llm_extraction = 1;
mem_config.llm_config = llm;
mem_config.enable_temporal_weighting = 1;

GV_MemoryLayer *memory = gv_memory_layer_create(db, &mem_config);

// 2. Store memories from conversation
const char *conversation = "User: My name is John and I work at Google.\n"
                           "Assistant: Nice to meet you, John!";

GV_MemoryCandidate candidates[10];
size_t count;
gv_memory_extract_candidates_from_conversation_llm(
    memory, conversation, "conv_123",
    candidates, 10, &count
);

for (size_t i = 0; i < count; i++) {
    float *embedding = gv_embedding_generate(embeddings, candidates[i].content);
    gv_memory_add(memory, candidates[i].content, embedding, NULL);
    free(embedding);
}

// 3. Retrieve memories
float *query_embedding = gv_embedding_generate(embeddings, "Where does John work?");

GV_MemorySearchOptions options = gv_memory_search_options_default();
options.temporal_weight = 0.3;  // Favor recent memories
options.importance_weight = 0.4;

GV_MemoryResult results[5];
int result_count = gv_memory_search_advanced(
    memory, query_embedding, 5, results,
    GV_DISTANCE_COSINE, &options
);

for (int i = 0; i < result_count; i++) {
    printf("Memory: %s (score: %.2f)\n",
           results[i].content, results[i].relevance_score);
    gv_memory_result_free(&results[i]);
}

// 4. Consolidate similar memories
GV_MemoryPair pairs[10];
int pair_count = gv_memory_find_similar(memory, 0.9, pairs, 10);
for (int i = 0; i < pair_count; i++) {
    gv_memory_consolidate_pair(memory, &pairs[i], GV_CONSOLIDATION_MERGE);
}

// 5. Cleanup
gv_memory_layer_destroy(memory);
gv_llm_destroy(llm);
gv_embedding_service_destroy(embeddings);
gv_db_close(db);
```

### Pure Vector Search (No Memory Layer)

```c
// Simple HNSW search
GV_HNSWConfig hnsw_config = gv_hnsw_config_default();
hnsw_config.M = 32;
hnsw_config.efSearch = 100;

GV_Database *db = gv_db_open_with_hnsw_config("vectors.db", 128, &hnsw_config);

// Insert vectors
for (int i = 0; i < num_vectors; i++) {
    gv_db_add_vector_with_metadata(db, vectors[i],
        (const char*[]){"id", "category"},
        (const char*[]){ids[i], categories[i]}, 2);
}

// Search with filter
GV_SearchResult results[10];
int count = gv_db_search_with_filter_expr(
    db, query, 10, results,
    GV_DISTANCE_COSINE,
    "category == \"science\""
);

gv_db_close(db);
```

---

## Glossary

| Term | Definition |
|------|------------|
| **ANN** | Approximate Nearest Neighbor - finding similar vectors quickly |
| **HNSW** | Hierarchical Navigable Small Worlds - graph-based ANN algorithm |
| **IVF** | Inverted File - partitioning vectors into clusters |
| **PQ** | Product Quantization - compressing vectors into codes |
| **Ebbinghaus Curve** | Memory retention decay formula: R = e^(-t/S) |
| **SM-2** | Spaced repetition algorithm for optimal review intervals |
| **TTR** | Type-Token Ratio - lexical diversity measure |
| **BM25** | Best Match 25 - statistical ranking function for text |
| **SoA** | Structure-of-Arrays - cache-efficient storage layout |
| **WAL** | Write-Ahead Log - durability through transaction logging |

---

## License

GigaVector is released under the MIT License. See LICENSE for details.
