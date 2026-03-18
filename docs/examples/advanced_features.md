# Advanced Features

## Index Configuration

### Fine-Tuning HNSW

```python
from gigavector import Database, IndexType, HNSWConfig

# High-quality HNSW configuration
high_quality_config = HNSWConfig(
    M=48,                    # More connections for better recall
    ef_construction=400,     # Higher construction quality
    ef_search=100,           # Higher search quality
    use_binary_quant=True,   # Enable binary quantization
    quant_rerank=50,         # Rerank top 50 candidates
    use_acorn=True,          # Enable ACORN for filtered search
    acorn_hops=2             # 2-hop exploration
)

with Database.open(
    "high_quality.db", dimension=128, index=IndexType.HNSW,
    hnsw_config=high_quality_config
) as db:
    # Add vectors
    for i in range(1000000):
        vector = [random.random() for _ in range(128)]
        db.add_vector(vector)

    # Search with high recall
    hits = db.search(query, k=10)
```

### Optimized IVFPQ Configuration

```python
from gigavector import Database, IndexType, IVFPQConfig, ScalarQuantConfig

# Memory-optimized IVFPQ
memory_optimized_config = IVFPQConfig(
    nlist=1024,              # More centroids for large datasets
    m=32,                    # More subquantizers for accuracy
    nbits=8,                 # Standard quantization
    nprobe=64,               # Probe more lists for recall
    train_iters=20,          # More training iterations
    default_rerank=64,       # Larger rerank pool
    use_scalar_quant=True,   # Enable scalar quantization
    scalar_quant_config=ScalarQuantConfig(
        bits=8,              # 8-bit scalar quantization
        per_dimension=False  # Global quantization
    ),
    oversampling_factor=2.0  # 2x oversampling for recall
)

with Database.open(
    None, dimension=256, index=IndexType.IVFPQ,
    ivfpq_config=memory_optimized_config
) as db:
    # Train with large dataset
    training_data = [
        [random.random() for _ in range(256)]
        for _ in range(10000)
    ]
    db.train_ivfpq(training_data)

    # Add millions of vectors
    for i in range(10000000):
        vector = [random.random() for _ in range(256)]
        db.add_vector(vector)
```

## Range Search

Range search finds all vectors within a specified distance threshold.

### C API

```c
// Range search: find all vectors within radius 0.5
float radius = 0.5f;
GV_SearchResult results[1000];  // Pre-allocated results array
size_t max_results = 1000;

int found = gv_db_range_search(
    db, query, 128, radius, results, max_results,
    GV_DISTANCE_EUCLIDEAN
);

printf("Found %d vectors within radius %f\n", found, radius);
for (int i = 0; i < found; i++) {
    printf("  Distance: %f, Index: %zu\n",
           results[i].distance, results[i].vector_index);
}
```

### Python API

```python
# Range search
radius = 0.5
hits = db.range_search(query, radius=radius, distance=DistanceType.EUCLIDEAN)

print(f"Found {len(hits)} vectors within radius {radius}")
for hit in hits:
    print(f"  Distance: {hit.distance}")
```

### Filtered Range Search

```python
# Range search with metadata filter
hits = db.range_search(
    query, radius=0.5, distance=DistanceType.EUCLIDEAN,
    filter_metadata=("category", "electronics")
)
```

## Sparse Vectors

GigaVector supports sparse vectors for efficient storage of high-dimensional sparse data.

### C API

```c
// Create sparse vector
GV_SparseVector *sparse = gv_sparse_vector_create(1000);  // 1000 dimensions

// Set non-zero values
gv_sparse_vector_set(sparse, 10, 0.5f);   // dimension 10 = 0.5
gv_sparse_vector_set(sparse, 50, 0.3f);   // dimension 50 = 0.3
gv_sparse_vector_set(sparse, 100, 0.8f);   // dimension 100 = 0.8

// Add to sparse index
GV_Database *db = gv_db_open("sparse.db", 1000, GV_INDEX_TYPE_SPARSE);
gv_db_add_sparse_vector(db, sparse);

// Search
GV_SparseVector *query_sparse = gv_sparse_vector_create(1000);
gv_sparse_vector_set(query_sparse, 10, 0.6f);
gv_sparse_vector_set(query_sparse, 50, 0.4f);

GV_SearchResult results[10];
int found = gv_db_search_sparse(db, query_sparse, 10, results, GV_DISTANCE_EUCLIDEAN);

gv_sparse_vector_destroy(sparse);
gv_sparse_vector_destroy(query_sparse);
```

### Python API

```python
from gigavector import Database, IndexType

# Create sparse database
with Database.open("sparse.db", dimension=1000, index=IndexType.SPARSE) as db:
    # Create sparse vector using indices and values
    indices = [10, 50, 100]   # non-zero dimensions
    values = [0.5, 0.3, 0.8]  # corresponding values

    db.add_sparse_vector(indices, values)

    # Search with sparse query
    query_indices = [10, 50]
    query_values = [0.6, 0.4]
    hits = db.search_sparse(query_indices, query_values, k=10)
```

## Graph Database and Knowledge Graph

GigaVector includes a full property graph database and a knowledge graph layer with vector embeddings.

### Building a Social Graph

```c
#include "gigavector/gv_graph_db.h"

GV_GraphDB *g = gv_graph_create(NULL);

// Create nodes
uint64_t alice = gv_graph_add_node(g, "Person");
uint64_t bob = gv_graph_add_node(g, "Person");
uint64_t charlie = gv_graph_add_node(g, "Person");
gv_graph_set_node_prop(g, alice, "name", "Alice");
gv_graph_set_node_prop(g, bob, "name", "Bob");
gv_graph_set_node_prop(g, charlie, "name", "Charlie");

// Create relationships
gv_graph_add_edge(g, alice, bob, "KNOWS", 1.0f);
gv_graph_add_edge(g, bob, charlie, "KNOWS", 1.0f);
gv_graph_add_edge(g, alice, charlie, "FRIENDS", 0.5f);

// Find shortest path
GV_GraphPath path;
if (gv_graph_shortest_path(g, alice, charlie, &path) == 0) {
    printf("Path length: %zu, weight: %.2f\n", path.length, path.total_weight);
    gv_graph_free_path(&path);
}

// Compute PageRank
float pr = gv_graph_pagerank(g, alice, 20, 0.85f);
printf("Alice PageRank: %.4f\n", pr);

// Connected components
uint64_t comps[3];
int num_comps = gv_graph_connected_components(g, comps, 3);
printf("Components: %d\n", num_comps);

gv_graph_save(g, "social.gvgr");
gv_graph_destroy(g);
```

### Knowledge Graph with Semantic Search

```python
from gigavector import KnowledgeGraph, KGConfig

kg = KnowledgeGraph(KGConfig(embedding_dimension=128))

# Build knowledge base
alice = kg.add_entity("Alice", "Person", embedding=[0.1] * 128)
bob = kg.add_entity("Bob", "Person", embedding=[0.2] * 128)
company = kg.add_entity("Anthropic", "Company", embedding=[0.5] * 128)

kg.add_relation(alice, "works_at", company)
kg.add_relation(bob, "works_at", company)
kg.add_relation(alice, "manages", bob)

# SPO triple queries
triples = kg.query_triples(predicate="works_at")
print(f"Found {len(triples)} 'works_at' triples")

# Semantic search
results = kg.search_similar([0.15] * 128, k=3)
for r in results:
    print(f"  {r.name} ({r.type}): similarity={r.similarity:.3f}")

# Hybrid search (vector + graph filters)
results = kg.hybrid_search(
    [0.1] * 128, entity_type="Person", predicate_filter="works_at", k=5
)

# Entity resolution (deduplicate)
resolved = kg.resolve_entity("Alice Smith", "Person", embedding=[0.1] * 128)
print(f"Resolved to entity {resolved}")

# Link prediction
predictions = kg.predict_links(alice, k=3)
for p in predictions:
    print(f"  Predicted: {p.entity_a} -> {p.entity_b} ({p.confidence:.3f})")

# Subgraph extraction
subgraph = kg.extract_subgraph(center=alice, radius=2)
print(f"Subgraph: {len(subgraph.entity_ids)} entities, {len(subgraph.relation_ids)} relations")

# Persistence
kg.save("knowledge.gvkg")
kg2 = KnowledgeGraph.load("knowledge.gvkg")
```
