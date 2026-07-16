# GigaVector DB Architecture: Correct Layer Design

## Layer Responsibilities (Source of Truth)

| Layer | Analogy | Owns | Answers |
|---|---|---|---|
| **Embedding Layer** (`GV_Database`) | Pinecone / Qdrant | Chunk vectors + ANN indexes | "What chunks are semantically similar to this query?" |
| **Graph Layer** (`GV_KnowledgeGraph`) | Neo4j | SPO triplets extracted from chunks | "What facts/relations exist in this document?" |
| **Memory Layer** (`GV_MemoryLayer`) | mem0 | Distilled user-specific facts | "What do I know about this user?" |

All three are linked by a **shared `chunk_id`** — the primary key that lets a result from any one layer be expanded with context from the other two.

---

## The Ingestion Pipeline (long document in)

```
User sends a long document
            │
            ▼
    ┌───────────────┐
    │    Chunker     │  overlap=128 tokens, size=512 tokens
    └───────┬───────┘
            │ produces chunks: [c0, c1, c2 ... cN]
            │ each chunk gets a shared ID:
            │   chunk_id = "{doc_id}:{chunk_index}"
            │   e.g.  "doc_7f3a:0042"
            │
            ▼
    for each chunk in parallel:
            │
     ┌──────┴──────────────────────────┐
     │                                 │                          │
     ▼                                 ▼                          ▼
┌─────────────────┐         ┌──────────────────┐      ┌──────────────────────┐
│  Embedding Layer │         │   Graph Layer     │      │    Memory Layer       │
│                  │         │                  │      │                      │
│  embed(chunk)    │         │  LLM extracts    │      │  LLM distills        │
│  → db_add_vector │         │  triplets from   │      │  key facts from      │
│    metadata: {   │         │  chunk text      │      │  chunk text          │
│    chunk_id,     │         │                  │      │                      │
│    doc_id,       │         │  kg_add_entity() │      │  memory_add()        │
│    chunk_index,  │         │  kg_add_relation()      │  metadata: {         │
│    char_start,   │         │  each triplet    │      │  chunk_id,           │
│    char_end      │         │  stores chunk_id │      │  doc_id,             │
│  }               │         │  as property     │      │  fact_type           │
└─────────────────┘         └──────────────────┘      │  }                   │
                                                        └──────────────────────┘
```

### Concrete example

Document: *"Alan Turing worked at Bletchley Park during WW2. He invented the Turing machine."*

After chunking and ingestion:

**Embedding Layer** stores:
```
vec_id=0  embedding=embed("Alan Turing worked at Bletchley Park...")
          metadata={ chunk_id="doc_abc:0", doc_id="doc_abc",
                     chunk_index=0, char_start=0, char_end=512 }
```

**Graph Layer** stores:
```
(Alan Turing,  WORKED_AT,   Bletchley Park)  { chunk_id: "doc_abc:0" }
(Alan Turing,  ACTIVE_DURING, WW2)           { chunk_id: "doc_abc:0" }
(Alan Turing,  INVENTED,    Turing Machine)  { chunk_id: "doc_abc:1" }
```

**Memory Layer** stores:
```
fact: "Alan Turing invented the Turing machine"
metadata={ chunk_id="doc_abc:1", doc_id="doc_abc",
           fact_type="invention", importance=0.9 }
```

---

## The Search Pipeline (query in)

```
User query: "What did Turing invent?"
                    │
                    ▼
            embed(query_text)
                    │
         ┌──────────┼──────────────────┐
         │          │                  │
         ▼          ▼                  ▼
  Embedding     Graph Layer        Memory Layer
  Layer         (triplet match)    (fact match)
  ANN search    LLM extracts       memory_search()
  db_search()   entities from      filtered by
  top-k chunks  query → pattern    user_id / doc_id
  by cosine     match triples
                kg_query_triples()

  results:       results:           results:
  [doc_abc:1,    (Alan Turing,      "Turing invented
   doc_abc:0,     INVENTED, ?)       Turing machine"
   doc_xyz:3]    → Turing Machine   chunk_id=doc_abc:1

         │          │                  │
         └──────────┴──────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │  chunk_id join │  ← the shared ID does the merge
            └───────┬───────┘
                    │
                    ▼
         for each unique chunk_id:
           score = vector_score   * 0.5
                 + triplet_hits   * 0.3   (how many graph results share this chunk)
                 + memory_hits    * 0.2   (how many memory facts share this chunk)

                    │
                    ▼
         fetch full chunk text from Embedding Layer metadata
         attach matched triplets from Graph Layer
         attach matched facts from Memory Layer
                    │
                    ▼
            return enriched results:
            {
              chunk_text:  "He invented the Turing machine.",
              chunk_id:    "doc_abc:1",
              vector_score: 0.91,
              triplets:    [(Alan Turing, INVENTED, Turing Machine)],
              facts:       ["Turing invented the Turing machine"],
            }
```

---

## The `chunk_id` as the Shared Key

Every layer stores `chunk_id` as metadata/property on every record it writes. This enables:

- **Expansion**: a vector hit → look up triplets with the same `chunk_id` to add structured context
- **Deduplication**: if three layers all return results for `chunk_id="doc_abc:1"`, it's one merged result with a boosted score, not three separate results
- **Provenance**: every fact and triplet is traceable back to the exact chunk (and therefore exact character range) in the source document
- **Deletion**: delete a document → delete all records across all three layers where `doc_id == that_doc`

### `chunk_id` format

```
chunk_id = "{doc_id}:{chunk_index:04d}"

doc_id   = sha256(source_url + ingestion_timestamp)[:16]
           e.g. "7f3a9b2c1d4e5f60"

example:   "7f3a9b2c1d4e5f60:0042"
```

---

## Data Structures

### Chunk (produced by chunker, consumed by all three writers)

```c
typedef struct {
    char    chunk_id[64];       /* "{doc_id}:{index:04d}" */
    char    doc_id[32];         /* sha256 prefix of source */
    size_t  chunk_index;        /* position in document */
    char   *text;               /* raw chunk text */
    size_t  char_start;         /* byte offset in original doc */
    size_t  char_end;
    float  *embedding;          /* produced by embedding service */
    size_t  embedding_dim;
} GV_Chunk;
```

### Enriched search result (returned to caller)

```c
typedef struct {
    char    chunk_id[64];
    char   *chunk_text;
    float   score;              /* merged score */
    float   vector_score;
    int     triplet_hit_count;
    int     memory_hit_count;
    GV_KGTriple  *triplets;     /* graph layer results for this chunk */
    size_t        triplet_count;
    char        **facts;        /* memory layer results for this chunk */
    size_t        fact_count;
} GV_EnrichedResult;
```

---

## What Needs to Be Built

### 1. Chunker (`src/storage/chunker.c`)

Does not exist yet. Needs:
- `gv_chunk_document(text, chunk_size, overlap, doc_id, out_chunks, out_count)`
- Splits on sentence boundaries where possible, falls back to token count
- Assigns `chunk_id = doc_id + ":" + zero_padded_index`
- Returns `GV_Chunk[]` with text + char offsets

### 2. Ingest coordinator (`src/storage/document_ingest.c`)

Does not exist yet. Needs:
- `gv_ingest_document(db, kg, memory_layer, llm, text, doc_id)`
- Calls chunker, then for each chunk in parallel:
  - `db_add_vector_with_rich_metadata()` → Embedding Layer
  - LLM extracts triplets → `kg_add_entity` + `kg_add_relation` with `chunk_id` property
  - LLM distills facts → `memory_add()` with `chunk_id` metadata
- Atomic at chunk granularity (partial ingest is recoverable)

### 3. Unified search (`src/storage/document_search.c`)

Does not exist yet. Needs:
- `gv_search_document(db, kg, memory_layer, query_text, query_embedding, k, out)`
- Fans out to all three layers in parallel
- Joins results on `chunk_id`, computes merged score
- Returns `GV_EnrichedResult[]`

### 4. Wire `chunk_id` into Graph Layer triplet properties

Currently `kg_add_relation` does not have a properties argument. Needs:
- `kg_add_relation_with_props(kg, subject, object, predicate, weight, props, n_props)`
- so `chunk_id` can be stored on each triple

### 5. `GV_KnowledgeGraph` → stop owning its own embeddings

- Remove `float *all_embeddings` from internal struct
- Add `GV_Database *vdb` field
- Replace `kg_search_similar` brute-force with `db_search(kg->vdb, ...)`

---

## Summary

```
Document
    │
    ▼
Chunker  ──────────────────────────────────── chunk_id binds all three
    │
    ├── embed each chunk ──────────────► Embedding Layer (fast ANN recall)
    ├── extract triplets from each chunk ► Graph Layer   (structured facts)
    └── distill facts from each chunk ──► Memory Layer  (user-relevant facts)

Query
    │
    ├── ANN search ─────────────────────► Embedding Layer
    ├── entity pattern match ────────────► Graph Layer
    └── fact search ─────────────────────► Memory Layer
    │
    └── join on chunk_id → merge scores → GV_EnrichedResult
```

The `chunk_id` is the only shared state. It costs nothing (just a string stored as metadata in each layer) and gives you full cross-layer joins at search time without tight coupling between the layers.
