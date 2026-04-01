# Basic Usage Examples

This page provides practical starter examples for common GigaVector workflows.

## Python: create, insert, search

```python
from gigavector import Database, IndexType, DistanceType

with Database.open("demo.db", dimension=4, index=IndexType.FLAT) as db:
    db.add_vector([1.0, 0.0, 0.0, 0.0], metadata={"category": "science"})
    db.add_vector([0.0, 1.0, 0.0, 0.0], metadata={"category": "tech"})

    hits = db.search([1.0, 0.0, 0.0, 0.0], k=2, distance=DistanceType.COSINE)
    for hit in hits:
        print(hit.distance, hit.vector.metadata)
```

## Python: filtered search

```python
hits = db.search(
    [1.0, 0.0, 0.0, 0.0],
    k=10,
    distance=DistanceType.COSINE,
    filter_metadata=("category", "science"),
)
```

## C: open database and add vector

```c
#include "gigavector/gigavector.h"

int main(void) {
    GV_Database *db = gv_db_open("demo.db", 4, GV_INDEX_TYPE_FLAT);
    if (!db) return 1;

    float v[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    gv_db_add_vector_with_metadata(db, v, 4, "category", "science");

    gv_db_close(db);
    return 0;
}
```

## SQL interface

```sql
SELECT * FROM vectors WHERE category = 'science' LIMIT 10;
SELECT COUNT(*) FROM vectors WHERE category <> 'archived';
```

## Next steps

- See [Advanced Features](advanced_features.md) for ANN tuning, hybrid search, and ranking pipelines.
- See [Usage Guide](../usage.md) for broader API examples.
