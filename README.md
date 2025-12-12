# GigaVector

A fast, modular vector database in C with optional Python bindings.

## Features
- KD-Tree, HNSW, and IVFPQ indexes
- Euclidean and cosine distances
- Rich per-vector metadata (multiple key-value pairs)
- Persistence with snapshot plus WAL replay
- Python bindings via `cffi` (published on PyPI as `gigavector`)

## Build (C library)
```bash
make lib        # builds static and shared libraries into build/lib/
make c-test     # runs C tests (needs LD_LIBRARY_PATH=build/lib)
```

## Python bindings
From PyPI:
```bash
pip install gigavector
```

From source:
```bash
cd python
python -m pip install .
```

## Quick start (Python)
```python
from gigavector import Database, DistanceType, IndexType

with Database.open("example.db", dimension=4, index=IndexType.KDTREE) as db:
    db.add_vector([1, 2, 3, 4], metadata={"tag": "sample", "owner": "user"})
    hits = db.search([1, 2, 3, 4], k=1, distance=DistanceType.EUCLIDEAN)
    print(hits[0].distance, hits[0].vector.metadata)
```

Persistence:
```python
db.save("example.db")          # snapshot
# On reopen, WAL is replayed automatically
with Database.open("example.db", dimension=4, index=IndexType.KDTREE) as db:
    ...
```

IVFPQ training (dimension must match):
```python
train = [[(i % 10) / 10.0 for _ in range(8)] for i in range(256)]
with Database.open(None, dimension=8, index=IndexType.IVFPQ) as db:
    db.train_ivfpq(train)
    db.add_vector([0.5] * 8)
```

## License
This project is licensed under the DBaJ-NC-CFL [License](./LICENCE).