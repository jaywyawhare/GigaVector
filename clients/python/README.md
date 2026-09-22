# GigaVector Python client (REST)

Standalone stdlib-only client for talking to a remote `gvserver` over HTTP.
(Distinct from the in-process cffi bindings under `python/src/gigavector`.)

```python
from gigavector import GigaVector, Distance

c = GigaVector("http://localhost:8080", api_key="secret")
c.add_vector([1, 0, 0, 0], {"tag": "a"})
for hit in c.search([1, 0, 0, 0], k=5, distance=Distance.COSINE):
    print(hit["id"], hit["distance"])
```

Run tests: `python3 -m unittest` (mock server, no C server needed). End-to-end:
`GIGAVECTOR_TEST_ADDR=http://localhost:8080 python3 -m unittest -k Integration`.

## LangChain

`gigavector_langchain.GigaVectorStore` is a LangChain `VectorStore` backed by a
remote GigaVector server (embeddings are computed client-side by any LangChain
`Embeddings`; the document text is kept in metadata so `similarity_search` can
rebuild `Document`s):

```python
from langchain_openai import OpenAIEmbeddings
from gigavector_langchain import GigaVectorStore

store = GigaVectorStore.from_texts(
    ["hello", "world"], OpenAIEmbeddings(),
    base_url="http://localhost:8080", api_key="secret",
)
docs = store.similarity_search("hello", k=3)
```

Requires `langchain-core` (`pip install langchain-core`). The LangChain tests
skip automatically when it is not installed.
