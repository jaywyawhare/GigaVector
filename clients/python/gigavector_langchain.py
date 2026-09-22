"""LangChain ``VectorStore`` backed by a remote GigaVector server.

Embeddings are computed client-side (by any LangChain ``Embeddings``); vectors
and metadata are stored/queried over the GigaVector REST API. The document text
is kept under a metadata key so ``similarity_search`` can reconstruct Documents.

    from langchain_openai import OpenAIEmbeddings  # any Embeddings
    from gigavector_langchain import GigaVectorStore

    store = GigaVectorStore.from_texts(
        ["hello", "world"], OpenAIEmbeddings(),
        base_url="http://localhost:8080", api_key="secret",
    )
    docs = store.similarity_search("hello", k=3)
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from gigavector import GigaVector, Distance


class GigaVectorStore(VectorStore):
    def __init__(self, client: GigaVector, embedding: Embeddings, *,
                 text_key: str = "text", distance: str = Distance.COSINE):
        self._client = client
        self._embedding = embedding
        self._text_key = text_key
        self._distance = distance

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(self, texts: Iterable[str],
                  metadatas: list[dict] | None = None, **kwargs: Any) -> list[str]:
        texts = list(texts)
        vectors = self._embedding.embed_documents(texts)
        ids: list[str] = []
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            md = dict(metadatas[i]) if metadatas else {}
            md[self._text_key] = text
            # The server stores string metadata values.
            md = {k: str(v) for k, v in md.items()}
            resp = self._client.add_vector(vec, md)
            indices = resp.get("indices") or []
            ids.append(str(indices[0]) if indices else str(i))
        return ids

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4,
                                    **kwargs: Any) -> list[Document]:
        hits = self._client.search(embedding, k, self._distance)
        docs = []
        for h in hits:
            md = dict(h.get("metadata") or {})
            text = md.pop(self._text_key, "")
            docs.append(Document(page_content=text, metadata=md))
        return docs

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return self.similarity_search_by_vector(self._embedding.embed_query(query), k, **kwargs)

    @classmethod
    def from_texts(cls, texts: list[str], embedding: Embeddings,
                   metadatas: list[dict] | None = None, *,
                   base_url: str = "http://localhost:8080",
                   api_key: str | None = None, **kwargs: Any) -> "GigaVectorStore":
        client = GigaVector(base_url, api_key=api_key)
        store = cls(client, embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store
