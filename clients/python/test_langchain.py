"""Tests for the LangChain VectorStore integration (mock server + fake embeddings).

Requires ``langchain-core`` (skipped automatically if not installed).
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document
    from gigavector import GigaVector
    from gigavector_langchain import GigaVectorStore
    HAVE_LC = True
except Exception:
    HAVE_LC = False


class FakeEmbeddings:  # duck-typed Embeddings
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0, 0.0]


def _make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(obj).encode())

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n)) if n else {}
            if self.path == "/vectors":
                idx = len(state["stored"])
                state["stored"].append(body.get("metadata", {}))
                self._send({"success": True, "inserted": 1, "indices": [idx]})
            elif self.path == "/search":
                k = body.get("k", 4)
                results = [{"id": i, "distance": 0.1 * i, "data": [], "metadata": md}
                           for i, md in enumerate(state["stored"])][:k]
                self._send({"results": results, "count": len(results)})
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


@unittest.skipUnless(HAVE_LC, "langchain-core not installed")
class LangChainStoreTest(unittest.TestCase):
    def setUp(self):
        self.state = {"stored": []}
        self.srv = HTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()

    def test_is_vectorstore(self):
        from langchain_core.vectorstores import VectorStore
        store = GigaVectorStore(GigaVector(self.url), FakeEmbeddings())
        self.assertIsInstance(store, VectorStore)

    def test_add_and_search(self):
        store = GigaVectorStore(GigaVector(self.url), FakeEmbeddings())
        ids = store.add_texts(["hello world", "foo"], [{"src": "a"}, {"src": "b"}])
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(self.state["stored"]), 2)
        # text was stored under the text metadata key
        self.assertEqual(self.state["stored"][0]["text"], "hello world")

        docs = store.similarity_search("hello world", k=2)
        self.assertEqual(len(docs), 2)
        self.assertIsInstance(docs[0], Document)
        self.assertEqual(docs[0].page_content, "hello world")
        self.assertEqual(docs[0].metadata.get("src"), "a")
        self.assertNotIn("text", docs[0].metadata)  # promoted to page_content

    def test_from_texts(self):
        store = GigaVectorStore.from_texts(
            ["a", "bb"], FakeEmbeddings(), base_url=self.url)
        docs = store.similarity_search("a", k=2)
        self.assertEqual({d.page_content for d in docs}, {"a", "bb"})


if __name__ == "__main__":
    unittest.main()
