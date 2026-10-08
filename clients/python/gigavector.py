"""Python client for the GigaVector HTTP (REST) API.

Standalone (stdlib only) - talks to a remote gvserver over HTTP. This is
distinct from the in-process cffi bindings under ``python/src/gigavector``.

    from gigavector import GigaVector, Distance

    c = GigaVector("http://localhost:8080", api_key="secret")
    c.add_vector([1, 0, 0, 0], {"tag": "a"})
    for hit in c.search([1, 0, 0, 0], k=5, distance=Distance.COSINE):
        print(hit["id"], hit["distance"])
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class Distance:
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    HAMMING = "hamming"


class GigaVectorError(Exception):
    """Raised for non-2xx responses."""

    def __init__(self, status: int, code: str | None, message: str | None):
        super().__init__(f"gigavector {status} {code or ''}: {message or ''}".strip())
        self.status = status
        self.code = code


class GigaVector:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = urllib.request.Request(self.base_url + path, data=data,
                                     method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {}
            raise GigaVectorError(e.code, obj.get("error"), obj.get("message")) from None

    def health(self) -> dict:
        return self._request("GET", "/health")

    def add_vector(self, data, metadata: dict | None = None) -> dict:
        body: dict[str, Any] = {"data": list(data)}
        if metadata:
            body["metadata"] = metadata
        return self._request("POST", "/vectors", body)

    def search(self, query, k: int, distance: str = Distance.EUCLIDEAN) -> list:
        res = self._request("POST", "/search",
                            {"query": list(query), "k": k, "distance": distance})
        return (res or {}).get("results", [])

    def stats(self) -> dict:
        return self._request("GET", "/stats")
