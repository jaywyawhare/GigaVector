"""Pure-Python HTTP server for the GigaVector dashboard.

Uses only ``http.server`` (stdlib) so no C dependencies are needed.
The server calls :class:`~gigavector.Database` methods directly.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gigavector._core import Database


class _Handler(BaseHTTPRequestHandler):
    """Request handler — dispatches to the Database held by the server."""

    server: "_DashboardHTTPServer"

    # Silence default stderr logging
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    # ── helpers ──────────────────────────────────────────────────────

    def _cors_headers(self) -> dict[str, str]:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
            "Access-Control-Max-Age": "86400",
        }

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, error: str, message: str) -> None:
        self._send_json({"error": error, "message": message}, status=status)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _parse_json_body(self) -> dict[str, Any] | None:
        raw = self._read_body()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @property
    def _db(self) -> "Database":
        return self.server.db

    # ── routing ──────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        if path == "/health":
            return self._handle_health()
        if path == "/stats":
            return self._handle_stats()
        if path == "/api/dashboard/info":
            return self._handle_dashboard_info()
        if path == "/vectors/scroll":
            return self._handle_vectors_scroll()
        m = re.match(r"^/vectors/(\d+)$", path)
        if m:
            return self._handle_vector_get(int(m.group(1)))
        if path.startswith("/dashboard"):
            return self._handle_static(path)

        self._send_error_json(404, "not_found", "Endpoint not found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        if path == "/vectors":
            return self._handle_vectors_post()
        if path == "/search":
            return self._handle_search()
        if path == "/search/range":
            return self._handle_search_range()
        if path == "/compact":
            return self._handle_compact()
        if path == "/save":
            return self._handle_save()

        self._send_error_json(404, "not_found", "Endpoint not found")

    def do_PUT(self) -> None:  # noqa: N802
        self.do_POST()

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        m = re.match(r"^/vectors/(\d+)$", path)
        if m:
            return self._handle_vector_delete(int(m.group(1)))
        self._send_error_json(404, "not_found", "Endpoint not found")

    # ── endpoint handlers ────────────────────────────────────────────

    def _handle_health(self) -> None:
        self._send_json({"status": "healthy", "vector_count": self._db.count})

    def _handle_stats(self) -> None:
        stats = self._db.get_stats()
        self._send_json({
            "total_vectors": self._db.count,
            "dimension": self._db.dimension,
            "total_inserts": stats.total_inserts,
            "total_queries": stats.total_queries,
            "total_range_queries": stats.total_range_queries,
        })

    def _handle_dashboard_info(self) -> None:
        from gigavector._core import IndexType
        idx_type = self._db._db.index_type
        try:
            idx_name = IndexType(idx_type).name
        except ValueError:
            idx_name = "UNKNOWN"
        self._send_json({
            "version": "0.8.0",
            "index_type": idx_name,
            "dimension": self._db.dimension,
            "vector_count": self._db.count,
        })

    def _handle_vectors_scroll(self) -> None:
        """Return a batch of vectors for visualization."""
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        offset = int(qs.get("offset", ["0"])[0])
        limit = int(qs.get("limit", ["200"])[0])
        limit = min(limit, 500)  # cap at 500

        total = self._db.count
        vectors = []
        end = min(offset + limit, total)
        for i in range(offset, end):
            try:
                vec = self._db.get_vector(i)
                if vec is not None:
                    vectors.append({"index": i, "data": vec})
            except Exception:
                pass
        self._send_json({
            "vectors": vectors,
            "total": total,
            "offset": offset,
            "limit": limit,
        })

    def _handle_vector_get(self, vid: int) -> None:
        vec = self._db.get_vector(vid)
        if vec is None:
            return self._send_error_json(404, "not_found", "Vector not found")
        self._send_json({"index": vid, "data": vec})

    def _handle_vector_delete(self, vid: int) -> None:
        try:
            self._db.delete_vector(vid)
            self._send_json({"success": True, "deleted_index": vid})
        except (RuntimeError, IndexError):
            self._send_error_json(404, "not_found", "Vector not found or already deleted")

    def _handle_vectors_post(self) -> None:
        body = self._parse_json_body()
        if body is None:
            return self._send_error_json(400, "invalid_json", "Invalid or missing JSON body")

        data = body.get("data")
        metadata = body.get("metadata")
        if not isinstance(data, list):
            return self._send_error_json(400, "invalid_request", "Missing 'data' array")

        try:
            self._db.add_vector(data, metadata=metadata)
            self._send_json({"success": True, "inserted": 1}, status=201)
        except Exception as e:
            self._send_error_json(500, "insert_failed", str(e))

    def _handle_search(self) -> None:
        body = self._parse_json_body()
        if body is None:
            return self._send_error_json(400, "invalid_json", "Invalid or missing JSON body")

        query = body.get("query")
        if not isinstance(query, list):
            return self._send_error_json(400, "invalid_request", "Missing 'query' array")

        k = int(body.get("k", 10))
        from gigavector._core import DistanceType
        dist_name = body.get("distance", "euclidean").upper()
        try:
            distance = DistanceType[dist_name]
        except KeyError:
            distance = DistanceType.EUCLIDEAN

        try:
            hits = self._db.search(query, k=k, distance=distance)
            results = []
            for h in hits:
                r: dict[str, Any] = {"distance": h.distance}
                if h.vector:
                    r["data"] = h.vector.data
                    r["metadata"] = h.vector.metadata
                results.append(r)
            self._send_json({"results": results, "count": len(results)})
        except Exception as e:
            self._send_error_json(500, "search_failed", str(e))

    def _handle_search_range(self) -> None:
        body = self._parse_json_body()
        if body is None:
            return self._send_error_json(400, "invalid_json", "Invalid or missing JSON body")

        query = body.get("query")
        if not isinstance(query, list):
            return self._send_error_json(400, "invalid_request", "Missing 'query' array")

        radius = float(body.get("radius", 1.0))
        max_results = int(body.get("max_results", 100))
        from gigavector._core import DistanceType
        dist_name = body.get("distance", "euclidean").upper()
        try:
            distance = DistanceType[dist_name]
        except KeyError:
            distance = DistanceType.EUCLIDEAN

        try:
            hits = self._db.range_search(query, radius=radius, max_results=max_results, distance=distance)
            results = [{"distance": h.distance} for h in hits]
            self._send_json({"results": results, "count": len(results)})
        except Exception as e:
            self._send_error_json(500, "search_failed", str(e))

    def _handle_compact(self) -> None:
        try:
            self._db.compact()
            self._send_json({"success": True, "message": "Compaction completed"})
        except Exception as e:
            self._send_error_json(500, "compact_failed", str(e))

    def _handle_save(self) -> None:
        try:
            self._db.save()
            self._send_json({"success": True, "message": "Database saved"})
        except Exception as e:
            self._send_error_json(500, "save_failed", str(e))

    # ── static file serving ──────────────────────────────────────────

    def _handle_static(self, url_path: str) -> None:
        from gigavector.dashboard import get_static_dir
        static_dir = get_static_dir()

        # Strip /dashboard prefix
        rel = url_path[len("/dashboard"):]
        if not rel or rel == "/":
            rel = "/index.html"
        if rel.startswith("/"):
            rel = rel[1:]

        # Security: reject path traversal
        if ".." in rel:
            return self._send_error_json(403, "forbidden", "Path traversal not allowed")

        filepath = os.path.join(static_dir, rel)
        if not os.path.isfile(filepath):
            return self._send_error_json(404, "not_found", "File not found")

        mime, _ = mimetypes.guess_type(filepath)
        if mime is None:
            mime = "application/octet-stream"

        with open(filepath, "rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Type", mime)
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _DashboardHTTPServer(HTTPServer):
    """HTTPServer subclass that carries a Database reference."""

    db: "Database"

    def __init__(self, db: "Database", server_address: tuple[str, int]) -> None:
        self.db = db
        super().__init__(server_address, _Handler)


class DashboardServer:
    """Pure-Python dashboard server for GigaVector.

    Uses :mod:`http.server` from the stdlib — no C HTTP library required.

    Example::

        from gigavector import Database, IndexType
        from gigavector.dashboard.server import DashboardServer

        db = Database.open(None, dimension=4, index=IndexType.FLAT)
        server = DashboardServer(db, port=6969)
        server.start()
        # Dashboard at http://localhost:6969/dashboard
        server.stop()
    """

    def __init__(self, db: "Database", *, host: str = "0.0.0.0", port: int = 6969) -> None:
        self._db = db
        self._host = host
        self._port = port
        self._httpd: _DashboardHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the dashboard server in a background daemon thread."""
        if self._httpd is not None:
            raise RuntimeError("Server is already running")
        self._httpd = _DashboardHTTPServer(self._db, (self._host, self._port))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}/dashboard"
