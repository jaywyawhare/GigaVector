"""Pure-Python HTTP server for the GigaVector dashboard.

Uses only ``http.server`` (stdlib) so no C dependencies are needed.
The server calls :class:`~gigavector.Database` methods directly.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from gigavector._core import Database

_GET_ROUTES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^/health$"), "_handle_health"),
    (re.compile(r"^/stats$"), "_handle_stats"),
    (re.compile(r"^/api/dashboard/info$"), "_handle_dashboard_info"),
    (re.compile(r"^/api/detailed-stats$"), "_handle_detailed_stats"),
    (re.compile(r"^/vectors/scroll$"), "_handle_vectors_scroll"),
    (re.compile(r"^/vectors/(\d+)$"), "_handle_vector_get"),
    (re.compile(r"^/api/namespaces$"), "_handle_namespaces_list"),
    (re.compile(r"^/api/namespaces/([^/]+)/info$"), "_handle_namespace_info"),
    (re.compile(r"^/api/tenants$"), "_handle_tenants_list"),
    (re.compile(r"^/api/tenants/([^/]+)/info$"), "_handle_tenant_info"),
    (re.compile(r"^/api/quotas/([^/]+)$"), "_handle_quota_get"),
    (re.compile(r"^/api/graph/stats$"), "_handle_graph_stats"),
    (re.compile(r"^/api/graph/node/(\d+)$"), "_handle_graph_node_get"),
    (re.compile(r"^/api/graph/edges/(\d+)$"), "_handle_graph_edges_get"),
    (re.compile(r"^/api/graph/bfs$"), "_handle_graph_bfs"),
    (re.compile(r"^/api/graph/shortest-path$"), "_handle_graph_shortest_path"),
    (re.compile(r"^/api/graph/pagerank/(\d+)$"), "_handle_graph_pagerank"),
    (re.compile(r"^/api/backups/header$"), "_handle_backup_header"),
]

_POST_ROUTES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^/vectors$"), "_handle_vectors_post"),
    (re.compile(r"^/search$"), "_handle_search"),
    (re.compile(r"^/search/range$"), "_handle_search_range"),
    (re.compile(r"^/compact$"), "_handle_compact"),
    (re.compile(r"^/save$"), "_handle_save"),
    (re.compile(r"^/api/sql/execute$"), "_handle_sql_execute"),
    (re.compile(r"^/api/sql/explain$"), "_handle_sql_explain"),
    (re.compile(r"^/api/backups$"), "_handle_backup_create"),
    (re.compile(r"^/api/backups/restore$"), "_handle_backup_restore"),
    (re.compile(r"^/api/import$"), "_handle_import"),
    (re.compile(r"^/api/semantic-search$"), "_handle_semantic_search"),
    (re.compile(r"^/api/namespaces$"), "_handle_namespace_create"),
    (re.compile(r"^/api/tenants$"), "_handle_tenant_create"),
    (re.compile(r"^/api/tenants/([^/]+)/promote$"), "_handle_tenant_promote"),
    (re.compile(r"^/api/rbac/roles$"), "_handle_rbac_role_create"),
    (re.compile(r"^/api/rbac/rules$"), "_handle_rbac_rule_add"),
    (re.compile(r"^/api/rbac/assign$"), "_handle_rbac_assign"),
    (re.compile(r"^/api/quotas$"), "_handle_quota_set"),
    (re.compile(r"^/api/graph/node$"), "_handle_graph_node_add"),
    (re.compile(r"^/api/graph/edge$"), "_handle_graph_edge_add"),
]

_DELETE_ROUTES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^/vectors/(\d+)$"), "_handle_vector_delete"),
    (re.compile(r"^/api/namespaces/([^/]+)$"), "_handle_namespace_delete"),
    (re.compile(r"^/api/tenants/([^/]+)$"), "_handle_tenant_delete"),
    (re.compile(r"^/api/rbac/roles/([^/]+)$"), "_handle_rbac_role_delete"),
    (re.compile(r"^/api/graph/node/(\d+)$"), "_handle_graph_node_remove"),
    (re.compile(r"^/api/graph/edge/(\d+)$"), "_handle_graph_edge_remove"),
]


class _Handler(BaseHTTPRequestHandler):
    """Request handler — dispatches to the Database held by the server."""

    server: "_DashboardHTTPServer"

    # Silence default stderr logging
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

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

    def _qs(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query)

    def _dispatch(self, routes: list[tuple[re.Pattern[str], str]]) -> bool:
        path = self.path.split("?")[0]
        for pattern, method_name in routes:
            m = pattern.match(path)
            if m:
                handler = getattr(self, method_name)
                groups = m.groups()
                if groups:
                    handler(*groups)
                else:
                    handler()
                return True
        return False

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self._dispatch(_GET_ROUTES):
            return
        path = self.path.split("?")[0]
        if path.startswith("/dashboard"):
            return self._handle_static(path)
        self._send_error_json(404, "not_found", "Endpoint not found")

    def do_POST(self) -> None:  # noqa: N802
        if self._dispatch(_POST_ROUTES):
            return
        self._send_error_json(404, "not_found", "Endpoint not found")

    def do_PUT(self) -> None:  # noqa: N802
        self.do_POST()

    def do_DELETE(self) -> None:  # noqa: N802
        if self._dispatch(_DELETE_ROUTES):
            return
        self._send_error_json(404, "not_found", "Endpoint not found")

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
            "version": "0.8.2",
            "index_type": idx_name,
            "dimension": self._db.dimension,
            "vector_count": self._db.count,
        })

    def _handle_detailed_stats(self) -> None:
        try:
            stats = self._db.get_detailed_stats()
            self._send_json(stats)
        except Exception as e:
            self._send_error_json(500, "stats_failed", str(e))

    def _handle_vectors_scroll(self) -> None:
        qs = self._qs()
        offset = int(qs.get("offset", ["0"])[0])
        limit = int(qs.get("limit", ["200"])[0])
        limit = min(limit, 500)

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

    def _handle_vector_get(self, vid: str) -> None:
        vid_int = int(vid)
        vec = self._db.get_vector(vid_int)
        if vec is None:
            return self._send_error_json(404, "not_found", "Vector not found")
        self._send_json({"index": vid_int, "data": vec})

    def _handle_vector_delete(self, vid: str) -> None:
        vid_int = int(vid)
        try:
            self._db.delete_vector(vid_int)
            self._send_json({"success": True, "deleted_index": vid_int})
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

        # Metadata filter support
        filter_meta = None
        filt = body.get("filter")
        if isinstance(filt, dict) and "key" in filt and "value" in filt:
            filter_meta = (str(filt["key"]), str(filt["value"]))

        try:
            kwargs: dict[str, Any] = {"k": k, "distance": distance}
            if filter_meta:
                kwargs["filter_metadata"] = filter_meta
            hits = self._db.search(query, **kwargs)
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

    def _handle_sql_execute(self) -> None:
        body = self._parse_json_body()
        if not body or "query" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'query'")
        engine = self.server.get_sql_engine()
        try:
            result = engine.execute(body["query"])
            self._send_json({
                "columns": result.column_names,
                "rows": [
                    {"index": result.indices[i] if i < len(result.indices) else None,
                     "distance": result.distances[i] if i < len(result.distances) else None,
                     "metadata": result.metadata_jsons[i] if i < len(result.metadata_jsons) else None}
                    for i in range(result.row_count)
                ],
                "row_count": result.row_count,
            })
        except Exception as e:
            self._send_error_json(400, "sql_error", str(e))

    def _handle_sql_explain(self) -> None:
        body = self._parse_json_body()
        if not body or "query" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'query'")
        engine = self.server.get_sql_engine()
        try:
            plan = engine.explain(body["query"])
            self._send_json({"plan": plan})
        except Exception as e:
            self._send_error_json(400, "sql_error", str(e))

    def _handle_backup_create(self) -> None:
        from gigavector._core import backup_create
        body = self._parse_json_body() or {}
        path = body.get("path")
        if not path:
            path = os.path.join(tempfile.gettempdir(), f"gv_backup_{int(time.time())}.gvb")
        try:
            result = backup_create(self._db, path)
            self._send_json({
                "success": True,
                "path": path,
                "original_size": result.original_size,
                "compressed_size": result.compressed_size,
            })
        except Exception as e:
            self._send_error_json(500, "backup_failed", str(e))

    def _handle_backup_header(self) -> None:
        from gigavector._core import backup_read_header
        qs = self._qs()
        path = qs.get("path", [""])[0]
        if not path:
            return self._send_error_json(400, "invalid_request", "Missing 'path' query parameter")
        try:
            hdr = backup_read_header(path)
            self._send_json({
                "version": hdr.version,
                "flags": hdr.flags,
                "created_at": hdr.created_at,
                "vector_count": hdr.vector_count,
                "dimension": hdr.dimension,
                "index_type": hdr.index_type,
                "original_size": hdr.original_size,
                "compressed_size": hdr.compressed_size,
            })
        except Exception as e:
            self._send_error_json(400, "header_failed", str(e))

    def _handle_backup_restore(self) -> None:
        from gigavector._core import backup_restore_to_db
        body = self._parse_json_body()
        if not body or "path" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'path'")
        try:
            new_db = backup_restore_to_db(body["path"])
            self.server.db = new_db
            self._send_json({"success": True, "vector_count": new_db.count, "dimension": new_db.dimension})
        except Exception as e:
            self._send_error_json(500, "restore_failed", str(e))

    def _handle_import(self) -> None:
        body = self._parse_json_body()
        if not body or "vectors" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'vectors' array")
        vectors = body["vectors"]
        if not isinstance(vectors, list):
            return self._send_error_json(400, "invalid_request", "'vectors' must be an array")
        inserted = 0
        errors = 0
        for v in vectors:
            data = v.get("data") if isinstance(v, dict) else v
            metadata = v.get("metadata") if isinstance(v, dict) else None
            if not isinstance(data, list):
                errors += 1
                continue
            try:
                self._db.add_vector(data, metadata=metadata)
                inserted += 1
            except Exception:
                errors += 1
        self._send_json({"inserted": inserted, "errors": errors, "total": len(vectors)})

    def _handle_semantic_search(self) -> None:
        body = self._parse_json_body()
        if not body or "text" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'text'")
        from gigavector._core import AutoEmbedConfig, AutoEmbedProvider, AutoEmbedder, DistanceType
        try:
            provider_name = body.get("provider", "openai").upper()
            try:
                provider = AutoEmbedProvider[provider_name]
            except KeyError:
                provider = AutoEmbedProvider.OPENAI
            config = AutoEmbedConfig(
                provider=provider,
                api_key=body.get("api_key", ""),
                model_name=body.get("model_name", "text-embedding-3-small"),
                dimension=self._db.dimension,
            )
            embedder = AutoEmbedder(config)
            try:
                embedding = embedder.embed_text(body["text"])
            finally:
                embedder.close()
            k = int(body.get("k", 10))
            dist_name = body.get("distance", "cosine").upper()
            try:
                distance = DistanceType[dist_name]
            except KeyError:
                distance = DistanceType.COSINE
            hits = self._db.search(embedding, k=k, distance=distance)
            results = []
            for h in hits:
                r: dict[str, Any] = {"distance": h.distance}
                if h.vector:
                    r["data"] = h.vector.data
                    r["metadata"] = h.vector.metadata
                results.append(r)
            self._send_json({"results": results, "count": len(results), "embedding_dim": len(embedding)})
        except Exception as e:
            self._send_error_json(500, "semantic_search_failed", str(e))

    def _handle_namespaces_list(self) -> None:
        mgr = self.server.get_namespace_mgr()
        try:
            names = mgr.list()
            self._send_json({"namespaces": names, "count": len(names)})
        except Exception as e:
            self._send_error_json(500, "namespace_error", str(e))

    def _handle_namespace_create(self) -> None:
        body = self._parse_json_body()
        if not body or "name" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'name'")
        from gigavector._core import NamespaceConfig, NSIndexType
        mgr = self.server.get_namespace_mgr()
        try:
            idx_str = body.get("index_type", "HNSW").upper()
            try:
                idx_type = NSIndexType[idx_str]
            except KeyError:
                idx_type = NSIndexType.HNSW
            config = NamespaceConfig(
                name=body["name"],
                dimension=int(body.get("dimension", self._db.dimension)),
                index_type=idx_type,
                max_vectors=int(body.get("max_vectors", 0)),
            )
            mgr.create(config)
            self._send_json({"success": True, "name": body["name"]}, status=201)
        except Exception as e:
            self._send_error_json(400, "namespace_error", str(e))

    def _handle_namespace_delete(self, name: str) -> None:
        mgr = self.server.get_namespace_mgr()
        try:
            mgr.delete(name)
            self._send_json({"success": True, "deleted": name})
        except Exception as e:
            self._send_error_json(400, "namespace_error", str(e))

    def _handle_namespace_info(self, name: str) -> None:
        mgr = self.server.get_namespace_mgr()
        try:
            ns = mgr.get(name)
            if ns is None:
                return self._send_error_json(404, "not_found", f"Namespace '{name}' not found")
            info = ns.get_info()
            self._send_json({
                "name": info.name,
                "dimension": info.dimension,
                "index_type": info.index_type,
                "vector_count": info.vector_count,
                "memory_bytes": info.memory_bytes,
                "created_at": info.created_at,
                "last_modified": info.last_modified,
            })
        except Exception as e:
            self._send_error_json(500, "namespace_error", str(e))

    def _handle_tenants_list(self) -> None:
        mgr = self.server.get_tiered_mgr()
        try:
            count = mgr.tenant_count
            self._send_json({"tenant_count": count})
        except Exception as e:
            self._send_error_json(500, "tenant_error", str(e))

    def _handle_tenant_create(self) -> None:
        body = self._parse_json_body()
        if not body or "tenant_id" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'tenant_id'")
        from gigavector._core import TenantTier
        mgr = self.server.get_tiered_mgr()
        try:
            tier_name = body.get("tier", "SHARED").upper()
            try:
                tier = TenantTier[tier_name]
            except KeyError:
                tier = TenantTier.SHARED
            mgr.add_tenant(body["tenant_id"], tier)
            self._send_json({"success": True, "tenant_id": body["tenant_id"], "tier": tier.name}, status=201)
        except Exception as e:
            self._send_error_json(400, "tenant_error", str(e))

    def _handle_tenant_delete(self, tenant_id: str) -> None:
        mgr = self.server.get_tiered_mgr()
        try:
            mgr.remove_tenant(tenant_id)
            self._send_json({"success": True, "deleted": tenant_id})
        except Exception as e:
            self._send_error_json(400, "tenant_error", str(e))

    def _handle_tenant_info(self, tenant_id: str) -> None:
        mgr = self.server.get_tiered_mgr()
        try:
            info = mgr.get_info(tenant_id)
            self._send_json({
                "tenant_id": info.tenant_id,
                "tier": info.tier.name if hasattr(info.tier, "name") else str(info.tier),
                "vector_count": info.vector_count,
                "memory_bytes": info.memory_bytes,
                "created_at": info.created_at,
                "last_active": info.last_active,
                "qps_avg": info.qps_avg,
            })
        except Exception as e:
            self._send_error_json(400, "tenant_error", str(e))

    def _handle_tenant_promote(self, tenant_id: str) -> None:
        body = self._parse_json_body()
        if not body or "tier" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'tier'")
        from gigavector._core import TenantTier
        mgr = self.server.get_tiered_mgr()
        try:
            tier_name = body["tier"].upper()
            tier = TenantTier[tier_name]
            mgr.promote(tenant_id, tier)
            self._send_json({"success": True, "tenant_id": tenant_id, "new_tier": tier.name})
        except Exception as e:
            self._send_error_json(400, "tenant_error", str(e))

    def _handle_rbac_role_create(self) -> None:
        body = self._parse_json_body()
        if not body or "name" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'name'")
        mgr = self.server.get_rbac_mgr()
        try:
            mgr.create_role(body["name"])
            self._send_json({"success": True, "role": body["name"]}, status=201)
        except Exception as e:
            self._send_error_json(400, "rbac_error", str(e))

    def _handle_rbac_role_delete(self, name: str) -> None:
        mgr = self.server.get_rbac_mgr()
        try:
            mgr.delete_role(name)
            self._send_json({"success": True, "deleted": name})
        except Exception as e:
            self._send_error_json(400, "rbac_error", str(e))

    def _handle_rbac_rule_add(self) -> None:
        body = self._parse_json_body()
        if not body or "role" not in body or "resource" not in body or "permissions" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'role', 'resource', or 'permissions'")
        mgr = self.server.get_rbac_mgr()
        try:
            mgr.add_rule(body["role"], body["resource"], int(body["permissions"]))
            self._send_json({"success": True})
        except Exception as e:
            self._send_error_json(400, "rbac_error", str(e))

    def _handle_rbac_assign(self) -> None:
        body = self._parse_json_body()
        if not body or "user_id" not in body or "role" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'user_id' or 'role'")
        mgr = self.server.get_rbac_mgr()
        try:
            mgr.assign_role(body["user_id"], body["role"])
            self._send_json({"success": True})
        except Exception as e:
            self._send_error_json(400, "rbac_error", str(e))

    def _handle_quota_set(self) -> None:
        body = self._parse_json_body()
        if not body or "tenant_id" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'tenant_id'")
        from gigavector._core import QuotaConfig
        mgr = self.server.get_quota_mgr()
        try:
            config = QuotaConfig(
                max_vectors=int(body.get("max_vectors", 0)),
                max_memory_bytes=int(body.get("max_memory_bytes", 0)),
                max_qps=float(body.get("max_qps", 0.0)),
                max_ips=float(body.get("max_ips", 0.0)),
                max_storage_bytes=int(body.get("max_storage_bytes", 0)),
                max_collections=int(body.get("max_collections", 0)),
            )
            mgr.set_quota(body["tenant_id"], config)
            self._send_json({"success": True})
        except Exception as e:
            self._send_error_json(400, "quota_error", str(e))

    def _handle_quota_get(self, tenant_id: str) -> None:
        mgr = self.server.get_quota_mgr()
        try:
            usage = mgr.get_usage(tenant_id)
            self._send_json({
                "tenant_id": tenant_id,
                "vectors_used": usage.vectors_used if hasattr(usage, "vectors_used") else 0,
                "memory_used": usage.memory_used if hasattr(usage, "memory_used") else 0,
            })
        except Exception as e:
            self._send_error_json(400, "quota_error", str(e))

    def _handle_graph_stats(self) -> None:
        gdb = self.server.get_graph_db()
        self._send_json({"node_count": gdb.node_count, "edge_count": gdb.edge_count})

    def _handle_graph_node_add(self) -> None:
        body = self._parse_json_body()
        if not body or "label" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'label'")
        gdb = self.server.get_graph_db()
        try:
            nid = gdb.add_node(body["label"])
            props = body.get("properties", {})
            for pk, pv in props.items():
                gdb.set_node_prop(nid, pk, str(pv))
            self._send_json({"id": nid, "label": body["label"]}, status=201)
        except Exception as e:
            self._send_error_json(500, "graph_error", str(e))

    def _handle_graph_node_get(self, nid: str) -> None:
        gdb = self.server.get_graph_db()
        nid_int = int(nid)
        try:
            label = gdb.get_node_label(nid_int)
            if label is None:
                return self._send_error_json(404, "not_found", "Node not found")
            neighbors = gdb.get_neighbors(nid_int)
            self._send_json({
                "id": nid_int, "label": label,
                "degree": gdb.degree(nid_int),
                "neighbors": neighbors[:50],
            })
        except Exception as e:
            self._send_error_json(400, "graph_error", str(e))

    def _handle_graph_node_remove(self, nid: str) -> None:
        gdb = self.server.get_graph_db()
        try:
            gdb.remove_node(int(nid))
            self._send_json({"success": True})
        except Exception as e:
            self._send_error_json(400, "graph_error", str(e))

    def _handle_graph_edge_add(self) -> None:
        body = self._parse_json_body()
        if not body or "source" not in body or "target" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'source' or 'target'")
        gdb = self.server.get_graph_db()
        try:
            eid = gdb.add_edge(
                int(body["source"]), int(body["target"]),
                body.get("label", ""),
                float(body.get("weight", 1.0)),
            )
            self._send_json({"id": eid}, status=201)
        except Exception as e:
            self._send_error_json(500, "graph_error", str(e))

    def _handle_graph_edge_remove(self, eid: str) -> None:
        gdb = self.server.get_graph_db()
        try:
            gdb.remove_edge(int(eid))
            self._send_json({"success": True})
        except Exception as e:
            self._send_error_json(400, "graph_error", str(e))

    def _handle_graph_edges_get(self, nid: str) -> None:
        gdb = self.server.get_graph_db()
        nid_int = int(nid)
        try:
            out_ids = gdb.get_edges_out(nid_int)
            edges = []
            for eid in out_ids[:100]:
                info = gdb.get_edge(eid)
                if info:
                    edges.append(info)
            self._send_json({"node_id": nid_int, "edges": edges})
        except Exception as e:
            self._send_error_json(400, "graph_error", str(e))

    def _handle_graph_bfs(self) -> None:
        qs = self._qs()
        start = int(qs.get("start", ["0"])[0])
        max_depth = int(qs.get("max_depth", ["5"])[0])
        gdb = self.server.get_graph_db()
        try:
            visited = gdb.bfs(start, max_depth=max_depth, max_count=200)
            nodes = []
            for nid in visited:
                label = gdb.get_node_label(nid)
                nodes.append({"id": nid, "label": label or ""})
            edges = []
            seen = set(visited)
            for nid in visited:
                for nb in gdb.get_neighbors(nid, max_count=50):
                    if nb in seen and nid < nb:
                        edges.append({"source": nid, "target": nb})
            self._send_json({"nodes": nodes, "edges": edges})
        except Exception as e:
            self._send_error_json(400, "graph_error", str(e))

    def _handle_graph_shortest_path(self) -> None:
        qs = self._qs()
        from_id = int(qs.get("from", ["0"])[0])
        to_id = int(qs.get("to", ["0"])[0])
        gdb = self.server.get_graph_db()
        try:
            path = gdb.shortest_path(from_id, to_id)
            if path is None:
                return self._send_json({"path": None, "message": "No path found"})
            self._send_json({
                "node_ids": path.node_ids,
                "edge_ids": path.edge_ids,
                "total_weight": path.total_weight,
            })
        except Exception as e:
            self._send_error_json(400, "graph_error", str(e))

    def _handle_graph_pagerank(self, nid: str) -> None:
        gdb = self.server.get_graph_db()
        try:
            rank = gdb.pagerank(int(nid))
            self._send_json({"node_id": int(nid), "pagerank": rank})
        except Exception as e:
            self._send_error_json(400, "graph_error", str(e))

    def _handle_static(self, url_path: str) -> None:
        from gigavector.dashboard import get_static_dir
        static_dir = get_static_dir()

        rel = url_path[len("/dashboard"):]
        if not rel or rel == "/":
            rel = "/index.html"
        if rel.startswith("/"):
            rel = rel[1:]

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
    """HTTPServer subclass that carries a Database reference and lazy managers."""

    db: "Database"

    def __init__(self, db: "Database", server_address: tuple[str, int]) -> None:
        self.db = db
        self._sql_engine: Any = None
        self._graph_db: Any = None
        self._namespace_mgr: Any = None
        self._tiered_mgr: Any = None
        self._rbac_mgr: Any = None
        self._quota_mgr: Any = None
        self._lock = threading.Lock()
        super().__init__(server_address, _Handler)

    def get_sql_engine(self) -> Any:
        if self._sql_engine is None:
            with self._lock:
                if self._sql_engine is None:
                    from gigavector._core import SQLEngine
                    self._sql_engine = SQLEngine(self.db)
        return self._sql_engine

    def get_graph_db(self) -> Any:
        if self._graph_db is None:
            with self._lock:
                if self._graph_db is None:
                    from gigavector._core import GraphDB
                    self._graph_db = GraphDB()
        return self._graph_db

    def get_namespace_mgr(self) -> Any:
        if self._namespace_mgr is None:
            with self._lock:
                if self._namespace_mgr is None:
                    from gigavector._core import NamespaceManager
                    self._namespace_mgr = NamespaceManager()
        return self._namespace_mgr

    def get_tiered_mgr(self) -> Any:
        if self._tiered_mgr is None:
            with self._lock:
                if self._tiered_mgr is None:
                    from gigavector._core import TieredManager
                    self._tiered_mgr = TieredManager()
        return self._tiered_mgr

    def get_rbac_mgr(self) -> Any:
        if self._rbac_mgr is None:
            with self._lock:
                if self._rbac_mgr is None:
                    from gigavector._core import RBACManager
                    self._rbac_mgr = RBACManager()
                    self._rbac_mgr.init_defaults()
        return self._rbac_mgr

    def get_quota_mgr(self) -> Any:
        if self._quota_mgr is None:
            with self._lock:
                if self._quota_mgr is None:
                    from gigavector._core import QuotaManager
                    self._quota_mgr = QuotaManager()
        return self._quota_mgr

    def close_managers(self) -> None:
        for mgr in (self._sql_engine, self._graph_db, self._namespace_mgr,
                     self._tiered_mgr, self._rbac_mgr, self._quota_mgr):
            if mgr is not None and hasattr(mgr, "close"):
                try:
                    mgr.close()
                except Exception:
                    pass


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
            self._httpd.close_managers()
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
