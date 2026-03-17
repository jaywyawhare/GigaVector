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
    (re.compile(r"^/api/collections$"), "_handle_collections_list"),
    (re.compile(r"^/api/collections/([^/]+)$"), "_handle_collection_get"),
    (re.compile(r"^/api/cluster/info$"), "_handle_cluster_info"),
    (re.compile(r"^/api/cluster/shards$"), "_handle_cluster_shards"),
    (re.compile(r"^/api/snapshots$"), "_handle_snapshots_list"),
    (re.compile(r"^/api/schema$"), "_handle_schema_get"),
    (re.compile(r"^/openapi\.json$"), "_handle_openapi_json"),
    (re.compile(r"^/docs$"), "_handle_swagger_ui"),
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
    (re.compile(r"^/api/collections$"), "_handle_collection_create"),
    (re.compile(r"^/api/points/batch$"), "_handle_points_batch"),
    (re.compile(r"^/api/snapshots$"), "_handle_snapshot_create"),
    (re.compile(r"^/api/snapshots/(\d+)/restore$"), "_handle_snapshot_restore"),
]

_DELETE_ROUTES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^/vectors/(\d+)$"), "_handle_vector_delete"),
    (re.compile(r"^/api/namespaces/([^/]+)$"), "_handle_namespace_delete"),
    (re.compile(r"^/api/tenants/([^/]+)$"), "_handle_tenant_delete"),
    (re.compile(r"^/api/rbac/roles/([^/]+)$"), "_handle_rbac_role_delete"),
    (re.compile(r"^/api/graph/node/(\d+)$"), "_handle_graph_node_remove"),
    (re.compile(r"^/api/graph/edge/(\d+)$"), "_handle_graph_edge_remove"),
    (re.compile(r"^/api/collections/([^/]+)$"), "_handle_collection_delete"),
    (re.compile(r"^/api/snapshots/(\d+)$"), "_handle_snapshot_delete"),
]


class _Handler(BaseHTTPRequestHandler):
    """Request handler — dispatches to the Database held by the server."""

    server: "_DashboardHTTPServer"

    # Silence default stderr logging
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    _CORS = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key, X-Collection",
        "Access-Control-Max-Age": "86400",
    }

    def _cors_headers(self) -> dict[str, str]:
        return self._CORS

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
        return self._get_active_db()

    def _get_active_db(self) -> "Database":
        collection = self.headers.get("X-Collection")
        if collection:
            mgr = self.server.get_collection_mgr()
            ns = mgr.get(collection)
            if ns is not None:
                return ns  # type: ignore[return-value]
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
        end = min(offset + limit, total)
        db = self._db
        vectors = []
        for i in range(offset, end):
            try:
                vec = db.get_vector(i)
            except Exception:
                continue
            if vec is not None:
                vectors.append({"index": i, "data": vec})
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

    def _compute_shard_routing(self, shard_key: Any) -> dict[str, Any] | None:
        if shard_key is None:
            return None
        try:
            shard_mgr = self.server.get_shard_mgr()
            shard_id = shard_mgr.get_shard_for_vector(hash(str(shard_key)) % (2**31))
            return {"shard_key": shard_key, "routed_to_shard": shard_id}
        except Exception:
            return {"shard_key": shard_key, "routed_to_shard": -1}

    def _handle_vectors_post(self) -> None:
        body = self._parse_json_body()
        if body is None:
            return self._send_error_json(400, "invalid_json", "Invalid or missing JSON body")

        data = body.get("data")
        metadata = body.get("metadata")
        if not isinstance(data, list):
            return self._send_error_json(400, "invalid_request", "Missing 'data' array")

        shard_info = self._compute_shard_routing(body.get("shard_key"))

        try:
            self._get_active_db().add_vector(data, metadata=metadata)
            resp: dict[str, Any] = {"success": True, "inserted": 1}
            if shard_info:
                resp["shard_routing"] = shard_info
            self._send_json(resp, status=201)
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
        from gigavector._core import DistanceType, IndexType
        dist_name = body.get("distance", "euclidean").upper()
        try:
            distance = DistanceType[dist_name]
        except KeyError:
            distance = DistanceType.EUCLIDEAN

        filter_meta = None
        filt = body.get("filter")
        if isinstance(filt, dict) and "key" in filt and "value" in filt:
            filter_meta = (str(filt["key"]), str(filt["value"]))

        oversampling_factor = body.get("oversampling_factor")
        shard_info = self._compute_shard_routing(body.get("shard_key"))

        try:
            db = self.server.db
            if oversampling_factor is not None and hasattr(db, '_db'):
                try:
                    idx_type = db._db.index_type
                    if idx_type == int(IndexType.IVFPQ):
                        rerank_top = int(k * float(oversampling_factor))
                        hits = db.search_ivfpq_opts(query, k, distance=distance, rerank_top=rerank_top)
                        results = []
                        for h in hits:
                            r: dict[str, Any] = {"distance": h.distance}
                            if h.vector:
                                r["data"] = h.vector.data
                                r["metadata"] = h.vector.metadata
                            results.append(r)
                        resp: dict[str, Any] = {"results": results, "count": len(results)}
                        if shard_info:
                            resp["shard_routing"] = shard_info
                        self._send_json(resp)
                        return
                except Exception:
                    pass  # Fall through to normal search

            kwargs: dict[str, Any] = {"k": k, "distance": distance}
            if filter_meta:
                kwargs["filter_metadata"] = filter_meta
            hits = self._get_active_db().search(query, **kwargs)
            results = []
            for h in hits:
                r2: dict[str, Any] = {"distance": h.distance}
                if h.vector:
                    r2["data"] = h.vector.data
                    r2["metadata"] = h.vector.metadata
                results.append(r2)
            resp2: dict[str, Any] = {"results": results, "count": len(results)}
            if shard_info:
                resp2["shard_routing"] = shard_info
            self._send_json(resp2)
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
                "success": result.success,
                "path": path,
                "bytes_processed": result.bytes_processed,
                "vectors_processed": result.vectors_processed,
                "elapsed_seconds": result.elapsed_seconds,
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
        batch_data: list[list[float]] = []
        errors = 0
        db = self._db
        for v in vectors:
            data = v.get("data") if isinstance(v, dict) else v
            if not isinstance(data, list):
                errors += 1
                continue
            batch_data.append(data)
        inserted = 0
        if batch_data:
            try:
                db.add_vectors(batch_data)
                inserted = len(batch_data)
            except Exception:
                for data in batch_data:
                    try:
                        db.add_vector(data)
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

    def _handle_collections_list(self) -> None:
        mgr = self.server.get_collection_mgr()
        try:
            names = mgr.list()
            collections = []
            for name in names:
                info = mgr.get_info(name)
                if info:
                    collections.append({
                        "name": info.name,
                        "dimension": info.dimension,
                        "index_type": info.index_type,
                        "vector_count": info.vector_count,
                        "memory_bytes": info.memory_bytes,
                    })
            self._send_json({"collections": collections, "count": len(collections)})
        except Exception as e:
            self._send_error_json(500, "collection_error", str(e))

    def _handle_collection_get(self, name: str) -> None:
        mgr = self.server.get_collection_mgr()
        try:
            info = mgr.get_info(name)
            if info is None:
                return self._send_error_json(404, "not_found", f"Collection '{name}' not found")
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
            self._send_error_json(500, "collection_error", str(e))

    def _handle_collection_create(self) -> None:
        body = self._parse_json_body()
        if not body or "name" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'name'")
        from gigavector._core import CollectionConfig
        mgr = self.server.get_collection_mgr()
        try:
            config = CollectionConfig(
                name=body["name"],
                dimension=int(body.get("dimension", self.server.db.dimension)),
                index_type=body.get("index_type", "HNSW").upper(),
                max_vectors=int(body.get("max_vectors", 0)),
            )
            mgr.create(config)
            self._send_json({"success": True, "name": body["name"]}, status=201)
        except Exception as e:
            self._send_error_json(400, "collection_error", str(e))

    def _handle_collection_delete(self, name: str) -> None:
        mgr = self.server.get_collection_mgr()
        try:
            mgr.delete(name)
            self._send_json({"success": True, "deleted": name})
        except Exception as e:
            self._send_error_json(400, "collection_error", str(e))

    def _handle_points_batch(self) -> None:
        body = self._parse_json_body()
        if not body or "ids" not in body:
            return self._send_error_json(400, "invalid_request", "Missing 'ids' array")
        ids = body["ids"]
        if not isinstance(ids, list):
            return self._send_error_json(400, "invalid_request", "'ids' must be an array")
        if len(ids) > 1000:
            return self._send_error_json(400, "invalid_request", "Maximum 1000 IDs per request")
        db = self._get_active_db()
        points = []
        for vid in ids:
            try:
                vid_int = int(vid)
                vec = db.get_vector(vid_int)
                points.append({"id": vid_int, "vector": vec})
            except Exception:
                points.append({"id": vid, "vector": None})
        self._send_json({"points": points, "count": len(points)})

    def _handle_cluster_info(self) -> None:
        try:
            cluster = self.server.get_cluster()
            local = cluster.get_local_node()
            stats = cluster.get_stats()
            self._send_json({
                "local_node": {
                    "node_id": local.node_id,
                    "address": local.address,
                    "role": local.role.name if hasattr(local.role, "name") else str(local.role),
                    "state": local.state.name if hasattr(local.state, "name") else str(local.state),
                },
                "total_nodes": stats.total_nodes,
                "active_nodes": stats.active_nodes,
                "total_shards": stats.total_shards,
                "total_vectors": stats.total_vectors,
                "healthy": cluster.is_healthy(),
            })
        except Exception as e:
            self._send_error_json(500, "cluster_error", str(e))

    def _handle_cluster_shards(self) -> None:
        try:
            shard_mgr = self.server.get_shard_mgr()
            shards = shard_mgr.list_shards()
            result = []
            for s in shards:
                result.append({
                    "shard_id": s.shard_id,
                    "node_address": s.node_address,
                    "state": s.state.name if hasattr(s.state, "name") else str(s.state),
                    "vector_count": s.vector_count,
                    "replica_count": s.replica_count,
                })
            self._send_json({"shards": result, "count": len(result)})
        except Exception as e:
            self._send_error_json(500, "shard_error", str(e))

    def _handle_snapshots_list(self) -> None:
        try:
            mgr = self.server.get_snapshot_mgr()
            snapshots = mgr.list_snapshots()
            result = []
            for s in snapshots:
                result.append({
                    "snapshot_id": s.snapshot_id,
                    "timestamp_us": s.timestamp_us,
                    "vector_count": s.vector_count,
                    "label": s.label,
                })
            self._send_json({"snapshots": result, "count": len(result)})
        except Exception as e:
            self._send_error_json(500, "snapshot_error", str(e))

    def _handle_snapshot_create(self) -> None:
        body = self._parse_json_body() or {}
        label = body.get("label", "")
        try:
            mgr = self.server.get_snapshot_mgr()
            db = self.server.db
            total = db.count
            dim = db.dimension
            vectors = []
            for i in range(min(total, 10000)):
                vec = db.get_vector(i)
                if vec is not None:
                    vectors.append(vec)
            sid = mgr.create_snapshot(vectors, dim, label=label)
            self._send_json({"success": True, "snapshot_id": sid, "vector_count": len(vectors)}, status=201)
        except Exception as e:
            self._send_error_json(500, "snapshot_error", str(e))

    def _handle_snapshot_restore(self, snapshot_id: str) -> None:
        self._send_error_json(501, "not_implemented", "Snapshot restore not yet implemented")

    def _handle_snapshot_delete(self, snapshot_id: str) -> None:
        try:
            mgr = self.server.get_snapshot_mgr()
            mgr.delete_snapshot(int(snapshot_id))
            self._send_json({"success": True, "deleted": int(snapshot_id)})
        except Exception as e:
            self._send_error_json(400, "snapshot_error", str(e))

    def _handle_schema_get(self) -> None:
        try:
            schema = self.server.get_schema()
            raw_json = schema.to_json()
            import json as _json
            try:
                parsed = _json.loads(raw_json)
            except Exception:
                parsed = {}
            fields = []
            if isinstance(parsed, dict):
                for fname, finfo in parsed.items():
                    if isinstance(finfo, dict):
                        fields.append({
                            "name": fname,
                            "type": finfo.get("type", "unknown"),
                            "required": finfo.get("required", False),
                            "indexed": finfo.get("indexed", False),
                        })
            self._send_json({"fields": fields, "field_count": schema.field_count, "raw": parsed})
        except Exception as e:
            self._send_error_json(500, "schema_error", str(e))

    def _handle_openapi_json(self) -> None:
        spec = _build_openapi_spec()
        self._send_json(spec)

    def _handle_swagger_ui(self) -> None:
        html = _SWAGGER_HTML
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    _static_cache: dict[str, tuple[str, bytes]] = {}

    def _handle_static(self, url_path: str) -> None:
        rel = url_path[len("/dashboard"):]
        if not rel or rel == "/":
            rel = "index.html"
        elif rel.startswith("/"):
            rel = rel[1:]

        from gigavector.dashboard import get_static_dir
        static_dir = os.path.realpath(get_static_dir())
        filepath = os.path.realpath(os.path.join(static_dir, rel))
        if not filepath.startswith(static_dir + os.sep) and filepath != static_dir:
            return self._send_error_json(403, "forbidden", "Path traversal not allowed")

        cached = _Handler._static_cache.get(rel)
        if cached is not None:
            mime, data = cached
        else:
            if not os.path.isfile(filepath):
                return self._send_error_json(404, "not_found", "File not found")
            mime, _ = mimetypes.guess_type(filepath)
            if mime is None:
                mime = "application/octet-stream"
            with open(filepath, "rb") as f:
                data = f.read()
            _Handler._static_cache[rel] = (mime, data)

        self.send_response(200)
        self.send_header("Content-Type", mime)
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


_ROUTE_METADATA: dict[str, dict[str, Any]] = {
    "GET /health": {
        "summary": "Health check",
        "tags": ["System"],
        "responses": {"200": {"description": "Server health status", "content": {"application/json": {"schema": {"type": "object", "properties": {"status": {"type": "string"}, "vector_count": {"type": "integer"}}}}}}},
    },
    "GET /stats": {
        "summary": "Database statistics",
        "tags": ["System"],
        "responses": {"200": {"description": "Aggregate stats"}},
    },
    "GET /api/dashboard/info": {
        "summary": "Dashboard info",
        "tags": ["System"],
        "responses": {"200": {"description": "Version, index type, dimension, count"}},
    },
    "GET /api/collections": {
        "summary": "List all collections",
        "tags": ["Collections"],
        "responses": {"200": {"description": "List of collections"}},
    },
    "GET /api/collections/{name}": {
        "summary": "Get collection info",
        "tags": ["Collections"],
        "parameters": [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}],
        "responses": {"200": {"description": "Collection details"}, "404": {"description": "Not found"}},
    },
    "POST /api/collections": {
        "summary": "Create a collection",
        "tags": ["Collections"],
        "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}, "dimension": {"type": "integer"}, "index_type": {"type": "string"}, "max_vectors": {"type": "integer"}}}}}},
        "responses": {"201": {"description": "Collection created"}, "400": {"description": "Invalid request"}},
    },
    "DELETE /api/collections/{name}": {
        "summary": "Delete a collection",
        "tags": ["Collections"],
        "parameters": [{"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}],
        "responses": {"200": {"description": "Collection deleted"}},
    },
    "POST /vectors": {
        "summary": "Add a vector",
        "tags": ["Vectors"],
        "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["data"], "properties": {"data": {"type": "array", "items": {"type": "number"}}, "metadata": {"type": "object"}, "shard_key": {"type": "string"}}}}}},
        "responses": {"201": {"description": "Vector inserted"}},
    },
    "GET /vectors/{id}": {
        "summary": "Get a vector by ID",
        "tags": ["Vectors"],
        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
        "responses": {"200": {"description": "Vector data"}, "404": {"description": "Not found"}},
    },
    "DELETE /vectors/{id}": {
        "summary": "Delete a vector",
        "tags": ["Vectors"],
        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
        "responses": {"200": {"description": "Vector deleted"}},
    },
    "GET /vectors/scroll": {
        "summary": "Scroll through vectors",
        "tags": ["Vectors"],
        "parameters": [
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 200}},
        ],
        "responses": {"200": {"description": "Paginated vector list"}},
    },
    "POST /api/points/batch": {
        "summary": "Batch retrieve points by IDs",
        "tags": ["Vectors"],
        "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["ids"], "properties": {"ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 1000}}}}}},
        "responses": {"200": {"description": "Batch point data"}},
    },
    "POST /search": {
        "summary": "Search for nearest neighbors",
        "tags": ["Search"],
        "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["query"], "properties": {"query": {"type": "array", "items": {"type": "number"}}, "k": {"type": "integer", "default": 10}, "distance": {"type": "string", "enum": ["euclidean", "cosine", "dot_product", "manhattan"]}, "filter": {"type": "object"}, "oversampling_factor": {"type": "number"}, "shard_key": {"type": "string"}}}}}},
        "responses": {"200": {"description": "Search results"}},
    },
    "POST /search/range": {
        "summary": "Range search",
        "tags": ["Search"],
        "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["query"], "properties": {"query": {"type": "array", "items": {"type": "number"}}, "radius": {"type": "number"}, "distance": {"type": "string"}}}}}},
        "responses": {"200": {"description": "Range search results"}},
    },
    "GET /api/cluster/info": {
        "summary": "Cluster information",
        "tags": ["Cluster"],
        "responses": {"200": {"description": "Cluster status and node info"}},
    },
    "GET /api/cluster/shards": {
        "summary": "List shards",
        "tags": ["Cluster"],
        "responses": {"200": {"description": "Shard list"}},
    },
    "GET /api/snapshots": {
        "summary": "List snapshots",
        "tags": ["Snapshots"],
        "responses": {"200": {"description": "Snapshot list"}},
    },
    "POST /api/snapshots": {
        "summary": "Create a snapshot",
        "tags": ["Snapshots"],
        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"label": {"type": "string"}}}}}},
        "responses": {"201": {"description": "Snapshot created"}},
    },
    "POST /api/snapshots/{id}/restore": {
        "summary": "Restore a snapshot",
        "tags": ["Snapshots"],
        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
        "responses": {"200": {"description": "Restore initiated"}},
    },
    "DELETE /api/snapshots/{id}": {
        "summary": "Delete a snapshot",
        "tags": ["Snapshots"],
        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
        "responses": {"200": {"description": "Snapshot deleted"}},
    },
    "GET /api/schema": {
        "summary": "Get payload schema",
        "tags": ["Schema"],
        "responses": {"200": {"description": "Schema fields and raw JSON"}},
    },
    "GET /api/namespaces": {
        "summary": "List namespaces",
        "tags": ["Namespaces"],
        "responses": {"200": {"description": "Namespace list"}},
    },
    "POST /api/namespaces": {
        "summary": "Create a namespace",
        "tags": ["Namespaces"],
        "requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}, "dimension": {"type": "integer"}, "index_type": {"type": "string"}}}}}},
        "responses": {"201": {"description": "Namespace created"}},
    },
    "GET /openapi.json": {
        "summary": "OpenAPI specification",
        "tags": ["System"],
        "responses": {"200": {"description": "OpenAPI 3.0 JSON spec"}},
    },
    "GET /docs": {
        "summary": "Swagger UI",
        "tags": ["System"],
        "responses": {"200": {"description": "Interactive API docs (HTML)"}},
    },
}


def _regex_to_openapi_path(pattern: re.Pattern[str]) -> str:
    """Convert a regex route pattern to an OpenAPI path template."""
    raw = pattern.pattern.lstrip("^").rstrip("$")
    raw = re.sub(r"\(\\d\+\)", "{id}", raw)
    raw = re.sub(r"\(\[\^/\]\+\)", "{name}", raw)
    return raw


def _build_openapi_spec() -> dict[str, Any]:
    """Generate an OpenAPI 3.0 spec from the registered routes."""
    paths: dict[str, dict[str, Any]] = {}

    method_routes = [
        ("get", _GET_ROUTES),
        ("post", _POST_ROUTES),
        ("delete", _DELETE_ROUTES),
    ]
    for method, routes in method_routes:
        for pattern, handler_name in routes:
            path = _regex_to_openapi_path(pattern)
            key = f"{method.upper()} {path}"
            meta = _ROUTE_METADATA.get(key, {})
            op: dict[str, Any] = {
                "operationId": handler_name.lstrip("_"),
                "summary": meta.get("summary", handler_name.replace("_handle_", "").replace("_", " ").title()),
                "tags": meta.get("tags", ["Other"]),
                "responses": meta.get("responses", {"200": {"description": "Success"}}),
            }
            if "parameters" in meta:
                op["parameters"] = meta["parameters"]
            if "requestBody" in meta:
                op["requestBody"] = meta["requestBody"]
            paths.setdefault(path, {})[method] = op

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "GigaVector API",
            "version": "0.8.2",
            "description": "High-performance vector database with LLM integration",
        },
        "servers": [{"url": "/", "description": "Local server"}],
        "paths": paths,
    }


_SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GigaVector API Docs</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
SwaggerUIBundle({
  url: '/openapi.json',
  dom_id: '#swagger-ui',
  deepLinking: true,
  presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
  layout: "BaseLayout",
});
</script>
</body>
</html>
"""


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
        self._collection_mgr: Any = None
        self._shard_mgr: Any = None
        self._cluster: Any = None
        self._snapshot_mgr: Any = None
        self._schema: Any = None
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
        return self._rbac_mgr

    def get_quota_mgr(self) -> Any:
        if self._quota_mgr is None:
            with self._lock:
                if self._quota_mgr is None:
                    from gigavector._core import QuotaManager
                    self._quota_mgr = QuotaManager()
        return self._quota_mgr

    def get_collection_mgr(self) -> Any:
        if self._collection_mgr is None:
            with self._lock:
                if self._collection_mgr is None:
                    from gigavector._core import CollectionManager
                    self._collection_mgr = CollectionManager()
        return self._collection_mgr

    def get_shard_mgr(self) -> Any:
        if self._shard_mgr is None:
            with self._lock:
                if self._shard_mgr is None:
                    from gigavector._core import ShardManager
                    self._shard_mgr = ShardManager()
        return self._shard_mgr

    def get_cluster(self) -> Any:
        if self._cluster is None:
            with self._lock:
                if self._cluster is None:
                    from gigavector._core import Cluster, ClusterConfig
                    cfg = ClusterConfig()
                    self._cluster = Cluster(cfg)
        return self._cluster

    def get_snapshot_mgr(self) -> Any:
        if self._snapshot_mgr is None:
            with self._lock:
                if self._snapshot_mgr is None:
                    from gigavector._core import SnapshotManager
                    self._snapshot_mgr = SnapshotManager()
        return self._snapshot_mgr

    def get_schema(self) -> Any:
        if self._schema is None:
            with self._lock:
                if self._schema is None:
                    from gigavector._core import Schema
                    self._schema = Schema()
        return self._schema

    def close_managers(self) -> None:
        for mgr in (self._sql_engine, self._graph_db, self._namespace_mgr,
                     self._tiered_mgr, self._rbac_mgr, self._quota_mgr,
                     self._collection_mgr, self._shard_mgr, self._cluster,
                     self._snapshot_mgr, self._schema):
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
