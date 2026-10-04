#include "admin/cluster.h"
#include "core/memory.h"
#include "admin/namespace.h"
#include "admin/replication.h"
#include "admin/shard.h"
#include "api/server.h"
#include "api/grpc.h"
#include "core/bloom.h"
#include "core/types.h"
#include "features/context_graph.h"
#include "features/graph_db.h"
#include "features/knowledge_graph.h"
#include "features/sql.h"
#include "multimodal/learned_sparse.h"
#include "search/phased_ranking.h"
#include "index/kdtree.h"
#include "index/diskann.h"
#include "multimodal/bm25.h"
#include "multimodal/embedding.h"
#include "multimodal/llm.h"
#include "schema/metadata.h"
#include "schema/vector.h"
#include "search/mmr.h"
#include "specialized/gpu.h"
#include "storage/backup.h"
#include "storage/database.h"
#include "storage/transaction.h"
#include "storage/bulk_import.h"
#include "search/recall.h"
#include "storage/memory_extraction.h"
#include "storage/memory_layer.h"
#include "features/knowledge_graph.h"
#include "storage/snapshot.h"
#include "storage/wal.h"
#include "storage/posting_list.h"
#include "features/geo.h"
#include "features/cypher.h"
#include "features/recommend.h"
#include "features/json_index.h"
#include "specialized/dedup.h"
#include "specialized/conditional.h"
#include "specialized/named_vectors.h"
#include "admin/ttl.h"
#include "admin/timetravel.h"
#include "admin/webhook.h"
#include "admin/cdc.h"
#include "admin/versioning.h"
#include "admin/tracing.h"
#include "admin/migration.h"
#include "api/alias.h"

#include <stdlib.h>

GV_Database *gv_db_open(const char *filepath, size_t dimension,
                        GV_IndexType index_type) {
  return db_open(filepath, dimension, index_type);
}

GV_Database *gv_db_open_with_hnsw_config(const char *filepath, size_t dimension,
                                         GV_IndexType index_type,
                                         const GV_HNSWConfig *hnsw_config) {
  return db_open_with_hnsw_config(filepath, dimension, index_type, hnsw_config);
}

GV_Database *gv_db_open_with_ivfpq_config(const char *filepath,
                                          size_t dimension,
                                          GV_IndexType index_type,
                                          const GV_IVFPQConfig *ivfpq_config) {
  return db_open_with_ivfpq_config(filepath, dimension, index_type,
                                   ivfpq_config);
}

void gv_db_close(GV_Database *db) { db_close(db); }

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension) {
  return db_add_vector(db, data, dimension);
}

int gv_db_add_vector_with_metadata(GV_Database *db, const float *data,
                                   size_t dimension, const char *metadata_key,
                                   const char *metadata_value) {
  return db_add_vector_with_metadata(db, data, dimension, metadata_key,
                                     metadata_value);
}

int gv_db_add_vector_with_rich_metadata(GV_Database *db, const float *data,
                                        size_t dimension,
                                        const char *const *metadata_keys,
                                        const char *const *metadata_values,
                                        size_t metadata_count) {
  return db_add_vector_with_rich_metadata(db, data, dimension, metadata_keys,
                                          metadata_values, metadata_count);
}

int gv_db_save(const GV_Database *db, const char *filepath) {
  return db_save(db, filepath);
}

/* ---- transactions (MVCC) ---- */
GV_DBTxn *gv_db_begin(GV_Database *db) { return db_begin(db); }
int gv_db_txn_add_vector(GV_DBTxn *txn, const float *data, size_t dimension) {
  return db_txn_add_vector(txn, data, dimension);
}
int gv_db_txn_delete(GV_DBTxn *txn, size_t vector_index) { return db_txn_delete(txn, vector_index); }
int gv_db_txn_search(GV_DBTxn *txn, const float *query, size_t k,
                     GV_SearchResult *results, GV_DistanceType metric) {
  return db_txn_search(txn, query, k, results, metric);
}
int gv_db_commit(GV_DBTxn *txn) { return db_commit(txn); }
int gv_db_rollback(GV_DBTxn *txn) { return db_rollback(txn); }
uint64_t gv_db_txn_gc(GV_Database *db, uint64_t safe_below) { return db_txn_gc(db, safe_below); }

/* ---- WiscKey value store (db-level opt-in KV separation) ---- */
int gv_db_value_store_enable(GV_Database *db, const char *path) { return db_value_store_enable(db, path); }
int gv_db_value_store_put(GV_Database *db, uint64_t key, const void *value, size_t len) { return db_value_store_put(db, key, value, len); }
int gv_db_value_store_get(GV_Database *db, uint64_t key, void **value_out, size_t *len_out) { return db_value_store_get(db, key, value_out, len_out); }
int gv_db_value_store_delete(GV_Database *db, uint64_t key) { return db_value_store_delete(db, key); }
int gv_db_value_store_gc(GV_Database *db) { return db_value_store_gc(db); }

/* ---- warm-up + recall eval + bulk import ---- */
int gv_db_warmup(GV_Database *db) { return db_warmup(db); }
double gv_db_evaluate_recall(const GV_Database *db, const float *queries, size_t nq,
                             size_t dim, size_t k, GV_DistanceType metric) {
  GV_RecallReport rep; if (db_evaluate_recall(db, queries, nq, dim, k, metric, &rep) != 0) return -1.0;
  return rep.recall;
}
int gv_db_import_csv(GV_Database *db, const char *path, char delimiter,
                     int has_header, int id_column) {
  return db_import_csv(db, path, delimiter, has_header, id_column, NULL);
}
int gv_db_import_jsonl(GV_Database *db, const char *path) {
  return db_import_jsonl(db, path, NULL);
}

int gv_db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type) {
  return db_search(db, query_data, k, results, distance_type);
}

int gv_db_search_filtered(const GV_Database *db, const float *query_data,
                          size_t k, GV_SearchResult *results,
                          GV_DistanceType distance_type, const char *filter_key,
                          const char *filter_value) {
  return db_search_filtered(db, query_data, k, results, distance_type,
                            filter_key, filter_value);
}

int gv_db_search_batch(const GV_Database *db, const float *queries,
                       size_t qcount, size_t k, GV_SearchResult *results,
                       GV_DistanceType distance_type) {
  return db_search_batch(db, queries, qcount, k, results, distance_type);
}

int gv_db_ivfpq_train(GV_Database *db, const float *data, size_t count,
                      size_t dimension) {
  return db_ivfpq_train(db, data, count, dimension);
}

void gv_replication_config_init(GV_ReplicationConfig *config) {
  replication_config_init(config);
}

GV_ReplicationManager *
gv_replication_create(GV_Database *db, const GV_ReplicationConfig *config) {
  return replication_create(db, config);
}

void gv_replication_destroy(GV_ReplicationManager *mgr) {
  replication_destroy(mgr);
}

int gv_replication_start(GV_ReplicationManager *mgr) {
  return replication_start(mgr);
}

int gv_replication_stop(GV_ReplicationManager *mgr) {
  return replication_stop(mgr);
}

int gv_replication_add_follower(GV_ReplicationManager *mgr, const char *node_id,
                                const char *address) {
  return replication_add_follower(mgr, node_id, address);
}

int gv_replication_sync_commit(GV_ReplicationManager *mgr,
                               uint32_t timeout_ms) {
  return replication_sync_commit(mgr, timeout_ms);
}

int gv_replication_leader_append_wal(GV_ReplicationManager *mgr,
                                     uint64_t entry_delta,
                                     uint64_t byte_delta) {
  return replication_leader_append_wal(mgr, entry_delta, byte_delta);
}

int gv_wal_truncate(GV_WAL *wal) { return wal_truncate(wal); }

GV_Database *gv_db_open_with_ivfflat_config(const char *filepath,
                                            size_t dimension,
                                            GV_IndexType index_type,
                                            const GV_IVFFlatConfig *config) {
  return db_open_with_ivfflat_config(filepath, dimension, index_type, config);
}

GV_Database *gv_db_open_with_ivfdisk_config(const char *filepath,
                                            size_t dimension,
                                            GV_IndexType index_type,
                                            const GV_IVFDiskConfig *config) {
  return db_open_with_ivfdisk_config(filepath, dimension, index_type, config);
}

GV_Database *gv_db_open_with_ivfsq8_config(const char *filepath,
                                           size_t dimension,
                                           GV_IndexType index_type,
                                           const GV_IVFSQ8Config *config) {
  return db_open_with_ivfsq8_config(filepath, dimension, index_type, config);
}

GV_Database *gv_db_open_with_ivfturboquant_config(const char *filepath,
                                                  size_t dimension,
                                                  GV_IndexType index_type,
                                                  const GV_IVFTurboQuantConfig *config) {
  return db_open_with_ivfturboquant_config(filepath, dimension, index_type, config);
}

GV_Database *gv_db_open_with_pq_config(const char *filepath, size_t dimension,
                                       GV_IndexType index_type,
                                       const GV_PQConfig *config) {
  return db_open_with_pq_config(filepath, dimension, index_type, config);
}

GV_Database *gv_db_open_with_lsh_config(const char *filepath, size_t dimension,
                                        GV_IndexType index_type,
                                        const GV_LSHConfig *config) {
  return db_open_with_lsh_config(filepath, dimension, index_type, config);
}

GV_Database *gv_db_open_from_memory(const void *data, size_t size,
                                    size_t dimension, GV_IndexType index_type) {
  return db_open_from_memory(data, size, dimension, index_type);
}

GV_Database *gv_db_open_mmap(const char *filepath, size_t dimension,
                             GV_IndexType index_type) {
  return db_open_mmap(filepath, dimension, index_type);
}

GV_IndexType gv_index_suggest(size_t dimension, size_t expected_count) {
  return index_suggest(dimension, expected_count);
}

size_t gv_index_suggest_bytes_per_vector(size_t dimension, size_t metadata_bytes_per_vector) {
  return index_suggest_bytes_per_vector(dimension, metadata_bytes_per_vector);
}

GV_IndexType gv_index_suggest_with_budget(size_t dimension, size_t expected_count,
                                          size_t max_memory_bytes, size_t bytes_per_vector) {
  return index_suggest_with_budget(dimension, expected_count, max_memory_bytes, bytes_per_vector);
}

void gv_db_get_stats(const GV_Database *db, GV_DBStats *out) {
  db_get_stats(db, out);
}

void gv_db_set_cosine_normalized(GV_Database *db, int enabled) {
  db_set_cosine_normalized(db, enabled);
}

int gv_db_delete_vector_by_index(GV_Database *db, size_t vector_index) {
  return db_delete_vector_by_index(db, vector_index);
}

int gv_db_update_vector(GV_Database *db, size_t vector_index,
                        const float *new_data, size_t dimension) {
  return db_update_vector(db, vector_index, new_data, dimension);
}

int gv_db_update_vector_metadata(GV_Database *db, size_t vector_index,
                                 const char *const *metadata_keys,
                                 const char *const *metadata_values,
                                 size_t metadata_count) {
  return db_update_vector_metadata(db, vector_index, metadata_keys,
                                   metadata_values, metadata_count);
}

int gv_db_ivfflat_train(GV_Database *db, const float *data, size_t count,
                        size_t dimension) {
  return db_ivfflat_train(db, data, count, dimension);
}

int gv_db_ivfdisk_train(GV_Database *db, const float *data, size_t count,
                        size_t dimension) {
  return db_ivfdisk_train(db, data, count, dimension);
}

int gv_db_ivfsq8_train(GV_Database *db, const float *data, size_t count,
                       size_t dimension) {
  return db_ivfsq8_train(db, data, count, dimension);
}

int gv_db_ivfturboquant_train(GV_Database *db, const float *data, size_t count,
                              size_t dimension) {
  return db_ivfturboquant_train(db, data, count, dimension);
}

int gv_db_pq_train(GV_Database *db, const float *data, size_t count,
                   size_t dimension) {
  return db_pq_train(db, data, count, dimension);
}

int gv_db_add_vectors(GV_Database *db, const float *data, size_t count,
                      size_t dimension) {
  return db_add_vectors(db, data, count, dimension);
}

int gv_db_add_vectors_with_metadata(GV_Database *db, const float *data,
                                    const char *const *keys,
                                    const char *const *values, size_t count,
                                    size_t dimension) {
  return db_add_vectors_with_metadata(db, data, keys, values, count, dimension);
}

int gv_db_add_sparse_vector(GV_Database *db, const uint32_t *indices,
                            const float *values, size_t nnz, size_t dimension,
                            const char *metadata_key,
                            const char *metadata_value) {
  return db_add_sparse_vector(db, indices, values, nnz, dimension, metadata_key,
                              metadata_value);
}

int gv_db_upsert(GV_Database *db, size_t vector_index, const float *data,
                 size_t dimension) {
  return db_upsert(db, vector_index, data, dimension);
}

int gv_db_upsert_with_metadata(GV_Database *db, size_t vector_index,
                               const float *data, size_t dimension,
                               const char *const *metadata_keys,
                               const char *const *metadata_values,
                               size_t metadata_count) {
  return db_upsert_with_metadata(db, vector_index, data, dimension,
                                 metadata_keys, metadata_values,
                                 metadata_count);
}

int gv_db_delete_vectors(GV_Database *db, const size_t *indices, size_t count) {
  return db_delete_vectors(db, indices, count);
}

int gv_db_search_with_filter_expr(const GV_Database *db,
                                  const float *query_data, size_t k,
                                  GV_SearchResult *results,
                                  GV_DistanceType distance_type,
                                  const char *filter_expr) {
  return db_search_with_filter_expr(db, query_data, k, results, distance_type,
                                    filter_expr);
}

int gv_db_search_ivfpq_opts(const GV_Database *db, const float *query_data,
                            size_t k, GV_SearchResult *results,
                            GV_DistanceType distance_type,
                            size_t nprobe_override, size_t rerank_top) {
  return db_search_ivfpq_opts(db, query_data, k, results, distance_type,
                              nprobe_override, rerank_top);
}

int gv_db_search_sparse(const GV_Database *db, const uint32_t *indices,
                        const float *values, size_t nnz, size_t k,
                        GV_SearchResult *results,
                        GV_DistanceType distance_type) {
  return db_search_sparse(db, indices, values, nnz, k, results, distance_type);
}

int gv_db_range_search(const GV_Database *db, const float *query_data,
                       float radius, GV_SearchResult *results,
                       size_t max_results, GV_DistanceType distance_type) {
  return db_range_search(db, query_data, radius, results, max_results,
                         distance_type);
}

int gv_db_range_search_filtered(const GV_Database *db, const float *query_data,
                                float radius, GV_SearchResult *results,
                                size_t max_results,
                                GV_DistanceType distance_type,
                                const char *filter_key,
                                const char *filter_value) {
  return db_range_search_filtered(db, query_data, radius, results, max_results,
                                  distance_type, filter_key, filter_value);
}

int gv_db_search_with_params(const GV_Database *db, const float *query_data,
                             size_t k, GV_SearchResult *results,
                             GV_DistanceType distance_type,
                             const GV_SearchParams *params) {
  return db_search_with_params(db, query_data, k, results, distance_type,
                               params);
}

int gv_db_scroll(const GV_Database *db, size_t offset, size_t limit,
                 GV_ScrollResult *results) {
  return db_scroll(db, offset, limit, results);
}

void gv_db_set_exact_search_threshold(GV_Database *db, size_t threshold) {
  db_set_exact_search_threshold(db, threshold);
}

void gv_db_set_force_exact_search(GV_Database *db, int enabled) {
  db_set_force_exact_search(db, enabled);
}

int gv_db_set_resource_limits(GV_Database *db,
                              const GV_ResourceLimits *limits) {
  return db_set_resource_limits(db, limits);
}

void gv_db_get_resource_limits(const GV_Database *db,
                               GV_ResourceLimits *limits) {
  db_get_resource_limits(db, limits);
}

size_t gv_db_get_memory_usage(const GV_Database *db) {
  return db_get_memory_usage(db);
}

size_t gv_db_get_concurrent_operations(const GV_Database *db) {
  return db_get_concurrent_operations(db);
}

int gv_db_start_background_compaction(GV_Database *db) {
  return db_start_background_compaction(db);
}

void gv_db_stop_background_compaction(GV_Database *db) {
  db_stop_background_compaction(db);
}

int gv_db_compact(GV_Database *db) { return db_compact(db); }

void gv_db_set_compaction_interval(GV_Database *db, size_t interval_sec) {
  db_set_compaction_interval(db, interval_sec);
}

void gv_db_set_wal_compaction_threshold(GV_Database *db,
                                        size_t threshold_bytes) {
  db_set_wal_compaction_threshold(db, threshold_bytes);
}

void gv_db_set_deleted_ratio_threshold(GV_Database *db, double ratio) {
  db_set_deleted_ratio_threshold(db, ratio);
}

int gv_db_get_detailed_stats(const GV_Database *db, GV_DetailedStats *out) {
  return db_get_detailed_stats(db, out);
}

void gv_db_free_detailed_stats(GV_DetailedStats *stats) {
  db_free_detailed_stats(stats);
}

int gv_db_health_check(const GV_Database *db) { return db_health_check(db); }

void gv_db_record_latency(GV_Database *db, uint64_t latency_us, int is_insert) {
  db_record_latency(db, latency_us, is_insert);
}

void gv_db_record_recall(GV_Database *db, double recall) {
  db_record_recall(db, recall);
}

size_t gv_database_count(const GV_Database *db) { return database_count(db); }

size_t gv_database_dimension(const GV_Database *db) {
  return database_dimension(db);
}

const float *gv_database_get_vector(const GV_Database *db, size_t index) {
  return database_get_vector(db, index);
}

int gv_db_export_json(const GV_Database *db, const char *filepath) {
  return db_export_json(db, filepath);
}

int gv_db_import_json(GV_Database *db, const char *filepath) {
  return db_import_json(db, filepath);
}

GV_Vector *gv_vector_create_from_data(size_t dimension, const float *data) {
  return vector_create_from_data(dimension, data);
}

int gv_vector_set_metadata(GV_Vector *vector, const char *key,
                           const char *value) {
  return vector_set_metadata(vector, key, value);
}

void gv_vector_destroy(GV_Vector *vector) { vector_destroy(vector); }

/* (Removed gv_kdtree_insert: it could never work — the real kdtree_insert
   operates on SoA storage + a vector index, not a standalone GV_Vector node,
   so the binding was a permanent -1 stub. Use a Database with
   IndexType.KDTREE for KD-tree indexing.) */

int gv_wal_append_insert(GV_WAL *wal, const float *data, size_t dimension,
                         const char *metadata_key, const char *metadata_value) {
  return wal_append_insert(wal, data, dimension, metadata_key, metadata_value);
}

int gv_wal_append_insert_rich(GV_WAL *wal, const float *data, size_t dimension,
                              const char *const *metadata_keys,
                              const char *const *metadata_values,
                              size_t metadata_count) {
  return wal_append_insert_rich(wal, data, dimension, metadata_keys,
                                metadata_values, metadata_count);
}

GV_LLM *gv_llm_create(const GV_LLMConfig *config) { return llm_create(config); }

void gv_llm_destroy(GV_LLM *llm) { llm_destroy(llm); }

int gv_llm_generate_response(GV_LLM *llm, const GV_LLMMessage *messages,
                             size_t message_count, const char *response_format,
                             GV_LLMResponse *response) {
  return llm_generate_response(llm, messages, message_count, response_format,
                               response);
}

void gv_llm_response_free(GV_LLMResponse *response) {
  llm_response_free(response);
}

void gv_llm_message_free(GV_LLMMessage *message) { llm_message_free(message); }

void gv_llm_messages_free(GV_LLMMessage *messages, size_t count) {
  llm_messages_free(messages, count);
}

const char *gv_llm_get_last_error(GV_LLM *llm) {
  return llm_get_last_error(llm);
}

const char *gv_llm_error_string(int error_code) {
  return llm_error_string(error_code);
}

GV_EmbeddingService *
gv_embedding_service_create(const GV_EmbeddingConfig *config) {
  return embedding_service_create(config);
}

void gv_embedding_service_destroy(GV_EmbeddingService *service) {
  embedding_service_destroy(service);
}

int gv_embedding_generate(GV_EmbeddingService *service, const char *text,
                          size_t *embedding_dim, float **embedding) {
  return embedding_generate(service, text, embedding_dim, embedding);
}

int gv_embedding_generate_batch(GV_EmbeddingService *service,
                                const char **texts, size_t text_count,
                                size_t **embedding_dims, float ***embeddings) {
  return embedding_generate_batch(service, texts, text_count, embedding_dims,
                                  embeddings);
}

GV_EmbeddingConfig gv_embedding_config_default(void) {
  return embedding_config_default();
}

void gv_embedding_config_free(GV_EmbeddingConfig *config) {
  embedding_config_free(config);
}

GV_EmbeddingCache *gv_embedding_cache_create(size_t max_size) {
  return embedding_cache_create(max_size);
}

void gv_embedding_cache_destroy(GV_EmbeddingCache *cache) {
  embedding_cache_destroy(cache);
}

int gv_embedding_cache_get(GV_EmbeddingCache *cache, const char *text,
                           size_t *embedding_dim, const float **embedding) {
  return embedding_cache_get(cache, text, embedding_dim, embedding);
}

int gv_embedding_cache_put(GV_EmbeddingCache *cache, const char *text,
                           size_t embedding_dim, const float *embedding) {
  return embedding_cache_put(cache, text, embedding_dim, embedding);
}

void gv_embedding_cache_clear(GV_EmbeddingCache *cache) {
  embedding_cache_clear(cache);
}

void gv_embedding_cache_stats(GV_EmbeddingCache *cache, size_t *size,
                              uint64_t *hits, uint64_t *misses) {
  embedding_cache_stats(cache, size, hits, misses);
}

GV_ContextGraph *gv_context_graph_create(const GV_ContextGraphConfig *config) {
  return context_graph_create(config);
}

void gv_context_graph_destroy(GV_ContextGraph *graph) {
  context_graph_destroy(graph);
}

int gv_context_graph_extract(GV_ContextGraph *graph, const char *text,
                             const char *user_id, const char *agent_id,
                             const char *run_id, GV_GraphEntity **entities,
                             size_t *entity_count,
                             GV_GraphRelationship **relationships,
                             size_t *relationship_count) {
  return context_graph_extract(graph, text, user_id, agent_id, run_id, entities,
                               entity_count, relationships, relationship_count);
}

int gv_context_graph_add_entities(GV_ContextGraph *graph,
                                  const GV_GraphEntity *entities,
                                  size_t entity_count) {
  return context_graph_add_entities(graph, entities, entity_count);
}

int gv_context_graph_add_relationships(
    GV_ContextGraph *graph, const GV_GraphRelationship *relationships,
    size_t relationship_count) {
  return context_graph_add_relationships(graph, relationships,
                                         relationship_count);
}

int gv_context_graph_search(GV_ContextGraph *graph,
                            const float *query_embedding, size_t embedding_dim,
                            const char *user_id, const char *agent_id,
                            const char *run_id, GV_GraphQueryResult *results,
                            size_t max_results) {
  return context_graph_search(graph, query_embedding, embedding_dim, user_id,
                              agent_id, run_id, results, max_results);
}

int gv_context_graph_get_related(GV_ContextGraph *graph, const char *entity_id,
                                 size_t max_depth, GV_GraphQueryResult *results,
                                 size_t max_results) {
  return context_graph_get_related(graph, entity_id, max_depth, results,
                                   max_results);
}

int gv_context_graph_delete_entities(GV_ContextGraph *graph,
                                     const char **entity_ids,
                                     size_t entity_count) {
  return context_graph_delete_entities(graph, entity_ids, entity_count);
}

int gv_context_graph_delete_relationships(GV_ContextGraph *graph,
                                          const char **relationship_ids,
                                          size_t relationship_count) {
  return context_graph_delete_relationships(graph, relationship_ids,
                                            relationship_count);
}

void gv_graph_entity_free(GV_GraphEntity *entity) { graph_entity_free(entity); }

void gv_graph_relationship_free(GV_GraphRelationship *relationship) {
  graph_relationship_free(relationship);
}

void gv_graph_query_result_free(GV_GraphQueryResult *result) {
  graph_query_result_free(result);
}

GV_ContextGraphConfig gv_context_graph_config_default(void) {
  return context_graph_config_default();
}

GV_MemoryLayerConfig gv_memory_layer_config_default(void) {
  return memory_layer_config_default();
}

GV_MemoryLayer *gv_memory_layer_create(GV_Database *db,
                                       const GV_MemoryLayerConfig *config) {
  return memory_layer_create(db, config);
}

void gv_memory_layer_destroy(GV_MemoryLayer *layer) {
  memory_layer_destroy(layer);
}

char *gv_memory_add(GV_MemoryLayer *layer, const char *content,
                    const float *embedding, GV_MemoryMetadata *metadata) {
  return memory_add(layer, content, embedding, metadata, NULL);
}

char *gv_memory_add_opts(GV_MemoryLayer *layer, const char *content,
                         const float *embedding, GV_MemoryMetadata *metadata,
                         int ingest_context) {
  return memory_add_opts(layer, content, embedding, metadata, NULL, ingest_context);
}

char **gv_memory_extract_from_conversation(GV_MemoryLayer *layer,
                                           const char *conversation,
                                           const char *conversation_id,
                                           float **embeddings,
                                           size_t *memory_count) {
  return memory_extract_from_conversation(layer, conversation, conversation_id,
                                          embeddings, memory_count);
}

char **gv_memory_extract_from_text(GV_MemoryLayer *layer, const char *text,
                                   const char *source, float **embeddings,
                                   size_t *memory_count) {
  return memory_extract_from_text(layer, text, source, embeddings,
                                  memory_count);
}

int gv_memory_extract_candidates_from_conversation_llm(
    GV_LLM *llm, const char *conversation, const char *conversation_id,
    int is_agent_memory, const char *custom_prompt, void *candidates,
    size_t max_candidates, size_t *actual_count) {
  return memory_extract_candidates_from_conversation_llm(
      llm, conversation, conversation_id, is_agent_memory, custom_prompt,
      (GV_MemoryCandidate *)candidates, max_candidates, actual_count);
}

int gv_memory_consolidate(GV_MemoryLayer *layer, double threshold,
                          int strategy) {
  return memory_consolidate(layer, threshold, strategy);
}

int gv_memory_search(GV_MemoryLayer *layer, const float *query_embedding,
                     size_t k, GV_MemoryResult *results,
                     GV_DistanceType distance_type) {
  return memory_search(layer, query_embedding, k, results, distance_type);
}

int gv_memory_search_filtered(GV_MemoryLayer *layer,
                              const float *query_embedding, size_t k,
                              GV_MemoryResult *results,
                              GV_DistanceType distance_type, int memory_type,
                              const char *source, time_t min_timestamp,
                              time_t max_timestamp) {
  return memory_search_filtered(layer, query_embedding, k, results,
                                distance_type, memory_type, source,
                                min_timestamp, max_timestamp);
}

GV_MemorySearchOptions gv_memory_search_options_default(void) {
  return memory_search_options_default();
}

int gv_memory_search_advanced(GV_MemoryLayer *layer,
                              const float *query_embedding, size_t k,
                              GV_MemoryResult *results,
                              GV_DistanceType distance_type,
                              const GV_MemorySearchOptions *options) {
  return memory_search_advanced(layer, query_embedding, k, results,
                                distance_type, options);
}

int gv_memory_get_related(GV_MemoryLayer *layer, const char *memory_id,
                          size_t k, GV_MemoryResult *results) {
  return memory_get_related(layer, memory_id, k, results);
}

int gv_memory_get(GV_MemoryLayer *layer, const char *memory_id,
                  GV_MemoryResult *result) {
  return memory_get(layer, memory_id, result);
}

int gv_memory_update(GV_MemoryLayer *layer, const char *memory_id,
                     const float *new_embedding,
                     GV_MemoryMetadata *new_metadata) {
  return memory_update(layer, memory_id, new_embedding, new_metadata);
}

int gv_memory_delete(GV_MemoryLayer *layer, const char *memory_id) {
  return memory_delete(layer, memory_id);
}

void gv_memory_result_free(GV_MemoryResult *result) {
  memory_result_free(result);
}

void gv_memory_metadata_free(GV_MemoryMetadata *metadata) {
  memory_metadata_free(metadata);
}

int gv_memory_link_create(GV_MemoryLayer *layer, const char *source_id,
                          const char *target_id, GV_MemoryLinkType link_type,
                          float strength, const char *reason) {
  return memory_link_create(layer, source_id, target_id, link_type, strength,
                            reason);
}

int gv_memory_link_remove(GV_MemoryLayer *layer, const char *source_id,
                          const char *target_id) {
  return memory_link_remove(layer, source_id, target_id);
}

int gv_memory_link_get(GV_MemoryLayer *layer, const char *memory_id,
                       GV_MemoryLink *links, size_t max_links) {
  return memory_link_get(layer, memory_id, links, max_links);
}

void gv_memory_link_free(GV_MemoryLink *link) { memory_link_free(link); }

int gv_memory_record_access(GV_MemoryLayer *layer, const char *memory_id,
                            float relevance) {
  return memory_record_access(layer, memory_id, relevance);
}

int gv_memory_layer_extract_context_entities(GV_MemoryLayer *layer, const char *text,
                                             char ***out_names, size_t *out_count) {
  return memory_layer_extract_context_entities(layer, text, out_names, out_count);
}

void gv_memory_layer_free_context_entity_names(char **names, size_t count) {
  memory_layer_free_context_entity_names(names, count);
}

int gv_gpu_available(void) { return gpu_available(); }

int gv_gpu_device_count(void) { return gpu_device_count(); }

int gv_gpu_get_device_info(int device_id, GV_GPUDeviceInfo *info) {
  return gpu_get_device_info(device_id, info);
}

void gv_gpu_config_init(GV_GPUConfig *config) { gpu_config_init(config); }

GV_GPUContext *gv_gpu_create(const GV_GPUConfig *config) {
  return gpu_create(config);
}

void gv_gpu_destroy(GV_GPUContext *ctx) { gpu_destroy(ctx); }

int gv_gpu_synchronize(GV_GPUContext *ctx) { return gpu_synchronize(ctx); }

GV_GPUIndex *gv_gpu_index_create(GV_GPUContext *ctx, const float *vectors,
                                 size_t count, size_t dimension) {
  return gpu_index_create(ctx, vectors, count, dimension);
}

GV_GPUIndex *gv_gpu_index_from_db(GV_GPUContext *ctx, GV_Database *db) {
  return gpu_index_from_db(ctx, db);
}

int gv_gpu_index_add(GV_GPUIndex *index, const float *vectors, size_t count) {
  return gpu_index_add(index, vectors, count);
}

int gv_gpu_index_remove(GV_GPUIndex *index, const size_t *indices,
                        size_t count) {
  return gpu_index_remove(index, indices, count);
}

int gv_gpu_index_update(GV_GPUIndex *index, const size_t *indices,
                        const float *vectors, size_t count) {
  return gpu_index_update(index, indices, vectors, count);
}

int gv_gpu_index_info(GV_GPUIndex *index, size_t *count, size_t *dimension,
                      size_t *memory_usage) {
  return gpu_index_info(index, count, dimension, memory_usage);
}

void gv_gpu_index_destroy(GV_GPUIndex *index) { gpu_index_destroy(index); }

int gv_gpu_compute_distances(GV_GPUContext *ctx, const float *queries,
                             size_t num_queries, const float *database,
                             size_t num_vectors, size_t dimension,
                             GV_GPUDistanceMetric metric, float *distances) {
  return gpu_compute_distances(ctx, queries, num_queries, database, num_vectors,
                               dimension, metric, distances);
}

int gv_gpu_index_compute_distances(GV_GPUIndex *index, const float *queries,
                                   size_t num_queries,
                                   GV_GPUDistanceMetric metric,
                                   float *distances) {
  return gpu_index_compute_distances(index, queries, num_queries, metric,
                                     distances);
}

int gv_gpu_knn_search(GV_GPUContext *ctx, const float *queries,
                      size_t num_queries, const float *database,
                      size_t num_vectors, size_t dimension,
                      const GV_GPUSearchParams *params, size_t *indices,
                      float *distances) {
  return gpu_knn_search(ctx, queries, num_queries, database, num_vectors,
                        dimension, params, indices, distances);
}

int gv_gpu_index_knn_search(GV_GPUIndex *index, const float *queries,
                            size_t num_queries,
                            const GV_GPUSearchParams *params, size_t *indices,
                            float *distances) {
  return gpu_index_knn_search(index, queries, num_queries, params, indices,
                              distances);
}

int gv_gpu_index_search(GV_GPUIndex *index, const float *query,
                        const GV_GPUSearchParams *params, size_t *indices,
                        float *distances) {
  return gpu_index_search(index, query, params, indices, distances);
}

int gv_gpu_batch_add(GV_GPUContext *ctx, GV_Database *db, const float *vectors,
                     size_t count) {
  return gpu_batch_add(ctx, db, vectors, count);
}

int gv_gpu_batch_search(GV_GPUContext *ctx, GV_Database *db,
                        const float *queries, size_t num_queries, size_t k,
                        size_t *indices, float *distances) {
  return gpu_batch_search(ctx, db, queries, num_queries, k, indices, distances);
}

int gv_gpu_get_stats(GV_GPUContext *ctx, GV_GPUStats *stats) {
  return gpu_get_stats(ctx, stats);
}

int gv_gpu_reset_stats(GV_GPUContext *ctx) { return gpu_reset_stats(ctx); }

void gv_server_config_init(GV_ServerConfig *config) {
  server_config_init(config);
}

GV_Server *gv_server_create(GV_Database *db, const GV_ServerConfig *config) {
  return server_create(db, config);
}

int gv_server_start(GV_Server *server) { return server_start(server); }

int gv_server_stop(GV_Server *server) { return server_stop(server); }

void gv_server_destroy(GV_Server *server) { server_destroy(server); }

int gv_server_is_running(const GV_Server *server) {
  return server_is_running(server);
}

int gv_server_get_stats(const GV_Server *server, GV_ServerStats *stats) {
  return server_get_stats(server, stats);
}

uint16_t gv_server_get_port(const GV_Server *server) {
  return server_get_port(server);
}

void gv_backup_options_init(GV_BackupOptions *options) {
  backup_options_init(options);
}

void gv_restore_options_init(GV_RestoreOptions *options) {
  restore_options_init(options);
}

GV_BackupResult *gv_backup_create(GV_Database *db, const char *backup_path,
                                  const GV_BackupOptions *options,
                                  GV_BackupProgressCallback progress,
                                  void *user_data) {
  return backup_create(db, backup_path, options, progress, user_data);
}

GV_BackupResult *gv_backup_create_from_file(const char *db_path,
                                            const char *backup_path,
                                            const GV_BackupOptions *options,
                                            GV_BackupProgressCallback progress,
                                            void *user_data) {
  return backup_create_from_file(db_path, backup_path, options, progress,
                                 user_data);
}

void gv_backup_result_free(GV_BackupResult *result) {
  backup_result_free(result);
}

GV_BackupResult *gv_backup_restore(const char *backup_path, const char *db_path,
                                   const GV_RestoreOptions *options,
                                   GV_BackupProgressCallback progress,
                                   void *user_data) {
  return backup_restore(backup_path, db_path, options, progress, user_data);
}

GV_BackupResult *gv_backup_restore_to_db(const char *backup_path,
                                         const GV_RestoreOptions *options,
                                         GV_Database **db) {
  return backup_restore_to_db(backup_path, options, db);
}

int gv_backup_read_header(const char *backup_path, GV_BackupHeader *header) {
  return backup_read_header(backup_path, header);
}

GV_BackupResult *gv_backup_verify(const char *backup_path,
                                  const char *decryption_key) {
  return backup_verify(backup_path, decryption_key);
}

int gv_backup_get_info(const char *backup_path, char *info_buf,
                       size_t buf_size) {
  return backup_get_info(backup_path, info_buf, buf_size);
}

GV_BackupResult *gv_backup_create_incremental(GV_Database *db,
                                              const char *backup_path,
                                              const char *base_backup_path,
                                              const GV_BackupOptions *options) {
  return backup_create_incremental(db, backup_path, base_backup_path, options);
}

GV_BackupResult *gv_backup_merge(const char *base_backup_path,
                                 const char **incremental_paths,
                                 size_t incremental_count,
                                 const char *output_path) {
  return backup_merge(base_backup_path, incremental_paths, incremental_count,
                      output_path);
}

int gv_backup_compute_checksum(const char *backup_path, char *checksum_out) {
  return backup_compute_checksum(backup_path, checksum_out);
}

void gv_shard_config_init(GV_ShardConfig *config) { shard_config_init(config); }

GV_ShardManager *gv_shard_manager_create(const GV_ShardConfig *config) {
  return shard_manager_create(config);
}

void gv_shard_manager_destroy(GV_ShardManager *mgr) {
  shard_manager_destroy(mgr);
}

int gv_shard_add(GV_ShardManager *mgr, uint32_t shard_id,
                 const char *node_address) {
  return shard_add(mgr, shard_id, node_address);
}

int gv_shard_remove(GV_ShardManager *mgr, uint32_t shard_id) {
  return shard_remove(mgr, shard_id);
}

int gv_shard_for_vector(GV_ShardManager *mgr, uint64_t vector_id) {
  return shard_for_vector(mgr, vector_id);
}

int gv_shard_for_key(GV_ShardManager *mgr, const void *key, size_t key_len) {
  return shard_for_key(mgr, key, key_len);
}

int gv_shard_get_info(GV_ShardManager *mgr, uint32_t shard_id,
                      GV_ShardInfo *info) {
  return shard_get_info(mgr, shard_id, info);
}

int gv_shard_list(GV_ShardManager *mgr, GV_ShardInfo **shards, size_t *count) {
  return shard_list(mgr, shards, count);
}

void gv_shard_free_list(GV_ShardInfo *shards, size_t count) {
  shard_free_list(shards, count);
}

int gv_shard_set_state(GV_ShardManager *mgr, uint32_t shard_id,
                       GV_ShardState state) {
  return shard_set_state(mgr, shard_id, state);
}

int gv_shard_rebalance_start(GV_ShardManager *mgr) {
  return shard_rebalance_start(mgr);
}

int gv_shard_rebalance_status(GV_ShardManager *mgr, double *progress) {
  return shard_rebalance_status(mgr, progress);
}

int gv_shard_rebalance_cancel(GV_ShardManager *mgr) {
  return shard_rebalance_cancel(mgr);
}

int gv_shard_attach_local(GV_ShardManager *mgr, uint32_t shard_id,
                          GV_Database *db) {
  return shard_attach_local(mgr, shard_id, db);
}

GV_Database *gv_shard_get_local_db(GV_ShardManager *mgr, uint32_t shard_id) {
  return shard_get_local_db(mgr, shard_id);
}

int gv_shard_migrate_vectors(GV_ShardManager *mgr, uint32_t from_shard,
                             uint32_t to_shard, size_t count) {
  return shard_migrate_vectors(mgr, from_shard, to_shard, count);
}

int gv_shard_migrate_vector_at(GV_ShardManager *mgr, uint32_t from_shard,
                               uint32_t to_shard, size_t vector_index,
                               size_t *out_new_index) {
  return shard_migrate_vector_at(mgr, from_shard, to_shard, vector_index,
                                 out_new_index);
}

GV_ReplicationRole gv_replication_get_role(GV_ReplicationManager *mgr) {
  return replication_get_role(mgr);
}

int gv_replication_step_down(GV_ReplicationManager *mgr) {
  return replication_step_down(mgr);
}

int gv_replication_request_leadership(GV_ReplicationManager *mgr) {
  return replication_request_leadership(mgr);
}

int gv_replication_remove_follower(GV_ReplicationManager *mgr,
                                   const char *node_id) {
  return replication_remove_follower(mgr, node_id);
}

int gv_replication_list_replicas(GV_ReplicationManager *mgr,
                                 GV_ReplicaInfo **replicas, size_t *count) {
  return replication_list_replicas(mgr, replicas, count);
}

void gv_replication_free_replicas(GV_ReplicaInfo *replicas, size_t count) {
  replication_free_replicas(replicas, count);
}

int64_t gv_replication_get_lag(GV_ReplicationManager *mgr) {
  return replication_get_lag(mgr);
}

int gv_replication_wait_sync(GV_ReplicationManager *mgr, size_t max_lag,
                             uint32_t timeout_ms) {
  return replication_wait_sync(mgr, max_lag, timeout_ms);
}

int gv_replication_get_stats(GV_ReplicationManager *mgr,
                             GV_ReplicationStats *stats) {
  return replication_get_stats(mgr, stats);
}

void gv_replication_free_stats(GV_ReplicationStats *stats) {
  replication_free_stats(stats);
}

int gv_replication_is_healthy(GV_ReplicationManager *mgr) {
  return replication_is_healthy(mgr);
}

int gv_replication_set_read_policy(GV_ReplicationManager *mgr,
                                   GV_ReadPolicy policy) {
  return replication_set_read_policy(mgr, policy);
}

GV_ReadPolicy gv_replication_get_read_policy(GV_ReplicationManager *mgr) {
  return replication_get_read_policy(mgr);
}

GV_Database *gv_replication_route_read(GV_ReplicationManager *mgr) {
  return replication_route_read(mgr);
}

int gv_replication_release_read(GV_ReplicationManager *mgr, GV_Database *db) {
  return replication_release_read(mgr, db);
}

int gv_replication_set_max_read_lag(GV_ReplicationManager *mgr,
                                    uint64_t max_lag) {
  return replication_set_max_read_lag(mgr, max_lag);
}

int gv_replication_register_follower_db(GV_ReplicationManager *mgr,
                                        const char *node_id,
                                        GV_Database *db) {
  return replication_register_follower_db(mgr, node_id, db);
}

int gv_replication_register_follower_memory(GV_ReplicationManager *mgr,
                                            const char *node_id,
                                            GV_MemoryLayer *layer) {
  return replication_register_follower_memory(mgr, node_id, layer);
}

GV_MemoryLayer *gv_replication_route_read_memory(GV_ReplicationManager *mgr) {
  return replication_route_read_memory(mgr);
}

void gv_grpc_config_init(GV_GrpcConfig *config) { grpc_config_init(config); }

GV_GrpcServer *gv_grpc_create(GV_Database *db, const GV_GrpcConfig *config) {
  return grpc_create(db, config);
}

int gv_grpc_start(GV_GrpcServer *server) { return grpc_start(server); }

int gv_grpc_stop(GV_GrpcServer *server) { return grpc_stop(server); }

void gv_grpc_destroy(GV_GrpcServer *server) { grpc_destroy(server); }

int gv_grpc_is_running(const GV_GrpcServer *server) {
  return grpc_is_running(server);
}

int gv_grpc_get_stats(const GV_GrpcServer *server, GV_GrpcStats *stats) {
  return grpc_get_stats(server, stats);
}

const char *gv_grpc_error_string(int error) { return grpc_error_string(error); }

int gv_grpc_encode_search_request(const float *query, size_t dimension, size_t k,
                                  int distance_type, uint8_t *buf, size_t buf_size,
                                  size_t *out_len) {
  return grpc_encode_search_request(query, dimension, k, distance_type, buf,
                                    buf_size, out_len);
}

int gv_grpc_decode_search_request(const uint8_t *buf, size_t len, float **query,
                                  size_t *dimension, size_t *k,
                                  int *distance_type) {
  return grpc_decode_search_request(buf, len, query, dimension, k, distance_type);
}

int gv_grpc_encode_add_request(const float *data, size_t dimension, uint8_t *buf,
                               size_t buf_size, size_t *out_len) {
  return grpc_encode_add_request(data, dimension, buf, buf_size, out_len);
}

int gv_grpc_encode_ivfdisk_train_request(const float *data, size_t count,
                                         size_t dimension, uint8_t *buf,
                                         size_t buf_size, size_t *out_len) {
  return grpc_encode_ivfdisk_train_request(data, count, dimension, buf, buf_size,
                                           out_len);
}

int gv_grpc_client_ivfdisk_train(const char *host, uint16_t port,
                                 const float *data, size_t count,
                                 size_t dimension, uint32_t timeout_ms) {
  return grpc_client_ivfdisk_train(host, port, data, count, dimension, timeout_ms);
}

int gv_grpc_client_search(const char *host, uint16_t port, const float *query,
                          size_t dimension, size_t k, int distance_type,
                          GV_GrpcSearchResponse *out, uint32_t timeout_ms) {
  return grpc_client_search(host, port, query, dimension, k, distance_type, out,
                            timeout_ms);
}

void gv_grpc_search_response_free(GV_GrpcSearchResponse *resp) {
  grpc_search_response_free(resp);
}

int gv_db_apply_wal_record(GV_Database *db, const uint8_t *record, size_t len) {
  return db_apply_wal_record(db, record, len);
}

const char *gv_db_wal_path(const GV_Database *db) {
  return db_wal_path(db);
}

void gv_cluster_config_init(GV_ClusterConfig *config) {
  cluster_config_init(config);
}

GV_Cluster *gv_cluster_create(const GV_ClusterConfig *config) {
  return cluster_create(config);
}

void gv_cluster_destroy(GV_Cluster *cluster) { cluster_destroy(cluster); }

int gv_cluster_start(GV_Cluster *cluster) { return cluster_start(cluster); }

int gv_cluster_stop(GV_Cluster *cluster) { return cluster_stop(cluster); }

int gv_cluster_get_local_node(GV_Cluster *cluster, GV_NodeInfo *info) {
  return cluster_get_local_node(cluster, info);
}

int gv_cluster_get_node(GV_Cluster *cluster, const char *node_id,
                        GV_NodeInfo *info) {
  return cluster_get_node(cluster, node_id, info);
}

int gv_cluster_list_nodes(GV_Cluster *cluster, GV_NodeInfo **nodes,
                          size_t *count) {
  return cluster_list_nodes(cluster, nodes, count);
}

void gv_cluster_free_node_info(GV_NodeInfo *info) {
  cluster_free_node_info(info);
}

void gv_cluster_free_node_list(GV_NodeInfo *nodes, size_t count) {
  cluster_free_node_list(nodes, count);
}

int gv_cluster_get_stats(GV_Cluster *cluster, GV_ClusterStats *stats) {
  return cluster_get_stats(cluster, stats);
}

GV_ShardManager *gv_cluster_get_shard_manager(GV_Cluster *cluster) {
  return cluster_get_shard_manager(cluster);
}

int gv_cluster_is_healthy(GV_Cluster *cluster) {
  return cluster_is_healthy(cluster);
}

int gv_cluster_wait_ready(GV_Cluster *cluster, uint32_t timeout_ms) {
  return cluster_wait_ready(cluster, timeout_ms);
}

GV_NamespaceManager *gv_namespace_manager_create(const char *base_path) {
  return namespace_manager_create(base_path);
}

void gv_namespace_manager_destroy(GV_NamespaceManager *mgr) {
  namespace_manager_destroy(mgr);
}

GV_Namespace *gv_namespace_create(GV_NamespaceManager *mgr,
                                  const GV_NamespaceConfig *config) {
  return namespace_create(mgr, config);
}

GV_Namespace *gv_namespace_get(GV_NamespaceManager *mgr, const char *name) {
  return namespace_get(mgr, name);
}

int gv_namespace_delete(GV_NamespaceManager *mgr, const char *name) {
  return namespace_delete(mgr, name);
}

int gv_namespace_list(GV_NamespaceManager *mgr, char ***names, size_t *count) {
  return namespace_list(mgr, names, count);
}

int gv_namespace_get_info(const GV_Namespace *ns, GV_NamespaceInfo *info) {
  return namespace_get_info(ns, info);
}

void gv_namespace_free_info(GV_NamespaceInfo *info) {
  namespace_free_info(info);
}

int gv_namespace_exists(GV_NamespaceManager *mgr, const char *name) {
  return namespace_exists(mgr, name);
}

int gv_namespace_add_vector(GV_Namespace *ns, const float *data,
                            size_t dimension) {
  return namespace_add_vector(ns, data, dimension);
}

int gv_namespace_add_vector_with_metadata(GV_Namespace *ns, const float *data,
                                          size_t dimension,
                                          const char *const *keys,
                                          const char *const *values,
                                          size_t meta_count) {
  return namespace_add_vector_with_metadata(ns, data, dimension, keys, values,
                                            meta_count);
}

int gv_namespace_search(const GV_Namespace *ns, const float *query, size_t k,
                        GV_SearchResult *results,
                        GV_DistanceType distance_type) {
  return namespace_search(ns, query, k, results, distance_type);
}

int gv_namespace_search_filtered(const GV_Namespace *ns, const float *query,
                                 size_t k, GV_SearchResult *results,
                                 GV_DistanceType distance_type,
                                 const char *filter_key,
                                 const char *filter_value) {
  return namespace_search_filtered(ns, query, k, results, distance_type,
                                   filter_key, filter_value);
}

int gv_namespace_delete_vector(GV_Namespace *ns, size_t vector_index) {
  return namespace_delete_vector(ns, vector_index);
}

size_t gv_namespace_count(const GV_Namespace *ns) {
  return namespace_count(ns);
}

int gv_namespace_save(GV_Namespace *ns) { return namespace_save(ns); }

int gv_namespace_manager_save_all(GV_NamespaceManager *mgr) {
  return namespace_manager_save_all(mgr);
}

int gv_namespace_manager_load_all(GV_NamespaceManager *mgr) {
  return namespace_manager_load_all(mgr);
}

GV_Database *gv_namespace_get_db(GV_Namespace *ns) {
  return namespace_get_db(ns);
}

void gv_namespace_config_init(GV_NamespaceConfig *config) {
  namespace_config_init(config);
}

GV_BloomFilter *gv_bloom_create(size_t expected_items, double fp_rate) {
  return bloom_create(expected_items, fp_rate);
}
void gv_bloom_destroy(GV_BloomFilter *bf) { bloom_destroy(bf); }
int gv_bloom_add(GV_BloomFilter *bf, const void *data, size_t len) {
  return bloom_add(bf, data, len);
}
int gv_bloom_add_string(GV_BloomFilter *bf, const char *str) {
  return bloom_add_string(bf, str);
}
int gv_bloom_check(const GV_BloomFilter *bf, const void *data, size_t len) {
  return bloom_check(bf, data, len);
}
int gv_bloom_check_string(const GV_BloomFilter *bf, const char *str) {
  return bloom_check_string(bf, str);
}
size_t gv_bloom_count(const GV_BloomFilter *bf) { return bloom_count(bf); }
double gv_bloom_fp_rate(const GV_BloomFilter *bf) { return bloom_fp_rate(bf); }
void gv_bloom_clear(GV_BloomFilter *bf) { bloom_clear(bf); }

void gv_bm25_config_init(GV_BM25Config *config) { bm25_config_init(config); }
GV_BM25Index *gv_bm25_create(const GV_BM25Config *config) {
  return bm25_create(config);
}
void gv_bm25_destroy(GV_BM25Index *index) { bm25_destroy(index); }
int gv_bm25_add_document(GV_BM25Index *index, size_t doc_id, const char *text) {
  return bm25_add_document(index, doc_id, text);
}
int gv_bm25_add_document_terms(GV_BM25Index *index, size_t doc_id,
                                const char **terms, size_t term_count) {
  return bm25_add_document_terms(index, doc_id, terms, term_count);
}
int gv_bm25_remove_document(GV_BM25Index *index, size_t doc_id) {
  return bm25_remove_document(index, doc_id);
}
int gv_bm25_update_document(GV_BM25Index *index, size_t doc_id, const char *text) {
  return bm25_update_document(index, doc_id, text);
}
int gv_bm25_search(GV_BM25Index *index, const char *query, size_t k,
                   GV_BM25Result *results) {
  return bm25_search(index, query, k, results);
}
int gv_bm25_score_document(GV_BM25Index *index, size_t doc_id,
                            const char *query, double *score) {
  return bm25_score_document(index, doc_id, query, score);
}
int gv_bm25_get_stats(const GV_BM25Index *index, GV_BM25Stats *stats) {
  return bm25_get_stats(index, stats);
}
size_t gv_bm25_get_doc_freq(const GV_BM25Index *index, const char *term) {
  return bm25_get_doc_freq(index, term);
}
int gv_bm25_has_document(const GV_BM25Index *index, size_t doc_id) {
  return bm25_has_document(index, doc_id);
}
int gv_bm25_save(const GV_BM25Index *index, const char *filepath) {
  return bm25_save(index, filepath);
}
GV_BM25Index *gv_bm25_load(const char *filepath) { return bm25_load(filepath); }

GV_SnapshotManager *gv_snapshot_manager_create(size_t max_snapshots) {
  return snapshot_manager_create(max_snapshots);
}
void gv_snapshot_manager_destroy(GV_SnapshotManager *mgr) {
  snapshot_manager_destroy(mgr);
}
uint64_t gv_snapshot_create(GV_SnapshotManager *mgr, size_t vector_count,
                             const float *vector_data, size_t dimension,
                             const char *label) {
  return snapshot_create(mgr, vector_count, vector_data, dimension, label);
}
GV_Snapshot *gv_snapshot_open(GV_SnapshotManager *mgr, uint64_t snapshot_id) {
  return snapshot_open(mgr, snapshot_id);
}
void gv_snapshot_close(GV_Snapshot *snap) { snapshot_close(snap); }
size_t gv_snapshot_count(const GV_Snapshot *snap) { return snapshot_count(snap); }
const float *gv_snapshot_get_vector(const GV_Snapshot *snap, size_t index) {
  return snapshot_get_vector(snap, index);
}
size_t gv_snapshot_dimension(const GV_Snapshot *snap) {
  return snapshot_dimension(snap);
}
int gv_snapshot_list(const GV_SnapshotManager *mgr, GV_SnapshotInfo *infos,
                     size_t max_infos) {
  return snapshot_list(mgr, infos, max_infos);
}
int gv_snapshot_delete(GV_SnapshotManager *mgr, uint64_t snapshot_id) {
  return snapshot_delete(mgr, snapshot_id);
}

void gv_mmr_config_init(GV_MMRConfig *config) { mmr_config_init(config); }
int gv_mmr_rerank(const float *query, size_t dimension,
                  const float *candidates, const size_t *candidate_indices,
                  const float *candidate_distances, size_t candidate_count,
                  size_t k, const GV_MMRConfig *config, GV_MMRResult *results) {
  return mmr_rerank(query, dimension, candidates, candidate_indices,
                    candidate_distances, candidate_count, k, config, results);
}
int gv_mmr_search(const void *db, const float *query, size_t dimension,
                  size_t k, size_t oversample, const GV_MMRConfig *config,
                  GV_MMRResult *results) {
  return mmr_search(db, query, dimension, k, oversample, config, results);
}

void gv_kg_config_init(GV_KGConfig *config) { kg_config_init(config); }

GV_KnowledgeGraph *gv_kg_create(const GV_KGConfig *config) {
  return kg_create(config);
}

void gv_kg_destroy(GV_KnowledgeGraph *kg) { kg_destroy(kg); }

uint64_t gv_kg_add_entity(GV_KnowledgeGraph *kg, const char *name,
                          const char *type, const float *embedding,
                          size_t dimension) {
  return kg_add_entity(kg, name, type, embedding, dimension);
}

int gv_kg_remove_entity(GV_KnowledgeGraph *kg, uint64_t entity_id) {
  return kg_remove_entity(kg, entity_id);
}

const GV_KGEntity *gv_kg_get_entity(const GV_KnowledgeGraph *kg,
                                    uint64_t entity_id) {
  return kg_get_entity(kg, entity_id);
}

int gv_kg_set_entity_prop(GV_KnowledgeGraph *kg, uint64_t entity_id,
                          const char *key, const char *value) {
  return kg_set_entity_prop(kg, entity_id, key, value);
}

const char *gv_kg_get_entity_prop(const GV_KnowledgeGraph *kg,
                                   uint64_t entity_id, const char *key) {
  return kg_get_entity_prop(kg, entity_id, key);
}

int gv_kg_find_entities_by_type(const GV_KnowledgeGraph *kg, const char *type,
                                uint64_t *out_ids, size_t max_count) {
  return kg_find_entities_by_type(kg, type, out_ids, max_count);
}

int gv_kg_find_entities_by_name(const GV_KnowledgeGraph *kg, const char *name,
                              uint64_t *out_ids, size_t max_count) {
  return kg_find_entities_by_name(kg, name, out_ids, max_count);
}

uint64_t gv_kg_add_relation(GV_KnowledgeGraph *kg, uint64_t subject,
                            const char *predicate, uint64_t object,
                            float weight) {
  return kg_add_relation(kg, subject, predicate, object, weight);
}

int gv_kg_remove_relation(GV_KnowledgeGraph *kg, uint64_t relation_id) {
  return kg_remove_relation(kg, relation_id);
}

const GV_KGRelation *gv_kg_get_relation(const GV_KnowledgeGraph *kg,
                                        uint64_t relation_id) {
  return kg_get_relation(kg, relation_id);
}

int gv_kg_set_relation_prop(GV_KnowledgeGraph *kg, uint64_t relation_id,
                            const char *key, const char *value) {
  return kg_set_relation_prop(kg, relation_id, key, value);
}

int gv_kg_query_triples(const GV_KnowledgeGraph *kg, const uint64_t *subject,
                        const char *predicate, const uint64_t *object,
                        GV_KGTriple *out, size_t max_count) {
  return kg_query_triples(kg, subject, predicate, object, out, max_count);
}

void gv_kg_free_triples(GV_KGTriple *triples, size_t count) {
  kg_free_triples(triples, count);
}

uint64_t gv_kg_add_relation_with_chunk(GV_KnowledgeGraph *kg, uint64_t subject,
                                       const char *predicate, uint64_t object,
                                       float weight, const char *chunk_id) {
  return kg_add_relation_with_chunk(kg, subject, predicate, object, weight,
                                    chunk_id);
}

int gv_kg_query_triples_by_chunk(const GV_KnowledgeGraph *kg,
                                 const char *chunk_id, GV_KGTriple *out,
                                 size_t max_count) {
  return kg_query_triples_by_chunk(kg, chunk_id, out, max_count);
}

int gv_kg_remove_relations_by_chunk(GV_KnowledgeGraph *kg,
                                    const char *chunk_id) {
  return kg_remove_relations_by_chunk(kg, chunk_id);
}

int gv_graph_wal_attach(GV_GraphDB *g, const char *snapshot_path) {
  return graph_wal_attach(g, snapshot_path);
}

int gv_graph_wal_checkpoint(GV_GraphDB *g) {
  return graph_wal_checkpoint(g);
}

uint64_t gv_graph_version(const GV_GraphDB *g) { return graph_version(g); }

int gv_server_set_graphs(GV_Server *server, GV_KnowledgeGraph *kg,
                         GV_GraphDB *graph) {
  return server_set_graphs(server, kg, graph);
}

int gv_kg_wal_attach(GV_KnowledgeGraph *kg, const char *snapshot_path) {
  return kg_wal_attach(kg, snapshot_path);
}

int gv_kg_wal_checkpoint(GV_KnowledgeGraph *kg) {
  return kg_wal_checkpoint(kg);
}

int gv_kg_expand_context(const GV_KnowledgeGraph *kg, const uint64_t *seeds,
                         size_t n_seeds, size_t radius, GV_KGTriple *out,
                         size_t max_count) {
  return kg_expand_context(kg, seeds, n_seeds, radius, out, max_count);
}

int gv_kg_remove_entity_prop(GV_KnowledgeGraph *kg, uint64_t entity_id,
                             const char *key) {
  return kg_remove_entity_prop(kg, entity_id, key);
}

int gv_kg_remove_relation_prop(GV_KnowledgeGraph *kg, uint64_t relation_id,
                               const char *key) {
  return kg_remove_relation_prop(kg, relation_id, key);
}

int gv_kg_search_similar(const GV_KnowledgeGraph *kg,
                         const float *query_embedding, size_t dimension,
                         size_t k, GV_KGSearchResult *results) {
  return kg_search_similar(kg, query_embedding, dimension, k, results);
}

int gv_kg_search_by_text(const GV_KnowledgeGraph *kg, const char *text,
                         const float *text_embedding, size_t dimension,
                         size_t k, GV_KGSearchResult *results) {
  return kg_search_by_text(kg, text, text_embedding, dimension, k, results);
}

void gv_kg_free_search_results(GV_KGSearchResult *results, size_t count) {
  kg_free_search_results(results, count);
}

int gv_kg_resolve_entity(GV_KnowledgeGraph *kg, const char *name,
                         const char *type, const float *embedding,
                         size_t dimension) {
  return kg_resolve_entity(kg, name, type, embedding, dimension);
}

int gv_kg_find_duplicates(const GV_KnowledgeGraph *kg, float threshold,
                          GV_KGLinkPrediction *out, size_t max_count) {
  return kg_find_duplicates(kg, threshold, out, max_count);
}

int gv_kg_merge_entities(GV_KnowledgeGraph *kg, uint64_t keep_id,
                         uint64_t merge_id) {
  return kg_merge_entities(kg, keep_id, merge_id);
}

int gv_kg_predict_links(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                        size_t k, GV_KGLinkPrediction *results) {
  return kg_predict_links(kg, entity_id, k, results);
}

int gv_kg_get_neighbors(const GV_KnowledgeGraph *kg, uint64_t entity_id,
                        uint64_t *out_ids, size_t max_count) {
  return kg_get_neighbors(kg, entity_id, out_ids, max_count);
}

int gv_kg_traverse(const GV_KnowledgeGraph *kg, uint64_t start,
                   size_t max_depth, uint64_t *out_ids, size_t max_count) {
  return kg_traverse(kg, start, max_depth, out_ids, max_count);
}

int gv_kg_shortest_path(const GV_KnowledgeGraph *kg, uint64_t from,
                        uint64_t to, uint64_t *path_ids, size_t max_len) {
  return kg_shortest_path(kg, from, to, path_ids, max_len);
}

int gv_kg_extract_subgraph(const GV_KnowledgeGraph *kg, uint64_t center,
                           size_t radius, GV_KGSubgraph *subgraph) {
  return kg_extract_subgraph(kg, center, radius, subgraph);
}

void gv_kg_free_subgraph(GV_KGSubgraph *subgraph) {
  kg_free_subgraph(subgraph);
}

int gv_kg_hybrid_search(const GV_KnowledgeGraph *kg,
                        const float *query_embedding, size_t dimension,
                        const char *entity_type, const char *predicate_filter,
                        size_t k, GV_KGSearchResult *results) {
  return kg_hybrid_search(kg, query_embedding, dimension, entity_type,
                          predicate_filter, k, results);
}

int gv_kg_get_stats(const GV_KnowledgeGraph *kg, GV_KGStats *stats) {
  return kg_get_stats(kg, stats);
}

float gv_kg_entity_centrality(const GV_KnowledgeGraph *kg,
                              uint64_t entity_id) {
  return kg_entity_centrality(kg, entity_id);
}

int gv_kg_get_entity_types(const GV_KnowledgeGraph *kg, char **out_types,
                           size_t max_count) {
  return kg_get_entity_types(kg, out_types, max_count);
}

int gv_kg_get_predicates(const GV_KnowledgeGraph *kg, char **out_predicates,
                         size_t max_count) {
  return kg_get_predicates(kg, out_predicates, max_count);
}

int gv_kg_save(const GV_KnowledgeGraph *kg, const char *path) {
  return kg_save(kg, path);
}

GV_KnowledgeGraph *gv_kg_load(const char *path) { return kg_load(path); }

GV_PostingCatalog *gv_posting_catalog_open(const char *base_dir, size_t sector_size) {
  return posting_catalog_open(base_dir, sector_size);
}

void gv_posting_catalog_close(GV_PostingCatalog *cat) {
  posting_catalog_close(cat);
}

void gv_posting_catalog_set_cache_mb(GV_PostingCatalog *cat, size_t cache_size_mb) {
  posting_catalog_set_cache_mb(cat, cache_size_mb);
}

void gv_posting_catalog_get_cache_stats(const GV_PostingCatalog *cat,
                                        GV_PostingCacheStats *out) {
  posting_catalog_get_cache_stats(cat, out);
}

void gv_posting_catalog_set_auto_live_count(GV_PostingCatalog *cat, int enabled) {
  posting_catalog_set_auto_live_count(cat, enabled);
}

int gv_posting_catalog_get_auto_live_count(const GV_PostingCatalog *cat) {
  return posting_catalog_get_auto_live_count(cat);
}

uint32_t gv_posting_catalog_segment_live_count(const GV_PostingCatalog *cat,
                                               uint64_t head_id, uint64_t sequence) {
  return posting_catalog_segment_live_count(cat, head_id, sequence);
}

size_t gv_posting_catalog_segment_count(const GV_PostingCatalog *cat) {
  return posting_catalog_segment_count(cat);
}

size_t gv_posting_catalog_head_live_count(GV_PostingCatalog *cat, uint64_t head_id) {
  return posting_catalog_head_live_count(cat, head_id);
}

int gv_posting_catalog_reconcile_live_counts(GV_PostingCatalog *cat) {
  return posting_catalog_reconcile_live_counts(cat);
}

int gv_posting_catalog_append_segment(GV_PostingCatalog *cat, uint64_t head_id,
                                      const GV_PostingWriteEntry *entries,
                                      size_t entry_count, size_t dimension) {
  return posting_catalog_append_segment(cat, head_id, entries, entry_count, dimension);
}

int gv_posting_catalog_append_segment_ex(GV_PostingCatalog *cat, uint64_t head_id,
                                         const GV_PostingWriteEntry *entries,
                                         size_t entry_count, size_t dimension,
                                         const GV_PostingSegmentParams *params) {
  return posting_catalog_append_segment_ex(cat, head_id, entries, entry_count,
                                           dimension, params);
}

int gv_posting_catalog_materialize_head(GV_PostingCatalog *cat, uint64_t head_id,
                                        GV_PostingHeadView *out) {
  return posting_catalog_materialize_head(cat, head_id, out);
}

void gv_posting_head_view_free(GV_PostingHeadView *view) {
  posting_head_view_free(view);
}

GV_SQLEngine *gv_sql_create(void *db) { return sql_create(db); }

void gv_sql_destroy(GV_SQLEngine *eng) { sql_destroy(eng); }

int gv_sql_execute(GV_SQLEngine *eng, const char *query, GV_SQLResult *result) {
  return sql_execute(eng, query, result);
}

void gv_sql_free_result(GV_SQLResult *result) { sql_free_result(result); }

const char *gv_sql_last_error(const GV_SQLEngine *eng) {
  return sql_last_error(eng);
}

int gv_sql_explain(GV_SQLEngine *eng, const char *query, char *plan,
                   size_t plan_size) {
  return sql_explain(eng, query, plan, plan_size);
}

GV_Pipeline *gv_pipeline_create(const void *db) { return pipeline_create(db); }

void gv_pipeline_destroy(GV_Pipeline *pipe) { pipeline_destroy(pipe); }

int gv_pipeline_add_phase(GV_Pipeline *pipe, const void *config) {
  return pipeline_add_phase(pipe, (const GV_PhaseConfig *)config);
}

void gv_pipeline_clear_phases(GV_Pipeline *pipe) {
  pipeline_clear_phases(pipe);
}

size_t gv_pipeline_phase_count(const GV_Pipeline *pipe) {
  return pipeline_phase_count(pipe);
}

int gv_pipeline_execute(GV_Pipeline *pipe, const float *query, size_t dimension,
                        size_t final_k, GV_PhasedResult *results) {
  return pipeline_execute(pipe, query, dimension, final_k, results);
}

int gv_pipeline_get_stats(const GV_Pipeline *pipe, GV_PipelineStats *stats) {
  return pipeline_get_stats(pipe, stats);
}

void gv_pipeline_free_stats(GV_PipelineStats *stats) {
  pipeline_free_stats(stats);
}

void gv_ls_config_init(GV_LearnedSparseConfig *config) {
  ls_config_init(config);
}

GV_LearnedSparseIndex *gv_ls_create(const GV_LearnedSparseConfig *config) {
  return ls_create(config);
}

void gv_ls_destroy(GV_LearnedSparseIndex *idx) { ls_destroy(idx); }

int gv_ls_insert(GV_LearnedSparseIndex *idx, const GV_SparseEntry *entries,
                 size_t count) {
  return ls_insert(idx, (const GV_LSSparseEntry *)entries, count);
}

int gv_ls_delete(GV_LearnedSparseIndex *idx, size_t doc_id) {
  return ls_delete(idx, doc_id);
}

int gv_ls_search(const GV_LearnedSparseIndex *idx, const GV_SparseEntry *query,
                 size_t query_count, size_t k, GV_LearnedSparseResult *results) {
  return ls_search(idx, (const GV_LSSparseEntry *)query, query_count, k,
                   results);
}

int gv_ls_search_with_threshold(const GV_LearnedSparseIndex *idx,
                                const GV_SparseEntry *query, size_t query_count,
                                float min_score, size_t k,
                                GV_LearnedSparseResult *results) {
  return ls_search_with_threshold(idx, (const GV_LSSparseEntry *)query,
                                  query_count, min_score, k, results);
}

int gv_ls_get_stats(const GV_LearnedSparseIndex *idx,
                    GV_LearnedSparseStats *stats) {
  return ls_get_stats(idx, stats);
}

size_t gv_ls_count(const GV_LearnedSparseIndex *idx) { return ls_count(idx); }

int gv_ls_save(const GV_LearnedSparseIndex *idx, const char *path) {
  return ls_save(idx, path);
}

GV_LearnedSparseIndex *gv_ls_load(const char *path) { return ls_load(path); }

void gv_graph_config_init(GV_GraphDBConfig *config) {
  graph_config_init(config);
}

GV_GraphDB *gv_graph_create(const GV_GraphDBConfig *config) {
  return graph_create(config);
}

void gv_graph_destroy(GV_GraphDB *g) { graph_destroy(g); }

uint64_t gv_graph_add_node(GV_GraphDB *g, const char *label) {
  return graph_add_node(g, label);
}

int gv_graph_remove_node(GV_GraphDB *g, uint64_t node_id) {
  return graph_remove_node(g, node_id);
}

const GV_GraphNode *gv_graph_get_node(const GV_GraphDB *g, uint64_t node_id) {
  return graph_get_node(g, node_id);
}

int gv_graph_set_node_prop(GV_GraphDB *g, uint64_t node_id, const char *key,
                           const char *value) {
  return graph_set_node_prop(g, node_id, key, value);
}

const char *gv_graph_get_node_prop(const GV_GraphDB *g, uint64_t node_id,
                                   const char *key) {
  return graph_get_node_prop(g, node_id, key);
}

int gv_graph_find_nodes_by_label(const GV_GraphDB *g, const char *label,
                                 uint64_t *out_ids, size_t max_count) {
  return graph_find_nodes_by_label(g, label, out_ids, max_count);
}

uint64_t gv_graph_add_edge(GV_GraphDB *g, uint64_t source, uint64_t target,
                           const char *label, float weight) {
  return graph_add_edge(g, source, target, label, weight);
}

int gv_graph_remove_edge(GV_GraphDB *g, uint64_t edge_id) {
  return graph_remove_edge(g, edge_id);
}

const GV_GraphEdge *gv_graph_get_edge(const GV_GraphDB *g, uint64_t edge_id) {
  return graph_get_edge(g, edge_id);
}

int gv_graph_set_edge_prop(GV_GraphDB *g, uint64_t edge_id, const char *key,
                           const char *value) {
  return graph_set_edge_prop(g, edge_id, key, value);
}

const char *gv_graph_get_edge_prop(const GV_GraphDB *g, uint64_t edge_id,
                                   const char *key) {
  return graph_get_edge_prop(g, edge_id, key);
}

int gv_graph_get_edges_out(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count) {
  return graph_get_edges_out(g, node_id, out_ids, max_count);
}

int gv_graph_get_edges_in(const GV_GraphDB *g, uint64_t node_id,
                          uint64_t *out_ids, size_t max_count) {
  return graph_get_edges_in(g, node_id, out_ids, max_count);
}

int gv_graph_get_neighbors(const GV_GraphDB *g, uint64_t node_id,
                           uint64_t *out_ids, size_t max_count) {
  return graph_get_neighbors(g, node_id, out_ids, max_count);
}

int gv_graph_bfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count) {
  return graph_bfs(g, start, max_depth, out_ids, max_count);
}

int gv_graph_dfs(const GV_GraphDB *g, uint64_t start, size_t max_depth,
                 uint64_t *out_ids, size_t max_count) {
  return graph_dfs(g, start, max_depth, out_ids, max_count);
}

int gv_graph_shortest_path(const GV_GraphDB *g, uint64_t from, uint64_t to,
                           GV_GraphPath *path) {
  return graph_shortest_path(g, from, to, path);
}

int gv_graph_all_paths(const GV_GraphDB *g, uint64_t from, uint64_t to,
                       size_t max_depth, GV_GraphPath *paths, size_t max_paths) {
  return graph_all_paths(g, from, to, max_depth, paths, max_paths);
}

void gv_graph_free_path(GV_GraphPath *path) { graph_free_path(path); }

float gv_graph_pagerank(const GV_GraphDB *g, uint64_t node_id,
                        size_t iterations, float damping) {
  return graph_pagerank(g, node_id, iterations, damping);
}

size_t gv_graph_degree(const GV_GraphDB *g, uint64_t node_id) {
  return graph_degree(g, node_id);
}

size_t gv_graph_in_degree(const GV_GraphDB *g, uint64_t node_id) {
  return graph_in_degree(g, node_id);
}

size_t gv_graph_out_degree(const GV_GraphDB *g, uint64_t node_id) {
  return graph_out_degree(g, node_id);
}

int gv_graph_connected_components(const GV_GraphDB *g, uint64_t *component_ids,
                                  size_t max_count) {
  return graph_connected_components(g, component_ids, max_count);
}

float gv_graph_clustering_coefficient(const GV_GraphDB *g, uint64_t node_id) {
  return graph_clustering_coefficient(g, node_id);
}

size_t gv_graph_node_count(const GV_GraphDB *g) { return graph_node_count(g); }

size_t gv_graph_edge_count(const GV_GraphDB *g) { return graph_edge_count(g); }

int gv_graph_save(const GV_GraphDB *g, const char *path) {
  return graph_save(g, path);
}

/* =========================================================================
 * Python-ABI forwarder wrappers (audit section B): thin gv_* shims for
 * feature modules whose _ffi.py declarations previously had no C symbol.
 * ========================================================================= */

GV_GeoIndex *gv_geo_create(void) { return geo_create(); }
void gv_geo_destroy(GV_GeoIndex *index) { geo_destroy(index); }
int gv_geo_insert(GV_GeoIndex *index, size_t point_index, double lat, double lng) { return geo_insert(index, point_index, lat, lng); }
int gv_geo_update(GV_GeoIndex *index, size_t point_index, double lat, double lng) { return geo_update(index, point_index, lat, lng); }
int gv_geo_remove(GV_GeoIndex *index, size_t point_index) { return geo_remove(index, point_index); }
int gv_geo_radius_search(const GV_GeoIndex *index, double lat, double lng, double radius_km, GV_GeoResult *results, size_t max_results) { return geo_radius_search(index, lat, lng, radius_km, results, max_results); }
int gv_geo_bbox_search(const GV_GeoIndex *index, const GV_GeoBBox *bbox, GV_GeoResult *results, size_t max_results) { return geo_bbox_search(index, bbox, results, max_results); }
int gv_geo_get_candidates(const GV_GeoIndex *index, double lat, double lng, double radius_km, size_t *out_indices, size_t max_count) { return geo_get_candidates(index, lat, lng, radius_km, out_indices, max_count); }
double gv_geo_distance_km(double lat1, double lng1, double lat2, double lng2) { return geo_distance_km(lat1, lng1, lat2, lng2); }
size_t gv_geo_count(const GV_GeoIndex *index) { return geo_count(index); }
int gv_geo_save(const GV_GeoIndex *index, const char *filepath) { return geo_save(index, filepath); }
GV_GeoIndex *gv_geo_load(const char *filepath) { return geo_load(filepath); }

void gv_recommend_config_init(GV_RecommendConfig *config) { recommend_config_init(config); }
int gv_recommend_by_id(const GV_Database *db, const size_t *positive_ids, size_t positive_count, const size_t *negative_ids, size_t negative_count, size_t k, const GV_RecommendConfig *config, GV_RecommendResult *results) { return recommend_by_id(db, positive_ids, positive_count, negative_ids, negative_count, k, config, results); }
int gv_recommend_by_vector(const GV_Database *db, const float *positive_vectors, size_t positive_count, const float *negative_vectors, size_t negative_count, size_t dimension, size_t k, const GV_RecommendConfig *config, GV_RecommendResult *results) { return recommend_by_vector(db, positive_vectors, positive_count, negative_vectors, negative_count, dimension, k, config, results); }
int gv_recommend_discover(const GV_Database *db, const float *target, const float *context, size_t dimension, size_t k, const GV_RecommendConfig *config, GV_RecommendResult *results) { return recommend_discover(db, target, context, dimension, k, config, results); }

GV_JSONPathIndex *gv_json_index_create(void) { return json_index_create(); }
void gv_json_index_destroy(GV_JSONPathIndex *idx) { json_index_destroy(idx); }
int gv_json_index_add_path(GV_JSONPathIndex *idx, const GV_JSONPathConfig *config) { return json_index_add_path(idx, config); }
int gv_json_index_remove_path(GV_JSONPathIndex *idx, const char *path) { return json_index_remove_path(idx, path); }
int gv_json_index_insert(GV_JSONPathIndex *idx, size_t vector_index, const char *json_str) { return json_index_insert(idx, vector_index, json_str); }
int gv_json_index_remove(GV_JSONPathIndex *idx, size_t vector_index) { return json_index_remove(idx, vector_index); }
int gv_json_index_lookup_string(const GV_JSONPathIndex *idx, const char *path, const char *value, size_t *out_indices, size_t max_count) { return json_index_lookup_string(idx, path, value, out_indices, max_count); }
int gv_json_index_lookup_int_range(const GV_JSONPathIndex *idx, const char *path, int64_t min_val, int64_t max_val, size_t *out_indices, size_t max_count) { return json_index_lookup_int_range(idx, path, min_val, max_val, out_indices, max_count); }
int gv_json_index_lookup_float_range(const GV_JSONPathIndex *idx, const char *path, double min_val, double max_val, size_t *out_indices, size_t max_count) { return json_index_lookup_float_range(idx, path, min_val, max_val, out_indices, max_count); }
size_t gv_json_index_count(const GV_JSONPathIndex *idx, const char *path) { return json_index_count(idx, path); }
int gv_json_index_save(const GV_JSONPathIndex *idx, const char *path_file) { return json_index_save(idx, path_file); }
GV_JSONPathIndex *gv_json_index_load(const char *path_file) { return json_index_load(path_file); }

GV_DedupIndex *gv_dedup_create(size_t dimension, const GV_DedupConfig *config) { return dedup_create(dimension, config); }
void gv_dedup_destroy(GV_DedupIndex *dedup) { dedup_destroy(dedup); }
int gv_dedup_check(GV_DedupIndex *dedup, const float *data, size_t dimension) { return dedup_check(dedup, data, dimension); }
int gv_dedup_insert(GV_DedupIndex *dedup, const float *data, size_t dimension) { return dedup_insert(dedup, data, dimension); }
int gv_dedup_scan(GV_DedupIndex *dedup, GV_DedupResult *results, size_t max_results) { return dedup_scan(dedup, results, max_results); }
size_t gv_dedup_count(const GV_DedupIndex *dedup) { return dedup_count(dedup); }
void gv_dedup_clear(GV_DedupIndex *dedup) { dedup_clear(dedup); }

GV_CondManager *gv_cond_create(void *db) { return cond_create(db); }
void gv_cond_destroy(GV_CondManager *mgr) { cond_destroy(mgr); }
uint64_t gv_cond_get_version(const GV_CondManager *mgr, size_t index) { return cond_get_version(mgr, index); }
/* Forwarders for the conditional-write API. cond_* return the GV_ConditionalResult
 * enum; the Python layer treats it as int (the enum's underlying type). */
int gv_cond_update_vector(GV_CondManager *mgr, size_t index, const float *new_data, size_t dimension,
                          const GV_Condition *conditions, size_t condition_count) {
    return (int)cond_update_vector(mgr, index, new_data, dimension, conditions, condition_count);
}
int gv_cond_update_metadata(GV_CondManager *mgr, size_t index, const char *key, const char *value,
                            const GV_Condition *conditions, size_t condition_count) {
    return (int)cond_update_metadata(mgr, index, key, value, conditions, condition_count);
}
int gv_cond_delete(GV_CondManager *mgr, size_t index,
                   const GV_Condition *conditions, size_t condition_count) {
    return (int)cond_delete(mgr, index, conditions, condition_count);
}
int gv_cond_batch_update(GV_CondManager *mgr, const size_t *indices, const float **vectors,
                         const GV_Condition **conditions, const size_t *condition_counts,
                         size_t batch_size, int *results) {
    return cond_batch_update(mgr, indices, vectors, conditions, condition_counts, batch_size,
                             (GV_ConditionalResult *)results);
}
int gv_cond_migrate_embedding(GV_CondManager *mgr, size_t index, const float *new_embedding,
                              size_t dimension, uint64_t expected_version) {
    return (int)cond_migrate_embedding(mgr, index, new_embedding, dimension, expected_version);
}

GV_NamedVectorStore *gv_named_vectors_create(void) { return named_vectors_create(); }
void gv_named_vectors_destroy(GV_NamedVectorStore *store) { named_vectors_destroy(store); }
int gv_named_vectors_add_field(GV_NamedVectorStore *store, const GV_VectorFieldConfig *config) { return named_vectors_add_field(store, config); }
int gv_named_vectors_remove_field(GV_NamedVectorStore *store, const char *name) { return named_vectors_remove_field(store, name); }
size_t gv_named_vectors_field_count(const GV_NamedVectorStore *store) { return named_vectors_field_count(store); }
int gv_named_vectors_get_field(const GV_NamedVectorStore *store, const char *name, GV_VectorFieldConfig *out) { return named_vectors_get_field(store, name, out); }
int gv_named_vectors_insert(GV_NamedVectorStore *store, size_t point_id, const GV_NamedVector *vectors, size_t vector_count) { return named_vectors_insert(store, point_id, vectors, vector_count); }
int gv_named_vectors_update(GV_NamedVectorStore *store, size_t point_id, const GV_NamedVector *vectors, size_t vector_count) { return named_vectors_update(store, point_id, vectors, vector_count); }
int gv_named_vectors_delete(GV_NamedVectorStore *store, size_t point_id) { return named_vectors_delete(store, point_id); }
int gv_named_vectors_search(const GV_NamedVectorStore *store, const char *field_name, const float *query, size_t k, GV_NamedSearchResult *results) { return named_vectors_search(store, field_name, query, k, results); }
const float *gv_named_vectors_get(const GV_NamedVectorStore *store, size_t point_id, const char *field_name) { return named_vectors_get(store, point_id, field_name); }
size_t gv_named_vectors_count(const GV_NamedVectorStore *store) { return named_vectors_count(store); }
int gv_named_vectors_save(const GV_NamedVectorStore *store, const char *filepath) { return named_vectors_save(store, filepath); }
GV_NamedVectorStore *gv_named_vectors_load(const char *filepath) { return named_vectors_load(filepath); }

void gv_ttl_config_init(GV_TTLConfig *config){ ttl_config_init(config); }
GV_TTLManager *gv_ttl_create(const GV_TTLConfig *config){ return ttl_create(config); }
void gv_ttl_destroy(GV_TTLManager *mgr){ ttl_destroy(mgr); }
int gv_ttl_set(GV_TTLManager *mgr, size_t vector_index, uint64_t ttl_seconds){ return ttl_set(mgr, vector_index, ttl_seconds); }
int gv_ttl_set_absolute(GV_TTLManager *mgr, size_t vector_index, uint64_t expire_at_unix){ return ttl_set_absolute(mgr, vector_index, expire_at_unix); }
int gv_ttl_get(const GV_TTLManager *mgr, size_t vector_index, uint64_t *expire_at){ return ttl_get(mgr, vector_index, expire_at); }
int gv_ttl_remove(GV_TTLManager *mgr, size_t vector_index){ return ttl_remove(mgr, vector_index); }
int gv_ttl_is_expired(const GV_TTLManager *mgr, size_t vector_index){ return ttl_is_expired(mgr, vector_index); }
int gv_ttl_get_remaining(const GV_TTLManager *mgr, size_t vector_index, uint64_t *remaining_seconds){ return ttl_get_remaining(mgr, vector_index, remaining_seconds); }
int gv_ttl_cleanup_expired(GV_TTLManager *mgr, GV_Database *db){ return ttl_cleanup_expired(mgr, db); }
int gv_ttl_start_background_cleanup(GV_TTLManager *mgr, GV_Database *db){ return ttl_start_background_cleanup(mgr, db); }
void gv_ttl_stop_background_cleanup(GV_TTLManager *mgr){ ttl_stop_background_cleanup(mgr); }
int gv_ttl_is_background_cleanup_running(const GV_TTLManager *mgr){ return ttl_is_background_cleanup_running(mgr); }
int gv_ttl_get_stats(const GV_TTLManager *mgr, GV_TTLStats *stats){ return ttl_get_stats(mgr, stats); }
int gv_ttl_set_bulk(GV_TTLManager *mgr, const size_t *indices, size_t count, uint64_t ttl_seconds){ return ttl_set_bulk(mgr, indices, count, ttl_seconds); }
int gv_ttl_get_expiring_before(const GV_TTLManager *mgr, uint64_t before_unix, size_t *indices, size_t max_indices){ return ttl_get_expiring_before(mgr, before_unix, indices, max_indices); }

void gv_tt_config_init(GV_TimeTravelConfig *config){ tt_config_init(config); }
GV_TimeTravelManager *gv_tt_create(const GV_TimeTravelConfig *config){ return tt_create(config); }
void gv_tt_destroy(GV_TimeTravelManager *mgr){ tt_destroy(mgr); }
uint64_t gv_tt_record_insert(GV_TimeTravelManager *mgr, size_t index, const float *vector, size_t dimension){ return tt_record_insert(mgr, index, vector, dimension); }
uint64_t gv_tt_record_update(GV_TimeTravelManager *mgr, size_t index, const float *old_vector, const float *new_vector, size_t dimension){ return tt_record_update(mgr, index, old_vector, new_vector, dimension); }
uint64_t gv_tt_record_delete(GV_TimeTravelManager *mgr, size_t index, const float *vector, size_t dimension){ return tt_record_delete(mgr, index, vector, dimension); }
int gv_tt_query_at_version(const GV_TimeTravelManager *mgr, uint64_t version_id, size_t index, float *output, size_t dimension){ return tt_query_at_version(mgr, version_id, index, output, dimension); }
int gv_tt_query_at_timestamp(const GV_TimeTravelManager *mgr, uint64_t timestamp, size_t index, float *output, size_t dimension){ return tt_query_at_timestamp(mgr, timestamp, index, output, dimension); }
size_t gv_tt_count_at_version(const GV_TimeTravelManager *mgr, uint64_t version_id){ return tt_count_at_version(mgr, version_id); }
uint64_t gv_tt_current_version(const GV_TimeTravelManager *mgr){ return tt_current_version(mgr); }
int gv_tt_list_versions(const GV_TimeTravelManager *mgr, GV_VersionEntry *out, size_t max_count){ return tt_list_versions(mgr, out, max_count); }
int gv_tt_gc(GV_TimeTravelManager *mgr){ return tt_gc(mgr); }
int gv_tt_save(const GV_TimeTravelManager *mgr, const char *path){ return tt_save(mgr, path); }
GV_TimeTravelManager *gv_tt_load(const char *path){ return tt_load(path); }

GV_WebhookManager *gv_webhook_create(void){ return webhook_create(); }
void gv_webhook_destroy(GV_WebhookManager *mgr){ webhook_destroy(mgr); }
int gv_webhook_register(GV_WebhookManager *mgr, const char *webhook_id, const GV_WebhookConfig *config){ return webhook_register(mgr, webhook_id, config); }
int gv_webhook_unregister(GV_WebhookManager *mgr, const char *webhook_id){ return webhook_unregister(mgr, webhook_id); }
int gv_webhook_pause(GV_WebhookManager *mgr, const char *webhook_id){ return webhook_pause(mgr, webhook_id); }
int gv_webhook_resume(GV_WebhookManager *mgr, const char *webhook_id){ return webhook_resume(mgr, webhook_id); }
int gv_webhook_list(const GV_WebhookManager *mgr, char ***out_ids, size_t *out_count){ return webhook_list(mgr, out_ids, out_count); }
void gv_webhook_free_list(char **ids, size_t count){ webhook_free_list(ids, count); }
int gv_webhook_fire(GV_WebhookManager *mgr, const GV_Event *event){ return webhook_fire(mgr, event); }
int gv_webhook_get_stats(const GV_WebhookManager *mgr, GV_WebhookStats *stats){ return webhook_get_stats(mgr, stats); }
int gv_webhook_subscribe(GV_WebhookManager *mgr, GV_EventType mask, void *cb, void *user_data){ return webhook_subscribe(mgr, mask, (GV_ChangeCallback)cb, user_data); }
int gv_webhook_unsubscribe(GV_WebhookManager *mgr, void *cb){ return webhook_unsubscribe(mgr, (GV_ChangeCallback)cb); }

void gv_cdc_config_init(GV_CDCConfig *config){ cdc_config_init(config); }
GV_CDCStream *gv_cdc_create(const GV_CDCConfig *config){ return cdc_create(config); }
void gv_cdc_destroy(GV_CDCStream *stream){ cdc_destroy(stream); }
int gv_cdc_publish(GV_CDCStream *stream, const GV_CDCEvent *event){ return cdc_publish(stream, event); }
int gv_cdc_subscribe(GV_CDCStream *stream, uint32_t event_mask, void *callback, void *user_data){ return cdc_subscribe(stream, event_mask, (GV_CDCCallback)callback, user_data); }
int gv_cdc_unsubscribe(GV_CDCStream *stream, int subscriber_id){ return cdc_unsubscribe(stream, subscriber_id); }
int gv_cdc_poll(GV_CDCStream *stream, GV_CDCCursor *cursor, GV_CDCEvent *events, size_t max_events){ return cdc_poll(stream, cursor, events, max_events); }
GV_CDCCursor gv_cdc_get_cursor(const GV_CDCStream *stream){ return cdc_get_cursor(stream); }
GV_CDCCursor gv_cdc_cursor_from_sequence(uint64_t seq){ return cdc_cursor_from_sequence(seq); }
size_t gv_cdc_pending_count(const GV_CDCStream *stream, const GV_CDCCursor *cursor){ return cdc_pending_count(stream, cursor); }

GV_VersionManager *gv_version_manager_create(size_t max_versions){ return version_manager_create(max_versions); }
void gv_version_manager_destroy(GV_VersionManager *mgr){ version_manager_destroy(mgr); }
uint64_t gv_version_create(GV_VersionManager *mgr, const float *data, size_t count, size_t dimension, const char *label){ return version_create(mgr, data, count, dimension, label); }
int gv_version_list(const GV_VersionManager *mgr, GV_VersionInfo *infos, size_t max_infos){ return version_list(mgr, infos, max_infos); }
int gv_version_count(const GV_VersionManager *mgr){ return version_count(mgr); }
int gv_version_get_info(const GV_VersionManager *mgr, uint64_t version_id, GV_VersionInfo *info){ return version_get_info(mgr, version_id, info); }
float *gv_version_get_data(const GV_VersionManager *mgr, uint64_t version_id, size_t *count_out, size_t *dimension_out){ return version_get_data(mgr, version_id, count_out, dimension_out); }
int gv_version_delete(GV_VersionManager *mgr, uint64_t version_id){ return version_delete(mgr, version_id); }
int gv_version_compare(const GV_VersionManager *mgr, uint64_t v1, uint64_t v2, size_t *added, size_t *removed, size_t *modified){ return version_compare(mgr, v1, v2, added, removed, modified); }

GV_QueryTrace *gv_trace_begin(void){ return trace_begin(); }
void gv_trace_end(GV_QueryTrace *trace){ trace_end(trace); }
void gv_trace_destroy(GV_QueryTrace *trace){ trace_destroy(trace); }
void gv_trace_span_start(GV_QueryTrace *trace, const char *name){ trace_span_start(trace, name); }
void gv_trace_span_end(GV_QueryTrace *trace){ trace_span_end(trace); }
void gv_trace_span_add(GV_QueryTrace *trace, const char *name, uint64_t duration_us){ trace_span_add(trace, name, duration_us); }
void gv_trace_set_metadata(GV_QueryTrace *trace, const char *metadata){ trace_set_metadata(trace, metadata); }
char *gv_trace_to_json(const GV_QueryTrace *trace){ return trace_to_json(trace); }
uint64_t gv_trace_get_time_us(void){ return trace_get_time_us(); }

void gv_db_set_bulk_load(GV_Database *db, int on){ db_set_bulk_load(db, on); }
void gv_db_set_cdc_stream(GV_Database *db, GV_CDCStream *stream){ db_set_cdc_stream(db, stream); }
GV_CDCStream *gv_db_get_cdc_stream(const GV_Database *db){ return db_get_cdc_stream(db); }
void gv_db_set_webhook_manager(GV_Database *db, GV_WebhookManager *mgr){ db_set_webhook_manager(db, mgr); }
GV_WebhookManager *gv_db_get_webhook_manager(const GV_Database *db){ return db_get_webhook_manager(db); }

GV_Migration *gv_migration_start(const float *source_data, size_t count, size_t dimension, int new_index_type, const void *new_index_config){ return migration_start(source_data, count, dimension, new_index_type, new_index_config); }
int gv_migration_get_info(const GV_Migration *mig, GV_MigrationInfo *info){ return migration_get_info(mig, info); }
int gv_migration_wait(GV_Migration *mig){ return migration_wait(mig); }
int gv_migration_cancel(GV_Migration *mig){ return migration_cancel(mig); }
void *gv_migration_take_index(GV_Migration *mig){ return migration_take_index(mig); }
void gv_migration_destroy(GV_Migration *mig){ migration_destroy(mig); }

GV_GraphDB *gv_graph_load(const char *path) { return graph_load(path); }

GV_CypherEngine *gv_cypher_create(GV_KnowledgeGraph *kg) { return cypher_create(kg); }
void gv_cypher_destroy(GV_CypherEngine *eng) { cypher_destroy(eng); }
int gv_cypher_set_parameter(GV_CypherEngine *eng, const char *name, const char *value) { return cypher_set_parameter(eng, name, value); }
int gv_cypher_execute(GV_CypherEngine *eng, const char *query, GV_CypherResult *result) { return cypher_execute(eng, query, result); }
void gv_cypher_free_result(GV_CypherResult *result) { cypher_free_result(result); }
const char *gv_cypher_last_error(const GV_CypherEngine *eng) { return cypher_last_error(eng); }

/* DiskANN forwarders — the Python DiskANNIndex binding calls these gv_-prefixed
 * symbols (declared in _ffi.py) but only the unprefixed diskann_* functions
 * existed, so every DiskANNIndex construction failed with an undefined symbol. */
void gv_diskann_config_init(GV_DiskANNConfig *config) { diskann_config_init(config); }
GV_DiskANNIndex *gv_diskann_create(size_t dimension, const GV_DiskANNConfig *config) { return diskann_create(dimension, config); }
void gv_diskann_destroy(GV_DiskANNIndex *index) { diskann_destroy(index); }
int gv_diskann_build(GV_DiskANNIndex *index, const float *data, size_t count, size_t dimension) { return diskann_build(index, data, count, dimension); }
int gv_diskann_insert(GV_DiskANNIndex *index, const float *data, size_t dimension) { return diskann_insert(index, data, dimension); }
int gv_diskann_search(const GV_DiskANNIndex *index, const float *query, size_t dimension, size_t k, GV_DiskANNResult *results) { return diskann_search(index, query, dimension, k, results); }
int gv_diskann_delete(GV_DiskANNIndex *index, size_t vector_index) { return diskann_delete(index, vector_index); }
int gv_diskann_get_stats(const GV_DiskANNIndex *index, GV_DiskANNStats *stats) { return diskann_get_stats(index, stats); }
int gv_diskann_save(const GV_DiskANNIndex *index, const char *filepath) { return diskann_save(index, filepath); }
GV_DiskANNIndex *gv_diskann_load(const char *filepath, const GV_DiskANNConfig *config) { return diskann_load(filepath, config); }
size_t gv_diskann_count(const GV_DiskANNIndex *index) { return diskann_count(index); }

/* --- Collection aliases: gv_-prefixed wrappers the CFFI layer resolves. --- */
GV_AliasManager *gv_alias_manager_create(void) { return alias_manager_create(); }
void gv_alias_manager_destroy(GV_AliasManager *mgr) { alias_manager_destroy(mgr); }
int gv_alias_create(GV_AliasManager *mgr, const char *alias_name, const char *collection_name) { return alias_create(mgr, alias_name, collection_name); }
int gv_alias_update(GV_AliasManager *mgr, const char *alias_name, const char *new_collection_name) { return alias_update(mgr, alias_name, new_collection_name); }
int gv_alias_delete(GV_AliasManager *mgr, const char *alias_name) { return alias_delete(mgr, alias_name); }
int gv_alias_exists(const GV_AliasManager *mgr, const char *alias_name) { return alias_exists(mgr, alias_name); }
int gv_alias_swap(GV_AliasManager *mgr, const char *alias_a, const char *alias_b) { return alias_swap(mgr, alias_a, alias_b); }
const char *gv_alias_resolve(const GV_AliasManager *mgr, const char *alias_name) { return alias_resolve(mgr, alias_name); }
int gv_alias_list(const GV_AliasManager *mgr, GV_AliasInfo **out_list, size_t *out_count) { return alias_list(mgr, out_list, out_count); }
void gv_alias_free_list(GV_AliasInfo *list, size_t count) { alias_free_list(list, count); }
int gv_alias_get_info(const GV_AliasManager *mgr, const char *alias_name, GV_AliasInfo *info) { return alias_get_info(mgr, alias_name, info); }
size_t gv_alias_count(const GV_AliasManager *mgr) { return alias_count(mgr); }
int gv_alias_save(const GV_AliasManager *mgr, const char *filepath) { return alias_save(mgr, filepath); }
GV_AliasManager *gv_alias_load(const char *filepath) { return alias_load(filepath); }

/* ===== Auto-generated gv_* CFFI forwarding wrappers =====
 * The high-level Python layer (CFFI) resolves gv_-prefixed symbols, but many
 * subsystems only exported their unprefixed C names, so those Python classes
 * failed at call time with undefined-symbol errors. These one-line forwarders
 * expose the gv_ names the bindings expect. */
#include "admin/cache.h"
#include "admin/sso.h"
#include "api/schema.h"
#include "index/codebook.h"
#include "index/hnsw_opt.h"
#include "multimodal/auto_embed.h"
#include "multimodal/fulltext.h"
#include "multimodal/inference.h"
#include "multimodal/late_interaction.h"
#include "multimodal/multimodal.h"
#include "multimodal/multivec.h"
#include "multimodal/muvera.h"
#include "multimodal/onnx.h"
#include "multimodal/payload_index.h"
#include "schema/tiered_tenant.h"
#include "search/consistency.h"
#include "search/filter_ops.h"
#include "search/group_search.h"
#include "search/hybrid_search.h"
#include "search/quant_rerank.h"
#include "search/ranking.h"
#include "search/score_threshold.h"
#include "security/auth.h"
#include "security/rbac.h"
#include "security/tls.h"
#include "specialized/agent.h"
#include "specialized/embedded.h"
#include "specialized/optimizer.h"
#include "specialized/point_id.h"
#include "specialized/quantization.h"
#include "storage/compression.h"
#include "storage/vacuum.h"

const char * gv_gpu_get_error(GV_GPUContext *ctx) { return gpu_get_error(ctx); }
const char * gv_server_error_string(int error) { return server_error_string(error); }
void gv_hybrid_config_init(GV_HybridConfig *config) { hybrid_config_init(config); }
GV_HybridSearcher * gv_hybrid_create(GV_Database *db, GV_BM25Index *bm25, const GV_HybridConfig *config) { return hybrid_create(db, bm25, config); }
void gv_hybrid_destroy(GV_HybridSearcher *searcher) { hybrid_destroy(searcher); }
int gv_hybrid_search(GV_HybridSearcher *searcher, const float *query_vector, const char *query_text, size_t k, GV_HybridResult *results) { return hybrid_search(searcher, query_vector, query_text, k, results); }
int gv_hybrid_search_with_stats(GV_HybridSearcher *searcher, const float *query_vector, const char *query_text, size_t k, GV_HybridResult *results, GV_HybridStats *stats) { return hybrid_search_with_stats(searcher, query_vector, query_text, k, results, stats); }
int gv_hybrid_search_vector_only(GV_HybridSearcher *searcher, const float *query_vector, size_t k, GV_HybridResult *results) { return hybrid_search_vector_only(searcher, query_vector, k, results); }
int gv_hybrid_search_text_only(GV_HybridSearcher *searcher, const char *query_text, size_t k, GV_HybridResult *results) { return hybrid_search_text_only(searcher, query_text, k, results); }
int gv_hybrid_set_weights(GV_HybridSearcher *searcher, double vector_weight, double text_weight) { return hybrid_set_weights(searcher, vector_weight, text_weight); }
void gv_auth_config_init(GV_AuthConfig *config) { auth_config_init(config); }
GV_AuthManager * gv_auth_create(const GV_AuthConfig *config) { return auth_create(config); }
void gv_auth_destroy(GV_AuthManager *auth) { auth_destroy(auth); }
int gv_auth_generate_api_key(GV_AuthManager *auth, const char *description, uint64_t expires_at, char *key_out, char *key_id_out) { return auth_generate_api_key(auth, description, expires_at, key_out, key_id_out); }
int gv_auth_add_api_key(GV_AuthManager *auth, const char *key_id, const char *key_hash, const char *description, uint64_t expires_at) { return auth_add_api_key(auth, key_id, key_hash, description, expires_at); }
int gv_auth_revoke_api_key(GV_AuthManager *auth, const char *key_id) { return auth_revoke_api_key(auth, key_id); }
int gv_auth_list_api_keys(GV_AuthManager *auth, GV_APIKey **keys, size_t *count) { return auth_list_api_keys(auth, keys, count); }
void gv_auth_free_api_keys(GV_APIKey *keys, size_t count) { auth_free_api_keys(keys, count); }
GV_AuthResult gv_auth_verify_api_key(GV_AuthManager *auth, const char *api_key, GV_Identity *identity) { return auth_verify_api_key(auth, api_key, identity); }
GV_AuthResult gv_auth_verify_jwt(GV_AuthManager *auth, const char *token, GV_Identity *identity) { return auth_verify_jwt(auth, token, identity); }
GV_AuthResult gv_auth_authenticate(GV_AuthManager *auth, const char *credential, GV_Identity *identity) { return auth_authenticate(auth, credential, identity); }
void gv_auth_free_identity(GV_Identity *identity) { auth_free_identity(identity); }
int gv_auth_generate_jwt(GV_AuthManager *auth, const char *subject, uint64_t expires_in, char *token_out, size_t token_size) { return auth_generate_jwt(auth, subject, expires_in, token_out, token_size); }
const char * gv_auth_result_string(GV_AuthResult result) { return auth_result_string(result); }
void * gv_multivec_create(size_t dimension, const GV_MultiVecConfig *config) { return multivec_create(dimension, config); }
void gv_multivec_destroy(void *index) { multivec_destroy(index); }
int gv_multivec_add_document(void *index, uint64_t doc_id, const float *chunks, size_t num_chunks, size_t dimension) { return multivec_add_document(index, doc_id, chunks, num_chunks, dimension); }
int gv_multivec_delete_document(void *index, uint64_t doc_id) { return multivec_delete_document(index, doc_id); }
int gv_multivec_search(void *index, const float *query, size_t k, GV_DocSearchResult *results, int distance_type) { return multivec_search(index, query, k, results, distance_type); }
size_t gv_multivec_count_documents(const void *index) { return multivec_count_documents(index); }
size_t gv_multivec_count_chunks(const void *index) { return multivec_count_chunks(index); }
GV_QueryOptimizer * gv_optimizer_create(void) { return optimizer_create(); }
void gv_optimizer_destroy(GV_QueryOptimizer *opt) { optimizer_destroy(opt); }
void gv_optimizer_update_stats(GV_QueryOptimizer *opt, const GV_CollectionStats *stats) { optimizer_update_stats(opt, stats); }
int gv_optimizer_plan(const GV_QueryOptimizer *opt, size_t k, int has_filter, double filter_selectivity, GV_QueryPlan *plan) { return optimizer_plan(opt, k, has_filter, filter_selectivity, plan); }
size_t gv_optimizer_recommend_ef_search(const GV_QueryOptimizer *opt, size_t k) { return optimizer_recommend_ef_search(opt, k); }
size_t gv_optimizer_recommend_nprobe(const GV_QueryOptimizer *opt, size_t k) { return optimizer_recommend_nprobe(opt, k); }
GV_PayloadIndex * gv_payload_index_create(void) { return payload_index_create(); }
void gv_payload_index_destroy(GV_PayloadIndex *idx) { payload_index_destroy(idx); }
int gv_payload_index_add_field(GV_PayloadIndex *idx, const char *name, GV_FieldType type) { return payload_index_add_field(idx, name, type); }
int gv_payload_index_field_count(const GV_PayloadIndex *idx) { return payload_index_field_count(idx); }
int gv_payload_index_insert_int(GV_PayloadIndex *idx, size_t vector_id, const char *field, int64_t value) { return payload_index_insert_int(idx, vector_id, field, value); }
int gv_payload_index_insert_float(GV_PayloadIndex *idx, size_t vector_id, const char *field, double value) { return payload_index_insert_float(idx, vector_id, field, value); }
int gv_payload_index_insert_string(GV_PayloadIndex *idx, size_t vector_id, const char *field, const char *value) { return payload_index_insert_string(idx, vector_id, field, value); }
int gv_payload_index_insert_bool(GV_PayloadIndex *idx, size_t vector_id, const char *field, int value) { return payload_index_insert_bool(idx, vector_id, field, value); }
int gv_payload_index_remove(GV_PayloadIndex *idx, size_t vector_id) { return payload_index_remove(idx, vector_id); }
size_t gv_payload_index_total_entries(const GV_PayloadIndex *idx) { return payload_index_total_entries(idx); }
void gv_cache_config_init(GV_CacheConfig *config) { cache_config_init(config); }
GV_Cache * gv_cache_create(const GV_CacheConfig *config) { return cache_create(config); }
void gv_cache_destroy(GV_Cache *cache) { cache_destroy(cache); }
void gv_cache_notify_mutation(GV_Cache *cache) { cache_notify_mutation(cache); }
void gv_cache_invalidate_all(GV_Cache *cache) { cache_invalidate_all(cache); }
int gv_cache_get_stats(const GV_Cache *cache, GV_CacheStats *stats) { return cache_get_stats(cache, stats); }
void gv_cache_reset_stats(GV_Cache *cache) { cache_reset_stats(cache); }
GV_Schema * gv_schema_create(uint32_t version) { return schema_create(version); }
void gv_schema_destroy(GV_Schema *schema) { schema_destroy(schema); }
int gv_schema_add_field(GV_Schema *schema, const char *name, GV_SchemaFieldType type, int required, const char *default_value) { return schema_add_field(schema, name, type, required, default_value); }
int gv_schema_remove_field(GV_Schema *schema, const char *name) { return schema_remove_field(schema, name); }
int gv_schema_has_field(const GV_Schema *schema, const char *name) { return schema_has_field(schema, name); }
size_t gv_schema_field_count(const GV_Schema *schema) { return schema_field_count(schema); }
int gv_schema_validate(const GV_Schema *schema, const char *const *keys, const char *const *values, size_t count) { return schema_validate(schema, keys, values, count); }
int gv_schema_is_compatible(const GV_Schema *old_schema, const GV_Schema *new_schema) { return schema_is_compatible(old_schema, new_schema); }
char * gv_schema_to_json(const GV_Schema *schema) { return schema_to_json(schema); }
GV_Codebook * gv_codebook_create(size_t dimension, size_t m, uint8_t nbits) { return codebook_create(dimension, m, nbits); }
void gv_codebook_destroy(GV_Codebook *cb) { codebook_destroy(cb); }
int gv_codebook_train(GV_Codebook *cb, const float *data, size_t count, size_t train_iters) { return codebook_train(cb, data, count, train_iters); }
int gv_codebook_encode(const GV_Codebook *cb, const float *vector, uint8_t *codes) { return codebook_encode(cb, vector, codes); }
int gv_codebook_decode(const GV_Codebook *cb, const uint8_t *codes, float *output) { return codebook_decode(cb, codes, output); }
int gv_codebook_save(const GV_Codebook *cb, const char *filepath) { return codebook_save(cb, filepath); }
GV_Codebook * gv_codebook_load(const char *filepath) { return codebook_load(filepath); }
GV_Codebook * gv_codebook_copy(const GV_Codebook *cb) { return codebook_copy(cb); }
GV_PointIDMap * gv_point_id_create(size_t initial_capacity) { return point_id_create(initial_capacity); }
void gv_point_id_destroy(GV_PointIDMap *map) { point_id_destroy(map); }
int gv_point_id_set(GV_PointIDMap *map, const char *string_id, size_t internal_index) { return point_id_set(map, string_id, internal_index); }
int gv_point_id_get(const GV_PointIDMap *map, const char *string_id, size_t *out_index) { return point_id_get(map, string_id, out_index); }
int gv_point_id_remove(GV_PointIDMap *map, const char *string_id) { return point_id_remove(map, string_id); }
int gv_point_id_has(const GV_PointIDMap *map, const char *string_id) { return point_id_has(map, string_id); }
const char * gv_point_id_reverse_lookup(const GV_PointIDMap *map, size_t internal_index) { return point_id_reverse_lookup(map, internal_index); }
int gv_point_id_generate_uuid(char *buf, size_t buf_size) { return point_id_generate_uuid(buf, buf_size); }
size_t gv_point_id_count(const GV_PointIDMap *map) { return point_id_count(map); }
int gv_point_id_save(const GV_PointIDMap *map, const char *filepath) { return point_id_save(map, filepath); }
GV_PointIDMap * gv_point_id_load(const char *filepath) { return point_id_load(filepath); }
void gv_tls_config_init(GV_TLSConfig *config) { tls_config_init(config); }
GV_TLSContext * gv_tls_create(const GV_TLSConfig *config) { return tls_create(config); }
void gv_tls_destroy(GV_TLSContext *ctx) { tls_destroy(ctx); }
int gv_tls_is_available(void) { return tls_is_available(); }
int gv_tls_cert_days_remaining(const GV_TLSContext *ctx) { return tls_cert_days_remaining(ctx); }
int gv_db_search_with_threshold(const void *db, const float *query_data, size_t k, int distance_type, float score_threshold, GV_ThresholdResult *results) { return db_search_with_threshold(db, query_data, k, distance_type, score_threshold, results); }
int gv_db_delete_by_filter(GV_Database *db, const char *filter_expr) { return db_delete_by_filter(db, filter_expr); }
int gv_db_update_metadata_by_filter(GV_Database *db, const char *filter_expr, const char **metadata_keys, const char **metadata_values, size_t metadata_count) { return db_update_metadata_by_filter(db, filter_expr, metadata_keys, metadata_values, metadata_count); }
int gv_db_count_by_filter(const GV_Database *db, const char *filter_expr) { return db_count_by_filter(db, filter_expr); }
void gv_auto_embed_config_init(GV_AutoEmbedConfig *config) { auto_embed_config_init(config); }
GV_AutoEmbedder * gv_auto_embed_create(const GV_AutoEmbedConfig *config) { return auto_embed_create(config); }
void gv_auto_embed_destroy(GV_AutoEmbedder *embedder) { auto_embed_destroy(embedder); }
float * gv_auto_embed_text(GV_AutoEmbedder *embedder, const char *text, size_t *out_dimension) { return auto_embed_text(embedder, text, out_dimension); }
int gv_auto_embed_get_stats(const GV_AutoEmbedder *embedder, GV_AutoEmbedStats *stats) { return auto_embed_get_stats(embedder, stats); }
void gv_group_search_config_init(GV_GroupSearchConfig *config) { group_search_config_init(config); }
int gv_group_search(const GV_Database *db, const float *query, size_t dimension, const GV_GroupSearchConfig *config, GV_GroupedResult *result) { return group_search(db, query, dimension, config, result); }
void gv_group_search_free_result(GV_GroupedResult *result) { group_search_free_result(result); }
void gv_late_interaction_config_init(GV_LateInteractionConfig *config) { late_interaction_config_init(config); }
GV_LateInteractionIndex * gv_late_interaction_create(const GV_LateInteractionConfig *config) { return late_interaction_create(config); }
void gv_late_interaction_destroy(GV_LateInteractionIndex *index) { late_interaction_destroy(index); }
int gv_late_interaction_add_doc(GV_LateInteractionIndex *index, const float *token_embeddings, size_t num_tokens) { return late_interaction_add_doc(index, token_embeddings, num_tokens); }
int gv_late_interaction_search(const GV_LateInteractionIndex *index, const float *query_tokens, size_t num_query_tokens, size_t k, GV_LateInteractionResult *results) { return late_interaction_search(index, query_tokens, num_query_tokens, k, results); }
size_t gv_late_interaction_count(const GV_LateInteractionIndex *index) { return late_interaction_count(index); }
void gv_vacuum_config_init(GV_VacuumConfig *config) { vacuum_config_init(config); }
GV_VacuumManager * gv_vacuum_create(GV_Database *db, const GV_VacuumConfig *config) { return vacuum_create(db, config); }
void gv_vacuum_destroy(GV_VacuumManager *mgr) { vacuum_destroy(mgr); }
int gv_vacuum_run(GV_VacuumManager *mgr) { return vacuum_run(mgr); }
int gv_vacuum_start_auto(GV_VacuumManager *mgr) { return vacuum_start_auto(mgr); }
int gv_vacuum_stop_auto(GV_VacuumManager *mgr) { return vacuum_stop_auto(mgr); }
double gv_vacuum_get_fragmentation(const GV_VacuumManager *mgr) { return vacuum_get_fragmentation(mgr); }
int gv_vacuum_get_stats(const GV_VacuumManager *mgr, GV_VacuumStats *stats) { return vacuum_get_stats(mgr, stats); }
GV_ConsistencyManager * gv_consistency_create(GV_ConsistencyLevel default_level) { return consistency_create(default_level); }
void gv_consistency_destroy(GV_ConsistencyManager *mgr) { consistency_destroy(mgr); }
int gv_consistency_set_default(GV_ConsistencyManager *mgr, GV_ConsistencyLevel level) { return consistency_set_default(mgr, level); }
GV_ConsistencyLevel gv_consistency_get_default(const GV_ConsistencyManager *mgr) { return consistency_get_default(mgr); }
uint64_t gv_consistency_new_session(GV_ConsistencyManager *mgr) { return consistency_new_session(mgr); }
void gv_compression_config_init(GV_CompressionConfig *config) { compression_config_init(config); }
GV_Compressor * gv_compression_create(const GV_CompressionConfig *config) { return compression_create(config); }
void gv_compression_destroy(GV_Compressor *comp) { compression_destroy(comp); }
size_t gv_compress(GV_Compressor *comp, const void *input, size_t input_len, void *output, size_t output_capacity) { return compress(comp, input, input_len, output, output_capacity); }
size_t gv_decompress(GV_Compressor *comp, const void *input, size_t input_len, void *output, size_t output_capacity) { return decompress(comp, input, input_len, output, output_capacity); }
size_t gv_compress_bound(const GV_Compressor *comp, size_t input_len) { return compress_bound(comp, input_len); }
int gv_compression_get_stats(const GV_Compressor *comp, GV_CompressionStats *stats) { return compression_get_stats(comp, stats); }
GV_RBACManager * gv_rbac_create(void) { return rbac_create(); }
void gv_rbac_destroy(GV_RBACManager *mgr) { rbac_destroy(mgr); }
int gv_rbac_create_role(GV_RBACManager *mgr, const char *role_name) { return rbac_create_role(mgr, role_name); }
int gv_rbac_delete_role(GV_RBACManager *mgr, const char *role_name) { return rbac_delete_role(mgr, role_name); }
int gv_rbac_add_rule(GV_RBACManager *mgr, const char *role_name, const char *resource, uint32_t permissions) { return rbac_add_rule(mgr, role_name, resource, permissions); }
int gv_rbac_assign_role(GV_RBACManager *mgr, const char *user_id, const char *role_name) { return rbac_assign_role(mgr, user_id, role_name); }
int gv_rbac_revoke_role(GV_RBACManager *mgr, const char *user_id, const char *role_name) { return rbac_revoke_role(mgr, user_id, role_name); }
int gv_rbac_check(const GV_RBACManager *mgr, const char *user_id, const char *resource, GV_Permission required) { return rbac_check(mgr, user_id, resource, required); }
int gv_rbac_init_defaults(GV_RBACManager *mgr) { return rbac_init_defaults(mgr); }
int gv_rbac_save(const GV_RBACManager *mgr, const char *filepath) { return rbac_save(mgr, filepath); }
GV_RBACManager * gv_rbac_load(const char *filepath) { return rbac_load(filepath); }
GV_RankExpr * gv_rank_expr_parse(const char *expression) { return rank_expr_parse(expression); }
GV_RankExpr * gv_rank_expr_create_weighted(size_t n, const char **signal_names, const double *weights) { return rank_expr_create_weighted(n, signal_names, weights); }
double gv_rank_expr_eval(const GV_RankExpr *expr, float vector_score, const GV_RankSignal *signals, size_t signal_count) { return rank_expr_eval(expr, vector_score, signals, signal_count); }
void gv_rank_expr_destroy(GV_RankExpr *expr) { rank_expr_destroy(expr); }
void gv_quant_config_init(GV_QuantConfig *config) { quant_config_init(config); }
GV_QuantCodebook * gv_quant_train(const float *vectors, size_t count, size_t dimension, const GV_QuantConfig *config) { return quant_train(vectors, count, dimension, config); }
int gv_quant_encode(const GV_QuantCodebook *cb, const float *vector, size_t dimension, uint8_t *codes) { return quant_encode(cb, vector, dimension, codes); }
float gv_quant_distance(const GV_QuantCodebook *cb, const float *query, size_t dimension, const uint8_t *codes) { return quant_distance(cb, query, dimension, codes); }
size_t gv_quant_code_size(const GV_QuantCodebook *cb, size_t dimension) { return quant_code_size(cb, dimension); }
int gv_quant_codebook_save(const GV_QuantCodebook *cb, const char *path) { return quant_codebook_save(cb, path); }
GV_QuantCodebook * gv_quant_codebook_load(const char *path) { return quant_codebook_load(path); }
void gv_quant_codebook_destroy(GV_QuantCodebook *cb) { quant_codebook_destroy(cb); }
float gv_quant_memory_ratio(const GV_QuantCodebook *cb, size_t dimension) { return quant_memory_ratio(cb, dimension); }
void gv_ft_config_init(GV_FTConfig *config) { ft_config_init(config); }
GV_FTIndex * gv_ft_create(const GV_FTConfig *config) { return ft_create(config); }
void gv_ft_destroy(GV_FTIndex *idx) { ft_destroy(idx); }
int gv_ft_add_document(GV_FTIndex *idx, size_t doc_id, const char *text) { return ft_add_document(idx, doc_id, text); }
int gv_ft_remove_document(GV_FTIndex *idx, size_t doc_id) { return ft_remove_document(idx, doc_id); }
int gv_ft_search(const GV_FTIndex *idx, const char *query, size_t limit, GV_FTResult *results) { return ft_search(idx, query, limit, results); }
int gv_ft_search_phrase(const GV_FTIndex *idx, const char *phrase, size_t limit, GV_FTResult *results) { return ft_search_phrase(idx, phrase, limit, results); }
int gv_ft_stem(const char *word, GV_FTLanguage lang, char *output, size_t output_size) { return ft_stem(word, lang, output, output_size); }
void gv_ft_free_results(GV_FTResult *results, size_t count) { ft_free_results(results, count); }
size_t gv_ft_doc_count(const GV_FTIndex *idx) { return ft_doc_count(idx); }
int gv_ft_save(const GV_FTIndex *idx, const char *path) { return ft_save(idx, path); }
GV_FTIndex * gv_ft_load(const char *path) { return ft_load(path); }
GV_HNSWInlineIndex * gv_hnsw_inline_create(size_t dimension, size_t max_elements, size_t M, size_t ef_construction, const GV_HNSWInlineConfig *config) { return hnsw_inline_create(dimension, max_elements, M, ef_construction, config); }
void gv_hnsw_inline_destroy(GV_HNSWInlineIndex *idx) { hnsw_inline_destroy(idx); }
int gv_hnsw_inline_insert(GV_HNSWInlineIndex *idx, const float *vector, size_t label) { return hnsw_inline_insert(idx, vector, label); }
int gv_hnsw_inline_search(const GV_HNSWInlineIndex *idx, const float *query, size_t k, size_t ef_search, size_t *labels, float *distances) { return hnsw_inline_search(idx, query, k, ef_search, labels, distances); }
int gv_hnsw_inline_rebuild(GV_HNSWInlineIndex *idx, const GV_HNSWRebuildConfig *config) { return hnsw_inline_rebuild(idx, config); }
int gv_hnsw_inline_rebuild_status(const GV_HNSWInlineIndex *idx, GV_HNSWRebuildStats *stats) { return hnsw_inline_rebuild_status(idx, stats); }
size_t gv_hnsw_inline_count(const GV_HNSWInlineIndex *idx) { return hnsw_inline_count(idx); }
int gv_hnsw_inline_save(const GV_HNSWInlineIndex *idx, const char *path) { return hnsw_inline_save(idx, path); }
GV_HNSWInlineIndex * gv_hnsw_inline_load(const char *path) { return hnsw_inline_load(path); }
int gv_onnx_available(void) { return onnx_available(); }
GV_ONNXModel * gv_onnx_load(const GV_ONNXConfig *config) { return onnx_load(config); }
void gv_onnx_destroy(GV_ONNXModel *model) { onnx_destroy(model); }
int gv_onnx_rerank(GV_ONNXModel *model, const char *query_text, const char **doc_texts, size_t doc_count, float *scores) { return onnx_rerank(model, query_text, doc_texts, doc_count, scores); }
int gv_onnx_embed(GV_ONNXModel *model, const char **texts, size_t text_count, float *embeddings, size_t dimension) { return onnx_embed(model, texts, text_count, embeddings, dimension); }
GV_Agent * gv_agent_create(const void *db, const GV_AgentConfig *config) { return agent_create(db, config); }
void gv_agent_destroy(GV_Agent *agent) { agent_destroy(agent); }
GV_AgentResult * gv_agent_query(GV_Agent *agent, const char *natural_language_query, size_t k) { return agent_query(agent, natural_language_query, k); }
GV_AgentResult * gv_agent_transform(GV_Agent *agent, const char *natural_language_instruction) { return agent_transform(agent, natural_language_instruction); }
GV_AgentResult * gv_agent_personalize(GV_Agent *agent, const char *query, const char *user_profile_json, size_t k) { return agent_personalize(agent, query, user_profile_json, k); }
void gv_agent_free_result(GV_AgentResult *result) { agent_free_result(result); }
void gv_agent_set_schema_hint(GV_Agent *agent, const char *schema_json) { agent_set_schema_hint(agent, schema_json); }
void gv_muvera_config_init(GV_MuveraConfig *config) { muvera_config_init(config); }
GV_MuveraEncoder * gv_muvera_create(const GV_MuveraConfig *config) { return muvera_create(config); }
void gv_muvera_destroy(GV_MuveraEncoder *enc) { muvera_destroy(enc); }
int gv_muvera_encode(const GV_MuveraEncoder *enc, const float *tokens, size_t num_tokens, float *output) { return muvera_encode(enc, tokens, num_tokens, output); }
size_t gv_muvera_output_dimension(const GV_MuveraEncoder *enc) { return muvera_output_dimension(enc); }
int gv_muvera_save(const GV_MuveraEncoder *enc, const char *path) { return muvera_save(enc, path); }
GV_MuveraEncoder * gv_muvera_load(const char *path) { return muvera_load(path); }
GV_SSOManager * gv_sso_create(const GV_SSOConfig *config) { return sso_create(config); }
void gv_sso_destroy(GV_SSOManager *mgr) { sso_destroy(mgr); }
int gv_sso_discover(GV_SSOManager *mgr) { return sso_discover(mgr); }
int gv_sso_get_auth_url(const GV_SSOManager *mgr, const char *state, char *url, size_t url_size) { return sso_get_auth_url(mgr, state, url, url_size); }
GV_SSOToken * gv_sso_validate_token(GV_SSOManager *mgr, const char *token_string) { return sso_validate_token(mgr, token_string); }
void gv_sso_free_token(GV_SSOToken *token) { sso_free_token(token); }
void gv_tiered_config_init(GV_TieredTenantConfig *config) { tiered_config_init(config); }
GV_TieredManager * gv_tiered_create(const GV_TieredTenantConfig *config) { return tiered_create(config); }
void gv_tiered_destroy(GV_TieredManager *mgr) { tiered_destroy(mgr); }
int gv_tiered_add_tenant(GV_TieredManager *mgr, const char *tenant_id, GV_TenantTier initial_tier) { return tiered_add_tenant(mgr, tenant_id, initial_tier); }
int gv_tiered_remove_tenant(GV_TieredManager *mgr, const char *tenant_id) { return tiered_remove_tenant(mgr, tenant_id); }
int gv_tiered_promote(GV_TieredManager *mgr, const char *tenant_id, GV_TenantTier new_tier) { return tiered_promote(mgr, tenant_id, new_tier); }
int gv_tiered_get_info(const GV_TieredManager *mgr, const char *tenant_id, GV_TenantInfo *info) { return tiered_get_info(mgr, tenant_id, info); }
int gv_tiered_record_usage(GV_TieredManager *mgr, const char *tenant_id, size_t vectors_delta, size_t memory_delta) { return tiered_record_usage(mgr, tenant_id, vectors_delta, memory_delta); }
size_t gv_tiered_tenant_count(const GV_TieredManager *mgr) { return tiered_tenant_count(mgr); }
int gv_tiered_save(const GV_TieredManager *mgr, const char *path) { return tiered_save(mgr, path); }
GV_TieredManager * gv_tiered_load(const char *path) { return tiered_load(path); }
void gv_inference_config_init(GV_InferenceConfig *config) { inference_config_init(config); }
GV_InferenceEngine * gv_inference_create(void *db, const GV_InferenceConfig *config) { return inference_create(db, config); }
void gv_inference_destroy(GV_InferenceEngine *eng) { inference_destroy(eng); }
int gv_inference_add(GV_InferenceEngine *eng, const char *text, const char *metadata_json) { return inference_add(eng, text, metadata_json); }
int gv_inference_search(GV_InferenceEngine *eng, const char *query_text, size_t k, GV_InferenceResult *results) { return inference_search(eng, query_text, k, results); }
void gv_inference_free_results(GV_InferenceResult *results, size_t count) { inference_free_results(results, count); }
void gv_embedded_config_init(GV_EmbeddedConfig *config) { embedded_config_init(config); }
GV_EmbeddedDB * gv_embedded_open(const GV_EmbeddedConfig *config) { return embedded_open(config); }
void gv_embedded_close(GV_EmbeddedDB *db) { embedded_close(db); }
int gv_embedded_add(GV_EmbeddedDB *db, const float *vector) { return embedded_add(db, vector); }
int gv_embedded_search(const GV_EmbeddedDB *db, const float *query, size_t k, int distance_type, GV_EmbeddedResult *results) { return embedded_search(db, query, k, distance_type, results); }
int gv_embedded_delete(GV_EmbeddedDB *db, size_t index) { return embedded_delete(db, index); }
size_t gv_embedded_count(const GV_EmbeddedDB *db) { return embedded_count(db); }
size_t gv_embedded_memory_usage(const GV_EmbeddedDB *db) { return embedded_memory_usage(db); }
int gv_embedded_save(const GV_EmbeddedDB *db, const char *path) { return embedded_save(db, path); }
GV_EmbeddedDB * gv_embedded_load(const char *path) { return embedded_load(path); }
int gv_embedded_compact(GV_EmbeddedDB *db) { return embedded_compact(db); }
void gv_media_config_init(GV_MediaConfig *config) { media_config_init(config); }
GV_MediaStore * gv_media_create(const GV_MediaConfig *config) { return media_create(config); }
void gv_media_destroy(GV_MediaStore *store) { media_destroy(store); }
int gv_media_store_blob(GV_MediaStore *store, size_t vector_index, GV_MediaType type, const void *data, size_t data_size, const char *filename, const char *mime_type) { return media_store_blob(store, vector_index, type, data, data_size, filename, mime_type); }
int gv_media_store_file(GV_MediaStore *store, size_t vector_index, GV_MediaType type, const char *file_path) { return media_store_file(store, vector_index, type, file_path); }
int gv_media_retrieve(const GV_MediaStore *store, size_t vector_index, void *buffer, size_t buffer_size, size_t *actual_size) { return media_retrieve(store, vector_index, buffer, buffer_size, actual_size); }
int gv_media_get_info(const GV_MediaStore *store, size_t vector_index, GV_MediaEntry *entry) { return media_get_info(store, vector_index, entry); }
int gv_media_delete(GV_MediaStore *store, size_t vector_index) { return media_delete(store, vector_index); }
int gv_media_exists(const GV_MediaStore *store, size_t vector_index) { return media_exists(store, vector_index); }
size_t gv_media_count(const GV_MediaStore *store) { return media_count(store); }
size_t gv_media_total_size(const GV_MediaStore *store) { return media_total_size(store); }
int gv_media_save_index(const GV_MediaStore *store, const char *path) { return media_save_index(store, path); }
GV_MediaStore * gv_media_load_index(const char *index_path, const char *storage_dir) { return media_load_index(index_path, storage_dir); }
