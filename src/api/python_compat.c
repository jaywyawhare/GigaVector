#include "admin/cluster.h"
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
#include "multimodal/bm25.h"
#include "multimodal/embedding.h"
#include "multimodal/llm.h"
#include "schema/metadata.h"
#include "schema/vector.h"
#include "search/mmr.h"
#include "specialized/gpu.h"
#include "storage/backup.h"
#include "storage/database.h"
#include "storage/memory_extraction.h"
#include "storage/memory_layer.h"
#include "features/knowledge_graph.h"
#include "storage/snapshot.h"
#include "storage/wal.h"
#include "storage/posting_list.h"

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

/* ── Database: open variants ── */

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

/* ── Database: misc ── */

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

/* ── Database: vector CRUD ── */

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

int gv_db_ivfsq8_train(GV_Database *db, const float *data, size_t count,
                       size_t dimension) {
  return db_ivfsq8_train(db, data, count, dimension);
}

int gv_db_ivfturboquant_train(GV_Database *db, const float *data, size_t count,
                              size_t dimension) {
  return db_ivfturboquant_train(db, data, count, dimension);
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

/* ── Database: search ── */

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

/* ── Database: exact search config ── */

void gv_db_set_exact_search_threshold(GV_Database *db, size_t threshold) {
  db_set_exact_search_threshold(db, threshold);
}

void gv_db_set_force_exact_search(GV_Database *db, int enabled) {
  db_set_force_exact_search(db, enabled);
}

/* ── Database: resource limits ── */

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

/* ── Database: compaction ── */

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

/* ── Database: observability ── */

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

/* ── Database: accessors ── */

size_t gv_database_count(const GV_Database *db) { return database_count(db); }

size_t gv_database_dimension(const GV_Database *db) {
  return database_dimension(db);
}

const float *gv_database_get_vector(const GV_Database *db, size_t index) {
  return database_get_vector(db, index);
}

/* ── Database: JSON import/export ── */

int gv_db_export_json(const GV_Database *db, const char *filepath) {
  return db_export_json(db, filepath);
}

int gv_db_import_json(GV_Database *db, const char *filepath) {
  return db_import_json(db, filepath);
}

/* ── Vector ── */

GV_Vector *gv_vector_create_from_data(size_t dimension, const float *data) {
  return vector_create_from_data(dimension, data);
}

int gv_vector_set_metadata(GV_Vector *vector, const char *key,
                           const char *value) {
  return vector_set_metadata(vector, key, value);
}

void gv_vector_destroy(GV_Vector *vector) { vector_destroy(vector); }

/* ── KD-tree: gv_kdtree_insert has a different signature from the underlying
   kdtree_insert (which requires SoA storage context). Stub returns -1. ── */
int gv_kdtree_insert(GV_KDNode **root, GV_Vector *point, size_t depth) {
  (void)root;
  (void)point;
  (void)depth;
  return -1;
}

/* ── WAL ── */

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

/* ── LLM ── */

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

/* ── Embedding ── */

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

/* ── Context graph ── */

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

/* ── Memory layer ── */

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

/* ── GPU ── */

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

/* ── Server ── */

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

/* ── Backup ── */

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

/* ── Shard ── */

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

/* ── Replication (missing wrappers) ── */

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

/* ── Cluster ── */

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

/* ── Namespace ── */

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

/* ── BloomFilter wrappers ─────────────────────────────────────────────────── */

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

/* ── BM25 wrappers ────────────────────────────────────────────────────────── */

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

/* ── SnapshotManager wrappers ─────────────────────────────────────────────── */

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

/* ── MMR wrappers ─────────────────────────────────────────────────────────── */

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

/* ── Knowledge graph wrappers ─────────────────────────────────────────────── */

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

void gv_free(void *ptr) { free(ptr); }

/* ── SQL engine wrappers ──────────────────────────────────────────────────── */

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

/* ── Phased ranking pipeline wrappers ───────────────────────────────────── */

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

/* ── Learned sparse index wrappers ────────────────────────────────────────── */

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

/* ── Graph database wrappers ──────────────────────────────────────────────── */

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

GV_GraphDB *gv_graph_load(const char *path) { return graph_load(path); }

/* ── Knowledge graph wrappers ─────────────────────────────────────────────── */

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

int gv_kg_search_similar(const GV_KnowledgeGraph *kg,
                         const float *query_embedding, size_t dimension,
                         size_t k, GV_KGSearchResult *results) {
  return kg_search_similar(kg, query_embedding, dimension, k, results);
}

int gv_kg_search_by_text(const GV_KnowledgeGraph *kg, const char *text,
                         const float *text_embedding, size_t dimension, size_t k,
                         GV_KGSearchResult *results) {
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

float gv_kg_entity_centrality(const GV_KnowledgeGraph *kg, uint64_t entity_id) {
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