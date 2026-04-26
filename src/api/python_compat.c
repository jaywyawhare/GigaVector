#include "admin/cluster.h"
#include "admin/namespace.h"
#include "admin/replication.h"
#include "admin/shard.h"
#include "api/server.h"
#include "core/bloom.h"
#include "features/context_graph.h"
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
#include "storage/snapshot.h"
#include "storage/wal.h"

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
  return memory_add(layer, content, embedding, metadata);
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