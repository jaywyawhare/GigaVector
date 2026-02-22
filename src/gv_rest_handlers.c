/**
 * @file gv_rest_handlers.c
 * @brief REST API endpoint handlers implementation.
 */

#include "gigavector/gv_rest_handlers.h"
#include "gigavector/gv_json.h"
#include "gigavector/gv_database.h"
#include "gigavector/gv_types.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>

/*  Response Helpers  */

GV_HttpResponse *gv_rest_response_json(GV_JsonValue *data) {
    if (!data) {
        return NULL;
    }

    GV_HttpResponse *response = calloc(1, sizeof(GV_HttpResponse));
    if (!response) {
        gv_json_free(data);
        return NULL;
    }

    response->body = gv_json_stringify(data, false);
    gv_json_free(data);

    if (!response->body) {
        free(response);
        return NULL;
    }

    response->body_length = strlen(response->body);
    response->status = GV_HTTP_200_OK;
    response->content_type = "application/json";

    return response;
}

GV_HttpResponse *gv_rest_response_error(GV_HttpStatus status,
                                         const char *error_code,
                                         const char *message) {
    GV_JsonValue *obj = gv_json_object();
    if (!obj) return NULL;

    gv_json_object_set(obj, "error", gv_json_string(error_code ? error_code : "error"));
    gv_json_object_set(obj, "message", gv_json_string(message ? message : "An error occurred"));

    GV_HttpResponse *response = gv_rest_response_json(obj);
    if (response) {
        response->status = status;
    }
    return response;
}

GV_HttpResponse *gv_rest_response_success(const char *message) {
    GV_JsonValue *obj = gv_json_object();
    if (!obj) return NULL;

    gv_json_object_set(obj, "success", gv_json_bool(true));
    if (message) {
        gv_json_object_set(obj, "message", gv_json_string(message));
    }

    return gv_rest_response_json(obj);
}

void gv_rest_response_free(GV_HttpResponse *response) {
    if (!response) return;
    free(response->body);
    free(response);
}

/*  Request Parsing Helpers  */

int gv_rest_parse_path_param(const char *url, const char *prefix,
                              char *param_out, size_t param_size) {
    if (!url || !prefix || !param_out || param_size == 0) {
        return -1;
    }

    size_t prefix_len = strlen(prefix);
    if (strncmp(url, prefix, prefix_len) != 0) {
        return -1;
    }

    const char *param = url + prefix_len;
    const char *end = strchr(param, '/');
    const char *query = strchr(param, '?');

    size_t len;
    if (end) {
        len = end - param;
    } else if (query) {
        len = query - param;
    } else {
        len = strlen(param);
    }

    if (len == 0 || len >= param_size) {
        return -1;
    }

    memcpy(param_out, param, len);
    param_out[len] = '\0';

    return 0;
}

int gv_rest_parse_query_param(const char *query_string, const char *key,
                               char *value_out, size_t value_size) {
    if (!query_string || !key || !value_out || value_size == 0) {
        return -1;
    }

    size_t key_len = strlen(key);
    const char *p = query_string;

    while (*p) {
        if (strncmp(p, key, key_len) == 0 && p[key_len] == '=') {
            const char *val_start = p + key_len + 1;
            const char *val_end = strchr(val_start, '&');
            size_t val_len = val_end ? (size_t)(val_end - val_start) : strlen(val_start);

            if (val_len >= value_size) {
                return -1;
            }

            memcpy(value_out, val_start, val_len);
            value_out[val_len] = '\0';
            return 0;
        }

        /* Move to next parameter */
        const char *next = strchr(p, '&');
        if (!next) break;
        p = next + 1;
    }

    return -1;
}

GV_JsonValue *gv_rest_parse_body(const GV_HttpRequest *request, GV_JsonError *error) {
    if (!request || !request->body || request->body_length == 0) {
        if (error) *error = GV_JSON_ERROR_NULL_INPUT;
        return NULL;
    }

    return gv_json_parse(request->body, error);
}

/*  Distance Type Parsing  */

static GV_DistanceType parse_distance_type(const char *str) {
    if (!str) return GV_DISTANCE_EUCLIDEAN;
    if (strcmp(str, "euclidean") == 0) return GV_DISTANCE_EUCLIDEAN;
    if (strcmp(str, "cosine") == 0) return GV_DISTANCE_COSINE;
    if (strcmp(str, "dot_product") == 0) return GV_DISTANCE_DOT_PRODUCT;
    if (strcmp(str, "manhattan") == 0) return GV_DISTANCE_MANHATTAN;
    return GV_DISTANCE_EUCLIDEAN;
}

/*  Endpoint Handlers  */

GV_HttpResponse *gv_rest_handle_health(const GV_HandlerContext *ctx,
                                        const GV_HttpRequest *request) {
    (void)request;

    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    int health = gv_db_health_check(ctx->db);
    const char *status_str;
    switch (health) {
        case 0:  status_str = "healthy"; break;
        case -1: status_str = "degraded"; break;
        default: status_str = "unhealthy"; break;
    }

    GV_JsonValue *obj = gv_json_object();
    gv_json_object_set(obj, "status", gv_json_string(status_str));
    gv_json_object_set(obj, "vector_count", gv_json_number((double)ctx->db->count));

    return gv_rest_response_json(obj);
}

GV_HttpResponse *gv_rest_handle_stats(const GV_HandlerContext *ctx,
                                       const GV_HttpRequest *request) {
    (void)request;

    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    GV_DBStats stats;
    gv_db_get_stats(ctx->db, &stats);
    size_t memory = gv_db_get_memory_usage(ctx->db);

    GV_JsonValue *obj = gv_json_object();
    gv_json_object_set(obj, "total_vectors", gv_json_number((double)ctx->db->count));
    gv_json_object_set(obj, "total_inserts", gv_json_number((double)stats.total_inserts));
    gv_json_object_set(obj, "total_queries", gv_json_number((double)stats.total_queries));
    gv_json_object_set(obj, "total_range_queries", gv_json_number((double)stats.total_range_queries));
    gv_json_object_set(obj, "memory_bytes", gv_json_number((double)memory));
    gv_json_object_set(obj, "dimension", gv_json_number((double)ctx->db->dimension));

    return gv_rest_response_json(obj);
}

GV_HttpResponse *gv_rest_handle_vectors_post(const GV_HandlerContext *ctx,
                                              const GV_HttpRequest *request) {
    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    GV_JsonError error;
    GV_JsonValue *body = gv_rest_parse_body(request, &error);
    if (!body) {
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_json",
                                       gv_json_error_string(error));
    }

    /* Check for batch vs single vector */
    GV_JsonValue *vectors_arr = gv_json_object_get(body, "vectors");
    GV_JsonValue *data_arr = gv_json_object_get(body, "data");

    size_t inserted = 0;
    GV_JsonValue *indices = gv_json_array();

    if (vectors_arr && gv_json_is_array(vectors_arr)) {
        /* Batch insert */
        size_t count = gv_json_array_length(vectors_arr);
        for (size_t i = 0; i < count; i++) {
            GV_JsonValue *vec_obj = gv_json_array_get(vectors_arr, i);
            GV_JsonValue *vec_data = gv_json_object_get(vec_obj, "data");

            if (!vec_data || !gv_json_is_array(vec_data)) {
                continue;
            }

            size_t dim = gv_json_array_length(vec_data);
            if (dim != ctx->db->dimension) {
                continue;
            }

            float *vec = malloc(dim * sizeof(float));
            if (!vec) continue;

            for (size_t j = 0; j < dim; j++) {
                GV_JsonValue *v = gv_json_array_get(vec_data, j);
                double num;
                if (gv_json_get_number(v, &num) == GV_JSON_OK) {
                    vec[j] = (float)num;
                }
            }

            /* Handle metadata */
            GV_JsonValue *metadata = gv_json_object_get(vec_obj, "metadata");
            int result;

            if (metadata && gv_json_is_object(metadata)) {
                size_t meta_count = gv_json_object_length(metadata);
                if (meta_count > 0) {
                    const char **keys = malloc(meta_count * sizeof(char *));
                    const char **values = malloc(meta_count * sizeof(char *));
                    char **value_buffers = malloc(meta_count * sizeof(char *));

                    if (keys && values && value_buffers) {
                        /* Extract all key-value pairs */
                        GV_JsonEntry *entries = metadata->data.object.entries;
                        for (size_t m = 0; m < meta_count; m++) {
                            keys[m] = entries[m].key;
                            /* Convert value to string if needed */
                            if (gv_json_is_string(entries[m].value)) {
                                values[m] = gv_json_get_string(entries[m].value);
                                value_buffers[m] = NULL;
                            } else {
                                /* Convert non-string values to string */
                                value_buffers[m] = gv_json_stringify(entries[m].value, false);
                                values[m] = value_buffers[m];
                            }
                        }

                        result = gv_db_add_vector_with_rich_metadata(ctx->db, vec, dim,
                                                                      keys, values, meta_count);

                        /* Free converted value buffers */
                        for (size_t m = 0; m < meta_count; m++) {
                            free(value_buffers[m]);
                        }
                    } else {
                        result = gv_db_add_vector(ctx->db, vec, dim);
                    }

                    free(keys);
                    free(values);
                    free(value_buffers);
                } else {
                    result = gv_db_add_vector(ctx->db, vec, dim);
                }
            } else {
                result = gv_db_add_vector(ctx->db, vec, dim);
            }

            free(vec);

            if (result == 0) {
                gv_json_array_push(indices, gv_json_number((double)(ctx->db->count - 1)));
                inserted++;
            }
        }
    } else if (data_arr && gv_json_is_array(data_arr)) {
        /* Single vector */
        size_t dim = gv_json_array_length(data_arr);
        if (dim != ctx->db->dimension) {
            gv_json_free(body);
            gv_json_free(indices);
            return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "dimension_mismatch",
                                           "Vector dimension does not match database");
        }

        float *vec = malloc(dim * sizeof(float));
        if (!vec) {
            gv_json_free(body);
            gv_json_free(indices);
            return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "memory_error",
                                           "Failed to allocate memory");
        }

        for (size_t j = 0; j < dim; j++) {
            GV_JsonValue *v = gv_json_array_get(data_arr, j);
            double num;
            if (gv_json_get_number(v, &num) == GV_JSON_OK) {
                vec[j] = (float)num;
            }
        }

        GV_JsonValue *metadata = gv_json_object_get(body, "metadata");
        int result;

        if (metadata && gv_json_is_object(metadata)) {
            size_t meta_count = gv_json_object_length(metadata);
            if (meta_count > 0) {
                GV_JsonEntry *entries = metadata->data.object.entries;
                result = gv_db_add_vector_with_metadata(ctx->db, vec, dim,
                                                         entries[0].key,
                                                         gv_json_get_string(entries[0].value));
            } else {
                result = gv_db_add_vector(ctx->db, vec, dim);
            }
        } else {
            result = gv_db_add_vector(ctx->db, vec, dim);
        }

        free(vec);

        if (result == 0) {
            gv_json_array_push(indices, gv_json_number((double)(ctx->db->count - 1)));
            inserted++;
        }
    } else {
        gv_json_free(body);
        gv_json_free(indices);
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_request",
                                       "Missing 'data' or 'vectors' field");
    }

    gv_json_free(body);

    GV_JsonValue *response = gv_json_object();
    gv_json_object_set(response, "success", gv_json_bool(true));
    gv_json_object_set(response, "inserted", gv_json_number((double)inserted));
    gv_json_object_set(response, "indices", indices);

    GV_HttpResponse *http_resp = gv_rest_response_json(response);
    if (http_resp) {
        http_resp->status = GV_HTTP_201_CREATED;
    }
    return http_resp;
}

GV_HttpResponse *gv_rest_handle_vectors_get(const GV_HandlerContext *ctx,
                                             const GV_HttpRequest *request,
                                             size_t vector_index) {
    (void)request;

    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    if (vector_index >= ctx->db->count) {
        return gv_rest_response_error(GV_HTTP_404_NOT_FOUND, "not_found",
                                       "Vector not found");
    }

    /* Get vector from SoA storage */
    if (!ctx->db->soa_storage) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Storage not available");
    }

    GV_SoAStorage *storage = ctx->db->soa_storage;
    if (vector_index >= storage->count) {
        return gv_rest_response_error(GV_HTTP_404_NOT_FOUND, "not_found",
                                       "Vector not found");
    }

    /* Check if deleted */
    if (storage->deleted && storage->deleted[vector_index]) {
        return gv_rest_response_error(GV_HTTP_404_NOT_FOUND, "not_found",
                                       "Vector has been deleted");
    }

    GV_JsonValue *obj = gv_json_object();
    gv_json_object_set(obj, "index", gv_json_number((double)vector_index));

    /* Add vector data */
    GV_JsonValue *data_arr = gv_json_array();
    float *vec_data = storage->data + vector_index * ctx->db->dimension;
    for (size_t i = 0; i < ctx->db->dimension; i++) {
        gv_json_array_push(data_arr, gv_json_number((double)vec_data[i]));
    }
    gv_json_object_set(obj, "data", data_arr);

    /* Add metadata */
    GV_JsonValue *meta_obj = gv_json_object();
    if (storage->metadata && storage->metadata[vector_index]) {
        GV_Metadata *meta = storage->metadata[vector_index];
        while (meta) {
            gv_json_object_set(meta_obj, meta->key, gv_json_string(meta->value));
            meta = meta->next;
        }
    }
    gv_json_object_set(obj, "metadata", meta_obj);

    return gv_rest_response_json(obj);
}

GV_HttpResponse *gv_rest_handle_vectors_put(const GV_HandlerContext *ctx,
                                             const GV_HttpRequest *request,
                                             size_t vector_index) {
    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    if (vector_index >= ctx->db->count) {
        return gv_rest_response_error(GV_HTTP_404_NOT_FOUND, "not_found",
                                       "Vector not found");
    }

    GV_JsonError error;
    GV_JsonValue *body = gv_rest_parse_body(request, &error);
    if (!body) {
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_json",
                                       gv_json_error_string(error));
    }

    GV_JsonValue *data_arr = gv_json_object_get(body, "data");
    if (data_arr && gv_json_is_array(data_arr)) {
        size_t dim = gv_json_array_length(data_arr);
        if (dim != ctx->db->dimension) {
            gv_json_free(body);
            return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "dimension_mismatch",
                                           "Vector dimension does not match database");
        }

        float *vec = malloc(dim * sizeof(float));
        if (!vec) {
            gv_json_free(body);
            return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "memory_error",
                                           "Failed to allocate memory");
        }

        for (size_t j = 0; j < dim; j++) {
            GV_JsonValue *v = gv_json_array_get(data_arr, j);
            double num;
            if (gv_json_get_number(v, &num) == GV_JSON_OK) {
                vec[j] = (float)num;
            }
        }

        int result = gv_db_update_vector(ctx->db, vector_index, vec, dim);
        free(vec);

        if (result != 0) {
            gv_json_free(body);
            return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "update_failed",
                                           "Failed to update vector");
        }
    }

    gv_json_free(body);
    return gv_rest_response_success("Vector updated");
}

GV_HttpResponse *gv_rest_handle_vectors_delete(const GV_HandlerContext *ctx,
                                                const GV_HttpRequest *request,
                                                size_t vector_index) {
    (void)request;

    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    int result = gv_db_delete_vector_by_index(ctx->db, vector_index);
    if (result != 0) {
        return gv_rest_response_error(GV_HTTP_404_NOT_FOUND, "not_found",
                                       "Vector not found or already deleted");
    }

    GV_JsonValue *obj = gv_json_object();
    gv_json_object_set(obj, "success", gv_json_bool(true));
    gv_json_object_set(obj, "deleted_index", gv_json_number((double)vector_index));

    return gv_rest_response_json(obj);
}

GV_HttpResponse *gv_rest_handle_search(const GV_HandlerContext *ctx,
                                        const GV_HttpRequest *request) {
    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    GV_JsonError error;
    GV_JsonValue *body = gv_rest_parse_body(request, &error);
    if (!body) {
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_json",
                                       gv_json_error_string(error));
    }

    /* Parse query vector */
    GV_JsonValue *query_arr = gv_json_object_get(body, "query");
    if (!query_arr || !gv_json_is_array(query_arr)) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_request",
                                       "Missing 'query' array");
    }

    size_t dim = gv_json_array_length(query_arr);
    if (dim != ctx->db->dimension) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "dimension_mismatch",
                                       "Query dimension does not match database");
    }

    float *query = malloc(dim * sizeof(float));
    if (!query) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "memory_error",
                                       "Failed to allocate memory");
    }

    for (size_t i = 0; i < dim; i++) {
        GV_JsonValue *v = gv_json_array_get(query_arr, i);
        double num;
        if (gv_json_get_number(v, &num) == GV_JSON_OK) {
            query[i] = (float)num;
        }
    }

    /* Parse k */
    GV_JsonValue *k_val = gv_json_object_get(body, "k");
    double k_num;
    size_t k = 10;
    if (k_val && gv_json_get_number(k_val, &k_num) == GV_JSON_OK) {
        k = (size_t)k_num;
    }

    /* Parse distance type */
    const char *dist_str = gv_json_get_string_path(body, "distance");
    GV_DistanceType distance = parse_distance_type(dist_str);

    /* Allocate results */
    GV_SearchResult *results = calloc(k, sizeof(GV_SearchResult));
    if (!results) {
        free(query);
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "memory_error",
                                       "Failed to allocate memory");
    }

    /* Perform search */
    clock_t start = clock();
    int found;

    /* Check for filter */
    GV_JsonValue *filter = gv_json_object_get(body, "filter");
    if (filter && gv_json_is_object(filter) && gv_json_object_length(filter) > 0) {
        GV_JsonEntry *entries = filter->data.object.entries;
        found = gv_db_search_filtered(ctx->db, query, k, results, distance,
                                       entries[0].key,
                                       gv_json_get_string(entries[0].value));
    } else {
        found = gv_db_search(ctx->db, query, k, results, distance);
    }

    clock_t end = clock();
    double latency_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    free(query);
    gv_json_free(body);

    if (found < 0) {
        free(results);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "search_failed",
                                       "Search operation failed");
    }

    /* Build response */
    GV_JsonValue *response = gv_json_object();
    GV_JsonValue *results_arr = gv_json_array();

    for (int i = 0; i < found; i++) {
        GV_JsonValue *result_obj = gv_json_object();
        gv_json_object_set(result_obj, "distance", gv_json_number(results[i].distance));

        /* Add vector data and metadata if available */
        if (results[i].vector) {
            GV_JsonValue *data_arr = gv_json_array();
            for (size_t j = 0; j < ctx->db->dimension; j++) {
                gv_json_array_push(data_arr, gv_json_number((double)results[i].vector->data[j]));
            }
            gv_json_object_set(result_obj, "data", data_arr);

            GV_JsonValue *meta_obj = gv_json_object();
            GV_Metadata *meta = results[i].vector->metadata;
            while (meta) {
                gv_json_object_set(meta_obj, meta->key, gv_json_string(meta->value));
                meta = meta->next;
            }
            gv_json_object_set(result_obj, "metadata", meta_obj);
        }

        gv_json_array_push(results_arr, result_obj);
    }

    free(results);

    gv_json_object_set(response, "results", results_arr);
    gv_json_object_set(response, "count", gv_json_number((double)found));
    gv_json_object_set(response, "latency_ms", gv_json_number(latency_ms));

    return gv_rest_response_json(response);
}

GV_HttpResponse *gv_rest_handle_search_range(const GV_HandlerContext *ctx,
                                              const GV_HttpRequest *request) {
    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    GV_JsonError error;
    GV_JsonValue *body = gv_rest_parse_body(request, &error);
    if (!body) {
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_json",
                                       gv_json_error_string(error));
    }

    /* Parse query vector */
    GV_JsonValue *query_arr = gv_json_object_get(body, "query");
    if (!query_arr || !gv_json_is_array(query_arr)) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_request",
                                       "Missing 'query' array");
    }

    size_t dim = gv_json_array_length(query_arr);
    if (dim != ctx->db->dimension) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "dimension_mismatch",
                                       "Query dimension does not match database");
    }

    float *query = malloc(dim * sizeof(float));
    if (!query) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "memory_error",
                                       "Failed to allocate memory");
    }

    for (size_t i = 0; i < dim; i++) {
        GV_JsonValue *v = gv_json_array_get(query_arr, i);
        double num;
        if (gv_json_get_number(v, &num) == GV_JSON_OK) {
            query[i] = (float)num;
        }
    }

    /* Parse radius */
    GV_JsonValue *radius_val = gv_json_object_get(body, "radius");
    double radius_num;
    float radius = 1.0f;
    if (radius_val && gv_json_get_number(radius_val, &radius_num) == GV_JSON_OK) {
        radius = (float)radius_num;
    }

    /* Parse max_results */
    GV_JsonValue *max_val = gv_json_object_get(body, "max_results");
    double max_num;
    size_t max_results = 100;
    if (max_val && gv_json_get_number(max_val, &max_num) == GV_JSON_OK) {
        max_results = (size_t)max_num;
    }

    /* Parse distance type */
    const char *dist_str = gv_json_get_string_path(body, "distance");
    GV_DistanceType distance = parse_distance_type(dist_str);

    /* Allocate results */
    GV_SearchResult *results = calloc(max_results, sizeof(GV_SearchResult));
    if (!results) {
        free(query);
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "memory_error",
                                       "Failed to allocate memory");
    }

    /* Perform range search */
    clock_t start = clock();
    int found = gv_db_range_search(ctx->db, query, radius, results, max_results, distance);
    clock_t end = clock();
    double latency_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    free(query);
    gv_json_free(body);

    if (found < 0) {
        free(results);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "search_failed",
                                       "Range search operation failed");
    }

    /* Build response */
    GV_JsonValue *response = gv_json_object();
    GV_JsonValue *results_arr = gv_json_array();

    for (int i = 0; i < found; i++) {
        GV_JsonValue *result_obj = gv_json_object();
        gv_json_object_set(result_obj, "distance", gv_json_number(results[i].distance));
        gv_json_array_push(results_arr, result_obj);
    }

    free(results);

    gv_json_object_set(response, "results", results_arr);
    gv_json_object_set(response, "count", gv_json_number((double)found));
    gv_json_object_set(response, "latency_ms", gv_json_number(latency_ms));

    return gv_rest_response_json(response);
}

GV_HttpResponse *gv_rest_handle_search_batch(const GV_HandlerContext *ctx,
                                              const GV_HttpRequest *request) {
    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    GV_JsonError error;
    GV_JsonValue *body = gv_rest_parse_body(request, &error);
    if (!body) {
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_json",
                                       gv_json_error_string(error));
    }

    /* Parse queries array */
    GV_JsonValue *queries_arr = gv_json_object_get(body, "queries");
    if (!queries_arr || !gv_json_is_array(queries_arr)) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_request",
                                       "Missing 'queries' array");
    }

    size_t qcount = gv_json_array_length(queries_arr);
    if (qcount == 0) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_400_BAD_REQUEST, "invalid_request",
                                       "Empty queries array");
    }

    /* Parse k */
    GV_JsonValue *k_val = gv_json_object_get(body, "k");
    double k_num;
    size_t k = 10;
    if (k_val && gv_json_get_number(k_val, &k_num) == GV_JSON_OK) {
        k = (size_t)k_num;
    }

    /* Parse distance type */
    const char *dist_str = gv_json_get_string_path(body, "distance");
    GV_DistanceType distance = parse_distance_type(dist_str);

    /* Allocate queries buffer */
    float *queries = malloc(qcount * ctx->db->dimension * sizeof(float));
    if (!queries) {
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "memory_error",
                                       "Failed to allocate memory");
    }

    for (size_t q = 0; q < qcount; q++) {
        GV_JsonValue *query_arr = gv_json_array_get(queries_arr, q);
        if (!query_arr || !gv_json_is_array(query_arr)) {
            continue;
        }
        for (size_t i = 0; i < ctx->db->dimension && i < gv_json_array_length(query_arr); i++) {
            GV_JsonValue *v = gv_json_array_get(query_arr, i);
            double num;
            if (gv_json_get_number(v, &num) == GV_JSON_OK) {
                queries[q * ctx->db->dimension + i] = (float)num;
            }
        }
    }

    /* Allocate results */
    GV_SearchResult *results = calloc(qcount * k, sizeof(GV_SearchResult));
    if (!results) {
        free(queries);
        gv_json_free(body);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "memory_error",
                                       "Failed to allocate memory");
    }

    /* Perform batch search */
    clock_t start = clock();
    int total = gv_db_search_batch(ctx->db, queries, qcount, k, results, distance);
    clock_t end = clock();
    double latency_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    free(queries);
    gv_json_free(body);

    if (total < 0) {
        free(results);
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "search_failed",
                                       "Batch search operation failed");
    }

    /* Build response */
    GV_JsonValue *response = gv_json_object();
    GV_JsonValue *batch_results = gv_json_array();

    for (size_t q = 0; q < qcount; q++) {
        GV_JsonValue *query_results = gv_json_array();
        for (size_t i = 0; i < k; i++) {
            GV_SearchResult *r = &results[q * k + i];
            if (r->vector) {
                GV_JsonValue *result_obj = gv_json_object();
                gv_json_object_set(result_obj, "distance", gv_json_number(r->distance));
                gv_json_array_push(query_results, result_obj);
            }
        }
        gv_json_array_push(batch_results, query_results);
    }

    free(results);

    gv_json_object_set(response, "results", batch_results);
    gv_json_object_set(response, "query_count", gv_json_number((double)qcount));
    gv_json_object_set(response, "latency_ms", gv_json_number(latency_ms));

    return gv_rest_response_json(response);
}

GV_HttpResponse *gv_rest_handle_compact(const GV_HandlerContext *ctx,
                                         const GV_HttpRequest *request) {
    (void)request;

    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    int result = gv_db_compact(ctx->db);
    if (result != 0) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "compact_failed",
                                       "Compaction operation failed");
    }

    return gv_rest_response_success("Compaction completed");
}

GV_HttpResponse *gv_rest_handle_save(const GV_HandlerContext *ctx,
                                      const GV_HttpRequest *request) {
    if (!ctx || !ctx->db) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Database not available");
    }

    const char *filepath = NULL;

    if (request->body && request->body_length > 0) {
        GV_JsonError error;
        GV_JsonValue *body = gv_rest_parse_body(request, &error);
        if (body) {
            filepath = gv_json_get_string_path(body, "filepath");
            gv_json_free(body);
        }
    }

    int result = gv_db_save(ctx->db, filepath);
    if (result != 0) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "save_failed",
                                       "Save operation failed");
    }

    GV_JsonValue *obj = gv_json_object();
    gv_json_object_set(obj, "success", gv_json_bool(true));
    gv_json_object_set(obj, "filepath", gv_json_string(filepath ? filepath :
                                                        (ctx->db->filepath ? ctx->db->filepath : "")));

    return gv_rest_response_json(obj);
}

/*  Router  */

GV_HttpResponse *gv_rest_route(const GV_HandlerContext *ctx,
                                const GV_HttpRequest *request) {
    if (!ctx || !request || !request->url) {
        return gv_rest_response_error(GV_HTTP_500_INTERNAL_ERROR, "internal_error",
                                       "Invalid request");
    }

    const char *url = request->url;

    /* Strip query string for matching */
    char url_path[256];
    const char *query_start = strchr(url, '?');
    if (query_start) {
        size_t len = query_start - url;
        if (len >= sizeof(url_path)) len = sizeof(url_path) - 1;
        memcpy(url_path, url, len);
        url_path[len] = '\0';
        url = url_path;
    }

    /* Route to handlers */

    /* GET /health */
    if (strcmp(url, "/health") == 0 && request->method == GV_HTTP_GET) {
        return gv_rest_handle_health(ctx, request);
    }

    /* GET /stats */
    if (strcmp(url, "/stats") == 0 && request->method == GV_HTTP_GET) {
        return gv_rest_handle_stats(ctx, request);
    }

    /* POST /vectors */
    if (strcmp(url, "/vectors") == 0 && request->method == GV_HTTP_POST) {
        return gv_rest_handle_vectors_post(ctx, request);
    }

    /* /vectors/{id} routes */
    if (strncmp(url, "/vectors/", 9) == 0) {
        char id_str[32];
        if (gv_rest_parse_path_param(url, "/vectors/", id_str, sizeof(id_str)) == 0) {
            char *endptr;
            size_t vector_index = strtoul(id_str, &endptr, 10);
            if (*endptr == '\0') {
                switch (request->method) {
                    case GV_HTTP_GET:
                        return gv_rest_handle_vectors_get(ctx, request, vector_index);
                    case GV_HTTP_PUT:
                        return gv_rest_handle_vectors_put(ctx, request, vector_index);
                    case GV_HTTP_DELETE:
                        return gv_rest_handle_vectors_delete(ctx, request, vector_index);
                    default:
                        return gv_rest_response_error(GV_HTTP_405_METHOD_NOT_ALLOWED,
                                                       "method_not_allowed",
                                                       "Method not allowed for this endpoint");
                }
            }
        }
    }

    /* POST /search */
    if (strcmp(url, "/search") == 0 && request->method == GV_HTTP_POST) {
        return gv_rest_handle_search(ctx, request);
    }

    /* POST /search/range */
    if (strcmp(url, "/search/range") == 0 && request->method == GV_HTTP_POST) {
        return gv_rest_handle_search_range(ctx, request);
    }

    /* POST /search/batch */
    if (strcmp(url, "/search/batch") == 0 && request->method == GV_HTTP_POST) {
        return gv_rest_handle_search_batch(ctx, request);
    }

    /* POST /compact */
    if (strcmp(url, "/compact") == 0 && request->method == GV_HTTP_POST) {
        return gv_rest_handle_compact(ctx, request);
    }

    /* POST /save */
    if (strcmp(url, "/save") == 0 && request->method == GV_HTTP_POST) {
        return gv_rest_handle_save(ctx, request);
    }

    /* 404 Not Found */
    return gv_rest_response_error(GV_HTTP_404_NOT_FOUND, "not_found",
                                   "Endpoint not found");
}
