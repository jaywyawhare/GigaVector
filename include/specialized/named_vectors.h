#ifndef GIGAVECTOR_GV_NAMED_VECTORS_H
#define GIGAVECTOR_GV_NAMED_VECTORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_NamedVectorStore GV_NamedVectorStore;

typedef struct {
    const char *name;       /* Field name (e.g., "title", "content") */
    size_t dimension;       /* Dimension for this field */
    int distance_type;      /* Default distance metric for this field */
} GV_VectorFieldConfig;

typedef struct {
    const char *field_name;
    const float *data;
    size_t dimension;
} GV_NamedVector;

typedef struct {
    size_t point_index;     /* Internal index of the point */
    float distance;         /* Distance from query */
    const char *field_name; /* Which field matched */
} GV_NamedSearchResult;

GV_NamedVectorStore *named_vectors_create(void);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param store store.
 */
void named_vectors_destroy(GV_NamedVectorStore *store);

/**
 * @brief Add an item.
 *
 * @param store store.
 * @param config Configuration to apply/output.
 * @return 0 on success, -1 on error.
 */
int named_vectors_add_field(GV_NamedVectorStore *store, const GV_VectorFieldConfig *config);
/**
 * @brief Perform the operation.
 *
 * @param store store.
 * @param name Name string.
 * @return 0 on success, -1 on error.
 */
int named_vectors_remove_field(GV_NamedVectorStore *store, const char *name);
/**
 * @brief Return the number of stored items.
 *
 * @param store store.
 * @return Count value.
 */
size_t named_vectors_field_count(const GV_NamedVectorStore *store);
/**
 * @brief Get a value.
 *
 * @param store store.
 * @param name Name string.
 * @param out Output buffer.
 * @return 0 on success, -1 on error.
 */
int named_vectors_get_field(const GV_NamedVectorStore *store, const char *name, GV_VectorFieldConfig *out);

int named_vectors_insert(GV_NamedVectorStore *store, size_t point_id,
                             const GV_NamedVector *vectors, size_t vector_count);
int named_vectors_update(GV_NamedVectorStore *store, size_t point_id,
                             const GV_NamedVector *vectors, size_t vector_count);
/**
 * @brief Delete an item.
 *
 * @param store store.
 * @param point_id Identifier.
 * @return 0 on success, -1 on error.
 */
int named_vectors_delete(GV_NamedVectorStore *store, size_t point_id);

int named_vectors_search(const GV_NamedVectorStore *store, const char *field_name,
                             const float *query, size_t k, GV_NamedSearchResult *results);

const float *named_vectors_get(const GV_NamedVectorStore *store, size_t point_id, const char *field_name);

/**
 * @brief Return the number of stored items.
 *
 * @param store store.
 * @return Count value.
 */
size_t named_vectors_count(const GV_NamedVectorStore *store);

/**
 * @brief Save state to a file.
 *
 * @param store store.
 * @param filepath Filesystem path.
 * @return 0 on success, -1 on error.
 */
int named_vectors_save(const GV_NamedVectorStore *store, const char *filepath);
GV_NamedVectorStore *named_vectors_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
