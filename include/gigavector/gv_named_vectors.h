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

/* Lifecycle */
GV_NamedVectorStore *gv_named_vectors_create(void);
void gv_named_vectors_destroy(GV_NamedVectorStore *store);

/* Field management */
int gv_named_vectors_add_field(GV_NamedVectorStore *store, const GV_VectorFieldConfig *config);
int gv_named_vectors_remove_field(GV_NamedVectorStore *store, const char *name);
size_t gv_named_vectors_field_count(const GV_NamedVectorStore *store);
int gv_named_vectors_get_field(const GV_NamedVectorStore *store, const char *name, GV_VectorFieldConfig *out);

/* Point operations - insert/update vectors for multiple fields at once */
int gv_named_vectors_insert(GV_NamedVectorStore *store, size_t point_id,
                             const GV_NamedVector *vectors, size_t vector_count);
int gv_named_vectors_update(GV_NamedVectorStore *store, size_t point_id,
                             const GV_NamedVector *vectors, size_t vector_count);
int gv_named_vectors_delete(GV_NamedVectorStore *store, size_t point_id);

/* Search a specific named field */
int gv_named_vectors_search(const GV_NamedVectorStore *store, const char *field_name,
                             const float *query, size_t k, GV_NamedSearchResult *results);

/* Get vector data for a specific point and field */
const float *gv_named_vectors_get(const GV_NamedVectorStore *store, size_t point_id, const char *field_name);

/* Count */
size_t gv_named_vectors_count(const GV_NamedVectorStore *store);

/* Save/load */
int gv_named_vectors_save(const GV_NamedVectorStore *store, const char *filepath);
GV_NamedVectorStore *gv_named_vectors_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
