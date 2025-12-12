from __future__ import annotations

import os
from pathlib import Path

from cffi import FFI

ffi = FFI()

# Keep this cdef in sync with the C headers.
ffi.cdef(
    """
typedef enum { GV_INDEX_TYPE_KDTREE = 0, GV_INDEX_TYPE_HNSW = 1, GV_INDEX_TYPE_IVFPQ = 2 } GV_IndexType;
typedef enum { GV_DISTANCE_EUCLIDEAN = 0, GV_DISTANCE_COSINE = 1 } GV_DistanceType;

typedef struct GV_Metadata {
    char *key;
    char *value;
    struct GV_Metadata *next;
} GV_Metadata;

typedef struct {
    size_t dimension;
    float *data;
    GV_Metadata *metadata;
} GV_Vector;

typedef struct GV_KDNode {
    GV_Vector *point;
    size_t axis;
    struct GV_KDNode *left;
    struct GV_KDNode *right;
} GV_KDNode;

typedef struct GV_WAL GV_WAL;

typedef struct GV_Database {
    size_t dimension;
    GV_IndexType index_type;
    GV_KDNode *root;
    void *hnsw_index;
    char *filepath;
    char *wal_path;
    GV_WAL *wal;
    int wal_replaying;
    void *rwlock;  // pthread_rwlock_t - opaque for FFI
    void *wal_mutex;  // pthread_mutex_t - opaque for FFI
    size_t count;
} GV_Database;

typedef struct {
    const GV_Vector *vector;
    float distance;
} GV_SearchResult;

GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type);
void gv_db_close(GV_Database *db);

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);
int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                    const char *metadata_key, const char *metadata_value);
int gv_db_add_vector_with_rich_metadata(GV_Database *db, const float *data, size_t dimension,
                                        const char *const *metadata_keys, const char *const *metadata_values,
                                        size_t metadata_count);
int gv_db_save(const GV_Database *db, const char *filepath);
int gv_db_ivfpq_train(GV_Database *db, const float *data, size_t count, size_t dimension);
int gv_db_add_vectors(GV_Database *db, const float *data, size_t count, size_t dimension);
int gv_db_add_vectors_with_metadata(GV_Database *db, const float *data,
                                    const char *const *keys, const char *const *values,
                                    size_t count, size_t dimension);

int gv_db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type);
int gv_db_search_filtered(const GV_Database *db, const float *query_data, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type,
                          const char *filter_key, const char *filter_value);
int gv_db_search_batch(const GV_Database *db, const float *queries, size_t qcount, size_t k,
                       GV_SearchResult *results, GV_DistanceType distance_type);

// Vector creation and metadata management
GV_Vector *gv_vector_create_from_data(size_t dimension, const float *data);
int gv_vector_set_metadata(GV_Vector *vector, const char *key, const char *value);
void gv_vector_destroy(GV_Vector *vector);

// Index insertion functions
int gv_kdtree_insert(GV_KDNode **root, GV_Vector *point, size_t depth);
int gv_hnsw_insert(void *index, GV_Vector *vector);
int gv_ivfpq_insert(void *index, GV_Vector *vector);

// WAL functions
int gv_wal_append_insert(GV_WAL *wal, const float *data, size_t dimension,
                         const char *metadata_key, const char *metadata_value);
int gv_wal_append_insert_rich(GV_WAL *wal, const float *data, size_t dimension,
                              const char *const *metadata_keys, const char *const *metadata_values,
                              size_t metadata_count);
"""
)


def _load_lib():
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent.parent  # .../GigaVector
    # Prefer freshly built library, fall back to packaged copy
    candidate_paths = [
        repo_root / "build" / "lib" / "libGigaVector.so",
        here / "libGigaVector.so",
    ]
    for lib_path in candidate_paths:
        if lib_path.exists():
            return ffi.dlopen(os.fspath(lib_path))
    raise FileNotFoundError(f"libGigaVector.so not found in {candidate_paths}")


lib = _load_lib()

