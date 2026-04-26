#pragma once

#include "storage/database.h"
#include "admin/replication.h"
#include "storage/wal.h"

#ifdef __cplusplus
extern "C" {
#endif

GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type);
void gv_db_close(GV_Database *db);

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension);
int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                   const char *metadata_key, const char *metadata_value);

int gv_db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type);

int gv_db_save(const GV_Database *db, const char *filepath);
int gv_db_delete_vector_by_index(GV_Database *db, size_t vector_index);

#ifdef __cplusplus
}
#endif
