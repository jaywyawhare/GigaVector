#ifndef GIGAVECTOR_STORAGE_DB_INTERNAL_H
#define GIGAVECTOR_STORAGE_DB_INTERNAL_H

/* Internal cross-module helpers shared between db_*.c translation units.
   Not part of the public API. */

#include "storage/database.h"
#include <stdint.h>

/* db_stats.c */
size_t db_estimate_vector_memory(size_t dimension);
void   db_update_memory_usage(GV_Database *db);
uint64_t db_get_time_us(void);

/* db_resource.c */
int  db_check_resource_limits(GV_Database *db, size_t additional_vectors,
                               size_t additional_memory);
void db_increment_concurrent_ops(GV_Database *db);
void db_decrement_concurrent_ops(GV_Database *db);

#endif
