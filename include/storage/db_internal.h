#ifndef GIGAVECTOR_STORAGE_DB_INTERNAL_H
#define GIGAVECTOR_STORAGE_DB_INTERNAL_H

/* Internal cross-module helpers shared between db_*.c translation units.
   Not part of the public API. */

#include "storage/database.h"
#include <stdint.h>
#include "admin/cdc.h"
#include "admin/webhook.h"

/* db_stats.c */
size_t db_estimate_vector_memory(size_t dimension);
void   db_update_memory_usage(GV_Database *db);
uint64_t db_get_time_us(void);

/* db_resource.c */
int  db_check_resource_limits(GV_Database *db, size_t additional_vectors,
                               size_t additional_memory);
void db_increment_concurrent_ops(GV_Database *db);
void db_decrement_concurrent_ops(GV_Database *db);

/* db_core (database.c) <-> db_io.c / db_maintenance.c */
void db_refresh_count(GV_Database *db);
int  db_save_locked(const GV_Database *db, const char *filepath);

char *db_build_wal_path(const char *filepath);
int   db_replay_wal(GV_Database *db);

int db_write_header(FILE *out, uint32_t dimension, uint64_t count,
                    uint32_t version, uint64_t checkpoint_gen);
/* Read the WAL checkpoint-generation sidecar next to @p wal_path; 0 if
 * missing/corrupt. Recovery skips WAL replay when this is < the snapshot's. */
uint64_t wal_read_ckpt_gen(const char *wal_path);
void db_fill_ivfdisk_search_vectors(GV_Database *db, GV_SearchResult *results,
                                    int n);

void db_normalize_vector(GV_Vector *vector);
void db_emit_change(GV_Database *db, GV_CDCEventType cdc_type,
                     GV_EventType event_type, size_t index,
                     const float *data, size_t dimension);

#endif
