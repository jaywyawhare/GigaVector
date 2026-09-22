#ifndef GIGAVECTOR_VALUE_STORE_H
#define GIGAVECTOR_VALUE_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file value_store.h
 * @brief WiscKey key-value separation: a uint64 key -> value store where keys
 *        (and their compact vlog offsets) live in an in-memory hash index while
 *        the values live in an append-only value log (see vlog.h).
 *
 * Updates and deletes leave dead value-log records behind; value_store_gc()
 * reclaims that space by compacting the log and rewriting the key->offset map.
 * This is the storage pattern that keeps a key index dense and cheap to scan
 * while values are written sequentially and reclaimed in bulk.
 */

typedef struct GV_ValueStore GV_ValueStore;

/** @brief Open (creating if absent) a value store backed by a log at @p path. */
GV_ValueStore *value_store_open(const char *path);

/**
 * @brief Insert or replace the value for @p key.
 * @return 0 on success, -1 on error. A prior value for @p key becomes garbage.
 */
int value_store_put(GV_ValueStore *vs, uint64_t key, const void *value, size_t len);

/**
 * @brief Fetch the value for @p key.
 * @param value_out Output: newly allocated buffer (caller free()s); NULL if len 0.
 * @param len_out Output: value length. Must be non-NULL.
 * @return 0 on success, -1 if the key is absent or on error.
 */
int value_store_get(GV_ValueStore *vs, uint64_t key, void **value_out, size_t *len_out);

/** @brief Remove @p key. @return 0 if removed, -1 if absent. */
int value_store_delete(GV_ValueStore *vs, uint64_t key);

/** @brief Number of live keys. */
size_t value_store_count(const GV_ValueStore *vs);

/** @brief Current on-disk value-log size in bytes. */
uint64_t value_store_disk_size(const GV_ValueStore *vs);

/**
 * @brief Garbage-collect the value log, dropping records no live key points at.
 * Rewrites the log and updates the key->offset map to the new offsets.
 * @return 0 on success, -1 on error.
 */
int value_store_gc(GV_ValueStore *vs);

/** @brief Flush and fsync the underlying log. */
int value_store_sync(GV_ValueStore *vs);

/** @brief Close the store and free resources. Safe with NULL. */
void value_store_close(GV_ValueStore *vs);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_VALUE_STORE_H */
