#ifndef GIGAVECTOR_GV_ID_BITMAP_H
#define GIGAVECTOR_GV_ID_BITMAP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file id_bitmap.h
 * @brief Compact "Roaring-lite" set of 64-bit ids.
 *
 * Stores a set of uint64_t ids as a sorted array of chunks, each chunk keyed by
 * the high 48 bits (@c id>>16) and holding a container over the low 16 bits
 * (@c id&0xFFFF). Containers are either a sorted uint16_t array (small
 * cardinality) or a dense 8192-byte bitmap (65536 bits). This gives compact
 * storage plus fast union/intersect for representing posting-list membership so
 * IVF multi-probe can combine candidate sets without materializing every entry.
 *
 * The ADT holds no global/static mutable state and is not internally locked;
 * callers synchronize concurrent access.
 */
typedef struct GV_IdBitmap GV_IdBitmap;

/** @brief Create an empty bitmap. Returns NULL on OOM. */
GV_IdBitmap *gv_id_bitmap_create(void);

/** @brief Free a bitmap. Safe on NULL. */
void gv_id_bitmap_free(GV_IdBitmap *b);

/** @brief Add @p id to the set. Idempotent. Returns 0 on success, -1 on OOM. */
int gv_id_bitmap_add(GV_IdBitmap *b, uint64_t id);

/** @brief Remove @p id if present. Returns 0 on success. */
int gv_id_bitmap_remove(GV_IdBitmap *b, uint64_t id);

/** @brief Test membership. Returns 1 if present, else 0. */
int gv_id_bitmap_contains(const GV_IdBitmap *b, uint64_t id);

/** @brief Number of ids in the set. */
uint64_t gv_id_bitmap_cardinality(const GV_IdBitmap *b);

/** @brief Remove all ids, retaining the (now empty) bitmap. */
void gv_id_bitmap_clear(GV_IdBitmap *b);

/** @brief Deep copy. Returns NULL on OOM (or if @p b is NULL). */
GV_IdBitmap *gv_id_bitmap_clone(const GV_IdBitmap *b);

/** @brief In-place union: @c dst |= src. Returns 0 on success, -1 on OOM. */
int gv_id_bitmap_or_into(GV_IdBitmap *dst, const GV_IdBitmap *src);

/** @brief New bitmap equal to @c a & b (intersection). NULL on OOM. */
GV_IdBitmap *gv_id_bitmap_and(const GV_IdBitmap *a, const GV_IdBitmap *b);

/** @brief New bitmap equal to @c a | b (union). NULL on OOM. */
GV_IdBitmap *gv_id_bitmap_or(const GV_IdBitmap *a, const GV_IdBitmap *b);

/**
 * @brief Iterate over members in ascending order.
 * @param fn Callback invoked per id; return 0 to continue, non-zero to stop.
 * @return 0 if iteration completed, else the non-zero value that stopped it.
 */
int gv_id_bitmap_iterate(const GV_IdBitmap *b,
                         int (*fn)(uint64_t id, void *ctx), void *ctx);

/**
 * @brief Serialize to a self-contained little-endian byte buffer.
 *
 * Allocates @p *out via gv_alloc and sets @p *out_len; caller frees with
 * gv_free. Returns 0 on success, -1 on error (bad args or OOM).
 */
int gv_id_bitmap_serialize(const GV_IdBitmap *b, uint8_t **out, size_t *out_len);

/**
 * @brief Deserialize a buffer produced by gv_id_bitmap_serialize().
 *
 * Robust against truncated/corrupt input (may parse untrusted on-disk data):
 * every field and length is bounds-checked before use, allocations are capped,
 * and any malformed input yields NULL rather than a crash. Returns a new
 * bitmap the caller must free, or NULL.
 */
GV_IdBitmap *gv_id_bitmap_deserialize(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_ID_BITMAP_H */
