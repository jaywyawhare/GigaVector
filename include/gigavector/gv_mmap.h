#ifndef GIGAVECTOR_GV_MMAP_H
#define GIGAVECTOR_GV_MMAP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle for a read-only memory-mapped file.
 */
typedef struct GV_MMap GV_MMap;

/**
 * @brief Open a file as a read-only memory mapping.
 *
 * The returned mapping can be used to access the entire file contents via
 * gv_mmap_data() / gv_mmap_size(). The file descriptor is owned by the
 * mapping and closed on gv_mmap_close().
 *
 * @param path Filesystem path to map.
 * @return New mapping handle on success, or NULL on error.
 */
GV_MMap *gv_mmap_open_readonly(const char *path);

/**
 * @brief Close a memory mapping and release associated resources.
 *
 * Safe to call with NULL; no action is taken.
 *
 * @param mm Mapping handle to close.
 */
void gv_mmap_close(GV_MMap *mm);

/**
 * @brief Get pointer to the mapped file contents.
 *
 * @param mm Mapping handle; must be non-NULL.
 * @return Pointer to read-only memory region backing the file, or NULL if
 *         mapping is invalid.
 */
const void *gv_mmap_data(const GV_MMap *mm);

/**
 * @brief Get size in bytes of the mapped file.
 *
 * @param mm Mapping handle; must be non-NULL.
 * @return Size of the mapping in bytes, or 0 on error.
 */
size_t gv_mmap_size(const GV_MMap *mm);

#ifdef __cplusplus
}
#endif

#endif


