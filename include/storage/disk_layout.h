#ifndef GIGAVECTOR_GV_DISK_LAYOUT_H
#define GIGAVECTOR_GV_DISK_LAYOUT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Default on-disk sector/page alignment shared by DiskANN and posting lists. */
#define GV_DISK_SECTOR_SIZE_DEFAULT 4096u

/** @deprecated Use GV_DISK_SECTOR_SIZE_DEFAULT. */
#define GV_POSTING_DEFAULT_SECTOR_SIZE GV_DISK_SECTOR_SIZE_DEFAULT

/**
 * @brief Return the default disk sector size (bytes).
 */
size_t gv_disk_default_sector_size(void);

/**
 * @brief Normalize a sector size (0 -> default).
 */
size_t gv_disk_normalize_sector_size(size_t sector_size);

#ifdef __cplusplus
}
#endif

#endif
