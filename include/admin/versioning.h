#ifndef GIGAVECTOR_GV_VERSIONING_H
#define GIGAVECTOR_GV_VERSIONING_H
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t version_id;
    uint64_t timestamp_us;
    size_t vector_count;
    size_t dimension;
    char label[128];
    size_t data_size_bytes;
} GV_VersionInfo;

typedef struct GV_VersionManager GV_VersionManager;

GV_VersionManager *version_manager_create(size_t max_versions);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param mgr Manager instance.
 */
void version_manager_destroy(GV_VersionManager *mgr);

uint64_t version_create(GV_VersionManager *mgr, const float *data,
                            size_t count, size_t dimension, const char *label);

/**
 * @brief List items.
 *
 * @param mgr Manager instance.
 * @param infos infos.
 * @param max_infos max_infos.
 * @return 0 on success, -1 on error.
 */
int version_list(const GV_VersionManager *mgr, GV_VersionInfo *infos, size_t max_infos);
/**
 * @brief Return the number of stored items.
 *
 * @param mgr Manager instance.
 * @return 0 on success, -1 on error.
 */
int version_count(const GV_VersionManager *mgr);

/**
 * @brief Retrieve information.
 *
 * @param mgr Manager instance.
 * @param version_id Identifier.
 * @param info Output information structure.
 * @return 0 on success, -1 on error.
 */
int version_get_info(const GV_VersionManager *mgr, uint64_t version_id, GV_VersionInfo *info);

float *version_get_data(const GV_VersionManager *mgr, uint64_t version_id,
                            size_t *count_out, size_t *dimension_out);

/**
 * @brief Delete an item.
 *
 * @param mgr Manager instance.
 * @param version_id Identifier.
 * @return 0 on success, -1 on error.
 */
int version_delete(GV_VersionManager *mgr, uint64_t version_id);

int version_compare(const GV_VersionManager *mgr, uint64_t v1, uint64_t v2,
                        size_t *added, size_t *removed, size_t *modified);

/**
 * @brief Save state to a file.
 *
 * @param mgr Manager instance.
 * @param out Output buffer.
 * @return 0 on success, -1 on error.
 */
int version_save(const GV_VersionManager *mgr, FILE *out);
/**
 * @brief Load state from a file.
 *
 * @param mgr_ptr mgr_ptr.
 * @param in Input file stream.
 * @return 0 on success, -1 on error.
 */
int version_load(GV_VersionManager **mgr_ptr, FILE *in);

#ifdef __cplusplus
}
#endif
#endif
