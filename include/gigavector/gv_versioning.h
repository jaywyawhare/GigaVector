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

GV_VersionManager *gv_version_manager_create(size_t max_versions);
void gv_version_manager_destroy(GV_VersionManager *mgr);

/* Create a version checkpoint (stores a full copy of vector data) */
uint64_t gv_version_create(GV_VersionManager *mgr, const float *data,
                            size_t count, size_t dimension, const char *label);

/* List all versions */
int gv_version_list(const GV_VersionManager *mgr, GV_VersionInfo *infos, size_t max_infos);
int gv_version_count(const GV_VersionManager *mgr);

/* Get version info */
int gv_version_get_info(const GV_VersionManager *mgr, uint64_t version_id, GV_VersionInfo *info);

/* Retrieve versioned data (caller must free returned pointer) */
float *gv_version_get_data(const GV_VersionManager *mgr, uint64_t version_id,
                            size_t *count_out, size_t *dimension_out);

/* Delete a specific version */
int gv_version_delete(GV_VersionManager *mgr, uint64_t version_id);

/* Compare two versions - returns number of differing vectors, or -1 on error */
int gv_version_compare(const GV_VersionManager *mgr, uint64_t v1, uint64_t v2,
                        size_t *added, size_t *removed, size_t *modified);

/* Save/load version manager state */
int gv_version_save(const GV_VersionManager *mgr, FILE *out);
int gv_version_load(GV_VersionManager **mgr_ptr, FILE *in);

#ifdef __cplusplus
}
#endif
#endif
