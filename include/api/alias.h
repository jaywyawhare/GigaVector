#ifndef GIGAVECTOR_GV_ALIAS_H
#define GIGAVECTOR_GV_ALIAS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_Database GV_Database;
typedef struct GV_AliasManager GV_AliasManager;

typedef struct {
    char *alias_name;
    char *collection_name;
    uint64_t created_at;
    uint64_t updated_at;
} GV_AliasInfo;

GV_AliasManager *alias_manager_create(void);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param mgr Manager instance.
 */
void alias_manager_destroy(GV_AliasManager *mgr);

/**
 * @brief Create a new instance.
 *
 * @param mgr Manager instance.
 * @param alias_name Alias name.
 * @param collection_name Collection/namespace name.
 * @return 0 on success, -1 on error.
 */
int alias_create(GV_AliasManager *mgr, const char *alias_name, const char *collection_name);
/**
 * @brief Update an item.
 *
 * @param mgr Manager instance.
 * @param alias_name Alias name.
 * @param new_collection_name New collection/namespace name.
 * @return 0 on success, -1 on error.
 */
int alias_update(GV_AliasManager *mgr, const char *alias_name, const char *new_collection_name);
/**
 * @brief Delete an item.
 *
 * @param mgr Manager instance.
 * @param alias_name Alias name.
 * @return 0 on success, -1 on error.
 */
int alias_delete(GV_AliasManager *mgr, const char *alias_name);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param alias_name Alias name.
 * @return 1 if true, 0 if false, -1 on error.
 */
int alias_exists(const GV_AliasManager *mgr, const char *alias_name);

/* Atomic swap: swap two aliases' targets in one operation */
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param alias_a alias_a.
 * @param alias_b alias_b.
 * @return 0 on success, -1 on error.
 */
int alias_swap(GV_AliasManager *mgr, const char *alias_a, const char *alias_b);

/* Resolve alias to collection name (returns NULL if not found) */
const char *alias_resolve(const GV_AliasManager *mgr, const char *alias_name);

/**
 * @brief List items.
 *
 * @param mgr Manager instance.
 * @param out_list Output list pointer (allocated by callee).
 * @param out_count Output item count.
 * @return 0 on success, -1 on error.
 */
int alias_list(const GV_AliasManager *mgr, GV_AliasInfo **out_list, size_t *out_count);
/**
 * @brief List items.
 *
 * @param list list.
 * @param count Number of items.
 */
void alias_free_list(GV_AliasInfo *list, size_t count);

/**
 * @brief Retrieve information.
 *
 * @param mgr Manager instance.
 * @param alias_name Alias name.
 * @param info Output information structure.
 * @return 0 on success, -1 on error.
 */
int alias_get_info(const GV_AliasManager *mgr, const char *alias_name, GV_AliasInfo *info);

/**
 * @brief Return the number of stored items.
 *
 * @param mgr Manager instance.
 * @return Count value.
 */
size_t alias_count(const GV_AliasManager *mgr);

/**
 * @brief Save state to a file.
 *
 * @param mgr Manager instance.
 * @param filepath Filesystem path.
 * @return 0 on success, -1 on error.
 */
int alias_save(const GV_AliasManager *mgr, const char *filepath);
GV_AliasManager *alias_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
