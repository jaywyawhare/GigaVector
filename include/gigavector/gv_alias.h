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

/* Lifecycle */
GV_AliasManager *gv_alias_manager_create(void);
void gv_alias_manager_destroy(GV_AliasManager *mgr);

/* Alias operations */
int gv_alias_create(GV_AliasManager *mgr, const char *alias_name, const char *collection_name);
int gv_alias_update(GV_AliasManager *mgr, const char *alias_name, const char *new_collection_name);
int gv_alias_delete(GV_AliasManager *mgr, const char *alias_name);
int gv_alias_exists(const GV_AliasManager *mgr, const char *alias_name);

/* Atomic swap: swap two aliases' targets in one operation */
int gv_alias_swap(GV_AliasManager *mgr, const char *alias_a, const char *alias_b);

/* Resolve alias to collection name (returns NULL if not found) */
const char *gv_alias_resolve(const GV_AliasManager *mgr, const char *alias_name);

/* List all aliases */
int gv_alias_list(const GV_AliasManager *mgr, GV_AliasInfo **out_list, size_t *out_count);
void gv_alias_free_list(GV_AliasInfo *list, size_t count);

/* Get info */
int gv_alias_get_info(const GV_AliasManager *mgr, const char *alias_name, GV_AliasInfo *info);

/* Count */
size_t gv_alias_count(const GV_AliasManager *mgr);

/* Persistence */
int gv_alias_save(const GV_AliasManager *mgr, const char *filepath);
GV_AliasManager *gv_alias_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
