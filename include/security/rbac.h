#ifndef GIGAVECTOR_GV_RBAC_H
#define GIGAVECTOR_GV_RBAC_H
#include <stddef.h>
#include <stdint.h>
#include "security/authz.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *resource;           /* Collection/namespace name, or "*" for all */
    uint32_t permissions;     /* Bitmask of GV_Permission */
} GV_RBACRule;

typedef struct GV_RBACManager GV_RBACManager;

GV_RBACManager *rbac_create(void);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param mgr Manager instance.
 */
void rbac_destroy(GV_RBACManager *mgr);

/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param role_name Name string.
 * @return 0 on success, -1 on error.
 */
int rbac_create_role(GV_RBACManager *mgr, const char *role_name);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param role_name Name string.
 * @return 0 on success, -1 on error.
 */
int rbac_delete_role(GV_RBACManager *mgr, const char *role_name);
int rbac_add_rule(GV_RBACManager *mgr, const char *role_name,
                      const char *resource, uint32_t permissions);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param role_name Name string.
 * @param resource resource.
 * @return 0 on success, -1 on error.
 */
int rbac_remove_rule(GV_RBACManager *mgr, const char *role_name, const char *resource);
/**
 * @brief Set a value.
 *
 * @param mgr Manager instance.
 * @param role_name Name string.
 * @param parent_role parent_role.
 * @return 0 on success, -1 on error.
 */
int rbac_set_inheritance(GV_RBACManager *mgr, const char *role_name, const char *parent_role);

/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param user_id Identifier.
 * @param role_name Name string.
 * @return 0 on success, -1 on error.
 */
int rbac_assign_role(GV_RBACManager *mgr, const char *user_id, const char *role_name);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param user_id Identifier.
 * @param role_name Name string.
 * @return 0 on success, -1 on error.
 */
int rbac_revoke_role(GV_RBACManager *mgr, const char *user_id, const char *role_name);
int rbac_get_user_roles(const GV_RBACManager *mgr, const char *user_id,
                            char ***out_roles, size_t *out_count);

int rbac_check(const GV_RBACManager *mgr, const char *user_id,
                   const char *resource, GV_Permission required);

/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param out_names Output parameter.
 * @param out_count Output item count.
 * @return 0 on success, -1 on error.
 */
int rbac_list_roles(const GV_RBACManager *mgr, char ***out_names, size_t *out_count);
/**
 * @brief List items.
 *
 * @param list list.
 * @param count Number of items.
 */
void rbac_free_string_list(char **list, size_t count);

/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @return 0 on success, -1 on error.
 */
int rbac_init_defaults(GV_RBACManager *mgr);  /* Creates admin, writer, reader roles */

/**
 * @brief Save state to a file.
 *
 * @param mgr Manager instance.
 * @param filepath Filesystem path.
 * @return 0 on success, -1 on error.
 */
int rbac_save(const GV_RBACManager *mgr, const char *filepath);
GV_RBACManager *rbac_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
