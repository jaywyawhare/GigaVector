#ifndef GIGAVECTOR_GV_AUTHZ_H
#define GIGAVECTOR_GV_AUTHZ_H

#include <stddef.h>
#include <stdint.h>

#include "gv_auth.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file gv_authz.h
 * @brief Authorization (RBAC) for GigaVector.
 *
 * Provides role-based access control for namespace and operation permissions.
 */

/**
 * @brief Permission flags.
 */
typedef enum {
    GV_PERM_NONE = 0,               /**< No permissions. */
    GV_PERM_READ = 1,               /**< Read vectors/search. */
    GV_PERM_WRITE = 2,              /**< Add/update vectors. */
    GV_PERM_DELETE = 4,             /**< Delete vectors. */
    GV_PERM_ADMIN = 8,              /**< Manage users/namespaces. */
    GV_PERM_ALL = 15                /**< All permissions. */
} GV_Permission;

/**
 * @brief Resource type.
 */
typedef enum {
    GV_RESOURCE_GLOBAL = 0,         /**< Global/database level. */
    GV_RESOURCE_NAMESPACE = 1,      /**< Specific namespace. */
    GV_RESOURCE_VECTOR = 2          /**< Specific vector. */
} GV_ResourceType;

/**
 * @brief Role definition.
 */
typedef struct {
    char *name;                     /**< Role name. */
    uint32_t permissions;           /**< Permission flags (GV_Permission). */
    char **allowed_namespaces;      /**< Allowed namespaces (NULL = all). */
    size_t namespace_count;         /**< Number of allowed namespaces. */
} GV_Role;

/**
 * @brief User-role assignment.
 */
typedef struct {
    char *subject;                  /**< User/service subject. */
    char **roles;                   /**< Assigned roles. */
    size_t role_count;              /**< Number of roles. */
} GV_UserRoles;

/**
 * @brief Authorization check result.
 */
typedef struct {
    int allowed;                    /**< 1 if allowed, 0 if denied. */
    const char *denied_reason;      /**< Reason for denial (if any). */
    const char *matched_role;       /**< Role that granted access (if any). */
} GV_AuthzResult;

/**
 * @brief Opaque authorization manager handle.
 */
typedef struct GV_AuthzManager GV_AuthzManager;

/* Authorization Manager Lifecycle */

/**
 * @brief Create an authorization manager.
 *
 * @return Authorization manager instance, or NULL on error.
 */
GV_AuthzManager *gv_authz_create(void);

/**
 * @brief Destroy an authorization manager.
 *
 * @param authz Authorization manager instance (safe to call with NULL).
 */
void gv_authz_destroy(GV_AuthzManager *authz);

/* Role Management */

/**
 * @brief Define a new role.
 *
 * @param authz Authorization manager.
 * @param name Role name.
 * @param permissions Permission flags.
 * @param namespaces Allowed namespaces (NULL = all namespaces).
 * @param namespace_count Number of namespaces.
 * @return 0 on success, -1 on error.
 */
int gv_authz_define_role(GV_AuthzManager *authz, const char *name,
                          uint32_t permissions, const char **namespaces,
                          size_t namespace_count);

/**
 * @brief Remove a role definition.
 *
 * @param authz Authorization manager.
 * @param name Role name.
 * @return 0 on success, -1 on error.
 */
int gv_authz_remove_role(GV_AuthzManager *authz, const char *name);

/**
 * @brief Get role by name.
 *
 * @param authz Authorization manager.
 * @param name Role name.
 * @param role Output role (caller owns, use gv_authz_free_role to free).
 * @return 0 on success, -1 on error.
 */
int gv_authz_get_role(GV_AuthzManager *authz, const char *name, GV_Role *role);

/**
 * @brief List all roles.
 *
 * @param authz Authorization manager.
 * @param roles Output role array.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int gv_authz_list_roles(GV_AuthzManager *authz, GV_Role **roles, size_t *count);

/**
 * @brief Free role structure.
 *
 * @param role Role to free.
 */
void gv_authz_free_role(GV_Role *role);

/**
 * @brief Free role array.
 *
 * @param roles Roles to free.
 * @param count Number of roles.
 */
void gv_authz_free_roles(GV_Role *roles, size_t count);

/* User-Role Assignment */

/**
 * @brief Assign a role to a user.
 *
 * @param authz Authorization manager.
 * @param subject User/service subject.
 * @param role_name Role name.
 * @return 0 on success, -1 on error.
 */
int gv_authz_assign_role(GV_AuthzManager *authz, const char *subject,
                          const char *role_name);

/**
 * @brief Revoke a role from a user.
 *
 * @param authz Authorization manager.
 * @param subject User/service subject.
 * @param role_name Role name.
 * @return 0 on success, -1 on error.
 */
int gv_authz_revoke_role(GV_AuthzManager *authz, const char *subject,
                          const char *role_name);

/**
 * @brief Get all roles for a user.
 *
 * @param authz Authorization manager.
 * @param subject User/service subject.
 * @param roles Output role names.
 * @param count Output count.
 * @return 0 on success, -1 on error.
 */
int gv_authz_get_user_roles(GV_AuthzManager *authz, const char *subject,
                             char ***roles, size_t *count);

/**
 * @brief Free user roles array.
 *
 * @param roles Roles to free.
 * @param count Number of roles.
 */
void gv_authz_free_user_roles(char **roles, size_t count);

/* Authorization Checks */

/**
 * @brief Check if identity has permission on a resource.
 *
 * @param authz Authorization manager.
 * @param identity Authenticated identity.
 * @param permission Required permission.
 * @param resource_type Resource type.
 * @param resource_name Resource name (namespace name, or NULL for global).
 * @param result Output authorization result.
 * @return 0 on success, -1 on error.
 */
int gv_authz_check(GV_AuthzManager *authz, const GV_Identity *identity,
                    GV_Permission permission, GV_ResourceType resource_type,
                    const char *resource_name, GV_AuthzResult *result);

/**
 * @brief Check if identity can read from namespace.
 *
 * Convenience function for gv_authz_check with GV_PERM_READ.
 *
 * @param authz Authorization manager.
 * @param identity Authenticated identity.
 * @param namespace_name Namespace name (NULL for global).
 * @return 1 if allowed, 0 if denied.
 */
int gv_authz_can_read(GV_AuthzManager *authz, const GV_Identity *identity,
                       const char *namespace_name);

/**
 * @brief Check if identity can write to namespace.
 *
 * @param authz Authorization manager.
 * @param identity Authenticated identity.
 * @param namespace_name Namespace name (NULL for global).
 * @return 1 if allowed, 0 if denied.
 */
int gv_authz_can_write(GV_AuthzManager *authz, const GV_Identity *identity,
                        const char *namespace_name);

/**
 * @brief Check if identity can delete from namespace.
 *
 * @param authz Authorization manager.
 * @param identity Authenticated identity.
 * @param namespace_name Namespace name (NULL for global).
 * @return 1 if allowed, 0 if denied.
 */
int gv_authz_can_delete(GV_AuthzManager *authz, const GV_Identity *identity,
                         const char *namespace_name);

/**
 * @brief Check if identity has admin privileges.
 *
 * @param authz Authorization manager.
 * @param identity Authenticated identity.
 * @return 1 if admin, 0 otherwise.
 */
int gv_authz_is_admin(GV_AuthzManager *authz, const GV_Identity *identity);

/* Built-in Roles */

/**
 * @brief Initialize built-in roles.
 *
 * Creates standard roles: admin, writer, reader.
 *
 * @param authz Authorization manager.
 * @return 0 on success, -1 on error.
 */
int gv_authz_init_builtin_roles(GV_AuthzManager *authz);

/**
 * @brief Get permission string representation.
 *
 * @param permission Permission flag.
 * @return Permission name string.
 */
const char *gv_authz_permission_string(GV_Permission permission);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_AUTHZ_H */
