#ifndef GIGAVECTOR_GV_RBAC_H
#define GIGAVECTOR_GV_RBAC_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_PERM_READ = 1,
    GV_PERM_WRITE = 2,
    GV_PERM_DELETE = 4,
    GV_PERM_ADMIN = 8,
    GV_PERM_ALL = 15
} GV_Permission;

typedef struct {
    char *resource;           /* Collection/namespace name, or "*" for all */
    uint32_t permissions;     /* Bitmask of GV_Permission */
} GV_RBACRule;

typedef struct {
    char *role_name;
    GV_RBACRule *rules;
    size_t rule_count;
    int inherits_from;        /* Index of parent role (-1 for none) */
} GV_Role;

typedef struct {
    char *user_id;
    char **role_names;
    size_t role_count;
} GV_UserRoles;

typedef struct GV_RBACManager GV_RBACManager;

/* Lifecycle */
GV_RBACManager *gv_rbac_create(void);
void gv_rbac_destroy(GV_RBACManager *mgr);

/* Role management */
int gv_rbac_create_role(GV_RBACManager *mgr, const char *role_name);
int gv_rbac_delete_role(GV_RBACManager *mgr, const char *role_name);
int gv_rbac_add_rule(GV_RBACManager *mgr, const char *role_name,
                      const char *resource, uint32_t permissions);
int gv_rbac_remove_rule(GV_RBACManager *mgr, const char *role_name, const char *resource);
int gv_rbac_set_inheritance(GV_RBACManager *mgr, const char *role_name, const char *parent_role);

/* User-role assignment */
int gv_rbac_assign_role(GV_RBACManager *mgr, const char *user_id, const char *role_name);
int gv_rbac_revoke_role(GV_RBACManager *mgr, const char *user_id, const char *role_name);
int gv_rbac_get_user_roles(const GV_RBACManager *mgr, const char *user_id,
                            char ***out_roles, size_t *out_count);

/* Authorization check */
int gv_rbac_check(const GV_RBACManager *mgr, const char *user_id,
                   const char *resource, GV_Permission required);

/* List roles */
int gv_rbac_list_roles(const GV_RBACManager *mgr, char ***out_names, size_t *out_count);
void gv_rbac_free_string_list(char **list, size_t count);

/* Built-in roles */
int gv_rbac_init_defaults(GV_RBACManager *mgr);  /* Creates admin, writer, reader roles */

/* Save/load */
int gv_rbac_save(const GV_RBACManager *mgr, const char *filepath);
GV_RBACManager *gv_rbac_load(const char *filepath);

#ifdef __cplusplus
}
#endif
#endif
