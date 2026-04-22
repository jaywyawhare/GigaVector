/**
 * @file authz.c
 * @brief Authorization (RBAC) implementation.
 */

#include "security/authz.h"
#include "core/utils.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* Internal Structures */

#define MAX_ROLES 64
#define MAX_USERS 256
#define MAX_NAMESPACES_PER_ROLE 32
#define MAX_ROLES_PER_USER 8

/**
 * @brief Role entry.
 */
typedef struct {
    char *name;
    uint32_t permissions;
    char *namespaces[MAX_NAMESPACES_PER_ROLE];
    size_t namespace_count;
} RoleEntry;

/**
 * @brief User entry.
 */
typedef struct {
    char *subject;
    char *roles[MAX_ROLES_PER_USER];
    size_t role_count;
} UserEntry;

/**
 * @brief Authorization manager structure.
 */
struct GV_AuthzManager {
    RoleEntry roles[MAX_ROLES];
    size_t role_count;

    UserEntry users[MAX_USERS];
    size_t user_count;

    pthread_rwlock_t rwlock;
};

/* Lifecycle */

GV_AuthzManager *authz_create(void) {
    GV_AuthzManager *authz = calloc(1, sizeof(GV_AuthzManager));
    if (!authz) return NULL;

    if (pthread_rwlock_init(&authz->rwlock, NULL) != 0) {
        free(authz);
        return NULL;
    }

    return authz;
}

void authz_destroy(GV_AuthzManager *authz) {
    if (!authz) return;

    /* Free roles */
    for (size_t i = 0; i < authz->role_count; i++) {
        free(authz->roles[i].name);
        for (size_t j = 0; j < authz->roles[i].namespace_count; j++) {
            free(authz->roles[i].namespaces[j]);
        }
    }

    /* Free users */
    for (size_t i = 0; i < authz->user_count; i++) {
        free(authz->users[i].subject);
        for (size_t j = 0; j < authz->users[i].role_count; j++) {
            free(authz->users[i].roles[j]);
        }
    }

    pthread_rwlock_destroy(&authz->rwlock);
    free(authz);
}

/* Internal Helpers */

static RoleEntry *find_role(GV_AuthzManager *authz, const char *name) {
    for (size_t i = 0; i < authz->role_count; i++) {
        if (strcmp(authz->roles[i].name, name) == 0) {
            return &authz->roles[i];
        }
    }
    return NULL;
}

static UserEntry *find_user(GV_AuthzManager *authz, const char *subject) {
    for (size_t i = 0; i < authz->user_count; i++) {
        if (strcmp(authz->users[i].subject, subject) == 0) {
            return &authz->users[i];
        }
    }
    return NULL;
}

static int namespace_allowed(RoleEntry *role, const char *namespace_name) {
    /* If no namespace restrictions, all are allowed */
    if (role->namespace_count == 0) return 1;

    /* NULL namespace = global, check if global is in allowed list */
    if (namespace_name == NULL) {
        for (size_t i = 0; i < role->namespace_count; i++) {
            if (strcmp(role->namespaces[i], "*") == 0) return 1;
        }
        return 0;
    }

    /* Check specific namespace */
    for (size_t i = 0; i < role->namespace_count; i++) {
        if (strcmp(role->namespaces[i], namespace_name) == 0 ||
            strcmp(role->namespaces[i], "*") == 0) {
            return 1;
        }
    }
    return 0;
}

/* Role Management */

int authz_define_role(GV_AuthzManager *authz, const char *name,
                          uint32_t permissions, const char **namespaces,
                          size_t namespace_count) {
    if (!authz || !name) return -1;

    pthread_rwlock_wrlock(&authz->rwlock);

    /* Check if role exists */
    RoleEntry *existing = find_role(authz, name);
    if (existing) {
        /* Update existing role */
        existing->permissions = permissions;

        /* Clear old namespaces */
        for (size_t i = 0; i < existing->namespace_count; i++) {
            free(existing->namespaces[i]);
        }
        existing->namespace_count = 0;

        /* Add new namespaces */
        for (size_t i = 0; i < namespace_count && i < MAX_NAMESPACES_PER_ROLE; i++) {
            existing->namespaces[i] = gv_dup_cstr(namespaces[i]);
            existing->namespace_count++;
        }

        pthread_rwlock_unlock(&authz->rwlock);
        return 0;
    }

    /* Add new role */
    if (authz->role_count >= MAX_ROLES) {
        pthread_rwlock_unlock(&authz->rwlock);
        return -1;
    }

    RoleEntry *role = &authz->roles[authz->role_count];
    role->name = gv_dup_cstr(name);
    role->permissions = permissions;
    role->namespace_count = 0;

    for (size_t i = 0; i < namespace_count && i < MAX_NAMESPACES_PER_ROLE; i++) {
        role->namespaces[i] = gv_dup_cstr(namespaces[i]);
        role->namespace_count++;
    }

    authz->role_count++;

    pthread_rwlock_unlock(&authz->rwlock);
    return 0;
}

int authz_remove_role(GV_AuthzManager *authz, const char *name) {
    if (!authz || !name) return -1;

    pthread_rwlock_wrlock(&authz->rwlock);

    for (size_t i = 0; i < authz->role_count; i++) {
        if (strcmp(authz->roles[i].name, name) == 0) {
            /* Free role data */
            free(authz->roles[i].name);
            for (size_t j = 0; j < authz->roles[i].namespace_count; j++) {
                free(authz->roles[i].namespaces[j]);
            }

            /* Shift remaining roles */
            for (size_t k = i; k < authz->role_count - 1; k++) {
                authz->roles[k] = authz->roles[k + 1];
            }
            authz->role_count--;

            pthread_rwlock_unlock(&authz->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&authz->rwlock);
    return -1;
}

int authz_get_role(GV_AuthzManager *authz, const char *name, GV_Role *role) {
    if (!authz || !name || !role) return -1;

    pthread_rwlock_rdlock(&authz->rwlock);

    RoleEntry *entry = find_role(authz, name);
    if (!entry) {
        pthread_rwlock_unlock(&authz->rwlock);
        return -1;
    }

    role->name = gv_dup_cstr(entry->name);
    role->permissions = entry->permissions;

    if (entry->namespace_count > 0) {
        role->allowed_namespaces = malloc(entry->namespace_count * sizeof(char *));
        for (size_t i = 0; i < entry->namespace_count; i++) {
            role->allowed_namespaces[i] = gv_dup_cstr(entry->namespaces[i]);
        }
        role->namespace_count = entry->namespace_count;
    } else {
        role->allowed_namespaces = NULL;
        role->namespace_count = 0;
    }

    pthread_rwlock_unlock(&authz->rwlock);
    return 0;
}

int authz_list_roles(GV_AuthzManager *authz, GV_Role **roles, size_t *count) {
    if (!authz || !roles || !count) return -1;

    pthread_rwlock_rdlock(&authz->rwlock);

    *count = authz->role_count;
    if (*count == 0) {
        *roles = NULL;
        pthread_rwlock_unlock(&authz->rwlock);
        return 0;
    }

    *roles = malloc(*count * sizeof(GV_Role));
    if (!*roles) {
        pthread_rwlock_unlock(&authz->rwlock);
        return -1;
    }

    for (size_t i = 0; i < *count; i++) {
        (*roles)[i].name = gv_dup_cstr(authz->roles[i].name);
        (*roles)[i].permissions = authz->roles[i].permissions;

        if (authz->roles[i].namespace_count > 0) {
            (*roles)[i].allowed_namespaces = malloc(authz->roles[i].namespace_count * sizeof(char *));
            for (size_t j = 0; j < authz->roles[i].namespace_count; j++) {
                (*roles)[i].allowed_namespaces[j] = gv_dup_cstr(authz->roles[i].namespaces[j]);
            }
            (*roles)[i].namespace_count = authz->roles[i].namespace_count;
        } else {
            (*roles)[i].allowed_namespaces = NULL;
            (*roles)[i].namespace_count = 0;
        }
    }

    pthread_rwlock_unlock(&authz->rwlock);
    return 0;
}

void authz_free_role(GV_Role *role) {
    if (!role) return;
    free(role->name);
    for (size_t i = 0; i < role->namespace_count; i++) {
        free(role->allowed_namespaces[i]);
    }
    free(role->allowed_namespaces);
    memset(role, 0, sizeof(*role));
}

void authz_free_roles(GV_Role *roles, size_t count) {
    if (!roles) return;
    for (size_t i = 0; i < count; i++) {
        authz_free_role(&roles[i]);
    }
    free(roles);
}

/* User-Role Assignment */

int authz_assign_role(GV_AuthzManager *authz, const char *subject,
                          const char *role_name) {
    if (!authz || !subject || !role_name) return -1;

    pthread_rwlock_wrlock(&authz->rwlock);

    /* Verify role exists */
    if (!find_role(authz, role_name)) {
        pthread_rwlock_unlock(&authz->rwlock);
        return -1;
    }

    /* Find or create user */
    UserEntry *user = find_user(authz, subject);
    if (!user) {
        if (authz->user_count >= MAX_USERS) {
            pthread_rwlock_unlock(&authz->rwlock);
            return -1;
        }
        user = &authz->users[authz->user_count++];
        user->subject = gv_dup_cstr(subject);
        user->role_count = 0;
    }

    /* Check if already assigned */
    for (size_t i = 0; i < user->role_count; i++) {
        if (strcmp(user->roles[i], role_name) == 0) {
            pthread_rwlock_unlock(&authz->rwlock);
            return 0;  /* Already assigned */
        }
    }

    /* Add role */
    if (user->role_count >= MAX_ROLES_PER_USER) {
        pthread_rwlock_unlock(&authz->rwlock);
        return -1;
    }

    user->roles[user->role_count++] = gv_dup_cstr(role_name);

    pthread_rwlock_unlock(&authz->rwlock);
    return 0;
}

int authz_revoke_role(GV_AuthzManager *authz, const char *subject,
                          const char *role_name) {
    if (!authz || !subject || !role_name) return -1;

    pthread_rwlock_wrlock(&authz->rwlock);

    UserEntry *user = find_user(authz, subject);
    if (!user) {
        pthread_rwlock_unlock(&authz->rwlock);
        return -1;
    }

    for (size_t i = 0; i < user->role_count; i++) {
        if (strcmp(user->roles[i], role_name) == 0) {
            free(user->roles[i]);
            for (size_t k = i; k < user->role_count - 1; k++) {
                user->roles[k] = user->roles[k + 1];
            }
            user->role_count--;
            pthread_rwlock_unlock(&authz->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&authz->rwlock);
    return -1;
}

int authz_get_user_roles(GV_AuthzManager *authz, const char *subject,
                             char ***roles, size_t *count) {
    if (!authz || !subject || !roles || !count) return -1;

    pthread_rwlock_rdlock(&authz->rwlock);

    UserEntry *user = find_user(authz, subject);
    if (!user) {
        *roles = NULL;
        *count = 0;
        pthread_rwlock_unlock(&authz->rwlock);
        return 0;
    }

    *count = user->role_count;
    if (*count == 0) {
        *roles = NULL;
        pthread_rwlock_unlock(&authz->rwlock);
        return 0;
    }

    *roles = malloc(*count * sizeof(char *));
    for (size_t i = 0; i < *count; i++) {
        (*roles)[i] = gv_dup_cstr(user->roles[i]);
    }

    pthread_rwlock_unlock(&authz->rwlock);
    return 0;
}

void authz_free_user_roles(char **roles, size_t count) {
    if (!roles) return;
    for (size_t i = 0; i < count; i++) {
        free(roles[i]);
    }
    free(roles);
}

/* Authorization Checks */

int authz_check(GV_AuthzManager *authz, const GV_Identity *identity,
                    GV_Permission permission, GV_ResourceType resource_type,
                    const char *resource_name, GV_AuthzResult *result) {
    if (!authz || !identity || !result) return -1;

    result->allowed = 0;
    result->denied_reason = "No matching role";
    result->matched_role = NULL;

    if (!identity->subject) {
        result->denied_reason = "No subject in identity";
        return 0;
    }

    pthread_rwlock_rdlock(&authz->rwlock);

    UserEntry *user = find_user(authz, identity->subject);
    if (!user) {
        pthread_rwlock_unlock(&authz->rwlock);
        result->denied_reason = "User not found";
        return 0;
    }

    /* Check each role */
    for (size_t i = 0; i < user->role_count; i++) {
        RoleEntry *role = find_role(authz, user->roles[i]);
        if (!role) continue;

        /* Check permission */
        if ((role->permissions & permission) == 0) continue;

        /* Check namespace access */
        if (resource_type == GV_RESOURCE_NAMESPACE ||
            resource_type == GV_RESOURCE_VECTOR) {
            if (!namespace_allowed(role, resource_name)) continue;
        }

        /* Access granted */
        result->allowed = 1;
        result->denied_reason = NULL;
        result->matched_role = role->name;
        pthread_rwlock_unlock(&authz->rwlock);
        return 0;
    }

    pthread_rwlock_unlock(&authz->rwlock);
    return 0;
}

int authz_can_read(GV_AuthzManager *authz, const GV_Identity *identity,
                       const char *namespace_name) {
    GV_AuthzResult result;
    if (authz_check(authz, identity, GV_PERM_READ, GV_RESOURCE_NAMESPACE,
                        namespace_name, &result) != 0) {
        return 0;
    }
    return result.allowed;
}

int authz_can_write(GV_AuthzManager *authz, const GV_Identity *identity,
                        const char *namespace_name) {
    GV_AuthzResult result;
    if (authz_check(authz, identity, GV_PERM_WRITE, GV_RESOURCE_NAMESPACE,
                        namespace_name, &result) != 0) {
        return 0;
    }
    return result.allowed;
}

int authz_can_delete(GV_AuthzManager *authz, const GV_Identity *identity,
                         const char *namespace_name) {
    GV_AuthzResult result;
    if (authz_check(authz, identity, GV_PERM_DELETE, GV_RESOURCE_NAMESPACE,
                        namespace_name, &result) != 0) {
        return 0;
    }
    return result.allowed;
}

int authz_is_admin(GV_AuthzManager *authz, const GV_Identity *identity) {
    GV_AuthzResult result;
    if (authz_check(authz, identity, GV_PERM_ADMIN, GV_RESOURCE_GLOBAL,
                        NULL, &result) != 0) {
        return 0;
    }
    return result.allowed;
}

/* Built-in Roles */

int authz_init_builtin_roles(GV_AuthzManager *authz) {
    if (!authz) return -1;

    /* Admin role - all permissions on all namespaces */
    if (authz_define_role(authz, "admin", GV_PERM_ALL, NULL, 0) != 0) {
        return -1;
    }

    /* Writer role - read and write */
    if (authz_define_role(authz, "writer", GV_PERM_READ | GV_PERM_WRITE, NULL, 0) != 0) {
        return -1;
    }

    /* Reader role - read only */
    if (authz_define_role(authz, "reader", GV_PERM_READ, NULL, 0) != 0) {
        return -1;
    }

    return 0;
}

const char *authz_permission_string(GV_Permission permission) {
    switch (permission) {
        case GV_PERM_NONE: return "none";
        case GV_PERM_READ: return "read";
        case GV_PERM_WRITE: return "write";
        case GV_PERM_DELETE: return "delete";
        case GV_PERM_ADMIN: return "admin";
        case GV_PERM_ALL: return "all";
        default: return "unknown";
    }
}
