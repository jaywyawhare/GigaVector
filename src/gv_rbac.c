/**
 * @file gv_rbac.c
 * @brief Fine-grained RBAC: collection-level and field-level permissions,
 *        custom roles with inheritance, save/load to JSON-like text format.
 */

#include "gigavector/gv_rbac.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

/*  Internal Constants  */

#define MAX_ROLES 64
#define MAX_RULES_PER_ROLE 32
#define MAX_USERS 256
#define MAX_ROLES_PER_USER 16
#define MAX_INHERITANCE_DEPTH 16

/*  Internal Structures  */

/**
 * @brief Internal rule entry for a single resource-permission pair.
 */
typedef struct {
    char *resource;
    uint32_t permissions;
} RuleEntry;

/**
 * @brief Internal role entry stored in the manager.
 */
typedef struct {
    char *name;
    RuleEntry rules[MAX_RULES_PER_ROLE];
    size_t rule_count;
    int parent_index;           /* Index into roles[], -1 for none */
} RoleEntry;

/**
 * @brief Internal user entry stored in the manager.
 */
typedef struct {
    char *user_id;
    char *role_names[MAX_ROLES_PER_USER];
    size_t role_count;
} UserEntry;

/**
 * @brief Opaque RBAC manager.
 */
struct GV_RBACManager {
    RoleEntry roles[MAX_ROLES];
    size_t role_count;

    UserEntry users[MAX_USERS];
    size_t user_count;

    pthread_rwlock_t rwlock;
};

/*  Internal Helpers  */

/**
 * @brief Find a role by name (caller must hold at least a read lock).
 */
static int find_role_index(const GV_RBACManager *mgr, const char *name) {
    for (size_t i = 0; i < mgr->role_count; i++) {
        if (strcmp(mgr->roles[i].name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief Find a user by id (caller must hold at least a read lock).
 */
static UserEntry *find_user(const GV_RBACManager *mgr, const char *user_id) {
    for (size_t i = 0; i < mgr->user_count; i++) {
        if (strcmp(mgr->users[i].user_id, user_id) == 0) {
            return (UserEntry *)&mgr->users[i];
        }
    }
    return NULL;
}

/**
 * @brief Check if a resource string matches a rule resource.
 *
 * Wildcard "*" matches everything. Otherwise exact match.
 */
static int resource_matches(const char *rule_resource, const char *target) {
    if (strcmp(rule_resource, "*") == 0) return 1;
    if (target == NULL) return 0;
    return strcmp(rule_resource, target) == 0;
}

/**
 * @brief Recursively check if a role (by index) grants the required permission
 *        on the given resource. Follows the inheritance chain up to
 *        MAX_INHERITANCE_DEPTH levels to prevent cycles.
 */
static int role_grants(const GV_RBACManager *mgr, int role_idx,
                       const char *resource, uint32_t required, int depth) {
    if (role_idx < 0 || role_idx >= (int)mgr->role_count) return 0;
    if (depth > MAX_INHERITANCE_DEPTH) return 0;

    const RoleEntry *role = &mgr->roles[role_idx];

    /* Check own rules */
    for (size_t i = 0; i < role->rule_count; i++) {
        if (resource_matches(role->rules[i].resource, resource)) {
            if ((role->rules[i].permissions & required) == required) {
                return 1;
            }
        }
    }

    /* Check parent role */
    if (role->parent_index >= 0) {
        return role_grants(mgr, role->parent_index, resource, required, depth + 1);
    }

    return 0;
}

/**
 * @brief Write a JSON-escaped string to file.
 */
static void write_json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const char *p = s; *p; p++) {
        switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\n': fputs("\\n", fp);  break;
            case '\r': fputs("\\r", fp);  break;
            case '\t': fputs("\\t", fp);  break;
            default:   fputc(*p, fp);     break;
        }
    }
    fputc('"', fp);
}

/*  Simple JSON-like Text Parser Helpers (for gv_rbac_load)  */

/**
 * @brief Skip whitespace characters.
 */
static void skip_ws(const char **pp) {
    while (**pp == ' ' || **pp == '\t' || **pp == '\n' || **pp == '\r') {
        (*pp)++;
    }
}

/**
 * @brief Expect and consume a specific character; return 0 on success.
 */
static int expect_char(const char **pp, char c) {
    skip_ws(pp);
    if (**pp != c) return -1;
    (*pp)++;
    return 0;
}

/**
 * @brief Parse a JSON string value. Caller must free the result.
 *        Returns NULL on failure.
 */
static char *parse_string(const char **pp) {
    skip_ws(pp);
    if (**pp != '"') return NULL;
    (*pp)++;

    size_t cap = 128;
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    while (**pp && **pp != '"') {
        char c = **pp;
        if (c == '\\') {
            (*pp)++;
            switch (**pp) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                default:   c = **pp; break;
            }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *tmp = realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
        buf[len++] = c;
        (*pp)++;
    }

    if (**pp != '"') { free(buf); return NULL; }
    (*pp)++;

    buf[len] = '\0';
    return buf;
}

/**
 * @brief Parse a non-negative integer. Returns -1 on failure.
 */
static int parse_int(const char **pp) {
    skip_ws(pp);
    if (**pp == '-') {
        (*pp)++;
        /* Only accept -1 for parent_index sentinel */
        if (**pp >= '0' && **pp <= '9') {
            int val = 0;
            while (**pp >= '0' && **pp <= '9') {
                val = val * 10 + (**pp - '0');
                (*pp)++;
            }
            return -val;
        }
        return -1;
    }
    if (**pp < '0' || **pp > '9') return -1;
    int val = 0;
    while (**pp >= '0' && **pp <= '9') {
        val = val * 10 + (**pp - '0');
        (*pp)++;
    }
    return val;
}

/**
 * @brief Expect a JSON key string followed by ':'. Returns the key or NULL.
 *        Caller must free.
 */
static char *parse_key(const char **pp) {
    char *key = parse_string(pp);
    if (!key) return NULL;
    skip_ws(pp);
    if (**pp != ':') { free(key); return NULL; }
    (*pp)++;
    return key;
}

/*  Lifecycle  */

GV_RBACManager *gv_rbac_create(void) {
    GV_RBACManager *mgr = calloc(1, sizeof(GV_RBACManager));
    if (!mgr) return NULL;

    if (pthread_rwlock_init(&mgr->rwlock, NULL) != 0) {
        free(mgr);
        return NULL;
    }

    return mgr;
}

void gv_rbac_destroy(GV_RBACManager *mgr) {
    if (!mgr) return;

    /* Free roles */
    for (size_t i = 0; i < mgr->role_count; i++) {
        free(mgr->roles[i].name);
        for (size_t j = 0; j < mgr->roles[i].rule_count; j++) {
            free(mgr->roles[i].rules[j].resource);
        }
    }

    /* Free users */
    for (size_t i = 0; i < mgr->user_count; i++) {
        free(mgr->users[i].user_id);
        for (size_t j = 0; j < mgr->users[i].role_count; j++) {
            free(mgr->users[i].role_names[j]);
        }
    }

    pthread_rwlock_destroy(&mgr->rwlock);
    free(mgr);
}

/*  Role Management  */

int gv_rbac_create_role(GV_RBACManager *mgr, const char *role_name) {
    if (!mgr || !role_name) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    /* Check duplicate */
    if (find_role_index(mgr, role_name) >= 0) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Check capacity */
    if (mgr->role_count >= MAX_ROLES) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    RoleEntry *role = &mgr->roles[mgr->role_count];
    role->name = strdup(role_name);
    if (!role->name) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }
    role->rule_count = 0;
    role->parent_index = -1;
    mgr->role_count++;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_rbac_delete_role(GV_RBACManager *mgr, const char *role_name) {
    if (!mgr || !role_name) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    int idx = find_role_index(mgr, role_name);
    if (idx < 0) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Free role data */
    free(mgr->roles[idx].name);
    for (size_t j = 0; j < mgr->roles[idx].rule_count; j++) {
        free(mgr->roles[idx].rules[j].resource);
    }

    /* Update parent references: any role that inherits from the deleted role
       gets its inheritance cleared. Any role that inherits from a role after
       the deleted one needs its index decremented. */
    for (size_t i = 0; i < mgr->role_count; i++) {
        if ((int)i == idx) continue;
        if (mgr->roles[i].parent_index == idx) {
            mgr->roles[i].parent_index = -1;
        } else if (mgr->roles[i].parent_index > idx) {
            mgr->roles[i].parent_index--;
        }
    }

    /* Shift remaining roles */
    for (size_t k = (size_t)idx; k < mgr->role_count - 1; k++) {
        mgr->roles[k] = mgr->roles[k + 1];
    }
    mgr->role_count--;

    /* Zero the now-unused slot to prevent stale pointers */
    memset(&mgr->roles[mgr->role_count], 0, sizeof(RoleEntry));

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_rbac_add_rule(GV_RBACManager *mgr, const char *role_name,
                      const char *resource, uint32_t permissions) {
    if (!mgr || !role_name || !resource) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    int idx = find_role_index(mgr, role_name);
    if (idx < 0) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    RoleEntry *role = &mgr->roles[idx];

    /* Check if a rule for this resource already exists; if so, update */
    for (size_t i = 0; i < role->rule_count; i++) {
        if (strcmp(role->rules[i].resource, resource) == 0) {
            role->rules[i].permissions = permissions;
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    /* Add new rule */
    if (role->rule_count >= MAX_RULES_PER_ROLE) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    role->rules[role->rule_count].resource = strdup(resource);
    if (!role->rules[role->rule_count].resource) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }
    role->rules[role->rule_count].permissions = permissions;
    role->rule_count++;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_rbac_remove_rule(GV_RBACManager *mgr, const char *role_name,
                         const char *resource) {
    if (!mgr || !role_name || !resource) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    int idx = find_role_index(mgr, role_name);
    if (idx < 0) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    RoleEntry *role = &mgr->roles[idx];

    for (size_t i = 0; i < role->rule_count; i++) {
        if (strcmp(role->rules[i].resource, resource) == 0) {
            free(role->rules[i].resource);
            /* Shift remaining rules */
            for (size_t k = i; k < role->rule_count - 1; k++) {
                role->rules[k] = role->rules[k + 1];
            }
            role->rule_count--;
            memset(&role->rules[role->rule_count], 0, sizeof(RuleEntry));
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;
}

int gv_rbac_set_inheritance(GV_RBACManager *mgr, const char *role_name,
                             const char *parent_role) {
    if (!mgr || !role_name) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    int child_idx = find_role_index(mgr, role_name);
    if (child_idx < 0) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* NULL parent_role clears inheritance */
    if (!parent_role) {
        mgr->roles[child_idx].parent_index = -1;
        pthread_rwlock_unlock(&mgr->rwlock);
        return 0;
    }

    int parent_idx = find_role_index(mgr, parent_role);
    if (parent_idx < 0) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Prevent self-inheritance */
    if (child_idx == parent_idx) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Detect cycles: walk parent chain from proposed parent and ensure we
       never reach child_idx */
    int cur = parent_idx;
    int depth = 0;
    while (cur >= 0 && depth < MAX_INHERITANCE_DEPTH) {
        if (cur == child_idx) {
            /* Cycle detected */
            pthread_rwlock_unlock(&mgr->rwlock);
            return -1;
        }
        cur = mgr->roles[cur].parent_index;
        depth++;
    }

    mgr->roles[child_idx].parent_index = parent_idx;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

/*  User-Role Assignment  */

int gv_rbac_assign_role(GV_RBACManager *mgr, const char *user_id,
                         const char *role_name) {
    if (!mgr || !user_id || !role_name) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    /* Verify role exists */
    if (find_role_index(mgr, role_name) < 0) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    /* Find or create user */
    UserEntry *user = find_user(mgr, user_id);
    if (!user) {
        if (mgr->user_count >= MAX_USERS) {
            pthread_rwlock_unlock(&mgr->rwlock);
            return -1;
        }
        user = &mgr->users[mgr->user_count];
        user->user_id = strdup(user_id);
        if (!user->user_id) {
            pthread_rwlock_unlock(&mgr->rwlock);
            return -1;
        }
        user->role_count = 0;
        mgr->user_count++;
    }

    /* Check if already assigned */
    for (size_t i = 0; i < user->role_count; i++) {
        if (strcmp(user->role_names[i], role_name) == 0) {
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;  /* Already assigned, treat as success */
        }
    }

    /* Add role */
    if (user->role_count >= MAX_ROLES_PER_USER) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    user->role_names[user->role_count] = strdup(role_name);
    if (!user->role_names[user->role_count]) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }
    user->role_count++;

    pthread_rwlock_unlock(&mgr->rwlock);
    return 0;
}

int gv_rbac_revoke_role(GV_RBACManager *mgr, const char *user_id,
                         const char *role_name) {
    if (!mgr || !user_id || !role_name) return -1;

    pthread_rwlock_wrlock(&mgr->rwlock);

    UserEntry *user = find_user(mgr, user_id);
    if (!user) {
        pthread_rwlock_unlock(&mgr->rwlock);
        return -1;
    }

    for (size_t i = 0; i < user->role_count; i++) {
        if (strcmp(user->role_names[i], role_name) == 0) {
            free(user->role_names[i]);
            for (size_t k = i; k < user->role_count - 1; k++) {
                user->role_names[k] = user->role_names[k + 1];
            }
            user->role_count--;
            user->role_names[user->role_count] = NULL;
            pthread_rwlock_unlock(&mgr->rwlock);
            return 0;
        }
    }

    pthread_rwlock_unlock(&mgr->rwlock);
    return -1;
}

int gv_rbac_get_user_roles(const GV_RBACManager *mgr, const char *user_id,
                            char ***out_roles, size_t *out_count) {
    if (!mgr || !user_id || !out_roles || !out_count) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    const UserEntry *user = find_user(mgr, user_id);
    if (!user || user->role_count == 0) {
        *out_roles = NULL;
        *out_count = 0;
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return 0;
    }

    *out_count = user->role_count;
    *out_roles = malloc(user->role_count * sizeof(char *));
    if (!*out_roles) {
        *out_count = 0;
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    for (size_t i = 0; i < user->role_count; i++) {
        (*out_roles)[i] = strdup(user->role_names[i]);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;
}

/*  Authorization Check  */

int gv_rbac_check(const GV_RBACManager *mgr, const char *user_id,
                   const char *resource, GV_Permission required) {
    if (!mgr || !user_id || !resource) return 0;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    const UserEntry *user = find_user(mgr, user_id);
    if (!user) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return 0;
    }

    /* For each assigned role, check if it (or its ancestors) grant access */
    for (size_t i = 0; i < user->role_count; i++) {
        int role_idx = find_role_index(mgr, user->role_names[i]);
        if (role_idx < 0) continue;

        if (role_grants(mgr, role_idx, resource, (uint32_t)required, 0)) {
            pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
            return 1;
        }
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;
}

/*  List Roles / Free Helpers  */

int gv_rbac_list_roles(const GV_RBACManager *mgr, char ***out_names,
                        size_t *out_count) {
    if (!mgr || !out_names || !out_count) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    *out_count = mgr->role_count;
    if (mgr->role_count == 0) {
        *out_names = NULL;
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return 0;
    }

    *out_names = malloc(mgr->role_count * sizeof(char *));
    if (!*out_names) {
        *out_count = 0;
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    for (size_t i = 0; i < mgr->role_count; i++) {
        (*out_names)[i] = strdup(mgr->roles[i].name);
    }

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;
}

void gv_rbac_free_string_list(char **list, size_t count) {
    if (!list) return;
    for (size_t i = 0; i < count; i++) {
        free(list[i]);
    }
    free(list);
}

/*  Built-in Roles  */

int gv_rbac_init_defaults(GV_RBACManager *mgr) {
    if (!mgr) return -1;

    /* admin: ALL permissions on all resources */
    if (gv_rbac_create_role(mgr, "admin") != 0) return -1;
    if (gv_rbac_add_rule(mgr, "admin", "*", GV_PERM_ALL) != 0) return -1;

    /* writer: READ | WRITE on all resources */
    if (gv_rbac_create_role(mgr, "writer") != 0) return -1;
    if (gv_rbac_add_rule(mgr, "writer", "*", GV_PERM_READ | GV_PERM_WRITE) != 0) return -1;

    /* reader: READ on all resources */
    if (gv_rbac_create_role(mgr, "reader") != 0) return -1;
    if (gv_rbac_add_rule(mgr, "reader", "*", GV_PERM_READ) != 0) return -1;

    return 0;
}

/*  Save (JSON-like text format)  */

int gv_rbac_save(const GV_RBACManager *mgr, const char *filepath) {
    if (!mgr || !filepath) return -1;

    pthread_rwlock_rdlock((pthread_rwlock_t *)&mgr->rwlock);

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
        return -1;
    }

    fprintf(fp, "{\n");

    /* Roles */
    fprintf(fp, "  \"roles\": [\n");
    for (size_t i = 0; i < mgr->role_count; i++) {
        const RoleEntry *role = &mgr->roles[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": ");
        write_json_string(fp, role->name);
        fprintf(fp, ",\n");

        /* Parent role name (resolve index to name, or "none") */
        fprintf(fp, "      \"inherits\": ");
        if (role->parent_index >= 0 && role->parent_index < (int)mgr->role_count) {
            write_json_string(fp, mgr->roles[role->parent_index].name);
        } else {
            fprintf(fp, "\"none\"");
        }
        fprintf(fp, ",\n");

        /* Rules */
        fprintf(fp, "      \"rules\": [\n");
        for (size_t j = 0; j < role->rule_count; j++) {
            fprintf(fp, "        { \"resource\": ");
            write_json_string(fp, role->rules[j].resource);
            fprintf(fp, ", \"permissions\": %u }", role->rules[j].permissions);
            if (j + 1 < role->rule_count) fprintf(fp, ",");
            fprintf(fp, "\n");
        }
        fprintf(fp, "      ]\n");

        fprintf(fp, "    }");
        if (i + 1 < mgr->role_count) fprintf(fp, ",");
        fprintf(fp, "\n");
    }
    fprintf(fp, "  ],\n");

    /* Users */
    fprintf(fp, "  \"users\": [\n");
    for (size_t i = 0; i < mgr->user_count; i++) {
        const UserEntry *user = &mgr->users[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"user_id\": ");
        write_json_string(fp, user->user_id);
        fprintf(fp, ",\n");

        fprintf(fp, "      \"roles\": [");
        for (size_t j = 0; j < user->role_count; j++) {
            if (j > 0) fprintf(fp, ", ");
            write_json_string(fp, user->role_names[j]);
        }
        fprintf(fp, "]\n");

        fprintf(fp, "    }");
        if (i + 1 < mgr->user_count) fprintf(fp, ",");
        fprintf(fp, "\n");
    }
    fprintf(fp, "  ]\n");

    fprintf(fp, "}\n");

    fclose(fp);

    pthread_rwlock_unlock((pthread_rwlock_t *)&mgr->rwlock);
    return 0;
}

/*  Load (JSON-like text format)  */

GV_RBACManager *gv_rbac_load(const char *filepath) {
    if (!filepath) return NULL;

    FILE *fp = fopen(filepath, "r");
    if (!fp) return NULL;

    /* Read entire file */
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize <= 0) { fclose(fp); return NULL; }
    fseek(fp, 0, SEEK_SET);

    char *data = malloc((size_t)fsize + 1);
    if (!data) { fclose(fp); return NULL; }

    size_t nread = fread(data, 1, (size_t)fsize, fp);
    fclose(fp);
    data[nread] = '\0';

    GV_RBACManager *mgr = gv_rbac_create();
    if (!mgr) { free(data); return NULL; }

    const char *p = data;

    /* We need to keep track of inheritance names and resolve after all roles
       are parsed, since parent roles may appear after child roles. */
    char *inherit_names[MAX_ROLES];
    memset(inherit_names, 0, sizeof(inherit_names));

    /* Expect top-level object */
    if (expect_char(&p, '{') != 0) goto fail;

    /* Parse top-level keys */
    while (*p) {
        skip_ws(&p);
        if (*p == '}') break;
        if (*p == ',') { p++; continue; }

        char *key = parse_key(&p);
        if (!key) goto fail;

        if (strcmp(key, "roles") == 0) {
            free(key);

            /* Parse roles array */
            if (expect_char(&p, '[') != 0) goto fail;

            while (*p) {
                skip_ws(&p);
                if (*p == ']') { p++; break; }
                if (*p == ',') { p++; continue; }

                /* Parse role object */
                if (expect_char(&p, '{') != 0) goto fail;

                char *role_name = NULL;
                char *inherits = NULL;

                /* Temporary storage for rules during parsing */
                char *rule_resources[MAX_RULES_PER_ROLE];
                uint32_t rule_permissions[MAX_RULES_PER_ROLE];
                size_t rule_count = 0;

                while (*p) {
                    skip_ws(&p);
                    if (*p == '}') { p++; break; }
                    if (*p == ',') { p++; continue; }

                    char *rkey = parse_key(&p);
                    if (!rkey) {
                        free(role_name);
                        free(inherits);
                        for (size_t ri = 0; ri < rule_count; ri++) free(rule_resources[ri]);
                        goto fail;
                    }

                    if (strcmp(rkey, "name") == 0) {
                        free(rkey);
                        role_name = parse_string(&p);
                    } else if (strcmp(rkey, "inherits") == 0) {
                        free(rkey);
                        inherits = parse_string(&p);
                    } else if (strcmp(rkey, "rules") == 0) {
                        free(rkey);

                        /* Parse rules array */
                        if (expect_char(&p, '[') != 0) {
                            free(role_name);
                            free(inherits);
                            goto fail;
                        }

                        while (*p) {
                            skip_ws(&p);
                            if (*p == ']') { p++; break; }
                            if (*p == ',') { p++; continue; }

                            /* Parse rule object */
                            if (expect_char(&p, '{') != 0) {
                                free(role_name);
                                free(inherits);
                                for (size_t ri = 0; ri < rule_count; ri++) free(rule_resources[ri]);
                                goto fail;
                            }

                            char *res = NULL;
                            uint32_t perms = 0;

                            while (*p) {
                                skip_ws(&p);
                                if (*p == '}') { p++; break; }
                                if (*p == ',') { p++; continue; }

                                char *rrkey = parse_key(&p);
                                if (!rrkey) {
                                    free(res);
                                    free(role_name);
                                    free(inherits);
                                    for (size_t ri = 0; ri < rule_count; ri++) free(rule_resources[ri]);
                                    goto fail;
                                }

                                if (strcmp(rrkey, "resource") == 0) {
                                    free(rrkey);
                                    res = parse_string(&p);
                                } else if (strcmp(rrkey, "permissions") == 0) {
                                    free(rrkey);
                                    int v = parse_int(&p);
                                    perms = (v >= 0) ? (uint32_t)v : 0;
                                } else {
                                    free(rrkey);
                                    /* Skip unknown value */
                                    parse_string(&p);
                                }
                            }

                            if (res && rule_count < MAX_RULES_PER_ROLE) {
                                rule_resources[rule_count] = res;
                                rule_permissions[rule_count] = perms;
                                rule_count++;
                            } else {
                                free(res);
                            }
                        }
                    } else {
                        free(rkey);
                        /* Skip unknown value */
                        skip_ws(&p);
                        if (*p == '"') {
                            char *tmp = parse_string(&p);
                            free(tmp);
                        } else if (*p == '[') {
                            /* Skip array: find matching ] */
                            int depth = 1;
                            p++;
                            while (*p && depth > 0) {
                                if (*p == '[') depth++;
                                else if (*p == ']') depth--;
                                p++;
                            }
                        } else {
                            parse_int(&p);
                        }
                    }
                }

                /* Add role to manager (without lock, we own it exclusively) */
                if (role_name && mgr->role_count < MAX_ROLES) {
                    size_t ri = mgr->role_count;
                    mgr->roles[ri].name = role_name;
                    mgr->roles[ri].rule_count = 0;
                    mgr->roles[ri].parent_index = -1;

                    for (size_t rr = 0; rr < rule_count; rr++) {
                        mgr->roles[ri].rules[rr].resource = rule_resources[rr];
                        mgr->roles[ri].rules[rr].permissions = rule_permissions[rr];
                    }
                    mgr->roles[ri].rule_count = rule_count;

                    /* Store inheritance name for later resolution */
                    if (inherits && strcmp(inherits, "none") != 0) {
                        inherit_names[ri] = inherits;
                    } else {
                        free(inherits);
                    }

                    mgr->role_count++;
                } else {
                    free(role_name);
                    free(inherits);
                    for (size_t rr = 0; rr < rule_count; rr++) {
                        free(rule_resources[rr]);
                    }
                }
            }
        } else if (strcmp(key, "users") == 0) {
            free(key);

            /* Parse users array */
            if (expect_char(&p, '[') != 0) goto fail;

            while (*p) {
                skip_ws(&p);
                if (*p == ']') { p++; break; }
                if (*p == ',') { p++; continue; }

                /* Parse user object */
                if (expect_char(&p, '{') != 0) goto fail;

                char *uid = NULL;
                char *parsed_roles[MAX_ROLES_PER_USER];
                size_t prole_count = 0;

                while (*p) {
                    skip_ws(&p);
                    if (*p == '}') { p++; break; }
                    if (*p == ',') { p++; continue; }

                    char *ukey = parse_key(&p);
                    if (!ukey) {
                        free(uid);
                        for (size_t ri = 0; ri < prole_count; ri++) free(parsed_roles[ri]);
                        goto fail;
                    }

                    if (strcmp(ukey, "user_id") == 0) {
                        free(ukey);
                        uid = parse_string(&p);
                    } else if (strcmp(ukey, "roles") == 0) {
                        free(ukey);

                        if (expect_char(&p, '[') != 0) {
                            free(uid);
                            for (size_t ri = 0; ri < prole_count; ri++) free(parsed_roles[ri]);
                            goto fail;
                        }

                        while (*p) {
                            skip_ws(&p);
                            if (*p == ']') { p++; break; }
                            if (*p == ',') { p++; continue; }

                            char *rn = parse_string(&p);
                            if (rn && prole_count < MAX_ROLES_PER_USER) {
                                parsed_roles[prole_count++] = rn;
                            } else {
                                free(rn);
                            }
                        }
                    } else {
                        free(ukey);
                        skip_ws(&p);
                        if (*p == '"') {
                            char *tmp = parse_string(&p);
                            free(tmp);
                        } else {
                            parse_int(&p);
                        }
                    }
                }

                /* Add user to manager */
                if (uid && mgr->user_count < MAX_USERS) {
                    size_t ui = mgr->user_count;
                    mgr->users[ui].user_id = uid;
                    mgr->users[ui].role_count = 0;

                    for (size_t rr = 0; rr < prole_count; rr++) {
                        mgr->users[ui].role_names[rr] = parsed_roles[rr];
                    }
                    mgr->users[ui].role_count = prole_count;

                    mgr->user_count++;
                } else {
                    free(uid);
                    for (size_t rr = 0; rr < prole_count; rr++) {
                        free(parsed_roles[rr]);
                    }
                }
            }
        } else {
            free(key);
            /* Skip unknown top-level value */
            skip_ws(&p);
            if (*p == '"') {
                char *tmp = parse_string(&p);
                free(tmp);
            } else if (*p == '[') {
                int depth = 1;
                p++;
                while (*p && depth > 0) {
                    if (*p == '[') depth++;
                    else if (*p == ']') depth--;
                    p++;
                }
            } else if (*p == '{') {
                int depth = 1;
                p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            } else {
                parse_int(&p);
            }
        }
    }

    /* Resolve inheritance names to indices */
    for (size_t i = 0; i < mgr->role_count; i++) {
        if (inherit_names[i]) {
            int pidx = find_role_index(mgr, inherit_names[i]);
            mgr->roles[i].parent_index = pidx;
            free(inherit_names[i]);
            inherit_names[i] = NULL;
        }
    }

    free(data);
    return mgr;

fail:
    /* Clean up inheritance names on failure */
    for (size_t i = 0; i < MAX_ROLES; i++) {
        free(inherit_names[i]);
    }
    free(data);
    gv_rbac_destroy(mgr);
    return NULL;
}
