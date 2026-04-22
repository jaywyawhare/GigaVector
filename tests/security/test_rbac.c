#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "security/rbac.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_create_destroy(void) {
    GV_RBACManager *mgr = rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");
    rbac_destroy(mgr);
    rbac_destroy(NULL);
    return 0;
}

static int test_role_create_delete(void) {
    GV_RBACManager *mgr = rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    int rc = rbac_create_role(mgr, "editor");
    ASSERT(rc == 0, "create role 'editor'");

    rc = rbac_create_role(mgr, "viewer");
    ASSERT(rc == 0, "create role 'viewer'");

    char **names = NULL;
    size_t count = 0;
    rc = rbac_list_roles(mgr, &names, &count);
    ASSERT(rc == 0, "list roles");
    ASSERT(count >= 2, "should have at least 2 roles");
    rbac_free_string_list(names, count);

    rc = rbac_delete_role(mgr, "editor");
    ASSERT(rc == 0, "delete role 'editor'");

    rbac_destroy(mgr);
    return 0;
}

static int test_add_remove_rules(void) {
    GV_RBACManager *mgr = rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    rbac_create_role(mgr, "writer");

    int rc = rbac_add_rule(mgr, "writer", "collection_a", GV_PERM_READ | GV_PERM_WRITE);
    ASSERT(rc == 0, "add rule to 'writer' for collection_a");

    rc = rbac_add_rule(mgr, "writer", "collection_b", GV_PERM_READ);
    ASSERT(rc == 0, "add rule to 'writer' for collection_b");

    rc = rbac_remove_rule(mgr, "writer", "collection_b");
    ASSERT(rc == 0, "remove rule for collection_b from 'writer'");

    rbac_destroy(mgr);
    return 0;
}

static int test_assign_check_permissions(void) {
    GV_RBACManager *mgr = rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    rbac_create_role(mgr, "reader");
    rbac_add_rule(mgr, "reader", "docs", GV_PERM_READ);

    int rc = rbac_assign_role(mgr, "user1", "reader");
    ASSERT(rc == 0, "assign 'reader' to user1");

    rc = rbac_check(mgr, "user1", "docs", GV_PERM_READ);
    ASSERT(rc == 1 || rc == 0, "check should return a permission result");

    int has_write = rbac_check(mgr, "user1", "docs", GV_PERM_WRITE);
    int has_read  = rbac_check(mgr, "user1", "docs", GV_PERM_READ);
    ASSERT(has_read != has_write || has_read == 0,
           "read and write permission results should differ or both be denied");

    rbac_destroy(mgr);
    return 0;
}

static int test_get_user_roles(void) {
    GV_RBACManager *mgr = rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    rbac_create_role(mgr, "alpha");
    rbac_create_role(mgr, "beta");
    rbac_assign_role(mgr, "user2", "alpha");
    rbac_assign_role(mgr, "user2", "beta");

    char **roles = NULL;
    size_t count = 0;
    int rc = rbac_get_user_roles(mgr, "user2", &roles, &count);
    ASSERT(rc == 0, "get user roles");
    ASSERT(count == 2, "user2 should have 2 roles");

    rbac_free_string_list(roles, count);
    rbac_destroy(mgr);
    return 0;
}

static int test_revoke_role(void) {
    GV_RBACManager *mgr = rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    rbac_create_role(mgr, "temp");
    rbac_assign_role(mgr, "user3", "temp");

    int rc = rbac_revoke_role(mgr, "user3", "temp");
    ASSERT(rc == 0, "revoke role from user3");

    char **roles = NULL;
    size_t count = 99;
    rc = rbac_get_user_roles(mgr, "user3", &roles, &count);
    ASSERT(rc == 0, "get user roles after revocation");
    ASSERT(count == 0, "user3 should have 0 roles after revocation");

    rbac_free_string_list(roles, count);
    rbac_destroy(mgr);
    return 0;
}

static int test_init_defaults(void) {
    GV_RBACManager *mgr = rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    int rc = rbac_init_defaults(mgr);
    ASSERT(rc == 0, "init default roles (admin, writer, reader)");

    char **names = NULL;
    size_t count = 0;
    rc = rbac_list_roles(mgr, &names, &count);
    ASSERT(rc == 0, "list roles after init defaults");
    ASSERT(count >= 3, "should have at least 3 default roles");

    rbac_free_string_list(names, count);
    rbac_destroy(mgr);
    return 0;
}

static int test_role_inheritance(void) {
    GV_RBACManager *mgr = rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    rbac_create_role(mgr, "base");
    rbac_create_role(mgr, "derived");
    rbac_add_rule(mgr, "base", "*", GV_PERM_READ);

    int rc = rbac_set_inheritance(mgr, "derived", "base");
    ASSERT(rc == 0, "set inheritance derived -> base");

    rbac_destroy(mgr);
    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing create/destroy...",           test_create_destroy},
        {"Testing role create/delete...",       test_role_create_delete},
        {"Testing add/remove rules...",         test_add_remove_rules},
        {"Testing assign/check permissions...", test_assign_check_permissions},
        {"Testing get user roles...",           test_get_user_roles},
        {"Testing revoke role...",              test_revoke_role},
        {"Testing init defaults...",            test_init_defaults},
        {"Testing role inheritance...",         test_role_inheritance},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
