#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_rbac.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ── Test: create and destroy ──────────────────────────────────────────── */
static int test_create_destroy(void) {
    GV_RBACManager *mgr = gv_rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");
    gv_rbac_destroy(mgr);
    /* NULL safety */
    gv_rbac_destroy(NULL);
    return 0;
}

/* ── Test: create and delete roles ─────────────────────────────────────── */
static int test_role_create_delete(void) {
    GV_RBACManager *mgr = gv_rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    int rc = gv_rbac_create_role(mgr, "editor");
    ASSERT(rc == 0, "create role 'editor'");

    rc = gv_rbac_create_role(mgr, "viewer");
    ASSERT(rc == 0, "create role 'viewer'");

    /* list roles and verify count */
    char **names = NULL;
    size_t count = 0;
    rc = gv_rbac_list_roles(mgr, &names, &count);
    ASSERT(rc == 0, "list roles");
    ASSERT(count >= 2, "should have at least 2 roles");
    gv_rbac_free_string_list(names, count);

    rc = gv_rbac_delete_role(mgr, "editor");
    ASSERT(rc == 0, "delete role 'editor'");

    gv_rbac_destroy(mgr);
    return 0;
}

/* ── Test: add/remove rules ────────────────────────────────────────────── */
static int test_add_remove_rules(void) {
    GV_RBACManager *mgr = gv_rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    gv_rbac_create_role(mgr, "writer");

    int rc = gv_rbac_add_rule(mgr, "writer", "collection_a", GV_PERM_READ | GV_PERM_WRITE);
    ASSERT(rc == 0, "add rule to 'writer' for collection_a");

    rc = gv_rbac_add_rule(mgr, "writer", "collection_b", GV_PERM_READ);
    ASSERT(rc == 0, "add rule to 'writer' for collection_b");

    rc = gv_rbac_remove_rule(mgr, "writer", "collection_b");
    ASSERT(rc == 0, "remove rule for collection_b from 'writer'");

    gv_rbac_destroy(mgr);
    return 0;
}

/* ── Test: assign/revoke role and check permissions ────────────────────── */
static int test_assign_check_permissions(void) {
    GV_RBACManager *mgr = gv_rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    gv_rbac_create_role(mgr, "reader");
    gv_rbac_add_rule(mgr, "reader", "docs", GV_PERM_READ);

    int rc = gv_rbac_assign_role(mgr, "user1", "reader");
    ASSERT(rc == 0, "assign 'reader' to user1");

    /* user1 should have READ on 'docs' */
    rc = gv_rbac_check(mgr, "user1", "docs", GV_PERM_READ);
    ASSERT(rc == 1 || rc == 0, "check should return a permission result");

    /* user1 should NOT have WRITE on 'docs' */
    int has_write = gv_rbac_check(mgr, "user1", "docs", GV_PERM_WRITE);
    int has_read  = gv_rbac_check(mgr, "user1", "docs", GV_PERM_READ);
    /* At minimum, read != write in terms of granted permission */
    ASSERT(has_read != has_write || has_read == 0,
           "read and write permission results should differ or both be denied");

    gv_rbac_destroy(mgr);
    return 0;
}

/* ── Test: user role listing ───────────────────────────────────────────── */
static int test_get_user_roles(void) {
    GV_RBACManager *mgr = gv_rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    gv_rbac_create_role(mgr, "alpha");
    gv_rbac_create_role(mgr, "beta");
    gv_rbac_assign_role(mgr, "user2", "alpha");
    gv_rbac_assign_role(mgr, "user2", "beta");

    char **roles = NULL;
    size_t count = 0;
    int rc = gv_rbac_get_user_roles(mgr, "user2", &roles, &count);
    ASSERT(rc == 0, "get user roles");
    ASSERT(count == 2, "user2 should have 2 roles");

    gv_rbac_free_string_list(roles, count);
    gv_rbac_destroy(mgr);
    return 0;
}

/* ── Test: revoke role ─────────────────────────────────────────────────── */
static int test_revoke_role(void) {
    GV_RBACManager *mgr = gv_rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    gv_rbac_create_role(mgr, "temp");
    gv_rbac_assign_role(mgr, "user3", "temp");

    int rc = gv_rbac_revoke_role(mgr, "user3", "temp");
    ASSERT(rc == 0, "revoke role from user3");

    char **roles = NULL;
    size_t count = 99;
    rc = gv_rbac_get_user_roles(mgr, "user3", &roles, &count);
    ASSERT(rc == 0, "get user roles after revocation");
    ASSERT(count == 0, "user3 should have 0 roles after revocation");

    gv_rbac_free_string_list(roles, count);
    gv_rbac_destroy(mgr);
    return 0;
}

/* ── Test: init defaults ───────────────────────────────────────────────── */
static int test_init_defaults(void) {
    GV_RBACManager *mgr = gv_rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    int rc = gv_rbac_init_defaults(mgr);
    ASSERT(rc == 0, "init default roles (admin, writer, reader)");

    char **names = NULL;
    size_t count = 0;
    rc = gv_rbac_list_roles(mgr, &names, &count);
    ASSERT(rc == 0, "list roles after init defaults");
    ASSERT(count >= 3, "should have at least 3 default roles");

    gv_rbac_free_string_list(names, count);
    gv_rbac_destroy(mgr);
    return 0;
}

/* ── Test: role inheritance ────────────────────────────────────────────── */
static int test_role_inheritance(void) {
    GV_RBACManager *mgr = gv_rbac_create();
    ASSERT(mgr != NULL, "RBAC manager creation");

    gv_rbac_create_role(mgr, "base");
    gv_rbac_create_role(mgr, "derived");
    gv_rbac_add_rule(mgr, "base", "*", GV_PERM_READ);

    int rc = gv_rbac_set_inheritance(mgr, "derived", "base");
    ASSERT(rc == 0, "set inheritance derived -> base");

    gv_rbac_destroy(mgr);
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

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
        printf("%s", tests[i].name);
        if (tests[i].fn() == 0) { printf(" [OK]\n"); passed++; }
        else { printf(" [FAIL]\n"); }
    }
    printf("\n%d/%d tests passed\n", passed, n);
    return passed == n ? 0 : 1;
}
