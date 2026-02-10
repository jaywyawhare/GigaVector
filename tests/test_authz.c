#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_authz.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* Helper to create a fake identity for testing */
static GV_Identity make_identity(const char *subject) {
    GV_Identity id;
    memset(&id, 0, sizeof(id));
    id.subject = strdup(subject);
    id.key_id = NULL;
    id.auth_time = 1000;
    id.expires_at = 0;
    id.claims = NULL;
    return id;
}

static void free_identity(GV_Identity *id) {
    free(id->subject);
    id->subject = NULL;
}

static int test_authz_create_destroy(void) {
    GV_AuthzManager *authz = gv_authz_create();
    ASSERT(authz != NULL, "authz manager creation should succeed");

    gv_authz_destroy(authz);

    /* Destroying NULL should be safe */
    gv_authz_destroy(NULL);
    return 0;
}

static int test_authz_define_role(void) {
    GV_AuthzManager *authz = gv_authz_create();
    ASSERT(authz != NULL, "authz creation");

    int rc = gv_authz_define_role(authz, "editor",
                                   GV_PERM_READ | GV_PERM_WRITE,
                                   NULL, 0);
    ASSERT(rc == 0, "defining role 'editor' should succeed");

    const char *namespaces[] = {"ns1", "ns2"};
    rc = gv_authz_define_role(authz, "ns_reader",
                               GV_PERM_READ,
                               namespaces, 2);
    ASSERT(rc == 0, "defining role 'ns_reader' with namespaces should succeed");

    gv_authz_destroy(authz);
    return 0;
}

static int test_authz_get_role(void) {
    GV_AuthzManager *authz = gv_authz_create();
    ASSERT(authz != NULL, "authz creation");

    gv_authz_define_role(authz, "viewer", GV_PERM_READ, NULL, 0);

    GV_Role role;
    memset(&role, 0, sizeof(role));
    int rc = gv_authz_get_role(authz, "viewer", &role);
    ASSERT(rc == 0, "getting role 'viewer' should succeed");
    ASSERT(role.name != NULL && strcmp(role.name, "viewer") == 0, "role name should be 'viewer'");
    ASSERT(role.permissions == GV_PERM_READ, "role permissions should be READ");

    gv_authz_free_role(&role);
    gv_authz_destroy(authz);
    return 0;
}

static int test_authz_assign_and_check(void) {
    GV_AuthzManager *authz = gv_authz_create();
    ASSERT(authz != NULL, "authz creation");

    gv_authz_define_role(authz, "writer", GV_PERM_READ | GV_PERM_WRITE, NULL, 0);

    int rc = gv_authz_assign_role(authz, "user_alice", "writer");
    ASSERT(rc == 0, "assigning role to user should succeed");

    GV_Identity alice = make_identity("user_alice");

    ASSERT(gv_authz_can_read(authz, &alice, NULL) == 1, "alice should be able to read");
    ASSERT(gv_authz_can_write(authz, &alice, NULL) == 1, "alice should be able to write");
    ASSERT(gv_authz_can_delete(authz, &alice, NULL) == 0, "alice should NOT be able to delete");

    free_identity(&alice);
    gv_authz_destroy(authz);
    return 0;
}

static int test_authz_admin_check(void) {
    GV_AuthzManager *authz = gv_authz_create();
    ASSERT(authz != NULL, "authz creation");

    gv_authz_define_role(authz, "superadmin", GV_PERM_ALL, NULL, 0);
    gv_authz_define_role(authz, "reader", GV_PERM_READ, NULL, 0);

    gv_authz_assign_role(authz, "admin_bob", "superadmin");
    gv_authz_assign_role(authz, "user_carol", "reader");

    GV_Identity bob = make_identity("admin_bob");
    GV_Identity carol = make_identity("user_carol");

    ASSERT(gv_authz_is_admin(authz, &bob) == 1, "bob should be admin");
    ASSERT(gv_authz_is_admin(authz, &carol) == 0, "carol should NOT be admin");

    free_identity(&bob);
    free_identity(&carol);
    gv_authz_destroy(authz);
    return 0;
}

static int test_authz_revoke_role(void) {
    GV_AuthzManager *authz = gv_authz_create();
    ASSERT(authz != NULL, "authz creation");

    gv_authz_define_role(authz, "editor", GV_PERM_READ | GV_PERM_WRITE, NULL, 0);
    gv_authz_assign_role(authz, "user_dave", "editor");

    GV_Identity dave = make_identity("user_dave");
    ASSERT(gv_authz_can_write(authz, &dave, NULL) == 1, "dave should be able to write before revoke");

    int rc = gv_authz_revoke_role(authz, "user_dave", "editor");
    ASSERT(rc == 0, "revoking role should succeed");

    ASSERT(gv_authz_can_write(authz, &dave, NULL) == 0, "dave should NOT be able to write after revoke");

    free_identity(&dave);
    gv_authz_destroy(authz);
    return 0;
}

static int test_authz_builtin_roles(void) {
    GV_AuthzManager *authz = gv_authz_create();
    ASSERT(authz != NULL, "authz creation");

    int rc = gv_authz_init_builtin_roles(authz);
    ASSERT(rc == 0, "initializing builtin roles should succeed");

    GV_Role *roles = NULL;
    size_t count = 0;
    rc = gv_authz_list_roles(authz, &roles, &count);
    ASSERT(rc == 0, "listing roles should succeed");
    ASSERT(count >= 3, "should have at least 3 builtin roles (admin, writer, reader)");

    gv_authz_free_roles(roles, count);
    gv_authz_destroy(authz);
    return 0;
}

static int test_authz_permission_string(void) {
    const char *s = gv_authz_permission_string(GV_PERM_READ);
    ASSERT(s != NULL && strlen(s) > 0, "READ permission string should be non-empty");

    s = gv_authz_permission_string(GV_PERM_WRITE);
    ASSERT(s != NULL && strlen(s) > 0, "WRITE permission string should be non-empty");

    s = gv_authz_permission_string(GV_PERM_DELETE);
    ASSERT(s != NULL && strlen(s) > 0, "DELETE permission string should be non-empty");

    s = gv_authz_permission_string(GV_PERM_ADMIN);
    ASSERT(s != NULL && strlen(s) > 0, "ADMIN permission string should be non-empty");

    s = gv_authz_permission_string(GV_PERM_NONE);
    ASSERT(s != NULL, "NONE permission string should be non-NULL");

    return 0;
}

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing authz create/destroy...", test_authz_create_destroy},
        {"Testing authz define role...", test_authz_define_role},
        {"Testing authz get role...", test_authz_get_role},
        {"Testing authz assign and check...", test_authz_assign_and_check},
        {"Testing authz admin check...", test_authz_admin_check},
        {"Testing authz revoke role...", test_authz_revoke_role},
        {"Testing authz builtin roles...", test_authz_builtin_roles},
        {"Testing authz permission string...", test_authz_permission_string},
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
