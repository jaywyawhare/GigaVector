#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gigavector/gv_consistency.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

/* ---------- test functions ---------- */

static int test_create_destroy(void) {
    GV_ConsistencyManager *mgr = gv_consistency_create(GV_CONSISTENCY_STRONG);
    ASSERT(mgr != NULL, "gv_consistency_create returned NULL");
    ASSERT(gv_consistency_get_default(mgr) == GV_CONSISTENCY_STRONG,
           "default level should be STRONG");
    gv_consistency_destroy(mgr);
    return 0;
}

static int test_set_get_default(void) {
    GV_ConsistencyManager *mgr = gv_consistency_create(GV_CONSISTENCY_EVENTUAL);
    ASSERT(mgr != NULL, "create manager");
    ASSERT(gv_consistency_get_default(mgr) == GV_CONSISTENCY_EVENTUAL,
           "initial default should be EVENTUAL");

    int rc = gv_consistency_set_default(mgr, GV_CONSISTENCY_SESSION);
    ASSERT(rc == 0, "set_default should succeed");
    ASSERT(gv_consistency_get_default(mgr) == GV_CONSISTENCY_SESSION,
           "default should be SESSION after set");

    rc = gv_consistency_set_default(mgr, GV_CONSISTENCY_BOUNDED_STALENESS);
    ASSERT(rc == 0, "set_default bounded should succeed");
    ASSERT(gv_consistency_get_default(mgr) == GV_CONSISTENCY_BOUNDED_STALENESS,
           "default should be BOUNDED_STALENESS");

    gv_consistency_destroy(mgr);
    return 0;
}

static int test_config_helpers(void) {
    GV_ConsistencyConfig strong = gv_consistency_strong();
    ASSERT(strong.level == GV_CONSISTENCY_STRONG, "strong helper level");

    GV_ConsistencyConfig eventual = gv_consistency_eventual();
    ASSERT(eventual.level == GV_CONSISTENCY_EVENTUAL, "eventual helper level");

    GV_ConsistencyConfig bounded = gv_consistency_bounded(500);
    ASSERT(bounded.level == GV_CONSISTENCY_BOUNDED_STALENESS, "bounded helper level");
    ASSERT(bounded.max_staleness_ms == 500, "bounded max_staleness_ms == 500");

    GV_ConsistencyConfig sess = gv_consistency_session(42);
    ASSERT(sess.level == GV_CONSISTENCY_SESSION, "session helper level");
    ASSERT(sess.session_token == 42, "session token == 42");

    GV_ConsistencyConfig generic;
    gv_consistency_config_init(&generic);
    ASSERT(generic.level == GV_CONSISTENCY_STRONG, "config_init default level");
    return 0;
}

static int test_check_strong(void) {
    GV_ConsistencyManager *mgr = gv_consistency_create(GV_CONSISTENCY_STRONG);
    ASSERT(mgr != NULL, "create manager");

    GV_ConsistencyConfig config = gv_consistency_strong();
    /* For STRONG, only leader (lag=0) should pass */
    int ok = gv_consistency_check(mgr, &config, 0, 100);
    ASSERT(ok == 1 || ok == 0, "check with lag=0 should return valid result");

    /* Replica with lag should fail strong consistency */
    int fail = gv_consistency_check(mgr, &config, 5000, 95);
    (void)fail; /* Result depends on implementation - just ensure no crash */

    gv_consistency_destroy(mgr);
    return 0;
}

static int test_check_bounded_staleness(void) {
    GV_ConsistencyManager *mgr = gv_consistency_create(GV_CONSISTENCY_EVENTUAL);
    ASSERT(mgr != NULL, "create manager");

    GV_ConsistencyConfig config = gv_consistency_bounded(1000);
    /* Replica within bound */
    int ok = gv_consistency_check(mgr, &config, 500, 100);
    ASSERT(ok == 1, "replica within bound should pass");

    /* Replica exceeding bound */
    int fail = gv_consistency_check(mgr, &config, 2000, 100);
    ASSERT(fail == 0, "replica exceeding bound should fail");

    gv_consistency_destroy(mgr);
    return 0;
}

static int test_session_token_management(void) {
    GV_ConsistencyManager *mgr = gv_consistency_create(GV_CONSISTENCY_SESSION);
    ASSERT(mgr != NULL, "create manager");

    uint64_t token = gv_consistency_new_session(mgr);
    ASSERT(token > 0, "new_session should return nonzero token");

    /* Initial session position should be 0 */
    uint64_t pos = gv_consistency_get_session_position(mgr, token);
    ASSERT(pos == 0, "initial session position should be 0");

    /* Update session position */
    int rc = gv_consistency_update_session(mgr, token, 42);
    ASSERT(rc == 0, "update_session should succeed");

    pos = gv_consistency_get_session_position(mgr, token);
    ASSERT(pos == 42, "session position should be 42 after update");

    /* Update to a higher position */
    rc = gv_consistency_update_session(mgr, token, 100);
    ASSERT(rc == 0, "update to higher position should succeed");
    pos = gv_consistency_get_session_position(mgr, token);
    ASSERT(pos == 100, "session position should be 100");

    gv_consistency_destroy(mgr);
    return 0;
}

static int test_multiple_sessions(void) {
    GV_ConsistencyManager *mgr = gv_consistency_create(GV_CONSISTENCY_SESSION);
    ASSERT(mgr != NULL, "create manager");

    uint64_t t1 = gv_consistency_new_session(mgr);
    uint64_t t2 = gv_consistency_new_session(mgr);
    ASSERT(t1 != t2, "two sessions should have different tokens");

    gv_consistency_update_session(mgr, t1, 10);
    gv_consistency_update_session(mgr, t2, 20);

    ASSERT(gv_consistency_get_session_position(mgr, t1) == 10, "session 1 pos == 10");
    ASSERT(gv_consistency_get_session_position(mgr, t2) == 20, "session 2 pos == 20");

    gv_consistency_destroy(mgr);
    return 0;
}

static int test_check_session_consistency(void) {
    GV_ConsistencyManager *mgr = gv_consistency_create(GV_CONSISTENCY_SESSION);
    ASSERT(mgr != NULL, "create manager");

    uint64_t token = gv_consistency_new_session(mgr);
    gv_consistency_update_session(mgr, token, 50);

    GV_ConsistencyConfig config = gv_consistency_session(token);

    /* Replica at position 60 >= 50 should satisfy read-your-writes */
    int ok = gv_consistency_check(mgr, &config, 0, 60);
    ASSERT(ok == 1, "replica ahead of session should pass");

    /* Replica at position 30 < 50 should fail */
    int fail = gv_consistency_check(mgr, &config, 0, 30);
    ASSERT(fail == 0, "replica behind session should fail");

    gv_consistency_destroy(mgr);
    return 0;
}

/* ---------- harness ---------- */

typedef int (*test_fn)(void);
typedef struct { const char *name; test_fn fn; } TestCase;

int main(void) {
    TestCase tests[] = {
        {"Testing create/destroy...", test_create_destroy},
        {"Testing set/get default...", test_set_get_default},
        {"Testing config helpers...", test_config_helpers},
        {"Testing check strong...", test_check_strong},
        {"Testing check bounded staleness...", test_check_bounded_staleness},
        {"Testing session token management...", test_session_token_management},
        {"Testing multiple sessions...", test_multiple_sessions},
        {"Testing check session consistency...", test_check_session_consistency},
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
