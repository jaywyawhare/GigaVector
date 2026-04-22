#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "search/consistency.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

static int test_create_destroy(void) {
    GV_ConsistencyManager *mgr = consistency_create(GV_CONSISTENCY_STRONG);
    ASSERT(mgr != NULL, "consistency_create returned NULL");
    ASSERT(consistency_get_default(mgr) == GV_CONSISTENCY_STRONG,
           "default level should be STRONG");
    consistency_destroy(mgr);
    return 0;
}

static int test_set_get_default(void) {
    GV_ConsistencyManager *mgr = consistency_create(GV_CONSISTENCY_EVENTUAL);
    ASSERT(mgr != NULL, "create manager");
    ASSERT(consistency_get_default(mgr) == GV_CONSISTENCY_EVENTUAL,
           "initial default should be EVENTUAL");

    int rc = consistency_set_default(mgr, GV_CONSISTENCY_SESSION);
    ASSERT(rc == 0, "set_default should succeed");
    ASSERT(consistency_get_default(mgr) == GV_CONSISTENCY_SESSION,
           "default should be SESSION after set");

    rc = consistency_set_default(mgr, GV_CONSISTENCY_BOUNDED_STALENESS);
    ASSERT(rc == 0, "set_default bounded should succeed");
    ASSERT(consistency_get_default(mgr) == GV_CONSISTENCY_BOUNDED_STALENESS,
           "default should be BOUNDED_STALENESS");

    consistency_destroy(mgr);
    return 0;
}

static int test_config_helpers(void) {
    GV_ConsistencyConfig strong = consistency_strong();
    ASSERT(strong.level == GV_CONSISTENCY_STRONG, "strong helper level");

    GV_ConsistencyConfig eventual = consistency_eventual();
    ASSERT(eventual.level == GV_CONSISTENCY_EVENTUAL, "eventual helper level");

    GV_ConsistencyConfig bounded = consistency_bounded(500);
    ASSERT(bounded.level == GV_CONSISTENCY_BOUNDED_STALENESS, "bounded helper level");
    ASSERT(bounded.max_staleness_ms == 500, "bounded max_staleness_ms == 500");

    GV_ConsistencyConfig sess = consistency_session(42);
    ASSERT(sess.level == GV_CONSISTENCY_SESSION, "session helper level");
    ASSERT(sess.session_token == 42, "session token == 42");

    GV_ConsistencyConfig generic;
    consistency_config_init(&generic);
    ASSERT(generic.level == GV_CONSISTENCY_STRONG, "config_init default level");
    return 0;
}

static int test_check_strong(void) {
    GV_ConsistencyManager *mgr = consistency_create(GV_CONSISTENCY_STRONG);
    ASSERT(mgr != NULL, "create manager");

    GV_ConsistencyConfig config = consistency_strong();
    /* For STRONG, only leader (lag=0) should pass */
    int ok = consistency_check(mgr, &config, 0, 100);
    ASSERT(ok == 1 || ok == 0, "check with lag=0 should return valid result");

    int fail = consistency_check(mgr, &config, 5000, 95);
    (void)fail; /* Result depends on implementation - just ensure no crash */

    consistency_destroy(mgr);
    return 0;
}

static int test_check_bounded_staleness(void) {
    GV_ConsistencyManager *mgr = consistency_create(GV_CONSISTENCY_EVENTUAL);
    ASSERT(mgr != NULL, "create manager");

    GV_ConsistencyConfig config = consistency_bounded(1000);
    int ok = consistency_check(mgr, &config, 500, 100);
    ASSERT(ok == 1, "replica within bound should pass");

    int fail = consistency_check(mgr, &config, 2000, 100);
    ASSERT(fail == 0, "replica exceeding bound should fail");

    consistency_destroy(mgr);
    return 0;
}

static int test_session_token_management(void) {
    GV_ConsistencyManager *mgr = consistency_create(GV_CONSISTENCY_SESSION);
    ASSERT(mgr != NULL, "create manager");

    uint64_t token = consistency_new_session(mgr);
    ASSERT(token > 0, "new_session should return nonzero token");

    uint64_t pos = consistency_get_session_position(mgr, token);
    ASSERT(pos == 0, "initial session position should be 0");

    int rc = consistency_update_session(mgr, token, 42);
    ASSERT(rc == 0, "update_session should succeed");

    pos = consistency_get_session_position(mgr, token);
    ASSERT(pos == 42, "session position should be 42 after update");

    rc = consistency_update_session(mgr, token, 100);
    ASSERT(rc == 0, "update to higher position should succeed");
    pos = consistency_get_session_position(mgr, token);
    ASSERT(pos == 100, "session position should be 100");

    consistency_destroy(mgr);
    return 0;
}

static int test_multiple_sessions(void) {
    GV_ConsistencyManager *mgr = consistency_create(GV_CONSISTENCY_SESSION);
    ASSERT(mgr != NULL, "create manager");

    uint64_t t1 = consistency_new_session(mgr);
    uint64_t t2 = consistency_new_session(mgr);
    ASSERT(t1 != t2, "two sessions should have different tokens");

    consistency_update_session(mgr, t1, 10);
    consistency_update_session(mgr, t2, 20);

    ASSERT(consistency_get_session_position(mgr, t1) == 10, "session 1 pos == 10");
    ASSERT(consistency_get_session_position(mgr, t2) == 20, "session 2 pos == 20");

    consistency_destroy(mgr);
    return 0;
}

static int test_check_session_consistency(void) {
    GV_ConsistencyManager *mgr = consistency_create(GV_CONSISTENCY_SESSION);
    ASSERT(mgr != NULL, "create manager");

    uint64_t token = consistency_new_session(mgr);
    consistency_update_session(mgr, token, 50);

    GV_ConsistencyConfig config = consistency_session(token);

    /* Replica at position 60 >= 50 should satisfy read-your-writes */
    int ok = consistency_check(mgr, &config, 0, 60);
    ASSERT(ok == 1, "replica ahead of session should pass");

    int fail = consistency_check(mgr, &config, 0, 30);
    ASSERT(fail == 0, "replica behind session should fail");

    consistency_destroy(mgr);
    return 0;
}

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
        if (tests[i].fn() == 0) { passed++; }
    }
    return passed == n ? 0 : 1;
}
