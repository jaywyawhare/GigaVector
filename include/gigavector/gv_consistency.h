#ifndef GIGAVECTOR_GV_CONSISTENCY_H
#define GIGAVECTOR_GV_CONSISTENCY_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_CONSISTENCY_STRONG = 0,           /* Read from leader only */
    GV_CONSISTENCY_EVENTUAL = 1,         /* Read from any replica */
    GV_CONSISTENCY_BOUNDED_STALENESS = 2,/* Read from replica within max lag */
    GV_CONSISTENCY_SESSION = 3           /* Read-your-writes within session */
} GV_ConsistencyLevel;

typedef struct {
    GV_ConsistencyLevel level;
    uint64_t max_staleness_ms;    /* For BOUNDED_STALENESS: max acceptable lag */
    uint64_t session_token;       /* For SESSION: monotonic token */
} GV_ConsistencyConfig;

typedef struct GV_ConsistencyManager GV_ConsistencyManager;

/* Lifecycle */
GV_ConsistencyManager *gv_consistency_create(GV_ConsistencyLevel default_level);
void gv_consistency_destroy(GV_ConsistencyManager *mgr);

/* Set default consistency level */
int gv_consistency_set_default(GV_ConsistencyManager *mgr, GV_ConsistencyLevel level);
GV_ConsistencyLevel gv_consistency_get_default(const GV_ConsistencyManager *mgr);

/* Check if a replica satisfies the consistency requirement */
int gv_consistency_check(const GV_ConsistencyManager *mgr, const GV_ConsistencyConfig *config,
                          uint64_t replica_lag_ms, uint64_t replica_position);

/* Session token management */
uint64_t gv_consistency_new_session(GV_ConsistencyManager *mgr);
int gv_consistency_update_session(GV_ConsistencyManager *mgr, uint64_t session_token, uint64_t write_position);
uint64_t gv_consistency_get_session_position(const GV_ConsistencyManager *mgr, uint64_t session_token);

/* Config helpers */
void gv_consistency_config_init(GV_ConsistencyConfig *config);
GV_ConsistencyConfig gv_consistency_strong(void);
GV_ConsistencyConfig gv_consistency_eventual(void);
GV_ConsistencyConfig gv_consistency_bounded(uint64_t max_staleness_ms);
GV_ConsistencyConfig gv_consistency_session(uint64_t token);

#ifdef __cplusplus
}
#endif
#endif
