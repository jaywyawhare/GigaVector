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

GV_ConsistencyManager *consistency_create(GV_ConsistencyLevel default_level);
/**
 * @brief Destroy an instance and free associated resources.
 *
 * @param mgr Manager instance.
 */
void consistency_destroy(GV_ConsistencyManager *mgr);

/**
 * @brief Set a value.
 *
 * @param mgr Manager instance.
 * @param level level.
 * @return 0 on success, -1 on error.
 */
int consistency_set_default(GV_ConsistencyManager *mgr, GV_ConsistencyLevel level);
/**
 * @brief Get a value.
 *
 * @param mgr Manager instance.
 * @return Result value.
 */
GV_ConsistencyLevel consistency_get_default(const GV_ConsistencyManager *mgr);

int consistency_check(const GV_ConsistencyManager *mgr, const GV_ConsistencyConfig *config,
                          uint64_t replica_lag_ms, uint64_t replica_position);

/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @return Value.
 */
uint64_t consistency_new_session(GV_ConsistencyManager *mgr);
/**
 * @brief Perform the operation.
 *
 * @param mgr Manager instance.
 * @param session_token session_token.
 * @param write_position write_position.
 * @return 0 on success, -1 on error.
 */
int consistency_update_session(GV_ConsistencyManager *mgr, uint64_t session_token, uint64_t write_position);
/**
 * @brief Get a value.
 *
 * @param mgr Manager instance.
 * @param session_token session_token.
 * @return Value.
 */
uint64_t consistency_get_session_position(const GV_ConsistencyManager *mgr, uint64_t session_token);

/**
 * @brief Initialize a configuration structure with default values.
 *
 * @param config Configuration to apply/output.
 */
void consistency_config_init(GV_ConsistencyConfig *config);
/**
 * @brief Perform the operation.
 *
 * @return Result value.
 */
GV_ConsistencyConfig consistency_strong(void);
/**
 * @brief Perform the operation.
 *
 * @return Result value.
 */
GV_ConsistencyConfig consistency_eventual(void);
/**
 * @brief Perform the operation.
 *
 * @param max_staleness_ms max_staleness_ms.
 * @return Result value.
 */
GV_ConsistencyConfig consistency_bounded(uint64_t max_staleness_ms);
/**
 * @brief Perform the operation.
 *
 * @param token token.
 * @return Result value.
 */
GV_ConsistencyConfig consistency_session(uint64_t token);

#ifdef __cplusplus
}
#endif
#endif
