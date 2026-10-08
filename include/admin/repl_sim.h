#ifndef GIGAVECTOR_GV_REPL_SIM_H
#define GIGAVECTOR_GV_REPL_SIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_ReplSim GV_ReplSim;

typedef struct {
    /** Drop probability in permille (0-1000). */
    uint32_t drop_permille;
    /** When set, no messages are delivered until heal. */
    int partitioned;
    /** Deliver queued messages in random order. */
    int reorder;
} GV_ReplSimFaultConfig;

GV_ReplSim *repl_sim_create(uint64_t rng_seed);
void repl_sim_destroy(GV_ReplSim *sim);

void repl_sim_set_faults(GV_ReplSim *sim, const GV_ReplSimFaultConfig *faults);
void repl_sim_heal(GV_ReplSim *sim);

/**
 * Enqueue a WAL record for a follower. May drop immediately per fault config.
 * @return 0 if queued, -1 if dropped or error.
 */
int repl_sim_enqueue_wal(GV_ReplSim *sim, const char *follower_id,
                         uint64_t entry_index, const uint8_t *record, size_t record_len);

/**
 * Deliver one pending WAL message to the follower (respecting partition/drop/reorder).
 * Caller must free @p *record.
 * @return 0 on delivery, -1 if none available.
 */
int repl_sim_deliver_wal(GV_ReplSim *sim, const char *follower_id,
                         uint64_t *entry_index, uint8_t **record, size_t *record_len);

/** Deliver all pending messages for a follower (used after heal). */
int repl_sim_flush_follower(GV_ReplSim *sim, const char *follower_id);

size_t repl_sim_pending_count(GV_ReplSim *sim, const char *follower_id);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_REPL_SIM_H */
