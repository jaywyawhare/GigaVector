/**
 * @file repl_sim.c
 * @brief In-memory replication transport with fault injection for DST.
 */

#include "admin/repl_sim.h"
#include "core/memory.h"
#include "admin/repl_transport.h"

#include <stdlib.h>
#include <string.h>

#define REPL_SIM_MAX_MSGS 4096

typedef struct {
    char *follower_id;
    uint64_t entry_index;
    uint8_t *record;
    size_t record_len;
} ReplSimMsg;

struct GV_ReplSim {
    uint64_t rng_state;
    GV_ReplSimFaultConfig faults;
    ReplSimMsg msgs[REPL_SIM_MAX_MSGS];
    size_t msg_count;
};

static uint64_t repl_sim_rng_next(GV_ReplSim *sim) {
    uint64_t x = sim->rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    sim->rng_state = x;
    return x * 2685821657736338717ULL;
}

static int repl_sim_should_drop_enqueue(GV_ReplSim *sim) {
    if (sim->faults.drop_permille == 0) return 0;
    return (repl_sim_rng_next(sim) % 1000U) < sim->faults.drop_permille;
}

static void repl_sim_free_msg(ReplSimMsg *msg) {
    if (!msg) return;
    gv_free(msg->follower_id);
    gv_free(msg->record);
    memset(msg, 0, sizeof(*msg));
}

GV_ReplSim *repl_sim_create(uint64_t rng_seed) {
    GV_ReplSim *sim = gv_calloc(1, sizeof(*sim));
    if (!sim) return NULL;
    sim->rng_state = rng_seed ? rng_seed : 0x475652454c4cULL;
    return sim;
}

void repl_sim_destroy(GV_ReplSim *sim) {
    if (!sim) return;
    for (size_t i = 0; i < sim->msg_count; i++) {
        repl_sim_free_msg(&sim->msgs[i]);
    }
    gv_free(sim);
}

void repl_sim_set_faults(GV_ReplSim *sim, const GV_ReplSimFaultConfig *faults) {
    if (!sim || !faults) return;
    sim->faults = *faults;
}

void repl_sim_heal(GV_ReplSim *sim) {
    if (!sim) return;
    sim->faults.partitioned = 0;
}

static ReplSimMsg *repl_sim_find_msg(GV_ReplSim *sim, const char *follower_id, size_t *idx_out) {
    if (sim->faults.reorder && sim->msg_count > 0) {
        size_t start = (size_t)(repl_sim_rng_next(sim) % sim->msg_count);
        for (size_t off = 0; off < sim->msg_count; off++) {
            size_t i = (start + off) % sim->msg_count;
            if (sim->msgs[i].follower_id &&
                strcmp(sim->msgs[i].follower_id, follower_id) == 0) {
                if (idx_out) *idx_out = i;
                return &sim->msgs[i];
            }
        }
        return NULL;
    }

    for (size_t i = 0; i < sim->msg_count; i++) {
        if (sim->msgs[i].follower_id &&
            strcmp(sim->msgs[i].follower_id, follower_id) == 0) {
            if (idx_out) *idx_out = i;
            return &sim->msgs[i];
        }
    }
    return NULL;
}

int repl_sim_enqueue_wal(GV_ReplSim *sim, const char *follower_id,
                         uint64_t entry_index, const uint8_t *record, size_t record_len) {
    if (!sim || !follower_id || !record || record_len == 0) return -1;
    if (sim->msg_count >= REPL_SIM_MAX_MSGS) return -1;
    if (repl_sim_should_drop_enqueue(sim)) return -1;

    ReplSimMsg *msg = &sim->msgs[sim->msg_count];
    msg->follower_id = gv_strdup(follower_id);
    msg->entry_index = entry_index;
    msg->record = (uint8_t *)gv_alloc(record_len);
    if (!msg->follower_id || !msg->record) {
        repl_sim_free_msg(msg);
        return -1;
    }
    memcpy(msg->record, record, record_len);
    msg->record_len = record_len;
    sim->msg_count++;
    return 0;
}

int repl_sim_deliver_wal(GV_ReplSim *sim, const char *follower_id,
                         uint64_t *entry_index, uint8_t **record, size_t *record_len) {
    if (!sim || !follower_id || !entry_index || !record || !record_len) return -1;
    if (sim->faults.partitioned) return -1;

    size_t idx = 0;
    ReplSimMsg *msg = repl_sim_find_msg(sim, follower_id, &idx);
    if (!msg) return -1;

    if (sim->faults.drop_permille > 0 &&
        (repl_sim_rng_next(sim) % 1000U) < sim->faults.drop_permille) {
        repl_sim_free_msg(msg);
        if (idx < sim->msg_count - 1) {
            sim->msgs[idx] = sim->msgs[sim->msg_count - 1];
        }
        sim->msg_count--;
        return -1;
    }

    *entry_index = msg->entry_index;
    *record = msg->record;
    *record_len = msg->record_len;
    msg->record = NULL;
    gv_free(msg->follower_id);
    memset(msg, 0, sizeof(*msg));
    if (idx < sim->msg_count - 1) {
        sim->msgs[idx] = sim->msgs[sim->msg_count - 1];
    }
    sim->msg_count--;
    return 0;
}

int repl_sim_flush_follower(GV_ReplSim *sim, const char *follower_id) {
    if (!sim || !follower_id) return -1;
    int delivered = 0;
    while (repl_sim_pending_count(sim, follower_id) > 0) {
        uint64_t entry_index = 0;
        uint8_t *record = NULL;
        size_t record_len = 0;
        if (repl_sim_deliver_wal(sim, follower_id, &entry_index, &record, &record_len) != 0) {
            break;
        }
        gv_free(record);
        delivered++;
    }
    return delivered;
}

size_t repl_sim_pending_count(GV_ReplSim *sim, const char *follower_id) {
    if (!sim || !follower_id) return 0;
    size_t n = 0;
    for (size_t i = 0; i < sim->msg_count; i++) {
        if (sim->msgs[i].follower_id &&
            strcmp(sim->msgs[i].follower_id, follower_id) == 0) {
            n++;
        }
    }
    return n;
}

static int repl_sim_filter_outbound(void *ctx, uint8_t msg_type,
                                    const uint8_t *payload, size_t payload_len) {
    GV_ReplSim *sim = (GV_ReplSim *)ctx;
    if (!sim || msg_type != 2 /* REPL_MSG_WAL */) return 0;
    (void)payload;
    (void)payload_len;
    if (sim->faults.partitioned) return -1;
    if (sim->faults.drop_permille > 0 &&
        (repl_sim_rng_next(sim) % 1000U) < sim->faults.drop_permille) {
        return -1;
    }
    return 0;
}

static int repl_sim_filter_inbound(void *ctx, uint8_t msg_type,
                                   const uint8_t *payload, size_t payload_len) {
    GV_ReplSim *sim = (GV_ReplSim *)ctx;
    if (!sim || msg_type != 2) return 0;
    (void)payload;
    (void)payload_len;
    if (sim->faults.partitioned) return -1;
    if (sim->faults.drop_permille > 0 &&
        (repl_sim_rng_next(sim) % 1000U) < sim->faults.drop_permille) {
        return -1;
    }
    return 0;
}

GV_ReplTransportHooks repl_sim_transport_hooks(GV_ReplSim *sim) {
    GV_ReplTransportHooks hooks;
    hooks.ctx = sim;
    hooks.filter_outbound = repl_sim_filter_outbound;
    hooks.filter_inbound = repl_sim_filter_inbound;
    return hooks;
}
