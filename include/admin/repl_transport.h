#ifndef GIGAVECTOR_GV_REPL_TRANSPORT_H
#define GIGAVECTOR_GV_REPL_TRANSPORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GV_ReplicationManager;
typedef struct GV_ReplicationManager GV_ReplicationManager;

struct GV_Database;
typedef struct GV_Database GV_Database;

typedef struct GV_ReplTransport GV_ReplTransport;

typedef enum {
    REPL_MSG_HELLO = 1,
    REPL_MSG_WAL = 2,
    REPL_MSG_ACK = 3,
    REPL_MSG_HEARTBEAT = 4,
    REPL_MSG_CATCHUP = 5,
    REPL_MSG_RESPONSE = 128
} GV_ReplMsgType;

GV_ReplTransport *repl_transport_create(GV_ReplicationManager *mgr);
void repl_transport_destroy(GV_ReplTransport *transport);

int repl_transport_start(GV_ReplTransport *transport);
int repl_transport_stop(GV_ReplTransport *transport);

/** Push the WAL record at @p entry_index (0-based) to connected remote followers. */
int repl_transport_broadcast_entry(GV_ReplTransport *transport, GV_Database *db,
                                   uint64_t entry_index);

/**
 * Optional fault-injection hooks for TCP transport (DST / repl_sim-style testing).
 * filter_outbound/filter_inbound return 0 to deliver, -1 to drop.
 */
typedef struct {
    void *ctx;
    int (*filter_outbound)(void *ctx, uint8_t msg_type, const uint8_t *payload, size_t payload_len);
    int (*filter_inbound)(void *ctx, uint8_t msg_type, const uint8_t *payload, size_t payload_len);
} GV_ReplTransportHooks;

void repl_transport_set_hooks(GV_ReplTransport *transport, const GV_ReplTransportHooks *hooks);
void repl_transport_clear_hooks(GV_ReplTransport *transport);

struct GV_ReplSim;
GV_ReplTransportHooks repl_sim_transport_hooks(struct GV_ReplSim *sim);

/**
 * Parse one replication wire frame from a buffer (same layout as TCP messages).
 * On success, @p *payload is heap-allocated; caller must gv_free().
 */
int repl_parse_frame_buffer(const uint8_t *data, size_t len, size_t max_bytes,
                            uint8_t *msg_type, uint32_t *request_id,
                            uint8_t **payload, size_t *payload_len);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_GV_REPL_TRANSPORT_H */
