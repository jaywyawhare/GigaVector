#ifndef GIGAVECTOR_RAFT_H
#define GIGAVECTOR_RAFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file raft.h
 * @brief A real Raft consensus core (leader election + log replication).
 *
 * This is a self-contained, transport-agnostic implementation of the Raft
 * algorithm (Ongaro & Ousterhout). It implements the parts the older simulated
 * replication lacked: per-entry {term,index} logs, RequestVote / AppendEntries
 * RPCs with the log-consistency check, commit advancement on a quorum of
 * matchIndex, randomised election timeouts, and the leader/candidate/follower
 * state machine with the Raft safety rules.
 *
 * The core is driven by two entry points — raft_tick() (clock) and raft_step()
 * (an incoming message) — and emits outbound messages through a caller-supplied
 * send callback. That keeps it independent of any particular transport, so it
 * can run over TCP or inside a deterministic in-memory simulation for testing.
 *
 * Log indices are 1-based; index 0 is the empty sentinel.
 */

typedef enum {
    GV_RAFT_FOLLOWER  = 0,
    GV_RAFT_CANDIDATE = 1,
    GV_RAFT_LEADER    = 2
} GV_RaftRole;

typedef enum {
    GV_RAFT_MSG_REQUEST_VOTE       = 1,
    GV_RAFT_MSG_REQUEST_VOTE_RESP  = 2,
    GV_RAFT_MSG_APPEND_ENTRIES      = 3,
    GV_RAFT_MSG_APPEND_ENTRIES_RESP = 4
} GV_RaftMsgType;

/** One replicated log entry. */
typedef struct {
    uint64_t term;   /**< Term in which the entry was created by a leader. */
    void    *data;   /**< Command payload (owned by the log). */
    size_t   len;    /**< Payload length. */
} GV_RaftEntry;

/**
 * A Raft RPC message. AppendEntries carries a (possibly empty) run of entries;
 * the pointer is borrowed for the duration of the send/step call — a transport
 * that queues the message must deep-copy the entries.
 */
typedef struct {
    GV_RaftMsgType type;
    uint64_t term;            /**< Sender's currentTerm. */
    int      from;            /**< Sender node id. */

    /* RequestVote */
    uint64_t last_log_index;  /**< Candidate's last log index. */
    uint64_t last_log_term;   /**< Candidate's last log term. */

    /* RequestVoteResp */
    int      vote_granted;

    /* AppendEntries */
    uint64_t prev_log_index;
    uint64_t prev_log_term;
    uint64_t leader_commit;
    const GV_RaftEntry *entries;
    size_t   n_entries;

    /* AppendEntriesResp */
    int      success;
    uint64_t match_index;     /**< On success: follower's last replicated index. */
} GV_RaftMsg;

/**
 * Callbacks into the host environment.
 * - send:    deliver @p msg to node @p to (transport owns copying if it queues).
 * - apply:   a committed entry at @p index is ready to apply to the state machine.
 * - persist: currentTerm/votedFor changed and should be made durable (may be NULL).
 */
typedef struct {
    void (*send)(void *ctx, int to, const GV_RaftMsg *msg);
    void (*apply)(void *ctx, uint64_t index, const void *data, size_t len);
    void (*persist)(void *ctx, uint64_t current_term, int voted_for);
    void *ctx;
} GV_RaftCallbacks;

typedef struct {
    uint32_t election_timeout_min_ms; /**< Lower bound of the randomised election timeout. */
    uint32_t election_timeout_max_ms; /**< Upper bound (exclusive). */
    uint32_t heartbeat_ms;            /**< Leader heartbeat interval. */
} GV_RaftConfig;

typedef struct GV_Raft GV_Raft;

/** Fill @p cfg with sensible defaults (150-300ms election, 50ms heartbeat). */
void raft_config_init(GV_RaftConfig *cfg);

/**
 * @brief Create a Raft node.
 * @param id This node's id.
 * @param peers Array of the OTHER nodes' ids.
 * @param n_peers Number of peers (cluster size = n_peers + 1).
 * @param cfg Timing config (NULL for defaults).
 * @param cb Host callbacks (send and apply required; persist optional).
 * @return Node handle, or NULL on error.
 */
GV_Raft *raft_create(int id, const int *peers, size_t n_peers,
                     const GV_RaftConfig *cfg, const GV_RaftCallbacks *cb);

/** @brief Advance the internal clock by @p ms; may trigger elections/heartbeats. */
void raft_tick(GV_Raft *r, uint32_t ms);

/** @brief Process an incoming message from node @p from. */
void raft_step(GV_Raft *r, int from, const GV_RaftMsg *msg);

/**
 * @brief Leader-only: append a client command to the log for replication.
 * @param index_out Output: the assigned 1-based log index (may be NULL).
 * @return 0 if accepted (this node is leader), -1 otherwise.
 */
int raft_submit(GV_Raft *r, const void *data, size_t len, uint64_t *index_out);

GV_RaftRole raft_role(const GV_Raft *r);
uint64_t    raft_current_term(const GV_Raft *r);
int         raft_leader_id(const GV_Raft *r);        /**< Believed leader, or -1. */
uint64_t    raft_commit_index(const GV_Raft *r);
uint64_t    raft_last_log_index(const GV_Raft *r);
int         raft_voted_for(const GV_Raft *r);

/** @brief Read the term of the log entry at 1-based @p index (0 if out of range). */
uint64_t    raft_log_term_at(const GV_Raft *r, uint64_t index);

/** @brief Destroy the node and free its log. Safe with NULL. */
void raft_destroy(GV_Raft *r);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_RAFT_H */
