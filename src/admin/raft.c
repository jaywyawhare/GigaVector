/*
 * raft.c - Raft consensus core (leader election + log replication).
 *
 * Transport-agnostic: driven by raft_tick() and raft_step(), emits messages via
 * the send callback. Implements the Raft safety rules - up-to-date vote check,
 * AppendEntries log-consistency check with conflict truncation, and commit
 * advancement restricted to current-term entries replicated on a quorum.
 *
 * See include/admin/raft.h for the API contract.
 */

#include <stdlib.h>
#include <string.h>

#include "admin/raft.h"

struct GV_Raft {
    int      id;
    int     *peers;
    size_t   n_peers;
    size_t   majority;      /* votes needed = floor(cluster/2) + 1 */

    /* Persistent state: current_term/voted_for are flushed via cb.persist, and
     * each log entry via cb.persist_log (truncations via cb.truncate_log), so a
     * durable host can reload it all with raft_restore() after a restart. */
    uint64_t current_term;
    int      voted_for;     /* -1 = none */
    GV_RaftEntry *log;      /* 0-based; array[k] == Raft index k+1 */
    size_t   log_len;
    size_t   log_cap;

    /* Volatile state. */
    GV_RaftRole role;
    int      leader_id;
    uint64_t commit_index;
    uint64_t last_applied;

    /* Leader state (parallel to peers[]). */
    uint64_t *next_index;
    uint64_t *match_index;

    /* Candidate state. `votes_granted` counts DISTINCT voters: `voted_by[slot]`
     * records whether peers[slot] has already granted a vote in the current
     * term, so a duplicated or retried RequestVoteResp cannot be counted twice.
     * Counting responses instead of voters let a single peer's vote be tallied
     * repeatedly, electing a leader without a real majority. */
    size_t   votes_granted;
    uint8_t *voted_by;      /* parallel to peers[]; 1 = granted this term */

    /* Timers (ms). */
    uint32_t election_elapsed;
    uint32_t election_timeout;
    uint32_t heartbeat_elapsed;
    uint32_t rng;

    GV_RaftConfig    cfg;
    GV_RaftCallbacks cb;
};

static uint32_t raft_rand(GV_Raft *r) {
    r->rng = r->rng * 1103515245u + 12345u;
    return (r->rng >> 16) & 0x7FFF;
}

static uint64_t last_log_index(const GV_Raft *r) { return r->log_len; }
static uint64_t log_term_at(const GV_Raft *r, uint64_t index) {
    if (index == 0 || index > r->log_len) return 0;
    return r->log[index - 1].term;
}
static uint64_t last_log_term(const GV_Raft *r) { return log_term_at(r, r->log_len); }

static int peer_slot(const GV_Raft *r, int id) {
    for (size_t i = 0; i < r->n_peers; i++) if (r->peers[i] == id) return (int)i;
    return -1;
}

static void reset_election_timer(GV_Raft *r) {
    r->election_elapsed = 0;
    uint32_t lo = r->cfg.election_timeout_min_ms;
    uint32_t hi = r->cfg.election_timeout_max_ms;
    uint32_t span = (hi > lo) ? (hi - lo) : 1;
    r->election_timeout = lo + (raft_rand(r) % span);
}

static void persist(GV_Raft *r) {
    if (r->cb.persist) r->cb.persist(r->cb.ctx, r->current_term, r->voted_for);
}

/* Append one entry. When do_persist is set, flush it durably via persist_log
 * BEFORE returning so the entry survives a restart; restore (replaying an
 * already-durable log) passes 0 to avoid re-persisting. */
static int log_append_ex(GV_Raft *r, uint64_t term, const void *data, size_t len,
                         int do_persist) {
    if (r->log_len == r->log_cap) {
        size_t nc = r->log_cap ? r->log_cap * 2 : 16;
        GV_RaftEntry *nl = (GV_RaftEntry *)realloc(r->log, nc * sizeof(GV_RaftEntry));
        if (!nl) return -1;
        r->log = nl; r->log_cap = nc;
    }
    void *copy = NULL;
    if (len) {
        copy = malloc(len);
        if (!copy) return -1;
        memcpy(copy, data, len);
    }
    r->log[r->log_len].term = term;
    r->log[r->log_len].data = copy;
    r->log[r->log_len].len  = len;
    r->log_len++;
    if (do_persist && r->cb.persist_log &&
        r->cb.persist_log(r->cb.ctx, r->log_len, term, copy, len) != 0) {
        /* Durable write failed: roll the entry back so it is never acknowledged
         * or committed (an un-acked entry cannot be lost). */
        r->log_len--;
        free(copy);
        r->log[r->log_len].data = NULL;
        return -1;
    }
    return 0;
}

static int log_append(GV_Raft *r, uint64_t term, const void *data, size_t len) {
    return log_append_ex(r, term, data, len, 1);
}

static int log_truncate(GV_Raft *r, uint64_t keep) { /* drop entries after index `keep` */
    /* Delete durably first; if that fails, leave the in-memory log intact so the
     * two stay consistent and the caller can reject the AppendEntries. */
    if (r->log_len > keep && r->cb.truncate_log &&
        r->cb.truncate_log(r->cb.ctx, keep) != 0)
        return -1;
    while (r->log_len > keep) {
        r->log_len--;
        free(r->log[r->log_len].data);
        r->log[r->log_len].data = NULL;
    }
    return 0;
}

static void apply_committed(GV_Raft *r) {
    while (r->last_applied < r->commit_index) {
        r->last_applied++;
        GV_RaftEntry *e = &r->log[r->last_applied - 1];
        if (r->cb.apply) r->cb.apply(r->cb.ctx, r->last_applied, e->data, e->len);
    }
}

static void become_follower(GV_Raft *r, uint64_t term) {
    r->role = GV_RAFT_FOLLOWER;
    if (term > r->current_term) {
        r->current_term = term;
        r->voted_for = -1;
        persist(r);
    }
}

static void send_request_vote(GV_Raft *r, int to) {
    GV_RaftMsg m; memset(&m, 0, sizeof(m));
    m.type = GV_RAFT_MSG_REQUEST_VOTE;
    m.term = r->current_term;
    m.from = r->id;
    m.last_log_index = last_log_index(r);
    m.last_log_term  = last_log_term(r);
    r->cb.send(r->cb.ctx, to, &m);
}

static void send_append_entries(GV_Raft *r, int slot) {
    uint64_t ni = r->next_index[slot];
    if (ni < 1) ni = 1;
    uint64_t prev_index = ni - 1;
    GV_RaftMsg m; memset(&m, 0, sizeof(m));
    m.type = GV_RAFT_MSG_APPEND_ENTRIES;
    m.term = r->current_term;
    m.from = r->id;
    m.prev_log_index = prev_index;
    m.prev_log_term  = log_term_at(r, prev_index);
    m.leader_commit  = r->commit_index;
    if (ni <= r->log_len) {
        m.entries   = &r->log[ni - 1];
        m.n_entries = r->log_len - (ni - 1);
    } else {
        m.entries = NULL; m.n_entries = 0;
    }
    r->cb.send(r->cb.ctx, r->peers[slot], &m);
}

static void maybe_become_leader(GV_Raft *r) {
    if (r->votes_granted < r->majority) return;
    r->role = GV_RAFT_LEADER;
    r->leader_id = r->id;
    for (size_t i = 0; i < r->n_peers; i++) {
        r->next_index[i]  = last_log_index(r) + 1;
        r->match_index[i] = 0;
    }
    r->heartbeat_elapsed = 0;
    for (size_t i = 0; i < r->n_peers; i++) send_append_entries(r, (int)i);
}

static void become_candidate(GV_Raft *r) {
    r->role = GV_RAFT_CANDIDATE;
    r->current_term++;
    r->voted_for = r->id;
    r->votes_granted = 1;   /* vote for self */
    if (r->voted_by && r->n_peers) memset(r->voted_by, 0, r->n_peers);
    r->leader_id = -1;
    persist(r);
    reset_election_timer(r);
    for (size_t i = 0; i < r->n_peers; i++) send_request_vote(r, r->peers[i]);
    maybe_become_leader(r); /* single-node cluster elects immediately */
}

/* Leader: advance commitIndex to the highest N replicated on a quorum whose
 * entry belongs to the current term (Raft's commit safety restriction). */
static void advance_commit(GV_Raft *r) {
    uint64_t new_commit = r->commit_index;
    for (uint64_t N = r->commit_index + 1; N <= last_log_index(r); N++) {
        if (log_term_at(r, N) != r->current_term) continue;
        size_t count = 1; /* leader replicates everything */
        for (size_t i = 0; i < r->n_peers; i++)
            if (r->match_index[i] >= N) count++;
        if (count >= r->majority) new_commit = N;
    }
    if (new_commit > r->commit_index) {
        r->commit_index = new_commit;
        apply_committed(r);
    }
}

static void handle_request_vote(GV_Raft *r, const GV_RaftMsg *msg) {
    GV_RaftMsg reply; memset(&reply, 0, sizeof(reply));
    reply.type = GV_RAFT_MSG_REQUEST_VOTE_RESP;
    reply.from = r->id;

    if (msg->term < r->current_term) {
        reply.term = r->current_term;
        reply.vote_granted = 0;
        r->cb.send(r->cb.ctx, msg->from, &reply);
        return;
    }
    if (msg->term > r->current_term) become_follower(r, msg->term);

    int up_to_date = (msg->last_log_term > last_log_term(r)) ||
                     (msg->last_log_term == last_log_term(r) &&
                      msg->last_log_index >= last_log_index(r));
    int grant = 0;
    if ((r->voted_for == -1 || r->voted_for == msg->from) && up_to_date) {
        grant = 1;
        r->voted_for = msg->from;
        persist(r);
        reset_election_timer(r);
    }
    reply.term = r->current_term;
    reply.vote_granted = grant;
    r->cb.send(r->cb.ctx, msg->from, &reply);
}

static void handle_request_vote_resp(GV_Raft *r, const GV_RaftMsg *msg) {
    if (msg->term > r->current_term) { become_follower(r, msg->term); return; }
    if (r->role != GV_RAFT_CANDIDATE || msg->term != r->current_term) return;
    if (msg->vote_granted) {
        int slot = peer_slot(r, msg->from);
        if (slot < 0) return;                 /* not a cluster member */
        if (r->voted_by && r->voted_by[slot]) return;  /* already counted this voter */
        if (r->voted_by) r->voted_by[slot] = 1;
        r->votes_granted++;
        maybe_become_leader(r);
    }
}

static void handle_append_entries(GV_Raft *r, const GV_RaftMsg *msg) {
    GV_RaftMsg reply; memset(&reply, 0, sizeof(reply));
    reply.type = GV_RAFT_MSG_APPEND_ENTRIES_RESP;
    reply.from = r->id;

    if (msg->term < r->current_term) {
        reply.term = r->current_term;
        reply.success = 0;
        r->cb.send(r->cb.ctx, msg->from, &reply);
        return;
    }
    /* Valid leader for this term: adopt term, step down to follower, reset timer. */
    become_follower(r, msg->term);
    r->role = GV_RAFT_FOLLOWER;
    r->leader_id = msg->from;
    reset_election_timer(r);

    /* Log-consistency check. */
    if (msg->prev_log_index > last_log_index(r) ||
        log_term_at(r, msg->prev_log_index) != msg->prev_log_term) {
        reply.term = r->current_term;
        reply.success = 0;
        r->cb.send(r->cb.ctx, msg->from, &reply);
        return;
    }

    /* Append / overwrite entries, truncating on the first conflict. On an append
     * (OOM) failure, reply failure at the last consistent index instead of a
     * false success - otherwise the leader would believe we hold an entry we do
     * not, and never resend it. */
    uint64_t idx = msg->prev_log_index;
    int append_ok = 1;
    for (size_t i = 0; i < msg->n_entries && append_ok; i++) {
        idx++;
        const GV_RaftEntry *e = &msg->entries[i];
        if (idx <= last_log_index(r)) {
            if (log_term_at(r, idx) != e->term) {
                if (log_truncate(r, idx - 1) != 0) append_ok = 0;
                else if (log_append(r, e->term, e->data, e->len) != 0) append_ok = 0;
            }
            /* else: already present and matching - skip */
        } else {
            if (log_append(r, e->term, e->data, e->len) != 0) append_ok = 0;
        }
    }

    if (!append_ok) {
        reply.term = r->current_term;
        reply.success = 0;
        r->cb.send(r->cb.ctx, msg->from, &reply);
        return;
    }

    if (msg->leader_commit > r->commit_index) {
        uint64_t last_new = msg->prev_log_index + msg->n_entries;
        uint64_t new_commit = (msg->leader_commit < last_new) ? msg->leader_commit : last_new;
        /* Never move commit_index backward: a reordered/stale AppendEntries can
         * carry a smaller prev_log_index+n_entries while leader_commit is still
         * ahead, which min() would otherwise regress the commit index to. */
        if (new_commit > r->commit_index) {
            r->commit_index = new_commit;
            apply_committed(r);
        }
    }

    reply.term = r->current_term;
    reply.success = 1;
    reply.match_index = msg->prev_log_index + msg->n_entries;
    r->cb.send(r->cb.ctx, msg->from, &reply);
}

static void handle_append_entries_resp(GV_Raft *r, const GV_RaftMsg *msg) {
    if (msg->term > r->current_term) { become_follower(r, msg->term); return; }
    if (r->role != GV_RAFT_LEADER || msg->term != r->current_term) return;
    int slot = peer_slot(r, msg->from);
    if (slot < 0) return;

    if (msg->success) {
        r->match_index[slot] = msg->match_index;
        r->next_index[slot]  = msg->match_index + 1;
        advance_commit(r);
    } else {
        if (r->next_index[slot] > 1) r->next_index[slot]--;
        send_append_entries(r, slot); /* retry with an earlier prefix */
    }
}

void raft_config_init(GV_RaftConfig *cfg) {
    if (!cfg) return;
    cfg->election_timeout_min_ms = 150;
    cfg->election_timeout_max_ms = 300;
    cfg->heartbeat_ms = 50;
}

GV_Raft *raft_create(int id, const int *peers, size_t n_peers,
                     const GV_RaftConfig *cfg, const GV_RaftCallbacks *cb) {
    if (!cb || !cb->send || !cb->apply) return NULL;
    GV_Raft *r = (GV_Raft *)calloc(1, sizeof(GV_Raft));
    if (!r) return NULL;
    r->id = id;
    r->n_peers = n_peers;
    if (n_peers) {
        r->peers       = (int *)malloc(n_peers * sizeof(int));
        r->next_index  = (uint64_t *)calloc(n_peers, sizeof(uint64_t));
        r->match_index = (uint64_t *)calloc(n_peers, sizeof(uint64_t));
        r->voted_by    = (uint8_t *)calloc(n_peers, sizeof(uint8_t));
        if (!r->peers || !r->next_index || !r->match_index || !r->voted_by) {
            free(r->peers); free(r->next_index); free(r->match_index);
            free(r->voted_by); free(r);
            return NULL;
        }
        memcpy(r->peers, peers, n_peers * sizeof(int));
    }
    r->majority = (n_peers + 1) / 2 + 1;
    r->current_term = 0;
    r->voted_for = -1;
    r->role = GV_RAFT_FOLLOWER;
    r->leader_id = -1;
    r->cfg = cfg ? *cfg : (GV_RaftConfig){0};
    if (!cfg) raft_config_init(&r->cfg);
    r->cb = *cb;
    r->rng = (uint32_t)id * 2654435761u + 1u;
    reset_election_timer(r);
    return r;
}

void raft_tick(GV_Raft *r, uint32_t ms) {
    if (!r) return;
    if (r->role == GV_RAFT_LEADER) {
        r->heartbeat_elapsed += ms;
        if (r->heartbeat_elapsed >= r->cfg.heartbeat_ms) {
            r->heartbeat_elapsed = 0;
            for (size_t i = 0; i < r->n_peers; i++) send_append_entries(r, (int)i);
        }
    } else {
        r->election_elapsed += ms;
        if (r->election_elapsed >= r->election_timeout) become_candidate(r);
    }
}

void raft_step(GV_Raft *r, int from, const GV_RaftMsg *msg) {
    if (!r || !msg) return;
    (void)from;
    switch (msg->type) {
        case GV_RAFT_MSG_REQUEST_VOTE:        handle_request_vote(r, msg); break;
        case GV_RAFT_MSG_REQUEST_VOTE_RESP:   handle_request_vote_resp(r, msg); break;
        case GV_RAFT_MSG_APPEND_ENTRIES:      handle_append_entries(r, msg); break;
        case GV_RAFT_MSG_APPEND_ENTRIES_RESP: handle_append_entries_resp(r, msg); break;
        default: break;
    }
}

int raft_submit(GV_Raft *r, const void *data, size_t len, uint64_t *index_out) {
    if (!r || r->role != GV_RAFT_LEADER) return -1;
    if (log_append(r, r->current_term, data, len) != 0) return -1;
    if (index_out) *index_out = last_log_index(r);
    /* Kick replication immediately (also covered by the next heartbeat). */
    for (size_t i = 0; i < r->n_peers; i++) send_append_entries(r, (int)i);
    advance_commit(r); /* single-node clusters commit right away */
    return 0;
}

int raft_restore(GV_Raft *r, uint64_t current_term, int voted_for,
                 uint64_t commit_index, const GV_RaftEntry *entries,
                 size_t n_entries) {
    if (!r) return -1;
    if (n_entries && !entries) return -1;
    r->current_term = current_term;
    r->voted_for = voted_for;
    /* Replay the already-durable log without re-persisting it. */
    for (size_t i = 0; i < n_entries; i++) {
        if (log_append_ex(r, entries[i].term, entries[i].data, entries[i].len, 0) != 0)
            return -1;
    }
    /* Re-apply the committed prefix to the state machine. commit_index is
     * volatile in Raft, so a recovered node would otherwise start with an empty
     * state machine until a leader re-advanced it (which never happens for a
     * single-node cluster). Clamp to the restored log length. */
    if (commit_index > r->log_len) commit_index = r->log_len;
    r->commit_index = commit_index;
    apply_committed(r);
    return 0;
}

GV_RaftRole raft_role(const GV_Raft *r)          { return r ? r->role : GV_RAFT_FOLLOWER; }
uint64_t    raft_current_term(const GV_Raft *r)  { return r ? r->current_term : 0; }
int         raft_leader_id(const GV_Raft *r)     { return r ? r->leader_id : -1; }
uint64_t    raft_commit_index(const GV_Raft *r)  { return r ? r->commit_index : 0; }
uint64_t    raft_last_log_index(const GV_Raft *r){ return r ? r->log_len : 0; }
int         raft_voted_for(const GV_Raft *r)     { return r ? r->voted_for : -1; }
uint64_t    raft_log_term_at(const GV_Raft *r, uint64_t index) { return r ? log_term_at(r, index) : 0; }

void raft_destroy(GV_Raft *r) {
    if (!r) return;
    for (size_t i = 0; i < r->log_len; i++) free(r->log[i].data);
    free(r->log);
    free(r->peers);
    free(r->next_index);
    free(r->match_index);
    free(r->voted_by);
    free(r);
}
