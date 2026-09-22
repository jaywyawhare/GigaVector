#!/usr/bin/env python3
"""Replay a Quint ITF trace of specs/quint/raft.qnt against the real
src/admin/raft.c, and report any divergence.

This is the bridge described in docs/formal_verification_plan.md §6. Quint's
own `quint-connect` helper is Rust-only, but ITF is language-neutral JSON, so
the driver is ~200 lines of Python over the shared library.

    quint run --main=raft5 --mbt --max-steps=8 \
              --out-itf=specs/quint/traces/foo.itf.json specs/quint/raft.qnt
    python3 specs/replay/replay_raft.py specs/quint/traces/foo.itf.json

Each trace step carries `mbt::actionTaken` plus `mbt::nondetPicks`, which
together say exactly which C call to make. After every step the driver compares
the model's per-node state against what raft.c reports through its accessors.

Exit status: 0 conformant, 1 divergence, 2 could not run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itf import Trace, strip_module_prefix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
LIB = ROOT / "build" / "lib" / "libGigaVector.so"

CDEF = """
typedef enum { GV_RAFT_FOLLOWER = 0, GV_RAFT_CANDIDATE = 1, GV_RAFT_LEADER = 2 } GV_RaftRole;
typedef enum {
    GV_RAFT_MSG_REQUEST_VOTE = 1, GV_RAFT_MSG_REQUEST_VOTE_RESP = 2,
    GV_RAFT_MSG_APPEND_ENTRIES = 3, GV_RAFT_MSG_APPEND_ENTRIES_RESP = 4
} GV_RaftMsgType;

typedef struct { uint64_t term; void *data; size_t len; } GV_RaftEntry;

typedef struct {
    GV_RaftMsgType type;
    uint64_t term;
    int from;
    uint64_t last_log_index;
    uint64_t last_log_term;
    int vote_granted;
    uint64_t prev_log_index;
    uint64_t prev_log_term;
    uint64_t leader_commit;
    const GV_RaftEntry *entries;
    size_t n_entries;
    int success;
    uint64_t match_index;
} GV_RaftMsg;

typedef struct {
    void (*send)(void *ctx, int to, const GV_RaftMsg *msg);
    void (*apply)(void *ctx, uint64_t index, const void *data, size_t len);
    void (*persist)(void *ctx, uint64_t current_term, int voted_for);
    void *ctx;
} GV_RaftCallbacks;

typedef struct {
    uint32_t election_timeout_min_ms;
    uint32_t election_timeout_max_ms;
    uint32_t heartbeat_ms;
} GV_RaftConfig;

typedef struct GV_Raft GV_Raft;

void raft_config_init(GV_RaftConfig *cfg);
GV_Raft *raft_create(int id, const int *peers, size_t n_peers,
                     const GV_RaftConfig *cfg, const GV_RaftCallbacks *cb);
void raft_tick(GV_Raft *r, uint32_t ms);
void raft_step(GV_Raft *r, int from, const GV_RaftMsg *msg);
int  raft_submit(GV_Raft *r, const void *data, size_t len, uint64_t *index_out);
GV_RaftRole raft_role(const GV_Raft *r);
uint64_t raft_current_term(const GV_Raft *r);
uint64_t raft_commit_index(const GV_Raft *r);
uint64_t raft_last_log_index(const GV_Raft *r);
int raft_voted_for(const GV_Raft *r);
void raft_destroy(GV_Raft *r);
"""

ROLE_NAME = {0: "Follower", 1: "Candidate", 2: "Leader"}
MSG_TYPE = {
    "RequestVote": 1,
    "RequestVoteResp": 2,
    "AppendEntries": 3,
    "AppendEntriesResp": 4,
}


def unwrap(pick):
    """mbt::nondetPicks wraps each value as {tag: Some|None, value: ...}."""
    if isinstance(pick, dict) and "tag" in pick:
        return pick["value"] if pick["tag"] == "Some" else None
    return pick


class RaftCluster:
    def __init__(self, ffi, lib, node_ids):
        self.ffi, self.lib, self.ids = ffi, lib, sorted(node_ids)
        self.nodes = {}
        cfg = ffi.new("GV_RaftConfig *")
        lib.raft_config_init(cfg)
        # The model drives message delivery explicitly, so the C send/apply
        # callbacks are sinks -- we never consume what the implementation emits.
        self._send = ffi.callback("void(void *, int, const GV_RaftMsg *)", lambda c, t, m: None)
        self._apply = ffi.callback("void(void *, uint64_t, const void *, size_t)",
                                   lambda c, i, d, l: None)
        self._keep = []
        for nid in self.ids:
            peers = [p for p in self.ids if p != nid]
            cpeers = ffi.new("int[]", peers)
            cb = ffi.new("GV_RaftCallbacks *")
            cb.send, cb.apply, cb.persist, cb.ctx = self._send, self._apply, ffi.NULL, ffi.NULL
            self._keep += [cpeers, cb, cfg]
            self.nodes[nid] = lib.raft_create(nid, cpeers, len(peers), cfg, cb)
            if self.nodes[nid] == ffi.NULL:
                raise RuntimeError(f"raft_create failed for node {nid}")

    def state(self):
        return {
            nid: {
                "role": ROLE_NAME[int(self.lib.raft_role(h))],
                "currentTerm": int(self.lib.raft_current_term(h)),
                "votedFor": int(self.lib.raft_voted_for(h)),
                "commitIndex": int(self.lib.raft_commit_index(h)),
            }
            for nid, h in self.nodes.items()
        }

    def close(self):
        for h in self.nodes.values():
            self.lib.raft_destroy(h)
        self.nodes.clear()


def build_msg(ffi, m):
    msg = ffi.new("GV_RaftMsg *")
    msg.type = MSG_TYPE[m["kind"]]
    msg.term = m["term"]
    msg.from_ = m["from"] if False else m["from"]  # field is literally `from`
    return msg


def apply_action(ffi, cluster, action, picks):
    """Translate one model action into the equivalent raft.c call."""
    if action in ("init",):
        return True
    if action == "timeout":
        n = unwrap(picks.get("n"))
        if n is None:
            return False
        # Any tick past election_timeout_max_ms forces become_candidate.
        cluster.lib.raft_tick(cluster.nodes[n], 100000)
        return True
    if action == "heartbeat":
        n = unwrap(picks.get("n"))
        if n is None:
            return False
        cluster.lib.raft_tick(cluster.nodes[n], 100000)
        return True
    if action == "clientRequest":
        n = unwrap(picks.get("n"))
        v = unwrap(picks.get("v")) or 0
        if n is None:
            return False
        payload = ffi.new("unsigned char[]", bytes([v & 0xFF]))
        out = ffi.new("uint64_t *")
        cluster.lib.raft_submit(cluster.nodes[n], payload, 1, out)
        return True
    if action in ("handleRequestVote", "handleVoteResp",
                  "handleAppendEntries", "handleAppendResp"):
        m = unwrap(picks.get("m"))
        if m is None:
            return False
        msg = ffi.new("GV_RaftMsg *")
        msg.type = MSG_TYPE[m["kind"]]
        msg.term = m["term"]
        setattr(msg, "from", m["from"])
        msg.last_log_index = m["lastLogIndex"]
        msg.last_log_term = m["lastLogTerm"]
        msg.vote_granted = 1 if m["voteGranted"] else 0
        msg.prev_log_index = m["prevLogIndex"]
        msg.prev_log_term = m["prevLogTerm"]
        msg.leader_commit = m["leaderCommit"]
        msg.entries = ffi.NULL
        msg.n_entries = 0
        msg.success = 1 if m["success"] else 0
        msg.match_index = m["matchIndex"]
        cluster.lib.raft_step(cluster.nodes[m["to"]], m["from"], msg)
        return True
    return False


def compare(model_state, impl_state, fields=("role", "currentTerm")):
    """Return a list of human-readable divergences."""
    out = []
    role = model_state.get("role", {})
    term = model_state.get("currentTerm", {})
    for nid, impl in sorted(impl_state.items()):
        if "role" in fields and nid in role and role[nid] != impl["role"]:
            out.append(f"node {nid}: model role={role[nid]} impl role={impl['role']}")
        if "currentTerm" in fields and nid in term and term[nid] != impl["currentTerm"]:
            out.append(f"node {nid}: model term={term[nid]} impl term={impl['currentTerm']}")
    return out


def replay(path, strict=True):
    try:
        from cffi import FFI
    except ImportError:
        print("cffi is required: pip install cffi", file=sys.stderr)
        return 2
    if not LIB.exists():
        print(f"shared library not found at {LIB}; run `make lib` first", file=sys.stderr)
        return 2

    ffi = FFI()
    ffi.cdef(CDEF)
    lib = ffi.dlopen(str(LIB))

    trace = Trace.load(path)
    actions = trace.actions
    if not any(actions):
        print(f"{path}: no mbt::actionTaken metadata; regenerate with `quint run --mbt`",
              file=sys.stderr)
        return 2

    first = strip_module_prefix(trace[0])
    node_ids = sorted(first.get("role", {}).keys())
    cluster = RaftCluster(ffi, lib, node_ids)

    print(f"replaying {Path(path).name}: {len(trace)} states, {len(node_ids)} nodes")
    divergences = []
    try:
        for i in range(1, len(trace)):
            st = strip_module_prefix(trace[i])
            picks = {k.rsplit("::", 1)[-1]: v
                     for k, v in trace[i].items() if k.endswith("nondetPicks")}
            picks = picks.get("nondetPicks", {}) or {}
            action = actions[i] or "?"
            if not apply_action(ffi, cluster, action, picks):
                print(f"  step {i}: action {action!r} not replayable, skipping")
                continue
            diffs = compare(st, cluster.state())
            status = "ok" if not diffs else "DIVERGE"
            print(f"  step {i}: {action:<20} {status}")
            for d in diffs:
                print(f"           {d}")
                divergences.append((i, action, d))
    finally:
        cluster.close()

    if divergences:
        print(f"\n{len(divergences)} divergence(s) between model and implementation.")
        return 1 if strict else 0
    print("\nconformant: implementation matched the model at every step.")
    return 0


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    strict = "--allow-divergence" not in argv[1:]
    if not args:
        d = ROOT / "specs" / "quint" / "traces"
        args = [str(p) for p in sorted(d.glob("*.itf.json"))]
        if not args:
            print("no traces found; generate with `quint run --mbt --out-itf=...`")
            return 2
    rc = 0
    for p in args:
        rc = max(rc, replay(p, strict=strict))
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
