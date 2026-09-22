// -*- mode: quint ---
//
// replication.qnt — model of leader/follower replication and read routing
// (src/admin/replication.c).
//
// State machine: GV_ReplicationRole {LEADER, FOLLOWER, CANDIDATE} crossed with
// GV_ReplicationState {SYNCING, STREAMING, LAGGING, DISCONNECTED}
// (include/admin/replication.h:29-44), plus read routing under GV_ReadPolicy
// {LEADER_ONLY, ROUND_ROBIN, LEAST_LAG, RANDOM} (replication.h:250).
//
// This is the layer the Raft core is NOT yet wired into
// (docs/plan-middle-status.md): replication.c runs its own election path.
// Specifying it before that integration is the point.

module replication_proto {
  const NODES: Set[int]
  const MAX_ENTRIES: int
  const MAX_READ_LAG: int

  var role: int -> str
  var state: int -> str
  var applied: int -> int          // entries applied per node
  var leaderLog: int               // entries appended on the leader
  var readPolicy: str
  // Reads served, recorded as (node served from, entries it had).
  var readsServed: Set[{ node: int, sawEntries: int }]
  // Entries the caller was told were durably committed.
  var ackedCommitted: Set[int]

  action init = all {
    role' = NODES.mapBy(n => if (n == 0) "leader" else "follower"),
    state' = NODES.mapBy(_ => "streaming"),
    applied' = NODES.mapBy(_ => 0),
    leaderLog' = 0,
    readPolicy' = "leader_only",
    readsServed' = Set(),
    ackedCommitted' = Set(),
  }

  def leaders: Set[int] = NODES.filter(n => role.get(n) == "leader")
  def majority: int = NODES.size() / 2 + 1

  /// A write on the leader.
  action leaderAppend = all {
    leaders.size() > 0,
    leaderLog < MAX_ENTRIES,
    leaderLog' = leaderLog + 1,
    applied' = NODES.mapBy(n => if (role.get(n) == "leader") leaderLog + 1 else applied.get(n)),
    role' = role, state' = state, readPolicy' = readPolicy,
    readsServed' = readsServed, ackedCommitted' = ackedCommitted,
  }

  /// A follower applies the next entry.
  action followerApply(n: int): bool = all {
    role.get(n) == "follower",
    applied.get(n) < leaderLog,
    applied' = applied.set(n, applied.get(n) + 1),
    state' = state.set(n,
      if (leaderLog - (applied.get(n) + 1) > MAX_READ_LAG) "lagging" else "streaming"),
    role' = role, leaderLog' = leaderLog, readPolicy' = readPolicy,
    readsServed' = readsServed, ackedCommitted' = ackedCommitted,
  }

  /// replication_sync_commit: succeeds only when a quorum has applied.
  action syncCommit(e: int): bool = all {
    e <= leaderLog,
    NODES.filter(n => applied.get(n) >= e).size() >= majority,
    ackedCommitted' = ackedCommitted.union(Set(e)),
    role' = role, state' = state, applied' = applied,
    leaderLog' = leaderLog, readPolicy' = readPolicy, readsServed' = readsServed,
  }

  action setPolicy(p: str): bool = all {
    readPolicy' = p,
    role' = role, state' = state, applied' = applied,
    leaderLog' = leaderLog, readsServed' = readsServed,
    ackedCommitted' = ackedCommitted,
  }

  /// replication_route_read: which replicas the policy admits.
  def routable(n: int): bool =
    if (readPolicy == "leader_only") role.get(n) == "leader"
    else if (readPolicy == "least_lag") leaderLog - applied.get(n) <= MAX_READ_LAG
    else state.get(n) != "disconnected"

  action routeRead(n: int): bool = all {
    routable(n),
    readsServed' = readsServed.union(Set({ node: n, sawEntries: applied.get(n) })),
    role' = role, state' = state, applied' = applied,
    leaderLog' = leaderLog, readPolicy' = readPolicy,
    ackedCommitted' = ackedCommitted,
  }

  /// replication_step_down
  action stepDown(n: int): bool = all {
    role.get(n) == "leader",
    role' = role.set(n, "follower"),
    state' = state, applied' = applied, leaderLog' = leaderLog,
    readPolicy' = readPolicy, readsServed' = readsServed,
    ackedCommitted' = ackedCommitted,
  }

  /// replication_request_leadership: a follower promotes itself only if no
  /// leader is currently known -- the guard that prevents split brain.
  action requestLeadership(n: int): bool = all {
    role.get(n) == "follower",
    leaders.size() == 0,
    role' = role.set(n, "leader"),
    state' = state, applied' = applied, leaderLog' = leaderLog,
    readPolicy' = readPolicy, readsServed' = readsServed,
    ackedCommitted' = ackedCommitted,
  }

  action step = any {
    leaderAppend,
    nondet n = NODES.oneOf()
    followerApply(n),
    nondet e = 1.to(MAX_ENTRIES).oneOf()
    syncCommit(e),
    nondet n = NODES.oneOf()
    routeRead(n),
    nondet n = NODES.oneOf()
    stepDown(n),
    nondet n = NODES.oneOf()
    requestLeadership(n),
    nondet p = Set("leader_only", "least_lag", "round_robin").oneOf()
    setPolicy(p),
  }

  // ---- invariants ---------------------------------------------------------

  /// Never two leaders at once.
  val singleLeader: bool = leaders.size() <= 1

  /// An entry acked as committed really is on a quorum.
  val ackImpliesDurable: bool =
    ackedCommitted.forall(e =>
      NODES.filter(n => applied.get(n) >= e).size() >= majority)

  /// Under LEADER_ONLY every served read came from a leader.
  val leaderOnlyRespected: bool =
    (readPolicy == "leader_only") implies
      readsServed.forall(r => r.sawEntries <= leaderLog)

  /// Under LEAST_LAG a served read never came from a replica beyond the bound.
  val readPolicyRespectsLag: bool =
    (readPolicy == "least_lag") implies
      readsServed.forall(r => leaderLog - r.sawEntries <= MAX_READ_LAG)

  /// A follower never claims more entries than the leader has.
  val noOverApply: bool = NODES.forall(n => applied.get(n) <= leaderLog)

  val safety: bool = and { singleLeader, ackImpliesDurable, noOverApply }
}

module replication {
  import replication_proto(NODES = Set(0, 1, 2), MAX_ENTRIES = 3, MAX_READ_LAG = 1).*

  /// A quorum-acked entry really is on a quorum.
  run quorumCommitTest =
    init
      .then(leaderAppend)
      .then(followerApply(1))
      .then(syncCommit(1))
      .expect(and { ackImpliesDurable, ackedCommitted == Set(1) })

  /// Step-down then promotion never yields two leaders.
  run failoverSingleLeaderTest =
    init
      .expect(singleLeader)
      .then(stepDown(0))
      .expect(leaders.size() == 0)
      .then(requestLeadership(1))
      .expect(and { singleLeader, role.get(1) == "leader" })
}
