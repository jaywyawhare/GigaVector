// -*- mode: quint -*-
//
// repl_wal_crash.qnt — CROSS-SUBSYSTEM model: replication (replication.qnt)
// composed with WAL durability (wal_recovery.qnt).
//
// Neither sub-spec can catch interaction bugs on its own:
//   * replication.qnt acks commits once a quorum has APPLIED, but knows
//     nothing about fsync -- an entry acked from OS page cache vanishes on a
//     machine crash.
//   * wal_recovery.qnt models durability but has no replication, so "acked"
//     never depends on remote state.
//
// This composition makes commit depend on BOTH subsystems and adds leader
// crash + failover:
//
//   syncCommit(e) requires  e <= walDurable   (WAL fsync gate, wal.c:45)
//                     AND    quorum applied    (replication quorum gate)
//
// Failover models the Raft election restriction that replication.c actually
// implements (src/admin/replication.c:170-176): a candidate whose log is not
// at least as up-to-date as every voter's is refused promotion. Without that
// guard a lagging follower can win, truncate the log below an ACKED entry,
// and the noAckedLoss invariant below goes FALSE -- which is the whole point
// of composing these two models.
//
// Machine-crash semantics follow wal_recovery.qnt: everything appended but
// not yet fsync()ed is lost; followers detect the truncation via heartbeat
// and roll back to the leader's new log end.

module repl_wal_proto {
  const NODES: Set[int]
  const MAX_ENTRIES: int

  var role: int -> str
  // Entries appended by the current leader (its WAL write position).
  var leaderLog: int
  // Prefix of the leader's WAL that is fsync()ed (durable across power loss).
  var walDurable: int
  // Entries applied per node; the leader applies locally on append.
  var applied: int -> int
  // Entries the caller was told were durably committed.
  var ackedCommitted: Set[int]

  action init = all {
    role' = NODES.mapBy(n => if (n == 0) "leader" else "follower"),
    leaderLog' = 0,
    walDurable' = 0,
    applied' = NODES.mapBy(_ => 0),
    ackedCommitted' = Set(),
  }

  def leaders: Set[int] = NODES.filter(n => role.get(n) == "leader")
  def majority: int = NODES.size() / 2 + 1

  /// Leader appends to its WAL (buffered; durable only after walSync).
  /// The leader applies the entry locally immediately.
  action leaderAppend =
    nondet l = NODES.oneOf()
    all {
      role.get(l) == "leader",
      leaderLog < MAX_ENTRIES,
      leaderLog' = leaderLog + 1,
      walDurable' = walDurable,
      applied' = applied.set(l, leaderLog + 1),
      role' = role, ackedCommitted' = ackedCommitted,
    }

  /// Group-commit boundary: wal_maybe_sync / wal_set_sync_interval fsyncs
  /// everything appended so far (wal.c:45-53).
  action walSync = all {
    walDurable < leaderLog,
    walDurable' = leaderLog,
    leaderLog' = leaderLog,
    applied' = applied,
    role' = role, ackedCommitted' = ackedCommitted,
  }

  /// Async shipping: follower n applies the next entry.
  action followerApply(n: int): bool = all {
    role.get(n) == "follower",
    applied.get(n) < leaderLog,
    applied' = applied.set(n, applied.get(n) + 1),
    leaderLog' = leaderLog, walDurable' = walDurable,
    role' = role, ackedCommitted' = ackedCommitted,
  }

  /// The ONLY path to an ack: fsynced on the leader AND held by a quorum.
  action syncCommit(e: int): bool = all {
    e <= walDurable,
    e <= leaderLog,
    NODES.filter(n => applied.get(n) >= e).size() >= majority,
    ackedCommitted' = ackedCommitted.union(Set(e)),
    leaderLog' = leaderLog, walDurable' = walDurable,
    applied' = applied, role' = role,
  }

  /// Power loss on the leader: the non-fsynced WAL tail is gone. Followers
  /// learn the truncation from the next heartbeat and roll back to it --
  /// there is no window where a follower stays ahead of the leader's disk.
  action machineCrash = all {
    walDurable < leaderLog,
    leaderLog' = walDurable,
    walDurable' = walDurable,
    applied' = NODES.mapBy(n =>
      if (role.get(n) == "leader") walDurable
      else if (applied.get(n) > walDurable) walDurable
      else applied.get(n)),
    role' = role, ackedCommitted' = ackedCommitted,
  }

  /// Graceful leader step-down; nobody has promoted yet.
  action stepDown(n: int): bool = all {
    role.get(n) == "leader",
    role' = role.set(n, "follower"),
    leaderLog' = leaderLog, walDurable' = walDurable,
    applied' = applied, ackedCommitted' = ackedCommitted,
  }

  /// Promotion under the Raft election restriction (replication.c:170):
  /// candidate n wins only when NO node has a longer log -- every other node
  /// would withhold its vote otherwise. The winner's replicated log becomes
  /// the authoritative WAL position.
  action requestLeadership(n: int): bool = all {
    role.get(n) == "follower",
    leaders.size() == 0,
    NODES.forall(m => applied.get(m) <= applied.get(n)),
    role' = role.set(n, "leader"),
    leaderLog' = applied.get(n),
    walDurable' = applied.get(n),
    applied' = applied, ackedCommitted' = ackedCommitted,
  }

  action step = any {
    leaderAppend,
    walSync,
    nondet n = NODES.oneOf()
    followerApply(n),
    nondet e = 1.to(MAX_ENTRIES).oneOf()
    syncCommit(e),
    machineCrash,
    nondet n = NODES.oneOf()
    stepDown(n),
    nondet n = NODES.oneOf()
    requestLeadership(n),
  }

  // ---- invariants ---------------------------------------------------------

  val singleLeader: bool = leaders.size() <= 1

  /// THE cross-subsystem property: an acked entry survives every crash and
  /// every failover. It needs BOTH gates -- drop `e <= walDurable` from
  /// syncCommit and machineCrash falsifies this within two steps; drop the
  /// election restriction from requestLeadership and failover falsifies it.
  val noAckedLoss: bool =
    ackedCommitted.forall(e => e <= leaderLog)

  /// An acked entry was really on a quorum at ack time; since logs only ever
  /// move to positions some node held, the quorum bound still shows here.
  val noOverApply: bool = NODES.forall(n => applied.get(n) <= leaderLog)

  /// The WAL durable prefix never exceeds what was appended, and the leader
  /// always holds its own durable prefix.
  val walSane: bool = walDurable <= leaderLog

  val safety: bool = and { singleLeader, noAckedLoss, noOverApply, walSane }
}

/// Production configuration: commit requires fsync + quorum, elections
/// require an up-to-date log.
module repl_wal_crash {
  import repl_wal_proto(NODES = Set(0, 1, 2), MAX_ENTRIES = 3).*

  /// Append -> fsync -> ship -> quorum ack. After the quorum ack the WAL is
  /// fully synced, so a machine crash has nothing left to lose, and failover
  /// keeps every acked entry.
  run committedSurvivesMachineCrashTest =
    init
      .then(leaderAppend)
      .then(walSync)
      .then(followerApply(1))
      .then(syncCommit(1))
      .expect(and {
        ackedCommitted == Set(1),
        walSane,
        // Fully fsynced + replicated: a crash now loses nothing.
        walDurable == leaderLog,
      })
      .then(stepDown(0))
      .then(requestLeadership(1))
      .expect(noAckedLoss)

  /// A lagging follower may NOT promote -- the replication.c:170 guard.
  /// Node 2 catches up with the old leader's log; node 1 stays one entry
  /// behind, so its promotion is refused and only node 2 wins.
  run laggingFollowerCannotWinTest =
    init
      .then(leaderAppend)
      .then(leaderAppend)
      .then(followerApply(2))
      .then(followerApply(2))
      .then(followerApply(1))
      .then(stepDown(0))
      // Election restriction: node 0 and node 2 both hold longer logs than
      // node 1's.
      .expect(not(NODES.forall(m => applied.get(m) <= applied.get(1))))
      .then(requestLeadership(2))
      .expect(and { role.get(2) == "leader", noAckedLoss })
}
