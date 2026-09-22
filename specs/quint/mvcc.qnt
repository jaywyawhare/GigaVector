// -*- mode: quint -*-
//
// mvcc.qnt — model of GigaVector's MVCC layer (src/specialized/mvcc.c).
//
// Models the visibility predicate `mvcc_version_visible` (mvcc.c:194) exactly,
// together with begin/commit/rollback/delete and `gv_mvcc_gc` (mvcc.c).
//
// Fidelity points:
//   * `snapshot_txn_id = gv_txn_id - 1` (mvcc.c, gv_txn_begin). A transaction's
//     snapshot therefore NUMERICALLY includes every lower-numbered transaction,
//     whether or not it had committed at begin time. Visibility compensates with
//     a liveness test against the active list -- see `repeatableRead`.
//   * GC reclaims a version iff delete_txn != 0, the deleter is not active, and
//     delete_txn < min(active ids) (mvcc.c, gv_mvcc_gc).
//   * rollback stamps delete_txn = self on inserts and clears self-stamped
//     deletes (mvcc.c, gv_txn_rollback).
//
// Every operation is modelled as atomic because mgr->mutex serialises them.
//
// Version array POSITIONS (and the GC remap of added_indices/deleted_indices)
// are an implementation detail below this model's abstraction; those are
// covered by the C tests. This model is about visibility and reclamation.

module mvcc_proto {
  type Version = {
    vid: int,          // stable identity; the C code uses an array position
    vectorIndex: int,
    createTxn: int,
    deleteTxn: int     // 0 = alive
  }

  const MAX_TXN: int
  const VECTORS: Set[int]

  pure val NO_TXN: int = 0

  var versions: Set[Version]
  var activeTxns: Set[int]
  var committedTxns: Set[int]
  var abortedTxns: Set[int]
  var nextTxnId: int
  var nextVid: int

  // Auxiliary history. Each entry records that transaction `txn` observed
  // version `vid` with visibility `vis`. Two entries differing only in `vis`
  // mean the same reader saw the same version both ways -- a snapshot
  // isolation violation.
  var observed: Set[{ txn: int, vid: int, vis: bool, own: bool }]

  /// The active set each transaction FROZE at begin time (mvcc.c
  /// snapshot_active, PostgreSQL's xip_list). Visibility consults this instead
  /// of the live active list, so a concurrent commit cannot retroactively
  /// change what an open reader sees.
  var snapshotActive: int -> Set[int]

  // ---- the implementation's visibility predicate (mvcc.c:194) -------------
  def isActive(t: int): bool = activeTxns.contains(t)

  /// Was `id` in flight when `reader` took its snapshot?
  def inSnapshot(reader: int, id: int): bool =
    snapshotActive.keys().contains(reader) and snapshotActive.get(reader).contains(id)

  def visible(v: Version, readerTxn: int, snapshot: int): bool =
    if (v.createTxn == readerTxn)
      v.deleteTxn != readerTxn                  // own write, unless self-deleted
    else if (v.createTxn > snapshot) false      // created after our snapshot
    else if (inSnapshot(readerTxn, v.createTxn)) false  // in flight at our snapshot
    else if (v.deleteTxn == NO_TXN) true        // never deleted
    else if (v.deleteTxn == readerTxn) false    // we deleted it ourselves
    else if (v.deleteTxn > snapshot) true       // deleted after our snapshot
    else if (inSnapshot(readerTxn, v.deleteTxn)) true   // deleter in flight at snapshot
    else false                                  // deleted by a committed txn

  def snapshotOf(t: int): int = t - 1

  def visibleTo(v: Version, t: int): bool = visible(v, t, snapshotOf(t))

  // ---- init ---------------------------------------------------------------
  action init = all {
    versions' = Set(),
    activeTxns' = Set(),
    committedTxns' = Set(),
    abortedTxns' = Set(),
    nextTxnId' = 1,
    nextVid' = 1,
    observed' = Set(),
    snapshotActive' = Map(),
  }

  // ---- actions ------------------------------------------------------------

  /// gv_txn_begin
  action begin = all {
    nextTxnId <= MAX_TXN,
    activeTxns' = activeTxns.union(Set(nextTxnId)),
    // Freeze the in-flight set for this reader's whole lifetime.
    snapshotActive' = snapshotActive.put(nextTxnId, activeTxns),
    nextTxnId' = nextTxnId + 1,
    versions' = versions, committedTxns' = committedTxns,
    abortedTxns' = abortedTxns, nextVid' = nextVid, observed' = observed,
  }

  /// gv_txn_add_vector — appends a version stamped with the writer's id.
  action addVector(t: int, vi: int): bool = all {
    activeTxns.contains(t),
    versions' = versions.union(Set({
      vid: nextVid, vectorIndex: vi, createTxn: t, deleteTxn: NO_TXN })),
    nextVid' = nextVid + 1,
    activeTxns' = activeTxns, committedTxns' = committedTxns,
    abortedTxns' = abortedTxns, nextTxnId' = nextTxnId, observed' = observed,
     snapshotActive' = snapshotActive,
  }

  /// gv_txn_delete_vector — stamps delete_txn on a VISIBLE version, refusing
  /// when another transaction already has a pending delete on it
  /// (the write-write conflict check).
  action deleteVector(t: int, v: Version): bool = all {
    activeTxns.contains(t),
    versions.contains(v),
    visibleTo(v, t),
    v.deleteTxn == NO_TXN or v.deleteTxn == t,     // conflict check
    versions' = versions.exclude(Set(v))
                        .union(Set({ ...v, deleteTxn: t })),
    activeTxns' = activeTxns, committedTxns' = committedTxns,
    abortedTxns' = abortedTxns, nextTxnId' = nextTxnId,
    nextVid' = nextVid, observed' = observed,
     snapshotActive' = snapshotActive,
  }

  /// A read. Records what the reader saw so that `repeatableRead` can compare
  /// observations of the same version by the same reader over time.
  action read(t: int, v: Version): bool = all {
    activeTxns.contains(t),
    versions.contains(v),
    // `own` records whether this version is one the reader itself wrote. A
    // transaction seeing the effect of its OWN insert or delete is correct
    // behaviour, not a snapshot violation, so those observations are excluded
    // from repeatableRead below.
    observed' = observed.union(Set({
      txn: t, vid: v.vid, vis: visibleTo(v, t),
      own: (v.createTxn == t or v.deleteTxn == t) })),
    versions' = versions, activeTxns' = activeTxns,
    committedTxns' = committedTxns, abortedTxns' = abortedTxns,
    nextTxnId' = nextTxnId, nextVid' = nextVid,
     snapshotActive' = snapshotActive,
  }

  /// gv_txn_commit — pending deletes simply become permanent.
  action commit(t: int): bool = all {
    activeTxns.contains(t),
    activeTxns' = activeTxns.exclude(Set(t)),
    committedTxns' = committedTxns.union(Set(t)),
    versions' = versions, abortedTxns' = abortedTxns,
    nextTxnId' = nextTxnId, nextVid' = nextVid, observed' = observed,
     snapshotActive' = snapshotActive,
  }

  /// gv_txn_rollback — stamp self-created versions deleted, clear self-stamped
  /// deletes on versions we did not create.
  action rollback(t: int): bool = all {
    activeTxns.contains(t),
    versions' = versions.map(v =>
      if (v.createTxn == t) { ...v, deleteTxn: t }
      else if (v.deleteTxn == t) { ...v, deleteTxn: NO_TXN }
      else v),
    activeTxns' = activeTxns.exclude(Set(t)),
    abortedTxns' = abortedTxns.union(Set(t)),
    committedTxns' = committedTxns,
    nextTxnId' = nextTxnId, nextVid' = nextVid, observed' = observed,
     snapshotActive' = snapshotActive,
  }

  /// gv_mvcc_gc — reclaim deleted versions below the active low-water mark.
  /// `min_active` is UINT64_MAX when nothing is active; MAX_TXN + 1 stands in.
  action gc = all {
    val minActive = if (activeTxns.size() == 0) MAX_TXN + 1
                    else activeTxns.fold(MAX_TXN + 1, (a, b) => if (b < a) b else a)
    versions' = versions.filter(v => not(and {
      v.deleteTxn != NO_TXN,
      not(isActive(v.deleteTxn)),
      // Not frozen into any live reader's snapshot -- required once visibility
      // consults frozen sets instead of the live active list.
      not(activeTxns.exists(t => inSnapshot(t, v.deleteTxn))),
      v.deleteTxn < minActive,
    })),
    activeTxns' = activeTxns, committedTxns' = committedTxns,
    abortedTxns' = abortedTxns, nextTxnId' = nextTxnId,
    nextVid' = nextVid, observed' = observed,
     snapshotActive' = snapshotActive,
  }

  action step = any {
    begin,
    gc,
    nondet t = 1.to(MAX_TXN).oneOf()
    nondet vi = VECTORS.oneOf()
    addVector(t, vi),
    nondet t = 1.to(MAX_TXN).oneOf()
    commit(t),
    nondet t = 1.to(MAX_TXN).oneOf()
    rollback(t),
    all {
      versions.size() > 0,
      nondet t = 1.to(MAX_TXN).oneOf()
      nondet v = versions.oneOf()
      any { deleteVector(t, v), read(t, v) },
    },
  }

  // ---- invariants ---------------------------------------------------------

  /// GC never reclaims a version that is still visible to an active reader.
  /// Stated as: for every active transaction, every version it can see is one
  /// the reclamation predicate would reject.
  val gcSafety: bool =
    activeTxns.forall(t =>
      versions.forall(v =>
        visibleTo(v, t) implies not(and {
          v.deleteTxn != NO_TXN,
          not(isActive(v.deleteTxn)),
          not(activeTxns.exists(u => inSnapshot(u, v.deleteTxn))),
          v.deleteTxn < (if (activeTxns.size() == 0) MAX_TXN + 1
                         else activeTxns.fold(MAX_TXN + 1, (a, b) => if (b < a) b else a)),
        })))

  /// A version created by an aborted transaction is never visible to anyone.
  val abortedInvisible: bool =
    versions.forall(v =>
      abortedTxns.contains(v.createTxn) implies
        activeTxns.forall(t => t != v.createTxn implies not(visibleTo(v, t))))

  /// Two active transactions never hold a pending delete on the same version.
  val noWriteWriteConflict: bool =
    versions.forall(v =>
      (v.deleteTxn != NO_TXN and isActive(v.deleteTxn)) implies
        activeTxns.filter(t => t == v.deleteTxn).size() <= 1)

  /// Snapshot isolation / repeatable read: a reader never observes the same
  /// version's visibility two different ways.
  ///
  /// This is the interesting one. `snapshot_txn_id = gv_txn_id - 1` means a
  /// transaction's snapshot numerically covers every lower id, including
  /// transactions that were still ACTIVE when it began. Visibility masks those
  /// with an `is_active` test -- so the instant such a transaction commits, it
  /// leaves the active list and its writes become retroactively visible to a
  /// reader that started earlier and has already read.
  /// Snapshot stability: a reader never observes the same version's visibility
  /// two different ways -- EXCLUDING versions it wrote itself, where a change
  /// is read-your-own-writes rather than a violation.
  val repeatableRead: bool =
    observed.forall(a => observed.forall(b =>
      (a.txn == b.txn and a.vid == b.vid and not(a.own) and not(b.own))
        implies a.vis == b.vis))

  /// The invariants that currently HOLD. `repeatableRead` is deliberately
  /// excluded: it is FALSE against today's implementation and is a confirmed
  /// finding (see docs/formal_verification_plan.md). Add it here once
  /// gv_txn_begin captures the active transaction set, so the fix cannot
  /// silently regress -- do not "fix" this bundle instead of the code.
  /// `repeatableRead` is now IN the bundle: with the frozen snapshot it holds.
  val safety: bool = and {
    gcSafety,
    abortedInvisible,
    noWriteWriteConflict,
    repeatableRead,
  }
}

/// Small instance: enough transactions to interleave a commit inside another
/// transaction's lifetime, which is what `repeatableRead` needs.
module mvcc {
  import mvcc_proto(MAX_TXN = 4, VECTORS = Set(1, 2)).*

  /// Two transactions overlap. T1 begins, T2 begins (snapshot = 1, which
  /// numerically covers T1). T1 inserts and T2 reads -- invisible, because T1
  /// is still active. T1 commits. T2 reads the same version again.
  run stableSnapshotTest =
    init
      .then(begin)                                  // T1
      .then(begin)                                  // T2, snapshot = 1
      .then(addVector(1, 1))                        // T1 inserts vid 1
      .then(all {
        nondet v = versions.oneOf()
        read(2, v)                                  // T2: not visible (T1 active)
      })
      .then(commit(1))                              // T1 commits
      .then(all {
        nondet v = versions.oneOf()
        read(2, v)                                  // T2 reads again
      })
      // After the fix T2's snapshot is stable: T1's commit does not
      // retroactively change what T2 can see.
      .expect(repeatableRead)
}
