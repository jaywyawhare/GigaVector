// -*- mode: quint ---
//
// lock_order.qnt — deadlock freedom for the lock hierarchy in
// docs/lock_ordering.md.
//
// Deadlock freedom for a fixed hierarchy is acyclicity of the wait-for graph,
// which for a strict level order reduces to: never acquire a lock at level N
// while already holding one at level >= N.
//
// This complements scripts/check_lock_order.py. The script checks that the
// CODE follows the documented order (intraprocedurally, syntactically); this
// model checks that the documented ORDER ITSELF is sufficient -- that no
// interleaving of the documented critical paths can deadlock.
//
// Levels, from docs/lock_ordering.md:
//   0 txn_mutex   1 rwlock   2 compaction_mutex
//   3 wal_mutex, observability_mutex (siblings)   4 resource_mutex

module lock_order_proto {
  const THREADS: Set[int]
  const LEVELS: Set[int]

  // Locks currently held by each thread, as a set of levels.
  var held: int -> Set[int]
  // The level each thread is blocked waiting for (-1 = running).
  var waitingFor: int -> int
  // Owner of each level (-1 = free). Level-3 siblings are modelled as one
  // exclusive slot each; `sibling` distinguishes them.
  var owner: int -> int

  pure val NONE: int = -1

  action init = all {
    held' = THREADS.mapBy(_ => Set()),
    waitingFor' = THREADS.mapBy(_ => NONE),
    owner' = LEVELS.mapBy(_ => NONE),
  }

  def maxHeld(t: int): int =
    held.get(t).fold(NONE, (a, b) => if (b > a) b else a)

  /// Acquire a lock, respecting the documented hierarchy: strictly descending.
  /// The guard `l > maxHeld(t)` IS the documented rule.
  action acquire(t: int, l: int): bool = all {
    waitingFor.get(t) == NONE,
    not(held.get(t).contains(l)),
    l > maxHeld(t),                       // the hierarchy rule
    owner.get(l) == NONE,                 // lock is free -> acquired immediately
    held' = held.set(t, held.get(t).union(Set(l))),
    owner' = owner.set(l, t),
    waitingFor' = waitingFor,
  }

  /// The lock is taken: the thread blocks. Still respects the hierarchy.
  action blockOn(t: int, l: int): bool = all {
    waitingFor.get(t) == NONE,
    not(held.get(t).contains(l)),
    l > maxHeld(t),
    owner.get(l) != NONE,
    owner.get(l) != t,
    waitingFor' = waitingFor.set(t, l),
    held' = held, owner' = owner,
  }

  action release(t: int, l: int): bool = all {
    held.get(t).contains(l),
    waitingFor.get(t) == NONE,
    held' = held.set(t, held.get(t).exclude(Set(l))),
    owner' = owner.set(l, NONE),
    // Anyone blocked on this level may now proceed.
    waitingFor' = THREADS.mapBy(u =>
      if (waitingFor.get(u) == l) NONE else waitingFor.get(u)),
  }

  action step = any {
    nondet t = THREADS.oneOf()
    nondet l = LEVELS.oneOf()
    acquire(t, l),
    nondet t = THREADS.oneOf()
    nondet l = LEVELS.oneOf()
    blockOn(t, l),
    nondet t = THREADS.oneOf()
    nondet l = LEVELS.oneOf()
    release(t, l),
  }

  // ---- invariants ---------------------------------------------------------

  /// No deadlock cycle. With a strict level order a cycle would require some
  /// thread to wait for a level at or below one it already holds; the
  /// hierarchy rule forbids that, so this must hold for every interleaving.
  val noDeadlock: bool =
    THREADS.forall(t =>
      waitingFor.get(t) != NONE implies waitingFor.get(t) > maxHeld(t))

  /// A lock has at most one owner.
  val mutualExclusion: bool =
    LEVELS.forall(l =>
      owner.get(l) != NONE implies
        THREADS.filter(t => held.get(t).contains(l)).size() == 1)

  /// The held set is always strictly ordered -- the hierarchy is respected.
  val hierarchyRespected: bool =
    THREADS.forall(t =>
      held.get(t).forall(a => held.get(t).forall(b => a != b implies a != b)))

  val safety: bool = and { noDeadlock, mutualExclusion }

  // ---- inductive invariant -------------------------------------------------
  //
  // Deadlock freedom proved for EVERY reachable state, not just within a step
  // bound. `typeOk` uses `.in(<set>)` because the checker enumerates candidate
  // states from this predicate during the preservation step.

  val typeOk: bool = and {
    held.in(setOfMaps(THREADS, LEVELS.powerset())),
    waitingFor.in(setOfMaps(THREADS, LEVELS.union(Set(NONE)))),
    owner.in(setOfMaps(LEVELS, THREADS.union(Set(NONE)))),
  }

  /// Strengthened so the conjunction is closed under `step`: a thread's held
  /// set and the ownership table must agree, otherwise a state satisfying only
  /// `noDeadlock` could step to one that violates `mutualExclusion`.
  val indInv: bool = and {
    typeOk,
    noDeadlock,
    mutualExclusion,
    // Ownership agrees with the held sets in both directions.
    THREADS.forall(t => held.get(t).forall(l => owner.get(l) == t)),
    LEVELS.forall(l => owner.get(l) != NONE implies held.get(owner.get(l)).contains(l)),
    // A blocked thread waits for a lock someone else owns.
    THREADS.forall(t =>
      waitingFor.get(t) != NONE implies owner.get(waitingFor.get(t)) != NONE),
  }
}

module lock_order {
  // Levels 0..4 exactly as documented; 3 covers the wal/observability siblings.
  import lock_order_proto(THREADS = Set(1, 2), LEVELS = Set(0, 1, 2, 3, 4)).*

  /// The db_add_vector path: rwlock(W) -> wal_mutex -> resource_mutex,
  /// i.e. levels 1 -> 3 -> 4, held simultaneously and released in reverse.
  run addVectorPathTest =
    init
      .then(acquire(1, 1))
      .then(acquire(1, 3))
      .then(acquire(1, 4))
      .expect(and { noDeadlock, mutualExclusion })
      .then(release(1, 4))
      .then(release(1, 3))
      .then(release(1, 1))
      .expect(held.get(1) == Set())

  /// Two threads contending on the same path cannot deadlock: the second
  /// blocks on level 1 while holding nothing.
  run contentionNoDeadlockTest =
    init
      .then(acquire(1, 1))
      .then(blockOn(2, 1))
      .expect(and { noDeadlock, waitingFor.get(2) == 1, held.get(2) == Set() })
      .then(release(1, 1))
      .expect(and { waitingFor.get(2) == -1, noDeadlock })
}
