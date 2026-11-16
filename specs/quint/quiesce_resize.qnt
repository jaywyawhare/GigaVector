// -*- mode: quint ---
//
// quiesce_resize.qnt — model of the in-flight-op counter that gates index
// resize/maintenance (src/storage/db_write.c, db_increment_concurrent_ops /
// db_decrement_concurrent_ops; the quiesce wait in the resize path).
//
// Every write op increments `concurrent_ops` on entry and MUST decrement it on
// EVERY exit path, success or error. A resize/maintenance op waits for the
// counter to reach 0 (quiescence) before mutating shared structures, so that no
// write is touching them while they move. The historical bug (commit 2ce37e5)
// was an error path of db_add_vector_with_metadata that returned WITHOUT
// decrementing: the counter leaked upward, so it reflected more in-flight ops
// than were really running. `counterAccurate` is exactly what that broke.

module quiesce_resize_proto {
  const OPS: Set[int]

  /// Ground truth: ops actually inside their critical section right now.
  var running: Set[int]
  /// The explicit counter the C maintains (db->concurrent_ops). It can drift
  /// from `running` if an exit path forgets to decrement -- that drift is the bug.
  var inflight: int
  /// A resize/maintenance pass is mutating shared structures.
  var resizing: bool

  action init = all {
    running' = Set(),
    inflight' = 0,
    resizing' = false,
  }

  /// A write op enters its critical section: bump the counter. New ops do not
  /// start while a resize is mutating structures (the resize holds the gate).
  action opEnter(o: int): bool = all {
    not(resizing),
    not(running.contains(o)),
    running' = running.union(Set(o)),
    inflight' = inflight + 1,
    resizing' = resizing,
  }

  /// Correct exit: leave the critical section AND decrement. Every real exit
  /// path -- the happy path and all error paths -- must look like this.
  action opExit(o: int): bool = all {
    running.contains(o),
    running' = running.exclude(Set(o)),
    inflight' = inflight - 1,
    resizing' = resizing,
  }

  /// BUGGY exit (the 2ce37e5 defect): an error path returns without
  /// decrementing. The op is gone from `running` but the counter still counts
  /// it, so it never falls back to 0 and the resize quiesce wait hangs.
  action opExitLeak(o: int): bool = all {
    running.contains(o),
    running' = running.exclude(Set(o)),
    inflight' = inflight,
    resizing' = resizing,
  }

  /// Resize starts only once the counter says the system is quiescent.
  action resizeStart = all {
    not(resizing),
    inflight == 0,
    resizing' = true,
    running' = running, inflight' = inflight,
  }

  action resizeEnd = all {
    resizing,
    resizing' = false,
    running' = running, inflight' = inflight,
  }

  action step = any {
    nondet o = OPS.oneOf()
    opEnter(o),
    nondet o = OPS.oneOf()
    opExit(o),
    resizeStart,
    resizeEnd,
  }

  // ---- invariants ---------------------------------------------------------

  /// The counter reflects reality: exactly the ops in flight, no leak.
  val counterAccurate: bool = inflight == running.size()

  /// Resize never runs concurrently with a live write op. This is the property
  /// quiescence exists to guarantee; it holds only because the counter is
  /// accurate, so a leaked counter (reading 0 while an op runs, or never
  /// reaching 0) defeats it.
  val resizeQuiescent: bool = resizing implies running == Set()

  val safety: bool = and { counterAccurate, resizeQuiescent }

  // ---- inductive invariant -------------------------------------------------
  val typeOk: bool = and {
    running.in(OPS.powerset()),
    inflight.in(0.to(OPS.size())),
    resizing.in(Set(false, true)),
  }

  /// counterAccurate is the strengthening that keeps resizeQuiescent closed
  /// under step: with it, resizeStart's `inflight == 0` guard forces `running`
  /// empty, so no resize can overlap a running op.
  val indInv: bool = and {
    typeOk,
    counterAccurate,
    resizeQuiescent,
  }
}

module quiesce_resize {
  import quiesce_resize_proto(OPS = Set(1, 2)).*

  /// A forgotten decrement leaves the counter disagreeing with reality.
  run leakBreaksCounterTest =
    init
      .then(opEnter(1))
      .expect(counterAccurate)
      .then(opExitLeak(1))
      .expect(not(counterAccurate))

  /// The correct exit keeps the counter exact, so quiescence is real.
  run cleanExitTest =
    init
      .then(opEnter(1))
      .then(opEnter(2))
      .then(opExit(1))
      .then(opExit(2))
      .expect(and { counterAccurate, inflight == 0 })
      .then(resizeStart)
      .expect(and { resizing, resizeQuiescent })
}
