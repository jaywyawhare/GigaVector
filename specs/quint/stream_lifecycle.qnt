// -*- mode: quint ---
//
// stream_lifecycle.qnt — model of a streaming consumer's stop/pause state
// machine (src/admin/streaming.c: stop_requested / pause_requested flags and
// the RUNNING/PAUSED/STOPPED state the worker loop drives).
//
// The control plane (caller thread) raises stop/pause/resume requests; the
// worker loop reconciles its state to those flags. The hazard behind commits
// 817fdef / 4fa2475 was these flags being read non-atomically, letting a
// stopping consumer get parked in PAUSED. The fix makes stop dominate pause:
// stream_stop sets stop_requested AND clears pause_requested (streaming.c:516),
// so the worker can never observe "paused and not stopping" once stop is set.
// `stopDominatesPause` encodes exactly that.

module stream_lifecycle_proto {
  type State = RUNNING | PAUSED | STOPPED

  var stopRequested: bool
  var pauseRequested: bool
  var state: State

  action init = all {
    stopRequested' = false,
    pauseRequested' = false,
    state' = RUNNING,
  }

  /// stream_pause: only a consumer that is not already stopping may be paused.
  action requestPause = all {
    not(stopRequested),
    pauseRequested' = true,
    stopRequested' = stopRequested, state' = state,
  }

  /// stream_resume: clear the pause request.
  action requestResume = all {
    pauseRequested' = false,
    stopRequested' = stopRequested, state' = state,
  }

  /// stream_stop (correct): raise stop AND clear pause, so stop dominates.
  action requestStop = all {
    stopRequested' = true,
    pauseRequested' = false,
    state' = state,
  }

  /// BUGGY stop (the pre-fix race): raise stop but leave pause_requested set.
  /// With non-atomic flags the worker could then still see pause and park.
  action requestStopUnsafe = all {
    stopRequested' = true,
    pauseRequested' = pauseRequested,
    state' = state,
  }

  /// The worker loop reconciles its visible state to the flags. Stop dominates
  /// pause dominates run -- the same precedence as the nested while loops in
  /// stream_consumer_loop.
  action consumerObserve = all {
    state' = if (stopRequested) STOPPED else if (pauseRequested) PAUSED else RUNNING,
    stopRequested' = stopRequested, pauseRequested' = pauseRequested,
  }

  /// stream_start reuse: a stopped consumer is restarted fresh.
  action restart = all {
    state == STOPPED,
    stopRequested' = false,
    pauseRequested' = false,
    state' = RUNNING,
  }

  action step = any {
    requestPause,
    requestResume,
    requestStop,
    consumerObserve,
    restart,
  }

  // ---- invariants ---------------------------------------------------------

  /// Once stop is requested, pause is never left set: stop clears it. This is
  /// what keeps the worker from parking in PAUSED while it should be stopping.
  val stopDominatesPause: bool = stopRequested implies not(pauseRequested)

  /// A consumer the worker shows as STOPPED really was asked to stop.
  val stoppedWasRequested: bool = (state == STOPPED) implies stopRequested

  val safety: bool = and { stopDominatesPause, stoppedWasRequested }

  // ---- inductive invariant -------------------------------------------------
  val typeOk: bool = and {
    stopRequested.in(Set(false, true)),
    pauseRequested.in(Set(false, true)),
    state.in(Set(RUNNING, PAUSED, STOPPED)),
  }

  val indInv: bool = and {
    typeOk,
    stopDominatesPause,
    stoppedWasRequested,
  }
}

module stream_lifecycle {
  import stream_lifecycle_proto.*

  /// A stop that forgets to clear pause breaks stop-dominance: the consumer is
  /// marked stopping yet still carries a live pause request.
  run unsafeStopBreaksDominanceTest =
    init
      .then(requestPause)
      .expect(pauseRequested)
      .then(requestStopUnsafe)
      .expect(not(stopDominatesPause))

  /// The correct stop clears pause, and the worker then settles on STOPPED
  /// rather than parking in PAUSED.
  run stopDominatesPauseTest =
    init
      .then(requestPause)
      .then(consumerObserve)
      .expect(state == PAUSED)
      .then(requestStop)
      .expect(and { stopRequested, not(pauseRequested) })
      .then(consumerObserve)
      .expect(state == STOPPED)
}
