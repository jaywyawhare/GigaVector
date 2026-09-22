// -*- mode: quint ---
//
// ab_test.qnt — model of A/B index routing (src/admin/ab_test.c) and
// conditional execution (src/specialized/conditional.c).
//
// `ab_test_route` (ab_test.c:92) picks arm A or B according to a split. The
// properties that matter for an experiment: an assignment must be STABLE (the
// same request always routes the same way, or the measurement is meaningless),
// results must be attributed to the arm that served them, and stopping the test
// must route everything back to A.

module ab_proto {
  const REQUESTS: Set[int]
  const SPLIT_NUMERATOR: int      // out of 10; e.g. 3 = 30% to arm B

  var running: bool
  var assignment: int -> str      // request -> "A" | "B" | "none"
  var recorded: Set[{ req: int, arm: str }]

  action init = all {
    running' = false,
    assignment' = REQUESTS.mapBy(_ => "none"),
    recorded' = Set(),
  }

  action experimentStart = all {
    not(running),
    running' = true,
    assignment' = assignment, recorded' = recorded,
  }

  action experimentStop = all {
    running,
    running' = false,
    assignment' = assignment, recorded' = recorded,
  }

  /// Deterministic bucketing: the arm is a pure function of the request id, so
  /// re-routing the same request always gives the same arm.
  def arm(r: int): str =
    if (not(running)) "A"
    else if (r % 10 < SPLIT_NUMERATOR) "B" else "A"

  action route(r: int): bool = all {
    assignment' = assignment.set(r, arm(r)),
    running' = running, recorded' = recorded,
  }

  /// ab_test_record: attribute a measurement to the arm that served it.
  action record(r: int): bool = all {
    assignment.get(r) != "none",
    recorded' = recorded.union(Set({ req: r, arm: assignment.get(r) })),
    running' = running, assignment' = assignment,
  }

  action step = any {
    experimentStart, experimentStop,
    nondet r = REQUESTS.oneOf()
    route(r),
    nondet r = REQUESTS.oneOf()
    record(r),
  }

  // ---- invariants ---------------------------------------------------------

  /// Routing is deterministic: for a fixed running state, the same request
  /// always gets the same arm. Without this the experiment measures noise.
  val assignmentStable: bool =
    REQUESTS.forall(r =>
      assignment.get(r) != "none" implies
        (assignment.get(r) == arm(r) or not(running)))

  /// Every measurement is attributed to a real arm.
  val recordedArmValid: bool =
    recorded.forall(m => m.arm == "A" or m.arm == "B")

  /// With the test stopped, everything routes to the control arm.
  val stoppedRoutesToControl: bool =
    not(running) implies REQUESTS.forall(r => arm(r) == "A")

  val safety: bool = and { recordedArmValid, stoppedRoutesToControl }
}

module ab_test {
  import ab_proto(REQUESTS = Set(1, 2, 3, 11, 12), SPLIT_NUMERATOR = 3).*

  /// The same request routes to the same arm every time.
  run stableAssignmentTest =
    init
      .then(experimentStart)
      .then(route(1))
      .expect(assignment.get(1) == "B")      // 1 % 10 = 1 < 3
      .then(route(11))
      .expect(assignment.get(11) == "B")     // 11 % 10 = 1, same bucket
      .then(route(3))
      .expect(and { assignment.get(3) == "A", assignmentStable })

  /// Stopping the test sends all traffic back to the control arm.
  run stopRoutesToControlTest =
    init
      .then(experimentStart)
      .then(route(1))
      .expect(assignment.get(1) == "B")
      .then(experimentStop)
      .expect(stoppedRoutesToControl)
}
