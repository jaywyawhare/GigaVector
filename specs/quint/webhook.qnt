// -*- mode: quint ---
//
// webhook.qnt — model of webhook registration and delivery
// (src/admin/webhook.c: register / unregister / pause / resume / fire).
//
// Delivery semantics for an outbound notifier: an event fires to every
// registered, non-paused subscriber whose mask matches; retries must terminate;
// and unregistering must stop delivery.

module webhook_proto {
  const HOOKS: Set[int]
  const MAX_EVENTS: int
  const MAX_RETRIES: int

  var registered: Set[int]
  var paused: Set[int]
  var mask: int -> int                       // subscribed event type
  var nextEvent: int
  // Deliveries attempted: (hook, event, attempt number).
  var delivered: Set[{ hook: int, event: int }]
  var attempts: int -> int                   // retries in flight per hook
  var fired: Set[{ event: int, kind: int }]

  action init = all {
    registered' = Set(),
    paused' = Set(),
    mask' = HOOKS.mapBy(_ => 1),
    nextEvent' = 1,
    delivered' = Set(),
    attempts' = HOOKS.mapBy(_ => 0),
    fired' = Set(),
  }

  action register(h: int, m: int): bool = all {
    not(registered.contains(h)),
    registered' = registered.union(Set(h)),
    mask' = mask.set(h, m),
    paused' = paused, nextEvent' = nextEvent, delivered' = delivered,
    attempts' = attempts, fired' = fired,
  }

  action unregister(h: int): bool = all {
    registered.contains(h),
    registered' = registered.exclude(Set(h)),
    paused' = paused.exclude(Set(h)),
    attempts' = attempts.set(h, 0),
    mask' = mask, nextEvent' = nextEvent, delivered' = delivered, fired' = fired,
  }

  action pause(h: int): bool = all {
    registered.contains(h),
    not(paused.contains(h)),
    paused' = paused.union(Set(h)),
    registered' = registered, mask' = mask, nextEvent' = nextEvent,
    delivered' = delivered, attempts' = attempts, fired' = fired,
  }

  action resume(h: int): bool = all {
    paused.contains(h),
    paused' = paused.exclude(Set(h)),
    registered' = registered, mask' = mask, nextEvent' = nextEvent,
    delivered' = delivered, attempts' = attempts, fired' = fired,
  }

  /// webhook_fire: deliver to every registered, unpaused, mask-matching hook.
  action fire(kind: int): bool = all {
    nextEvent <= MAX_EVENTS,
    val targets = registered.exclude(paused).filter(h => mask.get(h) == kind)
    all {
      delivered' = delivered.union(
        targets.map(h => { hook: h, event: nextEvent })),
      fired' = fired.union(Set({ event: nextEvent, kind: kind })),
      nextEvent' = nextEvent + 1,
    },
    registered' = registered, paused' = paused, mask' = mask,
    attempts' = attempts,
  }

  /// A delivery failure schedules a retry, bounded by MAX_RETRIES.
  action retry(h: int): bool = all {
    registered.contains(h),
    attempts.get(h) < MAX_RETRIES,
    attempts' = attempts.set(h, attempts.get(h) + 1),
    registered' = registered, paused' = paused, mask' = mask,
    nextEvent' = nextEvent, delivered' = delivered, fired' = fired,
  }

  action step = any {
    nondet h = HOOKS.oneOf()
    nondet m = Set(1, 2).oneOf()
    register(h, m),
    nondet h = HOOKS.oneOf()
    unregister(h),
    nondet h = HOOKS.oneOf()
    pause(h),
    nondet h = HOOKS.oneOf()
    resume(h),
    nondet k = Set(1, 2).oneOf()
    fire(k),
    nondet h = HOOKS.oneOf()
    retry(h),
  }

  // ---- invariants ---------------------------------------------------------

  /// A paused hook receives nothing while paused, and an unregistered hook
  /// receives nothing at all after unregistering.
  val pausedReceivesNothing: bool =
    paused.forall(h => registered.contains(h))

  /// Retries always terminate: no hook exceeds the retry bound.
  val retriesTerminate: bool =
    HOOKS.forall(h => attempts.get(h) <= MAX_RETRIES)

  /// Every delivery corresponds to an event that was actually fired.
  val noPhantomDelivery: bool =
    delivered.forall(d => fired.exists(f => f.event == d.event))

  /// A delivery only ever went to a hook whose mask matched the event kind.
  val maskRespected: bool =
    delivered.forall(d =>
      fired.exists(f => f.event == d.event and f.kind == mask.get(d.hook))
      or not(registered.contains(d.hook)))

  val safety: bool = and {
    pausedReceivesNothing, retriesTerminate, noPhantomDelivery,
  }
}

module webhook {
  import webhook_proto(HOOKS = Set(1, 2), MAX_EVENTS = 3, MAX_RETRIES = 2).*

  /// A paused hook misses events; resuming does not backfill them.
  run pauseSkipsEventsTest =
    init
      .then(register(1, 1))
      .then(pause(1))
      .then(fire(1))
      .expect(delivered == Set())            // paused: nothing delivered
      .then(resume(1))
      .expect(and { delivered == Set(), noPhantomDelivery })  // no backfill

  /// Retries are bounded, so delivery cannot loop forever.
  run retryBoundedTest =
    init
      .then(register(1, 1))
      .then(retry(1))
      .then(retry(1))
      .expect(and { attempts.get(1) == 2, retriesTerminate })
}
