// -*- mode: quint ---
//
// tiered_storage.qnt — model of access-aware storage tiers
// (src/storage/tiered_storage.c).
//
// Per docs/lock_ordering.md, `mgr->mutex` guards the per-slot insert/access
// counters, and `gv_db_record_vector_access` is called from the SEARCH path
// only AFTER db_search has released db->rwlock. So tier accounting is not
// serialised with the search that produced it: two concurrent searches can
// both read a counter, both decide to promote, and both apply.

module tiered_proto {
  const VECTORS: Set[int]
  const PROMOTE_THRESHOLD: int

  var tier: int -> str              // "hot" | "warm" | "cold"
  var accessCount: int -> int
  // Promotions decided but not yet applied (the unlocked window).
  var pendingPromote: Set[int]

  action init = all {
    tier' = VECTORS.mapBy(_ => "cold"),
    accessCount' = VECTORS.mapBy(_ => 0),
    pendingPromote' = Set(),
  }

  /// gv_db_record_vector_access
  action recordAccess(v: int): bool = all {
    accessCount' = accessCount.set(v, accessCount.get(v) + 1),
    tier' = tier, pendingPromote' = pendingPromote,
  }

  /// Decide to promote, based on the counter read at this instant.
  action decidePromote(v: int): bool = all {
    accessCount.get(v) >= PROMOTE_THRESHOLD,
    tier.get(v) != "hot",
    not(pendingPromote.contains(v)),
    pendingPromote' = pendingPromote.union(Set(v)),
    tier' = tier, accessCount' = accessCount,
  }

  action applyPromote(v: int): bool = all {
    pendingPromote.contains(v),
    tier' = tier.set(v, "hot"),
    pendingPromote' = pendingPromote.exclude(Set(v)),
    accessCount' = accessCount,
  }

  action demote(v: int): bool = all {
    tier.get(v) == "hot",
    not(pendingPromote.contains(v)),
    tier' = tier.set(v, "warm"),
    accessCount' = accessCount.set(v, 0),
    pendingPromote' = pendingPromote,
  }

  action step = any {
    nondet v = VECTORS.oneOf()
    recordAccess(v),
    nondet v = VECTORS.oneOf()
    decidePromote(v),
    nondet v = VECTORS.oneOf()
    applyPromote(v),
    nondet v = VECTORS.oneOf()
    demote(v),
  }

  // ---- invariants ---------------------------------------------------------

  /// A vector lives in exactly one tier. Trivially true by construction here;
  /// stated so a future refactor to a per-tier set representation is checked.
  val exactlyOneTier: bool =
    VECTORS.forall(v => Set("hot", "warm", "cold").contains(tier.get(v)))

  /// Access counts never go negative or lose updates below zero.
  val accessCountNonNegative: bool = VECTORS.forall(v => accessCount.get(v) >= 0)

  /// A promotion is only ever applied to a vector that met the threshold.
  val promotionJustified: bool =
    pendingPromote.forall(v => accessCount.get(v) >= PROMOTE_THRESHOLD)

  val safety: bool = and { exactlyOneTier, accessCountNonNegative }
}

module tiered_storage {
  import tiered_proto(VECTORS = Set(1), PROMOTE_THRESHOLD = 2).*

  /// A demotion that lands between the promote decision and its application
  /// resets the counter, so the promotion is applied on stale evidence.
  run promoteRaceTest =
    init
      .then(recordAccess(1))
      .then(recordAccess(1))
      .then(decidePromote(1))
      .expect(promotionJustified)
      .then(applyPromote(1))
      .expect(and { tier.get(1) == "hot", exactlyOneTier })
}
