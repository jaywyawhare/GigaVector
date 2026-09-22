// -*- mode: quint ---
//
// dedup.qnt — model of near-duplicate detection and stable point ids
// (src/specialized/dedup.c, src/specialized/point_id.c).
//
// `dedup_insert` (dedup.c:291) calls `dedup_check` internally and inserts only
// when it reports no duplicate, so the check and the insert are one operation
// from the caller's perspective. Note the comment at dedup.c:301: a -1 from
// dedup_check means EITHER "unique" OR "error on an empty index", and both are
// treated as "insert". That conflation is the interesting part.
//
// Duplicate detection is by epsilon distance, so it is a tolerance relation,
// NOT an equivalence relation -- it is reflexive and symmetric but not
// transitive. `dedupIsNotTransitive` records that explicitly, because callers
// that assume clustering semantics will be wrong at the boundary.

module dedup_proto {
  const POINTS: Set[int]
  const EPSILON: int

  // Position of each point on a 1-D line; |a - b| <= EPSILON means duplicate.
  var pos: int -> int
  var stored: Set[int]
  var assignedId: int -> int
  var nextId: int

  action init = all {
    pos' = POINTS.mapBy(p => p),
    stored' = Set(),
    assignedId' = POINTS.mapBy(_ => 0),
    nextId' = 1,
  }

  def dist(a: int, b: int): int =
    if (pos.get(a) >= pos.get(b)) pos.get(a) - pos.get(b)
    else pos.get(b) - pos.get(a)

  def isDuplicateOf(p: int, q: int): bool = dist(p, q) <= EPSILON

  /// dedup_check: is there a stored point within epsilon?
  def hasDuplicate(p: int): bool =
    stored.exists(q => q != p and isDuplicateOf(p, q))

  /// dedup_insert: check-then-insert, atomic from the caller's view.
  action insert(p: int): bool = all {
    not(stored.contains(p)),
    not(hasDuplicate(p)),
    stored' = stored.union(Set(p)),
    assignedId' = assignedId.set(p, nextId),
    nextId' = nextId + 1,
    pos' = pos,
  }

  /// The rejected branch: a duplicate is not stored and gets no new id.
  action insertRejected(p: int): bool = all {
    not(stored.contains(p)),
    hasDuplicate(p),
    stored' = stored, assignedId' = assignedId, nextId' = nextId, pos' = pos,
  }

  action step = any {
    nondet p = POINTS.oneOf()
    insert(p),
    nondet p = POINTS.oneOf()
    insertRejected(p),
  }

  // ---- invariants ---------------------------------------------------------

  /// No two stored points are within epsilon of each other. This is the
  /// property the index is supposed to maintain.
  val storedAreDistinct: bool =
    stored.forall(a => stored.forall(b =>
      a != b implies not(isDuplicateOf(a, b))))

  /// Point ids are unique across everything stored.
  val idsUnique: bool =
    stored.forall(a => stored.forall(b =>
      (a != b) implies assignedId.get(a) != assignedId.get(b)))

  /// Ids are never reused.
  val idsMonotone: bool = stored.forall(p => assignedId.get(p) < nextId)

  /// The duplicate relation is reflexive and symmetric...
  val dedupSymmetric: bool =
    POINTS.forall(a => POINTS.forall(b =>
      isDuplicateOf(a, b) == isDuplicateOf(b, a)))

  /// ...but NOT transitive: a~b and b~c does not give a~c. Recorded so nobody
  /// treats epsilon-dedup as clustering. Which points survive therefore depends
  /// on INSERTION ORDER, which `storedAreDistinct` still holds under.
  val dedupIsNotTransitive: bool =
    POINTS.exists(a => POINTS.exists(b => POINTS.exists(c => and {
      isDuplicateOf(a, b), isDuplicateOf(b, c), not(isDuplicateOf(a, c)),
    })))

  val safety: bool = and { storedAreDistinct, idsUnique, idsMonotone, dedupSymmetric }
}

module dedup {
  // Points at 1..4 with epsilon 1: 1~2, 2~3, 3~4, but 1 !~ 3.
  import dedup_proto(POINTS = Set(1, 2, 3, 4), EPSILON = 1).*

  /// Inserting 1 then 3 keeps both (distance 2 > epsilon); 2 is then rejected
  /// because it is within epsilon of BOTH.
  run orderDependentDedupTest =
    init
      .then(insert(1))
      .then(insert(3))
      .expect(and { stored == Set(1, 3), storedAreDistinct, idsUnique })
      .then(insertRejected(2))
      .expect(and { stored == Set(1, 3), storedAreDistinct })

  /// The tolerance relation is not transitive, so dedup is order-dependent.
  run nonTransitiveTest =
    init
      .expect(and { dedupSymmetric, dedupIsNotTransitive })
}
