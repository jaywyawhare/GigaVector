// -*- mode: quint ---
//
// versioning.qnt — model of vector versioning and time travel
// (src/admin/versioning.c, src/admin/timetravel.c).
//
// `version_manager_create(max_versions)` bounds retention, so old versions are
// evicted. Time-travel reads name a version id; the hazard is a read for a
// version that retention has already dropped, and a read that silently falls
// back to a DIFFERENT version instead of failing.

module versioning_proto {
  const MAX_VERSIONS: int
  const MAX_CREATES: int

  var versions: Set[int]          // retained version ids
  var nextId: int
  var created: Set[int]           // every id ever created
  // Time-travel reads: (requested version, version actually served).
  var reads: Set[{ asked: int, served: int }]

  action init = all {
    versions' = Set(),
    nextId' = 1,
    created' = Set(),
    reads' = Set(),
  }

  /// version_create; evicts the oldest once max_versions is reached.
  action versionCreate = all {
    nextId <= MAX_CREATES,
    val kept = versions.union(Set(nextId))
    val evicted = if (kept.size() > MAX_VERSIONS)
        kept.exclude(Set(kept.fold(nextId, (a, b) => if (b < a) b else a)))
      else kept
    all {
      versions' = evicted,
      created' = created.union(Set(nextId)),
      nextId' = nextId + 1,
    },
    reads' = reads,
  }

  action versionDelete(v: int): bool = all {
    versions.contains(v),
    versions' = versions.exclude(Set(v)),
    nextId' = nextId, created' = created, reads' = reads,
  }

  /// A time-travel read of a retained version: served exactly.
  action readAt(v: int): bool = all {
    versions.contains(v),
    reads' = reads.union(Set({ asked: v, served: v })),
    versions' = versions, nextId' = nextId, created' = created,
  }

  /// A read of an evicted version must FAIL, not silently serve a neighbour.
  /// Modelled as served = 0 (an explicit error).
  action readEvicted(v: int): bool = all {
    created.contains(v),
    not(versions.contains(v)),
    reads' = reads.union(Set({ asked: v, served: 0 })),
    versions' = versions, nextId' = nextId, created' = created,
  }

  action step = any {
    versionCreate,
    nondet v = 1.to(MAX_CREATES).oneOf()
    versionDelete(v),
    nondet v = 1.to(MAX_CREATES).oneOf()
    readAt(v),
    nondet v = 1.to(MAX_CREATES).oneOf()
    readEvicted(v),
  }

  // ---- invariants ---------------------------------------------------------

  /// Retention is honoured: never more than max_versions retained.
  val retentionBounded: bool = versions.size() <= MAX_VERSIONS

  /// Version ids are only ever handed out once and increase.
  val idsMonotone: bool = created.forall(v => v < nextId)

  /// A time-travel read either serves exactly the version asked for, or fails.
  /// It never silently substitutes a different version.
  val noSilentSubstitution: bool =
    reads.forall(r => r.served == r.asked or r.served == 0)

  /// Every retained version was actually created.
  val retainedWereCreated: bool = versions.forall(v => created.contains(v))

  val safety: bool = and {
    retentionBounded, idsMonotone, noSilentSubstitution, retainedWereCreated,
  }

  // ---- inductive invariant -------------------------------------------------
  pure val READ_DOMAIN: Set[{ asked: int, served: int }] =
    1.to(MAX_CREATES).map(a =>
      0.to(MAX_CREATES).map(sv => { asked: a, served: sv })).flatten()

  val typeOk: bool = and {
    versions.in(1.to(MAX_CREATES).powerset()),
    created.in(1.to(MAX_CREATES).powerset()),
    nextId.in(1.to(MAX_CREATES + 1)),
    reads.in(READ_DOMAIN.powerset()),
  }

  /// Retained versions must also have been created, and every created id must
  /// be below nextId -- both are needed to keep the conjunction closed.
  val indInv: bool = and {
    typeOk,
    retentionBounded,
    idsMonotone,
    noSilentSubstitution,
    retainedWereCreated,
    versions.subseteq(created),
  }
}

module versioning {
  import versioning_proto(MAX_VERSIONS = 2, MAX_CREATES = 4).*

  /// Creating past the retention bound evicts the oldest, and a read of the
  /// evicted version fails rather than serving a neighbour.
  run retentionEvictionTest =
    init
      .then(versionCreate)
      .then(versionCreate)
      .then(versionCreate)
      .expect(and { retentionBounded, versions == Set(2, 3) })
      .then(readEvicted(1))
      .expect(and { noSilentSubstitution, retentionBounded })
}
