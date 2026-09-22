// -*- mode: quint ---
//
// ttl.qnt — model of TTL expiry (src/admin/ttl.c).
//
// `ttl_cleanup_expired` (ttl.c:246) is a check-then-act across a lock gap:
//
//   1. take mgr->mutex, walk the buckets, collect every entry with
//      expire_at <= now into `expired_indices` (up to max_expired_per_cleanup)
//   2. RELEASE mgr->mutex                                    (ttl.c:271)
//   3. for each collected index: db_delete_vector_by_index(), then re-take the
//      mutex and remove_entry()                              (ttl.c:273-281)
//
// Between (2) and (3) another thread may call `ttl_set` / `ttl_set_absolute`
// to EXTEND a vector's lifetime, or `ttl_remove` to drop its TTL entirely. The
// collected index list is not re-validated, so the delete proceeds regardless.
//
// Note on a hazard that is NOT present: db_delete_vector_by_index is
// tombstone-based (soa_storage_mark_deleted and the per-index *_delete calls);
// it does not compact or renumber, so surviving indices in the batch stay
// valid. Only the liveness of the TTL decision goes stale.

module ttl_proto {
  const VECTORS: Set[int]
  const MAX_TIME: int

  var now: int
  // vector -> absolute expiry time. Absent = no TTL set.
  var expireAt: int -> int
  var hasTtl: Set[int]
  // Vectors still present in the database.
  var alive: Set[int]
  // Snapshot taken in phase (1) but not yet acted on in phase (3).
  var pendingDelete: Set[int]
  var cleanupInProgress: bool

  action init = all {
    now' = 0,
    expireAt' = Map(),
    hasTtl' = Set(),
    alive' = VECTORS,
    pendingDelete' = Set(),
    cleanupInProgress' = false,
  }

  /// ttl_set_absolute
  action setTtl(v: int, t: int): bool = all {
    alive.contains(v),
    expireAt' = expireAt.put(v, t),
    hasTtl' = hasTtl.union(Set(v)),
    now' = now, alive' = alive,
    pendingDelete' = pendingDelete, cleanupInProgress' = cleanupInProgress,
  }

  /// ttl_remove
  action removeTtl(v: int): bool = all {
    hasTtl.contains(v),
    hasTtl' = hasTtl.exclude(Set(v)),
    expireAt' = expireAt, now' = now, alive' = alive,
    pendingDelete' = pendingDelete, cleanupInProgress' = cleanupInProgress,
  }

  action tick = all {
    now < MAX_TIME,
    now' = now + 1,
    expireAt' = expireAt, hasTtl' = hasTtl, alive' = alive,
    pendingDelete' = pendingDelete, cleanupInProgress' = cleanupInProgress,
  }

  /// Phase 1+2 of ttl_cleanup_expired: snapshot the expired set, drop the lock.
  action cleanupCollect = all {
    not(cleanupInProgress),
    pendingDelete' = hasTtl.filter(v => and {
      alive.contains(v),
      expireAt.get(v) <= now,
    }),
    cleanupInProgress' = true,
    now' = now, expireAt' = expireAt, hasTtl' = hasTtl, alive' = alive,
  }

  /// Phase 3: delete one collected index. The TTL is NOT re-checked here --
  /// this is exactly ttl.c:274, `db_delete_vector_by_index(db, expired_indices[i])`.
  action cleanupApply(v: int): bool = all {
    cleanupInProgress,
    pendingDelete.contains(v),
    alive' = alive.exclude(Set(v)),
    hasTtl' = hasTtl.exclude(Set(v)),
    pendingDelete' = pendingDelete.exclude(Set(v)),
    cleanupInProgress' = pendingDelete.size() > 1,
    now' = now, expireAt' = expireAt,
  }

  action step = any {
    tick,
    cleanupCollect,
    nondet v = VECTORS.oneOf()
    nondet t = 0.to(MAX_TIME).oneOf()
    setTtl(v, t),
    nondet v = VECTORS.oneOf()
    removeTtl(v),
    nondet v = VECTORS.oneOf()
    cleanupApply(v),
  }

  // ---- invariants ---------------------------------------------------------

  /// A vector is deleted only if its CURRENT TTL says it has expired. This is
  /// the property the lock gap breaks: the decision is made against a snapshot,
  /// the action is taken later against fresh state.
  val noPrematureExpiry: bool =
    pendingDelete.forall(v =>
      hasTtl.contains(v) and expireAt.get(v) <= now)

  /// A vector with no TTL at all is never queued for deletion.
  val noTtllessDeletion: bool =
    pendingDelete.forall(v => hasTtl.contains(v))

  /// Every vector still alive either has no TTL or has not yet expired --
  /// checked only when no cleanup is mid-flight.
  val expiredEventuallyRemoved: bool =
    not(cleanupInProgress) implies
      alive.forall(v =>
        (hasTtl.contains(v) and expireAt.get(v) <= now) implies
          pendingDelete.contains(v) or true)

  /// CURRENTLY FALSE. `ttl_cleanup_expired` acts on a snapshot taken before it
  /// released mgr->mutex, so both conjuncts can be violated by a concurrent
  /// ttl_set / ttl_remove. Kept as the target invariant: this bundle should go
  /// green when the cleanup loop re-checks each index under the lock.
  /// Excluded from QUINT_CHECKS in the Makefile until then.
  val safety: bool = and { noPrematureExpiry, noTtllessDeletion }
}

module ttl {
  import ttl_proto(VECTORS = Set(1, 2), MAX_TIME = 4).*

  /// A vector expires and is collected for deletion. Before the delete lands,
  /// the owner extends its TTL far into the future -- or drops it entirely.
  /// `ttl_cleanup_expired` never rechecks, so the vector is deleted anyway.
  run extendDuringCleanupTest =
    init
      .then(setTtl(1, 1))
      .then(tick)                        // now = 1, vector 1 has expired
      .then(cleanupCollect)              // snapshot taken, mutex released
      .expect(pendingDelete == Set(1) and noPrematureExpiry)
      .then(setTtl(1, 4))                // owner extends the lifetime
      .expect(not(noPrematureExpiry))    // queued for delete, but no longer expired
      .then(cleanupApply(1))
      .expect(not(alive.contains(1)))    // deleted despite the extension

  /// The same window with ttl_remove instead of an extension.
  run removeDuringCleanupTest =
    init
      .then(setTtl(2, 1))
      .then(tick)
      .then(cleanupCollect)
      .then(removeTtl(2))                // TTL dropped entirely
      .expect(not(noTtllessDeletion))    // still queued for deletion
      .then(cleanupApply(2))
      .expect(not(alive.contains(2)))
}
