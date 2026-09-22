// -*- mode: quint ---
//
// shard_rebalance.qnt — model of shard ownership and vector migration
// (src/admin/shard.c).
//
// `migrate_vector_at_index` (shard.c:339) is COPY-THEN-DELETE:
//     db_add_vector_with_rich_metadata(to_db, ...)   // shard.c:385
//     ... copy metadata index entries ...
//     return db_delete_vector_by_index(from_db, idx) // shard.c:415, LAST
//
// Two consequences follow directly:
//   * During the window the vector exists on BOTH shards. A search that fans
//     out across shards can return it twice.
//   * If the final delete fails, the function returns -1 and the caller
//     (`shard_migrate_vector_at`, shard.c:437) reports the migration as failed
//     -- but the copy has already landed. The vector is now permanently on two
//     shards, and a retry duplicates it again.
//
// Copy-then-delete is the right order for durability (it never loses data);
// the gap is that failure is not distinguished from "nothing happened".

module shard_proto {
  const SHARDS: Set[int]
  const VECTORS: Set[int]

  // Which shards physically hold a copy of each vector.
  var holders: int -> Set[int]
  var shardState: int -> str          // "active" | "readonly" | "migrating" | "offline"
  var rebalancing: bool
  // Migrations that returned failure to the caller.
  var reportedFailed: Set[int]

  action init = all {
    holders' = VECTORS.mapBy(_ => Set(0)),   // every vector starts on shard 0
    shardState' = SHARDS.mapBy(_ => "active"),
    rebalancing' = false,
    reportedFailed' = Set(),
  }

  /// shard_set_state
  action setState(s: int, st: str): bool = all {
    shardState' = shardState.set(s, st),
    holders' = holders, rebalancing' = rebalancing,
    reportedFailed' = reportedFailed,
  }

  /// shard_rebalance_start (shard.c:491): refuses if already rebalancing.
  action rebalanceStart = all {
    not(rebalancing),
    rebalancing' = true,
    holders' = holders, shardState' = shardState,
    reportedFailed' = reportedFailed,
  }

  action rebalanceCancel = all {
    rebalancing,
    rebalancing' = false,
    holders' = holders, shardState' = shardState,
    reportedFailed' = reportedFailed,
  }

  /// Phase 1 of migrate_vector_at_index: the destination copy lands.
  action migrateCopy(v: int, from: int, dst: int): bool = all {
    from != dst,
    holders.get(v).contains(from),
    not(holders.get(v).contains(dst)),
    holders' = holders.set(v, holders.get(v).union(Set(dst))),
    shardState' = shardState, rebalancing' = rebalancing,
    reportedFailed' = reportedFailed,
  }

  /// Phase 2: the source delete succeeds; ownership transfers cleanly.
  action migrateDeleteOk(v: int, from: int): bool = all {
    holders.get(v).contains(from),
    holders.get(v).size() > 1,
    holders' = holders.set(v, holders.get(v).exclude(Set(from))),
    shardState' = shardState, rebalancing' = rebalancing,
    reportedFailed' = reportedFailed,
  }

  /// Phase 2 fails (shard.c:415 returns non-zero). The caller sees -1, but the
  /// destination copy stays. This is the leak.
  action migrateDeleteFail(v: int): bool = all {
    holders.get(v).size() > 1,
    reportedFailed' = reportedFailed.union(Set(v)),
    holders' = holders,
    shardState' = shardState, rebalancing' = rebalancing,
  }

  action step = any {
    rebalanceStart,
    rebalanceCancel,
    nondet v = VECTORS.oneOf()
    nondet a = SHARDS.oneOf()
    nondet b = SHARDS.oneOf()
    migrateCopy(v, a, b),
    nondet v = VECTORS.oneOf()
    nondet a = SHARDS.oneOf()
    migrateDeleteOk(v, a),
    nondet v = VECTORS.oneOf()
    migrateDeleteFail(v),
    nondet s = SHARDS.oneOf()
    setState(s, "readonly"),
  }

  // ---- invariants ---------------------------------------------------------

  /// No vector is ever lost: at least one shard always holds it.
  val noVectorLost: bool = VECTORS.forall(v => holders.get(v).size() >= 1)

  /// Each vector is owned by exactly one shard. Transiently false during any
  /// copy-then-delete migration -- that part is by design -- and PERMANENTLY
  /// false once a migration reports failure after its copy landed.
  val exactlyOneOwner: bool = VECTORS.forall(v => holders.get(v).size() == 1)

  /// A migration reported as failed left no trace. This is the property that
  /// distinguishes "failed cleanly" from "failed after copying".
  val failedMigrationIsClean: bool =
    reportedFailed.forall(v => holders.get(v).size() == 1)

  /// A cross-shard scan cannot return the same vector twice.
  val noDuplicateInScan: bool = exactlyOneOwner

  val safety: bool = noVectorLost
}

module shard_rebalance {
  import shard_proto(SHARDS = Set(0, 1), VECTORS = Set(1)).*

  /// A migration whose delete fails leaves the vector on both shards, while
  /// reporting failure to the caller. A retry would copy it a third time.
  run failedMigrationDuplicatesTest =
    init
      .expect(exactlyOneOwner)
      .then(migrateCopy(1, 0, 1))
      .expect(and {
        holders.get(1) == Set(0, 1),
        not(exactlyOneOwner),            // transient double-ownership, by design
        noVectorLost,
      })
      .then(migrateDeleteFail(1))
      .expect(and {
        not(failedMigrationIsClean),     // reported failed, but the copy remains
        holders.get(1) == Set(0, 1),
        noVectorLost,                    // data is never lost -- only duplicated
      })

  /// The happy path transfers ownership exactly.
  run cleanMigrationTest =
    init
      .then(migrateCopy(1, 0, 1))
      .then(migrateDeleteOk(1, 0))
      .expect(and { holders.get(1) == Set(1), exactlyOneOwner, noVectorLost })

  /// shard_rebalance_start refuses a second concurrent rebalance.
  run rebalanceMutualExclusionTest =
    init
      .then(rebalanceStart)
      .expect(rebalancing)
      .then(rebalanceCancel)
      .expect(not(rebalancing))
}
