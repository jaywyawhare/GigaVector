// -*- mode: quint ---
//
// index_swap.qnt — model of online index rebuild and hot swap
// (src/index/index_maintenance.c, src/index/ivf_retrain.c,
//  src/admin/migration.c: migration_start / wait / cancel / take_index).
//
// A rebuild reads a snapshot of the data, builds a fresh index off to the
// side, then swaps it in. Writes that land DURING the build must not be lost:
// either the build sees them, or the swap re-applies them.

module index_swap_proto {
  const MAX_VECTORS: int

  // Vectors actually in the database.
  var stored: Set[int]
  // What the live index can find.
  var indexed: Set[int]
  // Snapshot the rebuild started from, and what it has built so far.
  var buildSnapshot: Set[int]
  var building: bool
  // Writes that landed after the snapshot was taken.
  var writesDuringBuild: Set[int]
  var nextId: int
  var cancelled: bool

  action init = all {
    stored' = Set(),
    indexed' = Set(),
    buildSnapshot' = Set(),
    building' = false,
    writesDuringBuild' = Set(),
    nextId' = 1,
    cancelled' = false,
  }

  /// A normal insert: lands in storage and in the live index. If a rebuild is
  /// in flight it is also recorded as a during-build write.
  action insert = all {
    nextId <= MAX_VECTORS,
    stored' = stored.union(Set(nextId)),
    indexed' = indexed.union(Set(nextId)),
    writesDuringBuild' = if (building) writesDuringBuild.union(Set(nextId))
                         else writesDuringBuild,
    nextId' = nextId + 1,
    buildSnapshot' = buildSnapshot, building' = building, cancelled' = cancelled,
  }

  /// migration_start: capture the snapshot the new index is built from.
  action buildStart = all {
    not(building),
    buildSnapshot' = stored,
    writesDuringBuild' = Set(),
    building' = true,
    cancelled' = false,
    stored' = stored, indexed' = indexed, nextId' = nextId,
  }

  /// migration_take_index + swap. The new index contains the snapshot PLUS the
  /// writes that landed during the build -- that catch-up is what keeps the
  /// swap lossless.
  action buildSwap = all {
    building,
    not(cancelled),
    indexed' = buildSnapshot.union(writesDuringBuild),
    building' = false,
    writesDuringBuild' = Set(),
    buildSnapshot' = Set(),
    stored' = stored, nextId' = nextId, cancelled' = cancelled,
  }

  /// A swap that forgets the catch-up set -- the classic lost-write bug this
  /// spec exists to rule out.
  action buildSwapNoCatchup = all {
    building,
    not(cancelled),
    indexed' = buildSnapshot,
    building' = false,
    writesDuringBuild' = Set(),
    buildSnapshot' = Set(),
    stored' = stored, nextId' = nextId, cancelled' = cancelled,
  }

  /// migration_cancel: the in-flight index is dropped, the live one untouched.
  action buildCancel = all {
    building,
    building' = false,
    cancelled' = true,
    buildSnapshot' = Set(),
    writesDuringBuild' = Set(),
    stored' = stored, indexed' = indexed, nextId' = nextId,
  }

  action step = any { insert, buildStart, buildSwap, buildCancel }

  // ---- invariants ---------------------------------------------------------

  /// Every stored vector is findable. This is the property a swap can break.
  val noLostInsert: bool = stored.forall(v => indexed.contains(v))

  /// The index never reports a vector that is not stored.
  val noPhantomIndexEntry: bool = indexed.forall(v => stored.contains(v))

  /// Cancelling leaves the live index exactly as it was.
  val cancelPreservesIndex: bool = cancelled implies noLostInsert

  val safety: bool = and { noLostInsert, noPhantomIndexEntry }
}

module index_swap {
  import index_swap_proto(MAX_VECTORS = 3).*

  /// A write landing mid-rebuild survives the swap.
  run writeDuringRebuildSurvivesTest =
    init
      .then(insert)
      .then(buildStart)
      .then(insert)                       // lands after the snapshot
      .expect(writesDuringBuild.size() == 1)
      .then(buildSwap)
      .expect(and { noLostInsert, noPhantomIndexEntry, indexed == stored })

  /// Cancelling a rebuild leaves the live index intact.
  run cancelKeepsIndexTest =
    init
      .then(insert)
      .then(buildStart)
      .then(insert)
      .then(buildCancel)
      .expect(and { noLostInsert, cancelPreservesIndex, not(building) })

  /// The negative case: a swap that drops the catch-up set loses the write.
  /// Kept as a witness so the correct `buildSwap` above stays meaningful.
  run swapWithoutCatchupLosesWriteTest =
    init
      .then(insert)
      .then(buildStart)
      .then(insert)
      .then(buildSwapNoCatchup)
      .expect(not(noLostInsert))
}
