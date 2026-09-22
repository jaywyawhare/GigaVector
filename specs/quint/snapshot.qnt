// -*- mode: quint ---
//
// snapshot.qnt — model of snapshot and restore consistency
// (src/storage/snapshot.c, src/storage/backup.c).
//
// A snapshot must correspond to a single consistent cut of the write stream,
// not a smear across concurrent writes. Pairs with tests/corrupt_snapshot.sh,
// which supplies the byte-level adversary this model does not.

module snapshot_proto {
  const MAX_WRITES: int

  var committed: Set[int]           // writes durably in the database
  var nextId: int
  // A snapshot in progress: the cut it captured, and what it has copied.
  var snapCut: Set[int]
  var snapCopied: Set[int]
  var snapping: bool
  // Completed snapshot contents.
  var snapshotContents: Set[int]
  var haveSnapshot: bool
  var restored: Set[int]
  var haveRestored: bool
  // Which snapshot generation a restore came from. Without this, taking a NEW
  // snapshot after a restore would make the old, perfectly valid restore look
  // like a round-trip failure.
  var snapshotVersion: int
  var restoredVersion: int

  action init = all {
    committed' = Set(),
    nextId' = 1,
    snapCut' = Set(),
    snapCopied' = Set(),
    snapping' = false,
    snapshotContents' = Set(),
    haveSnapshot' = false,
    restored' = Set(),
    haveRestored' = false,
    snapshotVersion' = 0,
    restoredVersion' = -1,
  }

  action write = all {
    nextId <= MAX_WRITES,
    committed' = committed.union(Set(nextId)),
    nextId' = nextId + 1,
    snapCut' = snapCut, snapCopied' = snapCopied, snapping' = snapping,
    snapshotContents' = snapshotContents, haveSnapshot' = haveSnapshot,
    restored' = restored, haveRestored' = haveRestored,
    snapshotVersion' = snapshotVersion, restoredVersion' = restoredVersion,
  }

  /// Begin a snapshot: fix the cut.
  action snapBegin = all {
    not(snapping),
    snapCut' = committed,
    snapCopied' = Set(),
    snapping' = true,
    committed' = committed, nextId' = nextId,
    snapshotContents' = snapshotContents, haveSnapshot' = haveSnapshot,
    restored' = restored, haveRestored' = haveRestored,
    snapshotVersion' = snapshotVersion, restoredVersion' = restoredVersion,
  }

  /// Copy one record belonging to the cut. Writes that land after the cut are
  /// deliberately NOT copied -- that is what makes the snapshot a cut.
  action snapCopyOne(v: int): bool = all {
    snapping,
    snapCut.contains(v),
    not(snapCopied.contains(v)),
    snapCopied' = snapCopied.union(Set(v)),
    committed' = committed, nextId' = nextId, snapCut' = snapCut,
    snapping' = snapping, snapshotContents' = snapshotContents,
    haveSnapshot' = haveSnapshot, restored' = restored,
    haveRestored' = haveRestored,
    snapshotVersion' = snapshotVersion, restoredVersion' = restoredVersion,
  }

  action snapFinish = all {
    snapping,
    snapCopied == snapCut,
    snapshotContents' = snapCopied,
    haveSnapshot' = true,
    snapping' = false,
    snapCut' = Set(), snapCopied' = Set(),
    committed' = committed, nextId' = nextId,
    restored' = restored, haveRestored' = haveRestored,
    snapshotVersion' = snapshotVersion + 1, restoredVersion' = restoredVersion,
  }

  action restore = all {
    haveSnapshot,
    restored' = snapshotContents,
    haveRestored' = true,
    committed' = committed, nextId' = nextId, snapCut' = snapCut,
    snapCopied' = snapCopied, snapping' = snapping,
    snapshotContents' = snapshotContents, haveSnapshot' = haveSnapshot,
    snapshotVersion' = snapshotVersion, restoredVersion' = snapshotVersion,
  }

  action step = any {
    write, snapBegin, snapFinish, restore,
    nondet v = 1.to(MAX_WRITES).oneOf()
    snapCopyOne(v),
  }

  // ---- invariants ---------------------------------------------------------

  /// A finished snapshot contains only writes that had actually committed.
  val snapshotIsConsistentCut: bool =
    haveSnapshot implies snapshotContents.subseteq(committed)

  /// Restore reproduces the snapshot exactly -- no loss, no invention.
  val restoreRoundTrip: bool =
    (haveRestored and restoredVersion == snapshotVersion) implies
      restored == snapshotContents

  /// A restored database never contains a write that never happened.
  val noPhantomAfterRestore: bool =
    haveRestored implies restored.subseteq(committed)

  val safety: bool = and {
    snapshotIsConsistentCut, restoreRoundTrip, noPhantomAfterRestore,
  }
}

module snapshot {
  import snapshot_proto(MAX_WRITES = 3).*

  /// A write landing mid-snapshot is excluded from the cut, and the restore
  /// still round-trips exactly what was captured.
  run concurrentWriteDuringSnapshotTest =
    init
      .then(write)                       // write 1
      .then(snapBegin)                   // cut = {1}
      .then(write)                       // write 2 lands after the cut
      .then(snapCopyOne(1))
      .then(snapFinish)
      .expect(and {
        snapshotContents == Set(1),      // write 2 correctly excluded
        snapshotIsConsistentCut,
      })
      .then(restore)
      .expect(and { restored == Set(1), restoreRoundTrip, noPhantomAfterRestore })
}
