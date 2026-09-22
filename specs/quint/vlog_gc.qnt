// -*- mode: quint -*-
//
// vlog_gc.qnt — model of the WiscKey value log and its garbage collection
// (src/storage/vlog.c, src/storage/value_store.c).
//
// Reclaiming a value-log region that some key still points at is silent data
// corruption -- the worst failure mode in the tree, and one fuzzing will
// essentially never reach. This models the whole put/delete/GC/reopen cycle.
//
// Fidelity points:
//   * `value_store_gc` (value_store.c:231) snapshots live offsets from slots
//     with state == 1 only. A TOMBSTONED slot (state == 2) contributes nothing,
//     so both its original PUT record and its DEL marker are dropped by GC.
//   * `vlog_gc` (vlog.c:207) copies live records to `<path>.gc`, fsyncs, then
//     rename()s over the original, then the caller remaps map offsets old->new.
//   * `value_store_open` (value_store.c:127) rebuilds the map by scanning the
//     log: `vs_rebuild_cb` (value_store.c:117) upserts on PUT and erases on DEL,
//     in log order. So the map is fully derivable from the log.
//   * `value_store_delete` appends the DEL marker BEFORE tombstoning the slot,
//     so a failed append leaves both log and map holding the key.
//
// The in-memory offset remap after GC is not load-bearing for correctness
// (a reopen rebuilds from the log regardless), but it IS load-bearing for a
// process that keeps running -- `mapResolvesAfterGc` covers that.

module vlog_gc_proto {
  type Record = {
    off: int,          // byte offset in the log; unique and increasing
    key: int,
    isDelete: bool
  }

  const KEYS: Set[int]
  const MAX_RECORDS: int

  // The durable log, in physical order.
  var log: List[Record]
  // The in-memory index: key -> offset. Only live (state == 1) keys appear.
  var index: int -> int
  var liveKeys: Set[int]
  // Keys whose slot is tombstoned (state == 2).
  var tombstoned: Set[int]
  var nextOff: int

  // What each key's value SHOULD be, tracked independently of the storage
  // layer so that reclamation can be checked against ground truth.
  var expected: Set[int]

  // ---- helpers ------------------------------------------------------------
  def offsetsInLog: Set[int] = log.foldl(Set(), (acc, r) => acc.union(Set(r.off)))

  /// vs_rebuild_cb (value_store.c:117): replay the log in order, upserting on
  /// PUT and erasing on DEL. This is exactly what a reopen reconstructs.
  def rebuiltKeys: Set[int] =
    log.foldl(Set(), (acc, r) =>
      if (r.isDelete) acc.exclude(Set(r.key)) else acc.union(Set(r.key)))

  // ---- init ---------------------------------------------------------------
  action init = all {
    log' = List(),
    index' = Map(),
    liveKeys' = Set(),
    tombstoned' = Set(),
    nextOff' = 1,
    expected' = Set(),
  }

  // ---- actions ------------------------------------------------------------

  /// value_store_put (value_store.c:142): append, then upsert the map.
  action putKey(k: int): bool = all {
    log.length() < MAX_RECORDS,
    log' = log.append({ off: nextOff, key: k, isDelete: false }),
    index' = index.put(k, nextOff),
    liveKeys' = liveKeys.union(Set(k)),
    tombstoned' = tombstoned.exclude(Set(k)),
    nextOff' = nextOff + 1,
    expected' = expected.union(Set(k)),
  }

  /// value_store_delete (value_store.c:189): append the DEL marker first, then
  /// tombstone the slot. The marker's own offset is not recorded anywhere.
  action del(k: int): bool = all {
    liveKeys.contains(k),
    log.length() < MAX_RECORDS,
    log' = log.append({ off: nextOff, key: k, isDelete: true }),
    liveKeys' = liveKeys.exclude(Set(k)),
    tombstoned' = tombstoned.union(Set(k)),
    index' = index,
    nextOff' = nextOff + 1,
    expected' = expected.exclude(Set(k)),
  }

  /// value_store_gc + vlog_gc: keep exactly the records whose offset is in the
  /// live set, then remap the surviving offsets.
  action gc = all {
    val liveOffs = liveKeys.map(k => index.get(k))
    val kept = log.select(r => liveOffs.contains(r.off))
    // Records are physically copied, so surviving offsets are REWRITTEN to
    // their new positions. The index must be remapped to match -- modelling
    // only one side of that is exactly the bug value_store_gc's remap loop
    // (value_store.c:249) exists to prevent.
    val renumbered = kept.foldl({ lst: List(), i: 0 }, (acc, r) =>
      { lst: acc.lst.append({ ...r, off: acc.i + 1 }), i: acc.i + 1 }).lst
    all {
      log' = renumbered,
      index' = liveKeys.mapBy(k =>
        kept.foldl({ i: 0, found: 0 }, (acc, r) =>
          if (r.off == index.get(k)) { i: acc.i + 1, found: acc.i + 1 }
          else { i: acc.i + 1, found: acc.found }).found),
      liveKeys' = liveKeys,
      // GC dropped every tombstoned key's records, so the tombstones are gone.
      tombstoned' = Set(),
      nextOff' = kept.length() + 1,
      expected' = expected,
    },
  }

  /// A crash after the rename but before anything else: the process restarts
  /// and value_store_open rebuilds the index purely from the log.
  action reopen = all {
    liveKeys' = rebuiltKeys,
    index' = rebuiltKeys.mapBy(k =>
      log.foldl(0, (acc, r) => if (r.key == k and not(r.isDelete)) r.off else acc)),
    tombstoned' = Set(),
    log' = log, nextOff' = nextOff, expected' = expected,
  }

  action step = any {
    gc,
    reopen,
    nondet k = KEYS.oneOf()
    putKey(k),
    nondet k = KEYS.oneOf()
    del(k),
  }

  // ---- invariants ---------------------------------------------------------

  /// GC never reclaims a record that a live key still points at.
  val noLiveReclaim: bool =
    liveKeys.forall(k => offsetsInLog.contains(index.get(k)))

  /// Every index entry resolves to a record actually present in the log, and it
  /// is a PUT for that key -- not a delete marker, not another key's record.
  val mapResolvesAfterGc: bool =
    liveKeys.forall(k =>
      log.foldl(false, (acc, r) =>
        acc or (r.off == index.get(k) and r.key == k and not(r.isDelete))))

  /// The set of keys a reopen would reconstruct equals the set the running
  /// process believes is live. If these diverge, a restart silently changes the
  /// visible data -- either resurrecting a deleted key or losing a live one.
  val rebuildAgreesWithIndex: bool = rebuiltKeys == liveKeys

  /// Storage agrees with ground truth.
  val noLostKey: bool = expected.forall(k => liveKeys.contains(k))
  val noResurrectedKey: bool = liveKeys.forall(k => expected.contains(k))

  val safety: bool = and {
    noLiveReclaim,
    mapResolvesAfterGc,
    rebuildAgreesWithIndex,
    noLostKey,
    noResurrectedKey,
  }
}

module vlog_gc {
  import vlog_gc_proto(KEYS = Set(1, 2), MAX_RECORDS = 5).*

  /// GC drops a tombstoned key's PUT record and its DEL marker together, so a
  /// subsequent reopen must not resurrect the key.
  run deleteThenGcThenReopenTest =
    init
      .then(putKey(1))
      .then(putKey(2))
      .then(del(1))
      .then(gc)
      .expect(and { liveKeys == Set(2), log.length() == 1 })
      .then(reopen)
      .expect(and {
        liveKeys == Set(2),          // key 1 stays deleted across the restart
        rebuildAgreesWithIndex,
        noResurrectedKey,
        noLostKey,
      })

  /// A live key survives compaction and still resolves afterwards.
  run liveKeySurvivesGcTest =
    init
      .then(putKey(1))
      .then(putKey(2))
      .then(del(2))
      .then(gc)
      .expect(and { noLiveReclaim, mapResolvesAfterGc, liveKeys == Set(1) })
}
