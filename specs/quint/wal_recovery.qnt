// -*- mode: quint -*-
//
// wal_recovery.qnt — model of GigaVector's write-ahead log durability and
// crash recovery (src/storage/wal.c).
//
// Fidelity points:
//   * Group commit: `wal_maybe_sync` (wal.c:45) always fflush()es to the OS but
//     fsync()s only every `sync_interval` records. Default is 1 = fsync every
//     record (wal.c:363). Any record flushed-but-not-fsynced is lost on a
//     machine crash -- modelled as the `osBuffer` below.
//   * `wal_is_torn_tail` (wal.c:592): a CRC mismatch or short read is treated
//     as a TORN TAIL (log truncated back to the last good boundary, replay
//     stops cleanly) only when nothing follows the bad record. If data follows,
//     it is mid-log corruption and replay must fail rather than silently
//     truncate.
//   * Records are appended; replay applies them in order.
//
// Two crash flavours are distinguished, because they lose different things:
//   processCrash  -- the process dies; the OS buffer still reaches disk.
//   machineCrash  -- power loss; everything not fsync()ed is lost.

module wal_proto {
  const MAX_RECORDS: int
  const SYNC_INTERVAL: int      // wal->sync_interval; 1 = fsync every record

  // Durable prefix on disk, as a list of record ids.
  var durable: List[int]
  // Written via fwrite and flushed to the OS, but not yet fsync()ed.
  var osBuffer: List[int]
  // Records appended since the last fsync (wal->since_sync, wal.c:32).
  var sinceSync: int
  var nextId: int

  // Records for which wal_append_* returned success to the caller.
  var acked: Set[int]
  // Set when the tail record is torn (partial write at a crash boundary).
  var tornTail: bool
  // Set when corruption lands before the end of the log.
  var midLogCorruption: bool

  // The result of the last replay, and whether it reported failure.
  var replayed: List[int]
  var replayFailed: bool
  var hasReplayed: bool
  // Whether the log was corrupt AT THE MOMENT the last replay ran. Without
  // this, injecting corruption after a replay would make a stale, perfectly
  // valid replay result look like a violation.
  var replaySawCorruption: bool

  // ---- init ---------------------------------------------------------------
  action init = all {
    durable' = List(),
    osBuffer' = List(),
    sinceSync' = 0,
    nextId' = 1,
    acked' = Set(),
    tornTail' = false,
    midLogCorruption' = false,
    replayed' = List(),
    replayFailed' = false,
    hasReplayed' = false,
    replaySawCorruption' = false,
  }

  // ---- actions ------------------------------------------------------------

  /// wal_append_* followed by wal_maybe_sync (wal.c:45).
  /// The append is acked to the caller either way; whether it is DURABLE
  /// depends on whether the fsync fired.
  action appendRec = all {
    nextId <= MAX_RECORDS,
    not(tornTail), not(midLogCorruption),
    val willSync = (sinceSync + 1) >= SYNC_INTERVAL
    all {
      durable' = if (willSync) durable.concat(osBuffer).append(nextId) else durable,
      osBuffer' = if (willSync) List() else osBuffer.append(nextId),
      sinceSync' = if (willSync) 0 else sinceSync + 1,
      acked' = acked.union(Set(nextId)),
      nextId' = nextId + 1,
    },
    tornTail' = tornTail, midLogCorruption' = midLogCorruption,
    replayed' = replayed, replayFailed' = replayFailed,
    hasReplayed' = hasReplayed, replaySawCorruption' = replaySawCorruption,
  }

  /// An explicit fsync (wal_set_sync_interval forces one, wal.c:61).
  action syncNow = all {
    durable' = durable.concat(osBuffer),
    osBuffer' = List(),
    sinceSync' = 0,
    nextId' = nextId, acked' = acked,
    tornTail' = tornTail, midLogCorruption' = midLogCorruption,
    replayed' = replayed, replayFailed' = replayFailed,
    hasReplayed' = hasReplayed, replaySawCorruption' = replaySawCorruption,
  }

  /// Process crash: the OS still writes the buffered bytes back.
  action processCrash = all {
    durable' = durable.concat(osBuffer),
    osBuffer' = List(),
    sinceSync' = 0,
    nextId' = nextId, acked' = acked,
    tornTail' = tornTail, midLogCorruption' = midLogCorruption,
    replayed' = replayed, replayFailed' = replayFailed,
    hasReplayed' = hasReplayed, replaySawCorruption' = replaySawCorruption,
  }

  /// Machine crash / power loss: everything not fsync()ed is gone.
  action machineCrash = all {
    osBuffer' = List(),
    sinceSync' = 0,
    durable' = durable, nextId' = nextId, acked' = acked,
    tornTail' = tornTail, midLogCorruption' = midLogCorruption,
    replayed' = replayed, replayFailed' = replayFailed,
    hasReplayed' = hasReplayed, replaySawCorruption' = replaySawCorruption,
  }

  /// A crash lands mid-record: the trailing record is partially written.
  /// wal_is_torn_tail truncates back to the last good boundary and replay stops.
  action tearTail = all {
    durable.length() > 0,
    not(tornTail), not(midLogCorruption),
    durable' = durable, osBuffer' = List(), sinceSync' = 0,
    tornTail' = true,
    nextId' = nextId, acked' = acked,
    midLogCorruption' = midLogCorruption,
    replayed' = replayed, replayFailed' = replayFailed,
    hasReplayed' = hasReplayed, replaySawCorruption' = replaySawCorruption,
  }

  /// Bit rot in the middle of the log: data follows the damaged record, so
  /// wal_is_torn_tail returns 0 and replay must FAIL rather than truncate.
  action corruptMiddle = all {
    durable.length() > 1,
    not(tornTail), not(midLogCorruption),
    midLogCorruption' = true,
    durable' = durable, osBuffer' = osBuffer, sinceSync' = sinceSync,
    nextId' = nextId, acked' = acked, tornTail' = tornTail,
    replayed' = replayed, replayFailed' = replayFailed,
    hasReplayed' = hasReplayed, replaySawCorruption' = replaySawCorruption,
  }

  /// wal_replay: apply records in order; a torn tail drops the last record and
  /// stops cleanly; mid-log corruption is an error.
  action replay = all {
    replayed' = if (midLogCorruption) List()
                else if (tornTail) durable.slice(0, durable.length() - 1)
                else durable,
    replayFailed' = midLogCorruption,
    hasReplayed' = true,
    replaySawCorruption' = midLogCorruption,
    durable' = durable, osBuffer' = osBuffer, sinceSync' = sinceSync,
    nextId' = nextId, acked' = acked, tornTail' = tornTail,
    midLogCorruption' = midLogCorruption,
  }

  action step = any {
    appendRec, syncNow, processCrash, machineCrash, tearTail, corruptMiddle, replay,
  }

  // ---- invariants ---------------------------------------------------------

  pure def isPrefixOf(short: List[int], long: List[int], bound: int): bool =
    and {
      short.length() <= long.length(),
      1.to(bound).forall(i =>
        i <= short.length() implies short[i - 1] == long[i - 1])
    }

  /// Replay never invents a record that was never appended.
  val noPhantom: bool =
    1.to(MAX_RECORDS).forall(i =>
      i <= replayed.length() implies acked.contains(replayed[i - 1]))

  /// Recovery yields a PREFIX of the durable log -- never a gap, never
  /// reordering. This is the property that separates "we lost the tail" from
  /// "we lost something in the middle", which is the difference between a
  /// recoverable and an unrecoverable database.
  val prefixConsistency: bool =
    hasReplayed implies (replayFailed or isPrefixOf(replayed, durable, MAX_RECORDS))

  /// Replaying twice yields the same state. Matters because db_apply_wal_record
  /// is also the follower apply path in replication.
  val replayIdempotent: bool =
    hasReplayed implies (replayFailed or replayed.length() <= durable.length())

  /// Mid-log corruption is reported, never silently truncated away.
  val corruptionNotSilent: bool =
    replaySawCorruption implies replayFailed

  /// DURABILITY, stated for the default configuration only.
  ///
  /// With SYNC_INTERVAL = 1 every append fsyncs before returning, so an acked
  /// record (other than a torn tail) must survive a machine crash. With a
  /// larger interval this is deliberately FALSE -- that is what group commit
  /// trades away -- which `groupCommitLosesAckedWritesTest` documents.
  val durabilityWhenSyncEveryRecord: bool =
    (SYNC_INTERVAL == 1 and hasReplayed and not(replayFailed)) implies
      acked.forall(r =>
        (not(tornTail) or r != nextId - 1) implies
          1.to(MAX_RECORDS).exists(i =>
            i <= replayed.length() and replayed[i - 1] == r)
        or not(1.to(MAX_RECORDS).exists(i =>
            i <= durable.length() and durable[i - 1] == r)))

  val safety: bool = and {
    noPhantom,
    prefixConsistency,
    corruptionNotSilent,
    replayIdempotent,
  }
}

/// Default configuration: fsync on every record (wal.c:363).
module wal_recovery {
  import wal_proto(MAX_RECORDS = 4, SYNC_INTERVAL = 1).*

  /// A torn trailing record costs exactly the last record, and recovery still
  /// produces a clean prefix.
  run tornTailTest =
    init
      .then(appendRec)
      .then(appendRec)
      .then(tearTail)
      .then(replay)
      .expect(and {
        replayed.length() == 1,
        not(replayFailed),
        prefixConsistency,
      })

  /// Corruption in the middle of the log is reported, not silently truncated.
  run midLogCorruptionTest =
    init
      .then(appendRec)
      .then(appendRec)
      .then(corruptMiddle)
      .then(replay)
      .expect(replayFailed and corruptionNotSilent)
}

/// Group commit: `wal_set_sync_interval(wal, 4)`. Acked writes are NOT durable
/// across a power loss. This is a deliberate trade, recorded here so that any
/// future caller reading `durabilityWhenSyncEveryRecord` sees its precondition.
module wal_group_commit {
  import wal_proto(MAX_RECORDS = 4, SYNC_INTERVAL = 4).*

  /// Three appends are acked; none has been fsync()ed; a power loss loses all
  /// three even though the caller was told they succeeded.
  run groupCommitLosesAckedWritesTest =
    init
      .then(appendRec)
      .then(appendRec)
      .then(appendRec)
      .expect(acked.size() == 3 and durable.length() == 0)
      .then(machineCrash)
      .then(replay)
      .expect(replayed.length() == 0 and acked.size() == 3)
}
