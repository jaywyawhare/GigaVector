// -*- mode: quint ---
//
// bulk_import.qnt — model of batch ingestion
// (src/storage/bulk_import.c: db_import_csv / db_import_jsonl, and
//  src/storage/document_ingest.c).
//
// Both importers walk a file and insert row by row, filling a GV_ImportReport
// with counts. They are NOT transactional: a malformed row partway through
// leaves the rows before it committed. The property that matters is therefore
// not atomicity but ACCURATE REPORTING -- the caller must be able to tell
// exactly what landed, so a retry does not double-insert.

module bulk_proto {
  const ROWS: int

  var cursor: int             // next row to process
  var inserted: Set[int]
  var reportOk: int
  var reportFailed: int
  var aborted: bool
  // Rows that are malformed and will fail.
  var badRows: Set[int]

  action init = all {
    cursor' = 1,
    inserted' = Set(),
    reportOk' = 0,
    reportFailed' = 0,
    aborted' = false,
    badRows' = Set(2),        // row 2 is malformed
  }

  /// Process one good row: insert and count it.
  action importGood = all {
    not(aborted),
    cursor <= ROWS,
    not(badRows.contains(cursor)),
    inserted' = inserted.union(Set(cursor)),
    reportOk' = reportOk + 1,
    cursor' = cursor + 1,
    reportFailed' = reportFailed, aborted' = aborted, badRows' = badRows,
  }

  /// A malformed row: counted as failed, import continues (skip-and-continue).
  action importBadSkip = all {
    not(aborted),
    cursor <= ROWS,
    badRows.contains(cursor),
    reportFailed' = reportFailed + 1,
    cursor' = cursor + 1,
    inserted' = inserted, reportOk' = reportOk, aborted' = aborted,
    badRows' = badRows,
  }

  /// A malformed row that aborts the whole import. Rows already inserted stay.
  action importBadAbort = all {
    not(aborted),
    cursor <= ROWS,
    badRows.contains(cursor),
    aborted' = true,
    reportFailed' = reportFailed + 1,
    cursor' = cursor, inserted' = inserted, reportOk' = reportOk,
    badRows' = badRows,
  }

  /// The import is finished: every row processed, or aborted. Modelled as an
  /// explicit self-loop because Apalache reports a state with no enabled
  /// action as a DEADLOCK -- correct for a server, wrong for a batch job that
  /// is simply done. Without this the bounded check flags termination as a
  /// counterexample.
  action finished = all {
    aborted or cursor > ROWS,
    cursor' = cursor, inserted' = inserted, reportOk' = reportOk,
    reportFailed' = reportFailed, aborted' = aborted, badRows' = badRows,
  }

  action step = any { importGood, importBadSkip, importBadAbort, finished }

  // ---- invariants ---------------------------------------------------------

  /// The report's success count matches exactly what is in the database. This
  /// is what makes a retry safe: the caller knows where to resume.
  val reportMatchesReality: bool = reportOk == inserted.size()

  /// Every row is accounted for exactly once: processed = ok + failed.
  val everyRowAccounted: bool =
    (cursor - 1) == reportOk + reportFailed or aborted

  /// An aborted import still leaves the rows it already committed -- it is not
  /// atomic, and the report says so.
  val abortLeavesPrefix: bool =
    aborted implies inserted.forall(r => r < cursor)

  /// Nothing is inserted that was not a row in the file.
  val noPhantomRow: bool = inserted.forall(r => r >= 1 and r <= ROWS)

  val safety: bool = and {
    reportMatchesReality, everyRowAccounted, abortLeavesPrefix, noPhantomRow,
  }

  // ---- inductive invariant -------------------------------------------------
  val typeOk: bool = and {
    cursor.in(1.to(ROWS + 1)),
    inserted.in(1.to(ROWS).powerset()),
    reportOk.in(0.to(ROWS)),
    reportFailed.in(0.to(ROWS)),
    aborted.in(Set(false, true)),
    badRows.in(1.to(ROWS).powerset()),
  }

  /// Every inserted row is strictly behind the cursor, and the bad-row set is
  /// fixed -- both needed for the report/reality agreement to be preserved.
  val indInv: bool = and {
    typeOk,
    reportMatchesReality,
    abortLeavesPrefix,
    noPhantomRow,
    badRows == Set(2),
    inserted.forall(r => r < cursor),
    // Row accounting: everything before the cursor was counted exactly once,
    // and the good rows before the cursor are exactly the inserted ones. Both
    // are required to make the conjunction closed under `step`.
    // importBadAbort counts the offending row but does NOT advance the cursor,
    // so the accounting relation differs by one in the aborted case.
    reportOk + reportFailed == (if (aborted) cursor else cursor - 1),
    1.to(ROWS).forall(r =>
      (r < cursor and not(badRows.contains(r))) implies inserted.contains(r)),
    1.to(ROWS).forall(r => badRows.contains(r) implies not(inserted.contains(r))),
  }
}

module bulk_import {
  import bulk_proto(ROWS = 4).*

  /// Skip-and-continue: the bad row is counted as failed, the rest land, and
  /// the report exactly describes the database.
  run skipBadRowTest =
    init
      .then(importGood)        // row 1
      .then(importBadSkip)     // row 2 malformed
      .then(importGood)        // row 3
      .then(importGood)        // row 4
      .expect(and {
        inserted == Set(1, 3, 4),
        reportOk == 3, reportFailed == 1,
        reportMatchesReality, everyRowAccounted, noPhantomRow,
      })

  /// Abort-on-error is NOT a rollback: row 1 stays committed. The report still
  /// tells the caller exactly how far it got.
  run abortLeavesPrefixTest =
    init
      .then(importGood)        // row 1 committed
      .then(importBadAbort)    // row 2 aborts
      .expect(and {
        inserted == Set(1),
        aborted,
        reportMatchesReality,  // report says 1 succeeded, and 1 did
        abortLeavesPrefix,
      })
}
