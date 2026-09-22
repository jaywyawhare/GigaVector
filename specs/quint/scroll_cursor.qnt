// -*- mode: quint ---
//
// scroll_cursor.qnt — model of scroll/pagination cursors
// (src/storage/scroll_cursor.c: gv_db_cursor_open / next / close).
//
// A cursor walks results in pages. The classic pagination hazards are: a row
// inserted before the cursor position shifts everything and causes a SKIP, and
// a row deleted behind it causes a DUPLICATE. Which of these can happen depends
// on whether the cursor is offset-based or keyset-based.
//
// GigaVector's deletes are tombstone-based (indices are stable), which is what
// makes a stable cursor achievable here at all.

module scroll_proto {
  const MAX_ITEMS: int
  const PAGE: int

  // Items present, by stable index. Tombstoned items stay indexed.
  var live: Set[int]
  var nextIdx: int
  // Cursor position: the highest index already returned.
  var cursorAt: int
  var open: bool
  // Everything the cursor has handed back, in order.
  var returned: List[int]

  action init = all {
    live' = Set(),
    nextIdx' = 1,
    cursorAt' = 0,
    open' = false,
    returned' = List(),
  }

  action insert = all {
    nextIdx <= MAX_ITEMS,
    live' = live.union(Set(nextIdx)),
    nextIdx' = nextIdx + 1,
    cursorAt' = cursorAt, open' = open, returned' = returned,
  }

  /// Tombstone delete: the index is retired but never reused.
  action del(i: int): bool = all {
    live.contains(i),
    live' = live.exclude(Set(i)),
    nextIdx' = nextIdx, cursorAt' = cursorAt, open' = open, returned' = returned,
  }

  /// gv_db_cursor_open starts a NEW scroll. `returned` records one session, so
  /// it resets here -- re-scrolling from the start is not a duplicate, it is a
  /// different cursor.
  action cursorOpen = all {
    not(open),
    open' = true,
    cursorAt' = 0,
    returned' = List(),
    live' = live, nextIdx' = nextIdx,
  }

  /// gv_db_cursor_next: hand back up to PAGE live items strictly after
  /// cursorAt, then advance. Because deletes are tombstones the indices are
  /// stable, so keying off "last index returned" needs no offset arithmetic and
  /// concurrent writes cannot shift the cursor.
  ///
  /// Iterate over an ordered List (`range`), not a Set: Quint sets are
  /// unordered, so folding a filtered set would build the page in arbitrary
  /// order and the monotonicity check would be meaningless.
  action cursorNextRecording = all {
    open,
    val taken = range(1, MAX_ITEMS + 1).foldl(List(), (acc, i) =>
      if (acc.length() < PAGE and i > cursorAt and i < nextIdx and live.contains(i))
        acc.append(i)
      else acc)
    all {
      returned' = taken.foldl(returned, (acc, i) => acc.append(i)),
      cursorAt' = if (taken.length() > 0) taken[taken.length() - 1] else cursorAt,
    },
    live' = live, nextIdx' = nextIdx, open' = open,
  }

  action cursorClose = all {
    open,
    open' = false,
    live' = live, nextIdx' = nextIdx, cursorAt' = cursorAt, returned' = returned,
  }

  action step = any {
    insert,
    cursorOpen,
    cursorNextRecording,
    cursorClose,
    nondet i = 1.to(MAX_ITEMS).oneOf()
    del(i),
  }

  // ---- invariants ---------------------------------------------------------

  /// A cursor never returns the same index twice. Guaranteed because indices
  /// are stable (tombstone deletes) and the cursor advances strictly.
  val noDuplicates: bool =
    1.to(MAX_ITEMS).forall(a => 1.to(MAX_ITEMS).forall(b =>
      (a != b and a <= returned.length() and b <= returned.length())
        implies returned[a - 1] != returned[b - 1]))

  /// Returned indices strictly increase -- the cursor never goes backwards.
  val monotone: bool =
    1.to(MAX_ITEMS).forall(i =>
      (i + 1 <= returned.length()) implies returned[i - 1] < returned[i])

  /// The cursor never returns an index that was never allocated.
  val noPhantomRow: bool =
    1.to(MAX_ITEMS).forall(i =>
      (i <= returned.length()) implies returned[i - 1] < nextIdx)

  val safety: bool = and { noDuplicates, monotone, noPhantomRow }
}

module scroll_cursor {
  import scroll_proto(MAX_ITEMS = 4, PAGE = 2).*

  /// Inserting during a scroll cannot cause a skip or a duplicate, because
  /// indices are stable and the cursor keys off the last index returned.
  run insertDuringScrollTest =
    init
      .then(insert)
      .then(insert)
      .then(cursorOpen)
      .then(cursorNextRecording)
      .expect(and { returned == List(1, 2), noDuplicates, monotone })
      .then(insert)                          // lands after the cursor
      .then(cursorNextRecording)
      .expect(and { returned == List(1, 2, 3), noDuplicates, monotone })

  /// Deleting an already-returned row does not shift later rows, so the scroll
  /// continues without duplicating.
  run deleteDuringScrollTest =
    init
      .then(insert)
      .then(insert)
      .then(insert)
      .then(cursorOpen)
      .then(cursorNextRecording)
      .then(del(1))                          // behind the cursor
      .then(cursorNextRecording)
      .expect(and { noDuplicates, monotone, noPhantomRow })
}
