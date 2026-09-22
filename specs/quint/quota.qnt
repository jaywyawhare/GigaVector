// -*- mode: quint -*-
//
// quota.qnt — model of per-tenant quota enforcement (src/api/quota.c).
//
// The enforcement path is TWO separate locked operations:
//   gv_quota_check_insert  (quota.c:253) -- takes mgr->mutex, compares
//                          current_vectors + n against max_vectors, RELEASES
//                          the mutex, returns GV_QUOTA_OK. It reserves nothing.
//   gv_quota_record_insert (quota.c:328) -- takes mgr->mutex again and adds to
//                          current_vectors.
//
// Between those two calls the caller performs the actual insert. Two tenants'
// requests -- or two concurrent requests for the SAME tenant -- can both pass
// the check against the same pre-insert usage and then both record, pushing
// usage past the cap. `quotaNeverExceeded` targets exactly that window.
//
// The token bucket (`token_bucket_try_consume`, quota.c:294) is different: it
// CONSUMES under the same lock as the check, so the rate limit is atomic. Only
// the capacity quotas have the check/record split.

module quota_proto {
  const MAX_VECTORS: int        // cfg->max_vectors
  const REQUESTS: Set[int]      // concurrent request ids
  const BATCH: int              // vectors per request

  // usage.current_vectors
  var currentVectors: int
  // usage reserved by admitted-but-not-yet-committed requests (quota.c
  // reserved_vectors). Counted against the cap alongside currentVectors.
  var reservedVectors: int
  // Requests that passed gv_quota_check_insert but have not yet recorded.
  var checkedNotRecorded: Set[int]
  // Requests that have completed gv_quota_record_insert.
  var recorded: Set[int]
  var rejected: Set[int]

  action init = all {
    currentVectors' = 0,
    reservedVectors' = 0,
    checkedNotRecorded' = Set(),
    recorded' = Set(),
    rejected' = Set(),
  }

  /// gv_quota_check_insert. Note it compares against `currentVectors` only --
  /// in-flight requests that already passed the check are invisible to it.
  action checkInsert(r: int): bool = all {
    not(checkedNotRecorded.contains(r)),
    not(recorded.contains(r)),
    not(rejected.contains(r)),
    // FIXED (quota.c gv_quota_reserve_insert): the check and the claim happen
    // under ONE lock acquisition, and in-flight reservations count against the
    // cap -- so two concurrent callers cannot both be admitted.
    val admits = currentVectors + reservedVectors + BATCH <= MAX_VECTORS
    all {
      checkedNotRecorded' = if (admits) checkedNotRecorded.union(Set(r))
                            else checkedNotRecorded,
      reservedVectors' = if (admits) reservedVectors + BATCH else reservedVectors,
      rejected' = if (admits) rejected else rejected.union(Set(r)),
    },
    currentVectors' = currentVectors,
    recorded' = recorded,
  }

  /// gv_quota_release_insert: the insert failed, give the reservation back.
  action releaseInsert(r: int): bool = all {
    checkedNotRecorded.contains(r),
    reservedVectors' = reservedVectors - BATCH,
    checkedNotRecorded' = checkedNotRecorded.exclude(Set(r)),
    currentVectors' = currentVectors, recorded' = recorded, rejected' = rejected,
  }

  /// gv_quota_record_insert, after the caller's insert succeeded.
  /// gv_quota_commit_insert: convert the reservation into recorded usage.
  action recordInsert(r: int): bool = all {
    checkedNotRecorded.contains(r),
    currentVectors' = currentVectors + BATCH,
    reservedVectors' = reservedVectors - BATCH,
    checkedNotRecorded' = checkedNotRecorded.exclude(Set(r)),
    recorded' = recorded.union(Set(r)),
    rejected' = rejected,
  }

  /// gv_quota_record_delete: usage goes back down.
  action recordDelete(r: int): bool = all {
    recorded.contains(r),
    currentVectors' = currentVectors - BATCH,
    recorded' = recorded.exclude(Set(r)),
    checkedNotRecorded' = checkedNotRecorded, rejected' = rejected,
    reservedVectors' = reservedVectors,
  }

  action step = any {
    nondet r = REQUESTS.oneOf()
    checkInsert(r),
    nondet r = REQUESTS.oneOf()
    recordInsert(r),
    nondet r = REQUESTS.oneOf()
    recordDelete(r),
    nondet r = REQUESTS.oneOf()
    releaseInsert(r),
  }

  // ---- invariants ---------------------------------------------------------

  /// The tenant's recorded usage never exceeds its configured cap.
  val quotaNeverExceeded: bool = currentVectors <= MAX_VECTORS

  /// Usage is never negative (record_delete under-flow).
  val usageNonNegative: bool = currentVectors >= 0

  /// What an ATOMIC check-and-reserve would guarantee: admitted-but-not-yet
  /// recorded work, plus recorded work, still fits the cap. This is the
  /// property the current two-call design does not provide.
  val reservationSound: bool =
    currentVectors + reservedVectors <= MAX_VECTORS

  /// Now HOLDS: gv_quota_reserve_insert makes check-and-claim atomic and counts
  /// reservations against the cap.
  val safety: bool = and {
    quotaNeverExceeded, usageNonNegative, reservationSound,
  }
}

module quota {
  // Cap of 10, two concurrent requests of 6 each: either alone fits, together
  // they do not.
  import quota_proto(MAX_VECTORS = 10, REQUESTS = Set(1, 2), BATCH = 6).*

  /// The old TOCTOU is now impossible: the second reservation is refused
  /// because the first one's claim counts against the cap.
  run reserveIsAtomicTest =
    init
      .then(checkInsert(1))
      .expect(and { checkedNotRecorded == Set(1), reservedVectors == 6 })
      .then(checkInsert(2))
      .expect(and {
        rejected == Set(2),                // refused, not admitted
        reservationSound,
      })
      .then(recordInsert(1))
      .expect(and { currentVectors == 6, quotaNeverExceeded })

  /// A released reservation frees the capacity for someone else.
  run releaseFreesCapacityTest =
    init
      .then(checkInsert(1))
      .then(releaseInsert(1))
      .expect(reservedVectors == 0)
      .then(checkInsert(2))
      .expect(and { checkedNotRecorded == Set(2), reservationSound })
}
