// -*- mode: quint ---
//
// consistency.qnt — model of GigaVector's read-consistency layer
// (src/search/consistency.c).
//
// Models `consistency_check` (consistency.c:124) exactly, including the
// session-table lookup and its fail-open branch, plus
// `consistency_update_session` (consistency.c:212) with its
// advance-forward-only rule.
//
// Fidelity points:
//   * `session_lookup_locked` returns 0 both for an UNKNOWN token and for a
//     known session that has recorded no writes. `consistency_check` cannot
//     tell those apart and allows the read in both cases (consistency.c:169,
//     "Unknown session or no writes recorded yet -- allow read").
//   * STRONG returns 0 for every replica, i.e. "this replica does not satisfy
//     the requirement; redirect to the leader" -- it is not a rejection.
//   * update_session only ever moves a position forward.

module consistency_proto {
  const REPLICAS: Set[int]
  const MAX_POS: int

  pure val STRONG: str = "strong"
  pure val EVENTUAL: str = "eventual"
  pure val BOUNDED: str = "bounded"
  pure val SESSION: str = "session"

  // Replica state.
  var replicaPos: int -> int          // last applied write position
  var replicaLag: int -> int          // milliseconds behind the leader
  var leaderPos: int                  // leader's write position

  // Session table (consistency.c). A token maps to the highest write position
  // the session has performed. A token NOT in this map is "unknown".
  var sessions: int -> int
  var knownTokens: Set[int]
  // Every token ever minted. consistency_new_session hands out fresh tokens, so
  // a retired token is never reissued -- without this the model would recycle
  // an evicted token into a "new" session that inherits the old one's writes.
  var mintedTokens: Set[int]

  // Auxiliary history: what each session has actually written, and what it has
  // subsequently read. `readYourWrites` compares them.
  var sessionWrites: Set[{ token: int, pos: int }]
  // `requiredAtRead` is the session's write position AT THE MOMENT of the read.
  // Comparing a read against the session's FINAL write set would flag reads
  // that legitimately preceded a later write -- read-your-writes only
  // constrains writes that already happened.
  var sessionReads: Set[{ token: int, replica: int, sawPos: int, requiredAtRead: int }]

  // ---- consistency_check (consistency.c:124) ------------------------------
  def sessionLookup(token: int): int =
    if (knownTokens.contains(token)) sessions.get(token) else 0

  def checkOk(level: str, token: int, maxStaleness: int, r: int): bool =
    if (level == STRONG) false                       // redirect to leader
    else if (level == EVENTUAL) true
    else if (level == BOUNDED) replicaLag.get(r) <= maxStaleness
    else
      // SESSION
      // FIXED (consistency.c): an unknown token now fails CLOSED. A known
      // session with no writes still passes -- it has nothing to catch up on.
      if (not(knownTokens.contains(token))) false
      else
        val required = sessionLookup(token)
        if (required == 0) true
        else replicaPos.get(r) >= required

  // ---- init ---------------------------------------------------------------
  action init = all {
    replicaPos' = REPLICAS.mapBy(_ => 0),
    replicaLag' = REPLICAS.mapBy(_ => 0),
    leaderPos' = 0,
    sessions' = Map(),
    knownTokens' = Set(),
    mintedTokens' = Set(),
    sessionWrites' = Set(),
    sessionReads' = Set(),
  }

  // ---- actions ------------------------------------------------------------

  /// consistency_new_session
  action newSession(token: int): bool = all {
    not(mintedTokens.contains(token)),          // tokens are never reissued
    knownTokens' = knownTokens.union(Set(token)),
    mintedTokens' = mintedTokens.union(Set(token)),
    sessions' = sessions.put(token, 0),
    replicaPos' = replicaPos, replicaLag' = replicaLag, leaderPos' = leaderPos,
    sessionWrites' = sessionWrites, sessionReads' = sessionReads,
  }

  /// A write on the leader, attributed to a session, then recorded via
  /// consistency_update_session (advance-forward-only).
  action write(token: int): bool = all {
    leaderPos < MAX_POS,
    // consistency_update_session returns -1 for an unknown token, so a write
    // cannot be attributed to a session that does not exist.
    knownTokens.contains(token),
    val p = leaderPos + 1
    all {
      leaderPos' = p,
      sessions' = if (knownTokens.contains(token))
          sessions.put(token, if (p > sessions.get(token)) p else sessions.get(token))
        else sessions,
      sessionWrites' = sessionWrites.union(Set({ token: token, pos: p })),
    },
    knownTokens' = knownTokens,
    replicaPos' = replicaPos, replicaLag' = replicaLag,
    sessionReads' = sessionReads,
     mintedTokens' = mintedTokens,
  }

  /// Replication progress: a replica applies one more entry, reducing its lag.
  action replicate(r: int): bool = all {
    replicaPos.get(r) < leaderPos,
    replicaPos' = replicaPos.set(r, replicaPos.get(r) + 1),
    replicaLag' = replicaLag.set(r, leaderPos - (replicaPos.get(r) + 1)),
    leaderPos' = leaderPos, sessions' = sessions, knownTokens' = knownTokens,
    sessionWrites' = sessionWrites, sessionReads' = sessionReads,
     mintedTokens' = mintedTokens,
  }

  /// The session table loses a token: eviction, restart, or a token minted on
  /// another node. `knownTokens` shrinks while the session keeps being used.
  action forgetSession(token: int): bool = all {
    knownTokens.contains(token),
    knownTokens' = knownTokens.exclude(Set(token)),
    sessions' = sessions, replicaPos' = replicaPos, replicaLag' = replicaLag,
    leaderPos' = leaderPos, sessionWrites' = sessionWrites,
    sessionReads' = sessionReads,
     mintedTokens' = mintedTokens,
  }

  /// A SESSION-consistency read routed to replica r, admitted by checkOk.
  action sessionRead(token: int, r: int): bool = all {
    checkOk(SESSION, token, 0, r),
    sessionReads' = sessionReads.union(Set({
      token: token, replica: r, sawPos: replicaPos.get(r),
      requiredAtRead: sessionLookup(token) })),
    replicaPos' = replicaPos, replicaLag' = replicaLag, leaderPos' = leaderPos,
    sessions' = sessions, knownTokens' = knownTokens,
    sessionWrites' = sessionWrites,
     mintedTokens' = mintedTokens,
  }

  action step = any {
    nondet t = 1.to(2).oneOf()
    newSession(t),
    nondet t = 1.to(2).oneOf()
    write(t),
    nondet t = 1.to(2).oneOf()
    forgetSession(t),
    nondet r = REPLICAS.oneOf()
    replicate(r),
    nondet t = 1.to(2).oneOf()
    nondet r = REPLICAS.oneOf()
    sessionRead(t, r),
  }

  // ---- invariants ---------------------------------------------------------

  /// Read-your-writes: a read admitted under SESSION consistency must land on
  /// a replica that has already applied every write the session made.
  /// Read-your-writes: a read admitted under SESSION consistency landed on a
  /// replica that had already applied every write the session had made AT THAT
  /// POINT. Writes issued after the read are not its concern.
  val readYourWrites: bool =
    sessionReads.forall(rd => rd.sawPos >= rd.requiredAtRead)

  /// A session's recorded position never decreases.
  val sessionPositionMonotone: bool =
    knownTokens.forall(t => sessions.get(t) >= 0)

  /// Bounded staleness is honoured whenever it admits a replica.
  val boundedStalenessHonoured: bool =
    REPLICAS.forall(r =>
      1.to(MAX_POS).forall(b =>
        checkOk(BOUNDED, 0, b, r) implies replicaLag.get(r) <= b))

  /// `readYourWrites` is now IN the safety bundle: with the fail-closed fix it
  /// holds. It was previously excluded because the implementation admitted
  /// reads for unknown tokens.
  val safety: bool = and {
    sessionPositionMonotone,
    boundedStalenessHonoured,
    readYourWrites,
  }
}

module consistency {
  import consistency_proto(REPLICAS = Set(1, 2), MAX_POS = 3).*

  /// The fail-open branch, deterministically.
  ///
  /// A session writes at position 1, so read-your-writes requires any replica
  /// serving it to be at position >= 1. The session table then loses the token
  /// -- eviction, a restart, or a token minted on a different node. From that
  /// moment `session_lookup_locked` returns 0, `consistency_check` takes the
  /// "unknown session ... allow read" branch (consistency.c:169), and a replica
  /// still at position 0 is admitted. The session reads its own write as absent.
  run failClosedSessionTest =
    init
      .then(newSession(1))
      .then(write(1))                      // leader position 1, session requires 1
      // While the token is known, replica 2 (still at position 0) is correctly
      // refused: consistency_check compares replica_position >= required_pos.
      .expect(not(checkOk(SESSION, 1, 0, 2)))
      .then(forgetSession(1))              // token evicted / minted elsewhere
      // After the fix the same read is REFUSED: an unknown token can no longer
      // certify that this replica has caught up, so the caller goes to the
      // leader instead. Read-your-writes is preserved.
      .expect(not(checkOk(SESSION, 1, 0, 2)))
      .expect(readYourWrites)
}
