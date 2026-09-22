// -*- mode: quint -*-
//
// raft.qnt — model of GigaVector's Raft consensus core (src/admin/raft.c).
//
// This is a *faithful* model of that file, not of the Raft paper. Where the
// implementation deviates from Ongaro's algorithm the model follows the
// implementation, so that model checking reports the implementation's bugs
// rather than the paper's correctness.
//
// Deliberate fidelity points, each mirroring a specific line of raft.c:
//   * `votesGranted` is an integer COUNTER, exactly as `r->votes_granted`
//     (raft.c:41, incremented at raft.c:236). The Raft paper keeps a SET of
//     voters. See invariant `votesGrantedSound`.
//   * `majority = |NODES| / 2 + 1`, matching `(n_peers + 1) / 2 + 1` (raft.c:354)
//     where n_peers excludes self.
//   * commit advancement is restricted to current-term entries (raft.c:190).
//   * AppendEntries never moves commitIndex backwards (raft.c:300).
//   * `matchIndex` is assigned unconditionally on a success response
//     (raft.c:319) with no monotonicity guard. See `matchIndexMonotone`.
//
// The network is modelled as a monotonically growing SET of in-flight
// messages: delivery does not remove a message. That admits reordering,
// arbitrary delay, loss (never delivered) and DUPLICATION (delivered twice) --
// the standard TLA+ message-set idiom. Duplication is what exercises the
// counter above.
//
// Run:
//   quint typecheck specs/quint/raft.qnt
//   quint test      specs/quint/raft.qnt
//   quint verify --invariant=safety --max-steps=8 specs/quint/raft.qnt

module raft_proto {
  type Entry = { term: int, value: int }

  // One flat record for every message kind keeps the type uniform, which
  // Apalache handles far better than a sum type over records.
  type Msg = {
    kind: str,              // RequestVote | RequestVoteResp | AppendEntries | AppendEntriesResp
    from: int,
    to: int,
    term: int,
    lastLogIndex: int,      // RequestVote
    lastLogTerm: int,
    prevLogIndex: int,      // AppendEntries
    prevLogTerm: int,
    entries: List[Entry],
    leaderCommit: int,
    voteGranted: bool,      // RequestVoteResp
    success: bool,          // AppendEntriesResp
    matchIndex: int
  }

  type CommittedEntry = { index: int, term: int, value: int }

  // ---- cluster parameters -------------------------------------------------
  // Bounds keep the state space finite; raise them for deeper sweeps.
  const NODES: Set[int]
  const MAX_TERM: int
  const MAX_LOG: int
  /// true  -> model the FIXED raft.c: votes are deduplicated per voter id.
  /// false -> model the PRE-FIX raft.c: `r->votes_granted++` per response.
  const DEDUPE_VOTES: bool
  // raft.c:354 -- (n_peers + 1) / 2 + 1, where n_peers excludes self.
  pure val MAJORITY: int = NODES.size() / 2 + 1

  pure val NONE: int = -1

  pure val EMPTY_MSG: Msg = {
    kind: "", from: NONE, to: NONE, term: 0,
    lastLogIndex: 0, lastLogTerm: 0,
    prevLogIndex: 0, prevLogTerm: 0, entries: List(),
    leaderCommit: 0, voteGranted: false, success: false, matchIndex: 0
  }

  // ---- state --------------------------------------------------------------
  var role: int -> str                 // "Follower" | "Candidate" | "Leader"
  var currentTerm: int -> int
  var votedFor: int -> int             // NONE = not voted this term
  var log: int -> List[Entry]
  var commitIndex: int -> int
  var votesGranted: int -> int         // the counter, per raft.c:41
  var nextIndex: int -> (int -> int)
  var matchIndex: int -> (int -> int)
  var messages: Set[Msg]

  // Auxiliary history, not part of the implementation state. Accumulates every
  // entry that any node has ever considered committed, so that the completeness
  // and state-machine-safety invariants can be stated.
  var committed: Set[CommittedEntry]
  // Highest matchIndex ever recorded per (leader, peer), to detect regressions.
  var matchIndexHigh: int -> (int -> int)
  // Every vote ever granted, as (voter, term, candidate). Set semantics give
  // one-vote-per-voter-per-term for free, which is precisely the property the
  // implementation's integer counter is supposed to approximate.
  var votesCast: Set[{ voter: int, term: int, cand: int }]
  // Mirrors `r->voted_by[]` in the fixed raft.c: which peers have already been
  // counted toward this node's tally in its current term.
  var votedBy: int -> Set[int]

  // ---- log helpers (mirroring raft.c:60-65) -------------------------------
  pure def lastIndexOf(l: List[Entry]): int = l.length()

  pure def termAt(l: List[Entry], idx: int): int =
    if (idx <= 0 or idx > l.length()) 0 else l[idx - 1].term

  pure def lastTermOf(l: List[Entry]): int = termAt(l, l.length())

  // raft.c:217 — candidate's log is at least as up to date as ours.
  pure def upToDate(cndLastTerm: int, cndLastIdx: int, l: List[Entry]): bool =
    or {
      cndLastTerm > lastTermOf(l),
      cndLastTerm == lastTermOf(l) and cndLastIdx >= lastIndexOf(l)
    }

  // raft.c:271-285 — append with truncate-on-first-conflict, expressed as a
  // fold carrying (log so far, current raft index).
  //
  // The step operator is inlined as a lambda rather than a named definition:
  // Apalache requires the second argument of foldl to be a literal operator.
  pure def applyEntries(l: List[Entry], prevIdx: int, es: List[Entry]): List[Entry] =
    es.foldl({ lg: l, idx: prevIdx }, (acc, e) => {
      val i = acc.idx + 1
      val cur = acc.lg
      if (i <= cur.length())
        if (termAt(cur, i) != e.term)
          { lg: cur.slice(0, i - 1).append(e), idx: i }  // conflict: truncate, append
        else
          { lg: cur, idx: i }                             // present and matching
      else
        { lg: cur.append(e), idx: i }
    }).lg

  // ---- message constructors ----------------------------------------------
  pure def mkRequestVote(n: int, dst: int, term: int, l: List[Entry]): Msg =
    { ...EMPTY_MSG, kind: "RequestVote", from: n, to: dst, term: term,
      lastLogIndex: lastIndexOf(l), lastLogTerm: lastTermOf(l) }

  pure def mkVoteResp(n: int, dst: int, term: int, granted: bool): Msg =
    { ...EMPTY_MSG, kind: "RequestVoteResp", from: n, to: dst, term: term,
      voteGranted: granted }

  // raft.c:141-159
  pure def mkAppendEntries(n: int, dst: int, term: int, l: List[Entry],
                           ni: int, ci: int): Msg = {
    val niC = if (ni < 1) 1 else ni
    val prevIdx = niC - 1
    { ...EMPTY_MSG, kind: "AppendEntries", from: n, to: dst, term: term,
      prevLogIndex: prevIdx, prevLogTerm: termAt(l, prevIdx),
      entries: if (niC <= l.length()) l.slice(niC - 1, l.length()) else List(),
      leaderCommit: ci }
  }

  pure def mkAppendResp(n: int, dst: int, term: int, ok: bool, mi: int): Msg =
    { ...EMPTY_MSG, kind: "AppendEntriesResp", from: n, to: dst, term: term,
      success: ok, matchIndex: mi }

  // ---- init ---------------------------------------------------------------
  action init = all {
    role' = NODES.mapBy(_ => "Follower"),
    currentTerm' = NODES.mapBy(_ => 0),
    votedFor' = NODES.mapBy(_ => NONE),
    log' = NODES.mapBy(_ => List()),
    commitIndex' = NODES.mapBy(_ => 0),
    votesGranted' = NODES.mapBy(_ => 0),
    nextIndex' = NODES.mapBy(_ => NODES.mapBy(_ => 1)),
    matchIndex' = NODES.mapBy(_ => NODES.mapBy(_ => 0)),
    matchIndexHigh' = NODES.mapBy(_ => NODES.mapBy(_ => 0)),
    messages' = Set(),
    committed' = Set(),
    votesCast' = Set(),
    votedBy' = NODES.mapBy(_ => Set()),
  }

  // ---- actions ------------------------------------------------------------

  // raft.c:377 raft_tick -> become_candidate (raft.c:173).
  // MAJORITY > 1 for every cluster we check, so the single-node
  // maybe_become_leader shortcut at raft.c:182 cannot fire here.
  action timeout(n: int): bool = all {
    role.get(n) != "Leader",
    currentTerm.get(n) < MAX_TERM,
    val t = currentTerm.get(n) + 1
    all {
      role' = role.set(n, "Candidate"),
      currentTerm' = currentTerm.set(n, t),
      votedFor' = votedFor.set(n, n),
      votesGranted' = votesGranted.set(n, 1),          // vote for self
      votedBy' = votedBy.set(n, Set()),                // raft.c become_candidate
      messages' = messages.union(
        NODES.exclude(Set(n)).map(p => mkRequestVote(n, p, t, log.get(n)))),
    },
    log' = log, commitIndex' = commitIndex,
    nextIndex' = nextIndex, matchIndex' = matchIndex,
    matchIndexHigh' = matchIndexHigh, committed' = committed,
    votesCast' = votesCast,
  }

  // raft.c:204 handle_request_vote
  action handleRequestVote(m: Msg): bool = all {
    m.kind == "RequestVote",
    val n = m.to
    val stale = m.term < currentTerm.get(n)
    // become_follower(msg->term) at raft.c:215 resets votedFor on a term bump
    val termNow = if (m.term > currentTerm.get(n)) m.term else currentTerm.get(n)
    val votedNow = if (m.term > currentTerm.get(n)) NONE else votedFor.get(n)
    val roleNow = if (m.term > currentTerm.get(n)) "Follower" else role.get(n)
    val grant = and {
      not(stale),
      votedNow == NONE or votedNow == m.from,
      upToDate(m.lastLogTerm, m.lastLogIndex, log.get(n)),
    }
    all {
      role' = if (stale) role else role.set(n, roleNow),
      currentTerm' = if (stale) currentTerm else currentTerm.set(n, termNow),
      votedFor' = if (stale) votedFor
                  else votedFor.set(n, if (grant) m.from else votedNow),
      messages' = messages.union(Set(
        mkVoteResp(n, m.from, if (stale) currentTerm.get(n) else termNow, grant))),
      votesGranted' = if (stale or m.term <= currentTerm.get(n)) votesGranted
                      else votesGranted.set(n, 0),
      votesCast' = if (grant)
          votesCast.union(Set({ voter: n, term: termNow, cand: m.from }))
        else votesCast,
    },
    log' = log, commitIndex' = commitIndex,
    nextIndex' = nextIndex, matchIndex' = matchIndex,
    matchIndexHigh' = matchIndexHigh, committed' = committed,
    votedBy' = votedBy,
  }

  // raft.c:232 handle_request_vote_resp, including maybe_become_leader.
  // NOTE the unconditional increment: a duplicate delivery of the same
  // response increments twice. That is the implementation's behaviour.
  action handleVoteResp(m: Msg): bool = all {
    m.kind == "RequestVoteResp",
    val n = m.to
    val higher = m.term > currentTerm.get(n)
    // The fixed implementation ignores a response from a peer already counted.
    val alreadyCounted = DEDUPE_VOTES and votedBy.get(n).contains(m.from)
    val relevant = and {
      not(higher),
      role.get(n) == "Candidate",
      m.term == currentTerm.get(n),
      m.voteGranted,
      not(alreadyCounted),
    }
    val newVotes = if (relevant) votesGranted.get(n) + 1 else votesGranted.get(n)
    val wins = relevant and newVotes >= MAJORITY
    all {
      role' = if (higher) role.set(n, "Follower")
              else if (wins) role.set(n, "Leader") else role,
      currentTerm' = if (higher) currentTerm.set(n, m.term) else currentTerm,
      votedFor' = if (higher) votedFor.set(n, NONE) else votedFor,
      votesGranted' = if (higher) votesGranted.set(n, 0)
                      else votesGranted.set(n, newVotes),
      votedBy' = if (higher) votedBy.set(n, Set())
                 else if (relevant) votedBy.set(n, votedBy.get(n).union(Set(m.from)))
                 else votedBy,
      // maybe_become_leader (raft.c:161): reset indices and blast AppendEntries
      nextIndex' = if (wins)
          nextIndex.set(n, NODES.mapBy(_ => lastIndexOf(log.get(n)) + 1))
        else nextIndex,
      matchIndex' = if (wins) matchIndex.set(n, NODES.mapBy(_ => 0)) else matchIndex,
      messages' = if (wins)
          messages.union(NODES.exclude(Set(n)).map(p =>
            mkAppendEntries(n, p, currentTerm.get(n), log.get(n),
                            lastIndexOf(log.get(n)) + 1, commitIndex.get(n))))
        else messages,
    },
    log' = log, commitIndex' = commitIndex,
    matchIndexHigh' = matchIndexHigh, committed' = committed,
    votesCast' = votesCast,
  }

  // raft.c:393 raft_submit
  action clientRequest(n: int, v: int): bool = all {
    role.get(n) == "Leader",
    log.get(n).length() < MAX_LOG,
    val newLog = log.get(n).append({ term: currentTerm.get(n), value: v })
    all {
      log' = log.set(n, newLog),
      messages' = messages.union(NODES.exclude(Set(n)).map(p =>
        mkAppendEntries(n, p, currentTerm.get(n), newLog,
                        nextIndex.get(n).get(p), commitIndex.get(n)))),
    },
    role' = role, currentTerm' = currentTerm, votedFor' = votedFor,
    commitIndex' = commitIndex, votesGranted' = votesGranted,
    nextIndex' = nextIndex, matchIndex' = matchIndex,
    matchIndexHigh' = matchIndexHigh, committed' = committed,
    votesCast' = votesCast,
     votedBy' = votedBy,
  }

  // raft.c:369-374 heartbeat branch of raft_tick
  action heartbeat(n: int): bool = all {
    role.get(n) == "Leader",
    messages' = messages.union(NODES.exclude(Set(n)).map(p =>
      mkAppendEntries(n, p, currentTerm.get(n), log.get(n),
                      nextIndex.get(n).get(p), commitIndex.get(n)))),
    role' = role, currentTerm' = currentTerm, votedFor' = votedFor,
    log' = log, commitIndex' = commitIndex, votesGranted' = votesGranted,
    nextIndex' = nextIndex, matchIndex' = matchIndex,
    matchIndexHigh' = matchIndexHigh, committed' = committed,
    votesCast' = votesCast,
     votedBy' = votedBy,
  }

  // raft.c:241 handle_append_entries
  action handleAppendEntries(m: Msg): bool = all {
    m.kind == "AppendEntries",
    val n = m.to
    val stale = m.term < currentTerm.get(n)
    val termNow = if (m.term > currentTerm.get(n)) m.term else currentTerm.get(n)
    val votedNow = if (m.term > currentTerm.get(n)) NONE else votedFor.get(n)
    val l = log.get(n)
    // raft.c:259 log-consistency check
    val consistent = and {
      m.prevLogIndex <= lastIndexOf(l),
      termAt(l, m.prevLogIndex) == m.prevLogTerm,
    }
    val newLog = if (consistent) applyEntries(l, m.prevLogIndex, m.entries) else l
    val lastNew = m.prevLogIndex + m.entries.length()
    val candCommit = if (m.leaderCommit < lastNew) m.leaderCommit else lastNew
    // raft.c:300 never regress commitIndex
    val newCommit = if (consistent and candCommit > commitIndex.get(n))
                      candCommit else commitIndex.get(n)
    // Apalache requires a CONSTANT integer range, so iterate 1..MAX_LOG and
    // filter, rather than 1..newCommit.
    val newCommitted =
      1.to(MAX_LOG).filter(i => i <= newCommit and i <= newLog.length())
        .map(i => { index: i, term: newLog[i - 1].term, value: newLog[i - 1].value })
    all {
      role' = if (stale) role else role.set(n, "Follower"),
      currentTerm' = if (stale) currentTerm else currentTerm.set(n, termNow),
      votedFor' = if (stale) votedFor else votedFor.set(n, votedNow),
      votesGranted' = if (stale) votesGranted else votesGranted.set(n, 0),
      log' = if (stale) log else log.set(n, newLog),
      commitIndex' = if (stale) commitIndex else commitIndex.set(n, newCommit),
      committed' = if (stale) committed else committed.union(newCommitted),
      messages' = messages.union(Set(
        mkAppendResp(n, m.from, if (stale) currentTerm.get(n) else termNow,
                     not(stale) and consistent, lastNew))),
    },
    nextIndex' = nextIndex, matchIndex' = matchIndex,
    matchIndexHigh' = matchIndexHigh, votesCast' = votesCast,
     votedBy' = votedBy,
  }

  // raft.c:312 handle_append_entries_resp, including advance_commit (raft.c:187)
  action handleAppendResp(m: Msg): bool = all {
    m.kind == "AppendEntriesResp",
    val n = m.to
    val p = m.from
    val higher = m.term > currentTerm.get(n)
    val relevant = and {
      not(higher),
      role.get(n) == "Leader",
      m.term == currentTerm.get(n),
    }
    // raft.c:319 — assigned unconditionally, no max() guard
    val newMatch = if (relevant and m.success)
        matchIndex.get(n).set(p, m.matchIndex) else matchIndex.get(n)
    val newNext = if (relevant)
        if (m.success) nextIndex.get(n).set(p, m.matchIndex + 1)
        else nextIndex.get(n).set(p,
          if (nextIndex.get(n).get(p) > 1) nextIndex.get(n).get(p) - 1 else 1)
      else nextIndex.get(n)
    // advance_commit: highest N > commitIndex with current-term entry on a quorum
    val l = log.get(n)
    val cands = 1.to(MAX_LOG).filter(N => and {
      N > commitIndex.get(n),
      N <= lastIndexOf(l),
      termAt(l, N) == currentTerm.get(n),
      1 + NODES.exclude(Set(n)).filter(q => newMatch.get(q) >= N).size() >= MAJORITY,
    })
    val newCommit = if (relevant and m.success and cands.size() > 0)
        cands.fold(commitIndex.get(n), (a, b) => if (b > a) b else a)
      else commitIndex.get(n)
    val newCommitted =
      1.to(MAX_LOG).filter(i => i <= newCommit and i <= l.length())
        .map(i => { index: i, term: l[i - 1].term, value: l[i - 1].value })
    all {
      role' = if (higher) role.set(n, "Follower") else role,
      currentTerm' = if (higher) currentTerm.set(n, m.term) else currentTerm,
      votedFor' = if (higher) votedFor.set(n, NONE) else votedFor,
      votesGranted' = if (higher) votesGranted.set(n, 0) else votesGranted,
      matchIndex' = matchIndex.set(n, newMatch),
      nextIndex' = nextIndex.set(n, newNext),
      matchIndexHigh' = matchIndexHigh.set(n,
        NODES.mapBy(q =>
          val hi = matchIndexHigh.get(n).get(q)
          val cur = newMatch.get(q)
          if (cur > hi) cur else hi)),
      commitIndex' = commitIndex.set(n, newCommit),
      committed' = committed.union(newCommitted),
      // failure path retries with an earlier prefix (raft.c:324)
      messages' = if (relevant and not(m.success))
          messages.union(Set(mkAppendEntries(n, p, currentTerm.get(n), l,
                             newNext.get(p), commitIndex.get(n))))
        else messages,
    },
    log' = log, votesCast' = votesCast,
     votedBy' = votedBy,
  }

  action step = any {
    nondet n = NODES.oneOf()
    timeout(n),
    nondet n = NODES.oneOf()
    nondet v = Set(1, 2).oneOf()
    clientRequest(n, v),
    nondet n = NODES.oneOf()
    heartbeat(n),
    // Delivery leaves the message in flight: models reorder, delay and DUPLICATION.
    all {
      messages.size() > 0,
      nondet m = messages.oneOf()
      any {
        handleRequestVote(m),
        handleVoteResp(m),
        handleAppendEntries(m),
        handleAppendResp(m),
      },
    },
  }

  // ---- invariants ---------------------------------------------------------

  /// Raft "Election Safety": at most one leader per term.
  val electionSafety: bool =
    NODES.forall(a => NODES.forall(b =>
      (a != b and role.get(a) == "Leader" and role.get(b) == "Leader")
        implies currentTerm.get(a) != currentTerm.get(b)))

  /// The vote counter never exceeds the number of DISTINCT nodes that actually
  /// granted this candidate a vote in its current term (+1 for its own vote).
  ///
  /// Stated against the `votesCast` history rather than the live `votedFor`,
  /// because a node legitimately clears `votedFor` when it advances its term --
  /// comparing against live state would flag correct behaviour.
  ///
  /// This targets `r->votes_granted` being an int incremented per response
  /// (raft.c:41, raft.c:236) where the Raft paper keeps a set of voter ids.
  val votesGrantedSound: bool =
    NODES.forall(n =>
      (role.get(n) == "Candidate" or role.get(n) == "Leader") implies
        votesGranted.get(n) <=
          1 + votesCast.filter(v => and {
            v.term == currentTerm.get(n),
            v.cand == n,
          }).size())

  /// Raft "Log Matching": equal (index, term) implies equal prefixes.
  val logMatching: bool =
    NODES.forall(a => NODES.forall(b =>
      1.to(MAX_LOG).forall(i =>
        (i <= log.get(a).length() and i <= log.get(b).length() and
         termAt(log.get(a), i) == termAt(log.get(b), i))
          implies 1.to(MAX_LOG).forall(j =>
               j <= i implies log.get(a)[j - 1] == log.get(b)[j - 1]))))

  /// Raft "State Machine Safety": no two different entries ever commit at the
  /// same log index, anywhere in the cluster.
  val stateMachineSafety: bool =
    committed.forall(x => committed.forall(y =>
      (x.index == y.index) implies (x.term == y.term and x.value == y.value)))

  /// Raft "Leader Completeness": a current leader holds every committed entry.
  val leaderCompleteness: bool =
    NODES.forall(n => role.get(n) == "Leader" implies
      committed.forall(c =>
        c.term < currentTerm.get(n) implies and {
          c.index <= log.get(n).length(),
          log.get(n)[c.index - 1].value == c.value,
        }))

  /// matchIndex should never regress; raft.c:319 assigns it unconditionally, so
  /// a reordered success response can move it backwards.
  val matchIndexMonotone: bool =
    NODES.forall(n => NODES.forall(p =>
      matchIndex.get(n).get(p) >= matchIndexHigh.get(n).get(p)))

  /// Sanity: commitIndex never runs past the local log.
  val commitIndexBounded: bool =
    NODES.forall(n => commitIndex.get(n) <= log.get(n).length())

  /// The bundle checked in CI.
  val safety: bool = and {
    electionSafety,
    logMatching,
    stateMachineSafety,
    leaderCompleteness,
    commitIndexBounded,
  }

}

// ---- instances ------------------------------------------------------------

/// 3-node cluster. MAJORITY = 2, so a candidate reaches leadership on its first
/// granted vote and the response counter is never incremented twice.
module raft3 {
  import raft_proto(NODES = Set(0, 1, 2), MAX_TERM = 3, MAX_LOG = 2,
                    DEDUPE_VOTES = true).*

  /// A single election in term 1 makes exactly one leader.
  run electionTest =
    init
      .then(timeout(0))
      .then(all {
        messages.size() > 0,
        nondet m = messages.filter(x => x.kind == "RequestVote" and x.to == 1).oneOf()
        handleRequestVote(m)
      })
      .then(all {
        nondet m = messages.filter(x =>
          x.kind == "RequestVoteResp" and x.to == 0 and x.voteGranted).oneOf()
        handleVoteResp(m)
      })
      .expect(role.get(0) == "Leader" and electionSafety)
}

/// 5-node cluster. MAJORITY = 3, so a candidate stays a Candidate after its
/// first granted vote and keeps counting -- which is what exposes the
/// duplicate-response double count. Bounds are tightened to stay tractable.
module raft5 {
  import raft_proto(NODES = Set(0, 1, 2, 3, 4), MAX_TERM = 2, MAX_LOG = 1,
                    DEDUPE_VOTES = true).*

}

/// The PRE-FIX implementation, kept as a permanent regression witness:
/// `votes_granted` incremented per response rather than per distinct voter.
/// `voteDoubleCountTest` below is the counterexample that motivated the fix in
/// src/admin/raft.c, and it must keep failing `votesGrantedSound` here.
module raft5_prefix {
  import raft_proto(NODES = Set(0, 1, 2, 3, 4), MAX_TERM = 2, MAX_LOG = 1,
                    DEDUPE_VOTES = false).*

  /// Deterministic demonstration of the duplicate-vote double count.
  ///
  /// Node 2 stands for election in term 1. Exactly ONE peer (node 1) grants it
  /// a vote. That single response is delivered twice -- a duplicate, which the
  /// network is allowed to produce and which `raft_step` does not deduplicate.
  /// `r->votes_granted` is incremented per response (raft.c:236) rather than
  /// recording the voter id, so the second delivery pushes the count to 3 and
  /// `maybe_become_leader` (raft.c:161) promotes node 2 -- on 2 of 5 nodes.
  run voteDoubleCountTest =
    init
      .then(timeout(2))
      .then(all {
        nondet m = messages.filter(x => x.kind == "RequestVote" and x.to == 1).oneOf()
        handleRequestVote(m)
      })
      .then(all {
        nondet m = messages.filter(x =>
          x.kind == "RequestVoteResp" and x.to == 2 and x.voteGranted).oneOf()
        handleVoteResp(m)
      })
      // One genuine vote counted. Still short of MAJORITY = 3.
      .expect(votesGranted.get(2) == 2 and role.get(2) == "Candidate")
      .then(all {
        // The very same response, delivered a second time.
        nondet m = messages.filter(x =>
          x.kind == "RequestVoteResp" and x.to == 2 and x.voteGranted).oneOf()
        handleVoteResp(m)
      })
      .expect(and {
        votesCast.size() == 1,               // only one peer ever voted for node 2
        votesGranted.get(2) == 3,            // but the counter reads three
        role.get(2) == "Leader",             // so it takes leadership ...
        not(votesGrantedSound),              // ... unsoundly
      })
}
