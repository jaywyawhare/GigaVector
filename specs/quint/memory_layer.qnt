// -*- mode: quint ---
//
// memory_layer.qnt — model of the agent memory layer
// (src/storage/memory_layer.c, memory_consolidation.c, memory_extraction.c).
//
// Memories are written, consolidated (several merged into one summary), and
// deleted. The hazard specific to consolidation is that it REPLACES several
// records with one: if the replacement is not atomic with the removal, a crash
// or concurrent read sees either both or neither.

module memory_proto {
  const MAX_MEMORIES: int

  var memories: Set[int]
  var nextId: int
  // Which source memories each consolidated record subsumes.
  var subsumes: int -> Set[int]
  // Consolidation in flight: sources chosen, summary not yet installed.
  var pendingSources: Set[int]
  var consolidating: bool
  // Ground truth: facts that must remain retrievable, directly or via a summary.
  var facts: Set[int]

  action init = all {
    memories' = Set(),
    nextId' = 1,
    subsumes' = 1.to(MAX_MEMORIES).mapBy(_ => Set()),
    pendingSources' = Set(),
    consolidating' = false,
    facts' = Set(),
  }

  action remember = all {
    nextId <= MAX_MEMORIES,
    memories' = memories.union(Set(nextId)),
    facts' = facts.union(Set(nextId)),
    nextId' = nextId + 1,
    subsumes' = subsumes, pendingSources' = pendingSources,
    consolidating' = consolidating,
  }

  action forget(m: int): bool = all {
    memories.contains(m),
    not(pendingSources.contains(m)),
    memories' = memories.exclude(Set(m)),
    facts' = facts.exclude(subsumes.get(m).union(Set(m))),
    nextId' = nextId, subsumes' = subsumes,
    pendingSources' = pendingSources, consolidating' = consolidating,
  }

  /// Phase 1: pick sources to consolidate.
  action consolidateBegin(a: int, b: int): bool = all {
    not(consolidating),
    a != b,
    memories.contains(a),
    memories.contains(b),
    pendingSources' = Set(a, b),
    consolidating' = true,
    memories' = memories, nextId' = nextId, subsumes' = subsumes, facts' = facts,
  }

  /// Phase 2, done ATOMICALLY: install the summary and drop the sources in one
  /// step, so no observer sees the sources gone without the summary present.
  action consolidateCommit = all {
    consolidating,
    nextId <= MAX_MEMORIES,
    val srcFacts = pendingSources.fold(Set(), (acc, s) =>
      acc.union(subsumes.get(s)).union(Set(s)))
    all {
      memories' = memories.exclude(pendingSources).union(Set(nextId)),
      subsumes' = subsumes.set(nextId, srcFacts),
      nextId' = nextId + 1,
      pendingSources' = Set(),
      consolidating' = false,
    },
    facts' = facts,
  }

  /// The non-atomic variant: drop the sources BEFORE installing the summary.
  /// Kept as a witness for what `noFactLost` is protecting against.
  action consolidateDropFirst = all {
    consolidating,
    memories' = memories.exclude(pendingSources),
    pendingSources' = pendingSources,
    consolidating' = consolidating,
    nextId' = nextId, subsumes' = subsumes, facts' = facts,
  }

  action step = any {
    remember,
    consolidateCommit,
    nondet m = 1.to(MAX_MEMORIES).oneOf()
    forget(m),
    nondet a = 1.to(MAX_MEMORIES).oneOf()
    nondet b = 1.to(MAX_MEMORIES).oneOf()
    consolidateBegin(a, b),
  }

  // ---- invariants ---------------------------------------------------------

  /// Every remembered fact is still retrievable: either its own record exists,
  /// or a record that subsumes it does.
  val noFactLost: bool =
    facts.forall(f =>
      memories.contains(f) or memories.exists(m => subsumes.get(m).contains(f)))

  /// A fact is never reachable through two different summaries.
  val noDoubleSubsumption: bool =
    facts.forall(f =>
      memories.filter(m => subsumes.get(m).contains(f)).size() <= 1)

  /// Ids are never reused.
  val idsMonotone: bool = memories.forall(m => m < nextId)

  val safety: bool = and { noFactLost, noDoubleSubsumption, idsMonotone }
}

module memory_layer {
  import memory_proto(MAX_MEMORIES = 5).*

  /// Atomic consolidation preserves every fact.
  run consolidationPreservesFactsTest =
    init
      .then(remember)
      .then(remember)
      .expect(and { memories == Set(1, 2), noFactLost })
      .then(consolidateBegin(1, 2))
      .then(consolidateCommit)
      .expect(and {
        memories == Set(3),                  // sources replaced by the summary
        noFactLost,                          // both facts still reachable
        noDoubleSubsumption,
      })

  /// Dropping the sources before installing the summary loses both facts --
  /// the window that makes atomic replacement necessary.
  run nonAtomicConsolidationLosesFactsTest =
    init
      .then(remember)
      .then(remember)
      .then(consolidateBegin(1, 2))
      .then(consolidateDropFirst)
      .expect(not(noFactLost))
}
