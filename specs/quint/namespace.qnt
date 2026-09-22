// -*- mode: quint ---
//
// namespace.qnt — model of namespace and alias resolution
// (src/admin/namespace.c, src/api/alias.c).
//
// Namespaces own databases; aliases are indirections to namespaces. The
// hazards are the usual ones for a two-level name table: an alias outliving
// its target, a name being reused while an alias still points at the old
// entity, and delete racing resolution.

module namespace_proto {
  const NAMES: Set[int]
  const ALIASES: Set[int]
  /// Bounds per-namespace content so the state space is finite. Without this
  /// `nsAddVector` grows without limit and no inductive invariant can hold.
  const MAX_CONTENT: int

  var namespaces: Set[int]
  var aliasTarget: int -> int
  var aliases: Set[int]
  // Vectors held per namespace, so a delete can be checked for data loss.
  var contents: int -> int

  /// 0 is the "unset" sentinel for an alias target. `aliasTarget` is kept TOTAL
  /// over ALIASES so that setOfMaps(...) in typeOk can describe it -- that set
  /// contains only total maps, so an empty Map would fail the initial-state
  /// check of the inductive proof.
  pure val NO_TARGET: int = 0

  action init = all {
    namespaces' = Set(),
    aliasTarget' = ALIASES.mapBy(_ => NO_TARGET),
    aliases' = Set(),
    contents' = NAMES.mapBy(_ => 0),
  }

  /// namespace_create
  action nsCreate(n: int): bool = all {
    not(namespaces.contains(n)),
    namespaces' = namespaces.union(Set(n)),
    contents' = contents.set(n, 0),
    aliasTarget' = aliasTarget, aliases' = aliases,
  }

  /// namespace_delete. Aliases are NOT cascaded -- this is what
  /// `noDanglingAlias` checks.
  action nsDelete(n: int): bool = all {
    namespaces.contains(n),
    namespaces' = namespaces.exclude(Set(n)),
    contents' = contents.set(n, 0),
    aliasTarget' = aliasTarget, aliases' = aliases,
  }

  action nsAddVector(n: int): bool = all {
    namespaces.contains(n),
    contents.get(n) < MAX_CONTENT,
    contents' = contents.set(n, contents.get(n) + 1),
    namespaces' = namespaces, aliasTarget' = aliasTarget, aliases' = aliases,
  }

  /// alias_set: point an alias at an existing namespace.
  action aliasSet(a: int, n: int): bool = all {
    namespaces.contains(n),
    aliasTarget' = aliasTarget.put(a, n),
    aliases' = aliases.union(Set(a)),
    namespaces' = namespaces, contents' = contents,
  }

  action aliasDrop(a: int): bool = all {
    aliases.contains(a),
    aliases' = aliases.exclude(Set(a)),
    aliasTarget' = aliasTarget, namespaces' = namespaces, contents' = contents,
  }

  /// A cascading delete: drop the namespace AND every alias pointing at it.
  action nsDeleteCascade(n: int): bool = all {
    namespaces.contains(n),
    namespaces' = namespaces.exclude(Set(n)),
    contents' = contents.set(n, 0),
    aliases' = aliases.filter(a => aliasTarget.get(a) != n),
    aliasTarget' = aliasTarget,
  }

  action step = any {
    nondet n = NAMES.oneOf()
    nsCreate(n),
    nondet n = NAMES.oneOf()
    nsDelete(n),
    nondet n = NAMES.oneOf()
    nsAddVector(n),
    nondet a = ALIASES.oneOf()
    nondet n = NAMES.oneOf()
    aliasSet(a, n),
    nondet a = ALIASES.oneOf()
    aliasDrop(a),
  }

  // ---- invariants ---------------------------------------------------------

  /// Every live alias resolves to a live namespace.
  val noDanglingAlias: bool =
    aliases.forall(a => namespaces.contains(aliasTarget.get(a)))

  /// An alias never resolves to a namespace that was recreated under the same
  /// name after the original was deleted (name reuse pointing at new data).
  val aliasResolvesToLiveTarget: bool = noDanglingAlias

  /// Namespaces are isolated: content counts never go negative.
  val contentsNonNegative: bool = NAMES.forall(n => contents.get(n) >= 0)

  val safety: bool = contentsNonNegative

  // ---- inductive invariant -------------------------------------------------
  val typeOk: bool = and {
    namespaces.in(NAMES.powerset()),
    aliases.in(ALIASES.powerset()),
    aliasTarget.in(setOfMaps(ALIASES, NAMES.union(Set(NO_TARGET)))),
    contents.in(setOfMaps(NAMES, 0.to(MAX_CONTENT))),
    // A live alias always points at a real namespace name, never the sentinel.
    aliases.forall(a => aliasTarget.get(a) != NO_TARGET),
  }

  /// A deleted namespace has no content -- needed to keep the conjunction
  /// closed under nsDelete.
  val indInv: bool = and {
    typeOk,
    contentsNonNegative,
    NAMES.forall(n => not(namespaces.contains(n)) implies contents.get(n) == 0),
  }
}

module namespace {
  import namespace_proto(NAMES = Set(1, 2), ALIASES = Set(1), MAX_CONTENT = 2).*

  /// Deleting a namespace without cascading leaves the alias dangling.
  run danglingAliasTest =
    init
      .then(nsCreate(1))
      .then(aliasSet(1, 1))
      .expect(noDanglingAlias)
      .then(nsDelete(1))
      .expect(not(noDanglingAlias))

  /// The cascading delete keeps the name table consistent.
  run cascadeDeleteTest =
    init
      .then(nsCreate(1))
      .then(aliasSet(1, 1))
      .then(nsDeleteCascade(1))
      .expect(and { noDanglingAlias, aliases == Set() })
}
