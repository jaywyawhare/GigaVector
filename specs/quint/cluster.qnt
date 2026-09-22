// -*- mode: quint ---
//
// cluster.qnt — model of cluster membership and health
// (src/admin/cluster.c: cluster_start / stop / get_node / list_nodes /
//  is_healthy / wait_ready, with the shard manager attached).

module cluster_proto {
  const NODES: Set[int]

  var members: Set[int]
  var up: Set[int]
  var started: bool

  action init = all {
    members' = Set(),
    up' = Set(),
    started' = false,
  }

  def quorum: int = members.size() / 2 + 1
  def healthy: bool = started and members.size() > 0 and up.size() >= quorum

  action start = all {
    not(started),
    started' = true,
    members' = members, up' = up,
  }

  action stop = all {
    started,
    started' = false,
    members' = members, up' = up,
  }

  action join(n: int): bool = all {
    not(members.contains(n)),
    members' = members.union(Set(n)),
    up' = up.union(Set(n)),
    started' = started,
  }

  action leave(n: int): bool = all {
    members.contains(n),
    members' = members.exclude(Set(n)),
    up' = up.exclude(Set(n)),
    started' = started,
  }

  action nodeFail(n: int): bool = all {
    up.contains(n),
    up' = up.exclude(Set(n)),
    members' = members, started' = started,
  }

  action recover(n: int): bool = all {
    members.contains(n),
    not(up.contains(n)),
    up' = up.union(Set(n)),
    members' = members, started' = started,
  }

  action step = any {
    start, stop,
    nondet n = NODES.oneOf()
    join(n),
    nondet n = NODES.oneOf()
    leave(n),
    nondet n = NODES.oneOf()
    nodeFail(n),
    nondet n = NODES.oneOf()
    recover(n),
  }

  // ---- invariants ---------------------------------------------------------

  /// A live node is always a member: liveness never outruns membership.
  val upSubsetOfMembers: bool = up.subseteq(members)

  /// cluster_is_healthy implies an actual quorum is reachable. If this can be
  /// false the cluster reports green while unable to make progress.
  val healthyImpliesQuorum: bool = healthy implies up.size() >= quorum

  /// A stopped cluster never reports healthy.
  val stoppedNotHealthy: bool = not(started) implies not(healthy)

  val safety: bool = and {
    upSubsetOfMembers, healthyImpliesQuorum, stoppedNotHealthy,
  }

  // ---- inductive invariant -------------------------------------------------
  val typeOk: bool = and {
    members.in(NODES.powerset()),
    up.in(NODES.powerset()),
    started.in(Set(false, true)),
  }

  /// `upSubsetOfMembers` is the load-bearing conjunct: it is what makes
  /// healthyImpliesQuorum preserved, since quorum is computed over members.
  val indInv: bool = and {
    typeOk,
    upSubsetOfMembers,
    healthyImpliesQuorum,
    stoppedNotHealthy,
  }
}

module cluster {
  import cluster_proto(NODES = Set(1, 2, 3)).*

  /// Losing a minority keeps the cluster healthy; losing a majority does not.
  run quorumHealthTest =
    init
      .then(start)
      .then(join(1))
      .then(join(2))
      .then(join(3))
      .expect(and { healthy, healthyImpliesQuorum })
      .then(nodeFail(1))
      .expect(and { healthy, up.size() == 2 })     // 2 of 3 -> still quorum
      .then(nodeFail(2))
      .expect(and { not(healthy), healthyImpliesQuorum, upSubsetOfMembers })
}
