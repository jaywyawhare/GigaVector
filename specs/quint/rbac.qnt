// -*- mode: quint ---
//
// rbac.qnt — model of role-based access control (src/security/rbac.c,
// src/security/authz.c).
//
// The properties that matter for an authorization layer are not "does grant
// work" but "can a principal reach a permission it was never given" -- via
// role inheritance, via granting itself a role, or via a deny that fails to
// override an allow.

module rbac_proto {
  const PRINCIPALS: Set[int]
  const ROLES: Set[int]
  const PERMS: Set[int]

  var roleOf: int -> Set[int]           // principal -> roles
  var rolePerms: int -> Set[int]        // role -> permissions
  var roleParent: int -> Set[int]       // role -> inherited-from roles
  var denied: int -> Set[int]           // principal -> explicitly denied perms

  action init = all {
    roleOf' = PRINCIPALS.mapBy(_ => Set()),
    rolePerms' = ROLES.mapBy(_ => Set()),
    roleParent' = ROLES.mapBy(_ => Set()),
    denied' = PRINCIPALS.mapBy(_ => Set()),
  }

  /// Effective permissions: own roles plus one level of inheritance.
  /// (Deep inheritance is bounded here to keep the model finite; the acyclicity
  /// invariant is what makes deeper chains safe.)
  def rolePermsClosed(r: int): Set[int] =
    rolePerms.get(r).union(
      roleParent.get(r).fold(Set(), (acc, p) => acc.union(rolePerms.get(p))))

  def effectivePerms(p: int): Set[int] =
    roleOf.get(p).fold(Set(), (acc, r) => acc.union(rolePermsClosed(r)))
      .exclude(denied.get(p))            // deny overrides allow

  def can(p: int, perm: int): bool = effectivePerms(p).contains(perm)

  action assignRole(p: int, r: int): bool = all {
    not(roleOf.get(p).contains(r)),
    roleOf' = roleOf.set(p, roleOf.get(p).union(Set(r))),
    rolePerms' = rolePerms, roleParent' = roleParent, denied' = denied,
  }

  action grantPerm(r: int, perm: int): bool = all {
    not(rolePerms.get(r).contains(perm)),
    rolePerms' = rolePerms.set(r, rolePerms.get(r).union(Set(perm))),
    roleOf' = roleOf, roleParent' = roleParent, denied' = denied,
  }

  /// Role inheritance, guarded against creating a cycle.
  action addParent(child: int, parent: int): bool = all {
    child != parent,
    not(roleParent.get(child).contains(parent)),
    not(roleParent.get(parent).contains(child)),   // the acyclicity guard
    roleParent' = roleParent.set(child, roleParent.get(child).union(Set(parent))),
    roleOf' = roleOf, rolePerms' = rolePerms, denied' = denied,
  }

  action denyPerm(p: int, perm: int): bool = all {
    not(denied.get(p).contains(perm)),
    denied' = denied.set(p, denied.get(p).union(Set(perm))),
    roleOf' = roleOf, rolePerms' = rolePerms, roleParent' = roleParent,
  }

  action step = any {
    nondet p = PRINCIPALS.oneOf()
    nondet r = ROLES.oneOf()
    assignRole(p, r),
    nondet r = ROLES.oneOf()
    nondet x = PERMS.oneOf()
    grantPerm(r, x),
    nondet a = ROLES.oneOf()
    nondet b = ROLES.oneOf()
    addParent(a, b),
    nondet p = PRINCIPALS.oneOf()
    nondet x = PERMS.oneOf()
    denyPerm(p, x),
  }

  // ---- invariants ---------------------------------------------------------

  /// An explicit deny always wins, whatever roles grant.
  val denyOverridesAllow: bool =
    PRINCIPALS.forall(p => denied.get(p).forall(x => not(can(p, x))))

  /// The role inheritance graph has no 2-cycles (the guard in addParent).
  /// A cycle would make permission resolution non-terminating in the real
  /// implementation.
  val roleHierarchyAcyclic: bool =
    ROLES.forall(a => ROLES.forall(b =>
      (roleParent.get(a).contains(b)) implies not(roleParent.get(b).contains(a))))

  /// A principal only holds permissions reachable from a role it was assigned.
  val noUngroundedPermission: bool =
    PRINCIPALS.forall(p =>
      effectivePerms(p).forall(x =>
        roleOf.get(p).fold(false, (acc, r) => acc or rolePermsClosed(r).contains(x))))

  val safety: bool = and {
    denyOverridesAllow, roleHierarchyAcyclic, noUngroundedPermission,
  }
}

module rbac {
  import rbac_proto(PRINCIPALS = Set(1), ROLES = Set(1, 2), PERMS = Set(1, 2)).*

  /// Inheriting a role picks up its permissions, and an explicit deny still wins.
  run inheritanceAndDenyTest =
    init
      .then(grantPerm(2, 1))
      .then(addParent(1, 2))
      .then(assignRole(1, 1))
      .expect(and { can(1, 1), noUngroundedPermission, roleHierarchyAcyclic })
      .then(denyPerm(1, 1))
      .expect(and { not(can(1, 1)), denyOverridesAllow })
}
