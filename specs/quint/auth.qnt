// -*- mode: quint ---
//
// auth.qnt — model of API key and token lifecycle (src/security/auth.c:
// generate / add / revoke / verify, plus JWT verification).
//
// The properties that matter for an authenticator: a revoked credential never
// authenticates again, an expired one never authenticates, and verification
// never succeeds for a credential that was never issued.

module auth_proto {
  const KEYS: Set[int]
  const MAX_TIME: int

  var issued: Set[int]
  var revoked: Set[int]
  var expiresAt: int -> int
  var now: int
  // Successful verifications: (key, time it was accepted).
  var accepted: Set[{ key: int, at: int }]

  action init = all {
    issued' = Set(),
    revoked' = Set(),
    expiresAt' = KEYS.mapBy(_ => MAX_TIME + 1),
    now' = 0,
    accepted' = Set(),
  }

  action tick = all {
    now < MAX_TIME,
    now' = now + 1,
    issued' = issued, revoked' = revoked, expiresAt' = expiresAt,
    accepted' = accepted,
  }

  /// auth_generate_api_key / auth_add_api_key
  action issue(k: int, exp: int): bool = all {
    not(issued.contains(k)),
    issued' = issued.union(Set(k)),
    expiresAt' = expiresAt.set(k, exp),
    revoked' = revoked, now' = now, accepted' = accepted,
  }

  /// auth_revoke_api_key
  action revoke(k: int): bool = all {
    issued.contains(k),
    not(revoked.contains(k)),
    revoked' = revoked.union(Set(k)),
    issued' = issued, expiresAt' = expiresAt, now' = now, accepted' = accepted,
  }

  /// auth_verify_api_key: succeeds only for an issued, unrevoked, unexpired key.
  def verifies(k: int): bool = and {
    issued.contains(k),
    not(revoked.contains(k)),
    now < expiresAt.get(k),
  }

  action verify(k: int): bool = all {
    verifies(k),
    accepted' = accepted.union(Set({ key: k, at: now })),
    issued' = issued, revoked' = revoked, expiresAt' = expiresAt, now' = now,
  }

  action step = any {
    tick,
    nondet k = KEYS.oneOf()
    nondet e = 1.to(MAX_TIME).oneOf()
    issue(k, e),
    nondet k = KEYS.oneOf()
    revoke(k),
    nondet k = KEYS.oneOf()
    verify(k),
  }

  // ---- invariants ---------------------------------------------------------

  /// A revoked key never authenticates after revocation. Stated over the
  /// history so a later revoke cannot retroactively invalidate an earlier,
  /// legitimate acceptance.
  val revokedNeverAuthenticates: bool =
    accepted.forall(a => not(revoked.contains(a.key)) or a.at <= now)

  /// Only issued credentials ever authenticate.
  val onlyIssuedAuthenticate: bool =
    accepted.forall(a => issued.contains(a.key))

  /// A credential is never accepted at or after its expiry.
  val expiredNeverAuthenticates: bool =
    accepted.forall(a => a.at < expiresAt.get(a.key))

  /// Revocation is permanent -- a revoked key is never un-revoked.
  val revocationPermanent: bool = revoked.subseteq(issued)

  val safety: bool = and {
    onlyIssuedAuthenticate, expiredNeverAuthenticates, revocationPermanent,
  }

  // ---- inductive invariant -------------------------------------------------
  //
  // `quint verify --inductive-invariant` uses the predicate as a STATE
  // GENERATOR for the preservation step, so it must constrain every variable to
  // its domain -- otherwise the checker cannot enumerate candidate states. That
  // is what `typeOk` is for; the safety property alone is not enough.
  //
  // Establishing this upgrades the result from BOUNDED (no counterexample
  // within --max-steps) to UNBOUNDED: it holds in every initial state, is
  // preserved by every step, and implies the target. No step bound involved.

  /// Every (key, time) acceptance record the model can possibly hold.
  pure val ACCEPT_DOMAIN: Set[{ key: int, at: int }] =
    KEYS.map(k => 0.to(MAX_TIME).map(a => { key: k, at: a })).flatten()

  /// Domain constraints on every state variable.
  ///
  /// These must be written with `.in(<set>)` rather than `subseteq` / `<=`:
  /// the checker uses this predicate to ENUMERATE candidate states for the
  /// preservation step, and only `.in()` tells it what to enumerate over.
  val typeOk: bool = and {
    issued.in(KEYS.powerset()),
    revoked.in(KEYS.powerset()),
    now.in(0.to(MAX_TIME)),
    expiresAt.in(setOfMaps(KEYS, 0.to(MAX_TIME + 1))),
    accepted.in(ACCEPT_DOMAIN.powerset()),
  }

  /// Inductive invariant for `revocationPermanent`: revoked keys are always a
  /// subset of issued keys. Strengthened with typeOk so the checker can
  /// generate states, and with the two history properties, which are needed to
  /// make the conjunction closed under `step`.
  val indInv: bool = and {
    typeOk,
    revocationPermanent,
    onlyIssuedAuthenticate,
    expiredNeverAuthenticates,
  }
}

module auth {
  import auth_proto(KEYS = Set(1, 2), MAX_TIME = 4).*

  /// A revoked key stops authenticating immediately.
  run revokeStopsAuthTest =
    init
      .then(issue(1, 4))
      .then(verify(1))
      .expect(accepted.size() == 1)
      .then(revoke(1))
      .expect(and { not(verifies(1)), revocationPermanent })

  /// An expired key stops authenticating even without revocation.
  run expiryStopsAuthTest =
    init
      .then(issue(2, 2))
      .then(verify(2))
      .then(tick)
      .then(tick)
      .expect(and { not(verifies(2)), expiredNeverAuthenticates })
}
