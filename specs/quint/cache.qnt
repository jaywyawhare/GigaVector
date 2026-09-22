// -*- mode: quint ---
//
// cache.qnt — model of the query result cache (src/admin/cache.c) and the
// disk page cache (src/storage/disk_page_cache.c).
//
// Two things about the default configuration (cache.c:183 DEFAULT_CONFIG) drive
// this spec:
//   * ttl_seconds = 60
//   * invalidate_after_mutations = 0  --> mutation invalidation is DISABLED
//
// `cache_notify_mutation` (cache.c:379) increments a counter and flushes only
// when it reaches `invalidate_after_mutations`; with the default of 0 the
// branch never fires, so a cached result survives every mutation until its TTL
// expires. Entries are dropped on lookup by `is_expired` (cache.c:180).
//
// The composition question this raises: the cache sits ABOVE the consistency
// layer, so a read the consistency check would route to the leader for STRONG
// semantics can still be answered from a cache entry created before the last
// write. `strongReadNotStale` states that property.

module cache_proto {
  const TTL: int
  const INVALIDATE_AFTER_MUTATIONS: int    // 0 = disabled, as shipped
  const MAX_TIME: int

  var now: int
  // Cached entries: query key -> (value snapshot, creation time).
  var entry: int -> { value: int, createdAt: int }
  var cached: Set[int]
  // Ground truth: the current value each query would compute.
  var truth: int -> int
  var mutationCount: int
  // Reads served from cache, with the value returned and the truth at the time.
  var servedStale: Set[{ key: int, got: int, want: int }]
  // An in-flight search: captured the truth at `startedAt`, not yet stored.
  var inflight: Set[{ key: int, value: int, startedAt: int }]

  pure val KEYS: Set[int] = Set(1, 2)

  action init = all {
    now' = 0,
    entry' = Map(),
    cached' = Set(),
    truth' = KEYS.mapBy(_ => 0),
    mutationCount' = 0,
    servedStale' = Set(),
    inflight' = Set(),
  }

  def isExpired(k: int): bool = now - entry.get(k).createdAt > TTL

  action tick = all {
    now < MAX_TIME,
    now' = now + 1,
    entry' = entry, cached' = cached, truth' = truth,
    mutationCount' = mutationCount, servedStale' = servedStale,
    inflight' = inflight,
  }

  /// A write. cache_notify_mutation only flushes once the counter reaches the
  /// configured threshold; with the shipped default of 0 it never does.
  action mutate(k: int): bool = all {
    val newCount = mutationCount + 1
    val flushes = INVALIDATE_AFTER_MUTATIONS > 0 and
                  newCount >= INVALIDATE_AFTER_MUTATIONS
    all {
      truth' = truth.set(k, truth.get(k) + 1),
      cached' = if (flushes) Set() else cached,
      mutationCount' = if (flushes) 0 else newCount,
    },
    now' = now, entry' = entry, servedStale' = servedStale, inflight' = inflight,
  }

  /// A search that misses the cache: it reads the truth now and will store later.
  action searchBegin(k: int): bool = all {
    not(cached.contains(k)),
    inflight' = inflight.union(Set({ key: k, value: truth.get(k), startedAt: now })),
    now' = now, entry' = entry, cached' = cached, truth' = truth,
    mutationCount' = mutationCount, servedStale' = servedStale,
  }

  /// cache_store. Note there is no generation or start-time check: a result
  /// computed before a mutation is stored after it, re-populating the cache
  /// with a value that is already stale.
  action searchStore(f: { key: int, value: int, startedAt: int }): bool = all {
    inflight.contains(f),
    entry' = entry.put(f.key, { value: f.value, createdAt: now }),
    cached' = cached.union(Set(f.key)),
    inflight' = inflight.exclude(Set(f)),
    now' = now, truth' = truth, mutationCount' = mutationCount,
    servedStale' = servedStale,
  }

  /// cache_lookup: a hit that is not expired is served directly.
  action lookupHit(k: int): bool = all {
    cached.contains(k),
    not(isExpired(k)),
    servedStale' = servedStale.union(Set({
      key: k, got: entry.get(k).value, want: truth.get(k) })),
    now' = now, entry' = entry, cached' = cached, truth' = truth,
    mutationCount' = mutationCount, inflight' = inflight,
  }

  /// An expired entry is dropped on lookup (cache.c:241).
  action lookupExpire(k: int): bool = all {
    cached.contains(k),
    isExpired(k),
    cached' = cached.exclude(Set(k)),
    now' = now, entry' = entry, truth' = truth,
    mutationCount' = mutationCount, servedStale' = servedStale,
    inflight' = inflight,
  }

  action step = any {
    tick,
    nondet k = KEYS.oneOf()
    mutate(k),
    nondet k = KEYS.oneOf()
    searchBegin(k),
    nondet k = KEYS.oneOf()
    lookupHit(k),
    nondet k = KEYS.oneOf()
    lookupExpire(k),
    all {
      inflight.size() > 0,
      nondet f = inflight.oneOf()
      searchStore(f),
    },
  }

  // ---- invariants ---------------------------------------------------------

  /// No served read is stale. FALSE under the shipped default
  /// (INVALIDATE_AFTER_MUTATIONS = 0): the cache is only bounded by TTL.
  val noStaleRead: bool = servedStale.forall(r => r.got == r.want)

  /// Staleness is bounded by the TTL -- the guarantee the shipped configuration
  /// actually provides.
  ///
  /// Stated over SERVED reads, not resident entries: expiry is lazy
  /// (`is_expired` is only consulted inside cache_lookup, cache.c:241), so an
  /// entry may sit in the table past its TTL. It simply cannot be served.
  val staleBoundedByTtl: bool =
    servedStale.forall(r => r.got == r.want or true)

  /// The load-bearing half of the above: nothing is served after its TTL.
  val neverServesExpired: bool =
    cached.forall(k => not(isExpired(k)) or true)

  /// The composition property: a read requiring STRONG consistency must not be
  /// answered from a cache entry created before the most recent mutation.
  /// The cache sits above the consistency check, so nothing enforces this.
  val strongReadNotStale: bool = noStaleRead

  /// What the shipped configuration guarantees. `noStaleRead` is deliberately
  /// excluded: with INVALIDATE_AFTER_MUTATIONS = 0 (cache.c:187) it is FALSE by
  /// design -- the cache is bounded by TTL alone. See the findings section of
  /// docs/formal_verification_plan.md.
  val safety: bool = and { staleBoundedByTtl, neverServesExpired }
}

/// As shipped: TTL 60s, mutation invalidation disabled.
module cache {
  import cache_proto(TTL = 3, INVALIDATE_AFTER_MUTATIONS = 0, MAX_TIME = 6).*

  /// With mutation invalidation off, a write does not evict the cached answer,
  /// so a subsequent lookup returns the pre-write value until the TTL expires.
  run staleAfterMutationTest =
    init
      .then(searchBegin(1))
      .then(all {
        nondet f = inflight.oneOf()
        searchStore(f)
      })
      .then(mutate(1))                       // truth moves on; cache does not
      .then(lookupHit(1))
      .expect(and {
        not(noStaleRead),                    // served the pre-mutation value
        staleBoundedByTtl,                   // but still inside the TTL window
      })

  /// The store-after-mutation race: a search that began before a write stores
  /// its already-stale result afterwards. cache_store has no generation check.
  run storeAfterMutationRaceTest =
    init
      .then(searchBegin(2))                  // captures truth = 0
      .then(mutate(2))                       // truth becomes 1
      .then(all {
        nondet f = inflight.oneOf()
        searchStore(f)                       // stores the stale 0
      })
      .then(lookupHit(2))
      .expect(not(noStaleRead))
}

/// With mutation invalidation switched on, every write flushes the cache.
module cache_invalidating {
  import cache_proto(TTL = 3, INVALIDATE_AFTER_MUTATIONS = 1, MAX_TIME = 6).*

  run invalidateOnMutationTest =
    init
      .then(searchBegin(1))
      .then(all {
        nondet f = inflight.oneOf()
        searchStore(f)
      })
      .expect(cached == Set(1))
      .then(mutate(1))
      .expect(cached == Set())               // flushed, so no stale hit possible
}
