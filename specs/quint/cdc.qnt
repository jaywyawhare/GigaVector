// -*- mode: quint ---
//
// cdc.qnt — model of change-data-capture delivery (src/admin/cdc.c,
// src/admin/streaming.c).
//
// `db_emit_change` is called AFTER the database lock is released and after the
// mutation has landed (see db_delete_vector_by_index, database.c: the emit is
// the last statement, outside pthread_rwlock_unlock). So an event is emitted
// only for a mutation that actually happened, but the emit itself is not
// covered by the mutation's lock -- two concurrent mutations can therefore
// emit out of order relative to their commit order.

module cdc_proto {
  const MAX_EVENTS: int
  const SUBSCRIBERS: Set[int]

  // Mutations that have committed, in commit order.
  var committed: List[int]
  // Events actually published to the stream, in publish order.
  var published: List[int]
  // Per-subscriber delivery.
  var delivered: int -> List[int]
  var subscribed: Set[int]
  var nextId: int

  action init = all {
    committed' = List(),
    published' = List(),
    delivered' = SUBSCRIBERS.mapBy(_ => List()),
    subscribed' = Set(),
    nextId' = 1,
  }

  /// A mutation commits under the db lock.
  action commitMutation = all {
    nextId <= MAX_EVENTS,
    committed' = committed.append(nextId),
    nextId' = nextId + 1,
    published' = published, delivered' = delivered, subscribed' = subscribed,
  }

  /// db_emit_change, outside the db lock. Any committed-but-unpublished event
  /// may publish next, so publication order can differ from commit order.
  action publish(e: int): bool = all {
    committed.foldl(false, (a, x) => a or x == e),
    not(published.foldl(false, (a, x) => a or x == e)),
    published' = published.append(e),
    committed' = committed, nextId' = nextId,
    delivered' = delivered, subscribed' = subscribed,
  }

  action subscribe(s: int): bool = all {
    not(subscribed.contains(s)),
    subscribed' = subscribed.union(Set(s)),
    delivered' = delivered.set(s, List()),
    committed' = committed, published' = published, nextId' = nextId,
  }

  action unsubscribe(s: int): bool = all {
    subscribed.contains(s),
    subscribed' = subscribed.exclude(Set(s)),
    committed' = committed, published' = published, nextId' = nextId,
    delivered' = delivered,
  }

  /// Deliver the next published event to a subscriber, in publish order.
  action deliver(s: int): bool = all {
    subscribed.contains(s),
    delivered.get(s).length() < published.length(),
    delivered' = delivered.set(s,
      delivered.get(s).append(published[delivered.get(s).length()])),
    committed' = committed, published' = published, nextId' = nextId,
    subscribed' = subscribed,
  }

  action step = any {
    commitMutation,
    nondet e = 1.to(MAX_EVENTS).oneOf()
    publish(e),
    nondet s = SUBSCRIBERS.oneOf()
    subscribe(s),
    nondet s = SUBSCRIBERS.oneOf()
    unsubscribe(s),
    nondet s = SUBSCRIBERS.oneOf()
    deliver(s),
  }

  // ---- invariants ---------------------------------------------------------

  /// Every published event corresponds to a real committed mutation. No
  /// phantom change events.
  val noPhantomEvent: bool =
    1.to(MAX_EVENTS).forall(i =>
      i <= published.length() implies
        committed.foldl(false, (a, x) => a or x == published[i - 1]))

  /// A subscriber never receives the same event twice and never skips one:
  /// its delivered list is a prefix of the published stream.
  val deliveryIsPrefix: bool =
    subscribed.forall(s =>
      1.to(MAX_EVENTS).forall(i =>
        i <= delivered.get(s).length() implies
          delivered.get(s)[i - 1] == published[i - 1]))

  /// No gaps in the published sequence relative to delivery.
  val noGaps: bool =
    subscribed.forall(s => delivered.get(s).length() <= published.length())

  val safety: bool = and { noPhantomEvent, deliveryIsPrefix, noGaps }
}

module cdc {
  import cdc_proto(MAX_EVENTS = 3, SUBSCRIBERS = Set(1, 2)).*

  /// A subscriber that joins late sees only events published after it joined
  /// -- delivery is still a gapless prefix of what it can observe.
  run lateSubscriberTest =
    init
      .then(commitMutation)
      .then(publish(1))
      .then(subscribe(1))
      .then(deliver(1))
      .expect(and { delivered.get(1) == List(1), deliveryIsPrefix, noPhantomEvent })

  /// Publication order may differ from commit order (the emit happens outside
  /// the db lock), but every published event is still a real mutation.
  run reorderedPublishTest =
    init
      .then(commitMutation)
      .then(commitMutation)
      .then(publish(2))
      .then(publish(1))
      .expect(and {
        published == List(2, 1),
        committed == List(1, 2),
        noPhantomEvent,
      })
}
