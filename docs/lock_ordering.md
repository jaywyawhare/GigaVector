# Lock Ordering

This document defines the canonical lock acquisition order for GigaVector.
Violating this order causes deadlocks. All code that acquires multiple locks
must follow this hierarchy from top (acquired first) to bottom (acquired last).

## Hierarchy

```
Level 1 (outermost)
  db->rwlock           (pthread_rwlock_t) — guards all vector data, index, and count

Level 2
  db->compaction_mutex (pthread_mutex_t) — guards compaction_running flag and thread

Level 3
  db->wal_mutex        (pthread_mutex_t) — guards WAL writes; never held across IO blocking
  db->observability_mutex (pthread_mutex_t) — guards latency histograms and QPS tracking

Level 4 (innermost)
  db->resource_mutex   (pthread_mutex_t) — guards resource limits and memory counters
```

## Rules

1. Always acquire `db->rwlock` before any other `db` mutex.
2. Never acquire `db->wal_mutex` while holding `db->observability_mutex`, or vice versa
   (they are siblings at level 3 — do not hold both simultaneously).
3. `db->resource_mutex` is always innermost; never acquire another lock while holding it.
4. Index-internal locks (e.g. inside HNSW, IVF) are always acquired **after** `db->rwlock`
   and are opaque to the db layer — do not acquire any `db` mutex after entering an
   index-internal lock.
5. For read-only operations use `pthread_rwlock_rdlock`; multiple readers are safe in parallel.
   Writes (insert, delete, update, compact) must use `pthread_rwlock_wrlock`.

## Critical paths

| Operation         | Locks acquired (in order)              |
|-------------------|----------------------------------------|
| `db_add_vector`   | rwlock(W) → wal_mutex → resource_mutex |
| `db_search`       | rwlock(R)                              |
| `db_search_batch` | rwlock(R) [worker threads share rdlock] |
| `db_compact`      | rwlock(W)                              |
| `db_record_latency` | observability_mutex                  |
| `db_get_memory_usage` | resource_mutex                     |
| `db_get_detailed_stats` | rwlock(R) → observability_mutex  |
| WAL replay (`db_replay_wal`) | rwlock(W) → wal_mutex        |

## Module-level locks (outside GV_Database)

These locks are independent of the db hierarchy. Never hold a `db` lock when
acquiring these, and never acquire these while holding a `db` lock.

| Module           | Lock variable                     | Protects                        |
|------------------|-----------------------------------|---------------------------------|
| replication.c    | `repl->lock`                      | replication state and peer list |
| knowledge_graph.c| `graph->lock`                     | graph node and edge maps        |
| graph_db.c       | `gdb->lock`                       | graph adjacency list            |
| mvcc.c           | `mvcc->lock`                      | transaction version list        |
| namespace.c      | `ns->lock`                        | namespace registry              |
| shard.c          | `shard->lock`                     | shard routing table             |
| streaming.c      | `stream->lock`                    | stream subscriber list          |
| ttl.c            | `ttl->lock`                       | TTL expiration queue            |

## Adding new locks

When introducing a new lock:
1. Decide which level it belongs to in the hierarchy above.
2. Update this document.
3. Annotate every acquisition site with a comment citing the level (e.g.
   `/* level 3 — acquire after rwlock */`).
