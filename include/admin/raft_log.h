#ifndef GIGAVECTOR_RAFT_LOG_H
#define GIGAVECTOR_RAFT_LOG_H

#include <stddef.h>
#include <stdint.h>

#include "admin/raft.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file raft_log.h
 * @brief Crash-safe on-disk persistence for a Raft node's log + vote.
 *
 * Raft requires currentTerm, votedFor and the log to survive a restart (else a
 * node can double-vote or lose committed entries). This is an append-only file
 * of META / ENTRY / TRUNC records, fsync'd on every write, that backs the
 * raft_create() persist_log/truncate_log/persist callbacks. On open it replays
 * the records, then rewrites the file in compacted form so it stays bounded.
 */

typedef struct GV_RaftLog GV_RaftLog;

/** Open (creating if absent) the durable log at @p path, replaying its state.
 *  Returns NULL on error. */
GV_RaftLog *raft_log_open(const char *path);

/** Close and free (NULL-safe). Does not delete the file. */
void raft_log_close(GV_RaftLog *log);

/** Durably record currentTerm / votedFor. 0 on success, -1 on failure. */
int raft_log_set_meta(GV_RaftLog *log, uint64_t term, int voted_for);

/** Durably append the entry at 1-based @p index. 0 on success, -1 on failure. */
int raft_log_append(GV_RaftLog *log, uint64_t index, uint64_t term,
                    const void *data, size_t len);

/** Durably drop entries after 1-based @p keep_upto. 0 on success, -1 on failure. */
int raft_log_truncate(GV_RaftLog *log, uint64_t keep_upto);

/* Read-back for raft_restore() after raft_log_open(). */
uint64_t raft_log_term(const GV_RaftLog *log);
int      raft_log_voted_for(const GV_RaftLog *log);
size_t   raft_log_count(const GV_RaftLog *log);
/** Borrowed array of raft_log_count() entries (data pointers owned by the log). */
const GV_RaftEntry *raft_log_entries(const GV_RaftLog *log);

#ifdef __cplusplus
}
#endif

#endif /* GIGAVECTOR_RAFT_LOG_H */
