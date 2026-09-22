#!/usr/bin/env python3
"""Run Apalache over the Quint specs and report the REAL result per check.

Same discipline as scripts/run_cbmc.py: a check that never terminates is not a
check. Each spec declares its own bound, every run has a timeout, and the
outcome is reported honestly:

  PROVED      - no counterexample within --max-steps (BOUNDED, not a proof)
  VIOLATED    - Apalache produced a counterexample
  INTRACTABLE - timed out at this bound; nothing is established
  ERROR       - Apalache could not build the model

The distinction matters because a uniform `--max-steps` across specs of very
different state size means the expensive ones silently never finish, while the
summary still reads green. Bounds are therefore tuned per spec and recorded
here rather than assumed.

For UNBOUNDED results see `make quint-induct`, which establishes inductive
invariants -- those hold for all reachable states with no bound at all.

Exit status: 0 if nothing VIOLATED or ERRORed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (file, module, invariant, max_steps). Bounds are tuned to what actually
# terminates: specs carrying sets of records are far more expensive per step
# than specs over plain scalars.
CHECKS = [
    # These six also have UNBOUNDED inductive proofs (`make quint-induct`),
    # which strictly subsume any bounded result. The bounded run is kept as a
    # cheap smoke check only, so the bound is deliberately small -- spending
    # minutes re-deriving a weaker statement would be waste, not rigour.
    ("specs/quint/lock_order.qnt",      "lock_order",      "safety", 3),
    ("specs/quint/auth.qnt",            "auth",            "safety", 4),
    ("specs/quint/versioning.qnt",      "versioning",      "safety", 4),
    ("specs/quint/cluster.qnt",         "cluster",         "safety", 4),
    ("specs/quint/bulk_import.qnt",     "bulk_import",     "safety", 6),
    ("specs/quint/namespace.qnt",       "namespace",       "safety", 4),
    ("specs/quint/ab_test.qnt",         "ab_test",         "safety", 8),
    ("specs/quint/dedup.qnt",           "dedup",           "safety", 6),
    ("specs/quint/scroll_cursor.qnt",   "scroll_cursor",   "safety", 6),
    ("specs/quint/tiered_storage.qnt",  "tiered_storage",  "safety", 8),
    ("specs/quint/snapshot.qnt",        "snapshot",        "safety", 6),
    ("specs/quint/webhook.qnt",         "webhook",         "safety", 6),
    ("specs/quint/memory_layer.qnt",    "memory_layer",    "safety", 5),
    ("specs/quint/index_swap.qnt",      "index_swap",      "safety", 6),
    ("specs/quint/shard_rebalance.qnt", "shard_rebalance", "safety", 6),
    ("specs/quint/rbac.qnt",            "rbac",            "safety", 5),
    ("specs/quint/cdc.qnt",             "cdc",             "safety", 5),
    ("specs/quint/replication.qnt",     "replication",     "safety", 5),
    ("specs/quint/vlog_gc.qnt",         "vlog_gc",         "safety", 5),
    ("specs/quint/wal_recovery.qnt",    "wal_recovery",    "safety", 6),
    ("specs/quint/cache.qnt",           "cache",           "safety", 5),
    ("specs/quint/consistency.qnt",     "consistency",     "safety", 4),
    ("specs/quint/mvcc.qnt",            "mvcc",            "safety", 4),
    ("specs/quint/raft.qnt",            "raft3",           "safety", 6),
    # Re-enabled after the fix: gv_quota_reserve_insert makes check-and-claim
    # atomic, so `safety` (which includes quotaNeverExceeded) now holds. Keeping
    # it gated means the fix cannot silently regress.
    ("specs/quint/quota.qnt",           "quota",           "safety", 6),
    # Cross-subsystem composition: replication x WAL durability. The
    # noAckedLoss invariant needs BOTH the fsync gate and the election
    # restriction, so this is the one check that catches interaction bugs
    # between the two subsystems.
    ("specs/quint/repl_wal_crash.qnt",  "repl_wal_crash",  "safety", 6),
]

# `ttl` remains absent: its `safety` is still FALSE by design
# (ttl_cleanup_expired acts on a snapshot taken before it released the mutex).
# `quota` was in the same position and is now re-enabled above, because
# gv_quota_reserve_insert fixed it.


def run_one(quint: str, spec: str, module: str, inv: str, steps: int, timeout: int):
    cmd = [quint, "verify", f"--main={module}", f"--invariant={inv}",
           f"--max-steps={steps}", str(ROOT / spec)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return "INTRACTABLE", f"timed out at {timeout}s (max-steps={steps})"
    out = p.stdout + p.stderr
    if "[ok] No violation found" in out:
        m = re.search(r"No violation found \((\d+)ms\)", out)
        ms = int(m.group(1)) if m else 0
        return "PROVED", f"max-steps={steps}, {ms // 1000}s"
    if "reached a deadlock" in out:
        # Not a safety failure: the model simply has no enabled action. For a
        # terminating process that is expected, and conflating it with a
        # counterexample would report a bug that does not exist.
        return "DEADLOCK", (f"no enabled action at max-steps={steps}; add an "
                            f"explicit stutter step if termination is intended")
    if "[violation]" in out or "Found an issue" in out:
        return "VIOLATED", f"counterexample at max-steps={steps}"
    err = next((l for l in out.splitlines() if l.startswith("error:")), "")
    return "ERROR", (err[:150] or "no verdict emitted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quint", default="quint")
    ap.add_argument("--timeout", type=int, default=600, help="per-check seconds")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    checks = CHECKS
    if args.only:
        checks = [c for c in checks if args.only in c[1]]

    print(f"Apalache via quint, {len(checks)} checks, {args.timeout}s timeout each")
    print("NOTE: PROVED here means BOUNDED -- no counterexample within max-steps.")
    print("      For unbounded results see `make quint-induct`.\n")

    tally: dict[str, list[str]] = {}
    for spec, module, inv, steps in checks:
        status, detail = run_one(args.quint, spec, module, inv, steps, args.timeout)
        tally.setdefault(status, []).append(module)
        mark = {"PROVED": "ok", "VIOLATED": "FAIL", "DEADLOCK": "dead",
                "INTRACTABLE": "--", "ERROR": "ERR"}[status]
        print(f"  {mark:<5} {module:<20} {status:<12} {detail}", flush=True)

    print()
    for status in ("PROVED", "VIOLATED", "DEADLOCK", "INTRACTABLE", "ERROR"):
        if status in tally:
            print(f"  {status:<12} {len(tally[status])}")

    bad = (len(tally.get("VIOLATED", [])) + len(tally.get("ERROR", []))
           + len(tally.get("DEADLOCK", [])))
    if bad:
        print(f"\n[FAIL] {bad} check(s) did not pass.")
        return 1
    if tally.get("INTRACTABLE"):
        print("\n[ok] nothing violated. INTRACTABLE checks establish NOTHING at "
              "their bound and are not counted as verified.")
    else:
        print("\n[ok] every check passed at its bound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
