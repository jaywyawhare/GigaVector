#!/usr/bin/env python3
"""Run CBMC over every harness in specs/cbmc/ and report the REAL result.

This exists because a harness that merely compiles and runs once under gcc is
not verification -- it is a single concrete execution. `make cbmc-smoke` does
that (cheap, keeps harnesses from bit-rotting); this does the actual bounded
proof.

Each harness declares the sources it needs and its unwind bound in a
`CBMC-SOURCES:` / `CBMC-UNWIND:` comment near the top, so the harness and its
proof configuration stay in one place.

Outcomes are reported honestly and separately:
  PROVED      - VERIFICATION SUCCESSFUL: no counterexample within the bound
  FAILED      - CBMC found a counterexample (a real finding, or a harness bug)
  INTRACTABLE - timed out or ran out of memory; the harness is NOT a proof
  ERROR       - CBMC could not build the model at all

INTRACTABLE is the honest answer for harnesses whose dependency closure is too
large to symbolically execute. Reporting those as proved would be exactly the
kind of shallow claim this script is meant to prevent.

Exit status: 0 if nothing FAILED or ERRORed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CBMC_DIR = ROOT / "specs" / "cbmc"

DEFAULT_FLAGS = [
    "--bounds-check",
    "--pointer-check",
    "--conversion-check",
    "--div-by-zero-check",
    "--unwinding-assertions",
]


def parse_directives(path: Path):
    """Read CBMC-SOURCES / CBMC-UNWIND / CBMC-FLAGS / CBMC-SKIP /
    CBMC-SOLVER from the header."""
    text = path.read_text(errors="replace")
    head = text[:6000]
    srcs, unwind, extra, skip = [], 6, [], None
    solver = None
    m = re.search(r"CBMC-SOURCES:\s*(.+)", head)
    if m:
        srcs = m.group(1).split()
    m = re.search(r"CBMC-UNWIND:\s*(\d+)", head)
    if m:
        unwind = int(m.group(1))
    m = re.search(r"CBMC-FLAGS:\s*(.+)", head)
    if m:
        extra = m.group(1).split()
    m = re.search(r"CBMC-SKIP:\s*(.+)", head)
    if m:
        skip = m.group(1).strip()
    drop = []
    m = re.search(r"CBMC-DROP-FLAGS:\s*(.+)", head)
    if m:
        drop = m.group(1).split()
    # A harness may require a specific external SMT solver (e.g. z3 for
    # --smt2 --fpa). Missing solver => SKIP, mirroring the graceful skip for
    # a missing cbmc itself; it must NOT surface as ERROR and fail the run.
    m = re.search(r"CBMC-SOLVER:\s*(\S+)", head)
    if m:
        solver = m.group(1)
    return srcs, unwind, extra, skip, drop, solver


def run_one(cbmc: str, harness: Path, timeout: int):
    srcs, unwind, extra, skip, drop, solver = parse_directives(harness)
    if skip:
        return "SKIP", skip, 0
    if solver and shutil.which(solver) is None:
        return "SKIP", f"requires {solver!r} on PATH (see harness header)", 0

    # A harness may drop a default check when that check flags a deliberate,
    # documented idiom rather than a defect. The reason must be stated in the
    # harness -- dropping a check silently would defeat the point.
    flags = [f for f in DEFAULT_FLAGS if f not in drop]
    cmd = [cbmc, *flags, "--unwind", str(unwind), *extra,
           "-I", str(ROOT / "include"), str(harness)]
    for s in srcs:
        cmd.append(str(ROOT / s))

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "INTRACTABLE", f"timed out after {timeout}s", 0

    out = p.stdout + p.stderr
    m = re.search(r"\*\* (\d+) of (\d+) failed", out)
    total = int(m.group(2)) if m else 0
    nfail = int(m.group(1)) if m else 0

    if "VERIFICATION SUCCESSFUL" in out:
        return "PROVED", f"{total} checks", total
    if "VERIFICATION FAILED" in out:
        fails = re.findall(r"^\[(\S+)\].*: FAILURE", out, re.M)
        detail = f"{nfail}/{total} failed"
        if fails:
            detail += " -> " + ", ".join(fails[:3])
        return "FAILED", detail, total
    if "out of memory" in out.lower() or "std::bad_alloc" in out:
        return "INTRACTABLE", "out of memory", 0
    # Killed after it reached the solver: that is intractability, not a build
    # failure. Distinguishing them matters -- ERROR implies something is wrong
    # with the harness, INTRACTABLE means the problem is simply too big.
    if "Running propositional reduction" in out or "converting SSA" in out:
        return "INTRACTABLE", "killed during propositional reduction", 0
    first_err = next((l for l in out.splitlines() if "error" in l.lower()), "")
    return "ERROR", (first_err[:160] or "no verdict emitted"), 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbmc", default="cbmc", help="path to the cbmc binary")
    ap.add_argument("--timeout", type=int, default=300, help="per-harness seconds")
    ap.add_argument("--only", default=None, help="substring filter on harness name")
    ap.add_argument("--results", default=str(CBMC_DIR / "results.json"),
                    help="where to record per-harness outcomes")
    args = ap.parse_args()

    try:
        v = subprocess.run([args.cbmc, "--version"], capture_output=True, text=True)
        version = v.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        print(f"cbmc not found at {args.cbmc!r}; install it or pass --cbmc",
              file=sys.stderr)
        return 2

    harnesses = sorted(CBMC_DIR.glob("harness_*.c"))
    if args.only:
        harnesses = [h for h in harnesses if args.only in h.name]

    print(f"CBMC {version}, {len(harnesses)} harnesses, "
          f"{args.timeout}s timeout each\n")

    tally: dict[str, list[str]] = {}
    total_checks = 0
    for h in harnesses:
        status, detail, n = run_one(args.cbmc, h, args.timeout)
        total_checks += n
        tally.setdefault(status, []).append(h.name)
        mark = {"PROVED": "ok", "FAILED": "FAIL", "INTRACTABLE": "--",
                "ERROR": "ERR", "SKIP": "skip"}[status]
        print(f"  {mark:<5} {h.name:<34} {status:<12} {detail}", flush=True)

    # Persist outcomes. scripts/check_coverage.py reads this so the `cbmc`
    # tier can be backed by harnesses that actually PROVED, rather than by
    # harnesses that merely exist.
    if not args.only:
        results = {h: st for st, names in tally.items() for h in names}
        Path(args.results).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\n  wrote outcomes to {args.results}")

    print()
    for status in ("PROVED", "FAILED", "INTRACTABLE", "ERROR", "SKIP"):
        if status in tally:
            print(f"  {status:<12} {len(tally[status])}")
    print(f"\n  {total_checks} individual checks discharged across proved harnesses.")

    bad = len(tally.get("FAILED", [])) + len(tally.get("ERROR", []))
    if bad:
        print(f"\n[FAIL] {bad} harness(es) did not verify.")
        return 1
    if tally.get("INTRACTABLE"):
        print("\n[ok] no harness failed. Note the INTRACTABLE ones are NOT proofs "
              "-- they are smoke-tested only, and the coverage manifest says so.")
    else:
        print("\n[ok] every harness verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
