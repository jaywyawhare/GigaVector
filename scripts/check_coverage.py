#!/usr/bin/env python3
"""Enforce specs/coverage.yml: every C source file has a verification tier, and
every tier reference points at something that actually exists.

Two failure modes this exists to prevent:

  1. New code silently escaping verification. A file added to src/ without a
     manifest entry fails the build.
  2. The manifest lying. A `quint` entry naming a spec that was never written,
     or a `cbmc` entry naming a missing harness, fails the build -- otherwise
     the coverage table becomes aspirational documentation.

`property` and `exempt` are legitimate classifications, not verification. They
are counted separately and reported, so the real formal-coverage number is
always visible rather than hidden behind a green check.

A third failure mode this guards: a `cbmc` entry claiming a whole family of
files when the harness only exercises one representative. Claims are checked
against each harness's `CBMC-SOURCES:` directive -- the mechanical record of
what CBMC actually builds -- so coverage cannot inflate by association.

Exit status: 0 consistent, 1 inconsistent, 2 manifest unreadable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "specs" / "coverage.yml"
SRC = ROOT / "src"
QUINT_DIR = ROOT / "specs" / "quint"
CBMC_DIR = ROOT / "specs" / "cbmc"

TIERS = ("quint", "cbmc", "cbmc-pending", "property", "exempt")
FORMAL = ("quint", "cbmc")


def load_manifest(path: Path):
    try:
        import yaml
    except ImportError:
        print("PyYAML is required: pip install pyyaml", file=sys.stderr)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return None


def harness_sources(harness: Path) -> set[str]:
    """The sources a harness actually symbolically executes, from its
    `CBMC-SOURCES:` directive. This is the mechanical ground truth for what a
    cbmc claim is allowed to cover."""
    m = re.search(r"CBMC-SOURCES:\s*(.+)", harness.read_text(errors="replace")[:6000])
    if not m:
        return set()
    return {s.replace("src/", "") for s in m.group(1).split()}


def main() -> int:
    doc = load_manifest(MANIFEST)
    if not doc or "tiers" not in doc:
        print(f"error: {MANIFEST} missing or has no `tiers:` key", file=sys.stderr)
        return 2

    actual = {p.relative_to(SRC).as_posix() for p in SRC.rglob("*.c")}
    claimed: dict[str, tuple[str, str]] = {}
    duplicates: list[str] = []
    bad_refs: list[str] = []

    for tier, groups in (doc["tiers"] or {}).items():
        if tier not in TIERS:
            print(f"error: unknown tier {tier!r} in manifest", file=sys.stderr)
            return 2
        for group in groups or []:
            ref = group.get("ref", "-")
            # A tier reference must resolve to a real artifact.
            if tier == "quint" and not (QUINT_DIR / ref).exists():
                bad_refs.append(f"quint ref {ref!r} does not exist in specs/quint/")
            if tier.startswith("cbmc") and not (CBMC_DIR / ref).exists():
                bad_refs.append(f"cbmc ref {ref!r} does not exist in specs/cbmc/")
            # A cbmc entry may only claim files the harness genuinely executes.
            # Without this, one harness "covers" a whole family by association
            # and the coverage number silently inflates.
            if tier.startswith("cbmc") and (CBMC_DIR / ref).exists():
                exercised = harness_sources(CBMC_DIR / ref)
                if exercised:
                    for f in group.get("files", []) or []:
                        if f not in exercised:
                            bad_refs.append(
                                f"cbmc ref {ref!r} claims {f} but does not "
                                f"symbolically execute it (not in CBMC-SOURCES)")
            for f in group.get("files", []) or []:
                if f in claimed:
                    duplicates.append(f)
                claimed[f] = (tier, ref)

    unclassified = sorted(actual - claimed.keys())
    stale = sorted(claimed.keys() - actual)

    counts = {t: 0 for t in TIERS}
    for f, (tier, _) in claimed.items():
        if f in actual:
            counts[tier] += 1

    total = len(actual)
    formal = sum(counts[t] for t in FORMAL)

    print(f"coverage manifest: {MANIFEST.relative_to(ROOT)}")
    print(f"  {total} C source files under src/\n")
    for t in TIERS:
        pct = 100.0 * counts[t] / total if total else 0.0
        print(f"    {t:<9} {counts[t]:>4}  ({pct:5.1f}%)")
    print(f"\n    {'FORMAL':<9} {formal:>4}  ({100.0*formal/total:5.1f}%)"
          f"   [quint + cbmc]")

    # A harness that exists is not a harness that proved. If the last
    # `make cbmc` run left results behind, report how much of the cbmc tier is
    # actually backed by a PROVED outcome -- that is the number that means
    # "verified", and it is normally lower than the tier size.
    results_path = CBMC_DIR / "results.json"
    if results_path.exists():
        try:
            outcomes = json.loads(results_path.read_text())
        except (OSError, ValueError):
            outcomes = {}
        proved_files = unproved = 0
        unproved_by = {}
        for f, (tier, ref) in claimed.items():
            if tier != "cbmc" or f not in actual:
                continue
            st = outcomes.get(ref)
            if st == "PROVED":
                proved_files += 1
            else:
                unproved += 1
                unproved_by.setdefault(st or "NOT RUN", []).append(ref)
        print(f"\n  of the {counts['cbmc']} cbmc-tier files, "
              f"{proved_files} are backed by a harness that PROVED "
              f"({100.0*proved_files/total:.1f}% of the tree).")
        if unproved:
            detail = ", ".join(f"{k}: {len(set(v))}" for k, v in sorted(unproved_by.items()))
            print(f"  {unproved} are NOT currently proved ({detail}) - "
                  f"their harnesses exist but do not discharge.")

    ok = True
    if unclassified:
        ok = False
        print(f"\n[FAIL] {len(unclassified)} file(s) not in the manifest - new code "
              f"must be classified before it can merge:")
        for f in unclassified:
            print(f"    {f}")
    if stale:
        ok = False
        print(f"\n[FAIL] {len(stale)} manifest entr(ies) reference files that no "
              f"longer exist:")
        for f in stale:
            print(f"    {f}")
    if duplicates:
        ok = False
        print(f"\n[FAIL] {len(duplicates)} file(s) classified more than once:")
        for f in sorted(set(duplicates)):
            print(f"    {f}  ({claimed[f][0]})")
    if bad_refs:
        ok = False
        print(f"\n[FAIL] {len(bad_refs)} manifest reference(s) point at missing "
              f"artifacts - the manifest must not over-claim:")
        for r in bad_refs:
            print(f"    {r}")

    if ok:
        print("\n[ok] every source file is classified and every reference resolves.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
