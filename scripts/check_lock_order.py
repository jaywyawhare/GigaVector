#!/usr/bin/env python3
"""Static check that the code respects docs/lock_ordering.md.

Deadlock freedom for a fixed lock hierarchy is exactly: never acquire a lock at
level N while already holding one at level >= N. This script reads the
hierarchy out of docs/lock_ordering.md (the single source of truth) and walks
every function body in src/ checking that rule.

The analysis is intraprocedural and syntactic -- it tracks lock/unlock calls in
source order within a function. That is deliberate: it cannot prove deadlock
freedom across call boundaries, but it catches the drift that code review
misses, at zero toolchain cost. Cross-function ordering is the job of
specs/quint/lock_order.qnt.

Exit status: 0 clean, 1 violations found, 2 could not parse the hierarchy.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "lock_ordering.md"
SRC = ROOT / "src"

# pthread_mutex_lock(&db->wal_mutex) / pthread_rwlock_rdlock(&db->rwlock) / ...
ACQUIRE = re.compile(
    r"\bpthread_(?:mutex_lock|mutex_trylock|rwlock_rdlock|rwlock_wrlock|rwlock_tryrdlock|rwlock_trywrlock)"
    r"\s*\(\s*&?\s*(?:\(\s*pthread_\w+_t\s*\*\s*\)\s*&?)?"
    r"(?:\(\s*\w+\s*\*\s*\)\s*)?"
    r"[\w\[\]\.\-\>\(\) ]*?(\w+)\s*\)"
)
RELEASE = re.compile(
    r"\bpthread_(?:mutex_unlock|rwlock_unlock)"
    r"\s*\(\s*&?\s*(?:\(\s*pthread_\w+_t\s*\*\s*\)\s*&?)?"
    r"(?:\(\s*\w+\s*\*\s*\)\s*)?"
    r"[\w\[\]\.\-\>\(\) ]*?(\w+)\s*\)"
)
# A function definition at column 0: `type name(args) {`
FUNC_START = re.compile(r"^[A-Za-z_][\w \*]*\b(\w+)\s*\([^;]*\)\s*\{?\s*$")


def parse_hierarchy(doc_path):
    """Extract `name -> level` from the ``` block under ## Hierarchy."""
    if not doc_path.exists():
        return None
    text = doc_path.read_text()
    m = re.search(r"##\s*Hierarchy\s*\n+```(.*?)```", text, re.S)
    if not m:
        return None
    levels, current = {}, None
    for line in m.group(1).splitlines():
        lv = re.match(r"\s*Level\s+(\d+)", line)
        if lv:
            current = int(lv.group(1))
            continue
        # e.g. "  db->txn_mutex        (pthread_mutex_t) — serializes ..."
        name = re.match(r"\s+(?:\w+\s*->\s*)?(\w+)\s+\(", line)
        if name and current is not None:
            levels[name.group(1)] = current
    return levels or None


def iter_functions(path):
    """Yield (function_name, start_line, [source lines]) using brace depth."""
    lines = path.read_text(errors="replace").splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = FUNC_START.match(lines[i])
        if not m:
            i += 1
            continue
        # find the opening brace (same line or the next)
        j = i
        if "{" not in lines[i]:
            if i + 1 < n and lines[i + 1].strip() == "{":
                j = i + 1
            else:
                i += 1
                continue
        depth, body, k = 0, [], j
        while k < n:
            depth += lines[k].count("{") - lines[k].count("}")
            body.append(lines[k])
            if depth <= 0 and k >= j:
                break
            k += 1
        yield m.group(1), i + 1, body
        i = k + 1


def strip_comments(line):
    return re.sub(r"//.*$", "", re.sub(r"/\*.*?\*/", "", line))


def check_file(path, levels):
    violations = []
    for fname, fline, body in iter_functions(path):
        held = []  # [(lockname, level, lineno)]
        for off, raw in enumerate(body):
            line = strip_comments(raw)
            for m in RELEASE.finditer(line):
                name = m.group(1)
                held = [h for h in held if h[0] != name]
            for m in ACQUIRE.finditer(line):
                name = m.group(1)
                if name not in levels:
                    continue  # module-local lock, outside the db hierarchy
                lvl = levels[name]
                worse = [h for h in held if h[1] >= lvl]
                if worse:
                    violations.append({
                        "file": str(path.relative_to(ROOT))
                                if path.is_relative_to(ROOT) else str(path),
                        "line": fline + off,
                        "func": fname,
                        "acquired": f"{name} (level {lvl})",
                        "while_holding": ", ".join(
                            f"{h[0]} (level {h[1]}, line {fline + h[2]})" for h in worse),
                    })
                held.append((name, lvl, off))
    return violations


SELF_TEST_SRC = """
/* synthetic: must be flagged -- takes level 1 while holding level 3 */
int bad_order(GV_Database *db) {
    pthread_mutex_lock(&db->wal_mutex);
    pthread_rwlock_wrlock(&db->rwlock);
    pthread_rwlock_unlock(&db->rwlock);
    pthread_mutex_unlock(&db->wal_mutex);
    return 0;
}

/* synthetic: must be flagged -- level-3 siblings held simultaneously */
int bad_siblings(GV_Database *db) {
    pthread_mutex_lock(&db->wal_mutex);
    pthread_mutex_lock(&db->observability_mutex);
    pthread_mutex_unlock(&db->observability_mutex);
    pthread_mutex_unlock(&db->wal_mutex);
    return 0;
}

/* synthetic: must be clean -- descending order, per the documented hierarchy */
int good_order(GV_Database *db) {
    pthread_rwlock_wrlock(&db->rwlock);
    pthread_mutex_lock(&db->wal_mutex);
    pthread_mutex_unlock(&db->wal_mutex);
    pthread_mutex_lock(&db->resource_mutex);
    pthread_mutex_unlock(&db->resource_mutex);
    pthread_rwlock_unlock(&db->rwlock);
    return 0;
}

/* synthetic: must be clean -- sequential, never overlapping */
int good_sequential(GV_Database *db) {
    pthread_mutex_lock(&db->wal_mutex);
    pthread_mutex_unlock(&db->wal_mutex);
    pthread_mutex_lock(&db->observability_mutex);
    pthread_mutex_unlock(&db->observability_mutex);
    return 0;
}
"""


def self_test(levels):
    """Prove the checker is not vacuous: it must flag known-bad orderings and
    pass known-good ones. Run in CI alongside the real scan."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "synthetic.c"
        f.write_text(SELF_TEST_SRC)
        found = check_file(f, levels)
    flagged = {v["func"] for v in found}
    expect_bad = {"bad_order", "bad_siblings"}
    expect_clean = {"good_order", "good_sequential"}
    missed = expect_bad - flagged
    spurious = flagged & expect_clean
    if missed or spurious:
        print("[FAIL] self-test: the checker is not working as intended")
        if missed:
            print(f"        missed violations in: {', '.join(sorted(missed))}")
        if spurious:
            print(f"        false positives in:   {', '.join(sorted(spurious))}")
        return False
    print(f"[ok] self-test: flagged {', '.join(sorted(expect_bad))}; "
          f"passed {', '.join(sorted(expect_clean))}")
    return True


def main():
    levels = parse_hierarchy(DOC)
    if not levels:
        print(f"error: could not parse the lock hierarchy from {DOC}", file=sys.stderr)
        return 2

    if not self_test(levels):
        return 1
    print()

    print(f"lock hierarchy from {DOC.relative_to(ROOT)}:")
    for name, lvl in sorted(levels.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"  level {lvl}: {name}")
    print()

    all_v, nfiles = [], 0
    for path in sorted(SRC.rglob("*.c")):
        nfiles += 1
        all_v.extend(check_file(path, levels))

    if not all_v:
        print(f"[ok] {nfiles} files scanned, no lock-order violations "
              f"against the documented hierarchy.")
        return 0

    print(f"[FAIL] {len(all_v)} lock-order violation(s) in {nfiles} files:\n")
    for v in all_v:
        print(f"  {v['file']}:{v['line']}  in {v['func']}()")
        print(f"      acquires {v['acquired']}")
        print(f"      while holding {v['while_holding']}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
