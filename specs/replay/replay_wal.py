#!/usr/bin/env python3
"""Replay a Quint ITF trace of specs/quint/wal_recovery.qnt against the real
src/storage/wal.c, and report any divergence.

Companion to replay_raft.py (same ITF machinery, different driver). The two
crash flavours the spec distinguishes are emulated against a real file:

  processCrash  close + reopen the WAL handle; OS-buffered bytes persist.
  machineCrash  ftruncate the file back to the durable prefix, dropping
                exactly the records the model says were not fsync()ed yet.
                Every insert record is byte-identical in size (same vector,
                no metadata), so record boundaries are arithmetic.
  tearTail      cut the file mid-final-record; wal_is_torn_tail must truncate
                the torn tail and replay must stop cleanly.
  corruptMiddle flip one byte inside an early record; mid-log corruption must
                FAIL replay rather than silently truncate.

After every model `replay` step the driver compares the model's post-state
(`replayed` list length + `replayFailed` flag) with what wal_replay actually
delivered.

Exit status: 0 conformant, 1 divergence, 2 could not run.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itf import Trace, strip_module_prefix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
LIB = ROOT / "build" / "lib" / "libGigaVector.so"

CDEF = """
typedef struct GV_WAL GV_WAL;
GV_WAL *wal_open(const char *path, size_t dimension, uint32_t index_type);
int wal_append_insert(GV_WAL *wal, const float *data, size_t dimension,
                      const char *metadata_key, const char *metadata_value);
int wal_replay(const char *path, size_t expected_dimension,
               int (*on_insert)(void *ctx, const float *data, size_t dimension,
                                const char *metadata_key, const char *metadata_value),
               void *ctx, uint32_t expected_index_type);
void wal_close(GV_WAL *wal);
void wal_set_sync_interval(GV_WAL *wal, size_t interval);
"""

DIMENSION = 4


def unwrap(pick):
    if isinstance(pick, dict) and "tag" in pick:
        return pick["value"] if pick["tag"] == "Some" else None
    return pick


class WalRig:
    """A real WAL file driven through crash/recovery scenarios."""

    def __init__(self, ffi, lib, tmpdir):
        self.ffi, self.lib = ffi, lib
        self.path = os.path.join(tmpdir, "replay.wal")
        self.wal = None
        self.header_len = None
        self.record_len = None
        # Fixed payload: every insert record is byte-identical, so file offsets
        # are header_len + k * record_len.
        self.payload = [0.25, -0.5, 2.0, 0.0]

    def _open(self):
        self.wal = self.lib.wal_open(self.path.encode(), DIMENSION, 0)
        if self.wal == self.ffi.NULL:
            raise RuntimeError(f"wal_open failed for {self.path}")

    def _close(self):
        if self.wal is not None:
            self.lib.wal_close(self.wal)
            self.wal = None

    def _size(self):
        return os.stat(self.path).st_size

    def append_rec(self):
        vec = self.ffi.new("float[]", self.payload)
        rc = self.lib.wal_append_insert(self.wal, vec, DIMENSION,
                                        self.ffi.NULL, self.ffi.NULL)
        if rc != 0:
            raise RuntimeError("wal_append_insert failed")

    def _measure_header_and_record(self):
        """Write two probe records to a throwaway file to learn header/record
        sizes; every insert this driver writes is byte-identical, so record
        boundaries stay arithmetic for the whole trace."""
        probe = os.path.join(os.path.dirname(self.path), "probe.wal")
        wal = self.lib.wal_open(probe.encode(), DIMENSION, 0)
        if wal == self.ffi.NULL:
            raise RuntimeError("probe wal_open failed")
        vec = self.ffi.new("float[]", self.payload)
        if self.lib.wal_append_insert(wal, vec, DIMENSION,
                                      self.ffi.NULL, self.ffi.NULL) != 0:
            raise RuntimeError("probe wal_append_insert failed")
        one = os.stat(probe).st_size
        if self.lib.wal_append_insert(wal, vec, DIMENSION,
                                      self.ffi.NULL, self.ffi.NULL) != 0:
            raise RuntimeError("probe wal_append_insert failed")
        two = os.stat(probe).st_size
        self.record_len = two - one
        self.header_len = one - self.record_len
        self.lib.wal_close(wal)
        os.unlink(probe)

    def sync_now(self):
        # Forces fflush+fsync of everything pending (wal.c:55).
        self.lib.wal_set_sync_interval(self.wal, 1)

    def process_crash(self):
        self._close()
        self._open()

    def machine_crash(self, n_lost):
        """Drop the last n_lost (flushed-but-not-fsynced) records."""
        if n_lost <= 0 or self.record_len is None:
            return
        target = self._size() - n_lost * self.record_len
        self._close()
        with open(self.path, "r+b") as f:
            f.truncate(target)
        self._open()

    def tear_tail(self, n_durable):
        # Model keeps the torn record in `durable`; replay drops exactly it.
        target = self.header_len + max(n_durable - 1, 0) * self.record_len \
            + self.record_len // 2
        self._close()
        with open(self.path, "r+b") as f:
            f.truncate(target)
        self._open()

    def corrupt_middle(self):
        offset = self.header_len + self.record_len // 2
        self._close()
        with open(self.path, "r+b") as f:
            f.seek(offset)
            byte = f.read(1)
            f.seek(offset)
            f.write(bytes([byte[0] ^ 0xFF]))
        self._open()

    def replay(self):
        counter = {"n": 0}

        @self.ffi.callback("int(void *, const float *, size_t, const char *, const char *)")
        def on_insert(_ctx, _data, _dim, _k, _v):
            counter["n"] += 1
            return 0

        self._close()
        rc = self.lib.wal_replay(self.path.encode(), DIMENSION, on_insert,
                                 self.ffi.NULL, 0)
        self._open()
        return rc, counter["n"]

    def destroy(self):
        self._close()
        if os.path.exists(self.path):
            os.unlink(self.path)


def apply_action(rig, action, pre, picks):
    """Translate one model action into the equivalent wal.c manipulation.
    Returns True when the action was replayable."""
    if action in ("init", None):
        return True
    if action == "appendRec":
        rig.append_rec()
        return True
    if action == "syncNow":
        rig.sync_now()
        return True
    if action == "processCrash":
        rig.process_crash()
        return True
    if action == "machineCrash":
        rig.machine_crash(len(pre.get("osBuffer", [])))
        return True
    if action == "tearTail":
        rig.tear_tail(len(pre.get("durable", [])))
        return True
    if action == "corruptMiddle":
        rig.corrupt_middle()
        return True
    if action == "replay":
        return True  # compared by the caller, nothing to drive
    return False


def compare(post, rc, count):
    """Model's expected replay outcome vs what wal_replay delivered."""
    diffs = []
    failed = bool(post.get("replayFailed"))
    if failed != (rc != 0):
        diffs.append(f"replayFailed: model={failed} impl rc={rc}")
    if not failed:
        expect = len(post.get("replayed", []))
        if expect != count:
            diffs.append(f"replayed: model={expect} records impl={count} records")
    return diffs


def replay(path, strict=True):
    try:
        from cffi import FFI
    except ImportError:
        print("cffi is required: pip install cffi", file=sys.stderr)
        return 2
    if not LIB.exists():
        print(f"shared library not found at {LIB}; run `make lib` first",
              file=sys.stderr)
        return 2

    ffi = FFI()
    ffi.cdef(CDEF)
    lib = ffi.dlopen(str(LIB))

    trace = Trace.load(path)
    actions = trace.actions
    if not any(actions):
        print(f"{path}: no mbt::actionTaken metadata; regenerate with "
              "`quint run --mbt`", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="wal-replay-") as tmpdir:
        rig = WalRig(ffi, lib, tmpdir)
        print(f"replaying {Path(path).name}: {len(trace)} states")
        divergences = []
        try:
            for i in range(1, len(trace)):
                pre = strip_module_prefix(trace[i - 1])
                post = strip_module_prefix(trace[i])
                action = actions[i] or "?"
                if i == 1:
                    # Probe file layout, then open the real WAL, before
                    # driving anything.
                    rig._measure_header_and_record()
                    rig._open()
                if not apply_action(rig, action, pre, {}):
                    print(f"  step {i}: action {action!r} not replayable, skipping")
                    continue
                if action == "replay":
                    rc, count = rig.replay()
                    diffs = compare(post, rc, count)
                else:
                    diffs = []
                status = "ok" if not diffs else "DIVERGE"
                print(f"  step {i}: {action:<14} {status}")
                for d in diffs:
                    print(f"           {d}")
                    divergences.append((i, action, d))
        finally:
            rig.destroy()

    if divergences:
        print(f"\n{len(divergences)} divergence(s) between model and implementation.")
        return 1 if strict else 0
    print("\nconformant: implementation matched the model at every step.")
    return 0


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    strict = "--allow-divergence" not in argv[1:]
    if not args:
        d = ROOT / "specs" / "quint" / "traces"
        args = [str(p) for p in sorted(d.glob("*wal*conformance*.itf.json"))]
        if not args:
            print("no traces found; generate with `quint run --mbt --out-itf=...`",
                  file=sys.stderr)
            return 2
    rc = 0
    for p in args:
        rc = max(rc, replay(p, strict=strict))
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
