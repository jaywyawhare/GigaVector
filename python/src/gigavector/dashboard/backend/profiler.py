"""Nested scope profiler for the dashboard.

Mirrors the C-side scope profiler (``profile_e2e`` / ``backend/profiling``):
stack-based ``begin``/``end`` scopes aggregated per name with inclusive vs self
time, min/max, and allocator deltas, plus a flat span list (name, depth, start,
duration) that renders directly as an icicle/flamegraph.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional


def _s64(x: int) -> int:
    """Reinterpret an unsigned 64-bit allocator delta as signed for display."""
    return x - (1 << 64) if x >= (1 << 63) else x


class Profiler:
    """Single-threaded nested profiler. Use one instance per run."""

    def __init__(self, alloc_fn: Optional[Callable[[], dict]] = None) -> None:
        self._alloc = alloc_fn
        self._stats: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._ops: dict[str, int] = {}
        self._stack: list[dict[str, Any]] = []
        self._spans: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def _alloc_snap(self) -> tuple[int, int]:
        if self._alloc:
            try:
                a = self._alloc()
                return int(a["alloc_count"]), int(a["live_bytes"])
            except Exception:
                return 0, 0
        return 0, 0

    def begin(self, name: str) -> None:
        ac, lb = self._alloc_snap()
        self._stack.append({
            "name": name,
            "start": self._now_ms(),
            "child_ms": 0.0,
            "ac": ac,
            "lb": lb,
            "depth": len(self._stack),
        })

    def end(self, ops: int = 0) -> None:
        if not self._stack:
            return
        fr = self._stack.pop()
        incl = self._now_ms() - fr["start"]
        ac, lb = self._alloc_snap()
        allocs = ac - fr["ac"]
        net = _s64(lb - fr["lb"])
        self_ms = incl - fr["child_ms"]
        if self_ms < 0.0:
            self_ms = 0.0
        if self._stack:
            self._stack[-1]["child_ms"] += incl

        name = fr["name"]
        st = self._stats.get(name)
        if st is None:
            st = {
                "name": name,
                "calls": 0,
                "total_ms": 0.0,
                "self_ms": 0.0,
                "min_ms": float("inf"),
                "max_ms": 0.0,
                "allocs": 0,
                "net_bytes": 0,
            }
            self._stats[name] = st
            self._order.append(name)
        st["calls"] += 1
        st["total_ms"] += incl
        st["self_ms"] += self_ms
        st["min_ms"] = min(st["min_ms"], incl)
        st["max_ms"] = max(st["max_ms"], incl)
        st["allocs"] += allocs
        st["net_bytes"] += net
        if ops:
            self._ops[name] = self._ops.get(name, 0) + ops

        self._spans.append({
            "name": name,
            "depth": fr["depth"],
            "ts_ms": fr["start"],
            "dur_ms": incl,
            "self_ms": self_ms,
            "allocs": allocs,
            "net_bytes": net,
        })

    def scope(self, name: str, ops: int = 0) -> "_ScopeCtx":
        return _ScopeCtx(self, name, ops)

    def scopes(self) -> list[dict[str, Any]]:
        """Aggregated per-name stats, sorted by self time (hot spots first)."""
        out = []
        for name in self._order:
            st = dict(self._stats[name])
            if st["min_ms"] == float("inf"):
                st["min_ms"] = 0.0
            calls = st["calls"]
            st["avg_ms"] = st["total_ms"] / calls if calls else 0.0
            ops = self._ops.get(name, 0)
            st["ops_per_sec"] = (ops / (st["total_ms"] / 1000.0)) if ops and st["total_ms"] > 0 else 0.0
            out.append(st)
        out.sort(key=lambda s: s["self_ms"], reverse=True)
        return out

    def spans(self) -> list[dict[str, Any]]:
        """Flat span list ordered for flamegraph rendering (depth, then start)."""
        return sorted(self._spans, key=lambda s: (s["depth"], s["ts_ms"]))


class _ScopeCtx:
    def __init__(self, prof: Profiler, name: str, ops: int) -> None:
        self._prof = prof
        self._name = name
        self._ops = ops

    def __enter__(self) -> "_ScopeCtx":
        self._prof.begin(self._name)
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._prof.end(self._ops)
        return False
