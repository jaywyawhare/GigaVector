"""Minimal reader for the Informal Trace Format (ITF), the JSON format Quint
and Apalache emit with `--out-itf`.

Spec: https://apalache-mc.org/docs/adr/015adr-trace.html

The one thing a hand-rolled parser gets wrong is the integer encoding: ITF
forbids bare JSON numbers and wraps integers as {"#bigint": "42"}. Sets are
{"#set": [...]}, maps are {"#map": [[k, v], ...]}, and {"#unserializable": s}
marks a value Quint could not render.

Kept dependency-free on purpose so it runs in CI without installing itf-py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Unserializable:
    __slots__ = ("repr_",)

    def __init__(self, repr_: str) -> None:
        self.repr_ = repr_

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Unserializable({self.repr_!r})"


def decode(node: Any) -> Any:
    """Recursively convert an ITF JSON node into plain Python values."""
    if isinstance(node, list):
        return [decode(x) for x in node]
    if not isinstance(node, dict):
        # bool / str / None pass through. Bare numbers are illegal in ITF, but
        # tolerate them rather than failing a replay over a producer quirk.
        return node
    if "#bigint" in node:
        return int(node["#bigint"])
    if "#set" in node:
        return {_hashable(decode(x)) for x in node["#set"]}
    if "#map" in node:
        return {_hashable(decode(k)): decode(v) for k, v in node["#map"]}
    if "#tup" in node:
        return tuple(decode(x) for x in node["#tup"])
    if "#unserializable" in node:
        return Unserializable(node["#unserializable"])
    return {k: decode(v) for k, v in node.items()}


def _hashable(v: Any) -> Any:
    """Sets and map keys need hashable values; records decode to dicts."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, set):
        return frozenset(v)
    return v


class Trace:
    """One ITF trace: an ordered list of states, each a dict of variable -> value."""

    def __init__(self, raw: dict) -> None:
        self.meta = raw.get("#meta", {})
        self.vars = raw.get("vars", [])
        self.params = raw.get("params", [])
        self.loop = raw.get("loop")
        self.states = [
            {k: decode(v) for k, v in st.items() if not k.startswith("#")}
            for st in raw.get("states", [])
        ]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i: int) -> dict:
        return self.states[i]

    @property
    def actions(self) -> list[str | None]:
        """The action name recorded for each state, when the producer emits one.

        Quint records this under `mbt::actionTaken` when the spec is run with
        model-based-testing enabled; it is absent for plain `quint run` output.
        """
        out = []
        for st in self.states:
            a = st.get("mbt::actionTaken") or st.get("actionTaken")
            out.append(a if isinstance(a, str) else None)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def load_all(cls, directory: str | Path, pattern: str = "*.itf.json") -> list["Trace"]:
        return [cls.load(p) for p in sorted(Path(directory).glob(pattern))]


def strip_module_prefix(state: dict) -> dict:
    """Quint namespaces variables as `instance::module::var`. Replay drivers
    care about the bare name."""
    return {k.rsplit("::", 1)[-1]: v for k, v in state.items()}
