"""Dependency DAG + topological scheduling + partial-DAG selection.

Same Kahn topo-sort your 3__FeatureFramework.py already uses, plus the thing it
lacks: dbt-style selective execution. `select(..., "modified+")` rebuilds exactly
the changed blocks and their descendants; `select(..., "+target")` rebuilds a
target column and its ancestors. A one-block edit then costs 1 column, not 28GB.
"""
from __future__ import annotations

from collections import defaultdict, deque

from block import Block


class CycleError(ValueError):
    pass


class FeatureGraph:
    def __init__(self, blocks: list[Block]):
        self.blocks: dict[str, Block] = {}
        self._produced_by: dict[str, str] = {}
        for b in blocks:
            if b.name in self.blocks:
                raise ValueError(f"duplicate block name: {b.name}")
            self.blocks[b.name] = b
            for col in b.produces:
                self._produced_by[col] = b.name

        # edges: producer -> consumer
        self.fwd: dict[str, set[str]] = defaultdict(set)
        self.rev: dict[str, set[str]] = defaultdict(set)
        for b in blocks:
            for col in b.requires:
                src = self._produced_by.get(col)
                if src is not None and src != b.name:
                    self.fwd[src].add(b.name)
                    self.rev[b.name].add(src)

    # -- ordering -------------------------------------------------------------
    def topo_order(self, subset: set[str] | None = None) -> list[str]:
        nodes = set(self.blocks) if subset is None else set(subset)
        indeg = {n: len(self.rev[n] & nodes) for n in nodes}
        q = deque(sorted(n for n in nodes if indeg[n] == 0))
        order: list[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in sorted(self.fwd[n]):
                if m in nodes:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        q.append(m)
        if len(order) != len(nodes):
            raise CycleError("dependency cycle among: " + ", ".join(sorted(nodes - set(order))))
        return order

    # -- graph walks ----------------------------------------------------------
    def descendants(self, names: set[str]) -> set[str]:
        out, stack = set(), list(names)
        while stack:
            n = stack.pop()
            for m in self.fwd[n]:
                if m not in out:
                    out.add(m)
                    stack.append(m)
        return out

    def ancestors(self, names: set[str]) -> set[str]:
        out, stack = set(), list(names)
        while stack:
            n = stack.pop()
            for m in self.rev[n]:
                if m not in out:
                    out.add(m)
                    stack.append(m)
        return out

    def block_for_column(self, col: str) -> str | None:
        return self._produced_by.get(col)

    # -- selection ------------------------------------------------------------
    def select(self, spec: str, *, changed: set[str] | None = None) -> list[str]:
        """Resolve a dbt-style selector into a topo-ordered execution subset.

        Supported:
          "all"                     -> every block
          "modified+"               -> changed blocks + their descendants
          "+modified"               -> changed blocks + their ancestors
          "+modified+"              -> changed + ancestors + descendants
          "<block>" / "+<block>+"   -> a named block (+/- ancestors/descendants)
          "<block>+" / "+<block>"   -> descendants-only / ancestors-only variants
        """
        spec = spec.strip()
        if spec in ("all", "*", ""):
            return self.topo_order()

        want_anc = spec.startswith("+")
        want_desc = spec.endswith("+")
        core_name = spec.strip("+")

        if core_name == "modified":
            core = set(changed or ())
        elif core_name in self.blocks:
            core = {core_name}
        elif core_name in self._produced_by:
            core = {self._produced_by[core_name]}
        else:
            raise KeyError(f"unknown selector target: {core_name!r}")

        chosen = set(core)
        if want_anc:
            chosen |= self.ancestors(core)
        if want_desc:
            chosen |= self.descendants(core)
        return self.topo_order(chosen)
