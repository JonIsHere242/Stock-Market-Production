"""Execution engine — runs blocks in dependency order, with caching + profiling.

Preserves the two properties you care about:
  * PER-BLOCK PROFILING — every block is wall-clock timed, now annotated with a
    cache hit/miss flag so the timing report tells you what actually executed.
  * THE ROW CONTRACT — Block.run() enforces "add columns, never reshape".

The engine treats the frame it is handed as the unit of work (a single ticker for
per-ticker blocks, or a full panel for cross-sectional blocks). The caller decides
which; the engine just schedules, caches, and profiles. Incremental speed comes
from the cache: an unchanged block is an O(1) key lookup, not a recompute.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from cache import Cache
from graph import FeatureGraph
from hashing import data_version


class Engine:
    def __init__(self, blocks, cache: Cache, project_root: str | Path):
        self.graph = FeatureGraph(list(blocks))
        self.cache = cache
        self.project_root = Path(project_root)
        # code versions are expensive (closure walk) — compute once per engine
        self._cv = {b.name: b.code_version(self.project_root) for b in self.graph.blocks.values()}

    def changed_blocks(self) -> set[str]:
        return self.cache.detect_changed(list(self.graph.blocks.values()), self.project_root)

    def _order(self, only: set[str] | None) -> list[str]:
        if only is None:
            return self.graph.topo_order()
        closed = set(only) | self.graph.ancestors(set(only))
        return self.graph.topo_order(closed)

    def run(self, df: pd.DataFrame, only: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
        df = df.copy()
        order = self._order(only)
        dv: dict[str, str] = {c: data_version(df[c].to_numpy()) for c in df.columns}
        profile: dict[str, dict] = {}

        for name in order:
            b = self.graph.blocks[name]
            key = self.cache.cache_key(self._cv[name], dv, b.requires)
            t0 = time.perf_counter()
            cached = self.cache.get(key)
            if cached is not None:
                for col, content_hash in cached.items():
                    df[col] = self.cache.load_column(content_hash)
                    dv[col] = content_hash
                status = "hit"
            else:
                out = b.run(df)
                produced = {col: out[col].to_numpy() for col in b.produces}
                content = self.cache.put(key, produced)
                df = out
                for col, ch in content.items():
                    dv[col] = ch
                status = "miss"
            profile[name] = {
                "seconds": time.perf_counter() - t0,
                "cache": status,
                "n_rows": len(df),
                "produces": list(b.produces),
                "family": b.family,
            }
        return df, profile

    def finalize(self) -> None:
        """Persist the cache manifest + code-version snapshot after a run set."""
        self.cache.update_snapshot(list(self.graph.blocks.values()), self.project_root)
        self.cache.flush()


def summarize_profile(profiles: list[dict]) -> dict:
    """Aggregate per-ticker profiles into a report (mean/p95 per block, hit rate)."""
    agg: dict[str, list[float]] = {}
    hits = misses = 0
    for prof in profiles:
        for name, rec in prof.items():
            agg.setdefault(name, []).append(rec["seconds"])
            if rec["cache"] == "hit":
                hits += 1
            else:
                misses += 1

    def p95(xs: list[float]) -> float:
        s = sorted(xs)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    blocks = {
        name: {
            "calls": len(xs),
            "total_s": sum(xs),
            "mean_ms": 1000 * sum(xs) / len(xs),
            "p95_ms": 1000 * p95(xs),
            "max_ms": 1000 * max(xs),
        }
        for name, xs in agg.items()
    }
    total = hits + misses
    return {
        "hit_rate": hits / total if total else 0.0,
        "hits": hits,
        "misses": misses,
        "wall_s": sum(b["total_s"] for b in blocks.values()),
        "slowest": sorted(blocks.items(), key=lambda kv: -kv[1]["total_s"])[:10],
        "blocks": blocks,
    }
