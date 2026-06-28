"""Incremental materialization orchestrator — where everything composes.

This is the production loop the whole framework exists to enable: discover blocks,
detect what CODE changed since the last run, run per-ticker + panel passes through the
content-addressed cache (so unchanged blocks are O(1) hits, not recomputes), write the
result to the date-partitioned lake, and advance a per-run watermark.

A full rebuild becomes a DELTA rebuild:
  * edit one block  -> only it (and dependents) recompute, everything else from CAS;
  * rerun unchanged -> ~100% cache hits, near-zero work;
  * new trading day -> appended as a new lake partition.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cache import Cache
from engine import Engine, summarize_profile
from storage import FeatureLake


class Materializer:
    def __init__(self, blocks, cache: Cache, lake: FeatureLake, project_root: str | Path,
                 *, date_col: str = "Date"):
        self.blocks = list(blocks)
        self.cache = cache
        self.lake = lake
        self.root = project_root
        self.date_col = date_col
        self.per_ticker = [b for b in self.blocks if b.kind == "per_ticker"]
        self.panel = [b for b in self.blocks if b.kind == "panel"]
        self.wm_path = Path(cache.root) / "watermark.json"

    def _watermark(self) -> dict:
        if self.wm_path.is_file():
            try:
                return json.loads(self.wm_path.read_text("utf-8"))
            except Exception:
                return {}
        return {}

    def run(self, ticker_frames: dict[str, pd.DataFrame]) -> dict:
        # --- per-ticker pass (cached) ---
        pt_engine = Engine(self.per_ticker, self.cache, self.root)
        changed = sorted(pt_engine.changed_blocks())
        profiles, outs = [], []
        for tk, df in ticker_frames.items():
            out, prof = pt_engine.run(df)
            profiles.append(prof)
            outs.append(out)
        pt_engine.finalize()
        panel = pd.concat(outs, ignore_index=True)

        # --- panel / cross-sectional pass (cached) ---
        if self.panel:
            pn_engine = Engine(self.panel, self.cache, self.root)
            panel, pprof = pn_engine.run(panel)
            pn_engine.finalize()
            profiles.append(pprof)

        # --- persist to the lake ---
        lake_stats = self.lake.write(panel)

        # --- advance watermark ---
        last_date = panel[self.date_col].max()
        wm = self._watermark()
        wm["last_date"] = int(last_date) if pd.notna(last_date) else None
        wm["n_blocks"] = len(self.blocks)
        self.wm_path.write_text(json.dumps(wm, sort_keys=True), "utf-8")

        summary = summarize_profile(profiles)
        return {
            "changed_blocks": changed,
            "hit_rate": summary["hit_rate"],
            "hits": summary["hits"],
            "misses": summary["misses"],
            "n_feature_cols": panel.shape[1],
            "lake": lake_stats,
            "watermark": wm,
            "panel": panel,
        }
