"""
__trial_ledger.py  --  append-only trial ledger for feature EDA / screening.

WHY
---
This house's documented #1 failure mode is batch-screening false discovery: you
try N candidates, the best raw lift is a chance maximum, and a single-seed win
evaporates multi-seed (the +1.2pp false positive; feature_lab +0.60 -> -0.175).
Honest multiple-testing deflation (FDR, Deflated Sharpe, "is the best of M real")
is IMPOSSIBLE without knowing how many things you actually tried. Memory of "how
many candidates / seeds / transforms were screened" lives only in your head and is
always under-counted -- which is exactly why deflation done after the fact is fiction.

This module is that missing substrate: a tiny, dependency-light, append-only log
that every screening tool (__tail_screen, __diagnostics, __feature_lab, future
ablation wrappers) writes one row per (item, metric, seed) into. It never rewrites
or dedupes -- the whole point is an honest, monotonically-growing count of trials.

It is NOT analysis. It is the raw material the deflation layer (Romano-Wolf
step-down, BH-FDR, Deflated Sharpe) will later read. Keeping it append-only and
self-describing (every row carries its run params) means a future tool can answer
"how many INDEPENDENT family/seed combinations were tried before this winner" with
no guesswork.

The file is auto-skipped by the framework (leading __ keeps it out of block discovery).

SCHEMA  (fixed canonical columns; everything else goes to extra_json)
------
  ts          ISO timestamp of the flush
  run_id      stable id for one tool invocation (groups all rows of a run)
  tool        e.g. "tail_screen", "diagnostics", "ablation"
  item        the thing being scored: a feature column or a block/family name
  family      the owning block / family (optional)
  metric      e.g. "neut_lift", "ic_perday_mean", "ic_ir", "null_p95", "live_ic"
  value       float
  n_obs       sample size behind the value
  seed        the seed for this run (the house's noise axis)
  params_json run-level params (n tickers, folds, q, ...), denormalized onto every row
  extra_json  per-row extras (side, stable, foldmin, p_value, ...)

USAGE
-----
  from importlib import import_module  # or the load helper tools already use
  led = TrialLedger(tool="tail_screen", seed=42, params={"n": 500, "folds": 5})
  led.add_metrics("rdd_pt_loss_60", {"raw_lift": 1.31, "neut_lift": 1.18},
                  family="reference_dependence", n_obs=48000, side="top", stable=True)
  led.flush()                     # one append; returns the ledger path

  # later, read it back for deflation / trial-counting:
  df = TrialLedger.read()
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_LEDGER = ROOT / "Data" / "_eda_ledger" / "trial_ledger.csv"

CANON = ["ts", "run_id", "tool", "item", "family", "metric",
         "value", "n_obs", "seed", "params_json", "extra_json"]


def _json(obj) -> str:
    """Compact, sorted, NaN-safe JSON for the denormalized blob columns."""
    def _default(o):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:
            pass
        return str(o)
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)
    except Exception:
        return json.dumps(str(obj))


class TrialLedger:
    """Buffered, append-only writer. Buffer in memory, one append on flush()."""

    def __init__(self, tool: str, *, seed=None, params: dict | None = None,
                 ledger_path: str | Path | None = None, run_id: str | None = None,
                 enabled: bool = True):
        self.tool = str(tool)
        self.seed = seed
        self.params = dict(params or {})
        self.enabled = bool(enabled)
        self.path = Path(ledger_path) if ledger_path else DEFAULT_LEDGER
        # one run_id per invocation: timestamp (microsecond) + tool, monotonic & sortable
        self.run_id = run_id or f"{datetime.now().strftime('%Y%m%dT%H%M%S_%f')}_{self.tool}"
        self._params_json = _json(self.params)
        self._buf: list[dict] = []

    # -- low level -----------------------------------------------------------
    def add(self, item: str, metric: str, value, *, family: str = "",
            n_obs: int = 0, seed=None, **extra) -> None:
        if not self.enabled:
            return
        try:
            fval = float(value)
        except Exception:
            fval = float("nan")
        self._buf.append({
            "item": str(item),
            "family": str(family),
            "metric": str(metric),
            "value": fval,
            "n_obs": int(n_obs) if n_obs is not None else 0,
            "seed": self.seed if seed is None else seed,
            "extra_json": _json(extra) if extra else "",
        })

    # -- convenience: many metrics for one item ------------------------------
    def add_metrics(self, item: str, metrics: dict, *, family: str = "",
                    n_obs: int = 0, seed=None, **extra) -> None:
        for metric, value in metrics.items():
            if value is None:
                continue
            self.add(item, metric, value, family=family, n_obs=n_obs, seed=seed, **extra)

    # -- write ---------------------------------------------------------------
    def flush(self) -> Path:
        if not self.enabled or not self._buf:
            return self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        ts = datetime.now().isoformat(timespec="seconds")
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CANON, extrasaction="ignore")
            if new_file:
                w.writeheader()
            for r in self._buf:
                w.writerow({
                    "ts": ts,
                    "run_id": self.run_id,
                    "tool": self.tool,
                    "item": r["item"],
                    "family": r["family"],
                    "metric": r["metric"],
                    "value": r["value"],
                    "n_obs": r["n_obs"],
                    "seed": "" if r["seed"] is None else r["seed"],
                    "params_json": self._params_json,
                    "extra_json": r["extra_json"],
                })
        n = len(self._buf)
        self._buf.clear()
        return self.path

    def __len__(self) -> int:
        return len(self._buf)

    # -- read back -----------------------------------------------------------
    @staticmethod
    def read(ledger_path: str | Path | None = None):
        import pandas as pd
        p = Path(ledger_path) if ledger_path else DEFAULT_LEDGER
        if not p.exists():
            return pd.DataFrame(columns=CANON)
        return pd.read_csv(p)


if __name__ == "__main__":
    # smoke test / quick stats on the existing ledger
    import sys
    if not DEFAULT_LEDGER.exists():
        print(f"no ledger yet at {DEFAULT_LEDGER}")
        sys.exit(0)
    df = TrialLedger.read()
    print(f"ledger: {DEFAULT_LEDGER}  rows={len(df):,}")
    if len(df):
        print(f"  runs   : {df['run_id'].nunique()}")
        print(f"  tools  : {df['tool'].value_counts().to_dict()}")
        print(f"  metrics: {sorted(df['metric'].unique())}")
        print(f"  items  : {df['item'].nunique()} distinct")
        print(f"  span   : {df['ts'].min()} .. {df['ts'].max()}")
