"""Quick speed + sanity probe for candidate feature blocks.

    python tools/_gap_speedtest.py "FeatureTemplates/_cand_gap0903_*.py"

Loads 5 real tickers, runs compute() on each, reports median ms/ticker,
the produced-column contract check and the NaN share. Not a substitute for
FeatureDiscovery/validate_feature.py -- just a fast inner loop so codegen
agents can see their own timing before the real gate runs.
"""
from __future__ import annotations

import glob
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRICE_DIR = ROOT / "Data" / "PriceData"
COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def _frames(n: int = 5) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    for p in sorted(PRICE_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if not set(COLS[2:]).issubset(df.columns) or len(df) < 700:
            continue
        if "Date" not in df.columns:
            df = df.reset_index()
        if "Ticker" not in df.columns:
            df["Ticker"] = p.stem
        df = df[COLS].sort_values("Date").tail(750).reset_index(drop=True)
        out.append(df)
        if len(out) >= n:
            break
    return out


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "FeatureTemplates/_cand_gap0903_*.py"
    frames = _frames()
    if not frames:
        print("no usable price frames found")
        return
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        print(f"no files match {pattern}")
        return
    print(f"{'ms':>7} {'cols':>5} {'nan%':>6}  block")
    worst = 0.0
    for f in files:
        path = Path(f)
        try:
            mod = _load(path)
            produces = list(mod.METADATA["produces"])
            times, nan_share, missing = [], [], []
            for fr in frames:
                d = fr.copy()
                t0 = time.perf_counter()
                out = mod.compute(d)
                times.append((time.perf_counter() - t0) * 1000.0)
                miss = [c for c in produces if c not in out.columns]
                missing.extend(miss)
                got = [c for c in produces if c in out.columns]
                if got:
                    nan_share.append(float(out[got].isna().mean().mean()))
            ms = float(np.median(times))
            worst = max(worst, ms)
            flag = ""
            if missing:
                flag = f"  MISSING {sorted(set(missing))}"
            if ms > 100:
                flag += "  SLOW"
            print(f"{ms:7.1f} {len(produces):5d} {100 * float(np.mean(nan_share)):5.1f}%  {path.name}{flag}")
        except Exception as exc:  # noqa: BLE001
            print(f"{'ERR':>7} {'-':>5} {'-':>6}  {path.name}  {type(exc).__name__}: {exc}")
    print(f"\nworst median: {worst:.1f}ms (budget 100ms, gate quarantines >150ms)")


if __name__ == "__main__":
    main()
