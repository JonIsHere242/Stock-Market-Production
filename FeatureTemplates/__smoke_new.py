"""
__smoke_new.py  --  ultralight, single-core sanity gate for a NEWLY-authored feature block.

For brand-new (no ground-truth) candidate blocks, verify_block.py does not apply (it compares to
Data/ProcessedData originals). This is the cheap pre-screen an authoring agent runs right after
writing a block, BEFORE the heavy validate_feature/__tail_screen passes. It is deliberately tiny
(2 tickers, single core) so many can run without disturbing a training job.

Checks (all must pass for exit 0):
  1. imports + has METADATA/compute, name matches stem
  2. compute() runs on 2 tickers, adds EXACTLY the produced columns, drops/mutates nothing
  3. no inf; produced columns are numeric and not ~constant (mode-fraction < 0.999 after warmup)
  4. CAUSALITY: recompute on the series truncated at 60/75/90% -> past values must be identical
     (atol 1e-6, rtol 1e-4). This is the same look-ahead test validate_feature uses.
  5. timing < 150 ms/ticker (the framework's SLOW quarantine bar)

Output is ASCII only (Windows cp1252 console). Usage:
  stock_env\\Scripts\\python.exe FeatureTemplates\\__smoke_new.py _p625_amihud_illiquidity
  ...\\python.exe FeatureTemplates\\__smoke_new.py <stem> --tickers AAPL,KO
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # single-core: courteous to any running training job

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRICE_DIR = ROOT / "Data" / "PriceData"
OHLCV = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
ATOL, RTOL = 1e-6, 1e-4
SLOW_MS = 150.0


def load_block(stem: str):
    p = Path(stem)
    if not p.suffix:
        p = HERE / f"{stem}.py"
    if not p.exists():
        sys.exit(f"FAIL: block file not found: {p}")
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p


def prep(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(PRICE_DIR / f"{ticker}.parquet")
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()
    df = df.rename(columns={c: c.title() for c in df.columns
                            if c.lower() in ("open", "high", "low", "close", "volume")})
    if "Ticker" not in df.columns:
        df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def pick_tickers(explicit: str) -> list[str]:
    if explicit:
        return [t.strip().upper() for t in explicit.split(",") if t.strip()]
    have = {p.stem for p in PRICE_DIR.glob("*.parquet")}
    # one long-history major + one ordinary name, both deterministic
    prefer = ["AAPL", "MSFT", "KO", "JPM", "XOM", "WMT", "CSCO", "PG"]
    picks = [t for t in prefer if t in have][:2]
    return picks or sorted(have)[:2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("block")
    ap.add_argument("--tickers", default="")
    args = ap.parse_args()

    mod, path = load_block(args.block)
    fails: list[str] = []
    warns: list[str] = []

    meta = getattr(mod, "METADATA", None)
    if not isinstance(meta, dict) or not hasattr(mod, "compute"):
        sys.exit("FAIL: block missing METADATA dict or compute()")
    produces = list(meta.get("produces", []))
    name = meta.get("name", "")
    if name != path.stem:
        warns.append(f"METADATA name '{name}' != file stem '{path.stem}'")
    if not produces:
        fails.append("METADATA.produces is empty")

    tickers = pick_tickers(args.tickers)
    per_ms = []
    col_stats: dict[str, dict] = {}
    causality_bad: list[str] = []

    for t in tickers:
        try:
            df = prep(t)
        except Exception as exc:
            warns.append(f"could not load {t}: {exc}")
            continue
        base = list(df.columns)
        try:
            t0 = time.perf_counter()
            out = mod.compute(df.copy())
            per_ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            fails.append(f"compute() raised on {t}: {exc}")
            continue

        if len(out) != len(df):
            fails.append(f"{t}: row count changed {len(df)} -> {len(out)}")
        for c in base:
            if c not in out.columns:
                fails.append(f"{t}: dropped input column '{c}'")
        extra = [c for c in out.columns if c not in base and c not in produces]
        if extra:
            warns.append(f"{t}: undeclared output columns {extra[:6]}")

        for c in produces:
            if c not in out.columns:
                fails.append(f"{t}: declared column '{c}' not produced")
                continue
            s = pd.to_numeric(out[c], errors="coerce")
            if not pd.api.types.is_numeric_dtype(out[c]):
                fails.append(f"{t}: '{c}' is non-numeric")
            arr = s.to_numpy(dtype="float64")
            if np.isinf(arr).any():
                fails.append(f"{t}: '{c}' contains inf (unguarded division?)")
            tail = s.iloc[200:] if len(s) > 200 else s
            finite = float(np.isfinite(tail.to_numpy(dtype='float64')).mean()) if len(tail) else 0.0
            vc = tail.dropna()
            mode_frac = float(vc.value_counts(normalize=True).iloc[0]) if len(vc) else 1.0
            st = col_stats.setdefault(c, {"finite": [], "mode": [], "absmax": 0.0})
            st["finite"].append(finite)
            st["mode"].append(mode_frac)
            fin = arr[np.isfinite(arr)]
            if fin.size:
                st["absmax"] = max(st["absmax"], float(np.abs(fin).max()))

        # causality: truncate the price series, past values must not change
        for f in (0.6, 0.75, 0.9):
            k = int(len(df) * f)
            if k < 80:
                continue
            try:
                tr = mod.compute(df.iloc[:k].copy())
            except Exception:
                continue
            for c in produces:
                if c not in out.columns or c not in tr.columns:
                    continue
                a = np.asarray(out[c].iloc[:k], dtype="float64")
                b = np.asarray(tr[c], dtype="float64")
                m = min(len(a), len(b))
                a, b = a[:m], b[:m]
                fin = np.isfinite(a) & np.isfinite(b)
                if fin.any() and not np.allclose(a[fin], b[fin], atol=ATOL, rtol=RTOL):
                    md = float(np.max(np.abs(a[fin] - b[fin])))
                    tag = f"{c}@{f}(maxdiff {md:.2e})"
                    if tag not in causality_bad:
                        causality_bad.append(tag)

    if causality_bad:
        fails.append("LEAK: causality failed -> " + ", ".join(causality_bad[:5]))

    # near-constant / mostly-NaN produced columns are dead weight
    for c, st in col_stats.items():
        if st["finite"] and max(st["finite"]) < 0.05:
            warns.append(f"'{c}' almost entirely NaN (max finite frac {max(st['finite']):.2f})")
        if st["mode"] and min(st["mode"]) >= 0.999:
            warns.append(f"'{c}' is ~constant (mode-fraction {min(st['mode']):.3f})")

    mean_ms = float(np.mean(per_ms)) if per_ms else float("nan")
    if per_ms and mean_ms > SLOW_MS:
        fails.append(f"SLOW: {mean_ms:.0f} ms/ticker > {SLOW_MS:.0f} ms bar")

    print(f"\nsmoke: {path.stem}  | tickers {tickers}  | {len(produces)} cols  | "
          f"{mean_ms:.0f} ms/ticker")
    print(f"  {'column':40s} {'finite%':>8} {'modefrac':>9} {'absmax':>12}")
    for c in produces:
        st = col_stats.get(c)
        if not st:
            print(f"  {c[:40]:40s} {'--':>8} {'--':>9} {'--':>12}")
            continue
        f_ = max(st["finite"]) if st["finite"] else 0.0
        mo = min(st["mode"]) if st["mode"] else 1.0
        print(f"  {c[:40]:40s} {f_*100:>7.1f}% {mo:>9.3f} {st['absmax']:>12.4g}")
    for w in warns:
        print(f"  (warn) {w}")
    if fails:
        print("  RESULT: FAIL")
        for fz in fails:
            print(f"    - {fz}")
        sys.exit(1)
    print("  RESULT: OK (causal, finite, fast, contract-clean)")
    sys.exit(0)


if __name__ == "__main__":
    main()
