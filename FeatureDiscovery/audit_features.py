"""
audit_features.py  --  Bloat audit of the EXISTING (proven, no-underscore) feature set.

After ~3 years a feature set accumulates near-duplicates and dead columns. This runs the whole
framework over a ticker sample and answers "what is safe to cull":

  - DEAD-DATA : all-NaN or constant after warmup -> carries zero information -> cull.
  - REDUNDANT : |spearman| >= --corr with another feature -> a near-duplicate. We keep ONE per
                cluster (the one with the strongest out-of-sample IC) and mark the rest cull-able.
  - KEEP      : unique enough to stand on its own.

IMPORTANT / honest: a low univariate OOS IC is NOT a cull signal here. This strategy's edge lives
in COMBINATIONS (trees + tail discrimination), so a feature that looks weak alone can still matter
conditioned on others. We REPORT each feature's OOS IC as context, but only REDUNDANCY and
DEAD-DATA drive the cull recommendation. The real "does dropping it hurt the model" test is a
retrain/backtest -- this is the cheap screen that tells you where to look.

USAGE
-----
  python FeatureDiscovery/audit_features.py --n 25 --corr 0.95
  python FeatureDiscovery/audit_features.py --n 40 --corr 0.97 --out Data/PaperFeed/feature_audit.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
PRICE_DIR = ROOT / "Data" / "PriceData"
OUT_CSV   = ROOT / "Data" / "PaperFeed" / "feature_audit.csv"

_vf_spec = importlib.util.spec_from_file_location(
    "validate_feature", Path(__file__).resolve().parent / "validate_feature.py")
vf = importlib.util.module_from_spec(_vf_spec)
_vf_spec.loader.exec_module(vf)
OHLCV = vf.OHLCV


def build_matrix(n: int, seed: int, warmup: int):
    fw = vf._load_framework()
    blocks = fw.discover_blocks()
    col_block = {}
    for name, info in blocks.items():
        for c in info["meta"].get("produces", []):
            col_block[c] = name

    paths = sorted(PRICE_DIR.glob("*.parquet"))
    chosen = random.Random(seed).sample(paths, min(n, len(paths)))

    import contextlib, io
    parts = []
    for p in chosen:
        try:
            df = pd.read_parquet(p)
            if len(df) < warmup + 200:
                continue
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                res, _ = fw.run_pipeline_timed(df, verbose=False)
            res = res.iloc[warmup:].copy()           # drop warmup NaNs
            res["__fwd"] = vf.forward_logret(res)
            res["Date"] = pd.to_datetime(res["Date"], errors="coerce")
            num = res.select_dtypes(include="number")
            feat = [c for c in num.columns if c not in OHLCV and c != "__fwd"]
            parts.append(pd.concat([res[["Date", "__fwd"]], num[feat]], axis=1))
        except Exception:
            continue
    panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return panel, col_block, len(chosen)


def per_feature_stats(panel, feat_cols, oos_frac):
    cutoff = panel["Date"].quantile(1 - oos_frac)
    stats = {}
    for c in feat_cols:
        s = panel[c]
        nan_frac = float(s.isna().mean())
        nuniq = int(s.nunique(dropna=True))
        rec = {"nan_frac": round(nan_frac, 3), "nunique": nuniq,
               "ic_oos": float("nan"), "t_oos": float("nan"), "durable": False}
        if nan_frac >= 0.99 or nuniq <= 1:
            rec["dead"] = True
            stats[c] = rec
            continue
        rec["dead"] = False
        pair = panel[[c, "__fwd", "Date"]].dropna(subset=[c, "__fwd"])
        if len(pair) > 300:
            oos = pair[pair["Date"] >= cutoff]
            is_ = pair[pair["Date"] < cutoff]
            ic_oos = vf._spearman(oos[c], oos["__fwd"]) if len(oos) > 100 else float("nan")
            ic_is  = vf._spearman(is_[c], is_["__fwd"]) if len(is_) > 100 else float("nan")
            rec["ic_oos"] = ic_oos
            rec["t_oos"] = ic_oos * math.sqrt(len(oos)) if np.isfinite(ic_oos) else float("nan")
            rec["durable"] = bool(np.isfinite(ic_is) and np.isfinite(ic_oos)
                                  and np.sign(ic_is) == np.sign(ic_oos))
        stats[c] = rec
    return stats


def cluster(panel, dense_cols, stats, thresh, max_rows=6000):
    sub = panel[dense_cols]
    if len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = sub.rank().corr().abs()       # pearson-on-ranks = spearman, pairwise-complete
    np.fill_diagonal(corr.values, 0.0)

    # strongest |OOS IC| becomes the cluster keeper
    order = sorted(dense_cols, key=lambda c: -(abs(stats[c]["ic_oos"]) if np.isfinite(stats[c]["ic_oos"]) else 0))
    keeper = {}
    for c in order:
        if c in keeper:
            continue
        keeper[c] = c
        dups = corr.index[(corr[c] >= thresh)]
        for d in dups:
            if d not in keeper:
                keeper[d] = c

    nearest, maxc = {}, {}
    for c in dense_cols:
        row = corr[c]
        maxc[c] = float(row.max()) if len(row) else 0.0
        nearest[c] = row.idxmax() if len(row) and row.max() > 0 else ""
    return keeper, nearest, maxc


def main():
    ap = argparse.ArgumentParser(description="Audit the existing feature set for bloat")
    ap.add_argument("--n", type=int, default=25, help="ticker sample (default 25)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corr", type=float, default=0.95, help="redundancy threshold (default 0.95)")
    ap.add_argument("--oos_frac", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=252)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    print(f"Audit: running the framework on {args.n} tickers (this takes a minute) ...")
    panel, col_block, n_used = build_matrix(args.n, args.seed, args.warmup)
    if panel.empty:
        raise SystemExit("no data")
    feat_cols = [c for c in panel.columns if c not in ("Date", "__fwd")]
    print(f"  pooled {len(panel):,} rows x {len(feat_cols)} features from {n_used} tickers")

    stats = per_feature_stats(panel, feat_cols, args.oos_frac)
    dead = [c for c in feat_cols if stats[c]["dead"]]
    dense = [c for c in feat_cols if not stats[c]["dead"] and panel[c].notna().mean() > 0.4]
    print(f"  dead-data: {len(dead)}   |   dense (correlated): {len(dense)}")

    keeper, nearest, maxc = cluster(panel, dense, stats, args.corr)
    redundant = [c for c in dense if keeper.get(c) != c]
    keep = [c for c in dense if keeper.get(c) == c]

    # ---- per-feature CSV ----
    rows = []
    for c in feat_cols:
        st = stats[c]
        status = "DEAD-DATA" if st["dead"] else ("REDUNDANT" if keeper.get(c) != c else "KEEP")
        rows.append({"feature": c, "block": col_block.get(c, "?"),
                     "status": status, "nan_frac": st["nan_frac"], "nunique": st["nunique"],
                     "ic_oos": round(st["ic_oos"], 4) if np.isfinite(st["ic_oos"]) else "",
                     "t_oos": round(st["t_oos"], 2) if np.isfinite(st["t_oos"]) else "",
                     "durable": st["durable"],
                     "max_corr": round(maxc.get(c, float("nan")), 3) if c in maxc else "",
                     "duplicate_of": keeper.get(c, "") if keeper.get(c) != c else "",
                     "nearest": nearest.get(c, "")})
    out = pd.DataFrame(rows).sort_values(["status", "block", "feature"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # ---- report ----
    W = 78
    print("\n" + "=" * W)
    print(f"  FEATURE AUDIT   ({len(feat_cols)} features, corr>={args.corr}, OOS {args.oos_frac:.0%})")
    print("=" * W)
    n = len(feat_cols)
    print(f"  KEEP       {len(keep):>4}  ({len(keep)*100//n}%)")
    print(f"  REDUNDANT  {len(redundant):>4}  ({len(redundant)*100//n}%)   <- near-duplicates, safe to cull")
    print(f"  DEAD-DATA  {len(dead):>4}  ({len(dead)*100//n}%)   <- all-NaN/constant, definitely cull")
    print(f"  ==> cull candidates: {len(redundant)+len(dead)} of {n} "
          f"({(len(redundant)+len(dead))*100//n}%)")

    # per-block rollup, worst bloat first
    print("\n  Per-block (cull% = (redundant+dead) / cols):")
    print(f"  {'block':<28} {'cols':>5} {'keep':>5} {'redun':>6} {'dead':>5} {'cull%':>6}")
    print("  " + "-" * (W - 2))
    by_block = {}
    for c in feat_cols:
        b = col_block.get(c, "?")
        st = "DEAD" if stats[c]["dead"] else ("REDUN" if keeper.get(c) != c else "KEEP")
        d = by_block.setdefault(b, {"KEEP": 0, "REDUN": 0, "DEAD": 0})
        d[st] += 1
    def cullpct(d):
        tot = d["KEEP"] + d["REDUN"] + d["DEAD"]
        return (d["REDUN"] + d["DEAD"]) / tot if tot else 0
    for b, d in sorted(by_block.items(), key=lambda kv: -cullpct(kv[1])):
        tot = d["KEEP"] + d["REDUN"] + d["DEAD"]
        print(f"  {b:<28} {tot:>5} {d['KEEP']:>5} {d['REDUN']:>6} {d['DEAD']:>5} {cullpct(d)*100:>5.0f}%")

    # biggest redundant clusters
    from collections import Counter
    csize = Counter(keeper[c] for c in dense)
    big = [(k, v) for k, v in csize.items() if v > 1]
    big.sort(key=lambda kv: -kv[1])
    print(f"\n  Largest redundant clusters (keeper <- # members):")
    for k, v in big[:12]:
        members = [c for c in dense if keeper[c] == k and c != k][:4]
        more = "" if v - 1 <= 4 else f" +{v-1-4}"
        print(f"    {k:<30} x{v:<3}  e.g. {', '.join(members)}{more}")

    print("\n  " + "-" * (W - 2))
    print(f"  full per-feature verdict -> {args.out}")
    print("  NOTE: low univariate IC is NOT a cull signal (features matter in combination).")
    print("        Only REDUNDANT + DEAD-DATA are recommended culls; confirm with a retrain.")
    print("=" * W + "\n")


if __name__ == "__main__":
    main()
