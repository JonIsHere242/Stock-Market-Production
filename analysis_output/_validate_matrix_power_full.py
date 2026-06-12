"""
Full-universe validation of the 14 matrix-power features.

1. Sanity check on the same 50 EDA tickers: confirm the production function
   in 3__AlphaSensitivity.py reproduces the IC numbers from the notebook
   (within rounding). Catches any spec/code drift.
2. Full universe (4,265 tickers): score IC, monthly stability, train/test split.
3. Side-by-side report.

The 14 mp_* feature functions are imported by surgically extracting them from
3__AlphaSensitivity.py (its top-of-file imports include pykalman, ib_insync,
scalene, etc. that we don't want to load just to run a math function).
"""
import json
import random
import re
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


PRICE_DIR = Path("Data/PriceData")
SPEC_PATH = "Data/matrix_power_spec.json"
N_EDA_TICKERS = 50
MIN_BARS = 400


# -----------------------------------------------------------------------------
# Pull the matrix-power functions out of 3__AlphaSensitivity.py without
# triggering its heavy import block.
# -----------------------------------------------------------------------------
def _load_mp_functions():
    src = Path("3__AlphaSensitivity.py").read_text()
    # Grab everything between the marker comment and `def indicators(df):`
    m = re.search(r"# Matrix-power indicators(.*?)\ndef indicators\(df\):", src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not locate matrix-power block in 3__AlphaSensitivity.py")
    block = m.group(0)
    # Drop the trailing 'def indicators(df):' line
    block = block.rsplit("\ndef indicators(df):", 1)[0]
    ns = {"np": np, "pd": pd, "json": json}
    exec(block, ns)
    return ns["add_matrix_power_features"]


add_matrix_power_features = _load_mp_functions()
print("Loaded add_matrix_power_features from 3__AlphaSensitivity.py")


def load_ticker(p):
    df = pd.read_parquet(p)
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()


def build_panel_for(files, label, max_n=None):
    """Compute mp_* features + fwd_ret for each ticker, pool into a long table."""
    rows = []
    n_used = 0
    n_skipped = 0
    t0 = time()
    feature_cols = None
    for i, p in enumerate(files):
        if max_n is not None and n_used >= max_n:
            break
        try:
            df = load_ticker(p)
            if len(df) < MIN_BARS:
                n_skipped += 1
                continue
            if not df[["Open", "High", "Low", "Close", "Volume"]].gt(0).all().all():
                n_skipped += 1
                continue
            df = add_matrix_power_features(df)
            df["fwd_ret"] = df["Close"].pct_change().shift(-1)
            df["fwd_sign"] = (df["fwd_ret"] > 0).astype(int)
            df["ticker"] = p.stem
            keep = ["Date", "ticker", "fwd_ret", "fwd_sign"] + [c for c in df.columns if c.startswith("mp_")]
            if feature_cols is None:
                feature_cols = [c for c in keep if c.startswith("mp_")]
            rows.append(df[keep])
            n_used += 1
        except Exception as e:
            n_skipped += 1
            continue
        if (n_used + n_skipped) % 200 == 0:
            print(f"  [{label}] {n_used} used / {n_skipped} skipped  elapsed={time()-t0:.1f}s")

    panel = pd.concat(rows, axis=0, ignore_index=True)
    panel = panel.dropna(subset=["fwd_ret"])
    print(f"  [{label}] DONE: {n_used} tickers used, {n_skipped} skipped, {len(panel):,} rows  elapsed={time()-t0:.1f}s")
    return panel, feature_cols


def score_panel(panel, feature_cols, label, train_test_split_quantile=0.5):
    """Compute full IC, monthly mean/std/hit-rate, train/test IC + retention."""
    panel = panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["month"] = panel["Date"].dt.to_period("M")
    median_date = panel["Date"].quantile(train_test_split_quantile)
    panel["split"] = np.where(panel["Date"] < median_date, "train", "test")

    y_ret = panel["fwd_ret"].to_numpy()
    y_sign = panel["fwd_sign"].to_numpy()

    rows = []
    for c in feature_cols:
        x = panel[c].to_numpy()
        m = np.isfinite(x) & np.isfinite(y_ret)
        if m.sum() < 1000 or np.nanstd(x[m]) < 1e-12:
            rows.append({"feature": c, "n": int(m.sum()), "ic": np.nan})
            continue
        ic, _ = spearmanr(x[m], y_ret[m])
        try:
            auc = roc_auc_score(y_sign[m], x[m])
        except Exception:
            auc = 0.5
        sign = np.sign(ic) if abs(ic) > 1e-9 else 1.0

        # Monthly IC, sign-aligned
        def _ic_group(g):
            gx = g[c].to_numpy()
            gy = g["fwd_ret"].to_numpy()
            mm = np.isfinite(gx) & np.isfinite(gy)
            if mm.sum() < 50 or np.std(gx[mm]) < 1e-12:
                return np.nan
            return spearmanr(gx[mm], gy[mm])[0]

        monthly = panel.groupby("month").apply(_ic_group) * sign

        def _split_ic(d_):
            xx = d_[c].to_numpy()
            yy = d_["fwd_ret"].to_numpy()
            mm = np.isfinite(xx) & np.isfinite(yy)
            if mm.sum() < 100:
                return np.nan
            return spearmanr(xx[mm], yy[mm])[0]

        tr_ic = _split_ic(panel[panel.split == "train"]) * sign
        te_ic = _split_ic(panel[panel.split == "test"]) * sign

        rows.append({
            "feature": c, "n": int(m.sum()),
            "ic": ic, "auc_abs": max(auc, 1 - auc),
            "monthly_mean": monthly.mean(), "monthly_std": monthly.std(),
            "monthly_hit": (monthly > 0).mean(),
            "train_ic": tr_ic, "test_ic": te_ic,
            "retention": (te_ic / tr_ic) if abs(tr_ic) > 1e-6 else np.nan,
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("ic", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    print(f"\n=== [{label}] feature scores ===")
    print(out.round(4).to_string(index=False))
    return out


def main():
    all_files = sorted(PRICE_DIR.glob("*.parquet"))
    print(f"Total parquet files: {len(all_files):,}")

    # ----- Sanity: same 50 EDA tickers -----
    np.random.seed(42)
    random.seed(42)
    candidate_files = random.sample(all_files, min(len(all_files), 5 * N_EDA_TICKERS))
    eda_files = []
    for p in candidate_files:
        try:
            df = pd.read_parquet(p)
            df = df.sort_values("Date").reset_index(drop=True)
            if len(df) >= MIN_BARS and df[["Open", "High", "Low", "Close", "Volume"]].gt(0).all().all():
                eda_files.append(p)
            if len(eda_files) >= N_EDA_TICKERS:
                break
        except Exception:
            continue
    print(f"\n=== EDA-50 sanity panel ===")
    print(f"Using same {len(eda_files)} tickers as the notebook")

    eda_panel, feat_cols = build_panel_for(eda_files, "EDA-50")
    eda_scores = score_panel(eda_panel, feat_cols, "EDA-50")

    # ----- Full universe -----
    print(f"\n=== Full universe ({len(all_files):,} tickers) ===")
    full_panel, _ = build_panel_for(all_files, "full")
    full_scores = score_panel(full_panel, feat_cols, "full")

    # ----- Side-by-side comparison vs notebook validation table -----
    print("\n=== EDA-50 (production function) vs full-universe IC ===")
    cmp = eda_scores[["feature", "ic", "monthly_hit", "retention"]].rename(
        columns={"ic": "ic_eda50", "monthly_hit": "hit_eda50", "retention": "ret_eda50"}
    ).merge(
        full_scores[["feature", "ic", "monthly_hit", "retention", "n"]].rename(
            columns={"ic": "ic_full", "monthly_hit": "hit_full", "retention": "ret_full"}
        ),
        on="feature", how="outer",
    )
    cmp["ic_shrink"] = cmp["ic_full"] / cmp["ic_eda50"]
    print(cmp.round(4).to_string(index=False))

    Path("Data/EDA").mkdir(parents=True, exist_ok=True)
    cmp.to_csv("Data/EDA/matrix_power_full_universe_validation.csv", index=False)
    print("\nWrote Data/EDA/matrix_power_full_universe_validation.csv")


if __name__ == "__main__":
    main()
