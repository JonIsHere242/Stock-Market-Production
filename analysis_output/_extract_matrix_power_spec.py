"""
Re-derive deterministically the 14 winning matrix-power builders from the EDA notebook
and save them, along with the per-primitive scaling constants, to a JSON spec that
3__AlphaSensitivity.py can consume in production.

Re-runs only what's necessary:
  - Same ticker selection (seeds 42)
  - Same primitive computation
  - Same panel-wide scales (derived from those 50 tickers)
  - Same builder generation (seed 456) for d=4 then d=5
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_DIR = Path("Data/PriceData")
OUT_PATH = Path("Data/matrix_power_spec.json")
N_TICKERS = 50
MIN_BARS = 400
WINDOW = 20

# 14 winning feature names from the notebook cell 33 output (post-dedup, all stable)
WINNER_NAMES = [
    "d4_b185_sym_trace",
    "d4_b65_sym_frob",
    "d5_b108_sym_trace",
    "d4_b306_raw_top_left",
    "d5_b169_sym_top_left",
    "d4_b267_sym_trace",
    "d4_b291_sym_frob",
    "d4_b173_sym_top_left",
    "d4_b50_raw_top_left",
    "d4_b364_skew_top_left",
    "d4_b396_sym_trace",
    "d4_b291_sym_top_left",
    "d4_b477_skew_trace",
    "d5_b39_sym_trace",
]

PRIMITIVE_NAMES = [
    "log_ret", "log_vol",
    "range_pct", "body_pct",
    "upper_wick", "lower_wick",
    "rv", "z_close", "z_vol",
    "skew", "kurt", "mom5", "mom20",
    "sign_last",
]


def load_same_50_tickers():
    np.random.seed(42)
    random.seed(42)
    all_files = sorted(PRICE_DIR.glob("*.parquet"))
    candidate_files = random.sample(all_files, min(len(all_files), 5 * N_TICKERS))
    loaded = []
    for p in candidate_files:
        try:
            df = pd.read_parquet(p)
            df = df.sort_values("Date").reset_index(drop=True)
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df["Ticker"] = p.stem
            if len(df) >= MIN_BARS and df[["Open", "High", "Low", "Close", "Volume"]].gt(0).all().all():
                loaded.append(df)
            if len(loaded) >= N_TICKERS:
                break
        except Exception:
            continue
    return loaded


def primitives_panel(df, window=WINDOW):
    df = df.sort_values("Date").set_index(pd.DatetimeIndex(df["Date"]))
    O, H, L, C, V = (df[c].astype(float) for c in ["Open", "High", "Low", "Close", "Volume"])
    log_ret = np.log(C).diff()
    log_vol = np.log(V.replace(0, np.nan)).diff()
    range_pct = (H - L) / C
    body_pct = (C - O) / C
    upper_wick = (H - np.maximum(O, C)) / C
    lower_wick = (np.minimum(O, C) - L) / C
    rv = log_ret.rolling(window).std()
    z_close = (C - C.rolling(window).mean()) / (C.rolling(window).std() + 1e-9)
    z_vol = (V - V.rolling(window).mean()) / (V.rolling(window).std() + 1e-9)
    skew_ = log_ret.rolling(window).skew()
    kurt_ = log_ret.rolling(window).kurt()
    mom5 = np.log(C / C.shift(5))
    mom20 = np.log(C / C.shift(window))
    sign_last = np.sign(log_ret)
    return pd.DataFrame({
        "log_ret": log_ret, "log_vol": log_vol,
        "range_pct": range_pct, "body_pct": body_pct,
        "upper_wick": upper_wick, "lower_wick": lower_wick,
        "rv": rv, "z_close": z_close, "z_vol": z_vol,
        "skew": skew_, "kurt": kurt_, "mom5": mom5, "mom20": mom20,
        "sign_last": sign_last,
    })


def main():
    print("Loading the same 50 tickers used in the EDA notebook...")
    loaded = load_same_50_tickers()
    print(f"  loaded {len(loaded)} tickers; first 5: {[d.Ticker.iloc[0] for d in loaded[:5]]}")

    # Build aligned primitives panel exactly as the notebook does
    panels = []
    prim_per_ticker = []
    for df in loaded:
        # Mimic compute_panel's index alignment: starts at row WINDOW
        N = len(df)
        idx = pd.DatetimeIndex(df["Date"].values[WINDOW:N], name="Date")
        # placeholder panel just to align indices
        panels.append(pd.DataFrame(index=idx))

        pp = primitives_panel(df, WINDOW)
        pp = pp.reindex(idx)
        prim_per_ticker.append(pp)
    prim_full = pd.concat(prim_per_ticker, axis=0)

    prim_arr = prim_full.values.astype(np.float64)
    scales = np.nanstd(prim_arr, axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    print("Panel-wide primitive scales:")
    for n, s in zip(PRIMITIVE_NAMES, scales):
        print(f"  {n:12s} std={s:.6f}")

    # Re-derive builders (must match notebook seed/order)
    rng = np.random.default_rng(456)
    builders = {}
    for D, N_RAND in [(4, 500), (5, 250)]:
        idx_arr = rng.integers(0, len(PRIMITIVE_NAMES), size=(N_RAND, D, D))
        sgn_arr = rng.choice([-1.0, 1.0], size=(N_RAND, D, D))
        builders[D] = (idx_arr, sgn_arr)

    # Build the spec list
    spec_features = []
    for name in WINNER_NAMES:
        # parse name: dN_bM_VARIANT_EXTRACTOR
        rest = name[1:]  # drop leading 'd'
        d_str, rest = rest.split("_", 1)
        d = int(d_str)
        b_str, rest = rest.split("_", 1)
        assert b_str.startswith("b")
        builder_id = int(b_str[1:])
        variant, extractor = rest.split("_", 1)

        idx_arr, sgn_arr = builders[d]
        idx_mat = idx_arr[builder_id].tolist()
        sgn_mat = sgn_arr[builder_id].astype(int).tolist()

        spec_features.append({
            "name": name,
            "d": d,
            "builder_id": builder_id,
            "variant": variant,
            "extractor": extractor,
            "idx_matrix": idx_mat,
            "sign_matrix": sgn_mat,
        })

    spec = {
        "version": 1,
        "window": WINDOW,
        "primitive_names": PRIMITIVE_NAMES,
        "primitive_scales": scales.tolist(),
        "features": spec_features,
        "metadata": {
            "source_notebook": "10_MatrixPowerIndicators.ipynb",
            "n_tickers_in_eda": N_TICKERS,
            "min_bars_in_eda": MIN_BARS,
            "winner_dedup_corr_threshold": 0.9,
            "stability_filter": "monthly_hit >= 0.55 AND retention > 0.3",
            "rng_seed_for_builders": 456,
            "builder_configs": [{"d": 4, "n_random": 500}, {"d": 5, "n_random": 250}],
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(spec, indent=2))
    print(f"\nWrote spec for {len(spec_features)} features to {OUT_PATH}")
    print(f"  Spec file size: {OUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
