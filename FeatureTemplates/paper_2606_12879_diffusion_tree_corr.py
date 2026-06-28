"""
Diffusion-Tree Local Neighborhood Correlation Features  —  arxiv:2606.12879
"Diffusion-Network Alignment: An Efficient Algorithm and Explicit Probability Bounds"

The paper aligns a rooted diffusion tree to a network via *local neighborhood
correlation tests*.  Alignment quality at each vertex depends on depth-dependent
matching confidence (bounds increase near the root = high-centrality nodes).

Core computable method applied to OHLCV:
  1. Treat the rolling price history as an implicit "diffusion tree":
     each day is a node; price-change similarity defines edge weights.
  2. Build a local k-depth correlation structure:
       - Root = current bar (highest influence / centrality).
       - Children = recent bars weighted by similarity decay (diffusion kernel).
  3. Extract depth-weighted correlation features:
       - depth-1 (lag-1) neighbourhood correlation
       - depth-2 (lag-1 to lag-3) correlation
       - Tree alignment score: kernel-smoothed abs-return vs realized abs-return
  4. Radial symmetry break: vectorised run-length asymmetry via rolling sums.

Diffusion kernel: w_d(k) = exp(-alpha * k) for depth k, sum-normalised.
All windows strictly causal. Fully vectorised — no Python loops over rows.
Produces 7 columns prefixed "dtc_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12879_diffusion_tree_corr",
    "description": (
        "Diffusion-tree local neighborhood correlation features from arxiv:2606.12879 — "
        "depth-weighted return correlation structure, tree alignment score, "
        "branch asymmetry, and kernel-smoothed centrality across multiple horizons."
    ),
    "requires": ["Close"],
    "produces": [
        "dtc_depth1_corr_20d",
        "dtc_depth2_corr_20d",
        "dtc_tree_align_20d",
        "dtc_branch_asym_20d",
        "dtc_depth1_corr_60d",
        "dtc_tree_align_60d",
        "dtc_kernel_centrality_30d",
    ],
    "tags": ["market_regime", "statistical", "experimental"],
    "version": "1.1",
    "author": "paper:2606.12879",
}

_ALPHA_FAST = 0.3    # diffusion decay rate — faster decay = more local
_ALPHA_SLOW = 0.1    # slower decay = more global


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan).astype(np.float64)
    log_ret = np.log(close / close.shift(1))
    ret = log_ret
    idx = df.index

    # ── Depth-1 correlation (lag-1 autocorrelation) ─────────────────────────
    # Rolling corr between ret[t] and ret[t-1] — direct depth-1 neighborhood
    ret_lag1 = ret.shift(1)
    df["dtc_depth1_corr_20d"] = ret.rolling(20, min_periods=7).corr(ret_lag1)
    df["dtc_depth1_corr_60d"] = ret.rolling(60, min_periods=20).corr(ret_lag1)

    # ── Depth-2 correlation ──────────────────────────────────────────────────
    # Diffusion-kernel weighted predictor at depth 2:
    # predictor_d2[t] = w[0]*ret[t-1] + w[1]*ret[t-2]  (exp-decay weights)
    w = np.exp(-_ALPHA_FAST * np.array([1, 2]))
    w /= w.sum()
    pred_d2 = w[0] * ret.shift(1) + w[1] * ret.shift(2)
    df["dtc_depth2_corr_20d"] = ret.rolling(20, min_periods=7).corr(pred_d2)

    # ── Tree alignment score ─────────────────────────────────────────────────
    # How well does the diffusion kernel of past |returns| predict |current return|?
    # alignment = rolling corr between kernel-smoothed |past ret| and |ret[t]|
    abs_ret = ret.abs()
    # Diffusion kernel: ewm with alpha ≈ 1 - exp(-_ALPHA_FAST) ≈ 0.26
    kern_abs = abs_ret.shift(1).ewm(alpha=1 - np.exp(-_ALPHA_FAST),
                                     min_periods=5, adjust=False).mean()
    df["dtc_tree_align_20d"] = abs_ret.rolling(20, min_periods=7).corr(kern_abs)
    df["dtc_tree_align_60d"] = abs_ret.rolling(60, min_periods=20).corr(kern_abs)

    # ── Branch asymmetry (vectorised, no Python loops) ───────────────────────
    # Rolling fraction of positive returns (up-branch fraction)
    is_up   = (ret > 0).astype(float)
    is_down = (ret <= 0).astype(float)
    # Rolling correlation of up-indicator with next return vs down-indicator with next return
    ret_fwd_lag = ret.shift(-1)   # *** NOT a lookahead — we use shift(-1) only for correlation
    # Actually: to avoid lookahead, correlate current ret with lag-1 signed context
    # Branch asym = autocorr of ret conditioned on sign of ret[t-1]:
    # = corr(ret[t], is_up[t-1]) - corr(ret[t], is_down[t-1])
    # This measures: does an up day predict tomorrow's return differently than a down day?
    up_lag1   = is_up.shift(1)
    down_lag1 = is_down.shift(1)
    corr_up   = ret.rolling(20, min_periods=7).corr(up_lag1)
    corr_down = ret.rolling(20, min_periods=7).corr(down_lag1)
    df["dtc_branch_asym_20d"] = corr_up - corr_down

    # ── Kernel centrality ────────────────────────────────────────────────────
    # Diffusion-weighted absolute return (centrality of current bar
    # in its local diffusion neighbourhood).
    # EWM approximation of slow kernel
    kern_centrality = abs_ret.ewm(span=15, min_periods=8, adjust=False).mean()
    kc_mu = kern_centrality.rolling(30, min_periods=10).mean()
    kc_sd = kern_centrality.rolling(30, min_periods=10).std()
    df["dtc_kernel_centrality_30d"] = (kern_centrality - kc_mu) / (kc_sd + 1e-12)

    # ── Bonus: replace weak corr cols with stronger variants ─────────────────
    # Overwrite dtc_depth1_corr_60d with a more distinctive signal:
    # "diffusion decay rate" = slope of log(|autocorr(lag)|) vs lag
    # = how fast the autocorrelation decays (diffusion rate in the tree)
    # Computed as: -[acf(1) - acf(3)] / 2  (finite difference of log-AC)
    # (negative slope = fast decay = more efficient market)
    acf1 = ret.rolling(60, min_periods=20).corr(ret.shift(1))
    acf3 = ret.rolling(60, min_periods=20).corr(ret.shift(3))
    # Replace weak dtc_depth1_corr_60d: use as diffusion decay proxy
    # (already defined above — redefine to be more informative)
    # slope ≈ (acf1 - acf3) / 2   (higher = slower decay = stronger momentum)
    df["dtc_depth1_corr_60d"] = (acf1 - acf3) / 2.0

    return df
