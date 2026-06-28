"""
Momentary Trend-State Features (Proxy)  —  doi:10.1109/access.2023.3244680
"Deep Neural Network Architectures for Momentary Forecasting in Dry Bulk Markets:
 Robustness to the Impact of COVID-19"

PROXY RATIONALE:
The paper applies sparse DNN classifiers to classify the "momentary" state of dry bulk
shipping freight indices (Panamax, Supramax, Capesize) as Up / Flat / Down over very
short horizons. The architecture uses sparsity-constrained networks trained on lagged
index values. We cannot replicate:
  - The DNN architecture or weights
  - Freight index data (shipping-specific; not OHLCV)
  - Training data or sparsity constraints

PROXY IMPLEMENTED:
The paper's core insight is that "momentary forecasting" uses the IMMEDIATE multi-lag
pattern of the series (very short lookback, 1-5 lags) rather than statistical smoothing.
We implement:
  1. Momentary trend classification via short-window linear slope consistency across
     multiple time scales (1, 2, 3, 5, 10 lags) — capturing the DNN's multi-lag input structure.
  2. Trend-state entropy: consistency/disagreement across scales (analogous to the DNN
     classifying a single compound state across lags).
  3. Short-horizon directional momentum indicators at the paper's characteristic windows.
  4. A "sparsity proxy": fraction of recent lags showing the same sign as the current
     short-window trend (inspired by the paper's sparse architecture finding only a few
     inputs matter).

All features are per-ticker, causal, and OHLCV-only.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_access20233244_deep_neural_network",
    "description": (
        "Momentary trend-state proxy features inspired by doi:10.1109/access.2023.3244680 "
        "sparse DNN momentary forecasting; implements multi-lag slope consistency, trend-state "
        "entropy, and directional sparsity as OHLCV-only proxies (DNN/freight-index dropped)."
    ),
    "requires": ["Close"],
    "produces": [
        "mom_slope_1d",
        "mom_slope_3d",
        "mom_slope_5d",
        "mom_slope_10d",
        "mom_trend_agree_5",
        "mom_trend_agree_10",
        "mom_state_entropy_5",
        "mom_state_entropy_10",
        "mom_lag_sign_sparsity_5",
        "mom_lag_sign_sparsity_10",
    ],
    "tags": ["momentum", "trend", "experimental"],
    "version": "1.0",
    "author": (
        "proxy: doi:10.1109/access.2023.3244680 (sparse DNN momentary forecasting). "
        "DNN, sparsity constraints, and freight-index data dropped (not OHLCV-replicable). "
        "Replaced by multi-lag slope consistency and trend-state entropy proxies."
    ),
}


def _sign3(x: np.ndarray) -> np.ndarray:
    """Map to ternary state: +1 (up), 0 (flat), -1 (down). Flat band = 0.5% of value."""
    return np.where(np.abs(x) < 0.005, 0.0, np.sign(x))


def _ternary_entropy(states: np.ndarray) -> float:
    """Shannon entropy of a ternary (+1, 0, -1) state array. Max = log2(3) ~ 1.585."""
    states = states[np.isfinite(states)]
    if len(states) == 0:
        return np.nan
    total = len(states)
    ent = 0.0
    for v in [-1.0, 0.0, 1.0]:
        p = np.sum(states == v) / total
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentary trend-state proxy features.

    Each feature captures the multi-lag directional pattern the paper's DNN
    classifies into a momentary state. We use deterministic slope / sign
    consistency measures across windows of 1, 3, 5, 10 lags.
    """
    c = df["Close"].values.astype(np.float64)
    n = len(c)

    # --- Single-lag log returns at characteristic windows ---
    lret = np.full(n, np.nan)
    lret[1:] = np.log(c[1:] / np.where(c[:-1] > 0, c[:-1], np.nan))

    # Slopes at 1, 3, 5, 10 day windows (log-return over window / window)
    for w, col in [(1, "mom_slope_1d"), (3, "mom_slope_3d"),
                   (5, "mom_slope_5d"), (10, "mom_slope_10d")]:
        slope = np.full(n, np.nan)
        if w == 1:
            slope = lret.copy()
        else:
            for i in range(w, n):
                if c[i - w] > 0 and c[i] > 0:
                    slope[i] = np.log(c[i] / c[i - w]) / w
        df[col] = slope

    # --- Trend agreement: fraction of last W lag-returns sharing sign with current slope ---
    for W in (5, 10):
        agree = np.full(n, np.nan)
        for i in range(W, n):
            seg = lret[i - W + 1: i + 1]   # last W returns (inclusive)
            fin = seg[np.isfinite(seg)]
            if len(fin) < 3:
                continue
            cur_sign = np.sign(fin[-1]) if fin[-1] != 0 else 0.0
            if cur_sign == 0.0:
                agree[i] = 0.5
            else:
                agree[i] = float(np.mean(np.sign(fin) == cur_sign))
        df[f"mom_trend_agree_{W}"] = agree

    # --- State entropy: how mixed are the ternary states over the window ---
    for W in (5, 10):
        ent_arr = np.full(n, np.nan)
        for i in range(W, n):
            seg = lret[i - W + 1: i + 1]
            states = _sign3(seg[np.isfinite(seg)])
            if len(states) >= 3:
                ent_arr[i] = _ternary_entropy(states)
        df[f"mom_state_entropy_{W}"] = ent_arr

    # --- Lag sign sparsity (DNN sparsity proxy) ---
    # Fraction of last W lag-returns that are "active" (|ret| > median |ret| over window)
    # High sparsity = only a few lags dominate, matching sparse DNN architecture
    for W in (5, 10):
        sparse_arr = np.full(n, np.nan)
        for i in range(W, n):
            seg = lret[i - W + 1: i + 1]
            fin = seg[np.isfinite(seg)]
            if len(fin) < 3:
                continue
            med = np.median(np.abs(fin))
            if med < 1e-10:
                sparse_arr[i] = 0.0
            else:
                # fraction with |ret| > median (active lags)
                sparse_arr[i] = float(np.mean(np.abs(fin) > med))
        df[f"mom_lag_sign_sparsity_{W}"] = sparse_arr

    return df
