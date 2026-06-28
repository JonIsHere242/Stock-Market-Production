"""
Perception-Distortion Features  —  arxiv:2606.12987
"Diffusion Transformer World-Action Model for AV Scene Prediction"

The paper identifies a "perception-distortion frontier": high-fidelity
predictors (low distortion) differ from realistic-distribution predictors
(low perception error / KID). Regression to the mean = blurry but low-MSE;
distributional realism = sharp but high-MSE single-path.

Applied to OHLCV: decompose the rolling price-prediction problem into:
  - DISTORTION component: MSE of a naive AR(1) forecast vs realized return
  - PERCEPTION component: how far realized return is from rolling median
    (distributional surprise — median is mode proxy for unimodal distributions)
  - FRONTIER score: geometric mean of normalized distortion and perception
  - FID analog: Frechet-style distance between first-half and second-half
    return distributions in a rolling window (moment-based, vectorized)

All computations are fully vectorized using pandas rolling — no Python loops.

Produces 8 columns prefixed "pdc_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12987_perception_distortion",
    "description": (
        "Perception-distortion frontier features inspired by arxiv:2606.12987 — "
        "decompose rolling price prediction error into distortion (AR bias) and "
        "perception (distributional surprise) components, plus FID-analog "
        "distribution divergence between rolling sub-windows. Fully vectorized."
    ),
    "requires": ["Close"],
    "produces": [
        "pdc_distortion_21d",
        "pdc_distortion_63d",
        "pdc_perception_21d",
        "pdc_perception_63d",
        "pdc_frontier_21d",
        "pdc_frontier_63d",
        "pdc_fid_analog_42d",
        "pdc_composite",
    ],
    "tags": ["experimental", "volatility", "market_regime"],
    "version": "1.0",
    "author": "paper:2606.12987",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"].astype(np.float64)
    ret = np.log(C / C.shift(1))

    for w, suffix in [(21, "21d"), (63, "63d")]:
        mp = max(5, w // 3)

        # --- Distortion: rolling MSE of AR(1) forecast error ---
        # AR(1) prediction of ret[t] = rolling_mean of ret[t-w..t-1]
        roll_mean_lagged = ret.rolling(w, min_periods=mp).mean().shift(1)
        forecast_err = (ret - roll_mean_lagged) ** 2
        df[f"pdc_distortion_{suffix}"] = forecast_err.rolling(w, min_periods=mp).mean()

        # --- Perception: |ret - rolling_median| / rolling_std ---
        # Vectorized: rolling median (pandas), normalized by rolling std
        roll_med = ret.rolling(w, min_periods=mp).median().shift(1)
        roll_std = ret.rolling(w, min_periods=mp).std().shift(1)
        df[f"pdc_perception_{suffix}"] = (ret - roll_med).abs() / (roll_std + 1e-9)

        # --- Frontier: geometric mean of normalized distortion and perception ---
        d = df[f"pdc_distortion_{suffix}"]
        p = df[f"pdc_perception_{suffix}"]
        d_norm = d / (d.rolling(126, min_periods=w).mean() + 1e-12)
        p_norm = p / (p.rolling(126, min_periods=w).mean() + 1e-9)
        df[f"pdc_frontier_{suffix}"] = np.sqrt(d_norm.clip(lower=0) * p_norm.clip(lower=0))

    # --- FID analog: Frechet distance between first-half and second-half of 42d window ---
    # FD = (mu1-mu2)^2 + (sig1-sig2)^2  (vectorized via two rolling windows)
    half = 21
    full = 42
    mp_h = 5

    mu_full = ret.rolling(full, min_periods=full).mean()
    mu_half = ret.rolling(half, min_periods=mp_h).mean()
    std_full = ret.rolling(full, min_periods=full).std()
    std_half = ret.rolling(half, min_periods=mp_h).std()

    # First half [t-full, t-half]: mean = (mu_full * full - mu_half * half) / half
    mu1_num = mu_full * full - mu_half * half
    mu1 = mu1_num / half
    # Variance of first half via sum of squared deviations (Welford-style approximation)
    # Use variance difference: var_full*full ~ var1*half + var2*half + ...
    # Simplified: approximate var1 via the complement moment
    var_full = (std_full ** 2) * (full - 1)
    var_half = (std_half ** 2) * (half - 1)
    # Residual: total SS = SS1 + SS2 + cross-term; approximate SS1 from remainder
    ss_half = var_half  # second-half SS
    ss_total = var_full
    ss_first = (ss_total - ss_half).clip(lower=0)
    std1 = np.sqrt(ss_first / max(half - 1, 1))
    std2 = std_half / (half - 1 + 1e-9)  # already have std for second half

    df["pdc_fid_analog_42d"] = (mu1 - mu_half) ** 2 + (std1 - std_half) ** 2

    # Composite: z-score the frontier scores and average
    def _z(s, w=126, mp=42):
        return (s - s.rolling(w, min_periods=mp).mean()) / (
            s.rolling(w, min_periods=mp).std() + 1e-9
        )

    fz21 = _z(df["pdc_frontier_21d"])
    fz63 = _z(df["pdc_frontier_63d"])
    fid_z = _z(df["pdc_fid_analog_42d"])
    df["pdc_composite"] = (fz21 + fz63 + fid_z) / 3.0

    return df
