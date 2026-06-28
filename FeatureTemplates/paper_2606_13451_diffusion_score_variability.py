"""
Diffusion-trajectory noise variability features derived from:
  "Uncertainty Estimation for Molecular Diffusion Models"
  arXiv 2606.13451

The paper proposes measuring per-sample uncertainty in diffusion models by
tracking the VARIABILITY of the denoising network's noise prediction across
the generation trajectory (multiple denoising steps at different noise levels).
High variability across the trajectory = the model is uncertain about this sample.
Key finding: uncertainty score negatively correlates with sample quality.

We translate to OHLCV: treat the closing price as the "observed sample"
and define a family of "denoisers" (exponential smoothers at different scales,
analogous to denoising at different noise levels / timesteps in the trajectory).
The DISAGREEMENT among denoiser residuals is the uncertainty signal.

  - Residual from EMA-N = "noise prediction at noise level N" (larger N = more
    noise, i.e., coarser denoising step).
  - Cross-scale standard deviation of residuals = trajectory variability.
  - The SIGNED SPREAD (max resid - min resid) captures directional disagreement.
  - Rate of change of the spread = trajectory curvature (second-order signal).
  - Disagreement between fine-scale and coarse-scale residuals identifies
    whether the price has diverged from its slow trend (trend-break signal).

Feature family (7 cols):
  dsv_noise_spread        cross-scale residual std at each bar (uncertainty)
  dsv_noise_spread_z20    z-score of spread over 20-day window
  dsv_noise_spread_z60    z-score of spread over 60-day window
  dsv_trajectory_curv     2nd-difference of EMA-20 residual (curvature)
  dsv_fine_coarse_gap     EMA-5 residual minus EMA-40 residual (directional)
  dsv_spread_ratio        10d / 40d mean spread (regime shift detector)
  dsv_surprise_rank       percentile rank of today's spread in trailing 40 bars
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13451_diffusion_score_variability",
    "description": (
        "Cross-scale EMA denoiser disagreement and trajectory curvature as "
        "diffusion-style uncertainty signals; from arXiv 2606.13451 "
        "(Uncertainty Estimation for Molecular Diffusion Models)."
    ),
    "requires": ["Close"],
    "produces": [
        "dsv_noise_spread",
        "dsv_noise_spread_z20",
        "dsv_noise_spread_z60",
        "dsv_trajectory_curv",
        "dsv_fine_coarse_gap",
        "dsv_spread_ratio",
        "dsv_surprise_rank",
    ],
    "tags": ["volatility", "experimental"],
    "version": "1.1",
    "author": "paper:2606.13451",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan)
    log_p = np.log(close)

    # ---- Four "denoisers": EMA smoothers at 4 scales --------------------------
    # Residual = log_price - EMA(log_price, span) → "noise estimate at scale span"
    spans = [5, 10, 20, 40]
    resid = {}
    for sp in spans:
        ema       = log_p.ewm(span=sp, min_periods=sp // 2, adjust=False).mean()
        resid[sp] = log_p - ema

    resid_mat = np.column_stack([resid[sp].values for sp in spans])  # (n, 4)

    # ---- Cross-scale std (trajectory variability / uncertainty) ---------------
    with np.errstate(invalid="ignore"):
        noise_spread = np.nanstd(resid_mat, axis=1, ddof=0)
    all_nan = np.all(~np.isfinite(resid_mat), axis=1)
    noise_spread[all_nan] = np.nan

    spread_s = pd.Series(noise_spread, index=df.index)

    # ---- Fine-coarse gap: EMA-5 resid minus EMA-40 resid (directional) --------
    # Positive: fine-scale price above slow trend → upward divergence
    fine_coarse_gap = resid[5] - resid[40]

    # ---- Trajectory curvature: second difference of EMA-20 residual -----------
    trajectory_curv = resid[20].diff().diff()

    # ---- Z-scores of spread ---------------------------------------------------
    def z_score(s: pd.Series, w: int) -> pd.Series:
        mu  = s.rolling(w, min_periods=w // 3).mean()
        sig = s.rolling(w, min_periods=w // 3).std()
        return (s - mu) / sig.replace(0.0, np.nan)

    spread_z20 = z_score(spread_s, 20)
    spread_z60 = z_score(spread_s, 60)

    # ---- Spread ratio: short / long mean (regime shift) -----------------------
    short_mean  = spread_s.rolling(10, min_periods=3).mean()
    long_mean   = spread_s.rolling(40, min_periods=10).mean()
    spread_ratio = short_mean / long_mean.replace(0.0, np.nan)

    # ---- Surprise rank: pctile of today's spread in trailing 40 bars ----------
    def surprise_rank(s: pd.Series, w: int = 40) -> pd.Series:
        def _rank(x: np.ndarray) -> float:
            if len(x) < 3 or not np.isfinite(x[-1]):
                return np.nan
            return float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1)
        return s.rolling(w, min_periods=w // 3).apply(_rank, raw=True)

    surprise_rank_s = surprise_rank(spread_s, 40)

    # ---- Assign ---------------------------------------------------------------
    df["dsv_noise_spread"]     = spread_s
    df["dsv_noise_spread_z20"] = spread_z20
    df["dsv_noise_spread_z60"] = spread_z60
    df["dsv_trajectory_curv"]  = trajectory_curv
    df["dsv_fine_coarse_gap"]  = fine_coarse_gap
    df["dsv_spread_ratio"]     = spread_ratio
    df["dsv_surprise_rank"]    = surprise_rank_s

    return df
