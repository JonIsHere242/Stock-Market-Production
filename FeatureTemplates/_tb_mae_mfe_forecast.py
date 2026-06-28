"""
_tb_mae_mfe_forecast.py  --  CANDIDATE block for the `tb` (stop-aware) and `dd` (next-low) arms.

Forecasts the next bar's close-to-LOW excursion (Maximum Adverse Excursion -- the governing variable
of a tb stop-out AND the exact dd target) and close-to-HIGH excursion (MFE -> target hit). Emits a
Gaussian stop-probability, the empirical stop/target frequency, the conditional shortfall (CVaR) of
the low, and the dimensionless tb_recovery_ratio = target-prob / stop-prob (vol-scale invariant ->
survives the abs-vol neutralization the raw close-to-close axis does not).

The library FORECASTS no next-bar low: path_drawdown/_p625_drawdown_norms are TRAILING drawdown from a
running peak; orig_orig_stress_indicators is SAME-day open-to-low; the cvar blocks operate on
close-to-close returns -- none on the close-to-LOW excursion. Additive to _tb_empirical_touch_freq
(forecast model vs realized base rate).

Refs: Sweeney (1996) "Maximum Adverse Excursion" (Wiley); Barndorff-Nielsen, Kinnebrock & Shephard
(2010) realised semivariance. Candidate -- NOT promoted until the multi-seed (>=4) ablation gate.
"""

import numpy as np
import pandas as pd

try:                                  # standard-normal CDF, vectorized & NaN-safe
    from scipy.special import ndtr as _ndtr
except Exception:                      # pragma: no cover - fallback if scipy absent
    from math import erf as _erf
    _ndtr = np.vectorize(lambda z: 0.5 * (1.0 + _erf(z / np.sqrt(2.0))) if np.isfinite(z) else np.nan)

METADATA = {
    "name":        "_tb_mae_mfe_forecast",
    "description": "Next-bar adverse/favorable excursion forecast: Gaussian stop-prob at -1.9%, "
                   "empirical 126d stop/target freq, 63d CVaR(5%) of the next low, and the "
                   "dimensionless recovery ratio (target-prob/stop-prob). Sweeney 1996 MAE/MFE.",
    "requires":    ["High", "Low", "Close"],
    "produces":    ["mae_stop_prob_63", "mae_stop_freq_126", "mae_cvar05_63",
                    "mfe_tgt_prob_63", "mfe_tgt_freq_126", "tb_recovery_ratio_63"],
    "tags":        ["tb", "dd", "downside", "candidate"],
    "version":     "0.1",
    "author":      "alt-target-feature-research 2026-06-26 (tb/dd white-space)",
}

_STOP = -0.019
_TGT = 0.035


def _phi(x: pd.Series) -> pd.Series:
    return pd.Series(_ndtr(x.to_numpy(dtype=float)), index=x.index)


def _cvar05(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size < 5:
        return np.nan
    thr = np.quantile(a, 0.05)
    tail = a[a <= thr]
    return float(tail.mean()) if tail.size else np.nan


def compute(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["Close"].shift(1)
    dlow = df["Low"] / prev_close - 1.0            # close-to-low excursion (MAE), known at close t
    dhigh = df["High"] / prev_close - 1.0          # close-to-high excursion (MFE)

    # SHIFT(1): trailing stats see only past excursions
    dl = dlow.shift(1)
    dh = dhigh.shift(1)

    mu_dl = dl.rolling(63, min_periods=31).mean()
    sd_dl = dl.rolling(63, min_periods=31).std().replace(0, np.nan)
    df["mae_stop_prob_63"] = _phi((_STOP - mu_dl) / sd_dl)

    df["mae_stop_freq_126"] = (dl <= _STOP).astype(float).where(dl.notna()) \
        .rolling(126, min_periods=63).mean()

    df["mae_cvar05_63"] = dl.rolling(63, min_periods=20).apply(_cvar05, raw=True)

    mu_dh = dh.rolling(63, min_periods=31).mean()
    sd_dh = dh.rolling(63, min_periods=31).std().replace(0, np.nan)
    df["mfe_tgt_prob_63"] = 1.0 - _phi((_TGT - mu_dh) / sd_dh)

    df["mfe_tgt_freq_126"] = (dh >= _TGT).astype(float).where(dh.notna()) \
        .rolling(126, min_periods=63).mean()

    df["tb_recovery_ratio_63"] = (df["mfe_tgt_prob_63"]
                                  / (df["mae_stop_prob_63"] + 1e-4)).clip(0.0, 20.0)
    return df
