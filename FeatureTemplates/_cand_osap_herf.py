"""
osap_herf — Herfindahl Industry-Concentration proxy (per-ticker OHLCV + fundamentals)

CROSS-SECTIONAL NOTE: The canonical OSAP `herf` factor (Hou & Robinson 2006) is
inherently cross-sectional: it is the Herfindahl-Hirschman Index (HHI) of *industry*
sales concentration computed across ALL firms in a 4-digit SIC code.  High-HHI
(concentrated) industries earn higher returns, presumably as a distress/entry-barrier
premium.

Per-ticker proxy implemented here:
  1. osap_herf_rev_conc  — rolling 4-quarter coefficient-of-variation of TTM revenue
     (from PIT fundamentals).  A firm whose revenue comes in very smooth / concentrated
     bursts has high concentration risk analogous to an oligopolistic market.  Higher =
     more concentrated revenue dynamics.
  2. osap_herf_vol_hhi   — rolling 63-day Herfindahl of *daily dollar-volume share*
     within the window.  Captures whether trading activity is dominated by a few
     high-volume days (lumpy / event-driven) vs. spread uniformly.  Pure OHLCV.
  3. osap_herf_vol_hhi_z — 252-day rolling z-score of osap_herf_vol_hhi, capturing
     changes in the concentration regime.

Both revenue-based and volume-based concentration are monotone-transformable signals
that share the same economic spirit (concentration → premium) without leakage.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional: PIT fundamentals
# ---------------------------------------------------------------------------
_fund_spec = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_fund_spec)
try:
    _fund_spec.loader.exec_module(_fundamentals)
    _HAS_FUND = True
except Exception:
    _HAS_FUND = False

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_herf",
    "description": (
        "Per-ticker proxy for OSAP Herfindahl industry-concentration factor "
        "(Hou & Robinson 2006). Canonical signal is cross-sectional HHI of "
        "industry sales; implemented here as: (1) rolling CoV of PIT TTM revenue "
        "(revenue-concentration proxy) and (2) rolling 63-day Herfindahl of daily "
        "dollar-volume share within the window (trading-concentration proxy), plus "
        "its 252-day z-score. Predicted sign: +1 (high concentration -> premium)."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "osap_herf_rev_conc",
        "osap_herf_vol_hhi",
        "osap_herf_vol_hhi_z",
    ],
    "tags": ["concentration", "herfindahl", "liquidity", "fundamentals", "osap"],
    "version": "1.0",
    "author": "Hou & Robinson 2006 (OSAP: Chen-Zimmermann); per-ticker proxy impl.",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VOL_WIN = 63       # ~1 quarter of trading days for HHI window
_Z_WIN = 252        # 1 year for z-score baseline
_REV_QUARTERS = 8   # look back 8 quarterly obs for revenue CoV


def _rolling_hhi(dollar_vol: np.ndarray) -> float:
    """Herfindahl index of dollar-volume shares for a 1-D window array.

    HHI = sum(s_i^2) where s_i = dv_i / sum(dv).  Returns NaN if all zero.
    """
    total = dollar_vol.sum()
    if total <= 0.0:
        return np.nan
    shares = dollar_vol / total
    return float((shares ** 2).sum())


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # ------------------------------------------------------------------
    # 1. Dollar-volume HHI (pure OHLCV, always available)
    # ------------------------------------------------------------------
    dv = (df["Close"] * df["Volume"]).values.astype(np.float64)

    hhi_vals = np.full(n, np.nan, dtype=np.float64)
    if n >= _VOL_WIN:
        for i in range(_VOL_WIN - 1, n):
            window = dv[i - _VOL_WIN + 1 : i + 1]
            hhi_vals[i] = _rolling_hhi(window)

    df["osap_herf_vol_hhi"] = hhi_vals

    # z-score of HHI over trailing 252-day window
    hhi_series = pd.Series(hhi_vals, index=df.index)
    roll_mean = hhi_series.rolling(_Z_WIN, min_periods=max(1, _Z_WIN // 2)).mean()
    roll_std = hhi_series.rolling(_Z_WIN, min_periods=max(1, _Z_WIN // 2)).std(ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hhi_z = (hhi_series - roll_mean) / roll_std.replace(0.0, np.nan)
    df["osap_herf_vol_hhi_z"] = hhi_z.values

    # ------------------------------------------------------------------
    # 2. Revenue concentration via PIT fundamentals (CoV of TTM revenue)
    # ------------------------------------------------------------------
    rev_conc = np.full(n, np.nan, dtype=np.float64)

    if _HAS_FUND:
        try:
            df_f = _fundamentals.as_of(df, fields=["revenue_ttm"])
            rev_col = "fund_revenue_ttm"
            if rev_col in df_f.columns:
                rev = df_f[rev_col].values.astype(np.float64)
                rev_series = pd.Series(rev, index=df_f.index)
                # CoV = std / |mean| over trailing _REV_QUARTERS*63 days (~2 years)
                # Use a longer window to capture enough quarterly data points
                win = _REV_QUARTERS * 63
                r_mean = rev_series.rolling(win, min_periods=4).mean()
                r_std = rev_series.rolling(win, min_periods=4).std(ddof=1)
                abs_mean = r_mean.abs().replace(0.0, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cov_vals = (r_std / abs_mean).values
                # clip extreme outliers
                rev_conc = np.where(np.isfinite(cov_vals), cov_vals, np.nan)
                # drop any fund_ scratch cols not in produces
                drop_cols = [c for c in df_f.columns if c.startswith("fund_")]
                df_f = df_f.drop(columns=drop_cols, errors="ignore")
        except Exception:
            pass

    df["osap_herf_rev_conc"] = rev_conc

    return df
