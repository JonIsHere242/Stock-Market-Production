"""
DuPont asset-turnover trend feature block.

Spec: ext3_dupont_turnover_trend
Source: Round-4 expansion (osap_orgcap)

Captures improving operational efficiency via the DuPont decomposition:
  - asset_turnover = revenue_ttm / assets  (efficiency axis)
  - roa_dupont = net_margin * asset_turnover  (= net_income_ttm/assets, decomposed)
  - 252d linear trend of each (slope / mean), giving a drift signal orthogonal to level.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_dupont_turnover_trend",
    "description": (
        "DuPont decomposition trend features per ticker. "
        "asset_turnover = revenue_ttm / assets (operational efficiency). "
        "roa_dupont = net_margin * asset_turnover (profitability × efficiency). "
        "Produces the trailing 252-bar (≈1yr) linear trend (slope / mean) of each "
        "series so that the feature captures directional change rather than level. "
        "All fundamentals are PIT via as_of(filed_date) — no lookahead. "
        "Coverage ~84%; rows without fundamentals are NaN (expected)."
    ),
    "requires": ["Close"],  # Close used only as a time-axis anchor for merge_asof
    "produces": [
        "ext3_dupont_turnover_trend_at",         # asset-turnover level (latest PIT)
        "ext3_dupont_turnover_trend_at_slope",   # 252d normalised trend of asset turnover
        "ext3_dupont_turnover_trend_roa_slope",  # 252d normalised trend of DuPont ROA
    ],
    "tags": ["fundamentals", "dupont", "efficiency", "trend", "roa"],
    "version": "1.0",
    "author": "Spec: Round-4 expansion (osap_orgcap); impl Claude Sonnet 4.6",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normalised_slope(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling linear-regression slope over `window` bars, normalised by the
    rolling absolute mean so the output is scale-free (units: 1/bar).
    Returns NaN where the absolute mean is 0 or there are insufficient obs.
    No lookahead: uses only past values.
    """
    n = len(series)
    if n == 0:
        return series.copy()

    vals = series.to_numpy(dtype=float)
    out = np.full(n, np.nan)

    # Pre-compute x (centred) once for the full window
    w = min(window, n)
    x_full = np.arange(w, dtype=float)
    x_full -= x_full.mean()
    x_ss_full = float(np.dot(x_full, x_full))

    for i in range(w - 1, n):
        seg = vals[i - w + 1: i + 1]
        if np.isnan(seg).any():
            continue
        # Slope via closed-form OLS
        y_mean = seg.mean()
        if x_ss_full == 0.0:
            continue
        slope = float(np.dot(x_full, seg - y_mean)) / x_ss_full
        # Normalise by absolute mean to make scale-free
        abs_mean = abs(y_mean)
        if abs_mean == 0.0 or np.isnan(abs_mean):
            continue
        out[i] = slope / abs_mean

    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # -- load fundamentals helper -----------------------------------------------
    _spec2 = _ilu.spec_from_file_location(
        "_fundamentals",
        _P(__file__).resolve().parent / "_fundamentals.py",
    )
    _fundamentals = _ilu.module_from_spec(_spec2)
    _spec2.loader.exec_module(_fundamentals)

    # -- fetch PIT fundamentals fields ------------------------------------------
    df = _fundamentals.as_of(
        df,
        fields=["revenue_ttm", "assets", "net_margin"],
    )

    # -- asset turnover level ---------------------------------------------------
    rev = df["fund_revenue_ttm"].to_numpy(dtype=float)
    assets = df["fund_assets"].to_numpy(dtype=float)

    # Guard: assets == 0 → NaN; result is a per-row PIT value
    with np.errstate(divide="ignore", invalid="ignore"):
        at_arr = np.where(assets == 0, np.nan, rev / assets)

    at_series = pd.Series(at_arr, index=df.index)

    # -- DuPont ROA = net_margin * asset_turnover --------------------------------
    nm = df["fund_net_margin"].to_numpy(dtype=float)
    roa_arr = nm * at_arr  # NaN propagates naturally

    roa_series = pd.Series(roa_arr, index=df.index)

    # -- 252-bar normalised trend -----------------------------------------------
    at_slope = _normalised_slope(at_series, window=252)
    roa_slope = _normalised_slope(roa_series, window=252)

    # -- assign produced columns ------------------------------------------------
    df["ext3_dupont_turnover_trend_at"] = at_arr
    df["ext3_dupont_turnover_trend_at_slope"] = at_slope.values
    df["ext3_dupont_turnover_trend_roa_slope"] = roa_slope.values

    # -- drop scratch fund_ columns not in produces ----------------------------
    for col in ["fund_revenue_ttm", "fund_assets", "fund_net_margin"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
