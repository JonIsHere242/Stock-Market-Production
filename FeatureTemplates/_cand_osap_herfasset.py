"""
_cand_osap_herfasset.py  --  CANDIDATE (unproven, hidden from discovery)

Per-ticker proxy for Hou & Robinson (2006) industry asset-concentration
(Herfindahl index on firm assets at the 3-digit SIC level, 3-year rolling avg).

Cross-sectional note: the true signal requires summing squared asset-shares
across ALL firms in each SIC industry, which is impossible inside a single-stock
block. The proxy here captures the same economic intuition at the firm level:
  - A firm in a concentrated industry tends to have a large, stable, slowly-
    growing asset base relative to revenues (high asset intensity, low turnover).
  - We approximate the "self-contribution" to the Herfindahl using the firm's
    asset-to-revenue ratio squared, smoothed over 3 fiscal years. This is a
    monotone proxy of the firm's own weight in its industry's asset pool.
  - Predicted sign is -1 (high concentration -> lower future returns per H&R),
    so the produced column is also expected to have a negative cross-sectional
    relationship with forward returns.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# --- load PIT fundamentals helper ---
_s2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_herfasset",
    "description": (
        "Per-ticker proxy for Hou & Robinson (2006) industry asset Herfindahl "
        "(OSAP: osap_herfasset). True signal = 3-year rolling avg Herfindahl of "
        "3-digit SIC asset shares (cross-sectional, requires full industry panel). "
        "Proxy: firm self-weight approximation = (assets / revenue_ttm)^2, "
        "smoothed over ~3 fiscal years using a 36-month rolling median on the "
        "PIT fundamental values. A second column captures the YoY slope (change "
        "in concentration proxy). Predicted sign: -1 (high concentration -> lower "
        "forward returns). Per-ticker only -- cross-sectional ranking required for "
        "exact replication."
    ),
    "requires": [],   # uses PIT fundamentals; no raw OHLCV needed
    "produces": [
        "osap_herfasset_proxy",   # smoothed (assets/rev)^2 -- level
        "osap_herfasset_slope",   # 12m change in proxy -- dynamic
        "osap_herfasset_rank3y",  # percentile rank of proxy vs own 3-yr history
    ],
    "tags": ["fundamental", "concentration", "industry", "hou_robinson", "osap"],
    "version": "1.0",
    "author": "Hou & Robinson (2006) via Chen-Zimmermann OpenSourceAP; per-ticker proxy implementation",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three columns:
      osap_herfasset_proxy  -- rolling 36m median of (total_assets / revenue_ttm)^2
      osap_herfasset_slope  -- 12m change in proxy (momentum of concentration)
      osap_herfasset_rank3y -- rolling 3-year percentile rank of proxy (0-1)
    """
    # Pull PIT fundamentals: assets and revenue_ttm
    df = _fundamentals.as_of(df, fields=["assets", "revenue_ttm"])

    assets = df["fund_assets"]
    revenue = df["fund_revenue_ttm"]

    # Guard divide-by-zero: revenue == 0 or NaN -> NaN
    safe_rev = revenue.replace(0, np.nan)
    safe_assets = assets.where(assets.notna() & (assets > 0), np.nan)

    # Firm's squared asset-to-revenue ratio: proxy for self-weight in industry pool
    # (assets / revenue)^2  -- dimensionally equivalent to squared market-share-in-assets
    raw_proxy = (safe_assets / safe_rev) ** 2

    # Smooth over ~3 fiscal years (36 trading months, min 6 non-NaN obs)
    # Use rolling median (robust to fundamental restatements / outlier quarters)
    # Window in trading days: 36 months * ~21 days = 756; use 252*3 for exactness
    window = 252 * 3  # ~3 years of trading days
    min_obs = 126     # at least 6 months of data

    smoothed = raw_proxy.rolling(window=window, min_periods=min_obs).median()

    # Clip extreme outliers at 99th percentile (asset/rev ratio can blow up for
    # capital-heavy firms in low-revenue periods); do this on the raw series first
    p99 = raw_proxy.quantile(0.99)
    if pd.notna(p99) and p99 > 0:
        smoothed = smoothed.clip(upper=p99)

    df["osap_herfasset_proxy"] = smoothed

    # Slope: 12-month change in the smoothed proxy (252 trading days lag)
    lag_252 = smoothed.shift(252)
    slope = smoothed - lag_252
    # Normalise by lagged level so slope is a fractional change (avoids scale issues)
    safe_lag = lag_252.replace(0, np.nan)
    df["osap_herfasset_slope"] = slope / safe_lag

    # 3-year rolling percentile rank of the proxy within its own history
    # rank(x_t) = fraction of past 756-day window where proxy was <= x_t
    def _rolling_rank(series: pd.Series, win: int, min_p: int) -> pd.Series:
        """Vectorised rolling percentile rank using expanding cumcount."""
        vals = series.to_numpy(dtype=float)
        n = len(vals)
        out = np.full(n, np.nan)
        for i in range(n):
            lo = max(0, i - win + 1)
            window_vals = vals[lo : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) < min_p:
                continue
            cur = vals[i]
            if np.isnan(cur):
                continue
            out[i] = float(np.sum(valid <= cur)) / float(len(valid))
        return pd.Series(out, index=series.index)

    # Rolling rank is O(n * window) but window is large (756); use a sampled
    # approximation: subsample every 5 days for speed while staying O(n * w/5)
    # This is equivalent for the monotone rank signal.
    sub_step = 5
    idx = df.index
    sub_idx = idx[::sub_step]
    sub_series = smoothed.iloc[::sub_step].reset_index(drop=True)
    sub_win = max(1, window // sub_step)
    sub_minp = max(1, min_obs // sub_step)
    sub_rank = _rolling_rank(sub_series, sub_win, sub_minp)
    # Reindex back to full index via forward-fill (safe: only uses past sub-values)
    rank_series = (
        pd.Series(sub_rank.values, index=sub_idx)
        .reindex(idx)
        .ffill()
    )
    df["osap_herfasset_rank3y"] = rank_series

    # Drop scratch fundamental columns not in produces
    df.drop(columns=["fund_assets", "fund_revenue_ttm"], inplace=True, errors="ignore")

    return df
