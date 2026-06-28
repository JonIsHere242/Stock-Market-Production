"""
_cand_osap_meanrankrevgrowth.py

Candidate feature block: Mean Rank Revenue Growth
Spec ID: osap_meanrankrevgrowth

OSAP signal: "Mean Rank Revenue Growth" — Hou, Xue & Zhang (2020) q-factor replication /
Chen & Zimmermann Open Source Asset Pricing catalog.

Original construction (cross-sectional):
    Each month, for each stock rank its annual sales growth [(sale_t / sale_{t-1}) - 1] and
    multi-year growth [(sale_t / sale_{t-3}) - 1] percentile-rank across all stocks in the
    universe, then average the two ranks. High rank = persistent revenue grower. Predicts
    higher future returns (growth anomaly, possibly risk-based in q-factor framework).

PER-TICKER PROXY (this block):
    Cross-sectional ranking is impossible inside a per-ticker compute() call. We substitute
    WITHIN-STOCK TRAILING PERCENTILE RANKS over a lookback window (~4 years of quarterly
    filings = ~16 observations), which captures the same economic signal — how strong the
    firm's current revenue growth is relative to its own recent history. This is an honest
    proxy: it preserves the "growth momentum" dimension but loses the cross-sectional
    dispersion component. The description notes this limitation explicitly.

Produces:
    osap_meanrankrevgrowth_main   — mean of rolling within-stock rank of 1yr and 3yr revenue
                                    growth rates (0-1 scale; higher = stronger grower relative
                                    to own history)
    osap_meanrankrevgrowth_1yr    — rolling within-stock rank of 1-year TTM revenue growth
    osap_meanrankrevgrowth_3yr    — rolling within-stock rank of 3-year TTM revenue growth
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper (by file path — never 'from FeatureTemplates')
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_meanrankrevgrowth",
    "description": (
        "Mean Rank Revenue Growth (OSAP replication). "
        "Original: cross-sectional average rank of 1-year and 3-year annual sales growth "
        "(Hou, Xue & Zhang 2020 q-factor / Chen & Zimmermann Open Source Asset Pricing). "
        "Per-ticker proxy: rolling within-stock percentile rank of 1yr and 3yr TTM revenue "
        "growth over a trailing 4-year (16-quarter) window, averaged. "
        "Captures how robust current revenue growth is vs this firm's own history. "
        "Missing for ETFs/foreign tickers without SEC fundamentals coverage (~16% of universe)."
    ),
    "requires": [],  # pure fundamentals; no OHLCV needed beyond the join key
    "produces": [
        "osap_meanrankrevgrowth_main",
        "osap_meanrankrevgrowth_1yr",
        "osap_meanrankrevgrowth_3yr",
    ],
    "tags": ["fundamental", "revenue", "growth", "quality", "osap", "pit"],
    "version": "1.0.0",
    "author": "Hou, Xue & Zhang (2020); Chen & Zimmermann Open Source Asset Pricing. Per-ticker proxy by Claude.",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_LOOKBACK_QTRS = 16  # ~4 years of quarterly observations for within-stock ranking


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank: for each position t, what fraction of the trailing
    `window` values is <= s[t]? Returns values in [0, 1]; NaN where fewer than
    2 non-NaN values in window.
    """
    def _pct_rank_last(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.nan
        # rank of last element among all valid elements in window
        last = arr[-1]
        if np.isnan(last):
            return np.nan
        return float(np.sum(valid <= last) / len(valid))

    return s.rolling(window=window, min_periods=2).apply(_pct_rank_last, raw=True)


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Default outputs to NaN — always assigned so columns always exist
    df["osap_meanrankrevgrowth_main"] = np.nan
    df["osap_meanrankrevgrowth_1yr"] = np.nan
    df["osap_meanrankrevgrowth_3yr"] = np.nan

    # Pull PIT fundamentals — need TTM revenue to build growth rates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = _fundamentals.as_of(df, fields=["revenue_ttm"])
        except Exception:
            # No fundamentals for this ticker — return NaN columns as-is
            return df

    rev = df["fund_revenue_ttm"]

    # Guard: need at least some non-NaN revenue observations
    if rev.notna().sum() < 4:
        df.drop(columns=["fund_revenue_ttm"], errors="ignore", inplace=True)
        return df

    # Replace zero/negative revenue with NaN so divisions are safe
    rev_safe = rev.where(rev > 0, other=np.nan)

    # -----------------------------------------------------------------------
    # 1-year revenue growth: (rev_ttm_t / rev_ttm_{t-252}) - 1
    # Using 252 trading-day shift as annual lag (≈1 year on daily price frame).
    # fundamentals are already PIT-merged to daily rows so this shift is safe.
    # -----------------------------------------------------------------------
    rev_lag_1yr = rev_safe.shift(252)
    growth_1yr = (rev_safe / rev_lag_1yr.where(rev_lag_1yr > 0, other=np.nan)) - 1.0
    # cap extreme values to avoid numerical outliers dominating rank
    growth_1yr = growth_1yr.clip(-10.0, 10.0)

    # -----------------------------------------------------------------------
    # 3-year revenue growth: (rev_ttm_t / rev_ttm_{t-756}) - 1  (756 ≈ 3 * 252)
    # -----------------------------------------------------------------------
    rev_lag_3yr = rev_safe.shift(756)
    growth_3yr = (rev_safe / rev_lag_3yr.where(rev_lag_3yr > 0, other=np.nan)) - 1.0
    growth_3yr = growth_3yr.clip(-10.0, 10.0)

    # -----------------------------------------------------------------------
    # Rolling within-stock percentile rank over trailing 4-year window
    # Window size in days: _LOOKBACK_QTRS * 63 ≈ 4yr of quarterly obs spaced
    # over daily rows, but since fundamentals repeat until next filing, we use
    # a day-based window that captures ~16 distinct quarterly values.
    # -----------------------------------------------------------------------
    rank_window = _LOOKBACK_QTRS * 63  # ~1008 trading days ≈ 4 years

    rank_1yr = _rolling_pct_rank(growth_1yr, window=rank_window)
    rank_3yr = _rolling_pct_rank(growth_3yr, window=rank_window)

    # -----------------------------------------------------------------------
    # Mean rank: average of the two horizon ranks (NaN if both missing)
    # -----------------------------------------------------------------------
    stacked = pd.concat([rank_1yr, rank_3yr], axis=1)
    mean_rank = stacked.mean(axis=1, skipna=False)  # NaN if either component NaN
    # Fall back to whichever is available when only one horizon has data
    mean_rank_fallback = stacked.mean(axis=1, skipna=True)
    # Use strict mean where both exist, fallback otherwise
    mean_rank = mean_rank.where(mean_rank.notna(), other=mean_rank_fallback)
    # Where neither is available, result stays NaN
    mean_rank = mean_rank.where(
        rank_1yr.notna() | rank_3yr.notna(), other=np.nan
    )

    df["osap_meanrankrevgrowth_1yr"] = rank_1yr
    df["osap_meanrankrevgrowth_3yr"] = rank_3yr
    df["osap_meanrankrevgrowth_main"] = mean_rank

    # Drop scratch fundamentals column (not in produces)
    df.drop(columns=["fund_revenue_ttm"], errors="ignore", inplace=True)

    return df
