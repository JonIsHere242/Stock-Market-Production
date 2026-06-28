"""
R&D capital-to-assets (osap_rdcap)

Per-ticker implementation of the Li (2011) R&D capital factor from Chen-Zimmermann
OpenSourceAP. R&D capital is the depreciated sum of lagged R&D flows scaled by total
assets. Cross-sectional size screen (upper two-thirds of market cap = NaN) is
approximated per-ticker using rolling percentile of own market cap vs a fixed
threshold proxy (we cannot rank cross-sectionally, so the size-screen is omitted and
noted as a limitation in the description).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_rdcap",
    "description": (
        "R&D capital-to-assets: depreciated weighted sum of 5 lagged quarterly R&D "
        "expenditures (weights 1.0, 0.8, 0.6, 0.4, 0.2) scaled by total assets, "
        "using PIT SEC fundamentals. Missing R&D is treated as zero per Li (2011). "
        "Positive predicted sign (high R&D capital → higher future returns). "
        "Cross-sectional size screen (drop upper two-thirds market-cap) cannot be "
        "applied per-ticker; instead osap_rdcap_small flags rows where price * "
        "shares outstanding is below its own trailing 3-year 33rd-percentile as a "
        "rough proxy. Per-ticker proxy only — true cross-sectional rank not available."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_rdcap",          # R&D capital / assets (raw, all firms)
        "osap_rdcap_chg",      # quarter-over-quarter change in R&D capital ratio
        "osap_rdcap_small",    # R&D capital ratio masked to 'small-cap' rows only
    ],
    "tags": ["fundamentals", "rnd", "asset_composition", "accounting", "li2011"],
    "version": "1.0",
    "author": "Li (2011) via Chen-Zimmermann OpenSourceAP; per-ticker implementation",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute R&D capital-to-assets per Li (2011).

    Weights: rnd_cap = xrd_t + 0.8*xrd_{t-1} + 0.6*xrd_{t-2} + 0.4*xrd_{t-3} + 0.2*xrd_{t-4}
    where xrd is replaced by 0 if missing.
    rdcap_ratio = rnd_cap / assets (NaN if assets == 0 or missing).
    """
    # Initialise output columns to NaN
    df = df.copy()
    df["osap_rdcap"] = np.nan
    df["osap_rdcap_chg"] = np.nan
    df["osap_rdcap_small"] = np.nan

    # Need at least a few rows to be useful
    if len(df) < 2:
        return df

    # Pull PIT fundamentals: R&D (annual or TTM) and total assets
    # rnd_expense_ttm is the rolling 12-month R&D; assets is the balance-sheet total.
    # shares_outstanding for market-cap proxy.
    df = _fundamentals.as_of(
        df,
        fields=[
            "rnd_expense_ttm",
            "assets",
            "shares_outstanding",
        ],
    )

    # Replace missing R&D with 0 per spec ("replace xrd with 0 if missing")
    rnd = df["fund_rnd_expense_ttm"].fillna(0.0)
    assets = df["fund_assets"]

    # Build lagged R&D series (shift by quarters approximated as ~63 trading days each)
    # PIT fundamentals are already forward-filled at the as_of merge, so we shift
    # the resulting series to get t-1..t-4 readings.
    # Each "lag" here is one filing period back; with TTM data filed ~quarterly,
    # a 63-bar shift (~1 quarter of trading days) approximates each lag.
    QUARTER_BARS = 63

    rnd_t0 = rnd
    rnd_t1 = rnd.shift(QUARTER_BARS)
    rnd_t2 = rnd.shift(2 * QUARTER_BARS)
    rnd_t3 = rnd.shift(3 * QUARTER_BARS)
    rnd_t4 = rnd.shift(4 * QUARTER_BARS)

    # Depreciated R&D capital (fill shifted NaN with 0 per spec)
    rnd_cap = (
        1.0 * rnd_t0.fillna(0.0)
        + 0.8 * rnd_t1.fillna(0.0)
        + 0.6 * rnd_t2.fillna(0.0)
        + 0.4 * rnd_t3.fillna(0.0)
        + 0.2 * rnd_t4.fillna(0.0)
    )

    # Scale by assets; guard divide-by-zero
    safe_assets = assets.replace(0, np.nan)
    rdcap_ratio = rnd_cap / safe_assets
    # Clamp extreme values (e.g. tiny-asset firms) to [0, 10] to avoid inf
    rdcap_ratio = rdcap_ratio.clip(lower=0.0, upper=10.0)

    # Zero R&D capital is uninformative — if assets are available but rnd_cap == 0
    # AND assets are valid, keep the 0 (it tells us no R&D stock).
    # But if assets are missing entirely, force NaN.
    rdcap_ratio = rdcap_ratio.where(safe_assets.notna(), other=np.nan)

    df["osap_rdcap"] = rdcap_ratio

    # Quarter-over-quarter change in the ratio
    df["osap_rdcap_chg"] = rdcap_ratio - rdcap_ratio.shift(QUARTER_BARS)

    # Per-ticker small-cap proxy:
    # Market cap = Close * shares_outstanding.
    # Flag rows where mktcap is in the bottom third of its own trailing 3-year window.
    shares = df["fund_shares_outstanding"]
    mktcap = df["Close"] * shares.replace(0, np.nan)
    # Rolling 33rd-percentile over ~756 bars (3 years)
    WINDOW_SMALL = 756
    p33 = mktcap.rolling(WINDOW_SMALL, min_periods=WINDOW_SMALL // 4).quantile(0.333)
    is_small = mktcap <= p33  # True = bottom third = "small"
    # Mask: show rdcap_ratio only for rows classified as small
    df["osap_rdcap_small"] = rdcap_ratio.where(is_small, other=np.nan)

    # Drop scratch fund_ columns not in produces
    drop_cols = [c for c in df.columns if c.startswith("fund_")]
    df.drop(columns=drop_cols, inplace=True)

    return df
