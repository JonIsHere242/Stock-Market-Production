"""
osap_zerotrade  --  Zero-trade days illiquidity signal.

Source: Lesmond, Ogden & Trzcinka (1999) "A New Estimate of Transaction Costs",
        Review of Financial Studies 12(5), 1113-1141.
        Canonical OSAP (Open Source Asset Pricing) implementation documented in
        Chen & Zimmermann (2022) "Open Source Cross-Sectional Asset Pricing",
        Critical Finance Review.

Signal: fraction of trading days with zero return (or zero volume) over a
trailing window.  Higher values => less liquid => higher expected returns
(illiquidity premium, Amihud 2002 family).

Per-ticker proxy: identical to the paper definition -- no cross-sectional data
needed.  The original paper measures zero-return days as a proxy for binding
transaction costs (informed traders won't trade when the gain < round-trip cost).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_zerotrade",
    "description": (
        "Zero-trade illiquidity signal: fraction of days with zero daily return "
        "and/or zero volume over a trailing 63-day (~1 quarter) window, plus a "
        "252-day annual window variant and a momentum-of-illiquidity slope. "
        "Higher values indicate lower liquidity (Lesmond, Ogden & Trzcinka 1999; "
        "OSAP Chen & Zimmermann 2022). Per-ticker proxy -- no cross-section needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "osap_zerotrade_63d",    # fraction zero-return days, 63-day window
        "osap_zerotrade_252d",   # fraction zero-return days, 252-day window
        "osap_zerotrade_slope",  # change: 63d - prior-63d (momentum of illiquidity)
    ],
    "tags": ["liquidity", "illiquidity", "zero_trade", "osap", "lesmond"],
    "version": "1.0.0",
    "author": "Lesmond, Ogden & Trzcinka (1999); OSAP: Chen & Zimmermann (2022); block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute zero-trade illiquidity features per ticker.

    A day is classified as a 'zero-trade day' if its daily return is zero
    (|pct_change(Close)| == 0) OR its volume is zero.  Using the union of both
    conditions is conservative and robust to data quality issues where price is
    stale but volume is recorded (or vice versa).

    Windows:
      - 63-day  (~1 quarter, canonical short-window)
      - 252-day (~1 year)
      - slope   = 63d_window_t  -  63d_window_{t-63}  (direction of illiquidity change)
    """
    close = df["Close"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)
    n = len(df)

    # --- daily log return (NaN on first row) ---
    # Using pct change; zero-return days => |ret| == 0
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = np.where(
            close[:-1] == 0,
            np.nan,
            (close[1:] - close[:-1]) / close[:-1],
        )

    # --- zero-trade indicator: zero-return OR zero-volume ---
    zero_vol = (volume == 0) | np.isnan(volume)
    zero_ret = (ret == 0.0)  # exact zero only; NaN is not zero
    zero_flag = (zero_vol | zero_ret).astype(np.float64)
    # First row is NaN return => mark as NaN (unknown), not zero-trade
    zero_flag[0] = np.nan

    # --- rolling mean (fraction) over window ---
    s = pd.Series(zero_flag, index=df.index)

    win_short = 63
    win_long = 252

    frac_63 = s.rolling(win_short, min_periods=max(1, win_short // 2)).mean()
    frac_252 = s.rolling(win_long, min_periods=max(1, win_long // 2)).mean()

    # --- slope: current 63d fraction minus lagged 63d fraction (63 bars ago) ---
    slope = frac_63 - frac_63.shift(win_short)

    # --- guard: replace inf with NaN (shouldn't arise but be safe) ---
    def _clean(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan)

    df["osap_zerotrade_63d"] = _clean(frac_63).values
    df["osap_zerotrade_252d"] = _clean(frac_252).values
    df["osap_zerotrade_slope"] = _clean(slope).values

    return df
